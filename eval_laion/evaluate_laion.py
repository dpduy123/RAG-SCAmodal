"""
eval_laion/evaluate_laion.py

Standalone LAION Amodal evaluation script.
Follows the Ao et al. CVPR 2025 evaluation protocol:

  - CLIP Score: semantic alignment between completed object and category label
  - LPIPS: visual perceptual distance (visible region only)
  - Feature Similarity: CLIP image-feature cosine similarity (visible region only)
  - SSIM: structural similarity (visible region only)
  - PSNR: peak signal-to-noise ratio (visible region only)

Unlike COCOA, LAION has NO ground-truth masks. Visible masks are generated
automatically via SAM2 auto-segmentation + CLIP label matching.

Usage (from project root on Colab):
    from eval_laion.evaluate_laion import LAIONEvaluator
    evaluator = LAIONEvaluator(device="cuda")
    evaluator.evaluate(
        ann_file="dataset/LAION/laion_00000_annotation.json",
        img_dir="dataset/LAION",
        limit=50,
    )
"""

import os
import io
import json
import contextlib
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image

# ── Helpers ─────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _silence_output(enabled: bool = True):
    """Suppress stdout/stderr while debug logs keep rendering."""
    if not enabled:
        yield
        return
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield


def _apply_visible_mask(img: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    """
    Zero out RGB pixels outside the visible mask.
    Ao et al. protocol: compare visible part of the object with completed version.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    mask3 = visible_mask.astype(bool)[:, :, None]
    return np.where(mask3, img, 0).astype(img.dtype)


# ── Visual debug output ─────────────────────────────────────────────────────

def _save_debug_visualization(
    output_dir: str,
    basename: str,
    image: np.ndarray,
    visible_mask: np.ndarray,
    label: str,
    amodal_output: dict | None = None,
    clip_mask_score: float = 0.0,
    metrics: dict | None = None,
):
    """Save a multi-panel debug image showing every pipeline stage."""
    os.makedirs(output_dir, exist_ok=True)
    H, W = image.shape[:2]

    # Panel 1: Original image
    panel_orig = image.copy()

    # Panel 2: Image with visible mask overlay (green)
    panel_mask = image.copy()
    mask_overlay = np.zeros_like(image)
    mask_overlay[visible_mask.astype(bool)] = [0, 255, 0]
    panel_mask = cv2.addWeighted(panel_mask, 0.6, mask_overlay, 0.4, 0)

    # Panel 3: Masked region only
    panel_masked = _apply_visible_mask(image, visible_mask)

    panels = [panel_orig, panel_mask, panel_masked]
    titles = [
        f"Original ({W}x{H})",
        f"SAM Mask (CLIP={clip_mask_score:.3f})",
        f"Masked: '{label}'",
    ]

    # Panel 4+5: Amodal result (if available)
    if amodal_output is not None:
        rgba = amodal_output.get("inpainted_rgba")
        if rgba is not None:
            panel_result = rgba[:, :, :3].copy()
            panels.append(panel_result)
            titles.append("Amodal Completed")

        amodal_mask = amodal_output.get("amodal_mask")
        if amodal_mask is not None:
            panel_amodal = np.zeros_like(image)
            panel_amodal[amodal_mask.astype(bool)] = [255, 255, 255]
            panels.append(panel_amodal)
            titles.append("Amodal Mask")

    # Build composite image
    n = len(panels)
    panel_h = 300
    panel_w = int(panel_h * W / H)
    margin = 10
    text_h = 30

    canvas_w = n * (panel_w + margin) + margin
    canvas_h = panel_h + text_h + 2 * margin
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40

    for i, (panel, title) in enumerate(zip(panels, titles)):
        x = margin + i * (panel_w + margin)
        y = margin
        resized = cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        canvas[y:y + panel_h, x:x + panel_w] = resized
        cv2.putText(canvas, title, (x + 5, y + panel_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if metrics:
        metric_str = " | ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
            if isinstance(v, (int, float)) and k not in ("mask_clip_score",)
        )
        cv2.putText(canvas, metric_str, (margin, canvas_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)

    save_name = basename.replace(".jpg", "_debug.jpg").replace(".png", "_debug.jpg")
    save_path = os.path.join(output_dir, save_name)
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return save_path


# ── Main Evaluator ──────────────────────────────────────────────────────────

class LAIONEvaluator:
    """
    Self-contained evaluator for the LAION subset of the Ao et al. CVPR 2025
    benchmark. Handles mask generation, amodal completion, and metric computation.
    """

    # CSV columns in fixed order for incremental writes
    CSV_COLUMNS = [
        "filename", "category", "mask_clip_score",
        "CLIP_score", "LPIPS", "Feature_Similarity", "SSIM", "PSNR",
        "vlm_reasoning", "critic_score",
    ]

    def __init__(self, device="cuda", completer=None, segmenter=None):
        """
        Args:
            device:     "cuda" or "cpu"
            completer:  optional pre-loaded AmodalCompleter (avoids reloading ~25GB)
            segmenter:  optional pre-loaded SAMSegmenter (avoids reloading SAM2)
        """
        self.device = device
        self.results = []

        # ── Load pipeline components ──
        from amodal_completer import AmodalCompleter
        self.completer = completer if completer is not None else AmodalCompleter(device=device)
        self._owns_completer = completer is None

        from segmenter import SAMSegmenter
        self.segmenter = segmenter if segmenter is not None else SAMSegmenter(device=device)
        self._owns_segmenter = segmenter is None

        from eval_laion.mask_generator import CLIPMaskSelector
        self.mask_selector = CLIPMaskSelector(device=device)

    # ── Incremental CSV ─────────────────────────────────────────────────────

    def _init_csv(self, output_file: str):
        """Write CSV header. Called once at the start."""
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        df = pd.DataFrame(columns=self.CSV_COLUMNS)
        df.to_csv(output_file, index=False)

    def _append_csv(self, output_file: str, metrics: dict):
        """Append a single row to the CSV after each sample completes."""
        row = pd.DataFrame([metrics], columns=self.CSV_COLUMNS)
        row.to_csv(output_file, mode="a", header=False, index=False)

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        ann_file: str,
        img_dir: str,
        limit: int = None,
        verbose: bool = False,
        output_file: str = "results/laion_eval_results.csv",
        save_masks: bool = False,
        mask_output_dir: str = "results/laion_masks",
        save_debug: bool = False,
        debug_output_dir: str = "results/laion_debug",
    ):
        """
        Run the full LAION evaluation.

        Args:
            ann_file:          path to laion_00000_annotation.json
            img_dir:           directory containing image files
                               (img_dir + annotation filename = full path)
            limit:             max number of annotations to process (None = all)
            verbose:           show per-image internal logs (SD2 progress bars etc.)
            output_file:       CSV file for per-sample results (written incrementally)
            save_masks:        save auto-generated visible masks for inspection
            mask_output_dir:   directory for saved masks
            save_debug:        save multi-panel debug visualizations
            debug_output_dir:  directory for debug images
        """
        from metrics_utils import (
            calculate_lpips,
            calculate_clip_score,
            calculate_feature_similarity,
            calculate_ssim,
            calculate_psnr,
        )

        # Ensure output directories exist
        if save_masks:
            os.makedirs(mask_output_dir, exist_ok=True)
        if save_debug:
            os.makedirs(debug_output_dir, exist_ok=True)

        # ── Load annotations ──
        print(f"[LAION Eval] Loading annotations from {ann_file}...")
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        anns = data.get("annotations", [])
        if limit:
            anns = anns[:limit]

        total = len(anns)
        print(f"[LAION Eval] Total annotations: {total}")
        print(f"[LAION Eval] Image directory:    {img_dir}")
        print(f"[LAION Eval] Output CSV:         {output_file}")

        # ── Init incremental CSV ──
        self._init_csv(output_file)

        skipped = 0
        skip_reasons = {"image_not_found": 0, "image_read_error": 0,
                        "sam_no_masks": 0, "mask_error": 0, "pipeline_error": 0}

        print(f"\n[LAION Eval] Starting evaluation...")
        print(f"{'=' * 70}")

        for idx, ann in enumerate(anns):
            filename = ann["filename"]
            labels = ann.get("labels", [])
            label = labels[0] if labels else "object"
            basename = os.path.basename(filename)

            # ── 1. Load image ──
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                skip_reasons["image_not_found"] += 1
                skipped += 1
                print(f"  [{idx+1}/{total}] ❌ NOT FOUND: {img_path}")
                continue

            image = cv2.imread(img_path)
            if image is None:
                skip_reasons["image_read_error"] += 1
                skipped += 1
                print(f"  [{idx+1}/{total}] ❌ READ ERROR: {img_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            H, W = image.shape[:2]

            print(f"\n  [{idx+1}/{total}] 📷 {basename} ({W}×{H}) — label: '{label}'")

            # ── 2. Auto-generate visible mask (SAM2 + CLIP) ──
            try:
                print(f"    ▸ SAM2 auto-segment...", end=" ", flush=True)
                with _silence_output(enabled=not verbose):
                    sam_masks = self.segmenter.segment_everything(
                        image,
                        points_per_side=48,            # denser grid → more masks
                        pred_iou_thresh=0.70,          # lower → keep less confident masks
                        stability_score_thresh=0.80,   # lower → keep cat/hat sized objects
                        min_mask_region_area=200,
                    )
                print(f"found {len(sam_masks)} masks")

                if not sam_masks:
                    skip_reasons["sam_no_masks"] += 1
                    skipped += 1
                    print(f"    ❌ SAM2 found 0 masks — skipping")
                    continue

                print(f"    ▸ CLIP matching '{label}'...", end=" ", flush=True)
                visible_mask, mask_idx, clip_score_mask = \
                    self.mask_selector.select_best_mask(image, sam_masks, label)
                area_pct = visible_mask.sum() / (H * W) * 100
                print(f"mask #{mask_idx} (CLIP={clip_score_mask:.3f}, "
                      f"area={visible_mask.sum():,}px / {area_pct:.1f}%)")

                if save_masks:
                    mask_save = os.path.join(mask_output_dir,
                                             basename.replace(".jpg", "_mask.png"))
                    cv2.imwrite(mask_save, (visible_mask * 255).astype(np.uint8))

            except Exception as e:
                skip_reasons["mask_error"] += 1
                skipped += 1
                print(f"\n    ❌ Mask error: {e}")
                continue

            # ── 3. Run amodal completion pipeline ──
            try:
                print(f"    ▸ Amodal completion (Pix2Gestalt + VLM)...")
                with _silence_output(enabled=not verbose):
                    output = self.completer.complete(
                        image, visible_mask, all_masks=[],
                        mask_only=True,
                    )

                vlm_reasoning = output.get("vlm_reasoning", "N/A")
                amodal_mask = output.get("amodal_mask")
                missing_mask = output.get("missing_mask")

                print(f"    ▸ VLM Reasoning: {str(vlm_reasoning)[:100]}"
                      f"{'...' if len(str(vlm_reasoning)) > 100 else ''}")
                if missing_mask is not None:
                    miss_pct = missing_mask.sum() / (H * W) * 100
                    print(f"    ▸ Missing region: {missing_mask.sum():,}px ({miss_pct:.1f}%)")

                # ── 4. Compute metrics ──
                # Compare: A = visible object only  vs  B = complete version
                #   A: original pixels masked by visible_mask (neutral bg)
                #   B: original pixels masked by amodal_mask  (neutral bg)
                # This measures how much the amodal extension changes
                # the object appearance — a good mask adds coherent content.
                neutral_bg = np.full_like(image, 127, dtype=np.uint8)

                # A: visible-only on neutral bg
                vis3 = visible_mask[:, :, None].astype(np.float32)
                img_visible = (image * vis3 + neutral_bg * (1 - vis3)).astype(np.uint8)

                # B: complete (visible + missing) on neutral bg
                amodal3 = amodal_mask[:, :, None].astype(np.float32)
                img_complete = (image * amodal3 + neutral_bg * (1 - amodal3)).astype(np.uint8)

                print(f"    ▸ Computing metrics (visible vs complete)...", end=" ", flush=True)
                metrics = {
                    "filename": filename,
                    "category": label,
                    "mask_clip_score": clip_score_mask,
                    "CLIP_score": calculate_clip_score(img_complete, label),
                    "LPIPS": calculate_lpips(img_complete, img_visible),
                    "Feature_Similarity": calculate_feature_similarity(img_complete, img_visible),
                    "SSIM": calculate_ssim(img_complete, img_visible),
                    "PSNR": calculate_psnr(img_complete, img_visible),
                    "vlm_reasoning": vlm_reasoning,
                    "critic_score": None,
                }
                self.results.append(metrics)

                # ── 5. Append to CSV immediately ──
                self._append_csv(output_file, metrics)

                print(f"✅ CLIP={metrics['CLIP_score']:.2f}, "
                      f"LPIPS={metrics['LPIPS']:.4f}, "
                      f"SSIM={metrics['SSIM']:.4f}, "
                      f"PSNR={metrics['PSNR']:.2f}")
                print(f"    ▸ Saved to CSV ({len(self.results)}/{total} done)")

                # Save debug visualization
                if save_debug:
                    debug_path = _save_debug_visualization(
                        debug_output_dir, basename, image, visible_mask,
                        label, amodal_output=output,
                        clip_mask_score=clip_score_mask, metrics=metrics,
                    )
                    print(f"    ▸ Debug image: {debug_path}")

            except Exception as e:
                skip_reasons["pipeline_error"] += 1
                skipped += 1
                print(f"\n    ❌ Pipeline error: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Periodically free GPU cache
            if idx % 10 == 0 and self.device == "cuda":
                torch.cuda.empty_cache()

        # ── Final summary ──
        print(f"\n{'=' * 70}")
        print(f"[LAION Eval] ✅ Completed: {len(self.results)}/{total} samples")
        print(f"[LAION Eval]    Skipped: {skipped}")
        if any(v > 0 for v in skip_reasons.values()):
            print(f"[LAION Eval]    Skip breakdown:")
            for reason, count in skip_reasons.items():
                if count > 0:
                    print(f"      - {reason}: {count}")

        self._print_summary()
        print(f"\n[LAION Eval] 📊 Full results: {output_file}")
        return self.results

    # ── Summary statistics ──────────────────────────────────────────────────

    def _print_summary(self):
        """Print summary statistics from collected results."""
        if not self.results:
            print("[LAION Eval] No results to summarize.")
            return

        df = pd.DataFrame(self.results)

        print("\n" + "=" * 60)
        print("  LAION Evaluation Summary")
        print("=" * 60)
        metric_cols = ["CLIP_score", "LPIPS", "Feature_Similarity", "SSIM", "PSNR"]
        for col in metric_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"  {col:25s}: {mean_val:.4f} ± {std_val:.4f}")

        # Per-category breakdown (top 10)
        if "category" in df.columns and len(df) > 5:
            print(f"\n  Per-Category Breakdown (top 10):")
            print("  " + "-" * 56)
            cat_stats = df.groupby("category").agg({
                "CLIP_score": "mean",
                "LPIPS": "mean",
                "SSIM": "mean",
            }).round(4)
            cat_counts = df["category"].value_counts()
            cat_stats["count"] = cat_counts
            cat_stats = cat_stats.sort_values("count", ascending=False).head(10)
            for cat, row in cat_stats.iterrows():
                print(f"  {cat:20s} (n={int(row['count']):3d}): "
                      f"CLIP={row['CLIP_score']:.2f}, "
                      f"LPIPS={row['LPIPS']:.4f}, "
                      f"SSIM={row['SSIM']:.4f}")

        print("=" * 60)

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def cleanup(self):
        """Free all GPU memory."""
        if hasattr(self, "completer") and self._owns_completer:
            self.completer.cleanup()
            del self.completer

        if hasattr(self, "segmenter") and self._owns_segmenter:
            del self.segmenter

        if hasattr(self, "mask_selector"):
            self.mask_selector.cleanup()
            del self.mask_selector

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[LAION Eval] GPU memory cleared.")


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LAION Amodal Evaluation")
    parser.add_argument("--ann-file", type=str,
                        default="dataset/LAION/laion_00000_annotation.json")
    parser.add_argument("--img-dir", type=str, default="dataset/LAION")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="results/laion_eval_results.csv")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    evaluator = LAIONEvaluator(device="cuda")
    try:
        evaluator.evaluate(
            ann_file=args.ann_file,
            img_dir=args.img_dir,
            limit=args.limit,
            output_file=args.output,
            save_masks=args.save_masks,
            save_debug=args.save_debug,
            verbose=args.verbose,
        )
    finally:
        evaluator.cleanup()
