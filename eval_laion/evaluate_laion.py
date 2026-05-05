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
        img_dir="dataset/LAION",   # parent dir containing laion/00000/*.jpg
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
from tqdm import tqdm
import pandas as pd
from PIL import Image

# ── Silence helper ──────────────────────────────────────────────────────────

@contextlib.contextmanager
def _silence_output(enabled: bool = True):
    """Suppress stdout/stderr while the outer tqdm bar keeps rendering."""
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


# ── Main Evaluator ──────────────────────────────────────────────────────────

class LAIONEvaluator:
    """
    Self-contained evaluator for the LAION subset of the Ao et al. CVPR 2025
    benchmark. Handles mask generation, amodal completion, and metric computation.
    """

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
        # Import from project root (assumed in sys.path)
        from amodal_completer import AmodalCompleter
        self.completer = completer if completer is not None else AmodalCompleter(device=device)
        self._owns_completer = completer is None

        from segmenter import SAMSegmenter
        self.segmenter = segmenter if segmenter is not None else SAMSegmenter(device=device)
        self._owns_segmenter = segmenter is None

        # CLIP-based mask selector (reuses CLIP weights efficiently)
        from eval_laion.mask_generator import CLIPMaskSelector
        self.mask_selector = CLIPMaskSelector(device=device)

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        ann_file: str,
        img_dir: str,
        limit: int = None,
        verbose: bool = False,
        output_file: str = "laion_eval_results.csv",
        save_masks: bool = False,
        mask_output_dir: str = "results/laion_masks",
    ):
        """
        Run the full LAION evaluation.

        Args:
            ann_file:        path to laion_00000_annotation.json
            img_dir:         directory containing image files (parent of laion/00000/)
            limit:           max number of annotations to process (None = all)
            verbose:         show per-image internal logs
            output_file:     CSV file for per-sample results
            save_masks:      if True, save auto-generated visible masks for inspection
            mask_output_dir: directory for saved masks
        """
        # Import metrics from project-level utils
        from metrics_utils import (
            calculate_lpips,
            calculate_clip_score,
            calculate_feature_similarity,
            calculate_ssim,
            calculate_psnr,
        )

        # Load annotations
        print(f"[LAION Evaluator] Loading annotations from {ann_file}...")
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        anns = data.get("annotations", [])
        if limit:
            anns = anns[:limit]

        print(f"[LAION Evaluator] Evaluating on {len(anns)} samples...")

        if save_masks:
            os.makedirs(mask_output_dir, exist_ok=True)

        skipped = 0

        for idx, ann in enumerate(tqdm(anns, desc="LAION Eval")):
            filename = ann["filename"]
            labels = ann.get("labels", [])
            label = labels[0] if labels else "object"

            # ── 1. Load image ──
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                # Try stripping leading directory
                img_path = os.path.join(img_dir, os.path.basename(filename))
                if not os.path.exists(img_path):
                    if verbose:
                        print(f"  [SKIP] Image not found: {filename}")
                    skipped += 1
                    continue

            image = cv2.imread(img_path)
            if image is None:
                skipped += 1
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # ── 2. Auto-generate visible mask (SAM2 + CLIP) ──
            try:
                with _silence_output(enabled=not verbose):
                    sam_masks = self.segmenter.segment_everything(image)

                if not sam_masks:
                    if verbose:
                        print(f"  [SKIP] SAM2 found 0 masks for {filename}")
                    skipped += 1
                    continue

                visible_mask, mask_idx, clip_score_mask = \
                    self.mask_selector.select_best_mask(image, sam_masks, label)

                if verbose:
                    print(f"  [Mask] Selected mask {mask_idx} for '{label}' "
                          f"(CLIP={clip_score_mask:.3f}, area={visible_mask.sum()}px)")

                # Save mask for inspection
                if save_masks:
                    mask_save_path = os.path.join(
                        mask_output_dir,
                        os.path.basename(filename).replace(".jpg", "_mask.png")
                    )
                    cv2.imwrite(mask_save_path, (visible_mask * 255).astype(np.uint8))

            except Exception as e:
                print(f"  [ERROR] Mask generation failed for {filename}: {e}")
                skipped += 1
                continue

            # ── 3. Run amodal completion pipeline ──
            try:
                with _silence_output(enabled=not verbose):
                    output = self.completer.complete(
                        image, visible_mask, all_masks=[]
                    )
                pred_image = output["inpainted_rgba"][:, :, :3]

                # Ao et al. protocol: mask both images by visible region
                image_v = _apply_visible_mask(image, visible_mask)
                pred_v = _apply_visible_mask(pred_image, visible_mask)

                # ── 4. Compute metrics ──
                metrics = {
                    "filename": filename,
                    "category": label,
                    "mask_clip_score": clip_score_mask,  # how well SAM mask matched label
                    "CLIP_score": calculate_clip_score(pred_image, label),
                    "LPIPS": calculate_lpips(pred_v, image_v),
                    "Feature_Similarity": calculate_feature_similarity(pred_v, image_v),
                    "SSIM": calculate_ssim(pred_v, image_v),
                    "PSNR": calculate_psnr(pred_v, image_v),
                }
                self.results.append(metrics)

                if verbose:
                    print(f"  [OK] {filename} → CLIP={metrics['CLIP_score']:.2f}, "
                          f"LPIPS={metrics['LPIPS']:.4f}, SSIM={metrics['SSIM']:.4f}")

            except Exception as e:
                print(f"  [ERROR] Pipeline failed for {filename}: {e}")
                skipped += 1
                continue

            # Periodically free GPU cache
            if idx % 10 == 0 and self.device == "cuda":
                torch.cuda.empty_cache()

        print(f"\n[LAION Evaluator] Completed: {len(self.results)} samples, "
              f"Skipped: {skipped}")

        self._save_results(output_file)
        return self.results

    # ── Result saving ───────────────────────────────────────────────────────

    def _save_results(self, filename: str):
        """Save results to CSV and print summary statistics."""
        if not self.results:
            print("[LAION Evaluator] No results to save.")
            return

        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\n[LAION Evaluator] Results saved to {filename}")

        # Summary statistics
        print("\n" + "=" * 60)
        print("  LAION Evaluation Summary")
        print("=" * 60)
        metric_cols = ["CLIP_score", "LPIPS", "Feature_Similarity", "SSIM", "PSNR"]
        for col in metric_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"  {col:25s}: {mean_val:.4f} ± {std_val:.4f}")

        # Per-category breakdown (top 10 most frequent)
        if "category" in df.columns and len(df) > 5:
            print("\n  Per-Category Breakdown (top 10):")
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
        print("[LAION Evaluator] GPU memory cleared.")


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LAION Amodal Evaluation")
    parser.add_argument("--ann-file", type=str,
                        default="dataset/LAION/laion_00000_annotation.json")
    parser.add_argument("--img-dir", type=str, default="dataset/LAION")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="laion_eval_results.csv")
    parser.add_argument("--save-masks", action="store_true")
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
            verbose=args.verbose,
        )
    finally:
        evaluator.cleanup()
