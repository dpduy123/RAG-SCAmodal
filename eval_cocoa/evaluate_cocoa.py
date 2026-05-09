import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import io
import json
import contextlib
import torch
import numpy as np
import cv2
import gzip
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset


@contextlib.contextmanager
def _silence_output(enabled: bool = True):
    """
    Suppress stdout/stderr inside the block. Used to hide noisy per-step DDIM
    progress bars (and other internal logs) from the inner pipeline while the
    outer tqdm bar keeps rendering between iterations.
    """
    if not enabled:
        yield
        return
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield


def _apply_visible_mask(img: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    """
    Zero out RGB pixels outside the visible mask. Returns an RGB image of the
    same shape (H, W, 3) — pixels inside the mask keep their colour, pixels
    outside become (0, 0, 0). Used to align with the Ao et al. CVPR 2025
    evaluation protocol: compare the visible part of the object against the
    completed version.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    mask3 = visible_mask.astype(bool)[:, :, None]
    return np.where(mask3, img, 0).astype(img.dtype)
from amodal_completer import AmodalCompleter
from metrics_utils import (
    calculate_lpips,
    calculate_clip_score,
    calculate_feature_similarity,
    calculate_ssim,
    calculate_psnr,
)

class AmodalEvaluator:
    def __init__(self, device="cuda", completer=None):
        """
        Args:
            device:    "cuda" or "cpu"
            completer: optional pre-loaded AmodalCompleter — pass this to avoid
                       reloading ~25GB of model weights between notebook cells.
                       If None, a new AmodalCompleter is constructed (which will
                       still reuse class-level singletons if already loaded).
        """
        self.completer = completer if completer is not None else AmodalCompleter(device=device)
        self._owns_completer = completer is None
        self.results = []

    def cleanup(self):
        """
        Free VRAM. Only tears down the underlying models if this evaluator
        constructed its own completer; otherwise leaves it alone so the caller
        can keep reusing it.
        """
        if hasattr(self, 'completer'):
            if self._owns_completer:
                self.completer.cleanup()
            del self.completer
        
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Evaluator] GPU Memory cleared.")

    def evaluate_cocoa(self, ann_file, img_dir, limit=100, verbose=False):
        """
        Evaluate on the custom COCOA subset JSON.

        Args:
            verbose: if False (default), per-image internal logs (DDIM bar,
                     [VLM Reason], etc.) are silenced so only the outer tqdm
                     progress bar shows. Set True to debug a single image.
        """
        from pycocotools import mask as mask_utils
        
        print(f"[Evaluator] Loading subset annotations from {ann_file}...")
        
        # Support for .gz files to prevent sync issues
        if ann_file.endswith(".gz"):
            with gzip.open(ann_file, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        anns = data.get('annotations', [])
        if limit:
            anns = anns[:limit]

        print(f"[Evaluator] Evaluating COCOA on {len(anns)} objects...")

        for ann in tqdm(anns):
            img_path = os.path.join(img_dir, ann['filename'])
            
            if not os.path.exists(img_path):
                # Try cleaning filename if it has ./
                img_path = os.path.join(img_dir, ann['filename'].replace("./", ""))
                if not os.path.exists(img_path):
                    continue
                
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # 1. Decode Modal Mask (Visible)
            # COCOA usually provides visible_mask in 'visible_mask' field as RLE
            if 'visible_mask' in ann:
                visible_mask = mask_utils.decode(ann['visible_mask']).astype(bool)
            else:
                # Fallback to standard COCO 'segmentation' if it's the visible one
                visible_mask = self._decode_any_mask(ann.get('segmentation'), h, w)
            
            # 2. Decode Amodal Mask (Full Shape)
            # In full COCOA, the 'segmentation' field is the AMODAL shape
            gt_amodal_mask = self._decode_any_mask(ann.get('segmentation'), h, w)
            
            # Use COCOA's object name (e.g. "dog", "frisbee") for CLIP-text alignment.
            category = ann.get('name') or ann.get('category', '') or 'object'

            # Run Pipeline
            try:
                with _silence_output(enabled=not verbose):
                    output = self.completer.complete(image, visible_mask, all_masks=[])
                pred_image = output['inpainted_rgba'][:, :, :3]

                # Mask both images by visible_mask — Ao et al. protocol: compare
                # visible part of the object with the completed version.
                image_v = _apply_visible_mask(image, visible_mask)
                pred_v = _apply_visible_mask(pred_image, visible_mask)

                metrics = {
                    'filename': ann['filename'],
                    'category': category,
                    'CLIP_score': calculate_clip_score(pred_image, category),
                    'LPIPS': calculate_lpips(pred_v, image_v),
                    'Feature_Similarity': calculate_feature_similarity(pred_v, image_v),
                    'SSIM': calculate_ssim(pred_v, image_v),
                    'PSNR': calculate_psnr(pred_v, image_v),
                }
                self.results.append(metrics)
            except Exception as e:
                print(f"Error processing {ann['filename']}: {e}")

        self.save_results("cocoa_subset_results.csv")

    def _decode_any_mask(self, seg, h, w):
        from pycocotools import mask as mask_utils
        if not seg:
            return np.zeros((h, w), dtype=bool)
        
        if isinstance(seg, list):
            # Polygon format
            rles = mask_utils.frPyObjects(seg, h, w)
            return mask_utils.decode(rles).any(axis=2).astype(bool)
        elif isinstance(seg, dict) and 'counts' in seg:
            # RLE format
            return mask_utils.decode(seg).astype(bool)
        return np.zeros((h, w), dtype=bool)

    def evaluate_huggingface(self, dataset_name="shunk031/COCOA", split="validation", limit=100, verbose=False):
        """
        Evaluate directly from a Hugging Face dataset.
        Solves the storage issue by streaming/caching efficiently.
        """
        print(f"[Evaluator] Loading HF Dataset: {dataset_name} ({split})...")
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        
        if limit:
            # Take a subset for evaluation
            indices = range(min(limit, len(ds)))
            ds = ds.select(indices)

        print(f"[Evaluator] Starting evaluation on {len(ds)} samples...")

        for i, sample in enumerate(tqdm(ds)):
            image_pil = sample['image'].convert("RGB")
            image_np = np.array(image_pil)
            
            # Extract masks from HF dataset structure (Aligning with shunk031/COCOA)
            # Both modal_seg and amodal_seg are stored as PIL images in this dataset
            visible_mask = np.array(sample['modal_seg']).astype(bool)
            gt_amodal_mask = np.array(sample['amodal_seg']).astype(bool)
            
            # If mask is RGB, convert to single channel
            if len(visible_mask.shape) == 3:
                visible_mask = visible_mask[:, :, 0]
            if len(gt_amodal_mask.shape) == 3:
                gt_amodal_mask = gt_amodal_mask[:, :, 0]
            
            category = sample.get('class_name') or sample.get('category') or 'object'

            try:
                # Run Pipeline
                with _silence_output(enabled=not verbose):
                    output = self.completer.complete(image_np, visible_mask, all_masks=[])
                pred_image = output['inpainted_rgba'][:, :, :3]  # Get RGB from RGBA

                image_v = _apply_visible_mask(image_np, visible_mask)
                pred_v = _apply_visible_mask(pred_image, visible_mask)

                result = {
                    "id": i,
                    "category": category,
                    "CLIP_score": calculate_clip_score(pred_image, category),
                    "LPIPS": calculate_lpips(pred_v, image_v),
                    "Feature_Similarity": calculate_feature_similarity(pred_v, image_v),
                    "SSIM": calculate_ssim(pred_v, image_v),
                    "PSNR": calculate_psnr(pred_v, image_v),
                }
                self.results.append(result)

            except Exception as e:
                print(f"Error on sample {i}: {e}")

        self.save_results("hf_eval_results.csv")

    def save_results(self, filename):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\n[Evaluator] Results saved to {filename}")
        for col in ['CLIP_score', 'LPIPS', 'Feature_Similarity', 'SSIM', 'PSNR']:
            if col in df.columns:
                print(f"Mean {col}: {df[col].mean():.4f}")

if __name__ == "__main__":
    # Example usage (adjust paths for your Colab environment)
    # evaluator = AmodalEvaluator()
    # evaluator.evaluate_cocoa("path/to/cocoa.json", "path/to/images", limit=50)
    pass
