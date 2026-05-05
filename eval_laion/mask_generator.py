"""
eval_laion/mask_generator.py

Auto-generate visible masks for LAION images using SAM2 + CLIP label matching.

Since LAION has no ground-truth masks, we:
  1. Run SAM2 auto-segmentation to get all candidate masks
  2. Crop each mask region and score it against the category label via CLIP
  3. Select the mask with the highest CLIP similarity → visible mask

This mirrors the Ao et al. CVPR 2025 protocol where the visible mask is
produced by the pipeline itself rather than read from annotations.
"""

import numpy as np
import torch
from PIL import Image
from typing import Optional


class CLIPMaskSelector:
    """
    Given a set of SAM masks and a text label, selects the mask
    whose cropped region best matches the label in CLIP embedding space.
    """

    _clip_model = None
    _clip_processor = None

    def __init__(self, clip_model_id: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda"):
        self.device = device
        self._load_clip(clip_model_id)

    def _load_clip(self, clip_model_id: str):
        if CLIPMaskSelector._clip_model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        print("[CLIPMaskSelector] Loading CLIP model for mask selection...")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        CLIPMaskSelector._clip_model = CLIPModel.from_pretrained(
            clip_model_id, torch_dtype=dtype
        ).to(self.device)
        CLIPMaskSelector._clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        print("[CLIPMaskSelector] CLIP loaded.")

    def select_best_mask(
        self,
        image: np.ndarray,
        masks: list[dict],
        label: str,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.85,
    ) -> tuple[np.ndarray, int, float]:
        """
        Select the SAM mask that best matches `label` via CLIP.

        Args:
            image:          H×W×3 uint8 RGB
            masks:          list of SAM mask dicts (must have 'segmentation' key)
            label:          category label (e.g. "violin", "car")
            min_area_ratio: skip masks smaller than this fraction of image area
            max_area_ratio: skip masks larger than this (likely background)

        Returns:
            (best_mask_bool, best_index, best_score)
        """
        H, W = image.shape[:2]
        total_pixels = H * W

        # Filter masks by size
        candidates = []
        for i, m in enumerate(masks):
            seg = m["segmentation"].astype(bool)
            area_ratio = seg.sum() / total_pixels
            if min_area_ratio <= area_ratio <= max_area_ratio:
                candidates.append((i, seg))

        if not candidates:
            # Fallback: use the mask with largest area that isn't too big
            for i, m in enumerate(masks):
                seg = m["segmentation"].astype(bool)
                area_ratio = seg.sum() / total_pixels
                if area_ratio <= 0.95:
                    candidates.append((i, seg))
            if not candidates:
                # Absolute fallback: just use the first mask
                return masks[0]["segmentation"].astype(bool), 0, 0.0

        # Crop each candidate region and score against label
        crops = []
        for _, seg in candidates:
            crop = self._crop_masked_region(image, seg)
            crops.append(crop)

        scores = self._batch_clip_score(crops, label)
        best_local_idx = int(np.argmax(scores))
        best_global_idx = candidates[best_local_idx][0]
        best_mask = candidates[best_local_idx][1]
        best_score = float(scores[best_local_idx])

        return best_mask, best_global_idx, best_score

    def _crop_masked_region(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Crop the bounding box of the mask, set non-mask pixels to neutral gray.
        """
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return image.copy()
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        crop = image[y1:y2 + 1, x1:x2 + 1].copy()
        crop_mask = mask[y1:y2 + 1, x1:x2 + 1]
        crop[~crop_mask] = 128  # neutral gray background
        return crop

    @torch.no_grad()
    def _batch_clip_score(self, crops: list[np.ndarray], label: str) -> np.ndarray:
        """
        Compute CLIP cosine similarity between each crop and the text label.
        """
        model = CLIPMaskSelector._clip_model
        processor = CLIPMaskSelector._clip_processor

        # Prepare images
        pil_crops = [Image.fromarray(c) for c in crops]

        # Process text
        text_inputs = processor.tokenizer(
            [label] * len(crops),
            padding="max_length", max_length=77, truncation=True,
            return_tensors="pt",
        )
        # Process images
        image_inputs = processor.image_processor(pil_crops, return_tensors="pt")

        inputs = {
            "input_ids": text_inputs["input_ids"].to(self.device),
            "attention_mask": text_inputs["attention_mask"].to(self.device),
            "pixel_values": image_inputs["pixel_values"].to(self.device),
        }

        outputs = model(**inputs)
        # Cosine similarity per (image, text) pair along diagonal
        img_embeds = outputs.image_embeds  # (N, D)
        txt_embeds = outputs.text_embeds   # (N, D)

        # Normalize
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

        # Per-pair cosine similarity
        scores = (img_embeds * txt_embeds).sum(dim=-1)
        return scores.cpu().numpy()

    def cleanup(self):
        if CLIPMaskSelector._clip_model is not None:
            del CLIPMaskSelector._clip_model
            CLIPMaskSelector._clip_model = None
        if CLIPMaskSelector._clip_processor is not None:
            del CLIPMaskSelector._clip_processor
            CLIPMaskSelector._clip_processor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
