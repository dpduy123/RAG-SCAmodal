"""
pipeline/amodal_completer.py

Open-World Amodal Appearance Completion pipeline.
Given:
  - RGB image
  - visible mask of the target object (from SAM)
  - all other masks in the scene
  - optional text query

Returns: RGBA image (H×W×4) where the 4th channel is the amodal mask.

Pipeline (follows CVPR 2025 paper):
  1. Occlusion analysis  → identify which masks occlude the target
  2. Prompt selection    → CLIP-based best descriptor
  3. Image preparation   → isolate target on clean background
  4. Iterative inpainting → Stable Diffusion v2 inpainting
  5. Alpha blending      → seamless merge + RGBA output
"""

import numpy as np
from typing import Optional
import json
import os
import torch
import cv2
from agents import SemanticAgent, GeometryAgent, MultiAgentCritic
from memory_bank import ZillizMemoryBank
from dataclasses import dataclass
from PIL import Image

class AmodalCompleter:

    # Shared class-level instances to prevent redundant loading across instances
    _pipe = None
    _clip_model = None
    _clip_processor = None
    _shape_predictor = None
    _vlm = None
    _memory_bank = None
    _semantic_agent = None
    _geometry_agent = None
    _critic_agent = None

    def __init__(
        self,
        inpainting_model_id: str = "sd2-community/stable-diffusion-2-inpainting",
        clip_model_id: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize instances only if not already loaded globally
        if AmodalCompleter._shape_predictor is None:
            from amodal_shape_predictor import Pix2GestaltPredictor
            AmodalCompleter._shape_predictor = Pix2GestaltPredictor(device=self.device)
            AmodalCompleter._geometry_agent = GeometryAgent(AmodalCompleter._shape_predictor)
            
        if AmodalCompleter._vlm is None:
            from vlm_reasoner import VLMReasoner
            AmodalCompleter._vlm = VLMReasoner(device=self.device)
            AmodalCompleter._semantic_agent = SemanticAgent(AmodalCompleter._vlm)
            AmodalCompleter._critic_agent = MultiAgentCritic(AmodalCompleter._vlm)
            
        if AmodalCompleter._memory_bank is None:
            AmodalCompleter._memory_bank = ZillizMemoryBank(device=self.device)
            
        self._load_models(inpainting_model_id, clip_model_id)

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_models(self, inpainting_model_id: str, clip_model_id: str):
        if AmodalCompleter._pipe is not None:
             # Already loaded globally
             return

        print("[AmodalCompleter] Loading Stable Diffusion inpainting model...")
        from diffusers import StableDiffusionInpaintPipeline

        AmodalCompleter._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inpainting_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        )
        AmodalCompleter._pipe.to(self.device)
        AmodalCompleter._pipe.set_progress_bar_config(disable=True)

        # ── CLIP for Prompt Selection ──
        if AmodalCompleter._clip_model is None:
            print("[AmodalCompleter] Loading CLIP model...")
            from transformers import CLIPModel, CLIPProcessor
            clip_dtype = torch.float16 if self.device == "cuda" else torch.float32
            AmodalCompleter._clip_model = CLIPModel.from_pretrained(
                clip_model_id, torch_dtype=clip_dtype
            ).to(self.device)
            AmodalCompleter._clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        
        print("[AmodalCompleter] Models loaded.")

    def cleanup(self):
        """
        Explicitly removes all models from GPU memory.
        Use this when done with the pipeline to free ~25GB VRAM.
        """
        print("[AmodalCompleter] Cleaning up models and freeing VRAM...")
        
        if AmodalCompleter._shape_predictor is not None:
            AmodalCompleter._shape_predictor.cleanup()
            AmodalCompleter._shape_predictor = None
            
        if AmodalCompleter._vlm is not None:
            del AmodalCompleter._vlm
            AmodalCompleter._vlm = None

        if AmodalCompleter._pipe is not None:
            del AmodalCompleter._pipe
            AmodalCompleter._pipe = None
            
        if AmodalCompleter._clip_model is not None:
            del AmodalCompleter._clip_model
            AmodalCompleter._clip_model = None
            
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[AmodalCompleter] Cleanup complete.")


        print("[AmodalCompleter] Models loaded.")

    # ── Main entry point ───────────────────────────────────────────────────

    def complete(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        all_masks: list[dict],
        text_query: str = "",
        max_iter: int = 3,
        epsilon: float = 0.01,
        enable_critic: bool = True,
        critique_threshold: float = 7.0,
        max_critic_iter: int = 2,
        mask_only: bool = False,
    ) -> dict:
        """
        Args:
            image:               H×W×3 uint8 RGB
            visible_mask:        H×W bool — visible region of target object
            all_masks:           list of SAM mask dicts (other objects in scene)
            text_query:          optional text description of the target
            max_iter:            (legacy) max inpainting iterations
            epsilon:             (legacy) convergence threshold
            enable_critic:       run Stage 3 semantic critique loop
            critique_threshold:  τ — minimum acceptable critic score in [0, 10]
            max_critic_iter:     maximum number of synthesis attempts
            mask_only:           if True, skip SD2 diffusion entirely;
                                 output RGBA = visible pixels + amodal mask shape

        Returns:
            dict with input_image, visible_mask, amodal_mask, inpainted_rgba,
            vlm_reasoning, critique_history, final_score.
        """
        H, W = image.shape[:2]

        # ── RAG-SCAmodal: Retrieval-Augmented Generation ──────────────────
        print("[AmodalCompleter] Step 1: Object Crop & CLIP Embedding...")
        ys, xs = np.where(visible_mask)
        if len(xs) == 0:
            crop_np = image
        else:
            # Add 10% margin
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            w_m, h_m = int((x_max - x_min) * 0.1), int((y_max - y_min) * 0.1)
            x1, y1 = max(0, x_min - w_m), max(0, y_min - h_m)
            x2, y2 = min(W, x_max + w_m), min(H, y_max + h_m)
            crop_np = image[y1:y2, x1:x2].copy()
            # Gray out background within the crop
            crop_mask = visible_mask[y1:y2, x1:x2]
            crop_np[~crop_mask] = 127
            
        query_embed = AmodalCompleter._memory_bank.extract_dual_features(crop_np)
        
        print("[AmodalCompleter] Step 2: Retrieving Top-K shape priors from Memory Bank...")
        top_k_priors = AmodalCompleter._memory_bank.retrieve(query_embed, top_k=5)

        print("[AmodalCompleter] Step 3: Semantic reasoning (SemanticAgent)...")
        vlm_guidance = AmodalCompleter._semantic_agent.reason(image, visible_mask)
        print(f"[Semantic Reason]: {vlm_guidance}")

        print("[AmodalCompleter] Step 4: Geometric prediction (GeometryAgent - Best-of-N)...")
        # Geometry Agent now returns a list of hypotheses
        amodal_hypotheses = AmodalCompleter._geometry_agent.predict(image, visible_mask, top_k_priors, vlm_guidance, lambda_rag_threshold=0.6)
        
        # We will keep track of pad_info using the first hypothesis (M1) as reference
        pad_info = {"needs_outpaint": False, "pad_top": 0, "pad_bottom": 0, "pad_left": 0, "pad_right": 0}
        
        # ── OUTPAINTING DETECTION ─────────────────────────────────────────
        # If the missing region touches image edges, the object is cropped.
        # Expand the canvas so SD2 can generate content beyond boundaries.
        detect_info = self._detect_outpainting(missing_mask, H, W)
        
        # [TEMPORARILY DISABLED]
        detect_info["needs_outpaint"] = False 
        
        if detect_info["needs_outpaint"]:
            pad_info = detect_info
            print(f"[AmodalCompleter] 🔲 Outpainting detected! "
                  f"Expanding canvas: top={pad_info['pad_top']}, "
                  f"bottom={pad_info['pad_bottom']}, "
                  f"left={pad_info['pad_left']}, right={pad_info['pad_right']}")
            image, visible_mask, amodal_mask, missing_mask = \
                self._expand_for_outpainting(image, visible_mask, amodal_mask, pad_info)
            H, W = image.shape[:2]  # update dimensions

        # ── MASK-ONLY MODE ────────────────────────────────────────────────
        # Skip SD2 diffusion; return visible pixels + the best amodal mask shape (M1 by default).
        best_amodal_mask = amodal_hypotheses[0]
        if mask_only:
            print("[AmodalCompleter] mask_only=True → skipping SD2 inpainting")
            rgba = self._finalize_rgba(image, best_amodal_mask, H, W)
            return {
                "input_image": image,
                "visible_mask": visible_mask,
                "amodal_mask": best_amodal_mask,
                "inpainted_rgba": rgba,
                "vlm_reasoning": vlm_guidance,
                "critique_history": [],
                "final_score": None,
                "pad_info": pad_info,
            }

        # Fuse the two guidance signals via CLIP-verified Qwen reasoning.
        base_prompt, clip_info = self._select_prompt(
            image, visible_mask, qwen_text=vlm_guidance, user_query=text_query
        )
        target_img, _ = self._prepare_target_image(image, visible_mask)

        # ── SYNTHESIS + BEST-OF-N CRITIC LOOP ─────────
        sd_tok = AmodalCompleter._pipe.tokenizer
        prompt = self._truncate_to_tokens(
            f"{base_prompt}, centered, high quality, consistent lighting",
            sd_tok, max_tokens=70,
        )
        guidance_scale = 7.5
        num_inference_steps = 30
        
        best_candidate = None
        best_score = -1.0
        best_rgba = None
        best_amodal_mask = None
        best_critique = None
        
        critique_history = []
        
        for idx, current_amodal_mask in enumerate(amodal_hypotheses):
            current_missing_mask = current_amodal_mask & (~visible_mask.astype(bool))
            if not current_missing_mask.any():
                continue
                
            print(f"[AmodalCompleter] Synthesis attempt for Hypothesis M{idx+1}/{len(amodal_hypotheses)}")

            inpaint_img, _ = self._inpaint_step(
                target_img, current_missing_mask, prompt, H, W,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            blended_rgb = self._alpha_blend(image, inpaint_img, visible_mask, current_amodal_mask)
            rgba = self._finalize_rgba(blended_rgb, current_amodal_mask, H, W)

            if not enable_critic:
                best_rgba = rgba
                best_amodal_mask = current_amodal_mask
                break

            # Stage 3 — Semantic Critic: MultiAgentCritic evaluates the completed object
            critic_input = self._compose_for_critic(blended_rgb, current_amodal_mask)
            critique = AmodalCompleter._critic_agent.evaluate(critic_input, original_image_np=image)
            critique["hypothesis_id"] = f"M{idx+1}"
            critique_history.append(critique)
            
            print(f"[Critic M{idx+1}] score={critique['score']:.2f} "
                  f"(struct={critique['structural']}, tex={critique['texture']}, "
                  f"ctx={critique['context']})")

            if critique["score"] > best_score:
                best_score = critique["score"]
                best_rgba = rgba
                best_amodal_mask = current_amodal_mask
                best_critique = critique
                
        # If no missing regions were found in any hypothesis
        if best_rgba is None:
            best_amodal_mask = amodal_hypotheses[0]
            best_rgba = self._finalize_rgba(image, best_amodal_mask, H, W)
            
        print(f"[AmodalCompleter] Best-of-N selected Candidate with score {best_score:.2f}")

        # Persist VLM reasoning + critique trace for logging/inspection.
        try:
            with open("text_prompt.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "vlm_reasoning": vlm_guidance,
                        "clip_verification": clip_info,
                        "critique_history": [
                            {k: v for k, v in c.items() if k != "raw"}
                            for c in critique_history
                        ],
                    },
                    f, indent=4, ensure_ascii=False,
                )
        except Exception as e:
            print(f"[AmodalCompleter] Warning: Could not save text_prompt.json: {e}")

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return {
            "input_image": image,
            "visible_mask": visible_mask,
            "amodal_mask": best_amodal_mask,
            "inpainted_rgba": best_rgba,
            "vlm_reasoning": vlm_guidance,
            "clip_verification": clip_info,
            "critique_history": critique_history,
            "final_score": best_score if enable_critic else None,
            "pad_info": pad_info,
        }

    def _compose_for_critic(
        self, blended_rgb: np.ndarray, amodal_mask: np.ndarray
    ) -> np.ndarray:
        """
        Place the completed amodal object on a neutral white canvas so the critic
        evaluates the object itself rather than the surrounding scene clutter.
        """
        canvas = np.full_like(blended_rgb, 255, dtype=np.uint8)
        m = amodal_mask[:, :, np.newaxis] if amodal_mask.ndim == 2 else amodal_mask
        canvas = np.where(m, blended_rgb, canvas).astype(np.uint8)
        return canvas

    def _finalize_rgba(self, image, amodal_mask, H, W):
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = (amodal_mask * 255).astype(np.uint8)
        return rgba

    # ── Step 1: Occlusion analysis ─────────────────────────────────────────

    def _build_occluder_mask(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        all_masks: list[dict],
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        Identify which segments occlude the target object.

        Key insight: SAM produces non-overlapping masks, so adjacent masks
        (touching the target boundary) are likely occluders. We dilate the
        target mask to detect these adjacencies.
        """
        occluder = np.zeros((H, W), dtype=bool)

        # Dilate the visible mask to find nearby/adjacent masks
        # This is critical because SAM masks typically DON'T overlap
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dilated_target = cv2.dilate(
            visible_mask.astype(np.uint8), dilate_kernel
        ).astype(bool)
        target_border_zone = dilated_target & (~visible_mask)

        # Estimate full amodal extent using multiple strategies:
        # 1. Convex hull (captures overall shape)
        target_hull = self._convex_hull_mask(visible_mask, H, W)
        # 2. Large dilation (captures nearby hidden parts like legs)
        large_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
        dilated_full = cv2.dilate(
            visible_mask.astype(np.uint8), large_dilate
        ).astype(bool)
        # 3. Downward extension (legs/feet typically extend below visible body)
        downward_ext = self._extend_downward(visible_mask, H, W, extend_ratio=0.3)
        # Combine all estimates
        full_extent = target_hull | dilated_full | downward_ext
        # The "missing" region: inside full extent but not in visible mask
        estimated_hidden = full_extent & (~visible_mask)

        for i, m in enumerate(all_masks):
            seg = m["segmentation"].astype(bool)

            # Skip if this is the target mask itself
            if np.array_equal(seg, visible_mask):
                continue

            # Check 1: Direct overlap (rare with SAM, but possible)
            direct_overlap = seg & visible_mask
            # Check 2: Adjacent — mask touches the dilated target boundary
            adjacent_overlap = seg & target_border_zone
            # Check 3: Mask covers the estimated hidden region
            hidden_overlap = seg & estimated_hidden

            if not direct_overlap.any() and not adjacent_overlap.any() and not hidden_overlap.any():
                continue

            # Score how likely this mask is an occluder
            adjacency_score = adjacent_overlap.sum() / max(target_border_zone.sum(), 1)
            hidden_score = hidden_overlap.sum() / max(estimated_hidden.sum(), 1)

            # If the mask significantly overlaps with where we expect hidden parts
            # OR it's strongly adjacent to the target boundary → it's an occluder
            if hidden_score > 0.05 or adjacency_score > 0.03 or direct_overlap.any():
                try:
                    is_occluder = self._check_occlusion_order(image, visible_mask, seg)
                except Exception:
                    # Default: assume adjacent masks in the hidden region are occluders
                    is_occluder = hidden_score > 0.02 or adjacency_score > 0.02
                if is_occluder:
                    occluder |= seg

        # Handle boundary cases: expand along image edges
        occluder = self._handle_boundary_occlusion(visible_mask, occluder, H, W)

        # The actual region to inpaint: estimated hidden area covered by occluders
        # (not the full occluder — we only want to reveal what's behind it)
        inpaint_region = estimated_hidden | (occluder & full_extent)

        # Also add a border expansion to ensure smooth completion
        expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        inpaint_region = cv2.dilate(
            inpaint_region.astype(np.uint8), expand_kernel
        ).astype(bool)

        # Don't inpaint inside the already-visible mask
        inpaint_region = inpaint_region & (~visible_mask)

        print(f"[AmodalCompleter] Occluder masks found: {occluder.any()}")
        print(f"[AmodalCompleter] Estimated hidden region: {estimated_hidden.sum()} px")
        print(f"[AmodalCompleter] Inpaint region: {inpaint_region.sum()} px")

        return inpaint_region

    def _convex_hull_mask(
        self, mask: np.ndarray, H: int, W: int
    ) -> np.ndarray:
        """
        Compute the convex hull of a binary mask.
        The convex hull estimates the full amodal extent of the object.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return mask.copy()

        # Merge all contours and find convex hull
        all_points = np.concatenate(contours)
        hull = cv2.convexHull(all_points)

        hull_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        return hull_mask.astype(bool)

    def _extend_downward(
        self, mask: np.ndarray, H: int, W: int, extend_ratio: float = 0.3
    ) -> np.ndarray:
        """
        Extend the mask downward to account for hidden legs/feet.
        Objects like animals/people typically have parts extending below
        the visible body when occluded by foreground objects.
        """
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return mask.copy()

        # Get bottom edge of the visible mask
        x_min, x_max = xs.min(), xs.max()
        y_max = ys.max()
        mask_height = ys.max() - ys.min()

        # Extend downward by extend_ratio of the mask height
        extend_px = int(mask_height * extend_ratio)
        y_extend = min(y_max + extend_px, H - 1)

        # Create extension region (tapered trapezoid shape)
        extended = mask.copy()
        for y in range(y_max, y_extend):
            # Gradually narrow the extension
            progress = (y - y_max) / max(extend_px, 1)
            shrink = int((x_max - x_min) * 0.15 * progress)
            x_left = max(x_min + shrink, 0)
            x_right = min(x_max - shrink, W - 1)
            if x_left < x_right:
                extended[y, x_left:x_right] = True

        return extended

    def _check_occlusion_order(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> bool:
        """
        Check if candidate_mask occludes target_mask.
        Uses InstaOrderNet if available, otherwise falls back to heuristic.
        """
        try:
            return self._instaorder_check(image, target_mask, candidate_mask)
        except Exception:
            return self._heuristic_occlusion_check(target_mask, candidate_mask)

    def _instaorder_check(self, image, target_mask, candidate_mask) -> bool:
        """
        Use InstaOrderNet for pairwise occlusion order prediction.
        InstaOrderNet: https://github.com/HnKnA/InstaOrderNet
        """
        # InstaOrderNet takes (image, mask1, mask2) and returns occlusion probability
        # Install: pip install instaordernet
        import instaordernet
        model = instaordernet.load_model()  # cached after first call
        prob = model.predict_occlusion(image, target_mask, candidate_mask)
        return prob > 0.5  # candidate occludes target if prob > 0.5

    def _heuristic_occlusion_check(
        self, target_mask: np.ndarray, candidate_mask: np.ndarray
    ) -> bool:
        """
        Fallback heuristic: a segment likely occludes the target if:
        - It is adjacent to the target's boundary
        - It covers area where the target's convex hull extends
        - Smaller objects in front of larger ones are likely occluders
        """
        target_area = target_mask.sum()
        candidate_area = candidate_mask.sum()

        # Dilate target to check adjacency
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_target = cv2.dilate(target_mask.astype(np.uint8), kernel).astype(bool)
        adjacency = (candidate_mask & dilated_target & ~target_mask).sum()

        # If there's significant adjacency, likely an occluder
        if adjacency > 50:
            return True

        # Erode target mask to get interior
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        target_interior = cv2.erode(target_mask.astype(np.uint8), erode_kernel).astype(bool)
        target_boundary = target_mask & (~target_interior)

        # Overlap with boundary (not interior) → likely occlusion
        boundary_overlap = candidate_mask & target_boundary
        interior_overlap = candidate_mask & target_interior
        return boundary_overlap.sum() > interior_overlap.sum() * 0.3

    def _handle_boundary_occlusion(
        self,
        visible_mask: np.ndarray,
        occluder_mask: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        If the target touches image boundaries, dilate the occluder mask
        along those edges (the image frame itself is an occluder).
        """
        edges_touched = {
            "top": visible_mask[0, :].any(),
            "bottom": visible_mask[-1, :].any(),
            "left": visible_mask[:, 0].any(),
            "right": visible_mask[:, -1].any(),
        }

        if not any(edges_touched.values()):
            return occluder_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_occ = cv2.dilate(occluder_mask.astype(np.uint8), kernel).astype(bool)

        # Only add dilation near the touched edges
        edge_region = np.zeros((H, W), dtype=bool)
        margin = 30
        if edges_touched["top"]:    edge_region[:margin, :] = True
        if edges_touched["bottom"]: edge_region[-margin:, :] = True
        if edges_touched["left"]:   edge_region[:, :margin] = True
        if edges_touched["right"]:  edge_region[:, -margin:] = True

        occluder_mask = occluder_mask | (dilated_occ & edge_region)
        return occluder_mask

    # ── Step 2: CLIP verification of Qwen reasoning ────────────────────────

    @staticmethod
    def _truncate_words(text: str, max_words: int = 45) -> str:
        """Word-level truncation (rough fallback)."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    @staticmethod
    def _truncate_to_tokens(text: str, tokenizer, max_tokens: int = 70) -> str:
        """
        Token-accurate truncation using a HuggingFace tokenizer.
        Caps at `max_tokens` (leaving room for BOS/EOS in 77-token CLIP encoders).
        Falls back to word truncation if the tokenizer fails.
        """
        try:
            ids = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
            if len(ids) <= max_tokens:
                return text
            return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
        except Exception:
            return AmodalCompleter._truncate_words(text, max_words=max_tokens)

    def _select_prompt(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        qwen_text: str,
        user_query: str = "",
    ) -> tuple[str, dict]:
        """
        Use CLIP to VERIFY (not replace) Qwen's reasoning.

        Strategy:
          1. Truncate Qwen text to fit 77-token CLIP limit.
          2. Score Qwen text + a small set of generic fallbacks against the
             cropped visible region.
          3. Accept Qwen if its CLIP score is competitive with the best generic
             label (within `margin`) AND above an absolute floor.
          4. Otherwise fall back to the best generic label — treating the
             Qwen output as hallucinated / off-topic for the visible region.

        Returns: (prompt_for_SD, info_dict)
        """
        ys, xs = np.where(visible_mask)
        if len(xs) == 0:
            return user_query or "object", {"verified": False, "reason": "empty mask"}

        # Crop visible region; mask non-target pixels to neutral gray.
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        crop = image[y1:y2 + 1, x1:x2 + 1].copy()
        crop_mask = visible_mask[y1:y2 + 1, x1:x2 + 1]
        crop[~crop_mask] = 128
        crop_pil = Image.fromarray(crop)

        # Token-accurate truncation against CLIP's own tokenizer (77-token limit).
        clip_tok = AmodalCompleter._clip_processor.tokenizer
        qwen_short = self._truncate_to_tokens(qwen_text, clip_tok, max_tokens=70)

        fallbacks = [
            "object", "person", "animal", "vehicle", "furniture", "foreground object",
        ]
        if user_query and user_query.strip():
            fallbacks.insert(0, user_query.strip())
        candidates = [qwen_short] + fallbacks
        candidates = [c for c in candidates if c and c.strip()]

        # Tokenize text and process image SEPARATELY so we have full control over
        # truncation. The joint CLIPProcessor call has been observed to skip
        # length capping in transformers 5.0.0.
        text_inputs = clip_tok(
            candidates,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        # Hard guarantee: any token tensor we forward must be ≤ 77 long.
        if text_inputs["input_ids"].shape[1] > 77:
            text_inputs = {k: v[:, :77] for k, v in text_inputs.items()}

        image_inputs = AmodalCompleter._clip_processor.image_processor(
            [crop_pil] * len(candidates),
            return_tensors="pt",
        )
        clip_inputs = {
            "input_ids": text_inputs["input_ids"].to(self.device),
            "attention_mask": text_inputs["attention_mask"].to(self.device),
            "pixel_values": image_inputs["pixel_values"].to(self.device),
        }

        with torch.no_grad():
            outputs = AmodalCompleter._clip_model(**clip_inputs)
            sims = (outputs.image_embeds[0:1] @ outputs.text_embeds.T).squeeze(0)
            sims_list = sims.detach().cpu().tolist()

        qwen_score = sims_list[0]
        fallback_scores = sims_list[1:]
        best_fb_idx = max(range(len(fallback_scores)), key=lambda i: fallback_scores[i])
        best_fallback = candidates[1 + best_fb_idx]
        best_fb_score = fallback_scores[best_fb_idx]

        # Verification thresholds — cosine similarity in normalized CLIP space.
        ABS_FLOOR = 0.18      # below this Qwen is deemed clearly off-topic
        MARGIN = 0.03         # Qwen acceptable if at most 0.03 below best fallback

        verified = (qwen_score >= ABS_FLOOR) and (qwen_score >= best_fb_score - MARGIN)
        chosen = qwen_short if verified else best_fallback

        if not verified:
            print(f"[CLIP Verify] Qwen REJECTED "
                  f"(qwen={qwen_score:.3f}, best_fallback='{best_fallback}'={best_fb_score:.3f}) "
                  f"→ using fallback.")
        else:
            print(f"[CLIP Verify] Qwen ACCEPTED "
                  f"(qwen={qwen_score:.3f} vs best_fallback='{best_fallback}'={best_fb_score:.3f}).")

        info = {
            "verified": bool(verified),
            "qwen_clip_score": float(qwen_score),
            "best_fallback": best_fallback,
            "best_fallback_score": float(best_fb_score),
            "chosen": chosen,
        }
        return chosen, info

    # ── Step 3: Target image preparation ──────────────────────────────────

    # ── Outpainting support ──────────────────────────────────────────────

    def _detect_outpainting(
        self,
        missing_mask: np.ndarray,
        H: int,
        W: int,
        edge_margin: int = 5,
    ) -> dict:
        """
        Detect if the missing region touches image boundaries (outpainting case).
        Returns padding amounts for each edge.
        """
        touches_top    = missing_mask[:edge_margin, :].any()
        touches_bottom = missing_mask[-edge_margin:, :].any()
        touches_left   = missing_mask[:, :edge_margin].any()
        touches_right  = missing_mask[:, -edge_margin:].any()

        needs = touches_top or touches_bottom or touches_left or touches_right

        if not needs:
            return {"needs_outpaint": False,
                    "pad_top": 0, "pad_bottom": 0,
                    "pad_left": 0, "pad_right": 0}

        # Pad by 25% of image dimension on each touching edge (min 64px)
        pad_h = max(H // 4, 64)
        pad_w = max(W // 4, 64)

        return {
            "needs_outpaint": True,
            "pad_top":    pad_h if touches_top    else 0,
            "pad_bottom": pad_h if touches_bottom else 0,
            "pad_left":   pad_w if touches_left   else 0,
            "pad_right":  pad_w if touches_right  else 0,
        }

    def _expand_for_outpainting(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        amodal_mask: np.ndarray,
        pad_info: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Expand image and masks with padding for outpainting.
        The amodal mask is dilated into the padded region to give SD2
        a target area to generate content.

        Returns: (expanded_image, expanded_visible, expanded_amodal, expanded_missing)
        """
        pt = pad_info["pad_top"]
        pb = pad_info["pad_bottom"]
        pl = pad_info["pad_left"]
        pr = pad_info["pad_right"]

        # Pad image with neutral gray
        exp_image = cv2.copyMakeBorder(
            image, pt, pb, pl, pr,
            cv2.BORDER_CONSTANT, value=[127, 127, 127]
        )

        # Pad masks with zeros (False)
        exp_visible = cv2.copyMakeBorder(
            visible_mask.astype(np.uint8), pt, pb, pl, pr,
            cv2.BORDER_CONSTANT, value=0
        ).astype(bool)

        exp_amodal = cv2.copyMakeBorder(
            amodal_mask.astype(np.uint8), pt, pb, pl, pr,
            cv2.BORDER_CONSTANT, value=0
        ).astype(bool)

        # Dilate amodal mask into the padded region so SD2 knows where
        # to generate content. Use an elliptical kernel sized to the padding.
        max_pad = max(pt, pb, pl, pr)
        kernel_size = max_pad * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        exp_amodal_dilated = cv2.dilate(
            exp_amodal.astype(np.uint8), kernel
        ).astype(bool)

        # Only extend amodal into the padded border region,
        # keep the original interior amodal mask unchanged.
        eH, eW = exp_image.shape[:2]
        border_region = np.zeros((eH, eW), dtype=bool)
        border_region[:pt, :] = True                   # top pad
        if pb > 0:
            border_region[-pb:, :] = True              # bottom pad
        border_region[:, :pl] = True                   # left pad
        if pr > 0:
            border_region[:, -pr:] = True              # right pad

        # Final amodal = original padded amodal + dilated extension in borders
        exp_amodal = exp_amodal | (exp_amodal_dilated & border_region)

        # Missing = amodal but not visible
        exp_missing = exp_amodal & (~exp_visible)

        print(f"[Outpainting] Canvas: {image.shape[:2]} → {exp_image.shape[:2]}")
        print(f"[Outpainting] Amodal area: {amodal_mask.sum():,} → {exp_amodal.sum():,} px")
        print(f"[Outpainting] Missing area: {exp_missing.sum():,} px")

        return exp_image, exp_visible, exp_amodal, exp_missing

    def _prepare_target_image(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Isolate the target on a neutral background [127, 127, 127].
        This pure background provides a natural context-free canvas for inpainting.
        """
        background = np.full_like(image, [127, 127, 127], dtype=np.uint8)

        target_img = image.copy().astype(float)
        mask_3ch = visible_mask[:, :, np.newaxis]
        target_img = (target_img * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

        return target_img, background

    # ── Step 4: Inpainting step ────────────────────────────────────────────

    def _inpaint_step(
        self,
        image: np.ndarray,
        occluder_mask: np.ndarray,
        prompt: str,
        H: int,
        W: int,
        inpaint_size: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one iteration of Stable Diffusion inpainting.

        Args:
            image:         current inpainting canvas (H×W×3)
            occluder_mask: H×W bool — regions to inpaint
            prompt:        text conditioning for the inpainting
            H, W:          original image dimensions
            inpaint_size:  SD inference resolution

        Returns:
            (inpainted_image H×W×3, newly_reconstructed_mask H×W bool)
        """
        if not occluder_mask.any():
            return image, np.zeros((H, W), dtype=bool)

        # Resize to SD input resolution
        pil_image = Image.fromarray(image).resize((inpaint_size, inpaint_size), Image.LANCZOS)
        pil_mask = Image.fromarray((occluder_mask * 255).astype(np.uint8)).resize(
            (inpaint_size, inpaint_size), Image.NEAREST
        )

        with torch.inference_mode():
            result = AmodalCompleter._pipe(
                prompt=prompt,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=inpaint_size,
                width=inpaint_size,
            ).images[0]

        # Resize back to original resolution
        result_np = np.array(result.resize((W, H), Image.LANCZOS))

        # Newly reconstructed = pixels that were in occluder_mask and now have content
        newly_reconstructed = occluder_mask.copy()

        # Blend: only replace occluder pixels
        blended = image.copy()
        blended[occluder_mask] = result_np[occluder_mask]

        return blended, newly_reconstructed

    # ── Step 5: Alpha blending ─────────────────────────────────────────────

    def _alpha_blend(
        self,
        original_image: np.ndarray,
        inpainted_image: np.ndarray,
        visible_mask: np.ndarray,
        amodal_mask: np.ndarray,
        transition_px: int = 12,
    ) -> np.ndarray:
        """
        Blend original visible pixels with inpainted reconstructed pixels.
        Creates a smooth transition at the boundary of the visible region.

        Alpha = 1.0  inside visible_mask (use original pixels)
        Alpha = 0→1  in transition zone (blend)
        Alpha = 1.0  in reconstructed region (use inpainted pixels)
        """
        # Distance transform from visible mask boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (transition_px * 2 + 1,) * 2)
        eroded = cv2.erode(visible_mask.astype(np.uint8), kernel)
        transition_zone = visible_mask.astype(np.uint8) - eroded

        # Build alpha map
        dist = cv2.distanceTransform(visible_mask.astype(np.uint8), cv2.DIST_L2, 5)
        alpha_map = np.clip(dist / transition_px, 0, 1)
        alpha_map[~visible_mask] = 0
        alpha_3ch = alpha_map[:, :, np.newaxis]

        # Blend: original inside visible, inpainted outside
        blended = (
            original_image.astype(float) * alpha_3ch +
            inpainted_image.astype(float) * (1 - alpha_3ch)
        ).astype(np.uint8)

        # Only keep amodal region
        result = np.zeros_like(original_image)
        result[amodal_mask] = blended[amodal_mask]

        return result
