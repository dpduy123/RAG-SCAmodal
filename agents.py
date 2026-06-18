import numpy as np
from typing import List, Dict, Tuple
import cv2
from PIL import Image

class SemanticAgent:
    """
    SemanticAgent analyzes the object and reasons about its missing parts.
    It wraps the VLMReasoner to keep responsibilities separated.
    """
    def __init__(self, vlm_reasoner):
        self.vlm = vlm_reasoner

    def reason(self, image_np: np.ndarray, visible_mask: np.ndarray) -> str:
        """
        Produce a text reasoning of what is missing.
        """
        print("[SemanticAgent] Analyzing visible object and scene context...")
        reasoning = self.vlm.reason_occlusion(image_np, visible_mask)
        print(f"[SemanticAgent] Reasoning Output: {reasoning}")
        return reasoning


class MultiAgentCritic:
    """
    MultiAgentCritic evaluates the inpainted image from multiple perspectives.
    Currently it delegates to VLMReasoner's critique method, but in a true Multi-Agent
    setup, this could query different specialized prompts/models for Structural, Texture, etc.
    """
    def __init__(self, vlm_reasoner):
        self.vlm = vlm_reasoner

    def evaluate(self, completed_image_np: np.ndarray, original_image_np: np.ndarray) -> Dict:
        """
        Returns a dictionary with scores and feedback.
        """
        print("[MultiAgentCritic] Evaluating the completed image...")
        evaluation = self.vlm.critique(completed_image_np, original_image_np=original_image_np)
        return evaluation


class GeometryAgent:
    """
    GeometryAgent predicts the final amodal shape by combining Memory Bank's Top-K priors
    with the geometric refinement of Pix2Gestalt.
    """
    def __init__(self, shape_predictor):
        """
        shape_predictor: An instance of Pix2GestaltPredictor
        """
        self.predictor = shape_predictor

    def _align_prior(self, prior_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
        H, W = target_mask.shape
        ys_p, xs_p = np.where(prior_mask)
        if len(ys_p) == 0: return np.zeros_like(target_mask, dtype=bool)
        
        p_y1, p_y2 = ys_p.min(), ys_p.max()
        p_x1, p_x2 = xs_p.min(), xs_p.max()
        prior_crop = prior_mask[p_y1:p_y2+1, p_x1:p_x2+1].astype(np.uint8)
        
        ys_t, xs_t = np.where(target_mask)
        if len(ys_t) == 0: return np.zeros_like(target_mask, dtype=bool)
            
        t_y1, t_y2 = ys_t.min(), ys_t.max()
        t_x1, t_x2 = xs_t.min(), xs_t.max()
        
        target_w = t_x2 - t_x1 + 1
        scale = target_w / max((p_x2 - p_x1 + 1), 1)
        new_w = target_w
        new_h = int((p_y2 - p_y1 + 1) * scale)
        if new_h == 0 or new_w == 0: return np.zeros_like(target_mask, dtype=bool)
            
        prior_resized = cv2.resize(prior_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        aligned_mask = np.zeros_like(target_mask, dtype=bool)
        paste_y2 = min(t_y1 + new_h, H)
        paste_x2 = min(t_x1 + new_w, W)
        paste_h = paste_y2 - t_y1
        paste_w = paste_x2 - t_x1
        aligned_mask[t_y1:paste_y2, t_x1:paste_x2] = prior_resized[:paste_h, :paste_w] > 0
        return aligned_mask

    def predict(self, image_np: np.ndarray, visible_mask: np.ndarray, top_k_priors: List[Dict], semantic_reasoning: str, lambda_rag_threshold: float = 0.6) -> List[np.ndarray]:
        """
        Combines Top-K priors with Pix2Gestalt to produce multiple hypotheses (Best-of-N).
        Returns a list of 3 amodal masks: [M1, M2, M3].
        """
        print(f"[GeometryAgent] Generating 3 Multiple Hypotheses (Best-of-N)...")
        
        # 1. Base Hypothesis (M1): Pure Pix2Gestalt (Zero-Shot model prediction)
        raw_pred = self.predictor.predict_full_shape(image_np, visible_mask)
        m1 = raw_pred | visible_mask
        
        hypotheses = [m1]
        
        # Check retrieval confidence
        valid_priors = [p for p in top_k_priors if p.get("score", 0) >= lambda_rag_threshold]
        
        if len(valid_priors) == 0:
            print("[GeometryAgent] Retrieval confidence too low. Generating synthetic hypotheses (Dilation/Erosion).")
            # M2: Slightly dilated
            kernel = np.ones((5,5), np.uint8)
            m2 = cv2.dilate(m1.astype(np.uint8), kernel, iterations=1).astype(bool) | visible_mask
            # M3: Even more dilated or eroded
            m3 = cv2.dilate(m1.astype(np.uint8), kernel, iterations=2).astype(bool) | visible_mask
            hypotheses.extend([m2, m3])
        else:
            print(f"[GeometryAgent] High confidence priors found ({len(valid_priors)}). Fusing with RAG shapes.")
            # M2: Fuse with Top-1 Prior
            prior_1 = self._align_prior(valid_priors[0]["amodal_mask"], visible_mask)
            m2 = m1 | prior_1 | visible_mask
            hypotheses.append(m2)
            
            # M3: Fuse with Top-2 Prior (or dilate if only 1 valid prior)
            if len(valid_priors) > 1:
                prior_2 = self._align_prior(valid_priors[1]["amodal_mask"], visible_mask)
                m3 = m1 | prior_2 | visible_mask
            else:
                kernel = np.ones((5,5), np.uint8)
                m3 = cv2.dilate(m1.astype(np.uint8), kernel, iterations=1).astype(bool) | visible_mask
            hypotheses.append(m3)
            
        print(f"[GeometryAgent] Generated {len(hypotheses)} amodal hypotheses successfully.")
        return hypotheses
