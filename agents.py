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
            prior_1 = valid_priors[0]["amodal_mask"]
            # To handle shape mismatch, we just union them for this demo.
            m2 = m1 | prior_1 | visible_mask
            hypotheses.append(m2)
            
            # M3: Fuse with Top-2 Prior (or dilate if only 1 valid prior)
            if len(valid_priors) > 1:
                prior_2 = valid_priors[1]["amodal_mask"]
                m3 = m1 | prior_2 | visible_mask
            else:
                kernel = np.ones((5,5), np.uint8)
                m3 = cv2.dilate(m1.astype(np.uint8), kernel, iterations=1).astype(bool) | visible_mask
            hypotheses.append(m3)
            
        print(f"[GeometryAgent] Generated {len(hypotheses)} amodal hypotheses successfully.")
        return hypotheses
