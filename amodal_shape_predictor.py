import sys
import os
import torch
import numpy as np
from PIL import Image
import cv2

# Add BOTH Pix2Gestalt root and inner repo to python path
sys.path.insert(0, os.path.join(os.getcwd(), 'pix2gestalt'))
sys.path.insert(0, os.path.join(os.getcwd(), 'pix2gestalt', 'pix2gestalt'))

# --- Monkey-patch for Python 3.12 compatibility ---
import pkgutil
if not hasattr(pkgutil, 'ImpImporter'):
    class ImpImporter:
        pass
    pkgutil.ImpImporter = ImpImporter

import importlib.machinery
if not hasattr(importlib.machinery.FileFinder, "find_module"):
    def find_module(self, fullname, path=None):
        spec = self.find_spec(fullname)
        return spec.loader if spec is not None else None
    importlib.machinery.FileFinder.find_module = find_module
# --------------------------------------------------

try:
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.util import create_carvekit_interface # Advanced BG Matting
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"[AmodalShapePredictor] Core libraries missing: {e}")
    IMPORT_SUCCESS = False

class Pix2GestaltPredictor:
    def __init__(self, ckpt_path: str = "ckpt/epoch=000005.ckpt", device: str = "cuda"):
        """
        Loads the Pix2Gestalt amodal shape predictor from a raw PyTorch Lightning checkpoint.
        Requires ~24GB VRAM.
        """
        self.device = device
        self.model = None
        print(f"[AmodalShapePredictor] Loading Pix2Gestalt Checkpoint from {ckpt_path}...")
        
        if not os.path.exists(ckpt_path):
            print(f"[AmodalShapePredictor] Warning: {ckpt_path} not found. Did you run the wget command?")
            print("[AmodalShapePredictor] Falling back to Heuristic Predictor.")
            return
            
        if not IMPORT_SUCCESS:
            print("[AmodalShapePredictor] Warning: LDM or OmegaConf not installed.")
            print("[AmodalShapePredictor] Falling back to Heuristic Predictor.")
            return

        try:
            config = OmegaConf.load("pix2gestalt/pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml")
            model = instantiate_from_config(config.model)
            
            # Load raw 15.5GB weights
            # Disable weights_only because Lightning 1.x checkpoints contain pickled ModelCheckpoint objects
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                
            model.load_state_dict(state_dict, strict=False)
            self.model = model.to(self.device).eval()
            self.carvekit_interface = create_carvekit_interface()
            print("[AmodalShapePredictor] Pix2Gestalt loaded successfully.")
        except Exception as e:
            print(f"[AmodalShapePredictor] Warning: Failed to initialize network: {e}")
            print("[AmodalShapePredictor] Falling back to Heuristic Predictor.")
            self.model = None

    def cleanup(self):
        """
        Explicitly free Pix2Gestalt LDM from GPU memory.
        """
        print("[AmodalShapePredictor] Freeing Pix2Gestalt (15GB+)...")
        if self.model is not None:
            del self.model
            self.model = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_full_shape(self, image: np.ndarray, visible_mask: np.ndarray, prior_mask: np.ndarray = None, text_prompt: str = "") -> np.ndarray:
        """
        Args:
            image: HxWx3 uint8 RGB image
            visible_mask: HxW bool array of the visible part
            prior_mask: (Optional) HxW bool array of RAG geometric hint
            text_prompt: (Optional) Semantic reasoning text
        Returns:
            amodal_mask: HxW bool array of the predicted whole object shape
        """
        if self.model is None:
            return self._heuristic_fallback(visible_mask)

        # 1. Run full Pix2Gestalt Inference with Prior-Conditioning
        print("[AmodalShapePredictor] Synthesizing amodal object with Prior-Conditioned LDM...")
        if text_prompt:
            print(f"[AmodalShapePredictor] Applied Semantic Guidance: '{text_prompt[:50]}...'")
            
        # Early Fusion: Inject geometric prior into the diffusion's condition mask
        if prior_mask is not None:
            print("[AmodalShapePredictor] Injecting Memory Bank prior as geometric control hint.")
            condition_mask = visible_mask | prior_mask
            # Apply light closing to ensure solid hint
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            condition_mask = cv2.morphologyEx(condition_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        else:
            condition_mask = visible_mask
            print("[AmodalShapePredictor] No prior provided. Running Zero-Shot Geometry inference.")

        try:
            sys.path.insert(0, os.path.join(os.getcwd(), 'pix2gestalt', 'pix2gestalt'))
            from inference import run_pix2gestalt
            
            H, W = image.shape[:2]
            S = max(H, W)
            
            # Tính toán lượng padding để giữ ảnh ở trung tâm (tránh bóp méo hình dạng gấu/vật thể)
            pad_top = (S - H) // 2
            pad_bottom = S - H - pad_top
            pad_left = (S - W) // 2
            pad_right = S - W - pad_left
            
            # Tạo viền xám [127,127,127] cho ảnh gốc, viền đen cho mask
            square_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[127, 127, 127])
            square_mask = cv2.copyMakeBorder((condition_mask * 255).astype(np.uint8), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            
            # Ép về 256x256 (Pix2Gestalt model takes 256x256 uint8 numpy arrays)
            resized_image = cv2.resize(square_image, (256, 256), interpolation=cv2.INTER_AREA)
            resized_mask = cv2.resize(square_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            rgb_visible_mask = np.zeros((256, 256, 3), dtype=np.uint8)
            rgb_visible_mask[:,:,0] = resized_mask
            rgb_visible_mask[:,:,1] = resized_mask
            rgb_visible_mask[:,:,2] = resized_mask
            
            # The library returns a list of synthesized np arrays
            with torch.inference_mode():
                result_np = run_pix2gestalt(
                    model=self.model, 
                    device=self.device,
                    input_im=resized_image, 
                    visible_mask=rgb_visible_mask,
                    n_samples=1
                )[0]
                
            # 2. Extract Shape from Synthesized Output using Advanced Matting (CarveKit)
            # result_np is an RGB image [0, 255]. Carvekit expects PIL.
            result_pil = Image.fromarray(result_np)
            
            amodal_rgba = np.array(self.carvekit_interface([result_pil])[0])
            alpha_channel = amodal_rgba[:, :, 3]
            
            # 3. Post-processing the alpha mask to ensure clean borders
            amodal_mask = (alpha_channel > 0).astype(np.uint8)
            
            kernel = np.ones((5, 5), np.uint8)
            amodal_mask = cv2.morphologyEx(amodal_mask, cv2.MORPH_CLOSE, kernel)
            amodal_mask = cv2.morphologyEx(amodal_mask, cv2.MORPH_OPEN, kernel)
            
            # Phóng to amodal mask trở lại khung hình vuông S x S
            amodal_mask = cv2.resize((amodal_mask * 255).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
            
            # Cắt xén (Crop) lề Padding để trả về đúng khung H x W ban đầu
            if pad_bottom == 0:
                amodal_mask_cropped = amodal_mask[pad_top:, :]
            else:
                amodal_mask_cropped = amodal_mask[pad_top:-pad_bottom, :]
                
            if pad_right != 0:
                amodal_mask_cropped = amodal_mask_cropped[:, pad_left:-pad_right]
            else:
                amodal_mask_cropped = amodal_mask_cropped[:, pad_left:]
                
            amodal_mask_bool = amodal_mask_cropped > 127
            
            # Kết hợp với visible_mask gốc
            final_amodal_mask = amodal_mask_bool | visible_mask.astype(bool)
            return final_amodal_mask
            
        except Exception as e:
            print(f"[AmodalShapePredictor] Error during inference: {e}. Falling back to Heuristic.")
            return self._heuristic_fallback(visible_mask)

    def _heuristic_fallback(self, visible_mask: np.ndarray) -> np.ndarray:
        print("[AmodalShapePredictor] Expanding visible mask using heuristics...")
        amodal_mask = (visible_mask.copy() * 255).astype(np.uint8)
        
        kernel = np.ones((40, 40), np.uint8)
        amodal_mask = cv2.dilate(amodal_mask, kernel, iterations=1)
        
        ys, xs = np.where(visible_mask)
        if len(ys) > 0:
            height = ys.max() - ys.min()
            bottom_y = ys.max()
            extend_to = min(amodal_mask.shape[0], int(bottom_y + height * 0.4))
            x_min, x_max = xs.min(), xs.max()
            amodal_mask[bottom_y:extend_to, x_min:x_max] = 255
            
        return amodal_mask > 127
