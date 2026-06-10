import torch
from PIL import Image
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re

class VLMReasoner:
    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", device="cuda"):
        """
        Initializes the official Qwen3-VL model using the newest architecture class.
        """
        self.device = device
        print(f"[VLMReasoner] Loading Official {model_id} (FP16)...")
        
        try:
            # Using the specific Qwen3VL class as per official documentation
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            print(f"[VLMReasoner] {model_id} loaded successfully.")
        except Exception as e:
            print(f"[VLMReasoner] Failed to load model: {e}")
            print("Tip: Ensure you ran !pip install git+https://github.com/huggingface/transformers")
            self.model = None

    @torch.no_grad()
    def reason_occlusion(self, image_np, visible_mask=None):
        if self.model is None:
            return "ERROR: Model not loaded."

        import cv2
        if visible_mask is not None:
            # Highlight the target object with a red contour
            highlighted_img = image_np.copy()
            contours, _ = cv2.findContours(visible_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw contour in Red (RGB format: 255, 0, 0)
            cv2.drawContours(highlighted_img, contours, -1, (255, 0, 0), 3)
            image_pil = Image.fromarray(highlighted_img)
            prompt_text = "The target object is highlighted with a red outline in the image. Describe ONLY the missing or occluded parts of THIS specific target object to generate an inpainting prompt."
        else:
            image_pil = Image.fromarray(image_np)
            prompt_text = "Describe the missing parts of the target object for inpainting prompt."
        
        # Standard Qwen3-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": prompt_text}
                ],
            }
        ]

        # Official preparation method
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generating output
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        
        # Slicing out the prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    @torch.no_grad()
    def critique(self, completed_image_np, original_image_np=None):
        """
        Semantic Critic (Stage 3): evaluates the completed amodal object on 3 criteria.

        Args:
            completed_image_np: H×W×3 RGB of the inpainted object (preferably composed
                                on a clean canvas so the critic focuses on the object).
            original_image_np:  optional H×W×3 RGB of the source scene for context.

        Returns:
            dict with keys:
              - score (float, 0-10): mean of three criteria
              - structural, texture, context (float, 0-10)
              - feedback (str): one-sentence description of main issue
              - raw (str): full VLM output
        """
        if self.model is None:
            return {"score": 0.0, "structural": 0.0, "texture": 0.0,
                    "context": 0.0, "feedback": "model not loaded", "raw": ""}

        completed_pil = Image.fromarray(completed_image_np)
        content = [{"type": "image", "image": completed_pil}]
        if original_image_np is not None:
            content.append({"type": "image", "image": Image.fromarray(original_image_np)})

        instruction = (
            "You are a strict quality critic for amodal object completion. "
            "The first image is the reconstructed object (visible + hallucinated parts). "
            + ("The second image is the original scene for context. " if original_image_np is not None else "")
            + "Rate each criterion on a 0-10 integer scale:\n"
            "1. STRUCTURAL: shape symmetry and part-level connectivity (e.g. correct number of legs).\n"
            "2. TEXTURE: color, lighting and material continuity between visible and reconstructed regions.\n"
            "3. CONTEXT: semantic plausibility — does the completion match the object's identity?\n\n"
            "Respond EXACTLY in this format and nothing else:\n"
            "STRUCTURAL: <int>\n"
            "TEXTURE: <int>\n"
            "CONTEXT: <int>\n"
            "FEEDBACK: <one short sentence on the main flaw, or 'good' if no flaw>"
        )
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=120)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        def _parse(label, default=5.0):
            m = re.search(rf"{label}\s*:\s*([\d.]+)", output_text, re.IGNORECASE)
            try:
                return max(0.0, min(10.0, float(m.group(1)))) if m else default
            except Exception:
                return default

        structural = _parse("STRUCTURAL")
        texture = _parse("TEXTURE")
        context = _parse("CONTEXT")
        fb = re.search(r"FEEDBACK\s*:\s*(.+?)(?:\n|$)", output_text, re.IGNORECASE | re.DOTALL)
        feedback = fb.group(1).strip() if fb else ""

        return {
            "score": (structural + texture + context) / 3.0,
            "structural": structural,
            "texture": texture,
            "context": context,
            "feedback": feedback,
            "raw": output_text,
        }

    @torch.no_grad()
    def get_missing_region_boxes(self, image_np):
        """
        Uses Qwen-VL's grounding capability to find bounding boxes of occluded/missing parts.
        Returns: List of [ymin, xmin, ymax, xmax] in normalized coordinates (0-1000).
        """
        if self.model is None: return []

        image_pil = Image.fromarray(image_np)
        prompt = "Identify the locations of the hidden or missing parts of the main object in this image. Output the bounding boxes in the format [ymin, xmin, ymax, xmax]."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=200)
        output_text = self.processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        # Extract coordinates using regex: e.g., [123, 456, 789, 101]
        boxes = []
        pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches = re.findall(pattern, output_text)
        
        for m in matches:
            boxes.append([int(x) for x in m])
        
        return boxes
