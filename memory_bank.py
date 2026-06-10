import os
import torch
import numpy as np
from PIL import Image
from pymilvus import connections, Collection, utility
from dotenv import load_dotenv

class ZillizMemoryBank:
    """
    Memory Bank retrieves amodal shape priors (masks) from a Zilliz Cloud (Milvus) database
    based on the CLIP embedding of the cropped target object.
    """
    def __init__(self, collection_name="amodal_shapes", clip_model_id="openai/clip-vit-base-patch32", dinov2_model_id="facebook/dinov2-base", device=None):
        self.collection_name = collection_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Connect to Zilliz
        load_dotenv()
        self.zilliz_uri = os.getenv("ZILLIZ_CLUSTER_URI", "")
        self.zilliz_token = os.getenv("ZILLIZ_API_TOKEN", "")
        
        self.collection = None
        self._connect_milvus()
        
        # CLIP Models (loaded on demand to save memory if needed, but usually kept in memory)
        self.clip_model = None
        self.clip_processor = None
        self._clip_model_id = clip_model_id
        
        # DINOv2 Models
        self.dinov2_model = None
        self.dinov2_processor = None
        self._dinov2_model_id = dinov2_model_id

    def _connect_milvus(self):
        if not self.zilliz_uri or not self.zilliz_token:
            print("[ZillizMemoryBank] WARNING: ZILLIZ_CLUSTER_URI or ZILLIZ_API_TOKEN not found in .env.")
            print("[ZillizMemoryBank] Memory Bank will run in MOCK mode (returning empty/dummy shapes).")
            return
            
        try:
            print(f"[ZillizMemoryBank] Connecting to Zilliz Cloud at {self.zilliz_uri}...")
            connections.connect(
                alias="default",
                uri=self.zilliz_uri,
                token=self.zilliz_token
            )
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                print(f"[ZillizMemoryBank] Successfully loaded collection '{self.collection_name}'.")
            else:
                print(f"[ZillizMemoryBank] WARNING: Collection '{self.collection_name}' does not exist.")
        except Exception as e:
            print(f"[ZillizMemoryBank] Failed to connect or load collection: {e}")
            self.collection = None

    def _load_clip(self):
        if self.clip_model is None:
            print("[ZillizMemoryBank] Loading CLIP model for feature extraction...")
            from transformers import CLIPModel, CLIPProcessor
            clip_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.clip_model = CLIPModel.from_pretrained(
                self._clip_model_id, torch_dtype=clip_dtype
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self._clip_model_id)

    def _load_dinov2(self):
        if self.dinov2_model is None:
            print("[ZillizMemoryBank] Loading DINOv2 model for shape extraction...")
            from transformers import AutoImageProcessor, AutoModel
            dinov2_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.dinov2_processor = AutoImageProcessor.from_pretrained(self._dinov2_model_id)
            self.dinov2_model = AutoModel.from_pretrained(
                self._dinov2_model_id, torch_dtype=dinov2_dtype
            ).to(self.device)

    @torch.no_grad()
    def extract_dual_features(self, crop_image_np: np.ndarray) -> np.ndarray:
        """
        Extract and concatenate normalized CLIP and DINOv2 embeddings.
        """
        self._load_clip()
        self._load_dinov2()
        crop_pil = Image.fromarray(crop_image_np)
        
        # 1. CLIP features (Semantic)
        clip_inputs = self.clip_processor(images=crop_pil, return_tensors="pt")
        clip_pixel_values = clip_inputs["pixel_values"].to(self.device).to(self.clip_model.dtype)
        clip_embeds = self.clip_model.get_image_features(pixel_values=clip_pixel_values)
        clip_embeds = clip_embeds / clip_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # 2. DINOv2 features (Geometry/Shape)
        dino_inputs = self.dinov2_processor(images=crop_pil, return_tensors="pt")
        dino_pixel_values = dino_inputs["pixel_values"].to(self.device).to(self.dinov2_model.dtype)
        dino_outputs = self.dinov2_model(pixel_values=dino_pixel_values)
        dino_embeds = dino_outputs.pooler_output
        dino_embeds = dino_embeds / dino_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # 3. Combine Dual Features
        combined_embeds = torch.cat([clip_embeds, dino_embeds], dim=-1)
        combined_embeds = combined_embeds / combined_embeds.norm(p=2, dim=-1, keepdim=True)
        
        print("[ZillizMemoryBank] Extracted Dual Features (CLIP + DINOv2)")
        return combined_embeds.cpu().numpy()[0]

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        """
        Query the Memory Bank for the top_k most similar shapes.
        
        Returns:
            A list of dicts, each containing:
            - 'amodal_mask': np.ndarray (boolean mask of the retrieved shape)
            - 'score': float (similarity score)
            - 'id': int or str (database ID)
        """
        if self.collection is None:
            print("[ZillizMemoryBank] Running in MOCK mode. Returning dummy proposals to test Best-of-N.")
            # Fallback: create mock proposals with varying confidence scores
            h, w = 256, 256
            mock_mask1 = np.zeros((h, w), dtype=bool)
            mock_mask1[50:200, 50:200] = True
            
            mock_mask2 = np.zeros((h, w), dtype=bool)
            mock_mask2[80:180, 80:180] = True
            
            # 2 proposals: 1 low conf (0.45) - to trigger lambda_rag reduction, 1 higher conf (0.75)
            return [
                {"amodal_mask": mock_mask1, "score": 0.45, "id": "mock_1"},
                {"amodal_mask": mock_mask2, "score": 0.75, "id": "mock_2"}
            ]

        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            }
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",  # Assuming your vector field is named 'embedding'
                param=search_params,
                limit=top_k,
                output_fields=["amodal_mask_rle", "id"] # Assuming we store mask as RLE or bytes
            )
            
            proposals = []
            for hits in results:
                for hit in hits:
                    # In a real scenario, you decode the 'amodal_mask_rle' or path to the mask image
                    # Here we mock the mask decoding for now since the schema is not fully defined
                    # mask = decode_rle(hit.entity.get('amodal_mask_rle'))
                    # proposals.append({"amodal_mask": mask, "score": hit.distance, "id": hit.id})
                    pass
            
            # Temporary MOCK while schema is finalized
            print(f"[ZillizMemoryBank] Retrieved {len(proposals)} masks (MOCKED decode).")
            return proposals
            
        except Exception as e:
            print(f"[ZillizMemoryBank] Retrieval failed: {e}")
            return []
