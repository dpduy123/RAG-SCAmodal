import os
import sys
import json
import base64
import zlib
import argparse
import numpy as np
import torch
import cv2
import re
from tqdm import tqdm
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Thêm thư mục gốc vào path để import được memory_bank.py (Lùi 3 cấp: COCOA -> data_preparation -> Root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from memory_bank import ZillizMemoryBank

def encode_mask_to_string(mask_np: np.ndarray) -> str:
    mask_bytes = mask_np.astype(np.uint8).tobytes()
    compressed = zlib.compress(mask_bytes)
    encoded_str = base64.b64encode(compressed).decode('utf-8')
    h, w = mask_np.shape
    meta_str = f"{h},{w}|{encoded_str}"
    return meta_str

def setup_milvus_collection(collection_name="amodal_shapes", dim=1280):
    print(f"Checking collection '{collection_name}'...")
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists. Dropping it to recreate...")
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), 
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),         
        FieldSchema(name="amodal_mask_rle", dtype=DataType.VARCHAR, max_length=65535) 
    ]
    
    schema = CollectionSchema(fields, "Collection for Uncertainty-Aware RAG-SCAmodal shapes")
    
    print(f"Creating collection '{collection_name}'...")
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "AUTOINDEX", 
        "params": {}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Collection '{collection_name}' created and indexed successfully!")
    return collection

def main():
    parser = argparse.ArgumentParser(description="Index COCOA/fullCOCOA to Milvus")
    parser.add_argument("--json_path", type=str, default="dataset/fullCOCOA/annotations/COCO_amodal_train2014.json", help="Đường dẫn đến file JSON")
    parser.add_argument("--images_dir", type=str, default="dataset/fullCOCOA/images", help="Thư mục chứa ảnh")
    parser.add_argument("--start_idx", type=int, default=0, help="Vị trí bắt đầu (để chia nhỏ thành các part)")
    parser.add_argument("--end_idx", type=int, default=-1, help="Vị trí kết thúc (-1 là đến hết file)")
    parser.add_argument("--append", action="store_true", help="Nếu bật, sẽ thêm vào Collection hiện tại thay vì xoá tạo lại từ đầu")
    args = parser.parse_args()

    load_dotenv()
    zilliz_uri = os.getenv("ZILLIZ_CLUSTER_URI")
    zilliz_token = os.getenv("ZILLIZ_API_TOKEN")
    
    if not zilliz_uri or not zilliz_token:
        print("❌ LỖI: Chưa cấu hình ZILLIZ_CLUSTER_URI và ZILLIZ_API_TOKEN trong file .env")
        return

    print("🔌 Connecting to Zilliz Cloud...")
    connections.connect("default", uri=zilliz_uri, token=zilliz_token)
    
    COLLECTION_NAME = "amodal_shapes"
    VECTOR_DIM = 1280 
    
    if not args.append:
        collection = setup_milvus_collection(COLLECTION_NAME, VECTOR_DIM)
    else:
        print(f"🔄 Append mode ON, loading existing collection '{COLLECTION_NAME}'...")
        collection = Collection(name=COLLECTION_NAME)
        collection.load()
    
    print("🧠 Loading AI Models (CLIP + DINOv2)...")
    memory_bank = ZillizMemoryBank(collection_name=COLLECTION_NAME)
    
    # DANH SÁCH LOẠI TRỪ (TẬP TEST) ĐỂ TRÁNH DATA LEAKAGE
    test_exclude_list = set()
    exclude_file_path = "dataset/COCOA_test_set/img_filenames_cocoa.txt"
    if os.path.exists(exclude_file_path):
        with open(exclude_file_path, "r") as f:
            for line in f:
                name = line.strip().split('/')[-1]
                match = re.search(r'(COCO_[a-zA-Z0-9_]*?\d{12})\.jpg', name)
                if match:
                    test_exclude_list.add(match.group(1) + ".jpg")

    test_json_path = "dataset/COCOA_test_set/cocoa_annotation.json"
    if os.path.exists(test_json_path):
        with open(test_json_path, 'r') as f:
            test_data = json.load(f)
            for ann in test_data.get('annotations', []):
                url = ann.get('url', '')
                if url:
                    test_exclude_list.add(url.split('/')[-1])

    print(f"📂 Start indexing dataset... (Exclude {len(test_exclude_list)} test images)")
    
    if not os.path.exists(args.json_path):
        print(f"❌ LỖI: Không tìm thấy file {args.json_path}")
        return
        
    print(f"Loading {args.json_path}...")
    with open(args.json_path, "r") as f:
        cocoa_data = json.load(f)

    # Lấy map image_id sang file_name cho fullCOCOA
    img_id_to_filename = {}
    if "images" in cocoa_data:
        for img_info in cocoa_data["images"]:
            img_id_to_filename[img_info.get("id")] = img_info.get("file_name")
    
    annotations_list = cocoa_data.get("annotations", [])
    total_anns = len(annotations_list)
    
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else total_anns
    end_idx = min(end_idx, total_anns)

    print(f"▶️ Processing part from index {start_idx} to {end_idx} (Total: {total_anns})...")
    
    batch_embeddings = []
    batch_masks = []
    
    for i in tqdm(range(start_idx, end_idx), desc="Processing JSON"):
        img_data = annotations_list[i]
        
        # Tìm tên file ảnh
        if "image_id" in img_data and img_id_to_filename:
            img_name = img_id_to_filename.get(img_data["image_id"], "")
        else:
            img_name = img_data.get("filename", "")
            
        if img_name in test_exclude_list:
            continue
            
        img_path = os.path.join(args.images_dir, img_name)
        if not os.path.exists(img_path):
            continue 
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        H, W = img.shape[:2]
        
        # fullCOCOA có format regions kiểu array, hoặc đôi khi segmentation trực tiếp
        regions = img_data.get("regions", [])
        if not regions and "segmentation" in img_data:
            regions = [img_data] # Đưa chính nó thành 1 region nếu format phẳng
            
        for region in regions:
            if region.get("isStuff", 0) == 1:
                continue
                
            seg = region.get("segmentation")
            if not isinstance(seg, list) or len(seg) == 0:
                continue 
                
            # Đề phòng seg bị lồng (e.g. [[x1,y1...]])
            if isinstance(seg[0], list):
                seg = seg[0]
                
            if len(seg) < 6:
                continue

            ground_truth_mask = np.zeros((H, W), dtype=np.uint8)
            try:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(ground_truth_mask, [pts], 255)
            except Exception:
                continue
                
            x, y, w, h = cv2.boundingRect(pts)
            if w < 10 or h < 10:
                continue 
                
            m_x, m_y = int(w * 0.1), int(h * 0.1)
            x1, y1 = max(0, x - m_x), max(0, y - m_y)
            x2, y2 = min(W, x + w + m_x), min(H, y + h + m_y)
            crop_img_np = img[y1:y2, x1:x2]
            
            dual_vector = memory_bank.extract_dual_features(crop_img_np) 
            
            gt_mask_bool = ground_truth_mask > 127
            mask_string = encode_mask_to_string(gt_mask_bool)
            
            batch_embeddings.append(dual_vector.tolist())
            batch_masks.append(mask_string)
            
            if len(batch_embeddings) >= 500:
                collection.insert([batch_embeddings, batch_masks])
                batch_embeddings = []
                batch_masks = []

    if len(batch_embeddings) > 0:
        collection.insert([batch_embeddings, batch_masks])

    collection.flush()
    print(f"✅ Hoàn tất part {start_idx} đến {end_idx}!")
    print(f"📊 Số lượng hiện có trên DB: {collection.num_entities} vật thể.")

if __name__ == "__main__":
    main()
