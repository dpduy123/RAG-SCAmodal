import os
import json
import base64
import zlib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Import MemoryBank để tận dụng hàm trích xuất Dual Features (CLIP + DINOv2) đã viết sẵn
from memory_bank import ZillizMemoryBank

def encode_mask_to_string(mask_np: np.ndarray) -> str:
    """
    Nén ma trận boolean (Mask) thành một chuỗi văn bản nhỏ gọn.
    Sử dụng zlib (nén) + base64 (chuyển thành chữ) để lưu vào Vector DB.
    """
    # Chuyển boolean mask thành mảng byte (uint8)
    mask_bytes = mask_np.astype(np.uint8).tobytes()
    # Nén mảng byte để giảm dung lượng
    compressed = zlib.compress(mask_bytes)
    # Chuyển thành chuỗi văn bản an toàn
    encoded_str = base64.b64encode(compressed).decode('utf-8')
    
    # Kèm thêm thông tin chiều cao (H), rộng (W) để sau này decode dễ dàng
    h, w = mask_np.shape
    meta_str = f"{h},{w}|{encoded_str}"
    return meta_str

def setup_milvus_collection(collection_name="amodal_shapes", dim=1280):
    """
    Khởi tạo bảng (Collection) trên Zilliz Cloud với schema phù hợp.
    """
    print(f"Checking collection '{collection_name}'...")
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists. Dropping it to recreate...")
        utility.drop_collection(collection_name)
    
    # Định nghĩa cấu trúc bảng
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), # Khóa chính tự tăng
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),         # Vector 1280 chiều
        FieldSchema(name="amodal_mask_rle", dtype=DataType.VARCHAR, max_length=65535) # Chuỗi chứa mask nén
    ]
    
    schema = CollectionSchema(fields, "Collection for Uncertainty-Aware RAG-SCAmodal shapes")
    
    print(f"Creating collection '{collection_name}'...")
    collection = Collection(name=collection_name, schema=schema)
    
    # Tạo Index cho cột embedding để tìm kiếm siêu tốc
    index_params = {
        "metric_type": "COSINE",
        "index_type": "AUTOINDEX", # Zilliz Serverless khuyên dùng AUTOINDEX
        "params": {}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Collection '{collection_name}' created and indexed successfully!")
    return collection

def main():
    load_dotenv()
    zilliz_uri = os.getenv("ZILLIZ_CLUSTER_URI")
    zilliz_token = os.getenv("ZILLIZ_API_TOKEN")
    
    if not zilliz_uri or not zilliz_token:
        print("❌ LỖI: Chưa cấu hình ZILLIZ_CLUSTER_URI và ZILLIZ_API_TOKEN trong file .env")
        return

    # 1. Kết nối Zilliz Cloud
    print("🔌 Connecting to Zilliz Cloud...")
    connections.connect("default", uri=zilliz_uri, token=zilliz_token)
    
    # 2. Tạo Collection
    COLLECTION_NAME = "amodal_shapes"
    VECTOR_DIM = 1280 # 512 (CLIP) + 768 (DINOv2)
    collection = setup_milvus_collection(COLLECTION_NAME, VECTOR_DIM)
    
    # 3. Khởi tạo MemoryBank (chỉ để dùng ké model trích xuất vector)
    print("🧠 Loading AI Models (CLIP + DINOv2)...")
    memory_bank = ZillizMemoryBank(collection_name=COLLECTION_NAME)
    
    # =====================================================================
    # MÔ PHỎNG QUÁ TRÌNH DUYỆT DATASET COCOA (BẠN CẦN THAY BẰNG CODE ĐỌC FILE THẬT)
    # =====================================================================
    print("📂 Start indexing dataset...")
    
    # DANH SÁCH LOẠI TRỪ (TẬP TEST) ĐỂ TRÁNH DATA LEAKAGE
    # Giả sử bạn có 1 file text chứa tên các ảnh dùng để test (mỗi dòng 1 tên)
    test_exclude_list = set()
    exclude_file_path = "dataset/COCOA/test_images.txt"
    if os.path.exists(exclude_file_path):
        with open(exclude_file_path, "r") as f:
            test_exclude_list = set([line.strip() for line in f.readlines()])
        print(f"🛡️ Loaded {len(test_exclude_list)} test images to EXCLUDE from Memory Bank.")
    else:
        print("⚠️ Warning: No test exclude list found. All processed images will be indexed.")

    # GIẢ LẬP CÓ 10 VẬT THỂ (Trong thực tế bạn thay bằng vòng for duyệt 50,000 ảnh)
    num_dummy_instances = 10 
    
    batch_embeddings = []
    batch_masks = []
    
    for i in tqdm(range(num_dummy_instances), desc="Processing instances"):
        
        # --- BƯỚC A: ĐỌC DỮ LIỆU TỪ Ổ CỨNG (GIẢ LẬP) ---
        img_name = f"dummy_image_{i}.jpg" # Trong thực tế đây là tên file ảnh thật
        
        # KIỂM TRA DATA LEAKAGE: Nếu ảnh này thuộc tập Test, BỎ QUA không cho vào DB!
        if img_name in test_exclude_list:
            # print(f"Skipping {img_name} (Test set)")
            continue
            
        # Trong thực tế, bạn load ảnh và bbox tại đây:
        # img = cv2.imread(f"cocoa_data/images/{img_name}")
        # x_min, y_min, x_max, y_max = bbox
        # crop_img_np = img[y_min:y_max, x_min:x_max]
        # ground_truth_mask = cv2.imread(f"cocoa_data/masks/{mask_name}.png", 0) > 127
        
        # (Tạo data ảo để code không bị lỗi khi chạy thử)
        crop_img_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        ground_truth_mask = np.random.choice([True, False], size=(224, 224))
        
        # --- BƯỚC B: CHẠY QUA MODEL ĐỂ LẤY VECTOR (1280 chiều) ---
        # Chạy Dual Encoder (Mất vài chục mili-giây cho mỗi ảnh)
        dual_vector = memory_bank.extract_dual_features(crop_img_np) 
        
        # --- BƯỚC C: NÉN MASK ---
        mask_string = encode_mask_to_string(ground_truth_mask)
        
        # Lưu vào danh sách chờ (Batch)
        batch_embeddings.append(dual_vector.tolist())
        batch_masks.append(mask_string)
        
        # Nếu danh sách gom đủ 500 vật thể (hoặc là lượt cuối cùng) thì đẩy lên Cloud 1 lần cho lẹ
        if len(batch_embeddings) == 500 or i == num_dummy_instances - 1:
            data_to_insert = [
                batch_embeddings, # Cột embedding
                batch_masks       # Cột amodal_mask_rle
            ]
            collection.insert(data_to_insert)
            
            # Xóa tạm bộ nhớ đệm
            batch_embeddings = []
            batch_masks = []

    # 4. Hoàn tất
    collection.flush()
    print(f"✅ Hoàn tất! Đã đẩy thành công dữ liệu lên Zilliz Cloud.")
    print(f"📊 Số lượng hiện có trên DB: {collection.num_entities} vật thể.")

if __name__ == "__main__":
    main()
