# 🧠 Hướng dẫn toàn tập Zilliz Cloud (Milvus) cho RAG-SCAmodal

Tài liệu này tổng hợp toàn bộ quy trình thiết lập, quản lý và xử lý sự cố cho hệ thống **Memory Bank** (sử dụng Zilliz Cloud / Milvus) trong dự án RAG-SCAmodal.

---

## 1. Giới thiệu: Vai trò của Milvus
Trong dự án này, Milvus không lưu trữ ảnh thật. Nó lưu trữ một Collection tên là `amodal_shapes` với 3 trường dữ liệu chính:
* `id`: Mã định danh vật thể.
* `embedding` (1280 chiều): Vector nhận thức kết hợp từ **CLIP (512D - Ngữ nghĩa)** và **DINOv2 (768D - Hình học/Tư thế)**.
* `amodal_mask_rle`: Chuỗi văn bản nén (Base64 + Zlib RLE) chứa hình bóng (Amodal Mask) hoàn chỉnh của vật thể.

---

## 2. Thiết lập Môi trường (.env)
Để code có thể giao tiếp với Zilliz Cloud, bạn cần tạo một file tên là `.env` ở thư mục gốc của dự án với nội dung:

```env
# Zilliz Cloud / Milvus Configuration
ZILLIZ_CLUSTER_URI=https://in03-xxxxxxxx.api.gcp-us-west1.zillizcloud.com
ZILLIZ_API_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
*(File `.env` đã được cấu hình ẩn đi trong `.gitignore` để tránh bị lộ API Key khi đẩy lên GitHub).*

---

## 3. Quy trình Đẩy Dữ liệu (Indexing)
Để nạp dữ liệu từ dataset COCOA lên Milvus mà không bị rò rỉ dữ liệu (Data Leakage), hãy làm theo thứ tự sau:

### Bước 3.1: Chuẩn bị Ảnh
Hệ thống bắt buộc phải có ảnh gốc để mô hình DINOv2 trích xuất vector.
1. Trích xuất tên ảnh cần tải: `python3 data_preparation/COCOA/extract_image_names.py`
2. Tải ảnh về máy: `python3 scripts/dataset_downloader.py` (Ảnh sẽ lưu ở `dataset/COCOA/images`).

### Bước 3.2: Lọc bỏ Data Leakage
Chạy script để tự động xoá toàn bộ các hình ảnh thuộc tập Test ra khỏi các file Part JSON:
```bash
python3 data_preparation/COCOA/remove_test_data.py
```

### Bước 3.3: Đẩy Vector lên Zilliz
Mở Terminal (hoặc Google Colab) và chạy vòng lặp bash sau. 
* Lần chạy đầu tiên sẽ tạo mới Collection.
* Các lần chạy sau sẽ dùng cờ `--append` để nối thêm dữ liệu.

```bash
first=true
for file in dataset/COCOA/*.json; do
  if [ "$first" = true ]; then
    echo "🚀 Bắt đầu tạo mới Collection với file: $file..."
    python3 data_preparation/COCOA/index_cocoa_to_milvus.py --json_path "$file" --images_dir "dataset/COCOA/images"
    first=false
  else
    echo "🔄 Đang nối tiếp (append) file: $file..."
    python3 data_preparation/COCOA/index_cocoa_to_milvus.py --json_path "$file" --images_dir "dataset/COCOA/images" --append
  fi
done
```

---

## 4. Theo dõi & Xử lý sự cố (Troubleshooting)

### Câu hỏi: Tại sao "Loaded Entities" tăng mà "Data Size" vẫn là 0?
* **Loaded Entities** đếm số lượng Vector đang lơ lửng trên RAM (Growing Segments). Khi script đang chạy, dữ liệu được đẩy vào RAM nên con số này tăng liên tục.
* **Data Size** chỉ đếm dữ liệu đã được đóng gói và ghi cứng xuống đĩa (Sealed Segments).
* Lệnh `collection.flush()` có nhiệm vụ ép RAM ghi xuống đĩa, nhưng nó chỉ được gọi khi script chạy xong **100%** một file JSON. Do đó, hãy kiên nhẫn chờ file chạy xong, Data Size sẽ tự động tăng vọt.

### Script kiểm tra số lượng Vector thực tế trên DB
Nếu muốn kiểm tra nhanh DB có bao nhiêu Vector, chạy đoạn Python sau trên Colab:

```python
import os
from dotenv import load_dotenv
from pymilvus import connections, utility, Collection

load_dotenv(".env")
connections.connect("default", uri=os.getenv("ZILLIZ_CLUSTER_URI"), token=os.getenv("ZILLIZ_API_TOKEN"))

for col_name in utility.list_collections():
    collection = Collection(col_name)
    collection.flush() # Ép đồng bộ RAM xuống Disk
    print(f"Collection '{col_name}' có: {collection.num_entities:,} vật thể")
```

---

## 5. Bước Tiếp Theo: RAG Inference
Sau khi DB đã có đủ dữ liệu (ví dụ: >20,000 entities), Memory Bank đã sẵn sàng hoạt động.
Bạn có thể mở `COLAB_GUIDE.md`, làm theo **Bước 3: Chạy RAG-SCAmodal Pipeline** để tải một bức ảnh lên, chỉ định vật thể bị che khuất và xem hệ thống kết hợp Retrieval + Diffusion + LLM Critic để vẽ bù lại hình dáng một cách ma thuật.
