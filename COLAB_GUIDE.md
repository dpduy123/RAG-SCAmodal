# 🚀 Hướng dẫn chạy RAG-SCAmodal trên Google Colab Pro

Kiến trúc RAG-SCAmodal (Retrieval-Augmented Generation for Shape Completion) kết hợp Multi-Agent và Memory Bank để dự đoán phần bị che khuất siêu chính xác.

## Yêu cầu

- Tài khoản **Google Colab Pro** (hoặc Pro+)
- GPU: **T4 (16GB)** hoặc **A100 (40GB)**
- (Mới) Tài khoản **Zilliz Cloud** (Miễn phí) để làm Memory Bank.

### Môi trường đã kiểm chứng

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| diffusers | 0.37.1 |
| pymilvus | >=2.4.0 |
| GPU test | NVIDIA A100-SXM4-40GB |

---

## Bước 1: Tạo notebook & chọn GPU

1. Vào [Google Colab](https://colab.research.google.com/) → **New Notebook**
2. Chọn **Runtime → Change runtime type**
3. Chọn:
   - **Hardware accelerator**: `GPU`
   - **GPU type**: `T4` hoặc `A100` (nếu có)
4. Nhấn **Save**

---

## Bước 2: Thiết lập Môi trường & Database

### Cell 1 — Clone project & Cấu hình Zilliz Cloud

Bạn cần tạo một file `.env` chứa API Key của Zilliz Cloud để Memory Bank hoạt động. (Nếu bạn để trống URI, hệ thống sẽ tự động chạy ở chế độ MOCK - dùng dữ liệu giả).

```python
!git clone https://github.com/dpduy123/RAG-SCAmodal.git
%cd RAG-SCAmodal

import os
# ĐIỀN THÔNG TIN ZILLIZ CLOUD CỦA BẠN VÀO ĐÂY:
# (Nếu muốn dùng Milvus Lite chạy trực tiếp trên Colab, đổi URI thành "./milvus_local.db" và bỏ trống token)
zilliz_uri = "https://<your-cluster-url>.zillizcloud.com"
zilliz_token = "<your-api-token>"

with open(".env", "w") as f:
    f.write(f"ZILLIZ_CLUSTER_URI={zilliz_uri}\n")
    f.write(f"ZILLIZ_API_TOKEN={zilliz_token}\n")
print("✅ Đã thiết lập xong môi trường Memory Bank!")
```

### Cell 2 — Cài đặt tự động (1 lệnh duy nhất)

```python
%run colab_setup.py
```

### Cell 3 (Tùy chọn) — Đẩy dữ liệu COCOA lên Zilliz Cloud
> **Lưu ý:** Bạn chỉ cần chạy bước này **ĐÚNG 1 LẦN DUY NHẤT** ở lần setup đầu tiên để tạo Memory Bank. Những lần sau mở Colab lên để test ảnh thì hãy **bỏ qua** bước này!

1. Upload thư mục `dataset/COCOA/` (chứa các file JSON đã được chia nhỏ như `part1.json`, `part2.json`) lên Colab.
2. (Quan trọng) Chạy file lọc dữ liệu để loại bỏ tập Test, tránh Data Leakage:
   ```python
   !python3 data_preparation/COCOA/remove_test_data.py
   ```
3. Chạy vòng lặp để đẩy toàn bộ các file Part lên Zilliz Cloud. File đầu tiên sẽ khởi tạo Collection, các file sau dùng cờ `--append`:
   
   ```bash
   %%bash
   first=true
   for file in dataset/COCOA/*.json; do
     if [ "$first" = true ]; then
       echo "🚀 Bắt đầu tạo mới Collection với file: $file..."
       python3 data_preparation/COCOA/index_cocoa_to_milvus.py --json_path "$file"
       first=false
     else
       echo "🔄 Đang nối tiếp (append) file: $file..."
       python3 data_preparation/COCOA/index_cocoa_to_milvus.py --json_path "$file" --append
     fi
   done
   ```
*(Cứ mỗi 500 ảnh, hệ thống sẽ đẩy lên 1 batch. Quá trình này có thể tốn khá nhiều thời gian tùy số lượng dữ liệu của bạn).*

---

Script này sẽ tự động:
- ✅ Cài dependencies (kể cả pymilvus, Qwen-VL).
- ✅ Download SAM2.1 checkpoint.
- ✅ Load Pix2Gestalt & CLIP models.

---

## Bước 3: Chạy RAG-SCAmodal Pipeline

### Chạy trực tiếp trong notebook

```python
import torch
import numpy as np
from PIL import Image
from google.colab import files

# Upload ảnh
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
img = np.array(Image.open(image_path).convert("RGB"))
print(f"Image: {image_path} | Shape: {img.shape}")
```

```python
# ── Bước 1: SAM Segmentation ──
from segmenter import SAMSegmenter

segmenter = SAMSegmenter()
masks = segmenter.segment_everything(img, points_per_side=32)

# Xem các mask
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, min(4, len(masks)), figsize=(16, 4))
if len(masks) == 1: axes = [axes]
for i, ax in enumerate(axes):
    ax.imshow(img)
    ax.imshow(masks[i]["segmentation"], alpha=0.5, cmap="jet")
    ax.set_title(f"Mask {i}")
    ax.axis("off")
plt.show()

# Giải phóng VRAM
del segmenter
torch.cuda.empty_cache()
```

```python
# ── Bước 2: RAG Amodal Completion ──
from amodal_completer import AmodalCompleter

MASK_ID = 0  # <-- Đổi số này thành ID của vật thể bạn muốn vẽ bù
target_mask = masks[MASK_ID]["segmentation"].astype(bool)

completer = AmodalCompleter()

# Pipeline sẽ tự động chạy qua: 
# Object Crop -> CLIP -> Memory Bank -> Semantic Agent -> Geometry Agent -> Diffusion -> MultiAgentCritic
outputs = completer.complete(
    image=img,
    visible_mask=target_mask,
    all_masks=masks,
    max_iter=3,
)
print("✅ Amodal completion done!")
```

```python
# ── Xem kết quả ──
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(outputs["input_image"])
axes[0].set_title("Original")

axes[1].imshow(outputs["visible_mask"], cmap="gray")
axes[1].set_title(f"Visible Mask (#{MASK_ID})")

axes[2].imshow(outputs["amodal_mask"], cmap="gray")
axes[2].set_title("Amodal Mask (Geometry Agent)")

rgba_result = outputs["inpainted_rgba"]
axes[3].imshow(rgba_result)
axes[3].set_title("Amodal Result (RGBA)")

for ax in axes: ax.axis("off")
plt.show()

# In ra suy luận của Semantic Agent
print("\n📝 [Semantic Agent Reasoning]:")
print(outputs.get("vlm_reasoning", "No reasoning provided."))
```

---

## Bảng VRAM tham khảo (Kiến trúc RAG)

Hệ thống giờ đây có nhiều tác nhân (Agents) chạy đồng thời. Mức tiêu thụ VRAM như sau:

| Bước | VRAM sử dụng | Ghi chú |
|------|:---:|---------|
| SAM2 segment | ~6 GB | Giải phóng sau khi segment xong |
| CLIP + Memory Bank | ~1 GB | Mã hóa crop để query Milvus |
| Semantic Agent (Qwen-VL) | ~8 GB | Load mô hình 4B ở FP16 |
| Geometry Agent (Pix2Gestalt)| ~9 GB | Nắn chỉnh (warp) Top-K priors |
| SD2 inpainting | ~5 GB | Vẽ bù (Inpaint Appearance) |
| **Peak (Có optimize)** | **~18 GB** | Yêu cầu load/unload model hợp lý |
| **Peak (Giữ tất cả)** | **~29 GB** | Chạy mượt mà trên A100 |

> 💡 **Tip VRAM**: Nếu bạn chạy trên T4 (16GB), hệ thống có thể sẽ gặp lỗi Out of Memory khi load chung Qwen-VL và Pix2Gestalt. Khuyến nghị chạy từng model một (ví dụ: chạy Qwen lưu kết quả ra text, xóa Qwen, sau đó mới load Pix2Gestalt). Colab A100 có thể "cân" mượt mà toàn bộ hệ thống cùng lúc.
