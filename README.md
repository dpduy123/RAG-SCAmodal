# RAG-SCAmodal

A State-Of-The-Art **Retrieval-Augmented Generation (RAG)** Pipeline with Multi-Agent Systems for Amodal Segmentation and Completion on Google Colab.

This repository implements a highly sophisticated amodal completion pipeline. By leveraging a **Memory Bank** (vector database), large Vision-Language Models (VLMs), and geometric diffusion models, the pipeline accurately predicts the full shape of occluded objects and realistically inpaints their missing visual textures based on retrieved shape priors and semantic reasoning.

## 🚀 Key Features and Architecture

The system has aggressively transitioned from a heuristic iterative approach to a **Multi-Agent RAG Architecture**:

1. **Modal Segmentation (SAM 2)**: Detects the visible part of the occluded object and background elements.
2. **Dual Encoder (CLIP + DINOv2)**: Crops the target object and extracts a concatenated **1280-dimensional feature vector** capturing both semantic (CLIP) and geometric (DINOv2) traits.
3. **Memory Bank (Zilliz Cloud / Milvus)**: Searches a vector database using Cosine Similarity to retrieve the **Top-K most similar amodal shapes** (priors), returning a Confidence Score ($\lambda_{rag}$).
4. **Semantic Agent (Qwen3-VL)**: A Vision-Language Model that acts as the logic engine, analyzing the image to reason about which specific parts of the object are occluded.
5. **Geometry Agent (Pix2Gestalt)**: Fuses Top-K priors with its geometric knowledge to generate **Multiple Hypotheses (Best-of-N)** (e.g., $M_1, M_2, M_3$). If confidence is low, it safely falls back to Zero-Shot synthesis to avoid Data Leakage.
6. **Appearance Inpainting (Stable Diffusion v2)**: Realistically inpaints the missing visual textures onto all generated hypotheses, guided by the Semantic Agent's prompt.
7. **Best-of-N Multi-Agent Critic**: Evaluates the structural, textural, and contextual integrity of all candidate images and selects the single best outcome.

## 🛠️ Infrastructure & Setup

The entire workflow is heavily optimized to be run instantly on **Google Colab** (A100 or T4 GPUs).

Please see the comprehensive **[COLAB_GUIDE.md](./COLAB_GUIDE.md)** and **[database_setup_guide.md](./database_setup_guide.md)** for running instructions!

### 🗄️ Database Setup Script
We provide a standalone indexing script `index_cocoa_to_milvus.py` to seamlessly encode (CLIP+DINOv2) and compress (RLE) your dataset into Zilliz Cloud. It also supports a `test_images.txt` exclusion list to guarantee **Zero Data Leakage** during evaluation.

### Auto-Installation (`colab_setup.py`)
Our `colab_setup.py` automatically handles the heavily complex legacy installations, including:
- Fetching specific deep-learning legacy versions: `pytorch-lightning`, `einops`, `pymilvus`, OpenAI `CLIP`.
- Auto-bypassing PyTorch 2.6's `weights_only=True` unpickling blockers.
- Downloading all necessary multi-GB checkpoints directly.

## 📦 Pipeline Execution

```python
from segmenter import SAMSegmenter
from amodal_completer import AmodalCompleter

# 1. Init
segmenter = SAMSegmenter()
completer = AmodalCompleter() # Internally boots Memory Bank, VLM, Pix2Gestalt, and SD2

# 2. Get Visible Mask
masks = segmenter.segment_everything(image_rgb, points_per_side=32)
target_mask = masks[0]["segmentation"].astype(bool)

# 3. Full Multi-Agent RAG Pipeline
# Runs: Crop -> CLIP -> Milvus -> SemanticAgent -> GeometryAgent -> SD2 -> Critic
outputs = completer.complete(
    image=image_rgb,
    visible_mask=target_mask,
    all_masks=masks,
    max_iter=3
)

# returns dictionary: 
# outputs['input_image'], outputs['visible_mask'], outputs['amodal_mask'], outputs['inpainted_rgba'], outputs['vlm_reasoning']
```

## 💻 GPU VRAM Requirements 

* **SAM2**: ~6 GB
* **CLIP + Memory Bank**: ~1 GB
* **Semantic Agent (Qwen3-VL)**: ~8 GB
* **Geometry Agent (Pix2Gestalt)**: ~9 GB 
* **Stable Diffusion 2 (Inpaint)**: ~5 GB
* **Peak Usage**: **~18 GB** (Optimized with Sequential Model offloading on T4!) up to **~29GB** (Concurrent loaded on A100).

*(The pipeline requires a `.env` file containing `ZILLIZ_CLUSTER_URI` and `ZILLIZ_API_TOKEN` to connect to the Memory Bank. If omitted, it will safely run in Mock Mode).*
