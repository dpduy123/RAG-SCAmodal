# RAG-SCAmodal

A State-Of-The-Art **Retrieval-Augmented Generation (RAG)** Pipeline with Multi-Agent Systems for Amodal Segmentation and Completion on Google Colab.

This repository implements a highly sophisticated amodal completion pipeline. By leveraging a **Memory Bank** (vector database), large Vision-Language Models (VLMs), and geometric diffusion models, the pipeline accurately predicts the full shape of occluded objects and realistically inpaints their missing visual textures based on retrieved shape priors and semantic reasoning.

## 🚀 Key Features and Architecture

The system has aggressively transitioned from a heuristic iterative approach to a **Multi-Agent RAG Architecture**:

1. **Modal Segmentation (SAM 2)**: Detects the visible part of the occluded object and background elements.
2. **Object Crop & CLIP Encoder**: Crops the target object and extracts a 512-dimensional feature vector using `openai/clip-vit-base-patch32`.
3. **Memory Bank (Zilliz Cloud / Milvus)**: Searches a vector database (e.g., indexed from COCOA/LVIS datasets) using Cosine Similarity to retrieve the **Top-K most similar amodal shapes** (priors).
4. **Semantic Agent (Qwen3-VL)**: A Vision-Language Model that acts as the logic engine, analyzing the image to reason about which specific parts of the object are occluded.
5. **Geometry Agent (Pix2Gestalt)**: Combines the Top-K shape priors from the Memory Bank with the internal geometric knowledge of Pix2Gestalt to hallucinate and warp the perfect amodal shape mask.
6. **Appearance Inpainting (Stable Diffusion v2)**: Extracts the difference between the **Amodal Mask** and the **Visible Mask** and performs a realistic inpainting onto a neutral canvas, guided by the Semantic Agent's prompt.
7. **Multi-Agent Critic**: Evaluates the structural, textural, and contextual integrity of the inpainted output. If the score falls below a threshold, the prompt is tightened and generation loops again.

## 🛠️ Infrastructure & Setup

The entire workflow is heavily optimized to be run instantly on **Google Colab** (A100 or T4 GPUs).

Please see the comprehensive **[COLAB_GUIDE.md](./COLAB_GUIDE.md)** and **[database_setup_guide.md](./database_setup_guide.md)** for running instructions!

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
