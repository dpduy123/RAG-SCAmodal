"""
AmodalSeg — Google Colab Pro Setup Script
==========================================
Chạy trên Colab: %cd AP_ProtoSAM_Amodal && %run colab_setup.py

Colab Pro: chọn Runtime → Change runtime type → GPU → T4 hoặc A100
- T4 (16GB): đủ chạy, dùng attention slicing
- A100 (40GB): thoải mái, nhanh hơn ~3-4x
"""

# ══════════════════════════════════════════════════════════════
# Cell 1: Check GPU & Install dependencies
# ══════════════════════════════════════════════════════════════

import subprocess, sys, os

def run(cmd):
    print(f"▸ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Check GPU
print("=" * 50)
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("❌ No GPU! Go to Runtime → Change runtime type → GPU")
    sys.exit(1)
print("=" * 50)

# Install from requirements.txt
run("pip install -r requirements.txt")

# Install SAM2 (not on PyPI, install via git)
run("pip install git+https://github.com/facebookresearch/sam2.git")

# Download SAM2.1 checkpoint
SAM2_CKPT = "sam2.1_hiera_large.pt"
if not os.path.exists(SAM2_CKPT):
    run(f"wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/{SAM2_CKPT}")
    print(f"✅ Downloaded {SAM2_CKPT}")
else:
    print(f"✅ {SAM2_CKPT} already exists")

# --- Pix2Gestalt Manual Setup ---
print("\n[Pix2Gestalt] Fetching codebase and 15GB weights...")

# Clone Pix2Gestalt and its submodules
if not os.path.exists("pix2gestalt"):
    run("git clone https://github.com/cvlab-columbia/pix2gestalt.git")
    run("cd pix2gestalt && git submodule update --init --recursive")
    print("✅ Pix2Gestalt codebase cloned.")

# Download 15.5GB Checkpoint
if not os.path.exists("ckpt"):
    os.makedirs("ckpt")

if not os.path.exists("ckpt/epoch=000005.ckpt") or os.path.getsize("ckpt/epoch=000005.ckpt") < 1000000:
    # Xoá file rác bị lỗi tải (nếu có)
    if os.path.exists("ckpt/epoch=000005.ckpt"):
        os.remove("ckpt/epoch=000005.ckpt")
    # Tải trực tiếp từ server đại học Columbia thay vì HuggingFace bị dính LFS pointer lỗi
    run("wget -q --show-progress -c https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt -O ckpt/epoch=000005.ckpt")
    print("✅ Pix2Gestalt 15.5GB weights downloaded.")
else:
    print("✅ Pix2Gestalt weights already exist.")

print("\n✅ All dependencies installed and models loaded!")

# ══════════════════════════════════════════════════════════════
# Cell 2: VRAM monitoring utility
# ══════════════════════════════════════════════════════════════

def print_gpu_usage(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU {label}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Total: {total:.1f}GB")

# ══════════════════════════════════════════════════════════════
# Cell 3: Optimal settings per GPU type
# ══════════════════════════════════════════════════════════════

def get_optimal_settings():
    """Auto-detect GPU and return optimal pipeline settings."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "inpaint_size": 512, "max_iter": 2, "inference_steps": 20}

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)

    if gpu_mem >= 35:  # A100 40GB
        settings = {
            "device": "cuda",
            "inpaint_size": 768,      # Higher res inpainting
            "max_iter": 4,            # More iterations
            "inference_steps": 40,    # Better quality
            "points_per_side": 48,    # Denser SAM grid
            "guidance_scale": 8.0,
        }
        print(f"🚀 A100 mode ({gpu_mem:.0f}GB) — Full quality, 768px inpainting")

    elif gpu_mem >= 14:  # T4 16GB / V100 16GB
        settings = {
            "device": "cuda",
            "inpaint_size": 512,      # Standard SD resolution
            "max_iter": 3,            # Default iterations
            "inference_steps": 30,    # Good quality
            "points_per_side": 32,    # Default SAM grid
            "guidance_scale": 7.5,
        }
        print(f"⚡ T4/V100 mode ({gpu_mem:.0f}GB) — Balanced quality/speed")

    else:  # Low VRAM GPU
        settings = {
            "device": "cuda",
            "inpaint_size": 512,
            "max_iter": 2,
            "inference_steps": 20,    # Fewer steps = less VRAM peak
            "points_per_side": 24,    # Lighter SAM
            "guidance_scale": 7.0,
        }
        print(f"💾 Low-VRAM mode ({gpu_mem:.0f}GB) — Conservative settings")

    return settings

settings = get_optimal_settings()
print(f"Settings: {settings}")

# ══════════════════════════════════════════════════════════════
# Cell 4: Quick test — run pipeline directly without server
# ══════════════════════════════════════════════════════════════

def quick_test(image_path: str, mask_idx: int = 0):
    """
    Run the full pipeline on a single image without the FastAPI server.
    Useful for testing in Colab notebooks.
    """
    import numpy as np
    from PIL import Image

    # Add project to path
    sys.path.insert(0, ".")
    from segmenter import SAMSegmenter
    from amodal_completer import AmodalCompleter

    # Load image
    img = np.array(Image.open(image_path).convert("RGB"))
    print(f"Image shape: {img.shape}")
    print_gpu_usage("before models")

    # Segment
    print("\n[1/2] Running SAM segmentation...")
    segmenter = SAMSegmenter()
    masks = segmenter.segment_everything(img, **{
        k: v for k, v in settings.items()
        if k in ["points_per_side"]
    })
    print(f"Found {len(masks)} masks")
    print_gpu_usage("after SAM")

    # Free SAM to save VRAM for SD2
    del segmenter
    torch.cuda.empty_cache()
    print_gpu_usage("after SAM cleanup")

    # Amodal complete
    print(f"\n[2/2] Running amodal completion on mask {mask_idx}...")
    completer = AmodalCompleter()
    target_mask = masks[mask_idx]["segmentation"].astype(bool)
    outputs = completer.complete(
        image=img,
        visible_mask=target_mask,
        all_masks=masks,
        max_iter=settings.get("max_iter", 3),
    )
    rgba = outputs["inpainted_rgba"]
    print_gpu_usage("after completion")

    # Save result
    result = Image.fromarray(rgba, mode="RGBA")
    output_path = image_path.rsplit(".", 1)[0] + "_amodal.png"
    result.save(output_path)
    print(f"\n✅ Saved to {output_path}")

    # Display in Colab
    try:
        from IPython.display import display
        display(result)
    except ImportError:
        pass

    return rgba, masks


# Usage:
# rgba, masks = quick_test("your_image.jpg", mask_idx=0)

