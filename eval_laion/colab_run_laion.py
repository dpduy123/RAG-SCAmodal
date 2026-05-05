"""
eval_laion/colab_run_laion.py

Ready-to-run Colab cells for LAION evaluation.
Copy-paste each section into a Colab cell on A100 GPU.

Usage on Colab:
    %cd /content/AP_ProtoSAM_Amodal
    %run eval_laion/colab_run_laion.py
"""

# ══════════════════════════════════════════════════════════════
# Cell 1: Setup — Install dependencies & download models
# ══════════════════════════════════════════════════════════════

import subprocess, sys, os

def run(cmd):
    print(f"▸ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Ensure we're in project root
if not os.path.exists("amodal_completer.py"):
    print("❌ Please %cd to the ProtoSAM+Amodal project root first!")
    sys.exit(1)

# Run base setup (installs SAM2, Pix2Gestalt, SD2, etc.)
print("=" * 60)
print("  Step 1: Running base project setup...")
print("=" * 60)
run("python colab_setup.py")

# ══════════════════════════════════════════════════════════════
# Cell 2: Download LAION dataset from Kaggle
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Step 2: Downloading LAION dataset from Kaggle...")
print("=" * 60)

LAION_IMG_DIR = "dataset/LAION"

# Check if images already exist
sample_check = os.path.join(LAION_IMG_DIR, "laion", "00000")
if os.path.exists(sample_check) and len(os.listdir(sample_check)) > 10:
    print(f"✅ LAION images already present in {sample_check}")
else:
    print("Downloading from Kaggle...")
    print("⚠️  Make sure your Kaggle credentials are set up:")
    print("    1. Go to kaggle.com → Account → Create New API Token")
    print("    2. Upload kaggle.json to Colab, or run:")
    print('       os.environ["KAGGLE_USERNAME"] = "your_username"')
    print('       os.environ["KAGGLE_KEY"] = "your_key"')
    print()

    try:
        run("pip install -q kaggle")
        run(f"kaggle datasets download -d ralphsitinh/laionnn -p {LAION_IMG_DIR} --unzip")
        print(f"✅ LAION dataset downloaded to {LAION_IMG_DIR}")
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        print("\nManual alternative:")
        print(f"  1. Download from https://www.kaggle.com/datasets/ralphsitinh/laionnn")
        print(f"  2. Extract to {LAION_IMG_DIR}/")
        print(f"  3. Ensure images are at {LAION_IMG_DIR}/laion/00000/*.jpg")

# Verify dataset structure
ann_file = os.path.join(LAION_IMG_DIR, "laion_00000_annotation.json")
if os.path.exists(ann_file):
    import json
    with open(ann_file) as f:
        data = json.load(f)
    n_anns = len(data.get("annotations", []))
    print(f"✅ Annotation file found: {n_anns} entries")
else:
    print(f"❌ Annotation file not found at {ann_file}")

# ══════════════════════════════════════════════════════════════
# Cell 3: Run LAION Evaluation
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Step 3: Running LAION Amodal Evaluation")
print("=" * 60)

import torch
print(f"GPU: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")

from eval_laion.evaluate_laion import LAIONEvaluator

evaluator = LAIONEvaluator(device="cuda")

try:
    results = evaluator.evaluate(
        ann_file="dataset/LAION/laion_00000_annotation.json",
        img_dir="dataset/LAION",
        limit=None,          # None = đánh giá toàn bộ 177 samples
        verbose=False,       # True để debug từng ảnh
        output_file="results/laion_eval_results.csv",
        save_masks=True,     # Lưu mask tự động tạo để kiểm tra
        mask_output_dir="results/laion_masks",
    )
finally:
    evaluator.cleanup()

print("\n✅ LAION Evaluation Complete!")
print(f"📊 Results: results/laion_eval_results.csv")
print(f"🎭 Masks:   results/laion_masks/")
