"""
eval_laion/colab_run_laion.py

Ready-to-run Colab script for LAION evaluation.
Copy-paste into Colab cells on A100 GPU.

Usage:
    %cd /content/AP_ProtoSAM_Amodal
    %run eval_laion/colab_run_laion.py
"""

import subprocess, sys, os

def run(cmd):
    print(f"▸ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# ══════════════════════════════════════════════════════════════
# Cell 1: Setup
# ══════════════════════════════════════════════════════════════

if not os.path.exists("amodal_completer.py"):
    print("❌ Please %cd to the ProtoSAM+Amodal project root first!")
    sys.exit(1)

print("=" * 60)
print("  Step 1: Running base project setup...")
print("=" * 60)
run("python colab_setup.py")

# ══════════════════════════════════════════════════════════════
# Cell 2: Download LAION from Kaggle
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Step 2: Downloading LAION dataset from Kaggle...")
print("=" * 60)

LAION_DIR = "dataset/LAION"

try:
    run("pip install -q kaggle")
    run(f"kaggle datasets download -d ralphsitinh/laionnn -p {LAION_DIR} --unzip")
    print(f"✅ LAION dataset downloaded to {LAION_DIR}")
except Exception as e:
    print(f"⚠️  Kaggle download failed: {e}")
    print("Manual fix:")
    print(f"  1. Download from https://www.kaggle.com/datasets/ralphsitinh/laionnn")
    print(f"  2. Upload and extract to {LAION_DIR}/")

# ══════════════════════════════════════════════════════════════
# Cell 3: Diagnose paths (no GPU needed)
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Step 3: Diagnosing image paths...")
print("=" * 60)

run(f"python eval_laion/diagnose_paths.py --img-dir {LAION_DIR}")

# ══════════════════════════════════════════════════════════════
# Cell 4: Run evaluation
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Step 4: Running LAION Amodal Evaluation")
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
        limit=None,                            # None = all 177 samples
        verbose=False,
        output_file="results/laion_eval_results.csv",
        save_masks=True,
        mask_output_dir="results/laion_masks",
        save_debug=True,                       # Save visual debug panels
        debug_output_dir="results/laion_debug",
    )
finally:
    evaluator.cleanup()

print("\n✅ LAION Evaluation Complete!")
print(f"📊 Results CSV:    results/laion_eval_results.csv")
print(f"🎭 Masks:          results/laion_masks/")
print(f"🔍 Debug panels:   results/laion_debug/")
