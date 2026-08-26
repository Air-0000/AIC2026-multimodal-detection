"""
Train a pure-RGB RF-DETR baseline for the multi-modal detection challenge.

Usage:
    python3 train_rgb.py [EPOCHS] [OUTPUT_SUFFIX] [BATCH]
Defaults: EPOCHS=80, OUTPUT_SUFFIX=rgb_baseline, BATCH=auto
Example (smoke test):
    python3 train_rgb.py 1 smoke 2
"""
import sys
import os
from rfdetr import RFDETRBase

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
os.chdir(ROOT)

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 80
OUT = sys.argv[2] if len(sys.argv) > 2 else "rgb_baseline"
BATCH = sys.argv[3] if len(sys.argv) > 3 else "auto"

PRETRAIN = "vendor/rfdetr_models/rf-detr-base-o365.pth"
assert os.path.exists(PRETRAIN), f"pretrain weights missing: {PRETRAIN}"

print(f"[train] epochs={EPOCHS} out={OUT} batch={BATCH} pretrain={PRETRAIN}")

model = RFDETRBase(
    pretrain_weights=PRETRAIN,
    resolution=560,
)

model.train(
    dataset_dir="work_dir/rfdetr_dataset",
    output_dir=f"work_dir/{OUT}",
    batch_size=BATCH,
    epochs=EPOCHS,
    grad_accum_steps=4,
    num_workers=4,
    tensorboard=False,
    project="rgb_baseline",
    run=OUT,
)

print(f"[train] finished -> work_dir/{OUT}")
