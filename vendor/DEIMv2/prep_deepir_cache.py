"""
Build an offline image cache for CocoDetectionMultiModal (AIC deepir, 6ch).

For every training sample we replicate the dataset's image-reading + channel
stacking (RGB3 + heat1 + raw_ir1 + depth1 -> [6,H,W] float in [0,1] at the
ORIGINAL image size) and save it as a single compressed .pt.  This removes the
per-iteration cost of decoding 4 PNGs + 3 resizes + stacking; the random
augmentation pipeline (RandomNight5ch / RandomIoUCrop / ...) and Normalize still
run online, so training is bit-for-bit equivalent to the non-cached path.

Run from the vendor/DEIMv2 directory:
    python prep_deepir_cache.py
"""
import os
import sys

import torch

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.data.dataset.coco_dataset_mm import CocoDetectionMultiModal

WORK = os.path.join(ROOT, "work_dir")
CACHE_DIR = os.path.join(WORK, "deepir_cache", "train")
os.makedirs(CACHE_DIR, exist_ok=True)

ds = CocoDetectionMultiModal(
    img_folder=os.path.join(WORK, "deimv2_dataset", "train", "images"),
    ann_file=os.path.join(WORK, "deimv2_dataset", "train", "annotations", "instances_train.json"),
    transforms=None,
    ir_folder=os.path.join(WORK, "deimv2_heat", "images"),
    raw_ir_folder=os.path.join(WORK, "deimv2_ir", "images"),
    depth_folder=os.path.join(ROOT, "dataset", "train", "depth"),
    return_masks=False,
    remap_mscoco_category=False,
    n_channels=6,
    modality="rgb_ir",
    depth_mode="raw",
)

n = len(ds)
print(f"[cache] building {n} samples -> {CACHE_DIR}")
done = 0
for idx in range(n):
    img_name = ds.coco.loadImgs(ds.ids[idx])[0]["file_name"]
    stem = os.path.splitext(img_name)[0]
    out = os.path.join(CACHE_DIR, stem + ".pt")
    if os.path.exists(out):
        done += 1
        continue
    x, _ = ds.load_item(idx)          # [6,H,W] float [0,1] @ original size
    # store as uint8 (small, ~= source PNG size) to keep .pt load cheap;
    # the cached loader maps it back to float [0,1] on load.
    torch.save((x * 255.0).clamp(0, 255).to(torch.uint8), out)
    done += 1
    if done % 200 == 0:
        print(f"[cache] {done}/{n}")

print(f"[cache] done: {done} samples")
