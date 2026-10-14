"""
Build an offline image cache for the 3-channel-IR variant (AIC deepir, 7ch).

Layout: RGB(3) + heat(1) + raw_ir(1) + deleak_ir(1) + depth(1) = [7,H,W].

Identical to prep_deepir_cache.py except for the extra `deleak_ir_folder`
channel, and it writes to a SEPARATE cache dir so the existing 6ch cache
(used by the 2ch-IR baselines) is never touched.

Run from the vendor/DEIMv2 directory:
    python prep_deepir_cache3.py
"""
import os
import sys

import torch

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.data.dataset.coco_dataset_mm import CocoDetectionMultiModal

WORK = os.path.join(ROOT, "work_dir")
CACHE_DIR = os.path.join(WORK, "deepir_cache3", "train")
os.makedirs(CACHE_DIR, exist_ok=True)

ds = CocoDetectionMultiModal(
    img_folder=os.path.join(WORK, "deimv2_dataset", "train", "images"),
    ann_file=os.path.join(WORK, "deimv2_dataset", "train", "annotations", "instances_train.json"),
    transforms=None,
    ir_folder=os.path.join(WORK, "deimv2_heat", "images"),
    raw_ir_folder=os.path.join(WORK, "deimv2_ir", "images"),
    deleak_ir_folder=os.path.join(ROOT, "dataset", "train", "infrared_deleak"),
    depth_folder=os.path.join(ROOT, "dataset", "train", "depth"),
    return_masks=False,
    remap_mscoco_category=False,
    n_channels=7,
    modality="rgb_ir",
    depth_mode="raw",
)

n = len(ds)
print(f"[cache3] building {n} samples -> {CACHE_DIR}")
done = 0
for idx in range(n):
    img_name = ds.coco.loadImgs(ds.ids[idx])[0]["file_name"]
    stem = os.path.splitext(img_name)[0]
    out = os.path.join(CACHE_DIR, stem + ".pt")
    if os.path.exists(out):
        done += 1
        continue
    x, _ = ds.load_item(idx)          # [7,H,W] float [0,1] @ original size
    assert x.shape[0] == 7, f"expected 7 channels, got {x.shape[0]} for {stem}"
    torch.save((x * 255.0).clamp(0, 255).to(torch.uint8), out)
    done += 1
    if done % 200 == 0:
        print(f"[cache3] {done}/{n}")

print(f"[cache3] done: {done} samples")
