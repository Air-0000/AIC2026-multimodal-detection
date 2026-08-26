"""
Preprocess Depth maps for DEIMv2 multimodal training.

Generates a depth dataset aligned (by filename) with the RGB/IR images so that
the multimodal dataset class can pair them trivially:

    work_dir/deimv2_depth/
        train/images/<name>.png   (single-channel, normalized to [0,1])
        test/ images/<name>.png

Depth is treated as an independent geometric modality (orthogonal to the IR
"heat" channel). We only resize + min-max normalize; we do NOT attempt any
geometric registration with the visible image (consistent with the decision
that IR/depth alignment is not reliable and should be learned end-to-end).

Usage:
    conda run -n deimv2 python prep_depth.py
"""
import os
import numpy as np
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SRC_TRAIN = os.path.join(ROOT, "dataset/train/depth")
SRC_TEST = os.path.join(ROOT, "dataset/test/depth")
OUT_ROOT = os.path.join(ROOT, "work_dir/deimv2_depth")
OUT_TRAIN = os.path.join(OUT_ROOT, "train/images")
OUT_TEST = os.path.join(OUT_ROOT, "test/images")
TARGET_SIZE = 640  # DEIMv2 train resolution


def process_one(src_path, dst_path):
    im = Image.open(src_path)
    # Depth may be RGB-like or single channel; convert to grayscale luminance.
    if im.mode != "L":
        im = im.convert("L")
    im = im.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    # min-max normalize to [0,1] so the network sees a stable scale.
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo > 1e-6:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    out = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    out.save(dst_path)


def run(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    names = sorted(os.listdir(src_dir))
    done = 0
    for n in names:
        if not n.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem, _ = os.path.splitext(n)
        process_one(os.path.join(src_dir, n), os.path.join(dst_dir, stem + ".png"))
        done += 1
    print(f"[prep_depth] {src_dir} -> {dst_dir}: {done} images")


if __name__ == "__main__":
    run(SRC_TRAIN, OUT_TRAIN)
    run(SRC_TEST, OUT_TEST)
    # validation split: filenames come from deimv2_dataset/valid/images,
    # depth sources are paired from dataset/train/depth (valid is a subset of train).
    VAL_RGB = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/images")
    OUT_VALID = os.path.join(OUT_ROOT, "valid/images")
    if os.path.isdir(VAL_RGB):
        names = sorted(os.listdir(VAL_RGB))
        done = 0
        for n in names:
            if not n.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            src = os.path.join(SRC_TRAIN, n)
            if not os.path.exists(src):
                continue
            process_one(src, os.path.join(OUT_VALID, os.path.splitext(n)[0] + ".png"))
            done += 1
        print(f"[prep_depth] valid: {done} images")
    print("[prep_depth] done.")
