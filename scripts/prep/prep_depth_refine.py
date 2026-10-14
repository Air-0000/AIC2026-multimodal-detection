"""Depth refinement: small-hole filling + edge refinement.

Depth comes from a depth camera and is already registered to RGB (no geometric
alignment needed). We only clean it:
  1) median filter  -> remove salt-and-pepper / tiny holes
  2) grey closing    -> fill small structural gaps
  3) light gaussian  -> edge-preserving smoothing
Output is normalized to 8-bit PNG so the loader's /255.0 is correct for both
formerly-8bit and formerly-16bit depth inputs.
"""
import os
import glob
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SRC = os.path.join(ROOT, "dataset/train/depth")
OUT = os.path.join(ROOT, "work_dir/deimv2_depth/images")
os.makedirs(OUT, exist_ok=True)


def load_depth(p):
    arr = np.array(Image.open(p))
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr


def refine(d):
    d = ndi.median_filter(d, size=3)
    struct = ndi.generate_binary_structure(2, 1)
    d = ndi.grey_closing(d, size=3, structure=struct)
    d = ndi.gaussian_filter(d, sigma=0.6)
    return d


files = sorted(glob.glob(os.path.join(SRC, "*")))
print(f"[depth] {len(files)} files")
for i, f in enumerate(files):
    stem = os.path.splitext(os.path.basename(f))[0]
    d = refine(load_depth(f))
    d8 = np.clip(d * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(d8, mode="L").save(os.path.join(OUT, stem + ".png"))
    if (i + 1) % 400 == 0:
        print(f"[depth] {i + 1}/{len(files)}")
print("[depth] done ->", OUT)
