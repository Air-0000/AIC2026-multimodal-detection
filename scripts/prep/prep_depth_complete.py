"""Depth COMPLETION for the RGB-D detection ablation (single image only).

WHY THIS EXISTS
---------------
``prep_depth_variants.py`` only *re-encodes* the raw depth (linear / inv_eq / ...).
It keeps every real pixel and never fills holes, so a "dead" depth pixel (==0)
stays dead. This script instead treats the real depth as SPARSE / NOISY and
COMPLETES it into a clean dense map -- the depth-completion paradigm:

    Inputs (per image, no temporal info):
      * real depth : dataset/{train,test}/depth  (16-bit mm, or 8-bit relative;
                     two mixed sources, holes where depth<=0)
      * guide  (IR): work_dir/deimv2_ir/images   (same basename; night-clear)
                     fallback to heat: work_dir/deimv2_heat/images

    Completion:
      1. holes (depth<=0) are filled by Depth-Anything-V2 estimated METRIC depth.
         DAv2 runs on the IR image (night-clear structure) -> a strong DENSE prior
         even where the real depth sensor returned 0.
      2. where real depth IS valid, the real value is kept (it is the ground truth).
      3. an IR-guided filter does edge-preserving cleanup (keeps depth edges aligned
         to IR structure; no leaking across object boundaries).

    Output: 8-bit PNG, LINEAR encoding (mm / 19999) -- identical to the `linear`
    variant. So the ONLY difference vs `linear` is the COMPLETION step, which is
    exactly the single variable the ablation isolates.

Usage
-----
  PY=miniconda3/envs/deimv2/bin/python
  $PY prep_depth_complete.py --splits train test        # materialise (GPU)
  $PY prep_depth_complete.py --splits train --limit 20   # smoke
Then:
  $PY gen_rgd_configs.py --variants completion
  cd vendor/DEIMv2 && $PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rgd_completion.yml --use-amp
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import grey_dilation

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
OUT_ROOT = os.path.join(ROOT, "work_dir", "depth_variants", "completion")
DEPTH_MAX = 19999.0
U8_TO_MM = DEPTH_MAX / 255.0
CKPT = os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth")

sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2


def load_depth(path):
    """-> (mm float32, raw float32, is_16bit). Both AIC depth sources."""
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    is16 = a.dtype == np.uint16
    raw = a.astype(np.float32)
    mm = raw if is16 else raw * U8_TO_MM
    return mm, raw, is16


def load_guide(path, size):
    if path is None:
        return None
    g = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if g is None:
        return None
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    if g.shape[:2] != size:
        g = cv2.resize(g, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return g


def box(img, r):
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), normalize=True)


def guided_filter(I, p, r, eps):
    mean_I = box(I, r)
    mean_p = box(p, r)
    mean_Ip = box(I * p, r)
    var_I = box(I * I, r) - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return box(a, r) * I + box(b, r)


def find_guide(base, dirs):
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for e in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            c = os.path.join(d, base + e)
            if os.path.exists(c):
                return c
    return None


def complete(mm, raw, guide, model):
    """Return completed depth in mm. Holes filled by DAv2(IR) + real kept + IR-GF."""
    valid = raw > 0
    if guide is None:
        # DEGRADE: no IR/heat for this frame -> fill holes by iterative grey
        # dilation of the nearest valid real depth (no semantic prior).
        if valid.sum() == 0:
            return np.zeros_like(mm)
        filled = np.where(valid, mm, 0.0)
        for _ in range(4):
            filled = np.where(valid, mm, grey_dilation(filled, size=3))
            if (filled > 0).all():
                break
        return filled

    I = guide / (guide.max() + 1e-8)
    ir8 = np.clip(I * 255, 0, 255).astype(np.uint8)
    inp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
    with torch.no_grad():
        d_rel = model.infer_image(inp, 518)
    d_rel = cv2.resize(d_rel, (mm.shape[1], mm.shape[0]), interpolation=cv2.INTER_AREA)

    rv = d_rel[valid]
    tv = mm[valid]
    if rv.size < 50:
        # too few valid pixels to fit a scale: use the global mean ratio
        ratio = tv.mean() / (d_rel[valid].mean() + 1e-8) if valid.sum() else 1.0
        dav2_abs = np.clip(d_rel * ratio, 0, DEPTH_MAX)
    else:
        A = np.polyfit(rv, tv, 1)
        resid = tv - (A[0] * rv + A[1])
        m = np.abs(resid) < 3 * np.std(resid)
        if m.sum() > 10:
            A = np.polyfit(rv[m], tv[m], 1)
        dav2_abs = np.clip(A[0] * d_rel + A[1], 0, DEPTH_MAX)

    base = np.where(valid, mm, dav2_abs)          # dense: real kept, holes = DAv2
    gf = guided_filter(I, base, 6, 1e-3)           # IR-guided edge-preserving
    out = np.where(valid, mm, gf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--guide-dirs", nargs="+",
                    default=[os.path.join(ROOT, "work_dir", "deimv2_ir", "images"),
                             os.path.join(ROOT, "work_dir", "deimv2_heat", "images")])
    args = ap.parse_args()

    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.cuda().eval()

    for split in args.splits:
        ddir = os.path.join(ROOT, "dataset", split, "depth")
        fs = []
        for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            fs += glob.glob(os.path.join(ddir, ext))
        fs = sorted(fs)
        if args.limit:
            fs = fs[: args.limit]
        out_dir = os.path.join(OUT_ROOT, split)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{split}] {len(fs)} depth frames -> {out_dir}", flush=True)
        for i, f in enumerate(fs):
            base = os.path.splitext(os.path.basename(f))[0]
            mm, raw, _ = load_depth(f)
            gpath = find_guide(base, args.guide_dirs)
            guide = load_guide(gpath, mm.shape[:2]) if gpath else None
            out = complete(mm, raw, guide, model)
            out8 = np.clip(out / DEPTH_MAX * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, base + ".png"), out8)
            if (i + 1) % 200 == 0:
                print(f"  [{split}] {i + 1}/{len(fs)}", flush=True)
        n = len(glob.glob(os.path.join(out_dir, "*.png")))
        print(f"[{split}] wrote {n} files", flush=True)


if __name__ == "__main__":
    main()
