"""JBF + DA hybrid: near holes filled by JBF (edge-preserving), far holes by DA (clean).

Far/deep holes are poorly constrained for JBF (few valid neighbours, IR noise leaks),
so we fill them with the metric-aligned DAv2 distance; near holes (within BAND px of a
valid pixel) use JBF to keep sharp object edges.

Usage:
  python depth_refine_jbf_da_mix.py --splits train --limit 2 [--sc 0.15 --pre 4 --band 10]
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))
sys.path.insert(0, os.path.join(ROOT, "scripts/prep"))
from depth_anything_v2.dpt import DepthAnythingV2
from depth_refine_jbf import jbf_fill, colorize

CKPT = os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth")


def da_align(depth_path, ir_path, model):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    if ir.ndim == 3:
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    ir = ir.astype(np.float32)
    I = ir / (ir.max() + 1e-8)
    ir8 = np.clip(I * 255, 0, 255).astype(np.uint8)
    inp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
    with torch.no_grad():
        d_rel = model.infer_image(inp, 518)
    valid = depth > 1.0
    rv = d_rel[valid]
    tv = depth[valid]
    A = np.polyfit(rv, tv, 1)
    resid = tv - (A[0] * rv + A[1])
    m = np.abs(resid) < 3 * np.std(resid)
    A = np.polyfit(rv[m], tv[m], 1)
    dav2_abs = np.clip(A[0] * d_rel + A[1], 0, 65535)
    return depth, I, dav2_abs, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--r", type=int, default=7)
    ap.add_argument("--sc", type=float, default=0.15)
    ap.add_argument("--pre", type=float, default=4.0)
    ap.add_argument("--band", type=int, default=10,
                    help="near-hole radius (px) filled by JBF; beyond -> DA")
    args = ap.parse_args()
    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    for split in args.splits:
        dd = os.path.join(ROOT, "dataset", split, "depth")
        idir = os.path.join(ROOT, "dataset", split, "infrared")
        od = os.path.join(ROOT, "work_dir", "depth_refined_jbf_da_mix", split)
        files = sorted(glob.glob(os.path.join(dd, "*.png")))
        if args.limit:
            files = files[: args.limit]
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            gp = os.path.join(idir, base + ".png")
            if not os.path.exists(gp):
                continue
            depth, I, dav2_abs, valid = da_align(f, gp, model)
            if args.pre > 0:
                I = cv2.GaussianBlur(I, (0, 0), args.pre)
            jbf = jbf_fill(depth, I, args.r, 7.0, args.sc)
            dt = distance_transform_edt(~valid)
            out = np.where(valid, depth, dav2_abs).astype(np.float64)
            near = (~valid) & (dt <= args.band)
            out[near] = jbf[near]
            out = np.clip(out, 0, 65535).astype(np.uint16)
            os.makedirs(od, exist_ok=True)
            cv2.imwrite(os.path.join(od, base + ".png"), out)
            cv2.imwrite(os.path.join(od, base + "_color.png"),
                        colorize(out.astype(np.float32)))
            print("wrote", base, "near-holes", int(near.sum()))
    print("DONE")


if __name__ == "__main__":
    main()
