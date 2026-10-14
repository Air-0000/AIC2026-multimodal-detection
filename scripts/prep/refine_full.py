"""Batch depth refinement for the full AIC train+test splits.

Two methods, both emitting 16-bit metric PNG (same convention as the raw
camera depth, range ~0..19999) plus a turbo color map (fixed vmin/vmax so
raw/refined are comparable):

  da    -- DAv2-based ("DAv2 作引导"): holes filled with metric-aligned DAv2
          distance, valid pixels kept sharp (raw), hole edges sharpened by a
          guided filter whose GUIDE is the geometrically-correct DAv2 relative
          depth (so edges stay aligned to true geometry -> no collapse).
  morph -- pure morphological cleanup on the 16-bit raw (median + grey closing
          + gaussian). No guide -> no edge leakage. Large holes are left as-is.

Usage:
  PY=miniconda3/envs/deimv2/bin/python
  $PY refine_full.py --method da    --splits train test --limit 4   # smoke
  $PY refine_full.py --method da    --splits train test
  $PY refine_full.py --method morph --splits train test
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))


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


def colorize(d, vmin=0.0, vmax=19999.0):
    dn = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1) * 255
    return cv2.applyColorMap(dn.astype(np.uint8), cv2.COLORMAP_TURBO)


def da_refine(depth_path, ir_path, out_path, model, r=6, eps=1e-3, vmin=1.0):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype == np.uint8 or depth.ndim == 3:
        # only 16-bit single-channel metric PNG are valid inputs
        print(f"  ! skip (non-metric): {os.path.basename(depth_path)}")
        return None
    depth = depth.astype(np.float32)
    ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    if ir.ndim == 3:
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    ir = ir.astype(np.float32)
    I_ir = ir / (ir.max() + 1e-8)
    ir8 = np.clip(I_ir * 255, 0, 255).astype(np.uint8)
    inp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
    with torch.no_grad():
        d_rel = model.infer_image(inp, 518)
    d_rel = np.asarray(d_rel, dtype=np.float32).squeeze()
    valid = depth > vmin
    rv = d_rel[valid]
    tv = depth[valid]
    A = np.polyfit(rv, tv, 1)
    resid = tv - (A[0] * rv + A[1])
    m = np.abs(resid) < 3 * np.std(resid)
    A = np.polyfit(rv[m], tv[m], 1)
    dav2_abs = np.clip(A[0] * d_rel + A[1], 0, 65535)
    # DAv2 relative depth is the GUIDE (geometrically correct)
    I = (d_rel - d_rel.min()) / (d_rel.max() - d_rel.min() + 1e-8)
    base = np.where(valid, depth, dav2_abs)
    gf = guided_filter(I, base, r, eps)
    out = np.where(valid, depth, gf).astype(np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, np.clip(out, 0, 65535).astype(np.uint16))
    cv2.imwrite(os.path.splitext(out_path)[0] + "_color.png", colorize(out))
    return float((1 - valid.mean()) * 100)


def morph_refine(depth_path, out_path, vmin=1.0):
    from scipy import ndimage as ndi
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None or d.dtype == np.uint8 or d.ndim == 3:
        print(f"  ! skip (non-metric): {os.path.basename(depth_path)}")
        return None
    d = d.astype(np.float32)
    dn = d / 65535.0
    dn = ndi.median_filter(dn, size=3)
    struct = ndi.generate_binary_structure(2, 1)
    dn = ndi.grey_closing(dn, size=3, structure=struct)
    dn = ndi.gaussian_filter(dn, sigma=0.6)
    out = np.clip(dn * 65535.0, 0, 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)
    cv2.imwrite(os.path.splitext(out_path)[0] + "_color.png",
                colorize(out.astype(np.float32)))
    valid = d > vmin
    return float((1 - valid.mean()) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["da", "morph", "both"], default="both")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="process only N frames per split (smoke test)")
    ap.add_argument("--r", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-3)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if args.method in ("da", "both"):
        from depth_anything_v2.dpt import DepthAnythingV2
        ckpt = os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth")
        cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
        model = DepthAnythingV2(**cfg)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval().to(device)
        print(f"[DA] model on {device}")

    for split in args.splits:
        ddir = os.path.join(ROOT, "dataset", split, "depth")
        idir = os.path.join(ROOT, "dataset", split, "infrared")
        da_out = os.path.join(ROOT, "work_dir", "depth_refined_da_guided", split)
        mo_out = os.path.join(ROOT, "work_dir", "depth_refined_morph", split)
        files = []
        for ext in ("*.png", "*.PNG", "*.tif", "*.tiff"):
            files += glob.glob(os.path.join(ddir, ext))
        files = sorted(files)
        if args.limit:
            files = files[: args.limit]
        print(f"[{split}] {len(files)} frames  (method={args.method})")
        for i, f in enumerate(files):
            base = os.path.splitext(os.path.basename(f))[0]
            hole = 0.0
            if args.method in ("da", "both"):
                irp = None
                for e in (".png", ".PNG", ".tif", ".tiff"):
                    c = os.path.join(idir, base + e)
                    if os.path.exists(c):
                        irp = c
                        break
                if irp is None:
                    print(f"  ! {base}: no IR, skip DA")
                else:
                    hole = da_refine(os.path.join(ddir, os.path.basename(f)), irp,
                                     os.path.join(da_out, base + ".png"),
                                     model, args.r, args.eps)
            if args.method in ("morph", "both"):
                h2 = morph_refine(os.path.join(ddir, os.path.basename(f)),
                                  os.path.join(mo_out, base + ".png"))
                hole = hole if hole else (h2 or 0.0)
            if hole is None:
                hole = 0.0
            if (i + 1) % 100 == 0 or (args.limit and i + 1 == len(files)):
                print(f"  [{split}] {i + 1}/{len(files)}  hole={hole:.1f}%")
    print("DONE")


if __name__ == "__main__":
    main()
