"""Edge-preserving hole filling for AIC depth using Depth-Anything-V2 (DAv2).

DAv2 gives a metric-correct global distance (aligned to valid depth via a robust
linear fit), but local edges are over-smoothed. We fill holes with the aligned
DAv2 distance, then run an IR-guided guided filter (optionally pre-smoothed, so
flat object interiors don't pick up IR texture/false edges) -- valid pixels kept
sharp. The IR guide only sharpens edges where the (smoothed) DAv2 distance itself
also has a gradient, filtering out IR false edges.

Usage:
  python depth_refine_da_guided.py \
      --depth dataset/train/depth/000003_013_00000085.png \
      --ir    dataset/train/infrared/000003_013_00000085.png \
      --out   work_dir/depth_refined_da_guided/train/000003_013_00000085_refined.png \
      --psmooth 12 --crop 0.647 0.588 0.9 1.0 --zoom 2 --gray
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

CKPT = os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth")


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


def colorize(d, vmin, vmax):
    dn = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1) * 255
    return cv2.applyColorMap(dn.astype(np.uint8), cv2.COLORMAP_TURBO)


def grayize(d, vmin, vmax):
    return np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", required=True)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--guide", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--vmin", type=float, default=1.0)
    ap.add_argument("--crop", nargs=4, type=float, default=None,
                    metavar=("CX", "CY", "W", "H"))
    ap.add_argument("--zoom", type=int, default=2)
    ap.add_argument("--band", type=int, default=0,
                    help="if >0, IR-guided fill only within BAND px of a valid pixel")
    ap.add_argument("--psmooth", type=float, default=0.0,
                    help="gaussian sigma to pre-smooth DAv2 distance as guided-filter p")
    ap.add_argument("--gray", action="store_true", help="grayscale (linear) compare")
    args = ap.parse_args()

    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED).astype(np.float32)
    ir = cv2.imread(args.ir, cv2.IMREAD_UNCHANGED)
    if ir.ndim == 3:
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    ir = ir.astype(np.float32)
    I = ir / (ir.max() + 1e-8)
    if args.guide:
        g = cv2.imread(args.guide, cv2.IMREAD_UNCHANGED)
        if g.ndim == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        I = g.astype(np.float32) / (g.max() + 1e-8)

    valid = depth > args.vmin
    print("hole frac = %.1f%%" % (100 * (1 - valid.mean())))

    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    ir8 = np.clip(I * 255, 0, 255).astype(np.uint8)
    inp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
    with torch.no_grad():
        d_rel = model.infer_image(inp, 518)

    rv = d_rel[valid]
    tv = depth[valid]
    A = np.polyfit(rv, tv, 1)
    resid = tv - (A[0] * rv + A[1])
    m = np.abs(resid) < 3 * np.std(resid)
    A = np.polyfit(rv[m], tv[m], 1)
    print("align slope=%.3f intercept=%.1f" % (A[0], A[1]))
    dav2_abs = np.clip(A[0] * d_rel + A[1], 0, 65535)
    err = dav2_abs[valid] - depth[valid]
    print("DAv2 abs valid  MAE=%.1f  RMSE=%.1f" % (np.mean(np.abs(err)), np.sqrt(np.mean(err ** 2))))

    dav2_s = cv2.GaussianBlur(dav2_abs, (0, 0), args.psmooth) if args.psmooth > 0 else dav2_abs
    base = np.where(valid, depth, dav2_s)
    gf = guided_filter(I, base, args.r, args.eps)
    out = np.where(valid, depth, gf)
    if args.band > 0:
        dt = distance_transform_edt(~valid)
        band = (dt < args.band) & (~valid)
        out = depth.copy()
        out[band] = gf[band]
        out[~valid & ~band] = dav2_abs[~valid & ~band]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, np.clip(out, 0, 65535).astype(np.uint16))

    vlo, vhi = np.percentile(depth[valid], [1, 99])
    if args.gray:
        g3 = lambda d: grayize(d, vlo, vhi).astype(np.uint8)
        vis = lambda d: g3(d)
    else:
        vis = lambda d: colorize(d, vlo, vhi)
    comp = np.hstack([vis(depth), vis(dav2_abs), vis(out)])
    cpath = os.path.splitext(args.out)[0] + ("_graycmp.png" if args.gray else "_cmp.png")
    cv2.imwrite(cpath, comp)

    if args.crop:
        cx, cy, w, h = args.crop
        H, W = depth.shape
        x0 = int(max(0, (cx - w / 2) * W)); x1 = int(min(W, (cx + w / 2) * W))
        y0 = int(max(0, (cy - h / 2) * H)); y1 = int(min(H, (cy + h / 2) * H))
        z = args.zoom
        cr = lambda x: cv2.resize(x[y0:y1, x0:x1], (0, 0), fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        crop = np.hstack([cr(vis(depth)), cr(vis(dav2_abs)), cr(vis(out))])
        c2 = os.path.splitext(args.out)[0] + ("_graycrop.png" if args.gray else "_crop_cmp.png")
        cv2.imwrite(c2, crop)
        print("crop compare", c2)

    print("saved", args.out)
    print("compare", cpath)


if __name__ == "__main__":
    main()
