"""Joint-Bilateral-Filter hole filling for AIC depth -- new candidate vs da_guided.

Unlike the abandoned guided-filter variants, JBF weights valid neighbours by BOTH
spatial distance AND guidance-intensity similarity, so filled holes keep sharp
object edges instead of leaking RGB/IR texture. Holes (depth<=1) are filled by
iterative band propagation (each band uses already-filled inner pixels).

Usage (single):
  python depth_refine_jbf.py --depth D --guide G --out O [--r 7 --sc 0.3]
Batch (smoke only -- per-pixel Python loop is slow; use --limit):
  python depth_refine_jbf.py --splits train test --limit 4
"""
import argparse
import glob
import os

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = "/home/constant/CodeBuddy/Dev/AIC"


def colorize(d, vmin=0.0, vmax=19999.0):
    dn = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1) * 255
    return cv2.applyColorMap(dn.astype(np.uint8), cv2.COLORMAP_TURBO)


def jbf_fill(depth, G, r=7, sigma_s=7.0, sigma_c=0.3, max_band=80):
    valid = depth > 1.0
    out = depth.astype(np.float64).copy()
    G = (G / (G.max() + 1e-8)).astype(np.float32)
    H, W = depth.shape
    gy, gx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    sp = np.exp(-(gx**2 + gy**2) / (2 * sigma_s**2))
    for band in range(1, max_band + 1):
        dt = distance_transform_edt(~valid)
        mask = np.ceil(dt).astype(int) == band
        if not mask.any():
            break
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            x0, x1 = max(0, x - r), min(W, x + r + 1)
            sub_v = valid[y0:y1, x0:x1]
            if not sub_v.any():
                continue
            sub_d = out[y0:y1, x0:x1]
            sub_g = G[y0:y1, x0:x1]
            dy0, dy1 = y0 - y + r, y1 - y + r
            dx0, dx1 = x0 - x + r, x1 - x + r
            w = sp[dy0:dy1, dx0:dx1] * np.exp(
                -((sub_g - G[y, x]) ** 2) / (2 * sigma_c ** 2)) * sub_v
            s = w.sum()
            if s > 0:
                out[y, x] = np.sum(w * sub_d) / s
        valid = out > 1.0
    return np.clip(out, 0, 65535).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth")
    ap.add_argument("--guide")
    ap.add_argument("--out")
    ap.add_argument("--r", type=int, default=7)
    ap.add_argument("--sc", type=float, default=0.3, help="guide-intensity sigma")
    ap.add_argument("--pre", type=float, default=0.0, help="gaussian pre-smooth guide (denoise)")
    ap.add_argument("--outdir", default="depth_refined_jbf")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.splits:
        for split in args.splits:
            dd = os.path.join(ROOT, "dataset", split, "depth")
            gd = os.path.join(ROOT, "dataset", split, "infrared")
            od = os.path.join(ROOT, "work_dir", args.outdir, split)
            files = sorted(glob.glob(os.path.join(dd, "*.png")))
            if args.limit:
                files = files[: args.limit]
            print(f"[{split}] {len(files)} frames")
            for i, f in enumerate(files):
                base = os.path.splitext(os.path.basename(f))[0]
                gp = os.path.join(gd, base + ".png")
                if not os.path.exists(gp):
                    continue
                depth = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if depth is None or depth.dtype == np.uint8 or depth.ndim == 3:
                    continue
                g = cv2.imread(gp, cv2.IMREAD_UNCHANGED)
                if g.ndim == 3:
                    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
                if args.pre > 0:
                    g = cv2.GaussianBlur(g, (0, 0), args.pre)
                out = jbf_fill(depth.astype(np.float32), g.astype(np.float32),
                               args.r, args.sc)
                os.makedirs(od, exist_ok=True)
                cv2.imwrite(os.path.join(od, base + ".png"), out)
                cv2.imwrite(os.path.join(od, base + "_color.png"),
                            colorize(out.astype(np.float32)))
                if (i + 1) % 20 == 0 or i + 1 == len(files):
                    print(f"  {i + 1}/{len(files)}")
        print("DONE")
        return
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED).astype(np.float32)
    g = cv2.imread(args.guide, cv2.IMREAD_UNCHANGED)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    out = jbf_fill(depth, g.astype(np.float32), args.r, args.sc)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, out)
    cv2.imwrite(os.path.splitext(args.out)[0] + "_color.png",
                colorize(out.astype(np.float32)))
    print("saved", args.out)


if __name__ == "__main__":
    main()
