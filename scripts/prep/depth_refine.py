"""Refine real RGB-D camera depth maps using an aligned guidance image.

AIC depth comes from a real RGB-D camera (16-bit single-channel PNG, true
metric scale, range ~0..19999). The problem is sensor noise + holes (depth==0)
+ fuzzy edges, NOT estimation error. This script improves it with two classic,
fast, CPU-only steps guided by an aligned image (default: IR residual / heat,
which carries the useful night-time structure; visible RGB is near-black at
night and a poor guide):

  1. Hole filling  -- pixels with depth < --vmin (default 1.0, i.e. 0) are
     invalid; filled by Telea inpainting on the normalized depth.
  2. Edge-preserving denoise -- a guided filter (He et al., ECCV 2010) with the
     guidance luminance: smooths noise but KEEPS depth edges aligned to the
     guidance edges (the key fix for fuzzy/leaking depth boundaries).

No GPU, no extra deps (numpy + cv2 only). Does NOT touch the running training.

Usage:
  # one image -> refined 16-bit PNG
  python3 depth_refine.py dataset/train/depth/000002.png \
      --guide-dir work_dir/deimv2_heat/images --color -o work_dir/depth_refined_irheat/000002_refined.png

  # whole directory (IR residual preferred, fall back to raw IR)
  python3 depth_refine.py dataset/train/depth/ \
      --guide-dir work_dir/deimv2_heat/images --guide-dir2 work_dir/deimv2_ir/images \
      --out-dir work_dir/depth_refined_irheat/train
"""

import argparse
import glob
import os

import cv2
import numpy as np


def load_depth(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def load_guide(path):
    """Load a guidance image (IR residual / raw IR / visible) as a single
    [0,1] grayscale map. Accepts 1-ch or 3-ch inputs; translation/scale of the
    guide does not affect the guided filter's edge coefficients."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32) / 255.0


def _box(img, r):
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), normalize=True)


def guided_filter(I, p, r, eps):
    mean_I = _box(I, r)
    mean_p = _box(p, r)
    mean_Ip = _box(I * p, r)
    var_I = _box(I * I, r) - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return _box(a, r) * I + _box(b, r)


def refine(depth, I, r=6, eps=1e-3, vmin=1.0, inpaint_radius=5):
    invalid = depth < vmin
    frac = float(invalid.mean())
    dmin, dmax = np.nanmin(depth), np.nanmax(depth)
    span = (dmax - dmin) or 1.0
    dn = np.clip((depth - dmin) / span, 0, 1)
    dn8 = (dn * 255).astype(np.uint8)
    mask = (invalid * 255).astype(np.uint8)
    dn8_f = cv2.inpaint(dn8, mask, inpaint_radius, cv2.INPAINT_TELEA)
    depth_f = dn8_f.astype(np.float32) / 255.0 * span + dmin
    out = guided_filter(I, depth_f, r, eps)
    return out, frac


def _colorize(d):
    dn = np.clip((d - d.min()) / (d.max() - d.min() + 1e-6), 0, 1) * 255
    return cv2.applyColorMap(dn.astype(np.uint8), cv2.COLORMAP_TURBO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--rgb-dir", default=None, help="(alias) guidance dir, same basename")
    ap.add_argument("--guide-dir", default=None, help="primary guidance dir (e.g. IR residual/heat)")
    ap.add_argument("--guide-dir2", default=None, help="fallback guidance dir if missing in --guide-dir (e.g. raw IR)")
    ap.add_argument("--r", type=int, default=6, help="guided-filter radius (px)")
    ap.add_argument("--eps", type=float, default=1e-3, help="guided-filter regularization (smaller=more edge-preserving)")
    ap.add_argument("--vmin", type=float, default=1.0, help="depth below this counts as a hole")
    ap.add_argument("--color", action="store_true", help="also write a side-by-side color compare")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    guide_dir = args.guide_dir or args.rgb_dir
    guide_dir2 = args.guide_dir2
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

    if os.path.isdir(args.input):
        files = []
        for ext in ("*.png", "*.PNG", "*.tif", "*.tiff"):
            files += glob.glob(os.path.join(args.input, ext))
    else:
        files = [args.input]

    for f in sorted(files):
        depth = load_depth(f)
        base = os.path.splitext(os.path.basename(f))[0]
        guide = None
        for gd in (guide_dir, guide_dir2):
            if not gd:
                continue
            for e in exts:
                cand = os.path.join(gd, base + e)
                if os.path.exists(cand):
                    guide = cand
                    break
            if guide:
                break
        if guide is None:
            print(f"{f}: no matching guide -> skip")
            continue
        I = load_guide(guide)
        out, frac = refine(depth, I, args.r, args.eps, args.vmin)
        out16 = np.clip(out, 0, 65535).astype(np.uint16)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            outp = os.path.join(args.out_dir, base + "_refined.png")
        else:
            outp = args.out or (os.path.splitext(f)[0] + "_refined.png")
        cv2.imwrite(outp, out16)
        # always emit a colored visualization of the refined depth
        # (turbo, per-image min/max) so the colored map is generated directly
        cpath = os.path.splitext(outp)[0] + "_color.png"
        cv2.imwrite(cpath, _colorize(out))
        msg = (f"{f} + {guide}  holes={frac*100:.1f}%  -> {outp}  "
               f"range=({out.min():.1f},{out.max():.1f})  color={cpath}")
        if args.color:
            comp = np.hstack([_colorize(depth), _colorize(out)])
            cmp_path = os.path.splitext(outp)[0] + "_compare.png"
            cv2.imwrite(cmp_path, comp)
            msg += f"  compare={cmp_path}"
        print(msg)


if __name__ == "__main__":
    main()
