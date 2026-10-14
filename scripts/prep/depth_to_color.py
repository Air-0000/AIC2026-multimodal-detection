"""Convert (grayscale) depth maps to colored visualizations via a colormap.

AIC depth lives in dataset/train/depth (and dataset/test/depth) as 16-bit
single-channel PNG files. This script loads them with IMREAD_UNCHANGED (so the
16-bit range is preserved), normalizes to 0..255, and applies an OpenCV
colormap (turbo by default -- the perceptual rainbow used in most CV papers).

Usage:
  # one image -> prints its shape/value range, writes a colored copy
  python depth_to_color.py dataset/train/depth/000002.png -o work_dir/depth_color_demo.png

  # a whole directory (writes <name>_color.png next to each source)
  python depth_to_color.py dataset/train/depth/

  # fixed range across images (e.g. 16-bit meters*1000); default = per-image min/max
  python depth_to_color.py dataset/train/depth/ --vmin 0 --vmax 65535

Colormaps: turbo (default), jet, viridis, plasma, magma, inferno.
"""

import argparse
import glob
import os

import cv2
import numpy as np

_CMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "magma": cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO,
}


def load_depth(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:  # some depths are stored as RGB by mistake -> take luminance
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def to_color(depth, vmin=None, vmax=None, cmap=cv2.COLORMAP_TURBO):
    d = depth.copy()
    if vmin is None:
        vmin = float(np.nanmin(d)) if np.isfinite(d).any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(d)) if np.isfinite(d).any() else 1.0
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    d = np.clip(d, vmin, vmax)
    d = (d - vmin) / (vmax - vmin) * 255.0
    d = d.astype(np.uint8)
    return cv2.applyColorMap(d, cmap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="depth image or directory")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--vmin", type=float, default=None, help="fixed lower bound")
    ap.add_argument("--vmax", type=float, default=None, help="fixed upper bound")
    ap.add_argument("--cmap", default="turbo", choices=list(_CMAPS))
    ap.add_argument("--suffix", default="_color")
    ap.add_argument("--out-dir", default=None, help="write colored outputs here instead of next to sources")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not regenerate outputs that already exist (useful to backfill missing files)")
    args = ap.parse_args()

    cmap = _CMAPS[args.cmap]
    # NOTE: AIC depth mixes two sources -- 16-bit PNG (1920x1080, mm) and 8-bit
    # grayscale JPEG (640x360, 0..255). Both must be listed here, otherwise the
    # JPEG subset is silently skipped (it used to be, leaving depth_color short).
    if os.path.isdir(args.input):
        files = []
        for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG",
                    "*.tif", "*.tiff", "*.exr"):
            files += glob.glob(os.path.join(args.input, ext))
    else:
        files = [args.input]

    for f in sorted(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out = os.path.join(args.out_dir, stem + args.suffix + ".png")
        else:
            out = args.out or (os.path.splitext(f)[0] + args.suffix + ".png")
        if args.skip_existing and os.path.exists(out):
            continue
        d = load_depth(f)
        c = to_color(d, args.vmin, args.vmax, cmap)
        cv2.imwrite(out, c)
        print(f"{f}  shape={d.shape}  dtype={d.dtype}  "
              f"range=({d.min():.1f},{d.max():.1f})  ->  {out}")


if __name__ == "__main__":
    main()
