"""DA fill + segmented metric alignment + IR-semantic region denoise/fill.
1) DA relative depth aligned to metric via NEAR/MID/FAR piecewise linear fit.
2) IR coarse segmentation: CLAHE-stretch IR then quantized connected components
   (skimage-free). If that yields too few regions (flat IR), fall back to regular
   32x32 blocks. Within each region:
   - holes filled by region-valid median (connectivity-aware, no cross-edge smear)
   - outlier pixels (|d - region median| > thr) replaced by region median
If region is entirely hole, use dilated-neighbor valid median.
"""
import argparse, glob, os, sys
import cv2
import numpy as np
import torch
from scipy import ndimage
ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2
CKPT = os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth")

def colorize(d):
    dn = np.clip((d.astype(np.float32)) / 19999.0, 0, 1) * 255
    return cv2.applyColorMap(dn.astype(np.uint8), cv2.COLORMAP_TURBO)

def da_align_segmented(depth_path, ir_path, model, n_seg=3):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    if ir.ndim == 3: ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    ir = ir.astype(np.float32); I = ir / (ir.max() + 1e-8)
    ir8 = np.clip(I * 255, 0, 255).astype(np.uint8)
    inp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
    with torch.no_grad(): d_rel = model.infer_image(inp, 518)
    valid = depth > 1.0
    rv = d_rel[valid]; tv = depth[valid]
    rv_sorted = np.sort(rv)
    thr_rv = [rv_sorted[int(len(rv) * (i + 1) / n_seg) - 1] for i in range(n_seg - 1)]
    lo = -np.inf; segs = []
    for i in range(n_seg):
        hi = thr_rv[i] if i < n_seg - 1 else np.inf
        m = (rv >= lo) & (rv < hi)
        A = np.polyfit(rv[m], tv[m], 1) if m.sum() > 2 else (1.0, 0.0)
        segs.append((lo, hi, A[0], A[1])); lo = hi
    dav_full = np.empty_like(d_rel)
    for lo, hi, a, b in segs:
        m = (d_rel >= lo) & (d_rel < hi)
        dav_full[m] = a * d_rel[m] + b
    dav_full = np.clip(dav_full, 0, 65535)
    fit_err = np.mean(np.abs(dav_full[valid] - depth[valid]))
    return depth, I, dav_full, valid, fit_err

def block_labels(H, W, bh=32, bw=32):
    ys = np.arange(H) // bh
    xs = np.arange(W) // bw
    return (ys[:, None] * (W // bw + 1) + xs[None, :]).astype(int)

def ir_segment(I, levels=32, ds=2, pre=2.0, min_regions=20):
    H, W = I.shape
    ism = cv2.resize(I, (W // ds, H // ds))
    ism8 = np.clip(ism * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ism8 = clahe.apply(ism8)
    ism = ism8.astype(np.float32) / 255.0
    ism = cv2.GaussianBlur(ism, (0, 0), pre)
    q = np.clip((ism * (levels - 1)).astype(int), 0, levels - 1).astype(np.uint8)
    _, ls = cv2.connectedComponents(q, connectivity=4)
    labels = cv2.resize(ls.astype(np.float32), (W, H),
                        interpolation=cv2.INTER_NEAREST).astype(int)
    if len(np.unique(labels)) < min_regions:
        print("  ! IR too flat, fallback to 32x32 blocks (regions=%d)" % len(np.unique(labels)))
        labels = block_labels(H, W)
    return labels

def semantic_denoise_fill(base_da, valid, labels, thr=600):
    out = base_da.copy().astype(np.float64)
    bad_total = 0
    for lab in np.unique(labels):
        m = labels == lab
        vin = m & valid
        if vin.sum() > 0:
            med = np.median(base_da[vin])
        else:
            dil = ndimage.binary_dilation(m) & valid
            med = np.median(base_da[dil]) if dil.sum() > 0 else np.median(base_da[valid])
        out[m & (~valid)] = med
        bad = m & valid & (np.abs(base_da - med) > thr)
        out[bad] = med
        bad_total += bad.sum()
    return np.clip(out, 0, 65535).astype(np.uint16), bad_total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-seg", type=int, default=3)
    ap.add_argument("--thr", type=int, default=600)
    ap.add_argument("--levels", type=int, default=32)
    args = ap.parse_args()
    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**cfg); model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device); model.eval()
    for split in args.splits:
        dd = os.path.join(ROOT, "dataset", split, "depth")
        idir = os.path.join(ROOT, "dataset", split, "infrared")
        od = os.path.join(ROOT, "work_dir", "depth_refined_da_semantic", split)
        files = sorted(glob.glob(os.path.join(dd, "*.png")))
        if args.limit: files = files[: args.limit]
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            gp = os.path.join(idir, base + ".png")
            if not os.path.exists(gp): continue
            depth, I, dav_full, valid, fit_err = da_align_segmented(f, gp, model, args.n_seg)
            base_da = np.where(valid, depth, dav_full).astype(np.float32)
            labels = ir_segment(I, args.levels)
            out, bad = semantic_denoise_fill(base_da, valid, labels, args.thr)
            os.makedirs(od, exist_ok=True)
            cv2.imwrite(os.path.join(od, base + ".png"), out)
            cv2.imwrite(os.path.join(od, base + "_color.png"), colorize(out.astype(np.float32)))
            print("wrote", base, "fit_err=%.1f regions=%d sparkles=%d(%.3f%%)" % (
                fit_err, len(np.unique(labels)), bad, 100.0 * bad / valid.size))
    print("DONE")

if __name__ == "__main__":
    main()
