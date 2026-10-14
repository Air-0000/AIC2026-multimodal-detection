"""DA fill + stronger sparkle/outlier removal (iterative, larger window).

Holes filled by metric-aligned DAv2; isolated pixels deviating a lot from the
local median are replaced by the median (smooth env value). Median is edge-aware
so real object boundaries are preserved. Iterating cleans small clusters too.

2026-09-22 update -- "process ALL depth maps":
  * AIC depth mixes two sources: 16-bit PNG (1920x1080, 0..19999 mm) and 8-bit
    grayscale JPEG stored as 3-channel (640x360, 0..255 relative). Both are now
    processed (previously `*.png` only, so the JPEG subset was silently skipped).
  * IR is looked up by stem with ANY extension (depth/IR extensions always match,
    but assuming `.png` broke the JPEG subset).
  * --thr is expressed in MILLIMETRES (the PNG convention). For the 8-bit JPEG
    source it is rescaled by src_max/19999 so the outlier criterion carries over.
  * `phase2` split supported -> dataset/test/AIC2026_PHASE_2_1000 (1000 images
    disjoint from dataset/test, never processed before).
  * --skip-existing to backfill without recomputing, plus per-image try/except so
    one bad image cannot abort a multi-hour run.

Usage:
  # smoke test
  python depth_refine_da_denoise.py --splits train --limit 4
  # everything (train 2000 + test 1000 + phase2 1000)
  python depth_refine_da_denoise.py --splits train test phase2 --skip-existing
"""
import argparse, glob, os, sys, time, warnings
import cv2
import numpy as np
import torch
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2
CKPT = os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth")

# PNG source is metric mm with this nominal full scale; --thr is defined against it.
METRIC_FULL_SCALE = 19999.0

SPLITS = {
    "train":  (os.path.join(ROOT, "dataset/train/depth"),
               os.path.join(ROOT, "dataset/train/infrared")),
    "test":   (os.path.join(ROOT, "dataset/test/depth"),
               os.path.join(ROOT, "dataset/test/infrared")),
    "phase2": (os.path.join(ROOT, "dataset/test/AIC2026_PHASE_2_1000/depth"),
               os.path.join(ROOT, "dataset/test/AIC2026_PHASE_2_1000/infrared")),
}

EXTS = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")


def colorize(d, vmax=METRIC_FULL_SCALE):
    dn = np.clip(d.astype(np.float32) / max(vmax, 1e-6), 0, 1) * 255
    return cv2.applyColorMap(dn.astype(np.uint8), cv2.COLORMAP_TURBO)


def read_gray(path):
    """Read a depth or IR map as single-channel float32 (JPEG sources are 3-ch gray)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def nominal_scale_for(path):
    """Nominal full scale of a depth source: 16-bit PNG is metric mm, 8-bit JPEG is 0..255."""
    ext = os.path.splitext(path)[1].lower()
    return 255.0 if ext in (".jpg", ".jpeg") else METRIC_FULL_SCALE


def find_ir(idir, stem):
    for e in ("png", "PNG", "jpg", "JPG", "jpeg", "JPEG", "tif", "tiff"):
        p = os.path.join(idir, f"{stem}.{e}")
        if os.path.exists(p):
            return p
    return None


def da_align(depth, ir, model, nominal_scale, min_fit=32):
    """Align DAv2 relative depth to the source's own metric scale via robust linear fit.

    Depth maps that are entirely holes (17 across train/test/phase2, all from the
    8-bit JPEG source) have nothing to fit against; for those we fall back to the
    DAv2 relative depth normalized into the source's nominal range.
    """
    I = ir / (ir.max() + 1e-8)
    ir8 = np.clip(I * 255, 0, 255).astype(np.uint8)
    inp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
    with torch.no_grad():
        d_rel = model.infer_image(inp, 518)
    d_rel = cv2.resize(d_rel.astype(np.float32), (depth.shape[1], depth.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    valid = depth > 1.0
    if valid.sum() < min_fit:
        lo, hi = float(d_rel.min()), float(d_rel.max())
        dav2_abs = (d_rel - lo) / (hi - lo + 1e-8) * nominal_scale
        return depth, I, np.clip(dav2_abs, 0, 65535), valid, -1.0
    rv, tv = d_rel[valid], depth[valid]
    ok = np.isfinite(rv) & np.isfinite(tv)
    rv, tv = rv[ok], tv[ok]
    if rv.size < min_fit or np.ptp(rv) < 1e-6:
        lo, hi = float(d_rel.min()), float(d_rel.max())
        dav2_abs = (d_rel - lo) / (hi - lo + 1e-8) * nominal_scale
        return depth, I, np.clip(dav2_abs, 0, 65535), valid, -1.0
    A = _robust_fit(rv, tv)
    dav2_abs = np.clip(A[0] * d_rel + A[1], 0, 65535)
    fit_err = float(np.mean(np.abs(dav2_abs[valid] - depth[valid])))
    return depth, I, dav2_abs, valid, fit_err


def _robust_fit(rv, tv):
    """deg-1 robust fit; falls back to a median-ratio scale if SVD fails to converge."""
    try:
        A = np.polyfit(rv, tv, 1)
        resid = tv - (A[0] * rv + A[1])
        s = np.std(resid)
        if s > 0:
            m = np.abs(resid) < 3 * s
            if m.sum() > 32:
                A = np.polyfit(rv[m], tv[m], 1)
        if not np.all(np.isfinite(A)):
            raise np.linalg.LinAlgError("non-finite coefficients")
        return A
    except np.linalg.LinAlgError:
        scale = float(np.median(tv)) / (float(np.median(rv)) + 1e-8)
        return np.array([scale, 0.0])


def remove_sparkle(d, thr, k=5, iters=2):
    """Replace pixels deviating > thr from the local median by that median.

    cv2.medianBlur is used instead of scipy.ndimage.median_filter: on a 1080p map
    it is ~400x faster (0.002s vs 0.94s) and the result is identical away from the
    border (verified: mean |diff| < 1 on uint16 truncation). OpenCV caps ksize at 5
    for 16-bit input, so larger k is clamped.
    """
    k = int(k) | 1  # must be odd
    k = max(3, min(k, 5))
    out = d.astype(np.float32)
    total = np.zeros(out.shape, dtype=bool)
    for _ in range(iters):
        med = cv2.medianBlur(np.clip(out, 0, 65535).astype(np.uint16), k).astype(np.float32)
        diff = np.abs(out - med)
        mask = diff > thr
        out[mask] = med[mask]
        total |= mask
    return out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train"], choices=list(SPLITS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thr", type=float, default=600, help="outlier threshold in mm (PNG scale)")
    ap.add_argument("--k", type=int, default=5, help="median window, clamped to 3..5 (OpenCV 16-bit limit)")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--min-valid-frac", type=float, default=0.0,
                    help="skip images whose valid-pixel fraction is below this")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "work_dir/depth_refined_da_denoise"))
    args = ap.parse_args()

    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    t0 = time.time()
    n_done = n_skip = n_fail = 0
    for split in args.splits:
        dd, idir = SPLITS[split]
        od = os.path.join(args.out_root, split)
        os.makedirs(od, exist_ok=True)
        files = []
        for e in EXTS:
            files += glob.glob(os.path.join(dd, e))
        files = sorted(files)
        if args.limit:
            files = files[: args.limit]
        print(f"[{split}] {len(files)} files -> {od}", flush=True)
        for idx, f in enumerate(files):
            base = os.path.splitext(os.path.basename(f))[0]
            op = os.path.join(od, base + ".png")
            ocp = os.path.join(od, base + "_color.png")
            if args.skip_existing and os.path.exists(op) and os.path.exists(ocp):
                n_skip += 1
                continue
            gp = find_ir(idir, base)
            if gp is None:
                print(f"  ! no IR for {base}, skipped", flush=True)
                n_fail += 1
                continue
            try:
                depth = read_gray(f)
                ir = read_gray(gp)
                nom = nominal_scale_for(f)
                valid = depth > 1.0
                vf = valid.mean()
                if vf < args.min_valid_frac:
                    print(f"  ! {base}: valid frac {vf:.3f} < {args.min_valid_frac}, skipped", flush=True)
                    n_fail += 1
                    continue
                _, I, dav2_abs, valid, fit_err = da_align(depth, ir, model, nom)
                base_da = np.where(valid, depth, dav2_abs).astype(np.float32)
                # --thr is mm-scale; rescale for the 8-bit JPEG source (0..255)
                thr_eff = max(args.thr * (nom / METRIC_FULL_SCALE), 1.0)
                out, mask = remove_sparkle(base_da, thr_eff, args.k, args.iters)
                out = np.clip(out, 0, 65535).astype(np.uint16)
                src_max = float(depth[valid].max()) if valid.any() else nom
                cv2.imwrite(op, out)
                cv2.imwrite(ocp, colorize(out.astype(np.float32), vmax=max(src_max, 1.0)))
                n_done += 1
                if idx % 50 == 0 or n_done <= 5:
                    print(f"  [{idx}/{len(files)}] {base} holes={100*(1-vf):.1f}% "
                          f"fit_err={fit_err:.1f} thr_eff={thr_eff:.1f} "
                          f"sparkles={100.0*mask.sum()/mask.size:.3f}% "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # never abort a long run on one bad image
                print(f"  ! {base} FAILED: {type(e).__name__}: {e}", flush=True)
                n_fail += 1
    print(f"DONE done={n_done} skipped={n_skip} failed={n_fail} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
