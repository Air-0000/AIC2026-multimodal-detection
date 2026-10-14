"""DA-first then JBF: fill holes with metric-aligned DAv2, then edge-preserving
JOINT-bilateral refine on the dense result (guide=IR).

Key detail: JBF runs over the whole dense map, but the refined result is only
applied to the ORIGINALLY-HOLE region. Originally-valid pixels are kept intact,
so JBF only smooths/fills holes (IR-guided) instead of blurring real edges.
"""
import argparse, glob, os, sys
import cv2
import numpy as np
import torch
ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2
CKPT = os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth")

def colorize(d):
    dn = np.clip((d.astype(np.float32)) / 19999.0, 0, 1) * 255
    return cv2.applyColorMap(dn.astype(np.uint8), cv2.COLORMAP_TURBO)

def da_align(depth_path, ir_path, model):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    if ir.ndim == 3: ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    ir = ir.astype(np.float32); I = ir / (ir.max() + 1e-8)
    ir8 = np.clip(I * 255, 0, 255).astype(np.uint8)
    inp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
    with torch.no_grad(): d_rel = model.infer_image(inp, 518)
    valid = depth > 1.0
    rv = d_rel[valid]; tv = depth[valid]
    A = np.polyfit(rv, tv, 1); resid = tv - (A[0] * rv + A[1])
    m = np.abs(resid) < 3 * np.std(resid)
    A = np.polyfit(rv[m], tv[m], 1)
    dav2_abs = np.clip(A[0] * d_rel + A[1], 0, 65535)
    return depth, I, dav2_abs, valid

def jbf_smooth(base, G, r=7, sigma_s=7.0, sigma_c=0.3):
    bs = np.clip(base / 65535.0 * 255, 0, 255).astype(np.uint8)
    gs = np.clip((G / (G.max() + 1e-8)) * 255, 0, 255).astype(np.uint8)
    try:
        from cv2.ximgproc import jointBilateralFilter
        out = jointBilateralFilter(bs, gs, d=-1, sigmaColor=max(1.0, sigma_c * 255),
                                   sigmaSpace=sigma_s)
        return out.astype(np.float64) / 255.0 * 65535.0
    except Exception as e:
        print("  ! ximgproc unavailable, fallback self JBF (IR-guided):", e)
        H, W = base.shape; out = base.astype(np.float64).copy()
        Gg = (G / (G.max() + 1e-8)).astype(np.float32)
        gy, gx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        sp = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma_s ** 2))
        for y in range(H):
            for x in range(W):
                y0, y1 = max(0, y - r), min(H, y + r + 1)
                x0, x1 = max(0, x - r), min(W, x + r + 1)
                dy_top = y - y0; dx_left = x - x0
                sub_d = out[y0:y1, x0:x1]; sub_g = Gg[y0:y1, x0:x1]
                w = sp[r - dy_top:r + (y1 - y), r - dx_left:r + (x1 - x)] * np.exp(
                    -((sub_g - Gg[y, x]) ** 2) / (2 * sigma_c ** 2))
                s = w.sum()
                if s > 0: out[y, x] = np.sum(w * sub_d) / s
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--sc", type=float, default=0.15)
    ap.add_argument("--pre", type=float, default=4.0)
    args = ap.parse_args()
    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**cfg); model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device); model.eval()
    for split in args.splits:
        dd = os.path.join(ROOT, "dataset", split, "depth")
        idir = os.path.join(ROOT, "dataset", split, "infrared")
        od = os.path.join(ROOT, "work_dir", "depth_refined_da_then_jbf", split)
        files = sorted(glob.glob(os.path.join(dd, "*.png")))
        if args.limit: files = files[: args.limit]
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            gp = os.path.join(idir, base + ".png")
            if not os.path.exists(gp): continue
            depth, I, dav2_abs, valid = da_align(f, gp, model)
            if args.pre > 0: I = cv2.GaussianBlur(I, (0, 0), args.pre)
            base_da = np.where(valid, depth, dav2_abs).astype(np.float32)
            jbf = jbf_smooth(base_da, I, args.r, float(args.r), args.sc)
            # keep originally-valid pixels intact; only refine originally-hole region
            out = np.where(valid, depth, jbf).astype(np.float32)
            out = np.clip(out, 0, 65535).astype(np.uint16)
            os.makedirs(od, exist_ok=True)
            cv2.imwrite(os.path.join(od, base + ".png"), out)
            cv2.imwrite(os.path.join(od, base + "_color.png"), colorize(out.astype(np.float32)))
            print("wrote", base)
    print("DONE")

if __name__ == "__main__":
    main()
