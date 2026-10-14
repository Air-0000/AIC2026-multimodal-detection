#!/usr/bin/env python3
"""
Unified P0 Depth Generation & Baking Pipeline:
Integrates:
  1. Disparity-Space Linear Polynomial Fit (1/Z = a*d + b) (eliminates polyfit singularity)
  2. Bi-Modal Structural Guided Filter (Max-gradient RGB + IR) (snaps depth to physical borders)
  3. Sky & Infinity Clamping (locks sky blind zones Z=0 to True Infinity, eliminating fake 0.5m walls)
  4. Valid Sensor Mask Guarantee (retains 100% of real LiDAR/ToF physical measurements)

Output format: 16-bit PNG depth maps (millimeter scale, 0-65535mm).
Zero trained weights: pure physical geometry + zero-shot DAv2 foundation prior.
"""
import os
import sys
import glob
import time
import argparse
import cv2
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor

import warnings
warnings.filterwarnings("ignore", message=".*Polyfit.*")
warnings.filterwarnings("ignore", message=".*polyfit.*")
cv2.setNumThreads(2)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

DA_CKPT = os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth")

def load_da_model(device="cuda"):
    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(DA_CKPT, map_location="cpu"))
    return model.to(device).eval()

def guided_filter_core(guide_f, src_f, r=4, eps=1e-3):
    mean_I = cv2.boxFilter(guide_f, -1, (2*r+1, 2*r+1))
    mean_p = cv2.boxFilter(src_f, -1, (2*r+1, 2*r+1))
    mean_Ip = cv2.boxFilter(guide_f * src_f, -1, (2*r+1, 2*r+1))
    var_I = cv2.boxFilter(guide_f * guide_f, -1, (2*r+1, 2*r+1)) - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, (2*r+1, 2*r+1)) * guide_f + cv2.boxFilter(b, -1, (2*r+1, 2*r+1))

def process_single_depth(depth_raw, d_rel, vis, ir_gray, enable_sky_clamp=True):
    """
    Process one depth map using unified P0 + Sky Clamping pipeline.
    """
    H, W = depth_raw.shape
    valid = depth_raw > 1.0
    if valid.sum() < 32:
        d_min, d_max = d_rel.min(), d_rel.max()
        d_norm = (d_rel - d_min) / (d_max - d_min + 1e-8)
        dav2_abs = 20000.0 - d_norm * 15000.0
        return np.clip(np.round(dav2_abs), 500, 65535).astype(np.uint16)

    # 1. Disparity-Space Polynomial Fit (1/Z = a*d + b)
    Z_m = depth_raw[valid] / 1000.0
    disp_true = 1.0 / np.clip(Z_m, 0.5, 80.0)
    rv = d_rel[valid]

    d_min, d_max = rv.min(), rv.max()
    d_norm = (rv - d_min) / (d_max - d_min + 1e-8)
    d_rel_full_norm = (d_rel - d_min) / (d_max - d_min + 1e-8)

    try:
        A = np.polyfit(d_norm, disp_true, 1)
        resid = disp_true - (A[0] * d_norm + A[1])
        s = np.std(resid)
        if s > 0:
            m = np.abs(resid) < 3 * s
            if m.sum() > 32:
                A = np.polyfit(d_norm[m], disp_true[m], 1)
        a_disp, b_disp = A[0], A[1]
    except Exception:
        a_disp, b_disp = 1.0, 0.1

    disp_pred = np.clip(a_disp * d_rel_full_norm + b_disp, 0.01, 2.0)
    dav2_abs = np.clip((1.0 / disp_pred) * 1000.0, 500.0, 65535.0)
    base_fill = np.where(valid, depth_raw, dav2_abs)

    # 2. Bi-Modal Structural Guide (RGB + IR max gradient)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ir_enh = clahe.apply(np.clip(ir_gray, 0, 255).astype(np.uint8)).astype(np.float32) / 255.0
    rgb_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    gx_rgb = cv2.Sobel(rgb_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_rgb = cv2.Sobel(rgb_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_rgb = np.sqrt(gx_rgb**2 + gy_rgb**2)

    gx_ir = cv2.Sobel(ir_enh, cv2.CV_32F, 1, 0, ksize=3)
    gy_ir = cv2.Sobel(ir_enh, cv2.CV_32F, 0, 1, ksize=3)
    mag_ir = np.sqrt(gx_ir**2 + gy_ir**2)

    guide_f = np.where(mag_rgb > mag_ir, rgb_gray, ir_enh).astype(np.float32)
    gf = guided_filter_core(guide_f, base_fill.astype(np.float32), r=4, eps=1e-3)
    refined = np.where(valid, depth_raw, gf)

    # 3. Sky & Infinity Clamping
    if enable_sky_clamp:
        upper_mask = np.zeros((H, W), dtype=bool)
        upper_mask[:int(H * 0.40), :] = True
        ir_blur = cv2.GaussianBlur(ir_gray.astype(np.float32), (15, 15), 0)
        sky_mask = upper_mask & (~valid) & (ir_blur < np.percentile(ir_blur, 55))
        refined[sky_mask] = 65535.0

    # Ensure output is clipped to 16-bit uint range
    depth_16u = np.clip(np.round(refined), 0, 65535).astype(np.uint16)
    return depth_16u

def main():
    parser = argparse.ArgumentParser(description="Unified P0 Depth Preprocessing Pipeline")
    parser.add_argument("--src_dir", default="dataset/train", help="Source dataset split directory")
    parser.add_argument("--out_dir", default="dataset/train/depth_p0", help="Destination depth_p0 directory")
    parser.add_argument("--limit", type=int, default=10, help="Max images to process (default 10 for validation, 0 for all)")
    parser.add_argument("--stride", type=int, default=1, help="Stride for parallel workers")
    parser.add_argument("--offset", type=int, default=0, help="Offset for parallel workers")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DAv2 model on {device} (worker offset={args.offset}, stride={args.stride})...")
    model = load_da_model(device)

    depth_files = sorted(glob.glob(os.path.join(args.src_dir, "depth/*.*")))
    if args.limit > 0:
        depth_files = depth_files[:args.limit]
    depth_files = depth_files[args.offset::args.stride]

    print(f"Processing {len(depth_files)} depth images to {args.out_dir}...")
    t0 = time.time()

    for idx, df in enumerate(depth_files):
        stem = os.path.splitext(os.path.basename(df))[0]
        out_file = os.path.join(args.out_dir, f"{stem}.png")
        if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
            continue

        v_cand = [os.path.join(args.src_dir, "visible", f"{stem}.png"), os.path.join(args.src_dir, "visible", f"{stem}.jpg")]
        i_cand = [os.path.join(args.src_dir, "infrared", f"{stem}.png"), os.path.join(args.src_dir, "infrared", f"{stem}.jpg")]
        vp = next((p for p in v_cand if os.path.exists(p)), None)
        ip = next((p for p in i_cand if os.path.exists(p)), None)
        if not vp or not ip: continue

        vis = cv2.imread(vp)
        ir = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        if ir.ndim == 3: ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        depth_raw = cv2.imread(df, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth_raw.ndim == 3: depth_raw = depth_raw[..., 0]
        # Handle 8-bit vs 16-bit depth source: 8-bit is normalized [0, 255], scale to mm [0, 19999]
        if df.endswith(('.jpg', '.jpeg')) or depth_raw.max() <= 255.0:
            depth_raw = depth_raw * (19999.0 / 255.0)
        H, W = depth_raw.shape

        # DA inference
        I = ir.astype(np.float32) / (ir.max() + 1e-8)
        ir8 = np.clip(I * 255, 0, 255).astype(np.uint8)
        inp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
        with torch.no_grad():
            d_rel = model.infer_image(inp, 518)
        d_rel = cv2.resize(d_rel.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

        # Unified P0 + Sky Clamp
        d_out = process_single_depth(depth_raw, d_rel, vis, ir, enable_sky_clamp=True)

        out_file = os.path.join(args.out_dir, f"{stem}.png")
        cv2.imwrite(out_file, d_out)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(depth_files):
            dt = time.time() - t0
            fps = (idx + 1) / dt
            print(f"[{idx+1}/{len(depth_files)}] Saved {out_file} ({fps:.1f} img/s)", flush=True)

    print(f"Baking completed in {time.time()-t0:.2f}s.", flush=True)

if __name__ == "__main__":
    main()
