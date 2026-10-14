#!/usr/bin/env python3
"""
Bake Test Set Channel 4 for All-Inclusive Orthogonal Split (V18)
================================================================
Combines:
  1. Multi-scale MCA (from work_dir/test_lbg_mca)
  2. 3D Volumetric Curvature Gate Phi_VT (from depth_p0)
  3. 3D Disparity Geometric Fracture Step Edges (from depth_p0, alpha=0.30)
Outputs to:
  work_dir/test_lbg_orthogonal/ (1,000 files)
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

aic_root = '/root/autodl-tmp/AIC' if os.path.exists('/root/autodl-tmp/AIC') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

MCA_DIR = os.path.join(aic_root, 'work_dir/test_lbg_mca')
DEP_DIR = os.path.join(aic_root, 'dataset/test/AIC2026_PHASE_2_1000/depth_p0')
OUT_DIR = os.path.join(aic_root, 'work_dir/test_lbg_orthogonal')

def compute_phi_vt_and_3d_edges(depth_arr):
    depth_m = np.clip(depth_arr / 1000.0 if depth_arr.max() > 255.0 else depth_arr / 20.0, 0.5, 50.0)

    # 1. 3D surface normal & mean curvature divergence (Phi_VT)
    dz_dx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    div_n = np.abs(cv2.Sobel(-dz_dx / norm, cv2.CV_32F, 1, 0) + cv2.Sobel(-dz_dy / norm, cv2.CV_32F, 0, 1))

    flat_score = np.exp(- (div_n / 0.15)**2)
    phi_vt_gate = (1.0 - 0.75 * flat_score).astype(np.float32)

    # 2. Disparity-Domain 3D Geometric Fracture
    disp = 1.0 / np.clip(depth_m, 0.5, 50.0)
    gx_d = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    edge_3d = np.clip(np.sqrt(gx_d**2 + gy_d**2) * 4.0, 0.0, 1.0).astype(np.float32)

    return phi_vt_gate, edge_3d

def process_stem(stem):
    mca_p = os.path.join(MCA_DIR, f"{stem}.jpg") if os.path.exists(os.path.join(MCA_DIR, f"{stem}.jpg")) else os.path.join(MCA_DIR, f"{stem}.png")
    dep_p = os.path.join(DEP_DIR, f"{stem}.png")
    out_p = os.path.join(OUT_DIR, f"{stem}.png")

    mca_im = cv2.imread(mca_p, cv2.IMREAD_GRAYSCALE)
    dep_im = Image.open(dep_p)
    dep_arr = np.array(dep_im, dtype=np.float32)

    phi_vt, edge_3d = compute_phi_vt_and_3d_edges(dep_arr)
    ch4_np = np.clip(mca_im.astype(np.float32) * phi_vt + 0.30 * 255.0 * edge_3d, 0, 255).astype(np.uint8)

    cv2.imwrite(out_p, ch4_np)
    return stem

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mca_files = sorted([os.path.splitext(f)[0] for f in os.listdir(MCA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(mca_files)} files to bake in {MCA_DIR}...")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(process_stem, mca_files))

    dt = time.time() - t0
    print(f"Successfully baked {len(results)} orthogonal Ch4 files into {OUT_DIR} in {dt:.2f}s!")

if __name__ == '__main__':
    main()
