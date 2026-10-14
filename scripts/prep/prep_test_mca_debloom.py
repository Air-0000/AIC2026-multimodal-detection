#!/usr/bin/env python3
"""
Pre-bake Method IR-1 (MCA) and Method IR-3 (De-blooming) for Test Set
====================================================================
Fast 32-process parallel baking for 1,000 test images:
  - Input:  work_dir/test_aligned_ir, dataset/test/AIC2026_PHASE_2_1000/visible
  - Output: work_dir/test_lbg_mca, work_dir/test_aligned_debloom
"""

import os
import sys
import glob
import time
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image


def ir_method1_multiscale_mca(ir_u8, scales=(5, 11, 21, 41, 81)):
    """Method IR-1: 多尺度形态学分量分析 (Multi-Scale Morphological Top-Hat / MCA)"""
    out = np.zeros_like(ir_u8, dtype=np.float32)
    total_w = 0.0
    for k in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        wth = cv2.morphologyEx(ir_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        bth = cv2.morphologyEx(ir_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        w = 1.0 / np.sqrt(k)
        contrast_k = wth / (1.0 + 0.1 * bth)
        out += w * contrast_k
        total_w += w
    out = (out / total_w) * 1.6
    return np.clip(out, 0, 255).astype(np.uint8)


def ir_method3_anisotropic_thermal_deblooming(ir_u8, vis_bgr, n_iters=3, dt=0.20):
    """Method IR-3: 结构张量各向异性热晕逆收拢 (Anisotropic Thermal De-blooming)"""
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx_v = cv2.Sobel(vg, cv2.CV_32F, 1, 0, ksize=3)
    gy_v = cv2.Sobel(vg, cv2.CV_32F, 0, 1, ksize=3)
    mag_v = np.sqrt(gx_v**2 + gy_v**2)
    barrier = np.exp(- (mag_v / 0.12)**2)

    u = ir_u8.astype(np.float32)
    for _ in range(n_iters):
        gx_u = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
        gy_u = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
        mag_u = np.sqrt(gx_u**2 + gy_u**2)
        lap_u = cv2.Laplacian(u, cv2.CV_32F, ksize=3)
        shock = - np.sign(lap_u) * mag_u * barrier
        u = u + dt * shock
    return np.clip(u, 0, 255).astype(np.uint8)


def _process_single(task):
    stem, ir_p, vis_p, out_mca_p, out_deb_p = task
    try:
        ir_u8 = cv2.imread(ir_p, cv2.IMREAD_GRAYSCALE)
        vis_bgr = cv2.imread(vis_p)
        H, W = vis_bgr.shape[:2]
        if ir_u8.shape[:2] != (H, W):
            ir_u8 = cv2.resize(ir_u8, (W, H), interpolation=cv2.INTER_LINEAR)

        # 1. Debloom
        ir_deb = ir_method3_anisotropic_thermal_deblooming(ir_u8, vis_bgr)
        cv2.imwrite(out_deb_p, ir_deb)

        # 2. MCA from debloomed (or raw aligned)
        ir_mca = ir_method1_multiscale_mca(ir_deb)
        cv2.imwrite(out_mca_p, ir_mca)
        return True
    except Exception as e:
        print(f"Error {stem}: {e}")
        return False


def main():
    root = "/root/autodl-tmp/AIC"
    ir_dir = os.path.join(root, "work_dir/test_aligned_ir")
    vis_dir = os.path.join(root, "dataset/test/AIC2026_PHASE_2_1000/visible")
    out_mca_dir = os.path.join(root, "work_dir/test_lbg_mca")
    out_deb_dir = os.path.join(root, "work_dir/test_aligned_debloom")

    os.makedirs(out_mca_dir, exist_ok=True)
    os.makedirs(out_deb_dir, exist_ok=True)

    vis_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
    tasks = []
    for vf in vis_files:
        stem = os.path.splitext(os.path.basename(vf))[0]
        ir_f = os.path.join(ir_dir, f"{stem}.jpg")
        if not os.path.exists(ir_f):
            ir_f = os.path.join(ir_dir, f"{stem}.png")
        out_mca_f = os.path.join(out_mca_dir, f"{stem}.jpg")
        out_deb_f = os.path.join(out_deb_dir, f"{stem}.jpg")
        tasks.append((stem, ir_f, vf, out_mca_f, out_deb_f))

    print(f"Baking MCA + De-blooming for {len(tasks)} images with 32 workers...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(_process_single, tasks))

    dt = time.time() - t0
    success = sum(1 for r in results if r)
    print(f"Completed: {success}/{len(tasks)} in {dt:.1f}s ({len(tasks)/dt:.1f} FPS)")


if __name__ == '__main__':
    main()
