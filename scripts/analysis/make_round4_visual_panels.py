#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_round4_visual_panels.py
生成第四轮前沿范式(热力学生热反演THGI、3D地形高程HAG-DEM、3D接触阴影SSAO、结构张量角点STCE)的诊断图
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
IMG_DIR = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/images")
IR_DIR = os.path.join(ROOT, "dataset/train/infrared")
DEPTH_DIR = os.path.join(ROOT, "dataset/train/depth")
OUT_DIR = os.path.join(ROOT, "work_dir/audits/audit_round4_techniques")
os.makedirs(OUT_DIR, exist_ok=True)

def read_depth_m(base_name):
    p_png = os.path.join(DEPTH_DIR, base_name + ".png")
    p_jpg = os.path.join(DEPTH_DIR, base_name + ".jpg")
    p = p_png if os.path.exists(p_png) else p_jpg
    if os.path.exists(p):
        im = Image.open(p)
        arr = np.array(im).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        scale = 1000.0 if getattr(im, 'mode', '') in ('I;16', 'I') else (255.0 / 20.0 * 1000.0)
        return arr / scale
    return None

def process_sample(fname):
    base_name = os.path.splitext(fname)[0]
    rgb_path = os.path.join(IMG_DIR, fname)
    ir_path = os.path.join(IR_DIR, fname)
    depth_m = read_depth_m(base_name)
    
    if not (os.path.exists(rgb_path) and os.path.exists(ir_path) and depth_m is not None):
        return
        
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))
    depth_m = cv2.resize(depth_m, (rgb.shape[1], rgb.shape[0]))
    H, W = ir.shape
    
    # 1. THGI
    ir_smooth = cv2.GaussianBlur(ir.astype(np.float32), (5, 5), 1.2)
    lap = cv2.Laplacian(ir_smooth, cv2.CV_32F, ksize=3)
    active_heat_gen = np.clip(-lap, 0, 255)
    gx = cv2.Sobel(ir_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ir_smooth, cv2.CV_32F, 0, 1, ksize=3)
    heat_flux = np.sqrt(gx**2 + gy**2)
    thgi_map = np.clip((active_heat_gen * (1.0 + heat_flux / 15.0)) / 40.0, 0.0, 1.0)
    
    # 2. HAG-DEM
    valid = (depth_m > 0.5) & (depth_m < 18.0)
    fy = 1.07 * H
    cy = H / 2.0
    v_indices = np.arange(H)[:, None].repeat(W, axis=1).astype(np.float32)
    Y_m = (v_indices - cy) * depth_m / fy
    ground_band = valid.copy()
    ground_band[:int(H * 0.65), :] = False
    if ground_band.sum() > 50:
        slope, intercept = np.polyfit(depth_m[ground_band], Y_m[ground_band], deg=1)
        y_ref = slope * depth_m + intercept
        hag = -(Y_m - y_ref)
        hag[~valid] = 0.0
    else:
        hag = np.zeros_like(depth_m)
    hag_map = np.clip((hag - 0.15) / 1.5, 0.0, 1.0)
    
    # 3. Depth SSAO
    k_h = np.array([[-1, 2, -1]], dtype=np.float32)
    k_v = np.array([[-1], [2], [-1]], dtype=np.float32)
    concave_h = cv2.filter2D(depth_m, cv2.CV_32F, k_h)
    concave_v = cv2.filter2D(depth_m, cv2.CV_32F, k_v)
    ao_cavity = np.clip(np.abs(concave_h) + np.abs(concave_v), 0, 2.0)
    ao_cavity[~valid] = 0.0
    ssao_map = np.clip(ao_cavity / 0.8, 0.0, 1.0)
    
    # 4. STCE
    gx_ir = cv2.Sobel(ir.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy_ir = cv2.Sobel(ir.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    j_xx = cv2.GaussianBlur(gx_ir**2, (7, 7), 1.5)
    j_yy = cv2.GaussianBlur(gy_ir**2, (7, 7), 1.5)
    j_xy = cv2.GaussianBlur(gx_ir * gy_ir, (7, 7), 1.5)
    det_j = j_xx * j_yy - j_xy**2
    trace_j = j_xx + j_yy
    corner_energy = np.clip(det_j - 0.04 * (trace_j**2), 0, None)
    stce_map = np.clip(corner_energy / (trace_j + 1e-4) / 25.0, 0.0, 1.0)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"Original RGB ({fname})", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ir, cmap='inferno')
    axes[0, 1].set_title("Raw Infrared", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(thgi_map, cmap='hot')
    axes[0, 2].set_title("Method 1: Thermodynamic Heat Gen Inversion (THGI)\n[-∇²T Internal Energy Rate, Kills Passive Sun Heat]", fontsize=11, fontweight='bold', color='crimson')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(hag_map, cmap='jet')
    axes[1, 0].set_title("Method 2: 3D Height-Above-Ground DEM (HAG)\n[Ground Manifold Normalized Metric Clearance]", fontsize=11, fontweight='bold', color='darkblue')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ssao_map, cmap='magma')
    axes[1, 1].set_title("Method 3: Depth SSAO Contact Crevices\n[Ground-Anchor Cavity Occlusion]", fontsize=11, fontweight='bold', color='purple')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(stce_map, cmap='viridis')
    axes[1, 2].set_title("Method 4: Structure Tensor Corner Energy (STCE)\n[Suppresses 1D Lines, Boosts 2D Bounded Corners]", fontsize=11, fontweight='bold', color='darkgreen')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"round4_panel_{base_name}.png")
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    for s in ["00000095.jpg", "00000106.jpg", "00000123.jpg"]:
        if os.path.exists(os.path.join(IMG_DIR, s)):
            process_sample(s)
