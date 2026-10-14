#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_round3_visual_panels.py
生成第三轮全新范式(频域交叉谱、仿生视网膜、3D平面突起残差、多模态梯度余弦相干)的四联/五联对比诊断图
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
OUT_DIR = os.path.join(ROOT, "work_dir/audits/audit_round3_techniques")
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
    
    # 1. FCSPM (频域傅里叶交叉谱合成)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    f_ir = np.fft.fftshift(np.fft.fft2(ir.astype(np.float32)))
    f_l = np.fft.fftshift(np.fft.fft2(l_chan.astype(np.float32)))
    mag_ir = np.abs(f_ir)
    phase_ir = np.angle(f_ir)
    phase_l = np.angle(f_l)
    
    H, W = ir.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    radius = np.sqrt((x - cx)**2 + (y - cy)**2)
    cutoff = min(H, W) * 0.05
    w_p = np.clip((radius - cutoff) / (min(H, W) * 0.25 + 1e-4), 0.0, 1.0) * 0.45
    sin_p = (1.0 - w_p) * np.sin(phase_ir) + w_p * np.sin(phase_l)
    cos_p = (1.0 - w_p) * np.cos(phase_ir) + w_p * np.cos(phase_l)
    f_synth = mag_ir * np.exp(1j * np.arctan2(sin_p, cos_p))
    ir_fcspm = np.clip(np.real(np.fft.ifft2(np.fft.ifftshift(f_synth))), 0, 255).astype(np.uint8)
    
    # 2. Retinal DoG
    g_center = cv2.GaussianBlur(ir.astype(np.float32), (5, 5), 1.2)
    g_surround = cv2.GaussianBlur(ir.astype(np.float32), (25, 25), 6.0)
    dog = np.clip(g_center - 0.95 * g_surround, 0, 255)
    local_var = cv2.GaussianBlur(dog**2, (31, 31), 8.0)
    retinal_dog = np.clip(dog / (np.sqrt(local_var + 16.0) * 3.0), 0.0, 1.0)
    
    # 3. 3D PPR
    d_smooth = cv2.bilateralFilter(depth_m, d=15, sigmaColor=1.5, sigmaSpace=25)
    dz_dx = cv2.Sobel(d_smooth, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(d_smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag_grad = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    residual_3d = np.abs(depth_m - d_smooth) / mag_grad
    valid = (depth_m > 0.1) & (depth_m < 19.5)
    residual_3d[~valid] = 0.0
    ppr_map = np.clip((residual_3d - 0.15) / 0.85, 0.0, 1.0)
    
    # 4. LGOC
    gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx_rgb = cv2.Sobel(gray_rgb, cv2.CV_32F, 1, 0, ksize=3)
    gy_rgb = cv2.Sobel(gray_rgb, cv2.CV_32F, 0, 1, ksize=3)
    mag_rgb = np.sqrt(gx_rgb**2 + gy_rgb**2)
    gx_ir = cv2.Sobel(ir.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy_ir = cv2.Sobel(ir.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag_ir_sobel = np.sqrt(gx_ir**2 + gy_ir**2)
    dot = gx_rgb * gx_ir + gy_rgb * gy_ir
    cos_sim = np.abs(dot) / (mag_rgb * mag_ir_sobel + 1e-4)
    lgoc_map = cos_sim * np.clip(np.sqrt(mag_rgb * mag_ir_sobel) / 40.0, 0.0, 1.0)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"Original RGB ({fname})", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ir, cmap='inferno')
    axes[0, 1].set_title("Raw Infrared (Thermal Radiant Field)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(ir_fcspm, cmap='inferno')
    axes[0, 2].set_title("Dir-1: Fourier Cross-Spectral Phase Lock (FCSPM)\n[Sub-pixel Optical Contours + Thermal Energy]", fontsize=11, fontweight='bold', color='darkblue')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(retinal_dog, cmap='magma')
    axes[1, 0].set_title("Dir-2: Retinal Ganglion DoG Lateral Inhibition\n[Zeros Ambient Heat Dome, Fires on Targets]", fontsize=11, fontweight='bold', color='darkgreen')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ppr_map, cmap='viridis')
    axes[1, 1].set_title("Dir-3: 3D Planar Protrusion Residual (PPR)\n[Isolates >15cm True Physical Obstacles]", fontsize=11, fontweight='bold', color='purple')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(lgoc_map, cmap='hot')
    axes[1, 2].set_title("Dir-4: Multi-Modal Gradient Cosine Consistency (LGOC)\n[Kills Optical Reflections & Shadows]", fontsize=11, fontweight='bold', color='crimson')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"round3_panel_{base_name}.png")
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    sample_files = ["00000095.jpg", "00000106.jpg", "00000123.jpg"]
    for s in sample_files:
        if os.path.exists(os.path.join(IMG_DIR, s)):
            process_sample(s)
