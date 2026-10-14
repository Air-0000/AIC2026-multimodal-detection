#!/usr/bin/env python3
"""
Test CSSP (Cascaded Spectral-Spatial Perception) and ABC (Local Contrast / Attention)
on real AIC2026 test images to verify physical target enhancement and false-positive suppression.

Papers evaluated:
1. ABC (ICME 2023): Attention with Bilinear Correlation for Infrared Small Target Detection
   - Mechanism: Preserves shallow local contrast, prevents small targets from drowning in deep self-attention.
2. MODA / OSSDet (AAAI): Cascaded Spectral-Spatial Perception (CSSP) & Saliency Masking (CSFR)
   - Mechanism: Cross-spectral difference |F_rgb - F_ir| combined with spatial saliency gating.
"""

import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
RAW_PT = os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/raw_detections_v29_c5_peak.pt")
TEST_RGB_DIR = os.path.join(ROOT, "dataset/test/AIC2026_PHASE_2_1000/visible")
TEST_IR_DIR = os.path.join(ROOT, "work_dir/test_aligned_ir")
OUT_DIR = os.path.join(ROOT, "outputs/analysis/cssp_abc_evaluation")
FIG_DIR = os.path.join(ROOT, "outputs/figures/cssp_abc")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def compute_cssp_saliency(rgb_img, ir_img):
    """
    CSSP: Cascaded Spectral-Spatial joint Perception
    1. Cross-spectral difference: |RGB_gray - IR|
    2. Spatial gradient / structure weighting
    """
    if len(rgb_img.shape) == 3:
        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    else:
        rgb_gray = rgb_img
        
    rgb_f = rgb_gray.astype(np.float32) / 255.0
    ir_f = ir_img.astype(np.float32) / 255.0
    
    # 1. Spectral difference (cross-modality discrepancy)
    spectral_diff = np.abs(rgb_f - ir_f)
    
    # 2. Spatial gradient of IR and RGB
    grad_x_ir = cv2.Sobel(ir_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y_ir = cv2.Sobel(ir_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_ir = np.sqrt(grad_x_ir**2 + grad_y_ir**2)
    
    # 3. Cascaded Spatial-Spectral joint response
    # Small targets exhibit sharp localized spectral discrepancy and edge gradient
    saliency = spectral_diff * (1.0 + 2.0 * grad_mag_ir)
    
    # Local normalized enhancement
    saliency_norm = cv2.normalize(saliency, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return spectral_diff, saliency_norm

def compute_abc_local_contrast(ir_img, kernel_size=9):
    """
    ABC / RLCM (Relative Local Contrast Measure):
    Simulates ABC's shallow bilinear correlation by computing center-surround contrast.
    Center target T vs 8-direction background neighborhood.
    """
    ir_f = ir_img.astype(np.float32) / 255.0
    pad = kernel_size // 2
    h, w = ir_f.shape
    
    # Use morphological operations or box filters to compute local mean
    mean_filter = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    local_mean = cv2.filter2D(ir_f, -1, mean_filter)
    
    center_filter = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    c = kernel_size // 2
    center_filter[c-1:c+2, c-1:c+2] = 1.0 / 9.0
    center_mean = cv2.filter2D(ir_f, -1, center_filter)
    
    # Surrounding background = total mean - center mean
    surround_mean = (local_mean * (kernel_size*kernel_size) - center_mean * 9) / (kernel_size*kernel_size - 9)
    surround_mean = np.maximum(surround_mean, 1e-4)
    
    # Relative Local Contrast: (T - B) * (T / B)
    diff = center_mean - surround_mean
    lcm = np.maximum(diff, 0) * (center_mean / surround_mean)
    lcm_norm = cv2.normalize(lcm, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return lcm_norm

def extract_patch_metrics(rgb_img, ir_img, cx, cy, w, h):
    """
    Extract target patch and compute quantitative CSSP and ABC scores.
    """
    img_h, img_w = ir_img.shape
    x1 = int(max(0, (cx - w/2) * img_w))
    y1 = int(max(0, (cy - h/2) * img_h))
    x2 = int(min(img_w, (cx + w/2) * img_w))
    y2 = int(min(img_h, (cy + h/2) * img_h))
    
    # Expand margin for local background context (2x box)
    margin_w = max(10, int(w * img_w * 0.8))
    margin_h = max(10, int(h * img_h * 0.8))
    
    ctx_x1 = max(0, x1 - margin_w)
    ctx_y1 = max(0, y1 - margin_h)
    ctx_x2 = min(img_w, x2 + margin_w)
    ctx_y2 = min(img_h, y2 + margin_h)
    
    rgb_crop = rgb_img[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
    ir_crop = ir_img[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
    
    if rgb_crop.size == 0 or ir_crop.size == 0:
        return None
        
    spec_diff, cssp_sal = compute_cssp_saliency(rgb_crop, ir_crop)
    abc_lcm = compute_abc_local_contrast(ir_crop)
    
    # Target relative coordinates inside context crop
    tx1 = x1 - ctx_x1
    ty1 = y1 - ctx_y1
    tx2 = x2 - ctx_x1
    ty2 = y2 - ctx_y1
    
    target_mask = np.zeros(ir_crop.shape, dtype=bool)
    target_mask[ty1:ty2, tx1:tx2] = True
    bg_mask = ~target_mask
    
    target_cssp = np.mean(cssp_sal[target_mask]) if np.any(target_mask) else 0.0
    bg_cssp = np.mean(cssp_sal[bg_mask]) if np.any(bg_mask) else 1e-4
    cssp_ratio = (target_cssp + 1e-4) / (bg_cssp + 1e-4)
    
    target_abc = np.mean(abc_lcm[target_mask]) if np.any(target_mask) else 0.0
    bg_abc = np.mean(abc_lcm[bg_mask]) if np.any(bg_mask) else 1e-4
    abc_ratio = (target_abc + 1e-4) / (bg_abc + 1e-4)
    
    return {
        "rgb_crop": rgb_crop,
        "ir_crop": ir_crop,
        "spec_diff": spec_diff,
        "cssp_sal": cssp_sal,
        "abc_lcm": abc_lcm,
        "target_box_rel": (tx1, ty1, tx2, ty2),
        "cssp_score": float(target_cssp),
        "cssp_ratio": float(cssp_ratio),
        "abc_score": float(target_abc),
        "abc_ratio": float(abc_ratio),
        "combined_saliency": float(cssp_ratio * 0.5 + abc_ratio * 0.5)
    }

def main():
    print("[*] Loading raw model detections...")
    raw = torch.load(RAW_PT, weights_only=False)
    
    # Ground truth / verified physical drones
    true_targets = [
        ("00000012", 0.5377, 0.4578, 0.0680, 0.0611, "Verified Drone (00000012)"),
        ("00000422", 0.2871, 0.3646, 0.0210, 0.0150, "Verified Drone (00000422)"),
        ("00000511", 0.3440, 0.4350, 0.0180, 0.0140, "Rescued Weak Drone (00000511)"),
        ("00000299", 0.4430, 0.3800, 0.0250, 0.0220, "Moving Drone (00000299)"),
        ("00000338", 0.6120, 0.3200, 0.0300, 0.0250, "Moving Drone (00000338)"),
    ]
    
    # Confirmed false-positive background clutter killed by Physics V2
    false_positives = [
        ("00000071", 0.595, 0.635, 0.025, 0.025, "Static Fixture (00000071)"),
        ("00000076", 0.595, 0.635, 0.025, 0.025, "Static Fixture (00000076)"),
        ("00000037", 0.843, 0.594, 0.028, 0.026, "Static Fixture (00000037)"),
        ("00000290", 0.610, 0.740, 0.035, 0.030, "Building Shadow (00000290)"),
        ("00000466", 0.320, 0.880, 0.040, 0.020, "Ground Curb (00000466)"),
        ("00000256", 0.570, 0.770, 0.025, 0.035, "Static Pole (00000256)"),
    ]
    
    results_true = []
    results_fp = []
    
    print("\n--- Evaluating True Physical Drones with CSSP & ABC ---")
    for stem, cx, cy, w, h, label in true_targets:
        rgb_path = os.path.join(TEST_RGB_DIR, f"{stem}.jpg")
        ir_path = os.path.join(TEST_IR_DIR, f"{stem}.jpg")
        if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
            continue
        rgb = cv2.imread(rgb_path)
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        
        m = extract_patch_metrics(rgb, ir, cx, cy, w, h)
        if m:
            m["stem"] = stem
            m["label"] = label
            results_true.append(m)
            print(f"[TRUE DRONE] {label}: CSSP Ratio={m['cssp_ratio']:.2f}, ABC Ratio={m['abc_ratio']:.2f}, Combined={m['combined_saliency']:.2f}")

    print("\n--- Evaluating False-Positive Background Clutter with CSSP & ABC ---")
    for stem, cx, cy, w, h, label in false_positives:
        rgb_path = os.path.join(TEST_RGB_DIR, f"{stem}.jpg")
        ir_path = os.path.join(TEST_IR_DIR, f"{stem}.jpg")
        if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
            continue
        rgb = cv2.imread(rgb_path)
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        
        m = extract_patch_metrics(rgb, ir, cx, cy, w, h)
        if m:
            m["stem"] = stem
            m["label"] = label
            results_fp.append(m)
            print(f"[FALSE POS] {label}: CSSP Ratio={m['cssp_ratio']:.2f}, ABC Ratio={m['abc_ratio']:.2f}, Combined={m['combined_saliency']:.2f}")

    # Generate visual comparison panels
    print("\n[*] Generating high-resolution visual comparison panels...")
    selected_cases = [results_true[0], results_true[1], results_true[2], results_fp[0], results_fp[2], results_fp[3]]
    
    fig, axes = plt.subplots(len(selected_cases), 5, figsize=(20, 4 * len(selected_cases)))
    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    
    col_titles = [
        "Visible (RGB) Crop",
        "Aligned Thermal (IR)",
        "CSSP Cross-Spectral Diff |RGB-IR|",
        "CSSP Saliency Map",
        "ABC Local Contrast (LCM)"
    ]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=12, fontweight='bold', pad=10)
        
    for r, case in enumerate(selected_cases):
        tx1, ty1, tx2, ty2 = case["target_box_rel"]
        
        # 1. RGB
        rgb_disp = cv2.cvtColor(case["rgb_crop"], cv2.COLOR_BGR2RGB).copy()
        cv2.rectangle(rgb_disp, (tx1, ty1), (tx2, ty2), (0, 255, 0) if "Drone" in case["label"] else (255, 0, 0), 2)
        axes[r, 0].imshow(rgb_disp)
        axes[r, 0].set_ylabel(case["label"], fontsize=11, fontweight='bold')
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        
        # 2. IR
        axes[r, 1].imshow(case["ir_crop"], cmap='inferno')
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
        
        # 3. Spectral Diff
        axes[r, 2].imshow(case["spec_diff"], cmap='magma')
        axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])
        
        # 4. CSSP Saliency Map
        axes[r, 3].imshow(case["cssp_sal"], cmap='turbo')
        axes[r, 3].text(5, 15, f"CSSP: {case['cssp_ratio']:.2f}x", color='white', backgroundcolor='black', fontsize=10, fontweight='bold')
        axes[r, 3].set_xticks([]); axes[r, 3].set_yticks([])
        
        # 5. ABC LCM
        axes[r, 4].imshow(case["abc_lcm"], cmap='viridis')
        axes[r, 4].text(5, 15, f"ABC: {case['abc_ratio']:.2f}x", color='white', backgroundcolor='black', fontsize=10, fontweight='bold')
        axes[r, 4].set_xticks([]); axes[r, 4].set_yticks([])
        
    out_panel_fig = os.path.join(FIG_DIR, "cssp_abc_visual_comparison_panel.jpg")
    plt.savefig(out_panel_fig, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[+] Visual comparison panel saved: {out_panel_fig}")

    # Generate statistical comparison chart
    plt.figure(figsize=(10, 5))
    true_cssp = [x["cssp_ratio"] for x in results_true]
    fp_cssp = [x["cssp_ratio"] for x in results_fp]
    true_abc = [x["abc_ratio"] for x in results_true]
    fp_abc = [x["abc_ratio"] for x in results_fp]
    
    x_indices = np.arange(2)
    width = 0.35
    
    true_means = [np.mean(true_cssp), np.mean(true_abc)]
    true_stds = [np.std(true_cssp), np.std(true_abc)]
    fp_means = [np.mean(fp_cssp), np.mean(fp_abc)]
    fp_stds = [np.std(fp_cssp), np.std(fp_abc)]
    
    plt.bar(x_indices - width/2, true_means, width, yerr=true_stds, label='True Physical Drones (Signal Peak)', color='#2ca02c', capsize=5)
    plt.bar(x_indices + width/2, fp_means, width, yerr=fp_stds, label='False Background Clutter (Suppressed)', color='#d62728', capsize=5)
    
    plt.ylabel('Target-to-Background Contrast Ratio (x)', fontsize=12, fontweight='bold')
    plt.title('Separation Power: CSSP Spectral Saliency & ABC Local Contrast Measure', fontsize=13, fontweight='bold')
    plt.xticks(x_indices, ['CSSP (Cross-Spectral Difference)', 'ABC (Local Contrast Attention)'], fontsize=11, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_stat_fig = os.path.join(FIG_DIR, "cssp_abc_separation_statistics.png")
    plt.savefig(out_stat_fig, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[+] Separation statistics chart saved: {out_stat_fig}")
    
    print("\n=== Quantitative Summary ===")
    print(f"True Drones Mean CSSP Ratio: {np.mean(true_cssp):.2f}x (FP: {np.mean(fp_cssp):.2f}x) -> Separation Gap: {np.mean(true_cssp)/np.mean(fp_cssp):.2f}x")
    print(f"True Drones Mean ABC Ratio:  {np.mean(true_abc):.2f}x (FP: {np.mean(fp_abc):.2f}x) -> Separation Gap: {np.mean(true_abc)/np.mean(fp_abc):.2f}x")

if __name__ == "__main__":
    main()
