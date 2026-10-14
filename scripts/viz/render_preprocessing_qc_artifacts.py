#!/usr/bin/env python3
"""
Generate Preprocessing QC Comparison Artifacts:
1. Visible (RGB)
2. Raw Infrared vs Preprocessed Aligned Infrared (with False Color RGB-IR Alignment Overlay)
3. Raw Depth (8-bit sensor) vs Preprocessed P0 Metric Depth (16-bit guided millimeter depth)
"""

import os
import cv2
import numpy as np
from PIL import Image

OUT_DIR = "/home/constant/.gemini/antigravity/brain/46ad3bec-cfa4-4605-895d-0a4627a337ed"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = ["00000010", "00000018", "00000062"]

for stem in SAMPLES:
    vis_p = f"work_dir/audits/vis_qc_samples/test_visible/{stem}.jpg"
    raw_ir_p = f"work_dir/audits/vis_qc_samples/test_infrared/{stem}.jpg"
    aln_ir_p = f"work_dir/audits/vis_qc_samples/test_aligned_ir/{stem}.jpg"
    raw_dep_p = f"dataset/test/AIC2026_PHASE_2_1000/depth/{stem}.jpg"
    p0_dep_p = f"work_dir/audits/vis_qc_samples/test_depth_p0/{stem}.png"

    # 1. Load Visible
    vis = cv2.imread(vis_p)
    H, W = vis.shape[:2]

    # 2. Load Raw IR & Aligned IR
    raw_ir = cv2.imread(raw_ir_p, cv2.IMREAD_GRAYSCALE)
    aln_ir = cv2.imread(aln_ir_p, cv2.IMREAD_GRAYSCALE)
    if raw_ir.shape != (H, W):
        raw_ir = cv2.resize(raw_ir, (W, H))
    if aln_ir.shape != (H, W):
        aln_ir = cv2.resize(aln_ir, (W, H))

    # Difference map between Raw IR and Aligned IR (scaled for visibility)
    diff_ir = cv2.absdiff(aln_ir, raw_ir)
    diff_ir_vis = cv2.applyColorMap(cv2.equalizeHist(diff_ir), cv2.COLORMAP_MAGMA)

    # False color alignment check: Red = Vis Gray, Green = Aligned IR, Blue = 0
    vis_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    overlay_aln = np.zeros((H, W, 3), dtype=np.uint8)
    overlay_aln[:, :, 2] = vis_gray  # Red channel = visible
    overlay_aln[:, :, 1] = aln_ir    # Green channel = aligned IR
    overlay_aln[:, :, 0] = 0         # Where perfectly aligned, yellow edge contours appear!

    # False color with Raw IR:
    overlay_raw = np.zeros((H, W, 3), dtype=np.uint8)
    overlay_raw[:, :, 2] = vis_gray
    overlay_raw[:, :, 1] = raw_ir
    overlay_raw[:, :, 0] = 0

    # 3. Load Raw Depth & P0 Depth
    raw_dep = cv2.imread(raw_dep_p, cv2.IMREAD_GRAYSCALE)
    if raw_dep.shape != (H, W):
        raw_dep = cv2.resize(raw_dep, (W, H))
    raw_dep_cm = cv2.applyColorMap(raw_dep, cv2.COLORMAP_TURBO)

    # 16-bit P0 Depth
    p0_dep_16 = cv2.imread(p0_dep_p, cv2.IMREAD_UNCHANGED)
    if p0_dep_16 is not None:
        p0_norm = (p0_dep_16.astype(np.float32) / 20000.0).clip(0.0, 1.0)
        p0_dep_8 = (p0_norm * 255.0).astype(np.uint8)
        p0_dep_cm = cv2.applyColorMap(p0_dep_8, cv2.COLORMAP_TURBO)
    else:
        p0_dep_cm = raw_dep_cm.copy()

    # Create 2x3 Grid Visualization
    # Row 1: [Visible RGB] | [Raw IR (Sensor)] | [Aligned IR (Preprocessed)]
    # Row 2: [RGB-IR Overlay (Before vs After)] | [Raw Depth (Sensor 8-bit)] | [P0 Metric Depth (Preprocessed 16-bit)]

    tw, th = 640, 360
    v_vis = cv2.resize(vis, (tw, th))
    v_raw_ir = cv2.cvtColor(cv2.resize(raw_ir, (tw, th)), cv2.COLOR_GRAY2BGR)
    v_aln_ir = cv2.cvtColor(cv2.resize(aln_ir, (tw, th)), cv2.COLOR_GRAY2BGR)
    
    # Overlay comparison: Left half Raw Overlay, Right half Aligned Overlay
    v_overlay_aln = cv2.resize(overlay_aln, (tw, th))
    v_raw_dep = cv2.resize(raw_dep_cm, (tw, th))
    v_p0_dep = cv2.resize(p0_dep_cm, (tw, th))

    # Add labels
    def add_label(img, text, subtext=""):
        res = img.copy()
        cv2.rectangle(res, (0, 0), (tw, 45), (20, 20, 20), -1)
        cv2.putText(res, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        if subtext:
            cv2.putText(res, subtext, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return res

    p1 = add_label(v_vis, f"Visible RGB ({stem})", "High-Resolution Optical Stream")
    p2 = add_label(v_raw_ir, "Raw Infrared (Sensor)", "Unaligned Sensor Geometry")
    p3 = add_label(v_aln_ir, "Preprocessed Aligned IR", "Sub-pixel Warped & Destriped")
    p4 = add_label(v_overlay_aln, "RGB (Red) + Aligned IR (Green)", "Yellow = True Optical Phase Co-registration")
    p5 = add_label(v_raw_dep, "Raw Depth (Sensor)", "8-bit Lossy Compressed, Sparse & Noisy")
    p6 = add_label(v_p0_dep, "Preprocessed P0 Depth (Ours)", "16-bit Millimeter Guided Foundation Prior")

    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    canvas = np.vstack([row1, row2])

    out_p = os.path.join(OUT_DIR, f"qc_preprocess_{stem}.jpg")
    cv2.imwrite(out_p, canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"[+] Saved QC artifact: {out_p}")
