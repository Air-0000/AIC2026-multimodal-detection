#!/usr/bin/env python3
"""
High-Resolution Zoomed-in Crop Comparison:
Demonstrates pixel-level alignment and metric depth reconstruction on key targets.
"""

import os
import cv2
import numpy as np

OUT_DIR = "/home/constant/.gemini/antigravity/brain/46ad3bec-cfa4-4605-895d-0a4627a337ed"

# Sample 1: 00000018 (Car / Street Scene with strong parallax)
# Sample 2: 00000010 (Streetlight / UAV Scene)
for stem, (crop_y1, crop_y2, crop_x1, crop_x2), desc in [
    ("00000018", (120, 280, 180, 420), "Vehicle & Road Geometry"),
    ("00000010", (80, 240, 220, 460), "Streetlight & Infrastructure")
]:
    vis_p = f"work_dir/audits/vis_qc_samples/test_visible/{stem}.jpg"
    raw_ir_p = f"work_dir/audits/vis_qc_samples/test_infrared/{stem}.jpg"
    aln_ir_p = f"work_dir/audits/vis_qc_samples/test_aligned_ir/{stem}.jpg"
    raw_dep_p = f"dataset/test/AIC2026_PHASE_2_1000/depth/{stem}.jpg"
    p0_dep_p = f"work_dir/audits/vis_qc_samples/test_depth_p0/{stem}.png"

    vis = cv2.imread(vis_p)
    raw_ir = cv2.imread(raw_ir_p, cv2.IMREAD_GRAYSCALE)
    aln_ir = cv2.imread(aln_ir_p, cv2.IMREAD_GRAYSCALE)
    raw_dep = cv2.imread(raw_dep_p, cv2.IMREAD_GRAYSCALE)
    p0_dep_16 = cv2.imread(p0_dep_p, cv2.IMREAD_UNCHANGED)

    H, W = vis.shape[:2]
    raw_ir = cv2.resize(raw_ir, (W, H))
    aln_ir = cv2.resize(aln_ir, (W, H))
    raw_dep = cv2.resize(raw_dep, (W, H))

    raw_dep_cm = cv2.applyColorMap(raw_dep, cv2.COLORMAP_TURBO)
    p0_norm = (p0_dep_16.astype(np.float32) / 20000.0).clip(0.0, 1.0)
    p0_dep_cm = cv2.applyColorMap((p0_norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)

    # Edge overlay on Visible:
    # Canny edges of Aligned IR overlaid on Visible
    edges_aln = cv2.Canny(aln_ir, 50, 150)
    vis_edge_aln = vis.copy()
    vis_edge_aln[edges_aln > 0] = [0, 255, 0] # Green contours = Aligned IR

    edges_raw = cv2.Canny(raw_ir, 50, 150)
    vis_edge_raw = vis.copy()
    vis_edge_raw[edges_raw > 0] = [0, 0, 255] # Red contours = Raw IR (shows severe misalignment shift!)

    # Crops
    c_vis = vis[crop_y1:crop_y2, crop_x1:crop_x2]
    c_raw_edge = vis_edge_raw[crop_y1:crop_y2, crop_x1:crop_x2]
    c_aln_edge = vis_edge_aln[crop_y1:crop_y2, crop_x1:crop_x2]
    c_raw_dep = raw_dep_cm[crop_y1:crop_y2, crop_x1:crop_x2]
    c_p0_dep = p0_dep_cm[crop_y1:crop_y2, crop_x1:crop_x2]

    th, tw = 400, 500
    def prep_tile(tile, title, subtitle=""):
        res = cv2.resize(tile, (tw, th))
        cv2.rectangle(res, (0, 0), (tw, 48), (25, 25, 25), -1)
        cv2.putText(res, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(res, subtitle, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return res

    t1 = prep_tile(c_vis, f"Optical RGB ({stem})", f"Target Zoom: {desc}")
    t2 = prep_tile(c_raw_edge, "Raw IR Edges on RGB (Red)", "Notice Visible Phase Disparity & Shift")
    t3 = prep_tile(c_aln_edge, "Aligned IR Edges on RGB (Green)", "Tight Sub-Pixel Boundary Snapping")
    t4 = prep_tile(c_raw_dep, "Raw Sensor Depth", "Coarse, Quantized, No Object Contour")
    t5 = prep_tile(c_p0_dep, "Preprocessed P0 Depth", "Sharp Physical Silhouette & Continuous Metric")

    # Layout: Top row [RGB, Raw IR Edge, Aligned IR Edge], Bottom row [RGB, Raw Depth, P0 Depth]
    top = np.hstack([t1, t2, t3])
    bot = np.hstack([t1, t4, t5])
    canvas = np.vstack([top, bot])

    out_p = os.path.join(OUT_DIR, f"qc_zoom_crop_{stem}.jpg")
    cv2.imwrite(out_p, canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"[+] Saved zoomed crop artifact: {out_p}")
