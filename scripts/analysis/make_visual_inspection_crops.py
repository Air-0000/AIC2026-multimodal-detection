"""
Generate high-resolution side-by-side zoom crops for depth processing inspection,
and copy all visual artifact panels to the artifact directory.
"""
import os
import sys
import shutil
import cv2
import numpy as np
import torch

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, ROOT)
ART_DIR = "/home/constant/.gemini/antigravity/brain/7fd161eb-9712-4fa4-b0e6-4221a61b8664"
os.makedirs(ART_DIR, exist_ok=True)

from scripts.analysis.test_depth_geo_validation import (
    load_da_model, da_fill_and_filter, depth_to_surface_normal, get_gradient_map
)

def colorize_depth(d, vmin=0, vmax=15000):
    valid = d > 1.0
    dn = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1) * 255
    cm = cv2.applyColorMap(dn.astype(np.uint8), cv2.COLORMAP_TURBO)
    cm[~valid] = [30, 0, 30] # dark purple for holes
    return cm

def generate_zoom_strip(base_name, crop_box, title, model, device):
    v_path = f"dataset/train/visible/{base_name}.png"
    i_path = f"dataset/train/infrared/{base_name}.png"
    d_path = f"dataset/train/depth/{base_name}.png"

    vis = cv2.imread(v_path)
    ir = cv2.imread(i_path, cv2.IMREAD_UNCHANGED)
    if ir.ndim == 3: ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    depth_raw = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    H, W = depth_raw.shape

    # Process
    depth_refined, _, valid_mask = da_fill_and_filter(depth_raw, ir, model, device)
    normal_map = depth_to_surface_normal(depth_refined)

    # Gradients for 3-modal overlay
    e_rgb = get_gradient_map(vis)
    e_ir  = get_gradient_map(ir)
    e_geo = get_gradient_map(normal_map[..., :2])

    r_ch = (np.clip(e_rgb * 2.2, 0, 1) * 255).astype(np.uint8)
    g_ch = (np.clip(e_ir  * 2.2, 0, 1) * 255).astype(np.uint8)
    b_ch = (np.clip(e_geo * 2.5, 0, 1) * 255).astype(np.uint8)
    overlay = np.stack([b_ch, g_ch, r_ch], axis=-1)

    # Crop
    x1, y1, x2, y2 = crop_box
    x1, x2 = int(x1 * W), int(x2 * W)
    y1, y2 = int(y1 * H), int(y2 * H)

    size = (360, 360)
    c_vis = cv2.resize(vis[y1:y2, x1:x2], size)
    c_raw = cv2.resize(colorize_depth(depth_raw)[y1:y2, x1:x2], size)
    c_ref = cv2.resize(colorize_depth(depth_refined)[y1:y2, x1:x2], size)
    c_nor = cv2.resize((normal_map[y1:y2, x1:x2] * 255).astype(np.uint8), size)
    c_ovl = cv2.resize(overlay[y1:y2, x1:x2], size)

    # Label on tiles
    cv2.putText(c_vis, "A. Visible (RGB)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(c_raw, "B. Raw Depth (Holes)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(c_ref, "C. DA-Filled+Filtered", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(c_nor, "D. Surface Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(c_ovl, "E. 3-Modal Overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    strip = np.hstack([c_vis, c_raw, c_ref, c_nor, c_ovl])

    # Add header banner
    banner = np.zeros((50, strip.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    combined = np.vstack([banner, strip])

    out_file = os.path.join(ART_DIR, f"{base_name}_zoom_detail.jpg")
    cv2.imwrite(out_file, combined)
    print("Saved zoom detail:", out_file)
    return out_file

def main():
    model, device = load_da_model()

    # 1. Generate Zoom Details for 3 interesting cases
    generate_zoom_strip(
        "000002_083_00000169",
        (0.48, 0.32, 0.75, 0.58),
        "Scene 1 Detail: Tricycle & Bicycle Behind Trees (Dense Occlusion & Edge Sharpening)",
        model, device
    )

    generate_zoom_strip(
        "000006_010_00000389",
        (0.35, 0.72, 0.58, 1.00),
        "Scene 2 Detail: Close-range Vehicle Body & Ground Reflection Separation",
        model, device
    )

    generate_zoom_strip(
        "000003_018_00000167",
        (0.40, 0.15, 0.65, 0.65),
        "Scene 3 Detail: Backlight / Dark Silhouette Lamp Post & Canopy Boundary",
        model, device
    )

    # 2. Copy the 6 overview audit panels to the artifact directory
    panel_files = [
        "000002_083_00000169_audit_panel.jpg",
        "000006_010_00000389_audit_panel.jpg",
        "000003_018_00000167_audit_panel.jpg",
        "000002_081_00000047_audit_panel.jpg",
        "000004_009_00000321_audit_panel.jpg",
        "000002_082_00000061_audit_panel.jpg",
    ]
    src_dir = os.path.join(ROOT, "work_dir/visualizations/figures/depth_geo_validation")
    for pf in panel_files:
        src = os.path.join(src_dir, pf)
        dst = os.path.join(ART_DIR, pf)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print("Copied overview panel:", dst)

    print("All visual inspection assets copied successfully.")

if __name__ == "__main__":
    main()
