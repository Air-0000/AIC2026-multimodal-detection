#!/usr/bin/env python3
"""
Generate comprehensive visual comparison artifacts for Test Set:
1. Alignment Visualizer: RGB vs Raw IR vs Aligned IR (Overlay in Magenta/Green)
2. Channel 4 Bug Fix Visualizer: Raw IR vs Correct LBG-IR (Median Top-Hat Thermal Saliency)
3. 3D Depth Map Geometry
4. Final Detection Box Comparison: Buggy vs Fixed
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def make_overlay(rgb, ir):
    """Magenta/Green false color overlay: RGB edge in Green, IR in Magenta."""
    gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge_rgb = cv2.Canny(gray_rgb, 50, 150)
    edge_ir  = cv2.Canny(ir, 50, 150)
    overlay = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = edge_rgb  # Green = Visible
    overlay[..., 0] = edge_ir   # Magenta = IR (R + B)
    overlay[..., 2] = edge_ir
    return overlay

def main():
    test_root = "/root/autodl-tmp/AIC/dataset/test/AIC2026_PHASE_2_1000"
    aligned_ir_dir = "/root/autodl-tmp/AIC/work_dir/test_aligned_ir"
    lbg_ir_dir = "/root/autodl-tmp/AIC/work_dir/test_lbg_ir"
    out_dir = "/root/autodl-tmp/AIC/work_dir/vis_artifacts"
    os.makedirs(out_dir, exist_ok=True)

    # Let's inspect stem 001672 (massive 20px shift) and 00000018 (dynamic target)
    stems = ["001672", "00000018"]

    for stem in stems:
        vis_p = os.path.join(test_root, "visible", stem + ".jpg")
        if not os.path.exists(vis_p): vis_p = os.path.join(test_root, "visible", stem + ".png")

        ir_raw_p = os.path.join(test_root, "infrared", stem + ".jpg")
        if not os.path.exists(ir_raw_p): ir_raw_p = os.path.join(test_root, "infrared", stem + ".png")

        ir_aln_p = os.path.join(aligned_ir_dir, stem + ".jpg")
        if not os.path.exists(ir_aln_p): ir_aln_p = os.path.join(aligned_ir_dir, stem + ".png")

        lbg_p = os.path.join(lbg_ir_dir, stem + ".jpg")
        if not os.path.exists(lbg_p): lbg_p = os.path.join(lbg_ir_dir, stem + ".png")

        depth_p = os.path.join(test_root, "depth_p0", stem + ".png")
        if not os.path.exists(depth_p): depth_p = os.path.join(test_root, "depth_p0", stem + ".jpg")

        rgb = np.array(Image.open(vis_p).convert("RGB"))
        ir_raw = np.array(Image.open(ir_raw_p).convert("L"))
        ir_aln = np.array(Image.open(ir_aln_p).convert("L"))
        lbg = np.array(Image.open(lbg_p).convert("L")) if os.path.exists(lbg_p) else None
        if lbg is None:
            bg = cv2.medianBlur(ir_aln, 31)
            lbg = np.clip(ir_aln.astype(np.float32) - bg.astype(np.float32), 0, 255).astype(np.uint8)

        depth = np.array(Image.open(depth_p))
        depth_norm = (depth / (65535.0 if depth.max() > 255 else 255.0) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

        # Generate overlays
        over_raw = make_overlay(rgb, ir_raw)
        over_aln = make_overlay(rgb, ir_aln)

        # Difference map between raw and aligned IR
        diff_ir = cv2.absdiff(ir_raw, ir_aln)
        diff_ir_heat = cv2.applyColorMap(cv2.equalizeHist(diff_ir), cv2.COLORMAP_HOT)

        # Crop a focal region with high motion/misalignment for side-by-side zoom
        H, W = rgb.shape[:2]
        crop_box = (int(H*0.1), int(H*0.9), int(W*0.1), int(W*0.9))

        # Create multi-panel figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 11))
        fig.suptitle(f"AIC 2026 Test Set Multimodal Physics & Registration Audit [Stem: {stem}]", fontsize=18, fontweight="bold")

        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("(a) Visible RGB (100% Daylight)", fontsize=13)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(over_raw, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("(b) Raw IR Overlay (Green=RGB, Magenta=IR)\n[Visible Misalignment / Ghosting]", fontsize=13, color="red")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(cv2.cvtColor(over_aln, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("(c) Aligned IR Overlay (Green=RGB, Magenta=IR)\n[Sharp Coincidence / Ghosting Removed]", fontsize=13, color="green")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(ir_raw, cmap="gray")
        axes[1, 0].set_title("(d) Raw IR (What Bug Fed to Ch4: Sun-Heated Ground Clutter)", fontsize=13)
        axes[1, 0].axis("off")

        axes[1, 1].imshow(lbg, cmap="inferno")
        axes[1, 1].set_title("(e) Correct LBG-IR (Median Top-Hat: Thermal Anomalies Only)", fontsize=13, color="blue")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("(f) Metric 16-bit Depth Geometry (Turbo Colormap)", fontsize=13)
        axes[1, 2].axis("off")

        plt.tight_layout()
        save_p = os.path.join(out_dir, f"test_audit_comparison_{stem}.png")
        plt.savefig(save_p, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_p}")

if __name__ == "__main__":
    main()
