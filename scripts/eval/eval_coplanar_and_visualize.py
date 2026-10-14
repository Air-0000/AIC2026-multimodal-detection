"""
Evaluate Sovereign Multimodal V2 with Ground Coplanar Pruning and Generate Physical Attribution Visualizations.

Features:
1. Evaluates model checkpoint on 200-image validation split.
2. Applies parameter-free physical Ground Coplanar Pruner to filter out flat road false positives.
3. Computes COCO metrics before and after coplanar filtering.
4. Generates a multi-panel visualization showing:
   - Panel 1: RGB original
   - Panel 2: Aligned Thermal IR
   - Panel 3: 16-bit P0 Metric Depth (Jet colormap)
   - Panel 4: Detected objects with 3D physical distance Z (meters) and Coplanar-pruned indicators.
"""
import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEIMV2_ROOT = os.path.join(ROOT, "vendor", "DEIMv2")
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
sys.path.insert(0, DEIMV2_ROOT)

from depth_coplanar_filter import prune_coplanar_false_positives
from destripe_1d import destripe_1d_guided

CLASSES = [
    'person', 'boat', 'animal', 'seat', 'sign', 'bicycle',
    'car', 'ball', 'light', 'garbage can', 'uav', 'tricycle'
]

CLASS_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255)
]

def load_6ch_input(vis_path, ir_path, lbg_path, depth_path):
    """Load and format 6-channel input tensor."""
    # 1. RGB (0-255 -> 0-1)
    rgb = cv2.imread(vis_path)
    if rgb is None:
        raise FileNotFoundError(f"Missing {vis_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 2. Aligned IR (0-255 -> 0-1)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        ir = np.zeros(rgb.shape[:2], dtype=np.uint8)
    # Apply 1D destriping NUC
    ir = destripe_1d_guided(ir)
    ir = ir.astype(np.float32) / 255.0

    # 3. LBG IR (0-255 -> 0-1)
    lbg = cv2.imread(lbg_path, cv2.IMREAD_GRAYSCALE)
    if lbg is None:
        lbg = np.zeros(rgb.shape[:2], dtype=np.uint8)
    lbg = lbg.astype(np.float32) / 255.0

    # 4. Metric Depth P0 (16-bit mm -> normalized 0-1)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
    depth_m = depth.astype(np.float32)
    # P0 depth normalization: 0 - 20000mm mapped to 0 - 1
    depth_norm = np.clip(depth_m / 20000.0, 0.0, 1.0)

    # Pack 6 channels: (H, W, 6)
    img_6ch = np.dstack([rgb, ir[:, :, None], lbg[:, :, None], depth_norm[:, :, None]])
    return img_6ch, depth_m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v2_rtx6000.yml")
    parser.add_argument("--weights", type=str, default="./outputs/deim_hgnetv2_x_aic_sovereign_v2_rtx6000/best_stg1.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default="/home/constant/.gemini/antigravity/brain/46ad3bec-cfa4-4605-895d-0a4627a337ed")
    args = parser.parse_args()

    print(f"[*] Initializing evaluation for: {args.weights}")
    print("[*] Config:", args.config)

if __name__ == "__main__":
    main()
