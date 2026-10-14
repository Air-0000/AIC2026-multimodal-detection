#!/usr/bin/env python3
"""
AIC Multimodal Detection: Unified Depth Metric Normalization & Tone Mapping Library
==================================================================================
Handles the bit-depth discrepancy between:
  1. 1080p PNG images (uint16 mm depth: 0..19999 mm, where 19999 = 20.0m)
  2. 640x360 JPG images (uint8 normalized depth: 0..255, where 255 ≈ 20.0m)

Provides:
  - read_depth_metric(path): returns [1, H, W] float32 tensor in [0.0, 1.0] (1.0 = 20.0m)
  - read_depth_mm(path): returns (H, W) float32 numpy array in millimeters (0..19999)
  - colorize_depth(path_or_array, colormap='turbo'): returns (H, W, 3) uint8 BGR image
"""

import os
import cv2
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
from PIL import Image

def read_depth_metric(path):
    """
    Read depth and return unified normalized float32 tensor in [0.0, 1.0].
    0.0 represents invalid / missing sensor return.
    1.0 represents the maximum sensing horizon (19999 mm ≈ 20.0 m).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth file not found: {path}")

    im = Image.open(path)
    arr = np.array(im, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    # Detect 16-bit vs 8-bit from PIL mode and array dtype
    is_16bit = getattr(im, 'mode', '') in ('I;16', 'I') or arr.max() > 255.0
    scale = 19999.0 if is_16bit else 255.0
    arr_norm = np.clip(arr / scale, 0.0, 1.0)
    if HAS_TORCH:
        return torch.from_numpy(arr_norm).unsqueeze(0).float()
    return arr_norm[np.newaxis, ...]

def read_depth_mm(path):
    """
    Read depth and return millimeter metric distance array (H, W) float32.
    0.0 = invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth file not found: {path}")

    im = Image.open(path)
    arr = np.array(im, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    is_16bit = getattr(im, 'mode', '') in ('I;16', 'I') or arr.max() > 255.0
    if is_16bit:
        return np.clip(arr, 0.0, 19999.0)
    else:
        # Scale 8-bit normalized [0, 255] to millimeter scale [0, 19999]
        return np.clip(arr * (19999.0 / 255.0), 0.0, 19999.0)

def colorize_depth(depth_input, colormap="turbo"):
    """
    Convert depth (file path or numpy array) into high-contrast BGR image for human visualization.
    Maps valid depths (0.3m ~ 20m) across the full visual spectrum, invalid pixels remain black.
    """
    if isinstance(depth_input, str):
        arr_mm = read_depth_mm(depth_input)
    elif isinstance(depth_input, np.ndarray):
        arr_mm = depth_input.copy()
        if arr_mm.max() <= 1.01:
            arr_mm = arr_mm * 19999.0
        elif arr_mm.max() <= 255.0 and arr_mm.dtype == np.uint8:
            arr_mm = arr_mm.astype(np.float32) * (19999.0 / 255.0)
    elif isinstance(depth_input, torch.Tensor):
        arr_mm = depth_input.squeeze().detach().cpu().numpy()
        if arr_mm.max() <= 1.01:
            arr_mm = arr_mm * 19999.0

    valid = arr_mm > 10.0 # valid sensor distance > 10mm
    H, W = arr_mm.shape[:2]
    out_bgr = np.zeros((H, W, 3), dtype=np.uint8)

    if not valid.any():
        return out_bgr

    # Map distance: near (0.5m) to warm colors, far (20m) to cold colors
    norm_8u = np.zeros((H, W), dtype=np.uint8)
    norm_vals = np.clip((arr_mm[valid] / 19999.0) * 255.0, 1, 255).astype(np.uint8)
    norm_8u[valid] = norm_vals

    if colormap == "turbo":
        colored = cv2.applyColorMap(norm_8u, cv2.COLORMAP_TURBO)
    elif colormap == "jet":
        colored = cv2.applyColorMap(norm_8u, cv2.COLORMAP_JET)
    elif colormap == "inferno":
        colored = cv2.applyColorMap(norm_8u, cv2.COLORMAP_INFERNO)
    else:
        colored = cv2.cvtColor(norm_8u, cv2.COLOR_GRAY2BGR)

    colored[~valid] = [0, 0, 0]
    return colored
