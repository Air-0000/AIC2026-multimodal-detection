"""Thermal signal from the organiser-made IR files (forensics 2026-10-28 on all 2000 training images).

For the hehe / seq / seq_suppl / shuming sources the IR file is 0.85·T + 0.15·RGB per channel, the RGB layer pixel-aligned
with the visible image (high-pass regression gives 0.150, p5–p95 0.147–0.153; after removing it the colour vanishes
and T is ~0 outside the thermal window). For num6 / num8 the IR file is T itself. T is a thermal image the organisers
warped into the RGB frame (black outside its window) and is still offset from the RGB by a median 20–40 px at 1920
width, varying within the image (parallax) — that offset is left to the learned aligner.

decompose_thermal() estimates the blend factor from high frequencies only (the blended RGB layer is aligned with the
visible image, the thermal content is not, so fine details correlate only through the blended layer), snaps it to
0.15 or 0, removes the RGB layer and returns T with its window mask. Pure numpy / OpenCV, used by the data pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np

BLEND = 0.15
LUMA_BGR = (0.114, 0.587, 0.299)


def _highpass(x: np.ndarray, sigma: float) -> np.ndarray:
    return x - cv2.GaussianBlur(x, (0, 0), sigma)


def blend_factor(rgb: np.ndarray, ir: np.ndarray, work_width: int = 960, hp_sigma: float = 2.0) -> float:
    """Least-squares slope of high-pass(IR gray) on high-pass(RGB gray) over non-black IR pixels (uint8 HxWx3 inputs)."""
    h, w = rgb.shape[:2]
    s = min(1.0, work_width / w)
    if s < 1.0:
        size = (int(round(w * s)), int(round(h * s)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        ir = cv2.resize(ir, size, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY).astype(np.float32)
    nonblack = cv2.erode((ir.max(2) > 6).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    hg, ht = _highpass(g, hp_sigma) * nonblack, _highpass(t, hp_sigma) * nonblack
    den = float((hg * hg).sum())
    return float((ht * hg).sum() / den) if den > 0 else 0.0


def thermal_window(t: np.ndarray, threshold: float, min_blob: int = 30) -> np.ndarray:
    """Window of the warped thermal image: the filled convex hull of every blob above `threshold` of at least
    `min_blob` pixels. The window is a perspective-warped rectangle clipped by the frame, so it is convex, and a cold
    region inside it that touches its edge (the sky of the 640x360 pure-thermal source, T 0-3) can only be told from
    the black outside by this geometry. The first version filled holes from the border instead and dropped such
    regions, with the UAVs in them (review 2026-10-29; blended and 6-digit windows change by at most 0.11 %)."""
    m = (t > threshold).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_blob
    pts = cv2.findNonZero(keep[lab].astype(np.uint8))
    out = np.zeros(t.shape, np.uint8)
    if pts is not None:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool)


def decompose_thermal(rgb: np.ndarray, ir: np.ndarray, snap: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """rgb, ir: uint8 HxWx3 in the same channel order and size. Returns (T uint8 HxW, window uint8 HxW 0/255, blend)."""
    if ir.shape != rgb.shape:
        raise ValueError(f'IR {ir.shape} and RGB {rgb.shape} differ; resize the IR first')
    a = blend_factor(rgb, ir)
    if snap:
        a = BLEND if a > BLEND / 2 else 0.0
    t = (ir.astype(np.float32) - a * rgb.astype(np.float32)) / (1.0 - a)
    # luma weights (BGR order), not the channel mean: then (1 − a)·T + a·luma(RGB) is exactly the IR grey value
    # (PIL convert('L')) that the teammate's fusion used; they differ on coloured marks in some IR files
    t = np.clip(t @ np.array(LUMA_BGR, np.float32), 0, 255)
    # outside the window T is ~0 (blend sources) or exactly black (pure thermal)
    win = thermal_window(t, threshold=2.5 if a > 0 else 6.0)
    return np.rint(t).astype(np.uint8), (win * 255).astype(np.uint8), a
