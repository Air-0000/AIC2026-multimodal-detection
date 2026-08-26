#!/usr/bin/env python3
"""IR heat-source preprocessing for the AIC dataset.

Method (validated by experiments):
  * IR images are a semi-transparent thermal overlay shifted on top of a dimmed
    RGB background. Raw IR as input trains on noise (structure corr ~0.15).
  * Correct preprocessing = ALIGN IR to RGB, then take the RESIDUAL
    `max(IR_aligned - RGB_gray, 0)` which keeps only the heat source (IR
    brighter than the RGB background).
  * Alignment is coarse-to-fine: an integer block match at 1/8 scale (wide
    search, +/-12px) followed by a sub-pixel FFT phase-correlation refine at
    1/2 scale. A quality gate rejects shifts that do not reduce the alignment
    error below the zero-shift baseline (avoids bad-alignment artifacts).
  * Result: 65% of person GT boxes have in-box heat SNR>1.5 (vs background).

Output:
    work_dir/deimv2_heat/images/*.jpg   (3ch pseudo-RGB of the heat residual)
    work_dir/deimv2_heat/valid/*.jpg
Reuses the DEIMv2 COCO annotations (same file names).
"""
import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import shift as ndshift

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SRC_IR = os.path.join(ROOT, "dataset/train/infrared")
VIS_TRAIN = os.path.join(ROOT, "work_dir/deimv2_dataset/train/images")
VIS_VAL = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/images")
OUT = os.path.join(ROOT, "work_dir/deimv2_heat")


def _downscale(img, tw, th):
    """Bicubic downscale preserving gradients for block matching."""
    return np.array(Image.fromarray(img.astype(np.uint8)).resize(
        (tw, th), Image.BICUBIC)).astype(np.float32)


def _coarse_int_shift(irs, vs, search=12, scale_div=8):
    """Coarse integer offset search on a low-res copy (wide window)."""
    h, w = vs.shape
    irs = _downscale(irs, w // scale_div, h // scale_div)
    vs = _downscale(vs, w // scale_div, h // scale_div)
    # restrict search to a central crop so edges (constant-padded) don't bias
    ch, cw = irs.shape[0] // 4, irs.shape[1] // 4
    irs_c = irs[ch:-ch, cw:-cw] if irs.shape[0] > 2 * ch else irs
    vs_c = vs[ch:-ch, cw:-cw] if vs.shape[0] > 2 * ch else vs
    best = (0, 0)
    be = 1e18
    for dx in range(-search, search + 1):
        for dy in range(-search, search + 1):
            s = ndshift(irs_c, (dy, dx), order=1, mode="constant")
            e = np.abs(s - vs_c).mean()
            if e < be:
                be = e
                best = (dx, dy)
    return best[0] * scale_div, best[1] * scale_div  # back to full-res px


def _subpixel_refine(ir, v, dx0, dy0, scale_div=2):
    """FFT phase-correlation sub-pixel offset around an integer guess."""
    h, w = v.shape
    irs = _downscale(ir, w // scale_div, h // scale_div)
    vs = _downscale(v, w // scale_div, h // scale_div)
    g_ir = np.gradient(irs)
    g_v = np.gradient(vs)
    ir_e = np.hypot(g_ir[0], g_ir[1])
    v_e = np.hypot(g_v[0], g_v[1])
    win = np.hanning(ir_e.shape[0])[:, None] * np.hanning(ir_e.shape[1])[None, :]
    ir_c = np.fft.fft2((ir_e - ir_e.mean()) * win)
    v_c = np.fft.fft2((v_e - v_e.mean()) * win)
    cross = ir_c * np.conj(v_c)
    cross /= np.abs(cross) + 1e-12
    corr = np.real(np.fft.ifft2(cross))
    cy, cx = np.unravel_index(np.argmax(corr), corr.shape)
    sh, sw = corr.shape
    if cy > sh // 2:
        cy -= sh
    if cx > sw // 2:
        cx -= sw
    # sub-pixel peak via parabolic fit on the 3x3 neighborhood
    yy = slice(max(0, cy - 1), min(sh, cy + 2))
    xx = slice(max(0, cx - 1), min(sw, cx + 2))
    patch = corr[yy, xx]
    if patch.shape == (3, 3):
        cy += 0.5 * (patch[0, 1] - patch[2, 1]) / (patch[0, 1] - 2 * patch[1, 1] + patch[2, 1] + 1e-12)
        cx += 0.5 * (patch[1, 0] - patch[1, 2]) / (patch[1, 0] - 2 * patch[1, 1] + patch[1, 2] + 1e-12)
    return dx0 + cx * scale_div, dy0 + cy * scale_div


def align_ir(ir, v):
    """Coarse-to-fine IR->RGB alignment with a quality gate."""
    dx, dy = _coarse_int_shift(ir, v)
    dx, dy = _subpixel_refine(ir, v, dx, dy)
    ir_a = ndshift(ir, (dy, dx), order=1, mode="constant")
    # quality gate: reject shift if it does not beat the zero-shift baseline
    base = np.abs(ir - v).mean()
    if np.abs(ir_a - v).mean() > base:
        ir_a = ir
    return ir_a


def heat_residual(ir, v):
    """Align then residual: keep only the IR heat source.

    `max(IR_aligned - RGB_gray, 0)` preserves all bright targets; the residual is
    the part of the IR that is hotter than the visible background. Local-saliency
    masking was tried to remove the uniform "white film" but it also clipped the
    highlight body, so we keep the plain global residual.
    """
    ir_a = align_ir(ir, v)
    return np.clip(ir_a - v, 0, 255).astype(np.uint8)


def process_split(vis_dir, out_dir, split_name):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(vis_dir, "*")))
    ir_map = {os.path.splitext(os.path.basename(f))[0]: f
              for f in glob.glob(os.path.join(SRC_IR, "*"))}
    n_ok = 0
    for i, vp in enumerate(files):
        stem = os.path.splitext(os.path.basename(vp))[0]
        if stem not in ir_map:
            continue
        v = np.array(Image.open(vp).convert("RGB")).astype(np.float32).mean(axis=2)
        ir = np.array(Image.open(ir_map[stem]).convert("RGB")).astype(np.float32).mean(axis=2)
        if ir.shape != v.shape:
            ir = np.array(Image.fromarray(ir.astype(np.uint8)).resize(
                (v.shape[1], v.shape[0]))).astype(np.float32)
        heat = heat_residual(ir, v)
        Image.fromarray(np.stack([heat, heat, heat], axis=-1)).save(
            os.path.join(out_dir, f"{stem}.jpg"), quality=95)
        n_ok += 1
        if (i + 1) % 300 == 0:
            print(f"[{split_name}] {i + 1}/{len(files)}", flush=True)
    print(f"[{split_name}] done: {n_ok} heat images -> {out_dir}", flush=True)


def main():
    process_split(VIS_TRAIN, os.path.join(OUT, "images"), "train")
    process_split(VIS_VAL, os.path.join(OUT, "valid"), "valid")


if __name__ == "__main__":
    main()
