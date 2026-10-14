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
    work_dir/deimv2_heat/images/<source_name>   (3ch pseudo-RGB of the heat residual)  -- train
    work_dir/deimv2_heat/valid/<source_name>                                            -- valid
    work_dir/deimv2_heat/test/<source_name>                                             -- test
Output file names COPY the source visible-image names (extension included), so
they match the COCO `file_name` field and the deimv2_dataset layout exactly.

Usage:
    python prep_ir_heat.py [--mode heat|raw] [--split train|valid|test|all]
"""
import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import shift as ndshift

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SRC_IR = os.path.join(ROOT, "dataset/train/infrared")
SRC_IR_TEST = os.path.join(ROOT, "dataset/test/infrared")
VIS_TRAIN = os.path.join(ROOT, "work_dir/deimv2_dataset/train/images")
VIS_VAL = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/images")
VIS_TEST = os.path.join(ROOT, "dataset/test/visible")
# AIC2026 Phase 2 -- a second, DISJOINT 1000-image test set that lives next to
# dataset/test. Same modality layout as the phase-1 test split.
PHASE2 = os.path.join(ROOT, "dataset/test/AIC2026_PHASE_2_1000")
VIS_PHASE2 = os.path.join(PHASE2, "visible")
SRC_IR_PHASE2 = os.path.join(PHASE2, "infrared")
OUT = os.path.join(ROOT, "work_dir/deimv2_heat")
RAW_OUT = os.path.join(ROOT, "work_dir/deimv2_ir")


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


def process_split(vis_dir, ir_dir, out_dir, split_name, mode="heat"):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(vis_dir, "*")))
    ir_map = {os.path.splitext(os.path.basename(f))[0]: f
              for f in glob.glob(os.path.join(ir_dir, "*"))}
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
        if mode == "raw":
            # Step3: use the ALIGNED RAW IR grayscale as the 4th channel.
            # Unlike `heat`, we keep the full infrared structure (not just the
            # part brighter than RGB), so the IR stream can learn complementary
            # cues (cold/dark targets, edges) that the residual throws away.
            chan = np.clip(align_ir(ir, v), 0, 255).astype(np.uint8)
            img = Image.fromarray(chan).convert("L")   # store as a real grayscale
        else:
            heat = heat_residual(ir, v)
            # Keep the SOURCE file name (including its extension) so the heat file
            # matches COCO `file_name` / the RGB dataset layout exactly. The dataset
            # mixes .png and .jpg sources, and CocoDetectionMultiModal resolves the
            # heat path as `stem + ".png"` first, falling back to `file_name`; forcing
            # everything to .jpg silently hides the png-named entries (and previously
            # made the loader read a stale/legacy heat version).
            chan = heat
            img = Image.fromarray(np.stack([heat, heat, heat], axis=-1))
        name = os.path.basename(vp)
        out_path = os.path.join(out_dir, name)
        if mode == "raw":
            img.save(out_path)   # grayscale .png/.jpg both fine for convert("L")
        elif name.lower().endswith((".jpg", ".jpeg")):
            img.save(out_path, quality=95)
        else:
            img.save(out_path)
        n_ok += 1
        if (i + 1) % 300 == 0:
            print(f"[{split_name}/{mode}] {i + 1}/{len(files)}", flush=True)
    print(f"[{split_name}/{mode}] done: {n_ok} images -> {out_dir}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["heat", "raw"], default="heat",
                    help="heat=IR residual (Step1/2); raw=aligned raw IR gray (Step3)")
    ap.add_argument("--split", choices=["train", "valid", "test", "phase2", "all"], default="all",
                    help="which split(s) to (re)generate")
    args = ap.parse_args()
    out_root = RAW_OUT if args.mode == "raw" else OUT
    jobs = {
        "train": (VIS_TRAIN, SRC_IR, os.path.join(out_root, "images")),
        "valid": (VIS_VAL, SRC_IR, os.path.join(out_root, "valid")),
        "test": (VIS_TEST, SRC_IR_TEST, os.path.join(out_root, "test")),
        "phase2": (VIS_PHASE2, SRC_IR_PHASE2, os.path.join(out_root, "phase2")),
    }
    todo = ["train", "valid", "test"] if args.split == "all" else [args.split]
    for s in todo:
        vis_dir, ir_dir, out_dir = jobs[s]
        process_split(vis_dir, ir_dir, out_dir, s, args.mode)


if __name__ == "__main__":
    main()
