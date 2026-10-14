#!/usr/bin/env python3
"""IR preprocessing WITHOUT the align_ir() step.

Why this exists
---------------
`prep_ir_heat.py` runs `align_ir(IR, RGB_gray)` before computing the heat
residual / the raw channel. Measurement on 2026-09-26 showed that step is
actively HARMFUL on this dataset:

  * `cv2.phaseCorrelate(IR, RGB_gray)` -> best shift is within +-3 px on every
    sampled image, i.e. the AIC IR-RGB pairs are ALREADY registered out of the
    box. Nothing needs aligning.
  * `align_ir()` nevertheless shifts the IR by dy = 46..86 px (brute-force
    search: corr(IR_raw, IR_src) rises from 0.17-0.32 to 0.91-0.99 at that
    offset). It minimizes L1(|IR - RGB_gray|), which is a brightness-matching
    objective, not a structure-alignment objective -- so it walks the image off
    true alignment.
  * Damage: corr(IR_src, RGB_gray) = -0.41..-0.54 (a stable, physically
    sensible anti-correlation: hot targets are dark in visible light), while
    corr(IR_aligned, RGB_gray) = -0.23..+0.42 -- the relation is scrambled
    toward zero.

This script produces the deleak (heat) channel with alignment DISABLED:

    heat : clip(IR - RGB_gray, 0, 255)       (plain global residual = the only IR feature used)

The raw IR is NOT written here anymore -- the 6ch cache builder
(`prep_deepir_cache.py`) reads raw IR straight from the original
`dataset/train/infrared`, and `ir_in_chans=1` means only the deleak channel
enters the network. So this script's single output is the unified deleak:

    work_dir/deleak/{images,valid}

(file names copied verbatim from the visible split, extensions included, to
match the COCO `file_name` field)
"""
import os
import glob
import numpy as np
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SRC_IR = os.path.join(ROOT, "dataset/train/infrared")
VIS_TRAIN = os.path.join(ROOT, "work_dir/deimv2_dataset/train/images")
VIS_VAL = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/images")
OUT_HEAT = os.path.join(ROOT, "work_dir/deleak")


def process_split(vis_dir, ir_dir, out_heat, split_name):
    os.makedirs(out_heat, exist_ok=True)
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
        heat = np.clip(ir - v, 0, 255).astype(np.uint8)

        name = os.path.basename(vp)
        Image.fromarray(np.stack([heat, heat, heat], axis=-1)).save(
            os.path.join(out_heat, name),
            quality=95) if name.lower().endswith((".jpg", ".jpeg")) else \
            Image.fromarray(np.stack([heat, heat, heat], axis=-1)).save(
                os.path.join(out_heat, name))
        n_ok += 1
        if (i + 1) % 500 == 0:
            print(f"[{split_name}] {i + 1}/{len(files)}", flush=True)
    print(f"[{split_name}] done: {n_ok} -> heat {out_heat}", flush=True)


def main():
    jobs = [
        ("train", VIS_TRAIN, os.path.join(OUT_HEAT, "images")),
        ("valid", VIS_VAL, os.path.join(OUT_HEAT, "valid")),
    ]
    for name, vis, oh in jobs:
        process_split(vis, SRC_IR, oh, name)


if __name__ == "__main__":
    main()
