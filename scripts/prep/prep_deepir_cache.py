"""Offline cache builder for CocoDetectionMultiModalCached.

Reads each training/val sample exactly the way CocoDetectionMultiModal.load_item
does (RGB + heat + raw_ir + depth, all resized to RGB size, stacked to [C,H,W]
in [0,1]), then writes a single uint8 .pt per image into `out_dir/<stem>.pt`.

At train time CocoDetectionMultiModalCached loads these .pt files and applies the
SAME online augmentation pipeline (RandomNight5ch / RandomIoUCrop / Normalize /...),
so caching only removes the per-iter file decode+resize — it does NOT change any
augmentation or the model. Run once per dataset; supports resume (skips existing
.pt). Safe to rerun after a disk-full / interrupted build.
"""
import os
import sys
import argparse

import torch

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "DEIMv2"))

from engine.data.dataset.coco_dataset_mm import CocoDetectionMultiModal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="cache output dir (one .pt per image)")
    ap.add_argument("--img_folder", required=True)
    ap.add_argument("--ann_file", required=True)
    ap.add_argument("--ir_folder", default=None)
    ap.add_argument("--raw_ir_folder", default=None)
    ap.add_argument("--depth_folder", default=None)
    ap.add_argument("--n_channels", type=int, default=6)
    ap.add_argument("--modality", default="rgb_ir")
    ap.add_argument("--depth_mode", default="raw")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ds = CocoDetectionMultiModal(
        img_folder=args.img_folder,
        ann_file=args.ann_file,
        ir_folder=args.ir_folder,
        raw_ir_folder=args.raw_ir_folder,
        depth_folder=args.depth_folder,
        n_channels=args.n_channels,
        modality=args.modality,
        depth_mode=args.depth_mode,
    )
    n = len(ds)
    print(f"[cache] building {n} samples -> {args.out}", flush=True)

    done = 0
    for idx in range(n):
        img_name = ds.coco.loadImgs(ds.ids[idx])[0]["file_name"]
        stem = os.path.splitext(img_name)[0]
        out_path = os.path.join(args.out, stem + ".pt")
        if os.path.exists(out_path):
            done += 1
            continue
        x, _ = ds.load_item(idx)            # [C,H,W] float in [0,1]
        torch.save((x * 255.0).clamp(0, 255).to(torch.uint8), out_path)
        done += 1
        if done % 200 == 0 or done == n:
            print(f"[cache] {done}/{n}", flush=True)
    print(f"[cache] done {done}/{n}", flush=True)


if __name__ == "__main__":
    main()
