#!/usr/bin/env python3
"""全量批量叠加：deleak 热图 × RGB → 与 RGB 同名的叠加图。

  train: work_dir/deimv2_dataset/train/images  -> work_dir/rgb_heat_overlay/train
  valid: work_dir/deimv2_dataset/valid/images  -> work_dir/rgb_heat_overlay/valid

配方与 overlay_heat_on_rgb.overlay() 一致（resize→p2/p98 拉伸→JET→0.45 混色）。
缺热图的样本跳过并记录。输出 outputs/metrics/rgb_heat_overlay_manifest.json。
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, f"{ROOT}/scripts/prep")
from overlay_heat_on_rgb import find_file, load_heat, overlay  # noqa: E402

SPLITS = {
    "train": (f"{ROOT}/work_dir/deimv2_dataset/train/images",
              f"{ROOT}/work_dir/deleak/images"),
    "valid": (f"{ROOT}/work_dir/deimv2_dataset/valid/images",
              f"{ROOT}/work_dir/deleak/valid"),
}
ALPHA = 0.45


def main():
    manifest = {}
    for split, (rgb_dir, heat_dir) in SPLITS.items():
        out_dir = f"{ROOT}/work_dir/rgb_heat_overlay/{split}"
        os.makedirs(out_dir, exist_ok=True)
        done, skipped = 0, []
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(rgb_dir))
        for s in stems:
            rp = find_file(rgb_dir, s)
            hp = find_file(heat_dir, s)
            if rp is None:
                skipped.append([s, "no_rgb"])
                continue
            if hp is None:
                skipped.append([s, "no_heat"])
                continue
            rgb = cv2.imread(rp, cv2.IMREAD_COLOR)
            heat = load_heat(hp)
            ov, _ = overlay(rgb, heat, alpha=ALPHA)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(rp)), ov,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            done += 1
            if done % 400 == 0:
                print(f"[{split}] {done}/{len(stems)}", flush=True)
        manifest[split] = {"out": out_dir, "done": done, "skipped": skipped}
        print(f"[{split}] DONE done={done} skipped={len(skipped)}", flush=True)
    with open(f"{ROOT}/outputs/metrics/rgb_heat_overlay_manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
