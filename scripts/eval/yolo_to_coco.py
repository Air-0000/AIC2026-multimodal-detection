#!/usr/bin/env python3
"""Convert YOLO-format dataset (work_dir/rfdetr_dataset) to COCO JSON for DEIMv2.

YOLO layout (per split):
    <split>/images/<id>.jpg
    <split>/labels/<id>.txt        # "cls cx cy w h" (normalized 0..1)
Output (DEIMv2-compatible):
    work_dir/deimv2_dataset/<split>/images/...   (symlink or copy)
    work_dir/deimv2_dataset/<split>/annotations/instances_<split>.json
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image

CLASSES = ["person", "boat", "animal", "seat", "sign", "bicycle",
           "car", "ball", "light", "garbage can", "uav", "tricycle"]


def convert_split(src: Path, dst_split: Path, split: str, copy: bool = False):
    img_dir = src / "images"
    lbl_dir = src / "labels"
    out_img = dst_split / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    for img_path in img_files:
        img_id = len(images) + 1
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({
            "id": img_id, "file_name": img_path.name,
            "width": w, "height": h,
        })
        # copy / symlink image into COCO image folder
        target = out_img / img_path.name
        if copy:
            shutil.copyfile(img_path, target)
        elif not target.exists():
            os.symlink(img_path.resolve(), target)

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            # clip normalized YOLO coords into [0, 1] at read time to tolerate
            # minor export rounding errors (e.g. cx=1.006, h=1.0009)
            cx = min(1.0, max(0.0, cx))
            cy = min(1.0, max(0.0, cy))
            bw = min(1.0, max(0.0, bw))
            bh = min(1.0, max(0.0, bh))
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            bw_px, bh_px = bw * w, bh * h
            # clamp into image
            x = max(0.0, min(x, w - 1))
            y = max(0.0, min(y, h - 1))
            bw_px = max(1.0, min(bw_px, w - x))
            bh_px = max(1.0, min(bh_px, h - y))
            annotations.append({
                "id": ann_id, "image_id": img_id,
                # NOTE: DEIMv2 expects 0-indexed labels (0..num_classes-1); YOLO
                # classes are 0-indexed already, so pass them through directly.
                "category_id": cls,
                "bbox": [x, y, bw_px, bh_px],
                "area": bw_px * bh_px,
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
    out = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    ann_dir = dst_split / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    out_file = ann_dir / f"instances_{split}.json"
    with open(out_file, "w") as f:
        json.dump(out, f)
    print(f"[{split}] images={len(images)} annotations={len(annotations)} -> {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="work_dir/rfdetr_dataset")
    ap.add_argument("--dst", default="work_dir/deimv2_dataset")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlink")
    args = ap.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    convert_split(src / "train", dst / "train", "train", copy=args.copy)
    convert_split(src / "valid", dst / "valid", "val", copy=args.copy)


if __name__ == "__main__":
    main()
