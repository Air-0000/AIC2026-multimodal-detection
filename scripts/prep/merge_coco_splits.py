#!/usr/bin/env python3
"""Merge COCO annotation files of disjoint splits into one file, e.g. train 1800 + valid 200 -> all 2000.

The splits come from separate yolo_to_coco.py runs, so their image and annotation ids can collide. Images
and annotations are renumbered from 1 in input order; file names must be unique across inputs (they share
one img_folder) and the category lists must be identical. OUT must not exist.

Example (repo root):
  python scripts/prep/merge_coco_splits.py \
    --inputs work_dir/deimv2_dataset/train/annotations/instances_train.json \
             work_dir/deimv2_dataset/valid/annotations/instances_val.json \
    --out work_dir/deimv2_dataset/all2000/annotations/instances_all2000.json
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f'refusing: {a.out} exists')
    sources = [json.loads(f.read_text()) for f in a.inputs]
    categories = sources[0]['categories']
    if any(s['categories'] != categories for s in sources[1:]):
        sys.exit('category lists differ between inputs')
    names = Counter(im['file_name'] for s in sources for im in s['images'])
    dup = [n for n, c in names.items() if c > 1]
    if dup:
        sys.exit(f'{len(dup)} file names appear in more than one input: {dup[:5]}')
    images, annotations = [], []
    for f, s in zip(a.inputs, sources):
        new_id = {}
        for im in s['images']:
            new_id[im['id']] = len(images) + 1
            images.append({**im, 'id': new_id[im['id']]})
        orphans = [ann['id'] for ann in s['annotations'] if ann['image_id'] not in new_id]
        if orphans:
            sys.exit(f'{f}: {len(orphans)} annotations point to missing images: {orphans[:5]}')
        for ann in s['annotations']:
            annotations.append({**ann, 'id': len(annotations) + 1, 'image_id': new_id[ann['image_id']]})
        print(f"{f}: {len(s['images'])} images, {len(s['annotations'])} annotations")
    merged = {k: v for k, v in sources[0].items() if k not in ('images', 'annotations', 'categories')}
    merged.update(images=images, annotations=annotations, categories=categories)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(merged))
    per_cat = Counter(ann['category_id'] for ann in annotations)
    names = {c['id']: c['name'] for c in categories}
    print(f'merged: {len(images)} images, {len(annotations)} annotations -> {a.out}')
    print('per class: ' + ', '.join(f'{names[k]} {per_cat[k]}' for k in sorted(names)))
    print('sha256: ' + hashlib.sha256(a.out.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
