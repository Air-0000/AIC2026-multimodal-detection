"""Evaluate a submission dir against the AIC validation COCO annotations.

Usage:
    /home/constant/miniconda3/envs/deimv2/bin/python eval_submission.py [SUB_DIR] [ANN_JSON]
Defaults: SUB_DIR=work_dir/submission_wbf
          ANN_JSON=work_dir/deimv2_dataset/valid/annotations/instances_val.json
"""
import os
import sys
import glob
import json

import numpy as np

SUB_DIR = sys.argv[1] if len(sys.argv) > 1 else "work_dir/submission_wbf"
ANN_JSON = sys.argv[2] if len(sys.argv) > 2 else \
    "work_dir/deimv2_dataset/valid/annotations/instances_val.json"

from PIL import Image
import faster_coco_eval
faster_coco_eval.init_as_pycocotools()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- load GT ---
coco_gt = COCO(ANN_JSON)
img_meta = {img["id"]: img for img in coco_gt.dataset["images"]}
# image file_name -> id
name2id = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}

# --- load predictions from txt dir ---
detections = []
for txt in sorted(glob.glob(os.path.join(SUB_DIR, "*.txt"))):
    stem = os.path.splitext(os.path.basename(txt))[0]
    img_id = None
    for ext in (".jpg", ".png"):
        fname = stem + ext
        if fname in name2id:
            img_id = name2id[fname]
            break
    if img_id is None:
        continue
    W, H = img_meta[img_id]["width"], img_meta[img_id]["height"]
    for line in open(txt):
        parts = line.split()
        if len(parts) != 6:
            continue
        cls, cx, cy, bw, bh, conf = map(float, parts)
        x = (cx - bw / 2) * W
        y = (cy - bh / 2) * H
        detections.append({
            "image_id": img_id,
            "category_id": int(cls),
            "bbox": [x, y, bw * W, bh * H],
            "score": conf,
        })

print(f"[eval] {len(detections)} detections from {SUB_DIR}")
if not detections:
    print("[eval] no detections!")
    sys.exit(1)

coco_dt = coco_gt.loadRes(detections)
evaler = COCOeval(coco_gt, coco_dt, "bbox")
evaler.evaluate()
evaler.accumulate()
evaler.summarize()
stats = evaler.stats
print(f"\n[eval] mAP@50:95={stats[0]:.4f}  AP@50={stats[1]:.4f}  AP@75={stats[2]:.4f}")

# per-class AP using the cat id list from the GT annotation file
import collections
cat_list = [(c["id"], c["name"]) for c in coco_gt.dataset["categories"]]
cnt = collections.Counter(a["category_id"] for a in coco_gt.dataset["annotations"])
print("\n=== Per-class AP (order = categories in ann json) ===")
for i, (cid, nm) in enumerate(cat_list):
    # precision shape: (iou_thr, recall, cat_idx, area, maxDets)
    ap = evaler.eval["precision"][:, :, i, 0, 2]
    ap = ap[ap > -1]
    ap50 = evaler.eval["precision"][0, :, i, 0, 2]
    ap50 = ap50[ap50 > -1]
    print(f"  {nm:12s}: AP50:95={ap.mean():.4f}  AP50={ap50.mean():.4f}  GT={cnt.get(cid,0)}")
# detection count per class
dcnt = collections.Counter(d["category_id"] for d in detections)
print("\n=== Detections per class (pred) ===")
for cid, nm in cat_list:
    print(f"  {nm:12s}: pred={dcnt.get(cid,0)}  gt={cnt.get(cid,0)}")
