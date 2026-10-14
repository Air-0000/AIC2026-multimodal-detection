"""
Sweep the inference confidence threshold on the VALID set (which has labels)
to find the threshold that maximizes val mAP@50:95. We run inference once
with threshold=0.0 (returns all post-NMS boxes) and then filter in memory for
each threshold — equivalent to re-running predict(threshold=t) but far cheaper.

Usage:
    python3 eval_threshold.py
"""
import os
import glob
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRBase

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
VALID_IMG = os.path.join(ROOT, "work_dir/rfdetr_dataset/valid/images")
VALID_LBL = os.path.join(ROOT, "work_dir/rfdetr_dataset/valid/labels")
CKPT = os.path.join(ROOT, "work_dir/rgb_baseline/checkpoint_best_ema.pth")
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
NUM_CLASSES = 12

# ---------- build COCO ground truth ----------
img_paths = sorted(glob.glob(os.path.join(VALID_IMG, "*")))
gt = {
    "images": [],
    "annotations": [],
    "categories": [{"id": i + 1, "name": str(i)} for i in range(NUM_CLASSES)],
}
ann_id = 1
stems = []
stem_to_iid = {}
for idx, p in enumerate(img_paths):
    stem = os.path.splitext(os.path.basename(p))[0]
    stems.append(stem)
    iid = idx + 1  # 1-based, robust to non-numeric filenames
    stem_to_iid[stem] = iid
    im = Image.open(p)
    W, H = im.size
    gt["images"].append({"id": iid, "file_name": os.path.basename(p), "width": W, "height": H})
    lblp = os.path.join(VALID_LBL, stem + ".txt")
    if os.path.exists(lblp):
        for line in open(lblp):
            line = line.strip()
            if not line:
                continue
            cid, cx, cy, w, h = line.split()
            cid = int(cid)
            cx, cy, w, h = map(float, (cx, cy, w, h))
            x = (cx - w / 2) * W
            y = (cy - h / 2) * H
            bw, bh = w * W, h * H
            gt["annotations"].append(
                {"id": ann_id, "image_id": iid, "category_id": cid + 1,
                 "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0}
            )
            ann_id += 1
import tempfile
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
    json.dump(gt, tf)
    gt_json = tf.name
cocoGt = COCO(gt_json)
print(f"[gt] {len(gt['images'])} images, {len(gt['annotations'])} boxes")

# ---------- one inference pass (threshold=0 -> all post-NMS boxes) ----------
model = RFDETRBase(pretrain_weights=CKPT, resolution=560, num_classes=NUM_CLASSES)
print("[infer] running predict(threshold=0.0) on valid set ...")
preds = {}
for p in img_paths:
    stem = os.path.splitext(os.path.basename(p))[0]
    im = Image.open(p).convert("RGB")
    det = model.predict(im, threshold=0.0)
    preds[stem] = (
        np.asarray(det.xyxy, dtype=float),
        np.asarray(det.class_id, dtype=int),
        np.asarray(det.confidence, dtype=float),
    )
print(f"[infer] done ({len(preds)} images)")

# ---------- sweep thresholds ----------
results = {}
for t in THRESHOLDS:
    dts = []
    for stem in stems:
        iid = stem_to_iid[stem]
        xyxy, cid, conf = preds[stem]
        keep = conf >= t
        xyxy, cid, conf = xyxy[keep], cid[keep], conf[keep]
        for (x1, y1, x2, y2), c, cf in zip(xyxy, cid, conf):
            dts.append({
                "image_id": iid,
                "category_id": int(c) + 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(cf),
            })
    cocoDt = cocoGt.loadRes(dts)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    m = cocoEval.stats[0]
    results[t] = m
    print(f"  threshold={t:.1f}  val mAP@50:95 = {m:.4f}   (boxes={len(dts)})")

best_t = max(results, key=results.get)
print("=" * 50)
print("SWEEP RESULT:")
for t in THRESHOLDS:
    print(f"  t={t:.1f} -> mAP {results[t]:.4f}")
print(f"BEST threshold = {best_t}  (mAP {results[best_t]:.4f})")
print("=" * 50)
with open(os.path.join(ROOT, "work_dir/threshold_sweep.json"), "w") as f:
    json.dump({"results": {str(k): v for k, v in results.items()}, "best": best_t}, f, indent=2)
