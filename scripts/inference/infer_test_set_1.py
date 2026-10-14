#!/usr/bin/env python3
"""
Sovereign V21 Apex Soup - Test Set 1 Full Inference Pipeline
============================================================
Runs native multimodal inference on Phase 1 Test Set (1,000 images in dataset/test/):
1. Saves complete raw predictions cache (raw_predictions.pt) for zero-latency instant re-eval
2. Exports standard COCO-format predictions (coco_predictions.json) for official mAP evaluation
3. Generates official competition TXT prediction directories (tau >= 0.500 and tau >= 0.200)
4. Prepares for immediate ground-truth evaluation once labels are provided
"""

import os
import sys
import glob
import time
import json
import zipfile
import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF

aic_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in [aic_root, os.path.join(aic_root, 'vendor', 'DEIMv2'), os.path.join(aic_root, 'scripts', 'lib')]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from engine.core import YAMLConfig
from scripts.inference.generate_sovereign_v21_final_submission import (
    load_multimodal_tensor,
    class_wise_nms,
    CLASS_NAMES,
)


def main():
    parser = argparse.ArgumentParser(description="Test Set 1 Full Inference Pipeline")
    parser.add_argument("--config", default="vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_v14_rtx6000.yml")
    parser.add_argument("--weights", default="weights/sovereign_v21_apex_soup_581.pth")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--out-dir", default="work_dir/submissions/test1_predictions_sovereign_v21_apex_soup")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    txt_thr50_dir = os.path.join(args.out_dir, "txt_thr50")
    txt_thr20_dir = os.path.join(args.out_dir, "txt_thr20")
    os.makedirs(txt_thr50_dir, exist_ok=True)
    os.makedirs(txt_thr20_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(" Sovereign V21 Apex Soup - Test Set 1 Full Inference Engine ".center(70, "="))
    print("=" * 70)
    print(f"Model Config:     {args.config}")
    print(f"Weights:          {args.weights}")
    print(f"Test Set Dir:     {args.test_dir}")
    print(f"Output Dir:       {args.out_dir}")
    print(f"Inference Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    cfg = YAMLConfig(args.config)
    model = cfg.model.to(device).eval()
    ck = torch.load(args.weights, map_location="cpu")
    w = ck["ema"]["module"] if ("ema" in ck and "module" in ck["ema"]) else ck.get("model", ck)
    model.load_state_dict(w, strict=False)

    vis_dir = os.path.join(args.test_dir, "visible")
    img_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
    stems = [Path(f).stem for f in img_files if not f.endswith(".txt")]
    print(f"Discovered {len(stems)} official Phase 1 test images.\n")

    raw_cache = {}
    coco_predictions = []
    total_dets_50 = 0
    total_dets_20 = 0
    empty_50 = 0
    class_counter_50 = Counter()

    t0 = time.time()
    with torch.no_grad():
        for stem in tqdm(stems, desc="Inferring Test Set 1"):
            try:
                inp, orig_w, orig_h = load_multimodal_tensor(stem, args.test_dir, device)
            except Exception as e:
                print(f"\n[!] Error loading {stem}: {e}")
                with open(os.path.join(txt_thr50_dir, f"{stem}.txt"), "wb") as f:
                    f.write(b"")
                with open(os.path.join(txt_thr20_dir, f"{stem}.txt"), "wb") as f:
                    f.write(b"")
                empty_50 += 1
                continue

            outputs = model(inp)
            logits = outputs["pred_logits"][0]  # [300, 12]
            boxes = outputs["pred_boxes"][0]    # [300, 4] normalized cxcywh

            scores, labels = logits.sigmoid().max(dim=-1)

            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()
            boxes_np = boxes.cpu().numpy()

            # Save in raw cache
            raw_cache[stem] = {
                "orig_w": orig_w,
                "orig_h": orig_h,
                "scores": scores_np,
                "labels": labels_np,
                "boxes": boxes_np,
            }

            # 1. COCO Format Predictions (All detections >= 0.01 for standard COCO eval)
            for sc, lb, bx in zip(scores_np, labels_np, boxes_np):
                if sc >= 0.01:
                    cx, cy, bw, bh = bx
                    x1 = float(np.clip((cx - bw / 2.0) * orig_w, 0, orig_w))
                    y1 = float(np.clip((cy - bh / 2.0) * orig_h, 0, orig_h))
                    w_box = float(np.clip(bw * orig_w, 0, orig_w))
                    h_box = float(np.clip(bh * orig_h, 0, orig_h))
                    coco_predictions.append({
                        "image_id": stem,
                        "category_id": int(lb),
                        "bbox": [round(x1, 2), round(y1, 2), round(w_box, 2), round(h_box, 2)],
                        "score": round(float(sc), 5),
                    })

            # 2. Strict Submission Predictions (tau >= 0.500, NMS 0.55, Top-28)
            cands_50 = []
            for sc, lb, bx in zip(scores_np, labels_np, boxes_np):
                if sc >= 0.50:
                    cands_50.append((float(sc), int(lb), (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]))))
            cands_50.sort(key=lambda x: -x[0])
            clean_50 = class_wise_nms(cands_50, iou_thresh=0.55)[:28]

            out_lines_50 = []
            for sc, cid, (cx, cy, bw, bh) in clean_50:
                cx_c = float(np.clip(cx, 0.0, 1.0))
                cy_c = float(np.clip(cy, 0.0, 1.0))
                bw_c = float(np.clip(bw, 0.0, 1.0))
                bh_c = float(np.clip(bh, 0.0, 1.0))
                sc_c = float(np.clip(sc, 0.0, 1.0))
                out_lines_50.append(f"{cid} {cx_c:.6f} {cy_c:.6f} {bw_c:.6f} {bh_c:.6f} {sc_c:.6f}")
                class_counter_50[cid] += 1
                total_dets_50 += 1

            out_path_50 = os.path.join(txt_thr50_dir, f"{stem}.txt")
            with open(out_path_50, "wb") as f:
                if out_lines_50:
                    content = "\r\n".join(out_lines_50) + "\r\n"
                    f.write(content.encode("utf-8"))
                else:
                    f.write(b"")
                    empty_50 += 1

            # 3. Medium Threshold Predictions (tau >= 0.200, NMS 0.55, Top-100)
            cands_20 = []
            for sc, lb, bx in zip(scores_np, labels_np, boxes_np):
                if sc >= 0.20:
                    cands_20.append((float(sc), int(lb), (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]))))
            cands_20.sort(key=lambda x: -x[0])
            clean_20 = class_wise_nms(cands_20, iou_thresh=0.55)[:100]

            out_lines_20 = []
            for sc, cid, (cx, cy, bw, bh) in clean_20:
                cx_c = float(np.clip(cx, 0.0, 1.0))
                cy_c = float(np.clip(cy, 0.0, 1.0))
                bw_c = float(np.clip(bw, 0.0, 1.0))
                bh_c = float(np.clip(bh, 0.0, 1.0))
                sc_c = float(np.clip(sc, 0.0, 1.0))
                out_lines_20.append(f"{cid} {cx_c:.6f} {cy_c:.6f} {bw_c:.6f} {bh_c:.6f} {sc_c:.6f}")
                total_dets_20 += 1

            out_path_20 = os.path.join(txt_thr20_dir, f"{stem}.txt")
            with open(out_path_20, "wb") as f:
                if out_lines_20:
                    content = "\r\n".join(out_lines_20) + "\r\n"
                    f.write(content.encode("utf-8"))
                else:
                    f.write(b"")

    elapsed = time.time() - t0

    # Save raw predictions cache
    cache_path = os.path.join(args.out_dir, "raw_predictions.pt")
    torch.save(raw_cache, cache_path)
    print(f"\nSaved raw predictions cache to: {cache_path} ({len(raw_cache)} images)")

    # Save COCO predictions JSON
    coco_json_path = os.path.join(args.out_dir, "coco_predictions.json")
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_predictions, f)
    print(f"Saved COCO predictions JSON to: {coco_json_path} ({len(coco_predictions)} total detection entries)")

    # Pack thr50 to ZIP
    zip_path_50 = os.path.join(args.out_dir, "submission_test1_sovereign_v21_thr50.zip")
    with zipfile.ZipFile(zip_path_50, "w", zipfile.ZIP_DEFLATED) as z:
        for txt_f in sorted(glob.glob(os.path.join(txt_thr50_dir, "*.txt"))):
            z.write(txt_f, os.path.basename(txt_f))

    # Pack thr20 to ZIP
    zip_path_20 = os.path.join(args.out_dir, "submission_test1_sovereign_v21_thr20.zip")
    with zipfile.ZipFile(zip_path_20, "w", zipfile.ZIP_DEFLATED) as z:
        for txt_f in sorted(glob.glob(os.path.join(txt_thr20_dir, "*.txt"))):
            z.write(txt_f, os.path.basename(txt_f))

    print("\n" + "=" * 70)
    print(" TEST SET 1 INFERENCE COMPLETE ".center(70, "="))
    print("=" * 70)
    print(f"Total Test Images:        {len(stems)}")
    print(f"Total Detections (thr50): {total_dets_50} (avg {total_dets_50/len(stems):.2f} boxes/img)")
    print(f"Total Detections (thr20): {total_dets_20} (avg {total_dets_20/len(stems):.2f} boxes/img)")
    print(f"Empty Files (thr50):      {empty_50}")
    print(f"Time Taken:               {elapsed:.1f}s ({elapsed/len(stems)*1000:.1f}ms/img)")
    print("-------------------------------------------------------------------")
    print("Predictions Class Distribution (tau >= 0.500):")
    for cid in range(12):
        cname = CLASS_NAMES[cid]
        cnt = class_counter_50[cid]
        print(f"  [{cid:>2d}] {cname:<12s}: {cnt:>5d} ({cnt/len(stems):.2f}/img)")
    print("=" * 70)

    # Run check_submission
    check_script = os.path.join(aic_root, "scripts/eval/check_submission.py")
    if os.path.exists(check_script):
        print("\nRunning Official Submission Format Verification on thr50...")
        os.system(f"python3 {check_script} {txt_thr50_dir} {vis_dir}")


if __name__ == "__main__":
    main()
