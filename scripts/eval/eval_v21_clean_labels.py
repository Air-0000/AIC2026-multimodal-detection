#!/usr/bin/env python3
"""
Evaluate Sovereign V21 Apex Soup on 200 Validation Images:
Old Ground Truth vs New Quality Clean Ground Truth
===================================================
1. Loads Sovereign V21 Apex Soup (weights/sovereign_v21_apex_soup_581.pth)
2. Runs native 6-channel inference on the 200 validation images
3. Evaluates against:
   - Old Ground Truth: instances_val.json (1,523 boxes)
   - New Clean Ground Truth: instances_val_clean_quality.json (1,757 boxes, +234 newly annotated targets)
4. Displays side-by-side metric comparison:
   - Official mAP (0.50:0.95), AP50, AP75, and all 12 classes
"""

import os
import sys
import glob
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict

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
    CLASS_NAMES,
)
from scripts.eval.official_competition_evaluator import OfficialCompetitionEvaluator, NUM_CLASSES, ID2NAME


def main():
    config_path = "vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_v14_rtx6000.yml"
    weights_path = "weights/sovereign_v21_apex_soup_581.pth"
    old_gt_path = "work_dir/deimv2_dataset/valid/annotations/instances_val.json"
    clean_gt_path = "work_dir/deimv2_dataset/valid/annotations/instances_val_clean_quality.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" Sovereign V21 Apex Soup: Old GT vs New Clean GT Benchmark ".center(80, "="))
    print("=" * 80)

    cfg = YAMLConfig(config_path)
    model = cfg.model.to(device).eval()
    ck = torch.load(weights_path, map_location="cpu")
    w = ck["ema"]["module"] if ("ema" in ck and "module" in ck["ema"]) else ck.get("model", ck)
    model.load_state_dict(w, strict=False)

    with open(old_gt_path) as f:
        val_data = json.load(f)

    images = val_data["images"]
    print(f"Loaded {len(images)} validation images.")
    pred_dict = {}

    t0 = time.time()
    with torch.no_grad():
        for img in tqdm(images, desc="Evaluating 200 Val Images"):
            stem = os.path.splitext(img["file_name"])[0]
            inp, orig_w, orig_h = load_multimodal_tensor(stem, "dataset/train", device)
            out = model(inp)
            logits = out["pred_logits"][0]
            boxes = out["pred_boxes"][0]
            scores, labels = logits.sigmoid().max(dim=-1)

            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()
            boxes_np = boxes.cpu().numpy()

            # Convert normalized cxcywh to normalized xyxy
            cx = boxes_np[:, 0]
            cy = boxes_np[:, 1]
            bw = boxes_np[:, 2]
            bh = boxes_np[:, 3]
            x1 = np.clip(cx - bw / 2.0, 0.0, 1.0)
            y1 = np.clip(cy - bh / 2.0, 0.0, 1.0)
            x2 = np.clip(cx + bw / 2.0, 0.0, 1.0)
            y2 = np.clip(cy + bh / 2.0, 0.0, 1.0)
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            pred_dict[stem] = {
                "boxes": boxes_xyxy,
                "scores": scores_np,
                "labels": labels_np,
            }

    print(f"\nInference completed in {time.time() - t0:.2f}s.")

    print("\n" + "=" * 80)
    print(" 1. EVALUATION AGAINST OLD GROUND TRUTH (instances_val.json, 1523 boxes) ".center(80, "="))
    print("=" * 80)
    eval_old = OfficialCompetitionEvaluator(old_gt_path)
    old_res = eval_old.evaluate(pred_dict, score_thresh=0.005)
    eval_old.print_report(old_res, "Old Noisy Ground Truth Evaluation")

    print("\n" + "=" * 80)
    print(" 2. EVALUATION AGAINST NEW QUALITY CLEAN GROUND TRUTH (1757 boxes, +234) ".center(80, "="))
    print("=" * 80)
    eval_clean = OfficialCompetitionEvaluator(clean_gt_path)
    clean_res = eval_clean.evaluate(pred_dict, score_thresh=0.005)
    eval_clean.print_report(clean_res, "New Quality Clean Ground Truth Evaluation")

    print("\n" + "=" * 80)
    print(" SUMMARY HEAD-TO-HEAD COMPARISON: OLD NOISY GT vs NEW QUALITY CLEAN GT ".center(80, "="))
    print("=" * 80)
    print(f"{'Metric':<20} | {'Old Ground Truth':<18} | {'New Clean Quality GT':<20} | {'Delta':<10}")
    print("-" * 80)
    m_old = old_res["mAP_50_95"]
    m_new = clean_res["mAP_50_95"]
    d_m = m_new - m_old
    print(f"{'Official mAP@50:95':<20} | {m_old*100:6.2f}%            | {m_new*100:6.2f}%              | {'+' if d_m>=0 else ''}{d_m*100:5.2f}%")

    p50_o = old_res["mAP_50"]
    p50_n = clean_res["mAP_50"]
    d_50 = p50_n - p50_o
    print(f"{'Official mAP@50':<20} | {p50_o*100:6.2f}%            | {p50_n*100:6.2f}%              | {'+' if d_50>=0 else ''}{d_50*100:5.2f}%")

    p75_o = old_res["mAP_75"]
    p75_n = clean_res["mAP_75"]
    d_75 = p75_n - p75_o
    print(f"{'Official mAP@75':<20} | {p75_o*100:6.2f}%            | {p75_n*100:6.2f}%              | {'+' if d_75>=0 else ''}{d_75*100:5.2f}%")
    print("-" * 80)
    print("12 Categories AP@50:95 Breakdown:")
    for cid in range(NUM_CLASSES):
        cname = ID2NAME[cid]
        c_old = old_res["per_class"][cname]["ap50_95"]
        c_new = clean_res["per_class"][cname]["ap50_95"]
        delta = c_new - c_old
        sign = "+" if delta >= 0 else ""
        print(f"  [{cid:>2d}] {cname:<12s} | {c_old*100:6.2f}%            | {c_new*100:6.2f}%              | {sign}{delta*100:5.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
