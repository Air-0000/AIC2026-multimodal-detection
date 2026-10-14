#!/usr/bin/env python3
"""
Deep Confidence Score & Calibration Analysis: Sovereign V30 (Control) vs Sovereign V31 (Co-ATSS Test)
Directly tests whether Co-ATSS One-to-Many Auxiliary Supervision resolves the DETR low-confidence pathology (0.2~0.4).
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

# Set roots
_root = Path("/home/ailab/ssd/bigdata/hejianzhong/AIC_V29")
if not _root.exists():
    _root = Path("/home/constant/CodeBuddy/Dev/AIC")

sys.path.insert(0, str(_root / 'vendor/DEIMv2'))
sys.path.insert(0, str(_root / 'scripts/lib'))
os.chdir(str(_root / 'vendor/DEIMv2'))

from engine.core import YAMLConfig
from engine.data.dataset.coco_eval import CocoEvaluator

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

def box_iou_xyxy(b1, b2):
    # b1: [N, 4], b2: [M, 4]
    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)

def run_val_inference(model, postprocessor, val_dataloader, device="cuda:0"):
    results_raw = {}
    model.eval()
    print(f"[*] Running inference on {len(val_dataloader.dataset)} validation images...")
    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(val_dataloader):
            samples = samples.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            processed = postprocessor(outputs, orig_target_sizes, samples)
            for p, target in zip(processed, targets):
                img_id = int(target["image_id"].item())
                orig_w, orig_h = target["orig_size"].cpu().numpy()
                results_raw[img_id] = {
                    "labels": p["labels"].cpu(),
                    "boxes": p["boxes"].cpu(), # [N, 4] in [x1, y1, x2, y2]
                    "scores": p["scores"].cpu(),
                    "orig_w": float(orig_w),
                    "orig_h": float(orig_h),
                }
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(val_dataloader):
                print(f"    [{batch_idx+1:3d}/{len(val_dataloader)}] batches processed")
    return results_raw

def evaluate_coco(coco_gt, raw_dict):
    evaluator = CocoEvaluator(coco_gt, iou_types=['bbox'])
    res = {}
    for img_id, d in raw_dict.items():
        res[img_id] = {
            "labels": d["labels"],
            "boxes": d["boxes"],
            "scores": d["scores"],
        }
    evaluator.update(res)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.coco_eval["bbox"].stats
    
    # per-class
    per_class = {}
    coco_eval = evaluator.coco_eval["bbox"]
    cat_ids = coco_eval.params.catIds
    for cid in cat_ids:
        c_idx = cid - 1 if min(cat_ids) == 1 else cid
        # class AP
        prec = coco_eval.eval["precision"]
        # shape: [T, R, K, A, M] -> T=10 IoU, R=101 recall, K=classes, A=area(0=all), M=maxDets(2=100)
        p_c = prec[:, :, c_idx, 0, 2]
        p_c_50 = prec[0, :, c_idx, 0, 2]
        ap = np.mean(p_c[p_c > -1]) if np.any(p_c > -1) else 0.0
        ap50 = np.mean(p_c_50[p_c_50 > -1]) if np.any(p_c_50 > -1) else 0.0
        per_class[c_idx] = (float(ap), float(ap50))
        
    return {
        "mAP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_s": float(stats[3]),
        "AP_m": float(stats[4]),
        "AP_l": float(stats[5]),
        "per_class": per_class,
    }

def analyze_confidence_calibration(coco_gt, raw_dict, iou_threshold=0.5):
    """
    Match predictions to GT to inspect:
    - TP scores (true objects)
    - FP scores (background / hallucinations)
    - Per-class score distribution
    """
    matched_tp_scores = []
    matched_tp_by_class = defaultdict(list)
    fp_scores = []
    fp_by_class = defaultdict(list)
    gt_total_by_class = defaultdict(int)
    
    # Iterate each image
    for img_id, pred in raw_dict.items():
        p_boxes = pred["boxes"].numpy()
        p_labels = pred["labels"].numpy()
        p_scores = pred["scores"].numpy()
        
        # Sort predictions by score descending
        sort_idx = np.argsort(-p_scores)
        p_boxes = p_boxes[sort_idx]
        p_labels = p_labels[sort_idx]
        p_scores = p_scores[sort_idx]
        
        # Get GT annotations
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            if ann.get("iscrowd", 0): continue
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann["category_id"] if min(coco_gt.getCatIds()) == 0 else ann["category_id"] - 1)
            gt_total_by_class[gt_labels[-1]] += 1
            
        if len(gt_boxes) == 0:
            for s, l in zip(p_scores, p_labels):
                fp_scores.append(s)
                fp_by_class[l].append(s)
            continue
            
        gt_boxes = np.array(gt_boxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        
        if len(p_boxes) > 0:
            ious = box_iou_xyxy(p_boxes, gt_boxes) # [N_p, N_gt]
            for i in range(len(p_boxes)):
                plab = p_labels[i]
                psc = p_scores[i]
                
                # candidate GTs of same class that are not matched yet
                candidate_gt_idx = np.where((gt_labels == plab) & (~gt_matched))[0]
                if len(candidate_gt_idx) > 0:
                    cand_ious = ious[i, candidate_gt_idx]
                    best_cand = np.argmax(cand_ious)
                    best_iou = cand_ious[best_cand]
                    if best_iou >= iou_threshold:
                        matched_gt_idx = candidate_gt_idx[best_cand]
                        gt_matched[matched_gt_idx] = True
                        matched_tp_scores.append(psc)
                        matched_tp_by_class[plab].append(psc)
                        continue
                # Not matched to GT -> False Positive
                fp_scores.append(psc)
                fp_by_class[plab].append(psc)

    matched_tp_scores = np.array(matched_tp_scores) if len(matched_tp_scores) > 0 else np.array([0.0])
    fp_scores = np.array(fp_scores) if len(fp_scores) > 0 else np.array([0.0])
    
    # Calculate score bins for TPs: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
    tp_bins = {
        "[0.0, 0.20) Low": np.mean((matched_tp_scores >= 0.0) & (matched_tp_scores < 0.20)) * 100,
        "[0.20, 0.40) Mod-Low": np.mean((matched_tp_scores >= 0.20) & (matched_tp_scores < 0.40)) * 100,
        "[0.40, 0.60) Medium": np.mean((matched_tp_scores >= 0.40) & (matched_tp_scores < 0.60)) * 100,
        "[0.60, 0.80) Confident": np.mean((matched_tp_scores >= 0.60) & (matched_tp_scores < 0.80)) * 100,
        "[0.80, 1.00] High": np.mean((matched_tp_scores >= 0.80) & (matched_tp_scores <= 1.00)) * 100,
    }
    
    stats = {
        "tp_count": len(matched_tp_scores),
        "tp_mean": float(np.mean(matched_tp_scores)),
        "tp_median": float(np.median(matched_tp_scores)),
        "tp_p25": float(np.percentile(matched_tp_scores, 25)),
        "tp_p75": float(np.percentile(matched_tp_scores, 75)),
        "tp_p90": float(np.percentile(matched_tp_scores, 90)),
        "tp_bins": tp_bins,
        "fp_count": len(fp_scores),
        "fp_mean": float(np.mean(fp_scores)),
        "fp_median": float(np.median(fp_scores)),
        "by_class": {},
    }
    
    for c in range(12):
        c_tps = np.array(matched_tp_by_class[c]) if len(matched_tp_by_class[c]) > 0 else np.array([0.0])
        stats["by_class"][c] = {
            "name": CLASS_NAMES[c],
            "gt_total": gt_total_by_class[c],
            "tp_matched": len(matched_tp_by_class[c]),
            "recall50": (len(matched_tp_by_class[c]) / gt_total_by_class[c] * 100) if gt_total_by_class[c] > 0 else 0.0,
            "mean_score": float(np.mean(c_tps)),
            "median_score": float(np.median(c_tps)),
            "pct_gt_04": float(np.mean(c_tps >= 0.40) * 100) if len(matched_tp_by_class[c]) > 0 else 0.0,
            "pct_gt_06": float(np.mean(c_tps >= 0.60) * 100) if len(matched_tp_by_class[c]) > 0 else 0.0,
        }
        
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-v30", default="configs/deimv2/sovereign_v30_cssp_v29peak_2560.yml")
    parser.add_argument("--ckpt-v30", default="outputs/sovereign_v30_cssp_v29peak_2560/best_stg2.pth")
    parser.add_argument("--config-v31", default="configs/deimv2/sovereign_v31_ablation_probe.yml")
    parser.add_argument("--ckpt-v31", default="outputs/sovereign_v31_ablation_probe_2560/best_stg2.pth")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print("=" * 100)
    print(" Sovereign V30 vs V31: Confidence Distribution & Calibration Showdown ".center(100, "="))
    print("=" * 100)
    print(f"Device: {args.device}")

    # Build validation dataloader from V30 config
    cfg30 = YAMLConfig(args.config_v30)
    val_dataloader = cfg30.val_dataloader
    coco_gt = val_dataloader.dataset.coco

    # 1. Evaluate V30 Baseline
    print(f"\n[1/2] Loading Sovereign V30 Baseline ({args.ckpt_v30})...")
    model_v30 = cfg30.model.to(args.device)
    post_v30 = cfg30.postprocessor
    ck30 = torch.load(args.ckpt_v30, map_location=args.device, weights_only=False)
    st30 = ck30["ema"]["module"] if ("ema" in ck30 and "module" in ck30["ema"]) else (ck30["model"] if "model" in ck30 else ck30)
    model_v30.load_state_dict(st30, strict=False)
    v30_raw = run_val_inference(model_v30, post_v30, val_dataloader, device=args.device)
    del model_v30
    torch.cuda.empty_cache()

    # 2. Evaluate V31 Co-ATSS
    print(f"\n[2/2] Loading Sovereign V31 Co-ATSS Test ({args.ckpt_v31})...")
    cfg31 = YAMLConfig(args.config_v31)
    model_v31 = cfg31.model.to(args.device)
    post_v31 = cfg31.postprocessor
    ck31 = torch.load(args.ckpt_v31, map_location=args.device, weights_only=False)
    st31 = ck31["ema"]["module"] if ("ema" in ck31 and "module" in ck31["ema"]) else (ck31["model"] if "model" in ck31 else ck31)
    model_v31.load_state_dict(st31, strict=False)
    v31_raw = run_val_inference(model_v31, post_v31, val_dataloader, device=args.device)
    del model_v31
    torch.cuda.empty_cache()

    # Evaluate COCO
    print("\n[*] Evaluating standard COCO metrics...")
    coco_m30 = evaluate_coco(coco_gt, v30_raw)
    coco_m31 = evaluate_coco(coco_gt, v31_raw)

    # Analyze Confidence Calibration
    print("\n[*] Performing GT matching and confidence calibration analysis...")
    calib_v30 = analyze_confidence_calibration(coco_gt, v30_raw, iou_threshold=0.5)
    calib_v31 = analyze_confidence_calibration(coco_gt, v31_raw, iou_threshold=0.5)

    # ================= OUTPUT TABLES =================
    print("\n" + "=" * 100)
    print(" 1. DETECTION PERFORMANCE SHOWDOWN (200 Validation Images, 2.5K Resolution) ".center(100, "="))
    print("=" * 100)
    print(f"{'Metric':<18} | {'V30 Baseline (1-to-1)':<24} | {'V31 Co-ATSS (One-to-Many)':<26} | {'Delta':<12}")
    print("-" * 88)
    for k in ["mAP", "AP50", "AP75", "AP_s", "AP_m", "AP_l"]:
        val30 = coco_m30[k]
        val31 = coco_m31[k]
        delta = val31 - val30
        print(f"{k:<18} | {val30*100:>21.2f}% | {val31*100:>23.2f}% | {delta*100:>+10.2f}%")

    print("\n" + "=" * 100)
    print(" 2. TRUE POSITIVE QUERY CONFIDENCE CALIBRATION (Matched to Ground-Truth IoU>=0.5) ".center(100, "="))
    print("=" * 100)
    print(f"{'Confidence Metric':<24} | {'V30 Baseline':<20} | {'V31 Co-ATSS':<20} | {'Shift / Gain':<14}")
    print("-" * 88)
    print(f"{'Matched TPs Count':<24} | {calib_v30['tp_count']:>17d}  | {calib_v31['tp_count']:>17d}  | {calib_v31['tp_count']-calib_v30['tp_count']:>+12d}")
    print(f"{'Mean Score (True Positive)':<24} | {calib_v30['tp_mean']:>20.4f} | {calib_v31['tp_mean']:>20.4f} | {calib_v31['tp_mean']-calib_v30['tp_mean']:>+14.4f}")
    print(f"{'Median Score (TP)':<24} | {calib_v30['tp_median']:>20.4f} | {calib_v31['tp_median']:>20.4f} | {calib_v31['tp_median']-calib_v30['tp_median']:>+14.4f}")
    print(f"{'25% Percentile Score':<24} | {calib_v30['tp_p25']:>20.4f} | {calib_v31['tp_p25']:>20.4f} | {calib_v31['tp_p25']-calib_v30['tp_p25']:>+14.4f}")
    print(f"{'75% Percentile Score':<24} | {calib_v30['tp_p75']:>20.4f} | {calib_v31['tp_p75']:>20.4f} | {calib_v31['tp_p75']-calib_v30['tp_p75']:>+14.4f}")
    print(f"{'90% Percentile Score':<24} | {calib_v30['tp_p90']:>20.4f} | {calib_v31['tp_p90']:>20.4f} | {calib_v31['tp_p90']-calib_v30['tp_p90']:>+14.4f}")
    print("-" * 88)
    print(f"{'Background / FP Mean Score':<24} | {calib_v30['fp_mean']:>20.4f} | {calib_v31['fp_mean']:>20.4f} | {calib_v31['fp_mean']-calib_v30['fp_mean']:>+14.4f}")

    print("\n" + "=" * 100)
    print(" 3. TRUE POSITIVE SCORE BINS DISTRIBUTION (% of Real Objects) ".center(100, "="))
    print("=" * 100)
    print(f"{'Confidence Score Interval':<28} | {'V30 Baseline':<18} | {'V31 Co-ATSS':<18} | {'Shift':<12}")
    print("-" * 84)
    for b_name in calib_v30["tp_bins"].keys():
        b30 = calib_v30["tp_bins"][b_name]
        b31 = calib_v31["tp_bins"][b_name]
        d_b = b31 - b30
        print(f"{b_name:<28} | {b30:>15.1f}% | {b31:>15.1f}% | {d_b:>+10.1f}%")

    print("\n" + "=" * 100)
    print(" 4. PER-CLASS CONFIDENCE & RECALL BREAKDOWN ".center(100, "="))
    print("=" * 100)
    print(f"{'Class':<14} | {'V30 Mean Sc':<12} | {'V31 Mean Sc':<12} | {'Score Delta':<12} | {'V30 Recall':<11} | {'V31 Recall':<11} | {'V31 AP50':<10}")
    print("-" * 92)
    for c in range(12):
        c30 = calib_v30["by_class"][c]
        c31 = calib_v31["by_class"][c]
        d_sc = c31["mean_score"] - c30["mean_score"]
        c_name = CLASS_NAMES[c]
        ap50_31 = coco_m31["per_class"][c][1] * 100
        print(f"{c_name:<14} | {c30['mean_score']:>11.3f} | {c31['mean_score']:>11.3f} | {d_sc:>+11.3f} | {c30['recall50']:>10.1f}% | {c31['recall50']:>10.1f}% | {ap50_31:>9.1f}%")

    print("=" * 100)

if __name__ == "__main__":
    main()
