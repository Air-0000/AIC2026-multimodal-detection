#!/usr/bin/env python3
"""
Comprehensive Validation Evaluation Showdown:
Testing All Solutions on the 200 Validation Images with Ground Truth Annotations!

Models:
- Sovereign V29 Cycle 5 Peak (All-time champion, mAP 0.5808)
- Sovereign V29 Cycle 8 Decoder Aligned (mAP 0.5805)

Variants per Model:
1. Raw Native (top 100 queries, no thresholding)
2. Cap60 (Official submission threshold: 0.45 rare, 0.50 common, max 60)
3. Fusion 1C (Intra-Model Probabilistic Query Fusion + Scale-Aware Dual-Gate)
4. Scale Dual-Gate (Scale-aware gating without fusion)

Reports:
- mAP (0.50:0.95), AP50, AP75, AP_s, AP_m, AP_l
- 12-Class Breakdown (especially UAV AP & AP50)
"""

import os
import sys
import copy
import json
import math
import argparse
import contextlib
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

_root = Path("/home/constant/CodeBuddy/Dev/AIC")
if not _root.exists():
    _root = Path("/home/ailab/ssd/bigdata/hejianzhong/AIC_V29")
if not _root.exists():
    _root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

sys.path.insert(0, str(_root / 'vendor/DEIMv2'))
sys.path.insert(0, str(_root / 'scripts/lib'))
os.chdir(str(_root / 'vendor/DEIMv2'))

from engine.core import YAMLConfig
from engine.data.dataset.coco_eval import CocoEvaluator

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}
RARE_CLASSES = {1, 7, 9, 10, 11}

def iou_xyxy(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

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
            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(val_dataloader):
                print(f"    [{batch_idx+1:3d}/{len(val_dataloader)}] batches processed")
    return results_raw

def apply_postprocess(raw_dict, mode="raw", uav_small_thr=0.35, iou_merge=0.50):
    processed = {}
    for img_id, d in raw_dict.items():
        labels = d["labels"].numpy()
        boxes = d["boxes"].numpy()
        scores = d["scores"].numpy()
        w_f, h_f = d["orig_w"], d["orig_h"]

        if mode == "raw":
            # top 100 queries by score
            idx = np.argsort(-scores)[:100]
            if len(idx) > 0:
                processed[img_id] = {
                    "labels": torch.from_numpy(labels[idx]).long(),
                    "boxes": torch.from_numpy(boxes[idx]).float(),
                    "scores": torch.from_numpy(scores[idx]).float(),
                }
            else:
                processed[img_id] = {
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                }
        elif mode == "cap60":
            # standard Cap60
            kept = []
            for lab, box, sc in zip(labels, boxes, scores):
                cid = int(lab)
                thr = 0.45 if cid in RARE_CLASSES else 0.50
                if sc >= thr:
                    kept.append((lab, box, sc))
            kept.sort(key=lambda x: -x[2])
            kept = kept[:60]
            if kept:
                processed[img_id] = {
                    "labels": torch.tensor([x[0] for x in kept], dtype=torch.int64),
                    "boxes": torch.tensor(np.array([x[1] for x in kept]), dtype=torch.float32),
                    "scores": torch.tensor([x[2] for x in kept], dtype=torch.float32),
                }
            else:
                processed[img_id] = {
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                }

        elif mode == "dual_gate":
            kept = []
            for lab, box, sc in zip(labels, boxes, scores):
                cid = int(lab)
                bw = (box[2] - box[0]) / w_f
                bh = (box[3] - box[1]) / h_f
                area = bw * bh
                if cid == 10:
                    thr = uav_small_thr if area < 0.001 else 0.45
                elif cid in RARE_CLASSES:
                    thr = 0.45
                else:
                    thr = 0.50
                if sc >= thr:
                    kept.append((lab, box, sc))
            kept.sort(key=lambda x: -x[2])
            kept = kept[:60]
            if kept:
                processed[img_id] = {
                    "labels": torch.tensor([x[0] for x in kept], dtype=torch.int64),
                    "boxes": torch.tensor(np.array([x[1] for x in kept]), dtype=torch.float32),
                    "scores": torch.tensor([x[2] for x in kept], dtype=torch.float32),
                }
            else:
                processed[img_id] = {
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                }

        elif mode == "fusion_1c":
            # Non-UAV boxes
            kept = []
            for lab, box, sc in zip(labels, boxes, scores):
                cid = int(lab)
                if cid != 10:
                    thr = 0.45 if cid in RARE_CLASSES else 0.50
                    if sc >= thr:
                        kept.append((lab, box, sc))

            # Cluster UAVs
            uav_cands = [(lab, box, sc) for lab, box, sc in zip(labels, boxes, scores) if lab == 10 and sc >= 0.08]
            merged_uavs = []
            used = set()
            for i, u1 in enumerate(uav_cands):
                if i in used: continue
                cluster = [u1]
                used.add(i)
                b1_xy = [u1[1][0]/w_f, u1[1][1]/h_f, u1[1][2]/w_f, u1[1][3]/h_f]
                for j in range(i+1, len(uav_cands)):
                    if j in used: continue
                    u2 = uav_cands[j]
                    b2_xy = [u2[1][0]/w_f, u2[1][1]/h_f, u2[1][2]/w_f, u2[1][3]/h_f]
                    if iou_xyxy(b1_xy, b2_xy) >= iou_merge:
                        cluster.append(u2)
                        used.add(j)

                # Weight average box
                total_w = sum(x[2] for x in cluster)
                avg_box = np.sum([x[1] * x[2] for x in cluster], axis=0) / total_w
                # Probabilistic OR
                comb_sc = 1.0 - math.prod([1.0 - x[2] for x in cluster])
                comb_sc = min(0.98, max(cluster[0][2], comb_sc))

                bw = (avg_box[2] - avg_box[0]) / w_f
                bh = (avg_box[3] - avg_box[1]) / h_f
                area = bw * bh
                gate = uav_small_thr if area < 0.001 else 0.45
                if comb_sc >= gate:
                    merged_uavs.append((10, avg_box, comb_sc))

            all_kept = kept + merged_uavs
            all_kept.sort(key=lambda x: -x[2])
            all_kept = all_kept[:60]
            if all_kept:
                processed[img_id] = {
                    "labels": torch.tensor([x[0] for x in all_kept], dtype=torch.int64),
                    "boxes": torch.tensor(np.array([x[1] for x in all_kept]), dtype=torch.float32),
                    "scores": torch.tensor([x[2] for x in all_kept], dtype=torch.float32),
                }
            else:
                processed[img_id] = {
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                }

    return processed

def evaluate_predictions(coco_gt, processed_preds):
    evaluator = CocoEvaluator(coco_gt, ["bbox"])
    evaluator.update(processed_preds)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            evaluator.summarize()
            
    # Extract metrics from coco_eval
    ce = evaluator.coco_eval["bbox"]
    stats = ce.stats
    
    # per-class AP
    prec = ce.eval["precision"]
    cat_ids = list(ce.params.catIds)
    per_class = {}
    for k, cid in enumerate(cat_ids[:prec.shape[2]]):
        p_all = prec[:, :, k, 0, -1]
        p_all = p_all[p_all > -1]
        p50 = prec[0, :, k, 0, -1]
        p50 = p50[p50 > -1]
        ap = float(np.mean(p_all)) if p_all.size else 0.0
        ap50 = float(np.mean(p50)) if p50.size else 0.0
        per_class[cid] = (ap, ap50)

    metrics = {
        "mAP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_s": float(stats[3]),
        "AP_m": float(stats[4]),
        "AP_l": float(stats[5]),
        "per_class": per_class,
    }
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-c5", default="configs/deimv2/sovereign_v29_highres2560_anneal.yml")
    parser.add_argument("--ckpt-c5", default="outputs/sovereign_v29_cycle5_stage2_anneal_2560/best_stg2.pth")
    parser.add_argument("--config-c8", default="configs/deimv2/sovereign_v29_cycle8_decoder_head_align.yml")
    parser.add_argument("--ckpt-c8", default="outputs/sovereign_v29_cycle8_decoder_head_align/best_stg2.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-json", default="outputs/val_uav_showdown_results.json")
    args = parser.parse_args()

    print("=" * 105)
    print(" Sovereign V29 Validation Ground-Truth Showdown ".center(105, "="))
    print("=" * 105)

    cfg = YAMLConfig(args.config_c5)
    val_dataloader = cfg.val_dataloader
    coco_gt = val_dataloader.dataset.coco

    cache_c5 = _root / "outputs/val_c5_raw.pt"
    cache_c8 = _root / "outputs/val_c8_raw.pt"
    cache_c5.parent.mkdir(parents=True, exist_ok=True)

    # 1. Model C5 Inference
    if cache_c5.exists():
        print(f"\n[1/2] Loading cached C5 raw validation detections from {cache_c5}...")
        c5_raw = torch.load(cache_c5, map_location="cpu", weights_only=False)
    else:
        print(f"\n[1/2] Loading Sovereign V29 Cycle 5 Peak ({args.ckpt_c5})...")
        model_c5 = cfg.model.to(args.device)
        postprocessor = cfg.postprocessor
        ck5 = torch.load(args.ckpt_c5, map_location=args.device, weights_only=False)
        st5 = ck5["ema"]["module"] if ("ema" in ck5 and "module" in ck5["ema"]) else (ck5["model"] if "model" in ck5 else ck5)
        model_c5.load_state_dict(st5, strict=False)
        c5_raw = run_val_inference(model_c5, postprocessor, val_dataloader, device=args.device)
        torch.save(c5_raw, cache_c5)
        del model_c5
        torch.cuda.empty_cache()

    # 2. Model C8 Inference
    if cache_c8.exists():
        print(f"\n[2/2] Loading cached C8 raw validation detections from {cache_c8}...")
        c8_raw = torch.load(cache_c8, map_location="cpu", weights_only=False)
    else:
        print(f"\n[2/2] Loading Sovereign V29 Cycle 8 Aligned ({args.ckpt_c8})...")
        cfg8 = YAMLConfig(args.config_c8)
        model_c8 = cfg8.model.to(args.device)
        postprocessor8 = cfg8.postprocessor
        ck8 = torch.load(args.ckpt_c8, map_location=args.device, weights_only=False)
        st8 = ck8["ema"]["module"] if ("ema" in ck8 and "module" in ck8["ema"]) else (ck8["model"] if "model" in ck8 else ck8)
        model_c8.load_state_dict(st8, strict=False)
        c8_raw = run_val_inference(model_c8, postprocessor8, val_dataloader, device=args.device)
        torch.save(c8_raw, cache_c8)
        del model_c8
        torch.cuda.empty_cache()

    # 3. Benchmark All Variants
    variants = [
        ("C5 Raw Native (Top 100)", c5_raw, "raw", 0.35),
        ("C5 Cap60 (Standard)", c5_raw, "cap60", 0.45),
        ("C5 Dual-Gate (Thr=0.35)", c5_raw, "dual_gate", 0.35),
        ("★ C5 Fusion 1C (Thr=0.35)", c5_raw, "fusion_1c", 0.35),
        ("★ C5 Fusion 1C (Thr=0.30)", c5_raw, "fusion_1c", 0.30),
        ("★ C5 Fusion 1C (Thr=0.40)", c5_raw, "fusion_1c", 0.40),
        ("C8 Raw Native (Top 100)", c8_raw, "raw", 0.35),
        ("C8 Cap60 (Standard)", c8_raw, "cap60", 0.45),
        ("C8 Dual-Gate (Thr=0.35)", c8_raw, "dual_gate", 0.35),
        ("★ C8 Fusion 1C (Thr=0.35)", c8_raw, "fusion_1c", 0.35),
        ("★ C8 Fusion 1C (Thr=0.30)", c8_raw, "fusion_1c", 0.30),
        ("★ C8 Fusion 1C (Thr=0.40)", c8_raw, "fusion_1c", 0.40),
    ]

    print("\n[*] Evaluating COCO metrics across all variants...")
    results = {}
    for name, r_dict, mode, thr in variants:
        preds = apply_postprocess(r_dict, mode=mode, uav_small_thr=thr)
        m = evaluate_predictions(coco_gt, preds)
        results[name] = m
        print(f"    Done: {name:<30} -> mAP: {m['mAP']*100:.2f}%, UAV AP: {m['per_class'][10][0]*100:.2f}%")

    # 4. Print Overall Comparison Table
    print("\n" + "=" * 115)
    print(" GROUND TRUTH VALIDATION SET EVALUATION (200 IMAGES, mAP @ IoU=0.50:0.95) ".center(115, "="))
    print("=" * 115)
    header = f"{'Strategy / Variant':<32} | {'mAP':<7} | {'AP50':<7} | {'AP75':<7} | {'AP_small':<9} | {'AP_med':<7} | {'UAV AP50':<9} | {'UAV AP':<7}"
    print(header)
    print("-" * 115)
    for name in results:
        m = results[name]
        uav_ap, uav_ap50 = m["per_class"][10]
        print(f"{name:<32} | {m['mAP']*100:>5.2f}% | {m['AP50']*100:>5.2f}% | {m['AP75']*100:>5.2f}% | {m['AP_s']*100:>7.2f}% | {m['AP_m']*100:>5.2f}% | {uav_ap50*100:>7.2f}% | {uav_ap*100:>5.2f}%")
    print("=" * 115)

    # 5. Print 12-Class Detail Table
    print("\n" + "=" * 135)
    print(" 12-CLASS PER-CATEGORY AP COMPARISON ON VALIDATION SET ".center(135, "="))
    print("=" * 135)
    col_names = ["C5 Native", "C5 Cap60", "C5 F1C-35", "C8 Native", "C8 Cap60", "C8 F1C-35"]
    mapped_keys = [
        "C5 Raw Native (Top 100)",
        "C5 Cap60 (Standard)",
        "★ C5 Fusion 1C (Thr=0.35)",
        "C8 Raw Native (Top 100)",
        "C8 Cap60 (Standard)",
        "★ C8 Fusion 1C (Thr=0.35)",
    ]
    hdr = f"{'CID':<3} | {'Class Name':<12}" + "".join([f" | {k:<15}" for k in col_names])
    print(hdr)
    print("-" * 135)
    for cid in range(12):
        cname = CLASS_NAMES[cid]
        vals = []
        for mk in mapped_keys:
            ap_val = results[mk]["per_class"][cid][0] * 100.0
            vals.append(f"{ap_val:>6.2f}%")
        print(f"{cid:<3} | {cname:<12}" + "".join([f" | {v:<15}" for v in vals]))
    print("=" * 135)

    # Save to JSON
    json_path = _root / args.out_json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            "mAP": v["mAP"], "AP50": v["AP50"], "AP75": v["AP75"],
            "AP_s": v["AP_s"], "AP_m": v["AP_m"], "AP_l": v["AP_l"],
            "per_class": {str(cid): list(vals) for cid, vals in v["per_class"].items()}
        }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[✓] Results successfully saved to: {json_path}")

if __name__ == "__main__":
    main()
