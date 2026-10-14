#!/usr/bin/env python3
"""
AIC 2026: Comprehensive Quantitative Benchmark of All Recorded Viable Post-Processors
Tested on Sovereign Multimodal V3 (best_stg1.pth / best_stg2.pth).

Post-processing Techniques Evaluated:
1. Baseline: Raw Sovereign V3 Predictions (640x640 single-scale)
2. Waterline Reflection Rectifier (IEEE T-ITS WaterScenes / MODD2)
3. Closed-form Ground Coplanar False Positive Pruner (Physical 3D Depth)
4. 3D Metric Physical Scale Plausibility Filter (Camera Intrinsics H_3D = h * Z / f)
5. Intra-Class High-IoU Box Deduplication / Soft-WBF (IoU >= 0.82)
6. All-Combined Sovereign Master Post-Processing Pipeline
"""

import os
import sys
import copy
import json
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = "/root/autodl-tmp/AIC"
DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
sys.path.insert(0, os.path.join(ROOT, "scripts/lib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, DEIMV2_ROOT)
os.chdir(DEIMV2_ROOT)

from engine.core import YAMLConfig
from engine.data.dataset.coco_eval import CocoEvaluator
from depth_coplanar_filter import prune_coplanar_false_positives

CLASSES = [
    'person', 'boat', 'animal', 'seat', 'sign', 'bicycle',
    'car', 'ball', 'light', 'garbage can', 'uav', 'tricycle'
]

def run_eval(predictions_dict, base_ds, desc="Evaluation"):
    evaluator = CocoEvaluator(base_ds, ['bbox'])
    evaluator.cleanup()
    evaluator.update(predictions_dict)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    print(f"\n==================== {desc} ====================")
    evaluator.summarize()
    stats = evaluator.coco_eval['bbox'].stats
    return stats

# -------------------------------------------------------------------------
# 1. Waterline Reflection Rectifier (IEEE T-ITS WaterScenes)
# -------------------------------------------------------------------------
def rectify_boat_waterline(boxes, scores, labels, beta=0.96, max_aspect=1.15):
    """
    Vessels on water have horizontal extent. If H > max_aspect * W,
    the bottom of the box has captured water reflection ripples.
    Also applies a slight bottom retraction (beta=0.96) to eat away
    water reflection at the hull waterline.
    """
    rect_boxes = boxes.clone()
    rect_scores = scores.clone()
    for i in range(len(labels)):
        if labels[i] == 1: # boat
            b = rect_boxes[i]
            w = b[2] - b[0]
            h = b[3] - b[1]
            if h > max_aspect * w:
                # Clamp bottom to physical waterline
                rect_boxes[i, 3] = b[1] + w * 1.05
            else:
                # Slight waterline retraction
                rect_boxes[i, 3] = b[1] + h * beta
    return rect_boxes, rect_scores

# -------------------------------------------------------------------------
# 2. 3D Metric Physical Scale Plausibility Filter
# -------------------------------------------------------------------------
def filter_3d_metric_plausibility(boxes, scores, labels, depth_map, focal_length=640.0):
    """
    Computes physical real-world height H_3D = (h_pixel / focal_length) * Z_meters.
    Suppresses scores for detections with physically impossible dimensions.
    """
    if depth_map is None:
        return boxes, scores
    
    filt_scores = scores.clone()
    H, W = depth_map.shape[:2]
    
    for i in range(len(labels)):
        cid = int(labels[i])
        b = boxes[i]
        x1, y1, x2, y2 = [int(v.item()) for v in b]
        x1, x2 = max(0, x1), min(W - 1, x2)
        y1, y2 = max(0, y1), min(H - 1, y2)
        h_px = y2 - y1
        if h_px < 6:
            continue
            
        # Sample center depth
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        z_sample = depth_map[max(0, cy - 2):min(H, cy + 3), max(0, cx - 2):min(W, cx + 3)]
        valid_z = z_sample[(z_sample > 500.0) & (z_sample < 80000.0)]
        if len(valid_z) == 0:
            continue
            
        z_m = float(np.median(valid_z)) / 1000.0 # convert mm to meters
        h_3d = (h_px / focal_length) * z_m
        
        # Physical constraints:
        # person: 0.5m ~ 2.4m
        if cid == 0 and (h_3d > 2.6 or h_3d < 0.4):
            filt_scores[i] = filt_scores[i] * 0.70
        # car: 0.8m ~ 3.5m
        elif cid == 6 and (h_3d > 3.8 or h_3d < 0.6):
            filt_scores[i] = filt_scores[i] * 0.70
        # animal: 0.2m ~ 2.2m
        elif cid == 2 and (h_3d > 2.5 or h_3d < 0.15):
            filt_scores[i] = filt_scores[i] * 0.70
            
    return boxes, filt_scores

# -------------------------------------------------------------------------
# 3. Intra-Class High-IoU Soft-WBF (DETR Query Deduplication)
# -------------------------------------------------------------------------
def intra_class_wbf(boxes, scores, labels, iou_thresh=0.82):
    """
    Merges duplicate detections from sibling queries in DETR when IoU >= iou_thresh.
    Computes score-weighted box coordinates to tighten center and extents.
    """
    if len(boxes) <= 1:
        return boxes, scores, labels
        
    keep_boxes = []
    keep_scores = []
    keep_labels = []
    
    unique_classes = torch.unique(labels)
    for c in unique_classes:
        c_mask = (labels == c)
        c_boxes = boxes[c_mask]
        c_scores = scores[c_mask]
        
        # Sort descending by score
        order = torch.argsort(c_scores, descending=True)
        c_boxes = c_boxes[order]
        c_scores = c_scores[order]
        
        used = torch.zeros(len(c_boxes), dtype=torch.bool)
        for i in range(len(c_boxes)):
            if used[i]:
                continue
            b1 = c_boxes[i]
            s1 = c_scores[i]
            
            # Find matching boxes with high IoU
            x1 = torch.max(b1[0], c_boxes[i+1:, 0])
            y1 = torch.max(b1[1], c_boxes[i+1:, 1])
            x2 = torch.min(b1[2], c_boxes[i+1:, 2])
            y2 = torch.min(b1[3], c_boxes[i+1:, 3])
            inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (c_boxes[i+1:, 2] - c_boxes[i+1:, 0]) * (c_boxes[i+1:, 3] - c_boxes[i+1:, 1])
            iou = inter / (area1 + area2 - inter + 1e-6)
            
            match_idx = torch.where(iou >= iou_thresh)[0] + (i + 1)
            
            if len(match_idx) > 0:
                # Merge with weighted average
                all_b = torch.cat([b1.unsqueeze(0), c_boxes[match_idx]], dim=0)
                all_s = torch.cat([s1.unsqueeze(0), c_scores[match_idx]], dim=0)
                merged_b = (all_b * all_s.unsqueeze(-1)).sum(dim=0) / all_s.sum()
                merged_s = all_s.max()
                used[match_idx] = True
                keep_boxes.append(merged_b)
                keep_scores.append(merged_s)
            else:
                keep_boxes.append(b1)
                keep_scores.append(s1)
            keep_labels.append(c)
            used[i] = True
            
    if len(keep_boxes) == 0:
        return boxes, scores, labels
        
    return torch.stack(keep_boxes), torch.stack(keep_scores), torch.tensor(keep_labels, dtype=labels.dtype)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/deimv2/deim_hgnetv2_x_aic_sovereign_v3_rtx6000.yml'
    ck_path = sys.argv[2] if len(sys.argv) > 2 else 'outputs/deim_hgnetv2_x_aic_sovereign_v3_rtx6000/best_stg1.pth'
    print(f"Loading config: {cfg_path}")
    print(f"Loading checkpoint: {ck_path}")
    
    cfg = YAMLConfig(cfg_path)
    ck = torch.load(ck_path, map_location='cpu')
    sd = ck['ema']['module'] if 'ema' in ck else ck['model']
    cfg.model.load_state_dict(sd, strict=False)
    model = cfg.model.cuda().eval()
    postprocessor = cfg.postprocessor
    val_loader = cfg.val_dataloader
    base_ds = val_loader.dataset.coco
    
    # Pre-load depth maps for all 200 validation images
    print("[*] Pre-caching uint16 metric depth maps for validation set...")
    cached_depths = {}
    for img_id in base_ds.getImgIds():
        file_name = base_ds.loadImgs([img_id])[0]["file_name"]
        stem = os.path.splitext(file_name)[0]
        depth_path = f"/dev/shm/aic_cache/depth_p0/{stem}.png"
        if os.path.exists(depth_path):
            cached_depths[img_id] = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        else:
            cached_depths[img_id] = None
    print(f"[*] Successfully cached {len([d for d in cached_depths.values() if d is not None])} metric depth maps.")

    print("\n[*] Gathering raw predictions on validation set (200 images)...")
    collected_preds = {}
    with torch.no_grad():
        for samples, targets in val_loader:
            samples = samples.cuda()
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            outputs = model(samples)
            orig_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
            results = postprocessor(outputs, orig_sizes, samples)
            for t, r in zip(targets, results):
                img_id = t['image_id'].item()
                collected_preds[img_id] = {
                    'boxes': r['boxes'].cpu(),
                    'scores': r['scores'].cpu(),
                    'labels': r['labels'].cpu()
                }

    # 1. Clean Baseline
    base_preds = {k: copy.deepcopy(v) for k, v in collected_preds.items()}
    run_eval(base_preds, base_ds, desc="1. Clean Sovereign V3 Baseline")

    # 2. Waterline Reflection Rectifier (Boat)
    waterline_preds = {}
    for img_id, p in collected_preds.items():
        r_b, r_s = rectify_boat_waterline(p['boxes'], p['scores'], p['labels'], beta=0.96, max_aspect=1.15)
        waterline_preds[img_id] = {'boxes': r_b, 'scores': r_s, 'labels': p['labels']}
    run_eval(waterline_preds, base_ds, desc="2. + Physical Waterline Reflection Rectifier (Boat)")

    # 3. Closed-form Ground Coplanar False Positive Pruner
    coplanar_preds = {}
    for img_id, p in collected_preds.items():
        depth_np = cached_depths.get(img_id)
        if depth_np is not None:
            keep = prune_coplanar_false_positives(p['boxes'], p['scores'], p['labels'], depth_np, rel_thresh=0.010)
            k_mask = keep if isinstance(keep, torch.Tensor) else torch.from_numpy(keep)
            coplanar_preds[img_id] = {
                'boxes': p['boxes'][k_mask],
                'scores': p['scores'][k_mask],
                'labels': p['labels'][k_mask]
            }
        else:
            coplanar_preds[img_id] = copy.deepcopy(p)
    run_eval(coplanar_preds, base_ds, desc="3. + Closed-form Ground Coplanar Pruner (3D Depth)")

    # 4. 3D Metric Physical Scale Plausibility Filter
    scale_preds = {}
    for img_id, p in collected_preds.items():
        depth_np = cached_depths.get(img_id)
        s_b, s_s = filter_3d_metric_plausibility(p['boxes'], p['scores'], p['labels'], depth_np, focal_length=640.0)
        scale_preds[img_id] = {'boxes': s_b, 'scores': s_s, 'labels': p['labels']}
    run_eval(scale_preds, base_ds, desc="4. + 3D Metric Physical Scale Plausibility Filter")

    # 5. Intra-Class High-IoU Box Deduplication / Soft-WBF
    wbf_preds = {}
    for img_id, p in collected_preds.items():
        w_b, w_s, w_l = intra_class_wbf(p['boxes'], p['scores'], p['labels'], iou_thresh=0.82)
        wbf_preds[img_id] = {'boxes': w_b, 'scores': w_s, 'labels': w_l}
    run_eval(wbf_preds, base_ds, desc="5. + Intra-Class Box Deduplication (WBF IoU=0.82)")

    # 6. Full Combined Sovereign Peak Master Pipeline
    master_preds = {}
    for img_id, p in collected_preds.items():
        # A. Waterline reflection
        m_b, m_s = rectify_boat_waterline(p['boxes'], p['scores'], p['labels'], beta=0.96, max_aspect=1.15)
        # B. 3D metric scale
        depth_np = cached_depths.get(img_id)
        m_b, m_s = filter_3d_metric_plausibility(m_b, m_s, p['labels'], depth_np, focal_length=640.0)
        # C. Ground coplanar pruner
        if depth_np is not None:
            keep = prune_coplanar_false_positives(m_b, m_s, p['labels'], depth_np, rel_thresh=0.010)
            k_mask = keep if isinstance(keep, torch.Tensor) else torch.from_numpy(keep)
            m_b = m_b[k_mask]
            m_s = m_s[k_mask]
            m_l = p['labels'][k_mask]
        else:
            m_l = p['labels']
        # D. Intra-class WBF
        m_b, m_s, m_l = intra_class_wbf(m_b, m_s, m_l, iou_thresh=0.82)
        master_preds[img_id] = {'boxes': m_b, 'scores': m_s, 'labels': m_l}
    run_eval(master_preds, base_ds, desc="6. 【ALL-COMBINED】 Sovereign Master Peak Pipeline")

if __name__ == '__main__':
    main()
