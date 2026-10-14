"""
Comprehensive Benchmark of Literature Solutions for Sign & Boat:
1. Logit Calibration / Class-Decoupled Frequency Re-weighting (CVPR / NeurIPS)
2. Closed-form Ground Coplanar Pruner (Physical Depth Plane Fitting)
3. Water Reflection Aspect-Ratio Rectifier for Boat (IEEE T-ITS WaterScenes)
"""
import os
import sys
import copy
import json
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

# Frequency in train split (1800 imgs)
CLASS_COUNTS = {
    0: 5166, # person
    1: 126,  # boat (few-shot)
    2: 3168, # animal
    3: 657,  # seat
    4: 684,  # sign
    5: 666,  # bicycle
    6: 1404, # car
    7: 117,  # ball
    8: 1215, # light
    9: 315,  # garbage can
    10: 171, # uav
    11: 18   # tricycle
}

def logit_calibration(scores, labels, tau=0.3):
    """
    Logit calibration: \tilde{z}_c = z_c + tau * log(N_max / N_c)
    Boosts underrepresented classes (boat, sign, ball) in PR curve.
    """
    N_max = max(CLASS_COUNTS.values())
    calib_scores = scores.clone()
    eps = 1e-6
    # Logit transform
    logits = torch.log(scores.clamp(eps, 1.0 - eps) / (1.0 - scores.clamp(eps, 1.0 - eps)))
    for c in range(12):
        if c in [1, 4]: # Apply to boat and sign
            boost = tau * np.log(N_max / CLASS_COUNTS[c])
            mask = (labels == c)
            calib_scores[mask] = torch.sigmoid(logits[mask] + boost)
    return calib_scores

def rectify_boat_reflection(boxes, scores, labels):
    """
    Water Reflection Rectifier (WaterScenes / MODD2):
    Real vessels on water have aspect ratio W/H >= 0.8.
    If H > 1.2 * W, the bottom of the box has captured water reflection.
    Rectify by cutting off the lower reflection portion.
    """
    rect_boxes = boxes.clone()
    rect_scores = scores.clone()
    for i in range(len(labels)):
        if labels[i] == 1: # boat
            b = rect_boxes[i]
            w = b[2] - b[0]
            h = b[3] - b[1]
            if h > 1.2 * w:
                # Clamp bottom to water line
                rect_boxes[i, 3] = b[1] + w * 1.0
                rect_scores[i] = rect_scores[i] * 0.95 # slight score dampening
    return rect_boxes, rect_scores

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

    print("[*] Gathering raw predictions on validation set (200 images)...")
    collected_preds = {}
    depth_maps = {}

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

    # 1. Baseline
    base_preds = {}
    for img_id, p in collected_preds.items():
        base_preds[img_id] = {
            'boxes': p['boxes'],
            'scores': p['scores'],
            'labels': p['labels']
        }
    stats_base = run_eval(base_preds, base_ds, desc="1. Clean Sovereign V2 Baseline")

    # 2. Logit Calibration for boat & sign (tau=0.25)
    for tau in [0.15, 0.25, 0.35]:
        calib_preds = {}
        for img_id, p in collected_preds.items():
            new_scores = logit_calibration(p['scores'], p['labels'], tau=tau)
            calib_preds[img_id] = {
                'boxes': p['boxes'],
                'scores': new_scores,
                'labels': p['labels']
            }
        run_eval(calib_preds, base_ds, desc=f"2. Logit Calibration (tau={tau})")

    # 3. Water Reflection Rectifier for boat
    refl_preds = {}
    for img_id, p in collected_preds.items():
        r_boxes, r_scores = rectify_boat_reflection(p['boxes'], p['scores'], p['labels'])
        refl_preds[img_id] = {
            'boxes': r_boxes,
            'scores': r_scores,
            'labels': p['labels']
        }
    run_eval(refl_preds, base_ds, desc="3. Water Reflection Rectifier for Boat")

    # 4. Ground Coplanar Pruner
    coplanar_preds = {}
    for img_id, p in collected_preds.items():
        depth_path = f"/dev/shm/aic_cache/depth_p0/{img_id:08d}.png"
        depth_np = None
        if os.path.exists(depth_path):
            import cv2
            depth_np = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if depth_np is not None:
            depth_mm = depth_np.astype(np.float32)
            keep = prune_coplanar_false_positives(p['boxes'], p['scores'], p['labels'], depth_mm)
            k_mask = keep if isinstance(keep, torch.Tensor) else torch.from_numpy(keep)
            coplanar_preds[img_id] = {
                'boxes': p['boxes'][k_mask],
                'scores': p['scores'][k_mask],
                'labels': p['labels'][k_mask]
            }
        else:
            coplanar_preds[img_id] = p
    run_eval(coplanar_preds, base_ds, desc="4. Closed-form Ground Coplanar Pruner")

    # 5. Combined Best Pipeline
    combo_preds = {}
    for img_id, p in collected_preds.items():
        # A. Reflection rectifier
        r_boxes, r_scores = rectify_boat_reflection(p['boxes'], p['scores'], p['labels'])
        # B. Logit calib
        c_scores = logit_calibration(r_scores, p['labels'], tau=0.20)
        # C. Coplanar pruner
        depth_path = f"/dev/shm/aic_cache/depth_p0/{img_id:08d}.png"
        depth_np = None
        if os.path.exists(depth_path):
            import cv2
            depth_np = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_np is not None:
            depth_mm = depth_np.astype(np.float32)
            keep = prune_coplanar_false_positives(r_boxes, c_scores, p['labels'], depth_mm)
            k_mask = keep if isinstance(keep, torch.Tensor) else torch.from_numpy(keep)
            combo_preds[img_id] = {
                'boxes': r_boxes[k_mask],
                'scores': c_scores[k_mask],
                'labels': p['labels'][k_mask]
            }
        else:
            combo_preds[img_id] = {'boxes': r_boxes, 'scores': c_scores, 'labels': p['labels']}
    run_eval(combo_preds, base_ds, desc="5. [All-Combined] Sovereign V2 Peak Post-Processing")

if __name__ == "__main__":
    main()
