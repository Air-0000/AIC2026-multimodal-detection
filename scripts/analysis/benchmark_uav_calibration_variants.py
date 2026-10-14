#!/usr/bin/env python3
"""
Benchmark Scale-Aware Calibration & Intra-Model Query Fusion Variants
Comparing V29 Cycle 5 Peak against F5 Benchmark (58.105, 123 UAVs)
"""

import math
import torch
import os
import sys
sys.path.insert(0, '/home/constant/CodeBuddy/Dev/AIC')
from scripts.analysis.analyze_f5_overlap_sweep import parse_zip, to_xyxy, iou_xyxy, match_predictions

f5_zip = "work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip"
f5_data = parse_zip(f5_zip)
raw = torch.load("work_dir/submissions/v29_cycle5_peak/raw_detections_v29_c5_peak.pt", weights_only=False)

RARE_CLASSES = {1, 7, 9, 10, 11}

# Baseline Cap60
def get_baseline_cap60(raw_dict):
    res = {}
    for stem, bxs in raw_dict.items():
        filt = [b for b in bxs if (b[5] >= 0.45 if b[0] in RARE_CLASSES else b[5] >= 0.50)]
        filt.sort(key=lambda x: -x[5])
        res[stem] = filt[:60]
    return res

# Variant 1A: Scale-Aware Dual Thresholding for UAV
def get_variant_1a(raw_dict, uav_small_thr=0.30, area_thr=0.001):
    res = {}
    for stem, bxs in raw_dict.items():
        filt = []
        for b in bxs:
            cid, cx, cy, w, h, sc = b
            if cid == 10: # UAV
                t = uav_small_thr if (w * h < area_thr) else 0.45
            elif cid in RARE_CLASSES:
                t = 0.45
            else:
                t = 0.50
            if sc >= t:
                filt.append(b)
        filt.sort(key=lambda x: -x[5])
        res[stem] = filt[:60]
    return res

# Variant 1B: Intra-Model Query Fusion for UAV (probabilistic OR)
def get_variant_1b(raw_dict, iou_merge=0.50):
    res = {}
    for stem, bxs in raw_dict.items():
        non_uav = [b for b in bxs if b[0] != 10 and (b[5] >= 0.45 if b[0] in RARE_CLASSES else b[5] >= 0.50)]
        
        uavs = [b for b in bxs if b[0] == 10 and b[5] >= 0.08]
        merged_uavs = []
        used = set()
        for i, b in enumerate(uavs):
            if i in used: continue
            cluster = [b]
            used.add(i)
            b_xy = to_xyxy(*b[1:5])
            for j in range(i+1, len(uavs)):
                if j in used: continue
                b2 = uavs[j]
                if iou_xyxy(b_xy, to_xyxy(*b2[1:5])) >= iou_merge:
                    cluster.append(b2)
                    used.add(j)
            
            total_w = sum(x[5] for x in cluster)
            avg_cx = sum(x[1] * x[5] for x in cluster) / total_w
            avg_cy = sum(x[2] * x[5] for x in cluster) / total_w
            avg_w = sum(x[3] * x[5] for x in cluster) / total_w
            avg_h = sum(x[4] * x[5] for x in cluster) / total_w
            
            comb_sc = 1.0 - math.prod([1.0 - x[5] for x in cluster])
            comb_sc = min(0.98, max(cluster[0][5], comb_sc))
            
            if comb_sc >= 0.45:
                merged_uavs.append((10, avg_cx, avg_cy, avg_w, avg_h, comb_sc))
        
        all_b = non_uav + merged_uavs
        all_b.sort(key=lambda x: -x[5])
        res[stem] = all_b[:60]
    return res

# Variant 1C: Combined (Intra-Model Fusion + Scale-Aware Gating)
def get_variant_1c(raw_dict, uav_small_thr=0.35, iou_merge=0.50):
    res = {}
    for stem, bxs in raw_dict.items():
        non_uav = [b for b in bxs if b[0] != 10 and (b[5] >= 0.45 if b[0] in RARE_CLASSES else b[5] >= 0.50)]
        
        uavs = [b for b in bxs if b[0] == 10 and b[5] >= 0.08]
        merged_uavs = []
        used = set()
        for i, b in enumerate(uavs):
            if i in used: continue
            cluster = [b]
            used.add(i)
            b_xy = to_xyxy(*b[1:5])
            for j in range(i+1, len(uavs)):
                if j in used: continue
                b2 = uavs[j]
                if iou_xyxy(b_xy, to_xyxy(*b2[1:5])) >= iou_merge:
                    cluster.append(b2)
                    used.add(j)
            
            total_w = sum(x[5] for x in cluster)
            avg_cx = sum(x[1] * x[5] for x in cluster) / total_w
            avg_cy = sum(x[2] * x[5] for x in cluster) / total_w
            avg_w = sum(x[3] * x[5] for x in cluster) / total_w
            avg_h = sum(x[4] * x[5] for x in cluster) / total_w
            comb_sc = 1.0 - math.prod([1.0 - x[5] for x in cluster])
            comb_sc = min(0.98, max(cluster[0][5], comb_sc))
            
            area = avg_w * avg_h
            gate = uav_small_thr if area < 0.001 else 0.45
            if comb_sc >= gate:
                merged_uavs.append((10, avg_cx, avg_cy, avg_w, avg_h, comb_sc))
        
        all_b = non_uav + merged_uavs
        all_b.sort(key=lambda x: -x[5])
        res[stem] = all_b[:60]
    return res

tests = [
    ("Baseline Cap60", get_baseline_cap60(raw)),
    ("1A: Scale Dual-Gate (Thr=0.32)", get_variant_1a(raw, uav_small_thr=0.32)),
    ("1A: Scale Dual-Gate (Thr=0.30)", get_variant_1a(raw, uav_small_thr=0.30)),
    ("1B: Intra-Query Fusion (Thr=0.45)", get_variant_1b(raw)),
    ("1C: Fusion + Dual-Gate (Thr=0.38)", get_variant_1c(raw, uav_small_thr=0.38)),
    ("1C: Fusion + Dual-Gate (Thr=0.35)", get_variant_1c(raw, uav_small_thr=0.35)),
    ("1C: Fusion + Dual-Gate (Thr=0.30)", get_variant_1c(raw, uav_small_thr=0.30)),
]

print("=" * 110)
print(f"{'Variant':<35} | {'Total':<6} | {'Consensus':<9} | {'F5 Recall':<10} | {'Cand Prec':<10} | {'UAV Recall (out of 123)'}")
print("-" * 110)
for name, d in tests:
    m = match_predictions(f5_data, d, iou_thr=0.50)
    u_f5 = m["per_class_ref"][10]
    u_mat = m["per_class_matched"][10]
    u_tot = m["per_class_cand"][10]
    print(f"{name:<35} | {m['total_cand']:<6} | {m['matched_same']:<9} | {m['recall_ref']:>8.2f}% | {m['precision_cand']:>8.2f}% | {u_mat}/{u_f5} ({u_mat/u_f5*100:.1f}%) [Total: {u_tot}]")
print("=" * 110)
