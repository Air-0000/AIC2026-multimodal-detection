#!/usr/bin/env python3
"""
Generate Sovereign V29 Cycle 5 Safe Seeded Fusion Submission Package:
- Defends against False Positive Inflation (Ghost Detections)
- Triple-Defense Gate:
  1. Leader Seed Threshold: cluster max score must be >= 0.26
  2. Bounded Boost: comb_score = min(leader_score + 0.15, prob_or)
  3. Argmax Geometry: preserve highest-confidence query's box coordinates
"""

import os
import sys
import math
import zipfile
import torch
import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
RAW_PT = os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/raw_detections_v29_c5_peak.pt")
OUT_ZIP = os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/submission_v29_c5_safe_seeded_fusion_thr35.zip")

RARE_CLASSES = {1, 7, 9, 10, 11} # boat, ball, garbage can, uav, tricycle

def to_xyxy(cx, cy, w, h):
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

def iou_xyxy(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0.0

raw = torch.load(RAW_PT, weights_only=False)

def build_safe_seeded_submission(raw_dict, seed_thr=0.26, noise_floor=0.10, max_boost=0.15, iou_merge=0.50):
    res = {}
    rescued_total = 0
    ghosts_eliminated = 0

    for stem, bxs in raw_dict.items():
        # 1. Non-UAV boxes use standard Cap60
        non_uav = [b for b in bxs if b[0] != 10 and (b[5] >= 0.45 if b[0] in RARE_CLASSES else b[5] >= 0.50)]
        
        # 2. UAV queries
        uav_cands = [b for b in bxs if b[0] == 10 and b[5] >= noise_floor]
        used = set()
        safe_uavs = []

        for i, b in enumerate(uav_cands):
            if i in used: continue
            cluster = [b]
            used.add(i)
            b_xy = to_xyxy(*b[1:5])
            for j in range(i+1, len(uav_cands)):
                if j in used: continue
                b2 = uav_cands[j]
                if iou_xyxy(b_xy, to_xyxy(*b2[1:5])) >= iou_merge:
                    cluster.append(b2)
                    used.add(j)
            
            cluster.sort(key=lambda x: -x[5])
            leader = cluster[0]
            max_sc = leader[5]

            # DEFENSE 1: Leader Seed Constraint
            if max_sc < seed_thr:
                ghosts_eliminated += 1
                continue

            # DEFENSE 2: Bounded Boost
            if len(cluster) == 1:
                comb_sc = max_sc
            else:
                prob_or = 1.0 - math.prod([1.0 - x[5] for x in cluster])
                comb_sc = min(max_sc + max_boost, prob_sc := min(0.98, prob_or))

            # DEFENSE 3: Argmax box coordinates
            cx, cy, w, h = leader[1:5]
            area = w * h
            gate = 0.35 if area < 0.003 else 0.45

            if comb_sc >= gate:
                safe_uavs.append((10, cx, cy, w, h, comb_sc))
                if max_sc < gate:
                    rescued_total += 1

        all_boxes = non_uav + safe_uavs
        all_boxes.sort(key=lambda x: -x[5])
        res[stem] = all_boxes[:60]

    return res, rescued_total, ghosts_eliminated

safe_dict, rescued, ghosts = build_safe_seeded_submission(raw)
print(f"Safe Seeded Fusion complete: {rescued} weak UAVs safely rescued, {ghosts} ghost clusters destroyed!")

# Package into zip
os.makedirs(os.path.dirname(OUT_ZIP), exist_ok=True)
with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for stem in sorted(safe_dict.keys()):
        boxes = safe_dict[stem]
        lines = []
        for b in boxes:
            cid, cx, cy, w, h, sc = b
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {sc:.6f}\r\n")
        zf.writestr(f"{stem}.txt", "".join(lines).encode('utf-8'))

print(f"[✔] Successfully exported: {OUT_ZIP} ({sum(len(v) for v in safe_dict.values())} boxes)")

# Check compliance
cmd = f"/home/constant/miniconda3/envs/deimv2/bin/python {ROOT}/scripts/eval/check_submission.py {OUT_ZIP} {ROOT}/dataset/test/AIC2026_PHASE_2_1000/visible"
os.system(cmd)
