#!/usr/bin/env python3
"""
Generate Sovereign V29 Cycle 5 Physics-Verified Submission Package
=================================================================
Applies the 4 Rigid Real-World Physical Laws to filter and verify all UAV detections:
- Law 1: Gravity & Flight Corridor (y_bottom <= 0.85)
- Law 2: Newtonian Kinematics & Static Rigidity (suppress stationary ground poles across images)
- Law 3: Perspective Scale & Aerodynamic Symmetry (0.45 <= w/h <= 2.2, area <= 0.35%)
- Law 4: Thermodynamic Heat Dissipation Verification
"""

import os
import sys
import math
import zipfile
import torch
import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "scripts/lib"))
from physical_drone_verifier import detect_static_infrastructure, verify_physics

RAW_PT = os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/raw_detections_v29_c5_peak.pt")
OUT_ZIP = os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/submission_v29_c5_physics_verified.zip")
UNZIP_DIR = os.path.join(ROOT, "work_dir/submissions/v29_c5_physics_verified")

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

# Phase 1: Mine all UAV candidates across all 1000 images
candidate_pool = []
for stem, bxs in sorted(raw.items()):
    uav_queries = [b for b in bxs if b[0] == 10 and b[5] >= 0.15]
    used = set()
    for i, b in enumerate(uav_queries):
        if i in used: continue
        cluster = [b]
        used.add(i)
        b_xy = to_xyxy(*b[1:5])
        for j in range(i+1, len(uav_queries)):
            if j in used: continue
            b2 = uav_queries[j]
            if iou_xyxy(b_xy, to_xyxy(*b2[1:5])) >= 0.50:
                cluster.append(b2)
                used.add(j)
        
        cluster.sort(key=lambda x: -x[5])
        leader = cluster[0]
        max_sc = leader[5]
        
        if len(cluster) == 1:
            comb_sc = max_sc
        else:
            prob_or = 1.0 - math.prod([1.0 - x[5] for x in cluster])
            comb_sc = min(max_sc + 0.15, prob_or)
            
        candidate_pool.append({
            "stem": stem,
            "cx": leader[1], "cy": leader[2], "w": leader[3], "h": leader[4],
            "max_sc": max_sc, "comb_sc": comb_sc,
            "cluster_len": len(cluster)
        })

print(f"[*] Total mined candidate clusters: {len(candidate_pool)}")

# Phase 2: Detect Stationary Ground Infrastructure across images
static_indices = detect_static_infrastructure(candidate_pool, coord_tol=0.018)
print(f"[*] Stationary ground poles detected: {len(static_indices)} clusters across images")

# Phase 3: Apply 4 Rigid Physical Laws & Build Verified Submissions
final_submissions = {}
physics_killed_reasons = []
rescued_true_drones = 0
native_kept = 0

for idx, cand in enumerate(candidate_pool):
    stem = cand["stem"]
    cx, cy, w, h = cand["cx"], cand["cy"], cand["w"], cand["h"]
    comb_sc = cand["comb_sc"]
    max_sc = cand["max_sc"]
    is_static = (idx in static_indices)
    
    passed, reason = verify_physics(cand, is_static=is_static)
    
    if stem not in final_submissions:
        # Load non-UAV standard boxes for this image
        non_uav = [b for b in raw[stem] if b[0] != 10 and (b[5] >= 0.45 if b[0] in RARE_CLASSES else b[5] >= 0.50)]
        final_submissions[stem] = non_uav

    if max_sc >= 0.45:
        # Native high confidence: require passing at least ground cutoff (Law 1)
        if (cy + h/2.0) <= 0.88:
            final_submissions[stem].append((10, cx, cy, w, h, max_sc))
            native_kept += 1
        else:
            physics_killed_reasons.append((stem, cx, cy, "Native box touching ground/curb"))
    else:
        # Weak candidate: MUST pass all rigid physical laws!
        if passed and max_sc >= 0.26 and comb_sc >= 0.35:
            final_submissions[stem].append((10, cx, cy, w, h, comb_sc))
            rescued_true_drones += 1
        else:
            physics_killed_reasons.append((stem, cx, cy, reason))

print("\n=== Physics Engine Audit Results ===")
print(f"  Native high-confidence drones retained: {native_kept}")
print(f"  Genuinely physical weak drones rescued: {rescued_true_drones}")
print(f"  False positives killed by physics laws: {len(physics_killed_reasons)}")
print(f"  Total physics-verified UAV detections: {native_kept + rescued_true_drones}")

# Ensure all 1000 test stems are present
all_test_stems = sorted(raw.keys())
for s in all_test_stems:
    if s not in final_submissions:
        final_submissions[s] = [b for b in raw[s] if b[0] != 10 and (b[5] >= 0.45 if b[0] in RARE_CLASSES else b[5] >= 0.50)]
    final_submissions[s].sort(key=lambda x: -x[5])
    final_submissions[s] = final_submissions[s][:60]

# Package to zip & unzip
os.makedirs(os.path.dirname(OUT_ZIP), exist_ok=True)
os.makedirs(UNZIP_DIR, exist_ok=True)

with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for s in all_test_stems:
        boxes = final_submissions[s]
        lines = []
        for b in boxes:
            cid, cx, cy, w, h, sc = b
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {sc:.6f}\r\n")
        content_str = "".join(lines)
        zf.writestr(f"{s}.txt", content_str.encode('utf-8'))
        with open(os.path.join(UNZIP_DIR, f"{s}.txt"), "w", encoding="utf-8") as f_out:
            f_out.write(content_str)

tot_boxes = sum(len(v) for v in final_submissions.values())
print(f"\n[✔] Successfully generated Physics-Verified Package: {OUT_ZIP} ({tot_boxes} total boxes)")

# Check compliance
cmd = f"/home/constant/miniconda3/envs/deimv2/bin/python {ROOT}/scripts/eval/check_submission.py {UNZIP_DIR} {ROOT}/dataset/test/AIC2026_PHASE_2_1000/visible"
os.system(cmd)
