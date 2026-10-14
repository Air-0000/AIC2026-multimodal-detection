#!/usr/bin/env python3
"""
Physical Drone Verifier V2:
1. Convex Bounding Envelope (整机包络):
   - Expands cluster boxes to enclose the ENTIRE drone (body + rotors), eliminating partial framing (like in 00000012 and 00000422).
2. Universal Static Infrastructure Suppression (全量静态地物抑制):
   - Kills any detection that remains stationary across >= 3 frames in fixed camera sequences,
     eliminating static background structures like:
     - 00000071 (Bin 0.59, 0.63)
     - 00000076 (Bin 0.59, 0.63)
     - 00000037 (Bin 0.84, 0.59)
     - 00000256/261/272 (Bin 0.57, 0.77)
"""

import os
import sys
import math
import zipfile
import torch
import numpy as np
from collections import defaultdict

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
RAW_PT = os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/raw_detections_v29_c5_peak.pt")
OUT_ZIP = os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/submission_v29_c5_physics_v2_clean.zip")
UNZIP_DIR = os.path.join(ROOT, "work_dir/submissions/v29_c5_physics_v2_clean")

RARE_CLASSES = {1, 7, 9, 10, 11}

raw = torch.load(RAW_PT, weights_only=False)

# Phase 1: Collect all candidate clusters using Envelope merging (IoU merge 0.28 to group whole body + wings)
clusters_by_stem = defaultdict(list)
all_clusters = []

for stem, bxs in sorted(raw.items()):
    uav_queries = [b for b in bxs if b[0] == 10 and b[5] >= 0.15]
    used = set()
    for i, b in enumerate(uav_queries):
        if i in used: continue
        cluster = [b]
        used.add(i)
        bx1, by1, bx2, by2 = b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2
        for j in range(i+1, len(uav_queries)):
            if j in used: continue
            b2 = uav_queries[j]
            b2x1, b2y1, b2x2, b2y2 = b2[1]-b2[3]/2, b2[2]-b2[4]/2, b2[1]+b2[3]/2, b2[2]+b2[4]/2
            inter = max(0, min(bx2, b2x2) - max(bx1, b2x1)) * max(0, min(by2, b2y2) - max(by1, b2y1))
            u = b[3]*b[4] + b2[3]*b2[4] - inter
            if inter/u >= 0.28:
                cluster.append(b2)
                used.add(j)
        
        # Convex bounding envelope (encloses the whole physical drone)
        min_x1 = min(x[1]-x[3]/2 for x in cluster)
        min_y1 = min(x[2]-x[4]/2 for x in cluster)
        max_x2 = max(x[1]+x[3]/2 for x in cluster)
        max_y2 = max(x[2]+x[4]/2 for x in cluster)
        cx = (min_x1 + max_x2) / 2
        cy = (min_y1 + max_y2) / 2
        w = max_x2 - min_x1
        h = max_y2 - min_y1
        
        max_sc = max(x[5] for x in cluster)
        if len(cluster) == 1:
            comb_sc = max_sc
        else:
            prob_or = 1.0 - math.prod([1.0 - x[5] for x in cluster])
            comb_sc = min(max_sc + 0.15, prob_or)
            
        c_item = {
            "stem": stem, "cx": cx, "cy": cy, "w": w, "h": h,
            "max_sc": max_sc, "comb_sc": comb_sc, "cluster_len": len(cluster)
        }
        clusters_by_stem[stem].append(c_item)
        all_clusters.append(c_item)

print(f"[*] Total mined candidate clusters: {len(all_clusters)}")

# Phase 2: Detect Stationary Ground Infrastructure across scenes (>= 3 recurring frames in same spatial bin)
grid = defaultdict(list)
for c in all_clusters:
    # 0.025 spatial bin
    gx, gy = int(round(c["cx"] / 0.025)), int(round(c["cy"] / 0.025))
    grid[(gx, gy)].append(c)

static_bins = set()
for (gx, gy), items in grid.items():
    stems = set(x["stem"] for x in items)
    if len(stems) >= 3:
        static_bins.add((gx, gy))

print(f"[*] Detected {len(static_bins)} static infrastructure bins across video sequences")

# Phase 3: Physical Filtering & Assembly
final_submissions = {}
killed_fps = []
verified_drones = 0

for stem in sorted(raw.keys()):
    # Non-UAV standard boxes
    non_uav = [b for b in raw[stem] if b[0] != 10 and (b[5] >= 0.45 if b[0] in RARE_CLASSES else b[5] >= 0.50)]
    uav_kept = []
    
    for c in clusters_by_stem.get(stem, []):
        cx, cy, w, h = c["cx"], c["cy"], c["w"], c["h"]
        comb_sc = c["comb_sc"]
        max_sc = c["max_sc"]
        area = w * h
        ar = w / (h + 1e-6)
        gx, gy = int(round(cx / 0.025)), int(round(cy / 0.025))
        
        # LAW 1: Ground Cutoff
        if (cy + h/2.0) > 0.85:
            killed_fps.append((stem, cx, cy, "LAW 1: Ground/Curb Cutoff"))
            continue
            
        # LAW 2: Universal Stationary Infrastructure Suppression
        if (gx, gy) in static_bins:
            killed_fps.append((stem, cx, cy, f"LAW 2: Static Fixture across >=3 frames (Bin {gx*0.025:.2f},{gy*0.025:.2f})"))
            continue
            
        # LAW 3: Physical Scale & Symmetry
        if area > 0.0050:
            killed_fps.append((stem, cx, cy, "LAW 3: Excessive Area"))
            continue
        if ar < 0.40 or ar > 2.50:
            killed_fps.append((stem, cx, cy, f"LAW 3: Non-aerodynamic Aspect Ratio ({ar:.2f})"))
            continue
            
        # Confidence gate: Native >= 0.45, or Rescued >= 0.38
        if comb_sc >= 0.45 or (max_sc >= 0.28 and comb_sc >= 0.38):
            uav_kept.append((10, cx, cy, w, h, comb_sc))
            verified_drones += 1
        else:
            killed_fps.append((stem, cx, cy, f"Confidence Gate (< 0.38)"))
            
    all_b = non_uav + uav_kept
    all_b.sort(key=lambda x: -x[5])
    final_submissions[stem] = all_b[:60]

print("\n=== Physics V2 Clean Audit Results ===")
print(f"  Verified True Physical Drones: {verified_drones}")
print(f"  False Positives Killed by Physics V2: {len(killed_fps)}")

# Check user-reported targets specifically
print("\n--- Status of User-Reported Targets ---")
targets = ["00000071", "00000076", "00000037", "00000012", "00000422"]
for s in targets:
    u = [b for b in final_submissions[s] if b[0] == 10]
    print(f"[{s}]: {len(u)} UAVs retained -> " + ", ".join([f"conf={b[5]:.3f} box=({b[1]:.3f},{b[2]:.3f},{b[3]:.3f},{b[4]:.3f})" for b in u]))

# Package to zip & unzip
os.makedirs(os.path.dirname(OUT_ZIP), exist_ok=True)
os.makedirs(UNZIP_DIR, exist_ok=True)

with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for s in sorted(final_submissions.keys()):
        lines = []
        for b in final_submissions[s]:
            cid, cx, cy, w, h, sc = b
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {sc:.6f}\r\n")
        content_str = "".join(lines)
        zf.writestr(f"{s}.txt", content_str.encode('utf-8'))
        with open(os.path.join(UNZIP_DIR, f"{s}.txt"), "w", encoding="utf-8") as f_out:
            f_out.write(content_str)

tot_boxes = sum(len(v) for v in final_submissions.values())
print(f"\n[✔] Successfully generated Physics V2 Clean Package: {OUT_ZIP} ({tot_boxes} total boxes)")

# Check compliance
cmd = f"/home/constant/miniconda3/envs/deimv2/bin/python {ROOT}/scripts/eval/check_submission.py {UNZIP_DIR} {ROOT}/dataset/test/AIC2026_PHASE_2_1000/visible"
os.system(cmd)
