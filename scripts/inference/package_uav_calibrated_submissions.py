#!/usr/bin/env python3
"""
Generate Calibrated & Query-Fused Submissions from V29 Cycle 5 Peak (2.5K Native)
Packages:
1. Variant 1A: Scale-Aware Dual-Gate (Threshold=0.32 for small UAVs, 0.45 for large UAVs, F5 standard for others)
2. Variant 1B: Intra-Model Query Fusion (Probabilistic OR aggregation for overlapping queries, Threshold=0.45)
3. Variant 1C: Fusion + Scale-Aware Dual-Gate (Thr=0.35 for fused small UAVs)
"""

import os
import sys
import math
import zipfile
import torch

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, ROOT)
from scripts.analysis.analyze_f5_overlap_sweep import to_xyxy, iou_xyxy

RAW_PT = os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/raw_detections_v29_c5_peak.pt")
OUT_DIR = os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak")

RARE_CLASSES = {1, 7, 9, 10, 11} # boat, ball, garbage can, uav, tricycle
COMMON_CLASSES = {0, 2, 3, 4, 5, 6, 8}

def package_zip(detections, zip_path, max_cap=60):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for stem in sorted(detections.keys()):
            boxes = sorted(detections[stem], key=lambda x: -x[5])
            if max_cap > 0:
                boxes = boxes[:max_cap]
            lines = []
            for b in boxes:
                cid, cx, cy, w, h, sc = b
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {sc:.6f}\r\n")
            zf.writestr(f"{stem}.txt", "".join(lines).encode('utf-8'))
    print(f"[✔] Successfully created package: {zip_path} ({sum(len(v) for v in detections.values())} boxes)")

def generate_variant_1a(raw_dict, uav_small_thr=0.32, area_thr=0.001):
    res = {}
    for stem, bxs in raw_dict.items():
        filt = []
        for b in bxs:
            cid, cx, cy, w, h, sc = b
            if cid == 10:
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

def generate_variant_1b(raw_dict, iou_merge=0.50):
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

def generate_variant_1c(raw_dict, uav_small_thr=0.35, iou_merge=0.50):
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

def main():
    print("[*] Loading raw detections...")
    raw = torch.load(RAW_PT, weights_only=False)
    print(f"[✔] Loaded {len(raw)} images from {RAW_PT}")

    pkg_1a = os.path.join(OUT_DIR, "submission_v29_c5_scale_dual_gate_thr32.zip")
    pkg_1b = os.path.join(OUT_DIR, "submission_v29_c5_intra_query_fusion_thr45.zip")
    pkg_1c = os.path.join(OUT_DIR, "submission_v29_c5_fusion_dual_gate_thr35.zip")

    print("[*] Generating Variant 1A: Scale-Aware Dual-Gate (Thr32)...")
    d_1a = generate_variant_1a(raw, uav_small_thr=0.32)
    package_zip(d_1a, pkg_1a)

    print("[*] Generating Variant 1B: Intra-Model Query Fusion (Thr45)...")
    d_1b = generate_variant_1b(raw)
    package_zip(d_1b, pkg_1b)

    print("[*] Generating Variant 1C: Fusion + Dual-Gate (Thr35)...")
    d_1c = generate_variant_1c(raw, uav_small_thr=0.35)
    package_zip(d_1c, pkg_1c)

    print("\n[✔] All packages generated successfully!")

if __name__ == "__main__":
    main()
