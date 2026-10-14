#!/usr/bin/env python3
"""
Deep Head-to-Head Comparison:
F5 Peak Benchmark (submission_F5_cap60thr45_50.zip, 58.105)
vs
Sovereign V23 Flip TTA (submission_v23_flip_tta_thr45.zip, 6,226 boxes)
"""

import os
import sys
import zipfile
import json
from collections import defaultdict, Counter
import numpy as np

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}
RARE_CLASSES = {1, 7, 9, 10, 11}

def parse_zip(zip_path):
    data = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            stem = os.path.splitext(os.path.basename(name))[0]
            raw = zf.read(name).decode("utf-8").strip()
            boxes = []
            if raw:
                for line in raw.replace("\r\n", "\n").split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        cid = int(parts[0])
                        cx, cy, bw, bh, sc = map(float, parts[1:6])
                        boxes.append((cid, cx, cy, bw, bh, sc))
            boxes.sort(key=lambda x: -x[5])
            data[stem] = boxes
    return data

def iou_xyxy(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / ua if ua > 0 else 0.0

def to_xyxy(cx, cy, bw, bh):
    return [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]

def main():
    f5_zip = "work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip"
    v23_zip = "work_dir/submissions/v23_unfiltered_top100/submission_v23_flip_tta_thr45.zip"

    print("=" * 96)
    print(" F5 PEAK (58.105) vs SOVEREIGN V23 FLIP TTA (thr=0.45) DEEP COLLISION AUDIT ".center(96, "="))
    print("=" * 96)

    f5_data = parse_zip(f5_zip)
    v23_data = parse_zip(v23_zip)

    stems = sorted(set(f5_data.keys()) | set(v23_data.keys()))
    print(f"\n[+] Total Images Analyzed: {len(stems)}")

    # 1. Global Box Count and Score Distribution
    f5_scores = [b[5] for v in f5_data.values() for b in v]
    v23_scores = [b[5] for v in v23_data.values() for b in v]

    f5_empty = sum(1 for v in f5_data.values() if len(v) == 0)
    v23_empty = sum(1 for v in v23_data.values() if len(v) == 0)

    print("\n[1] Global Scale & Confidence Distribution:")
    print(f"  {'Metric':<25} {'F5 Peak (58.105)':<25} {'V23 Flip TTA (thr45)':<25} {'Delta (V23 - F5)':<20}")
    print("  " + "-" * 90)
    print(f"  {'Total Box Count':<25} {len(f5_scores):<25,d} {len(v23_scores):<25,d} {len(v23_scores) - len(f5_scores):+10,d}")
    print(f"  {'Avg Boxes / Image':<25} {len(f5_scores)/1000:<25.2f} {len(v23_scores)/1000:<25.2f} {len(v23_scores)/1000 - len(f5_scores)/1000:+10.2f}")
    print(f"  {'Empty Images (0 box)':<25} {f5_empty:<25d} {v23_empty:<25d} {v23_empty - f5_empty:+10d}")
    print(f"  {'Confidence Mean':<25} {np.mean(f5_scores):<25.4f} {np.mean(v23_scores):<25.4f} {np.mean(v23_scores) - np.mean(f5_scores):+10.4f}")
    print(f"  {'Confidence Median':<25} {np.median(f5_scores):<25.4f} {np.median(v23_scores):<25.4f} {np.median(v23_scores) - np.median(f5_scores):+10.4f}")
    print(f"  {'Confidence P25':<25} {np.percentile(f5_scores, 25):<25.4f} {np.percentile(v23_scores, 25):<25.4f} {np.percentile(v23_scores, 25) - np.percentile(f5_scores, 25):+10.4f}")
    print(f"  {'Confidence P75':<25} {np.percentile(f5_scores, 75):<25.4f} {np.percentile(v23_scores, 75):<25.4f} {np.percentile(v23_scores, 75) - np.percentile(f5_scores, 75):+10.4f}")
    print(f"  {'Confidence P90':<25} {np.percentile(f5_scores, 90):<25.4f} {np.percentile(v23_scores, 90):<25.4f} {np.percentile(v23_scores, 90) - np.percentile(f5_scores, 90):+10.4f}")
    print(f"  {'Confidence Min':<25} {np.min(f5_scores):<25.4f} {np.min(v23_scores):<25.4f} {np.min(v23_scores) - np.min(f5_scores):+10.4f}")
    print(f"  {'Confidence Max':<25} {np.max(f5_scores):<25.4f} {np.max(v23_scores):<25.4f} {np.max(v23_scores) - np.max(f5_scores):+10.4f}")

    # 2. Per-Class Box Distribution
    f5_classes = Counter([b[0] for v in f5_data.values() for b in v])
    v23_classes = Counter([b[0] for v in v23_data.values() for b in v])

    print("\n[2] Per-Class Box Breakdown (All 12 Categories):")
    print(f"  {'CID':<4} {'Class Name':<14} {'F5 Peak':<12} {'F5 Share':<10} {'V23 TTA':<12} {'V23 Share':<10} {'Delta':<10} {'Ratio (V23/F5)':<15}")
    print("  " + "-" * 90)
    for cid in range(12):
        cname = CLASS_NAMES[cid]
        fc = f5_classes[cid]
        vc = v23_classes[cid]
        fs = fc / len(f5_scores) * 100
        vs = vc / len(v23_scores) * 100
        delta = vc - fc
        ratio = f"{vc/fc:.2f}x" if fc > 0 else "N/A"
        is_rare = "*" if cid in RARE_CLASSES else " "
        print(f" {is_rare}{cid:<3} {cname:<14} {fc:<12,d} {fs:<9.2f}% {vc:<12,d} {vs:<9.2f}% {delta:+10,d} {ratio:<15}")
    print("  (* indicates rare class)")

    # 3. Spatial IoU Matching & Recall Analysis
    print("\n[3] Spatial IoU Matching & Cross-Model Concordance:")
    for thr in [0.30, 0.50, 0.70, 0.85]:
        matched_same = 0
        matched_diff = 0
        ious = []
        per_class_f5_matched = Counter()
        per_class_f5_total = Counter()

        for stem in stems:
            f5_b = f5_data.get(stem, [])
            v23_b = v23_data.get(stem, [])
            used_v23 = set()

            for ba in f5_b:
                per_class_f5_total[ba[0]] += 1
                ba_xyxy = to_xyxy(*ba[1:5])
                best_iou = thr
                best_vi = -1
                for vi, vb in enumerate(v23_b):
                    if vi in used_v23 or ba[0] != vb[0]:
                        continue
                    vb_xyxy = to_xyxy(*vb[1:5])
                    iou = iou_xyxy(ba_xyxy, vb_xyxy)
                    if iou >= best_iou:
                        best_iou = iou
                        best_vi = vi
                if best_vi >= 0:
                    used_v23.add(best_vi)
                    matched_same += 1
                    per_class_f5_matched[ba[0]] += 1
                    ious.append(best_iou)

            for ba in f5_b:
                ba_xyxy = to_xyxy(*ba[1:5])
                best_iou = thr
                best_vi = -1
                for vi, vb in enumerate(v23_b):
                    if vi in used_v23:
                        continue
                    vb_xyxy = to_xyxy(*vb[1:5])
                    iou = iou_xyxy(ba_xyxy, vb_xyxy)
                    if iou >= best_iou:
                        best_iou = iou
                        best_vi = vi
                if best_vi >= 0:
                    used_v23.add(best_vi)
                    matched_diff += 1

        recall_f5 = matched_same / len(f5_scores) * 100
        precision_v23 = matched_same / len(v23_scores) * 100
        mean_iou = np.mean(ious) if ious else 0.0
        print(f"  [IoU >= {thr:.2f}]:")
        print(f"    - Concordant Pairs (Same Class): {matched_same:,} / {len(f5_scores):,} ({recall_f5:.2f}% of F5 captured by V23)")
        print(f"    - Concordance of V23 boxes:      {matched_same:,} / {len(v23_scores):,} ({precision_v23:.2f}% of V23 matches F5)")
        print(f"    - Class Confusion (IoU>=thr, Diff Class): {matched_diff:,}")
        print(f"    - Mean IoU of Common Detections: {mean_iou:.4f}")

    # Detailed Per-Class Concordance at IoU >= 0.50
    print("\n[4] Per-Class Concordance Breakdown (at IoU >= 0.50):")
    print(f"  {'CID':<4} {'Class Name':<14} {'F5 Total':<10} {'V23 Captured':<14} {'Capture Rate':<14} {'V23 Total':<10} {'V23 Precision':<14}")
    print("  " + "-" * 85)
    matched_same = 0
    per_class_f5_matched = Counter()
    per_class_v23_matched = Counter()
    ious = []

    for stem in stems:
        f5_b = f5_data.get(stem, [])
        v23_b = v23_data.get(stem, [])
        used_v23 = set()

        for ba in f5_b:
            ba_xyxy = to_xyxy(*ba[1:5])
            best_iou = 0.50
            best_vi = -1
            for vi, vb in enumerate(v23_b):
                if vi in used_v23 or ba[0] != vb[0]:
                    continue
                vb_xyxy = to_xyxy(*vb[1:5])
                iou = iou_xyxy(ba_xyxy, vb_xyxy)
                if iou >= best_iou:
                    best_iou = iou
                    best_vi = vi
            if best_vi >= 0:
                used_v23.add(best_vi)
                per_class_f5_matched[ba[0]] += 1
                per_class_v23_matched[ba[0]] += 1

    for cid in range(12):
        cname = CLASS_NAMES[cid]
        fc = f5_classes[cid]
        vc = v23_classes[cid]
        cap = per_class_f5_matched[cid]
        cap_rate = f"{cap/fc*100:.2f}%" if fc > 0 else "N/A"
        prec = f"{cap/vc*100:.2f}%" if vc > 0 else "N/A"
        print(f"  {cid:<4} {cname:<14} {fc:<10,d} {cap:<14,d} {cap_rate:<14} {vc:<10,d} {prec:<14}")

    # 5. Semantic Confusions (IoU >= 0.50 but Diff Class)
    confusions = Counter()
    for stem in stems:
        f5_b = f5_data.get(stem, [])
        v23_b = v23_data.get(stem, [])
        used_v23 = set()

        # Mark same first
        for ba in f5_b:
            ba_xyxy = to_xyxy(*ba[1:5])
            best_iou = 0.50
            best_vi = -1
            for vi, vb in enumerate(v23_b):
                if vi in used_v23 or ba[0] != vb[0]:
                    continue
                vb_xyxy = to_xyxy(*vb[1:5])
                iou = iou_xyxy(ba_xyxy, vb_xyxy)
                if iou >= best_iou:
                    best_iou = iou
                    best_vi = vi
            if best_vi >= 0:
                used_v23.add(best_vi)

        # Find diff
        for ba in f5_b:
            ba_xyxy = to_xyxy(*ba[1:5])
            best_iou = 0.50
            best_vi = -1
            for vi, vb in enumerate(v23_b):
                if vi in used_v23:
                    continue
                vb_xyxy = to_xyxy(*vb[1:5])
                iou = iou_xyxy(ba_xyxy, vb_xyxy)
                if iou >= best_iou:
                    best_iou = iou
                    best_vi = vi
            if best_vi >= 0:
                used_v23.add(best_vi)
                confusions[(ba[0], v23_b[best_vi][0])] += 1

    print("\n[5] Top Cross-Class Semantic Confusions (IoU >= 0.50):")
    print(f"  {'F5 Class':<16} -> {'V23 Class':<16} : {'Confusion Count':<10}")
    print("  " + "-" * 50)
    for (ca, cb), cnt in confusions.most_common(10):
        print(f"  {CLASS_NAMES[ca]:<16} -> {CLASS_NAMES[cb]:<16} : {cnt:<10,d}")

    # 6. File-Level Top Discrepancies
    file_diffs = []
    for stem in stems:
        f5_b = f5_data.get(stem, [])
        v23_b = v23_data.get(stem, [])
        diff = abs(len(f5_b) - len(v23_b))
        file_diffs.append((stem, len(f5_b), len(v23_b), diff))

    file_diffs.sort(key=lambda x: -x[3])
    print("\n[6] Top 10 Images with Largest Box Count Divergence:")
    print(f"  {'Filename':<20} {'F5 Count':<12} {'V23 Count':<12} {'Divergence (|Delta|)':<20}")
    print("  " + "-" * 66)
    for stem, fc, vc, df in file_diffs[:10]:
        print(f"  {stem+'.txt':<20} {fc:<12d} {vc:<12d} {df:<20d}")

    print("\n" + "=" * 96)

if __name__ == "__main__":
    main()
