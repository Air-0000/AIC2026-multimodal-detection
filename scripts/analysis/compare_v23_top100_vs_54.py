#!/usr/bin/env python3
"""
Comprehensive Collision & Comparative Analysis: Unfiltered V23 Top-100 vs 54.28 Baseline
========================================================================================
Performs deep multi-dimensional collision audit between:
1. Baseline (54.28 score): work_dir/submissions/submission_clean_single/submission_clean_single.zip
2. V23 Native Top-100: work_dir/submissions/v23_unfiltered_top100/submission_v23_native_unfiltered_top100.zip
3. V23 Flip TTA Top-100: work_dir/submissions/v23_unfiltered_top100/submission_v23_flip_tta_unfiltered_top100.zip
"""

import os
import sys
import zipfile
from collections import defaultdict, Counter
import numpy as np

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}


def parse_zip(zip_path, top_k=100):
    """
    Parses zip file and returns {stem: list of (cid, cx, cy, bw, bh, score)}
    If top_k > 0, keeps top_k per image by score.
    """
    data = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            stem = os.path.splitext(os.path.basename(name))[0]
            raw = zf.read(name).decode("utf-8").strip()
            boxes = []
            if raw:
                for line in raw.split("\r\n"):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        cid = int(parts[0])
                        cx, cy, bw, bh, sc = map(float, parts[1:6])
                        boxes.append((cid, cx, cy, bw, bh, sc))
            boxes.sort(key=lambda x: -x[5])
            if top_k > 0:
                boxes = boxes[:top_k]
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


def compare_datasets(baseline_data, target_data, target_name="V23"):
    stems = sorted(set(baseline_data.keys()) | set(target_data.keys()))
    
    # 1. Box counts & scores
    base_scores = []
    tgt_scores = []
    base_class_cnt = Counter()
    tgt_class_cnt = Counter()
    
    base_total = sum(len(boxes) for boxes in baseline_data.values())
    tgt_total = sum(len(boxes) for boxes in target_data.values())
    
    for boxes in baseline_data.values():
        for b in boxes:
            base_scores.append(b[5])
            base_class_cnt[b[0]] += 1
            
    for boxes in target_data.values():
        for b in boxes:
            tgt_scores.append(b[5])
            tgt_class_cnt[b[0]] += 1

    # 2. Spatial Overlap Analysis (Greedy IoU Matching)
    matched_same_class = 0
    matched_diff_class = 0
    class_confusion = Counter() # (base_cid, tgt_cid)
    
    base_matched_cnt = 0
    tgt_matched_cnt = 0
    
    ious_matched = []
    
    for stem in stems:
        b_boxes = baseline_data.get(stem, [])
        t_boxes = target_data.get(stem, [])
        
        b_used = set()
        t_used = set()
        
        # Sort by score descending
        # Try same class match first at IoU >= 0.50
        for ti, tb in enumerate(t_boxes):
            t_box_xyxy = to_xyxy(*tb[1:5])
            best_iou = 0.50
            best_bi = -1
            for bi, bb in enumerate(b_boxes):
                if bi in b_used:
                    continue
                if bb[0] == tb[0]: # Same class
                    b_box_xyxy = to_xyxy(*bb[1:5])
                    iou = iou_xyxy(t_box_xyxy, b_box_xyxy)
                    if iou >= best_iou:
                        best_iou = iou
                        best_bi = bi
            if best_bi >= 0:
                b_used.add(best_bi)
                t_used.add(ti)
                matched_same_class += 1
                ious_matched.append(best_iou)

        # Cross class match for remaining boxes
        for ti, tb in enumerate(t_boxes):
            if ti in t_used:
                continue
            t_box_xyxy = to_xyxy(*tb[1:5])
            best_iou = 0.50
            best_bi = -1
            for bi, bb in enumerate(b_boxes):
                if bi in b_used:
                    continue
                b_box_xyxy = to_xyxy(*bb[1:5])
                iou = iou_xyxy(t_box_xyxy, b_box_xyxy)
                if iou >= best_iou:
                    best_iou = iou
                    best_bi = bi
            if best_bi >= 0:
                b_used.add(best_bi)
                t_used.add(ti)
                matched_diff_class += 1
                class_confusion[(b_boxes[best_bi][0], tb[0])] += 1

    unique_tgt = tgt_total - matched_same_class - matched_diff_class
    unique_base = base_total - matched_same_class - matched_diff_class

    return {
        "target_name": target_name,
        "base_total": base_total,
        "tgt_total": tgt_total,
        "base_scores": base_scores,
        "tgt_scores": tgt_scores,
        "base_class_cnt": base_class_cnt,
        "tgt_class_cnt": tgt_class_cnt,
        "matched_same_class": matched_same_class,
        "matched_diff_class": matched_diff_class,
        "unique_tgt": unique_tgt,
        "unique_base": unique_base,
        "class_confusion": class_confusion,
        "ious_matched": ious_matched,
    }


def print_report(res, title="COMPARISON REPORT"):
    print("\n" + "=" * 80)
    print(f" {title}: 54.28 Baseline vs {res['target_name']} ".center(80, "="))
    print("=" * 80)
    
    print("\n[1] Overall Statistics:")
    print(f"  - 54.28 Baseline Total Dets: {res['base_total']:,} (Avg: {res['base_total']/1000:.1f}/img)")
    print(f"  - {res['target_name']} Total Dets:   {res['tgt_total']:,} (Avg: {res['tgt_total']/1000:.1f}/img)")
    print(f"  - Delta:                     {res['tgt_total'] - res['base_total']:+,} ({((res['tgt_total']/res['base_total'])-1)*100:+.2f}%)")

    b_sc = np.array(res['base_scores'])
    t_sc = np.array(res['tgt_scores'])
    print("\n[2] Confidence Score Distribution:")
    print(f"  Metric       54.28 Baseline       {res['target_name']}")
    print(f"  Mean:        {b_sc.mean():.4f}               {t_sc.mean():.4f}")
    print(f"  Median:      {np.median(b_sc):.4f}               {np.median(t_sc):.4f}")
    print(f"  P10:         {np.percentile(b_sc, 10):.4f}               {np.percentile(t_sc, 10):.4f}")
    print(f"  P75:         {np.percentile(b_sc, 75):.4f}               {np.percentile(t_sc, 75):.4f}")
    print(f"  P90:         {np.percentile(b_sc, 90):.4f}               {np.percentile(t_sc, 90):.4f}")
    print(f"  Max:         {b_sc.max():.4f}               {t_sc.max():.4f}")

    print("\n[3] Per-Class Breakdown (Category-by-Category Shift):")
    print(f"  {'CID':<4} {'Class Name':<14} {'54.28 Base':<12} {res['target_name']:<12} {'Diff':<10} {'Ratio':<8}")
    print("  " + "-" * 65)
    for cid in range(12):
        cname = CLASS_NAMES[cid]
        bc = res['base_class_cnt'][cid]
        tc = res['tgt_class_cnt'][cid]
        diff = tc - bc
        ratio = f"{tc/bc:.2f}x" if bc > 0 else "N/A"
        print(f"  {cid:<4} {cname:<14} {bc:<12,d} {tc:<12,d} {diff:+<10,d} {ratio:<8}")

    print("\n[4] Spatial Overlap & Agreement Audit (IoU >= 0.50):")
    agree_pct_tgt = (res['matched_same_class'] / res['tgt_total']) * 100 if res['tgt_total'] > 0 else 0
    agree_pct_base = (res['matched_same_class'] / res['base_total']) * 100 if res['base_total'] > 0 else 0
    mean_iou = np.mean(res['ious_matched']) if res['ious_matched'] else 0
    print(f"  - Concordant Detections (Same Class & IoU >= 0.50): {res['matched_same_class']:,} boxes")
    print(f"    * Overlap Rate relative to 54.28 Base: {agree_pct_base:.2f}%")
    print(f"    * Overlap Rate relative to {res['target_name']}:   {agree_pct_tgt:.2f}%")
    print(f"    * Mean IoU of Matched Targets:          {mean_iou:.4f}")
    print(f"  - Cross-Class Contradictions (IoU >= 0.50, Different Class): {res['matched_diff_class']:,} boxes")
    print(f"  - Novel Detections exclusive to {res['target_name']} (Discovered): {res['unique_tgt']:,} boxes")
    print(f"  - Baseline Detections exclusive to 54.28 (Pruned/Missed):    {res['unique_base']:,} boxes")

    if res['class_confusion']:
        print("\n[5] Top 5 Category Contradictions (Base Class -> Target Class):")
        for (bcid, tcid), cnt in res['class_confusion'].most_common(5):
            print(f"    {CLASS_NAMES[bcid]} -> {CLASS_NAMES[tcid]}: {cnt:,} occurrences")

    print("=" * 80 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", default="work_dir/submissions/submission_clean_single/submission_clean_single.zip")
    parser.add_argument("--v23-native-zip", default="work_dir/submissions/v23_unfiltered_top100/submission_v23_native_unfiltered_top100.zip")
    parser.add_argument("--v23-flip-zip", default="work_dir/submissions/v23_unfiltered_top100/submission_v23_flip_tta_unfiltered_top100.zip")
    args = parser.parse_args()

    print(f"Loading Base: {args.base_zip}")
    base_data = parse_zip(args.base_zip, top_k=100) # strictly cap at top-100 to match official rule

    if os.path.exists(args.v23_native_zip):
        print(f"Loading Native V23: {args.v23_native_zip}")
        v23_native_data = parse_zip(args.v23_native_zip, top_k=100)
        res_native = compare_datasets(base_data, v23_native_data, target_name="V23 Native Top-100")
        print_report(res_native, title="NATIVE V23 vs 54.28 BASELINE")

    if os.path.exists(args.v23_flip_zip):
        print(f"Loading Flip TTA V23: {args.v23_flip_zip}")
        v23_flip_data = parse_zip(args.v23_flip_zip, top_k=100)
        res_flip = compare_datasets(base_data, v23_flip_data, target_name="V23 Flip TTA Top-100")
        print_report(res_flip, title="FLIP TTA V23 vs 54.28 BASELINE")


if __name__ == "__main__":
    main()
