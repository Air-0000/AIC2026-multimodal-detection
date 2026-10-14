#!/usr/bin/env python3
"""
Three-Way Deep Collision Analysis:
1. F5 Official Benchmark (58.105 Peak, cap60thr45_50)
2. 54.28 Baseline (submission_clean_single.zip, 73,685 boxes)
3. V23 Multimodal Top-100 (Native & Flip TTA)
4. F5 Raw Top-100 (53.100 Baseline)
"""

import os
import sys
import zipfile
import argparse
from collections import defaultdict, Counter
import numpy as np

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}
RARE_CLASSES = {1, 7, 9, 10, 11}

def parse_zip(zip_path, top_k=0):
    data = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            stem = os.path.splitext(os.path.basename(name))[0]
            raw = zf.read(name).decode("utf-8").strip()
            boxes = []
            if raw:
                # Handle CRLF and LF
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

def match_datasets(data_a, data_b, iou_thr=0.50):
    stems = sorted(set(data_a.keys()) | set(data_b.keys()))
    matched_same = 0
    matched_diff = 0
    total_a = sum(len(v) for v in data_a.values())
    total_b = sum(len(v) for v in data_b.values())
    ious = []
    per_class_a = Counter()
    per_class_matched = Counter()

    for stem in stems:
        boxes_a = data_a.get(stem, [])
        boxes_b = data_b.get(stem, [])
        used_b = set()

        for ba in boxes_a:
            per_class_a[ba[0]] += 1
            b_xyxy = to_xyxy(*ba[1:5])
            best_iou = iou_thr
            best_bi = -1
            for bi, bb in enumerate(boxes_b):
                if bi in used_b or ba[0] != bb[0]:
                    continue
                bb_xyxy = to_xyxy(*bb[1:5])
                iou = iou_xyxy(b_xyxy, bb_xyxy)
                if iou >= best_iou:
                    best_iou = iou
                    best_bi = bi
            if best_bi >= 0:
                used_b.add(best_bi)
                matched_same += 1
                per_class_matched[ba[0]] += 1
                ious.append(best_iou)

        for ba in boxes_a:
            b_xyxy = to_xyxy(*ba[1:5])
            best_iou = iou_thr
            best_bi = -1
            for bi, bb in enumerate(boxes_b):
                if bi in used_b:
                    continue
                bb_xyxy = to_xyxy(*bb[1:5])
                iou = iou_xyxy(b_xyxy, bb_xyxy)
                if iou >= best_iou:
                    best_iou = iou
                    best_bi = bi
            if best_bi >= 0:
                used_b.add(best_bi)
                matched_diff += 1

    return {
        "total_a": total_a,
        "total_b": total_b,
        "matched_same": matched_same,
        "matched_diff": matched_diff,
        "unique_a": total_a - matched_same - matched_diff,
        "unique_b": total_b - matched_same - matched_diff,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "per_class_a": per_class_a,
        "per_class_matched": per_class_matched,
    }

def filter_by_threshold_policy(data, thr_rare=0.45, thr_common=0.50, max_per_img=60):
    filtered = {}
    for stem, boxes in data.items():
        eligible = []
        for b in boxes:
            thr = thr_rare if b[0] in RARE_CLASSES else thr_common
            if b[5] >= thr:
                eligible.append(b)
        rare = sorted([b for b in eligible if b[0] in RARE_CLASSES], key=lambda x: -x[5])
        common = sorted([b for b in eligible if b[0] not in RARE_CLASSES], key=lambda x: -x[5])
        kept = (rare + common)[:max_per_img]
        kept.sort(key=lambda x: -x[5])
        filtered[stem] = kept
    return filtered

def run_deep_comparison(f5_zip, base_zip, v23_native_zip, v23_tta_zip=None, f5_raw_zip=None):
    print("\n" + "=" * 96)
    print(" COMPREHENSIVE MULTI-WAY ARCHITECTURAL COLLISION AUDIT ".center(96, "="))
    print(" F5 Benchmark (58.105) vs 54.28 Baseline vs Sovereign V23 Multimodal ".center(96, " "))
    print("=" * 96)

    print("\n[+] Loading and parsing packages...")
    f5_data = parse_zip(f5_zip)
    base_data = parse_zip(base_zip)
    v23_data = parse_zip(v23_native_zip)
    v23_tta_data = parse_zip(v23_tta_zip) if v23_tta_zip and os.path.exists(v23_tta_zip) else None
    f5_raw_data = parse_zip(f5_raw_zip) if f5_raw_zip and os.path.exists(f5_raw_zip) else None

    # Helper stats
    def stats(d):
        scs = [b[5] for v in d.values() for b in v]
        cnt = Counter([b[0] for v in d.values() for b in v])
        non_empty = sum(1 for v in d.values() if len(v) > 0)
        return {
            "count": len(scs),
            "non_empty": non_empty,
            "mean": np.mean(scs) if scs else 0,
            "med": np.median(scs) if scs else 0,
            "p75": np.percentile(scs, 75) if scs else 0,
            "p90": np.percentile(scs, 90) if scs else 0,
            "p95": np.percentile(scs, 95) if scs else 0,
            "classes": cnt,
        }

    s_f5 = stats(f5_data)
    s_base = stats(base_data)
    s_v23 = stats(v23_data)
    s_v23_tta = stats(v23_tta_data) if v23_tta_data else None

    # 1. Macro Radar
    print("\n[1] Global Scale & Score Calibration Radar:")
    hdr = f"  {'Metric':<25} {'F5 Peak (58.105)':<20} {'54.28 Baseline':<20} {'V23 Native Top100':<20}"
    if s_v23_tta:
        hdr += f" {'V23 Flip TTA':<15}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def row(label, k, fmt="{:<20}"):
        l_str = f"  {label:<25} "
        l_str += fmt.format(f"{s_f5[k]:,}" if isinstance(s_f5[k], int) else f"{s_f5[k]:.4f}") + " "
        l_str += fmt.format(f"{s_base[k]:,}" if isinstance(s_base[k], int) else f"{s_base[k]:.4f}") + " "
        l_str += fmt.format(f"{s_v23[k]:,}" if isinstance(s_v23[k], int) else f"{s_v23[k]:.4f}") + " "
        if s_v23_tta:
            l_str += fmt.format(f"{s_v23_tta[k]:,}" if isinstance(s_v23_tta[k], int) else f"{s_v23_tta[k]:.4f}")
        print(l_str)

    row("Total Box Count", "count")
    print(f"  {'Avg Boxes / Image':<25} {s_f5['count']/1000:<20.2f} {s_base['count']/1000:<20.2f} {s_v23['count']/1000:<20.2f}" + (f" {s_v23_tta['count']/1000:<15.2f}" if s_v23_tta else ""))
    print(f"  {'Active Images (>0 box)':<25} {s_f5['non_empty']:<20d} {s_base['non_empty']:<20d} {s_v23['non_empty']:<20d}" + (f" {s_v23_tta['non_empty']:<15d}" if s_v23_tta else ""))
    row("Confidence Mean", "mean")
    row("Confidence Median", "med")
    row("Confidence P75", "p75")
    row("Confidence P90", "p90")
    row("Confidence P95", "p95")

    # 2. Simulated F5 Threshold Policy on 54.28 and V23
    print("\n[2] What happens if we apply F5's Exact Policy (.45/.50, max 60) to all models?")
    base_filtered = filter_by_threshold_policy(base_data, 0.45, 0.50, 60)
    v23_filtered = filter_by_threshold_policy(v23_data, 0.45, 0.50, 60)
    sb_f = stats(base_filtered)
    sv_f = stats(v23_filtered)
    print(f"  - F5 Peak (Native):        {s_f5['count']:,} boxes (Avg {s_f5['count']/1000:.2f}/img)")
    print(f"  - 54.28 Base filtered:     {sb_f['count']:,} boxes (Avg {sb_f['count']/1000:.2f}/img) [Retained: {sb_f['count']/s_base['count']*100:.2f}%]")
    print(f"  - V23 Native filtered:     {sv_f['count']:,} boxes (Avg {sv_f['count']/1000:.2f}/img) [Retained: {sv_f['count']/s_v23['count']*100:.2f}%]")
    if s_v23_tta:
        v23_tta_filtered = filter_by_threshold_policy(v23_tta_data, 0.45, 0.50, 60)
        svt_f = stats(v23_tta_filtered)
        print(f"  - V23 Flip TTA filtered:   {svt_f['count']:,} boxes (Avg {svt_f['count']/1000:.2f}/img) [Retained: {svt_f['count']/s_v23_tta['count']*100:.2f}%]")

    # 3. Per-Class Box Count Breakdown
    print("\n[3] Per-Class Box Distribution Comparison:")
    chdr = f"  {'CID':<4} {'Class Name':<14} {'F5 Peak':<12} {'54.28 Base':<12} {'V23 Native':<12} {'54 filtered':<12} {'V23 filtered':<12}"
    print(chdr)
    print("  " + "-" * (len(chdr) - 2))
    for cid in range(12):
        cname = CLASS_NAMES[cid]
        fc = s_f5["classes"][cid]
        bc = s_base["classes"][cid]
        vc = s_v23["classes"][cid]
        bcf = sb_f["classes"][cid]
        vcf = sv_f["classes"][cid]
        print(f"  {cid:<4} {cname:<14} {fc:<12,d} {bc:<12,d} {vc:<12,d} {bcf:<12,d} {vcf:<12,d}")

    # 4. Spatial Overlap Audit (IoU >= 0.50) against F5 Peak
    print("\n[4] Spatial Overlap Audit against F5 Peak (58.105, IoU >= 0.50):")
    m_f5_base = match_datasets(f5_data, base_data)
    m_f5_v23 = match_datasets(f5_data, v23_data)
    print(f"  A. F5 Peak vs 54.28 Base (Unfiltered Top-100):")
    print(f"     - Common Targets (Same Class & IoU >= 0.5): {m_f5_base['matched_same']:,} / {s_f5['count']:,} ({m_f5_base['matched_same']/s_f5['count']*100:.2f}% of F5)")
    print(f"     - Mean IoU of Common Targets:               {m_f5_base['mean_iou']:.4f}")
    print(f"     - Class Mismatches (IoU >= 0.5, Diff Class):{m_f5_base['matched_diff']:,}")
    print(f"     - Unique to F5 (54.28 missed entirely):      {m_f5_base['unique_a']:,}")

    print(f"\n  B. F5 Peak vs V23 Native (Unfiltered Top-100):")
    print(f"     - Common Targets (Same Class & IoU >= 0.5): {m_f5_v23['matched_same']:,} / {s_f5['count']:,} ({m_f5_v23['matched_same']/s_f5['count']*100:.2f}% of F5)")
    print(f"     - Mean IoU of Common Targets:               {m_f5_v23['mean_iou']:.4f}")
    print(f"     - Class Mismatches (IoU >= 0.5, Diff Class):{m_f5_v23['matched_diff']:,}")
    print(f"     - Unique to F5 (V23 missed entirely):       {m_f5_v23['unique_a']:,}")

    if s_v23_tta:
        m_f5_v23_tta = match_datasets(f5_data, v23_tta_data)
        print(f"\n  C. F5 Peak vs V23 Flip TTA (Unfiltered Top-100):")
        print(f"     - Common Targets (Same Class & IoU >= 0.5): {m_f5_v23_tta['matched_same']:,} / {s_f5['count']:,} ({m_f5_v23_tta['matched_same']/s_f5['count']*100:.2f}% of F5)")
        print(f"     - Mean IoU of Common Targets:               {m_f5_v23_tta['mean_iou']:.4f}")

    # 5. Overlap between F5 Peak and Filtered Versions
    print(f"\n  D. F5 Peak vs Filtered Versions (.45/.50, max 60):")
    m_f5_base_f = match_datasets(f5_data, base_filtered)
    m_f5_v23_f = match_datasets(f5_data, v23_filtered)
    print(f"     - F5 Peak vs 54.28 Filtered: {m_f5_base_f['matched_same']:,} matched ({m_f5_base_f['matched_same']/s_f5['count']*100:.2f}% of F5)")
    print(f"     - F5 Peak vs V23 Filtered:   {m_f5_v23_f['matched_same']:,} matched ({m_f5_v23_f['matched_same']/s_f5['count']*100:.2f}% of F5)")

    print("\n" + "=" * 96 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f5-zip", required=True)
    parser.add_argument("--base-zip", default="work_dir/submissions/submission_clean_single/submission_clean_single.zip")
    parser.add_argument("--v23-zip", default="work_dir/submissions/v23_unfiltered_top100/submission_v23_native_unfiltered_top100.zip")
    parser.add_argument("--v23-tta-zip", default="work_dir/submissions/v23_unfiltered_top100/submission_v23_flip_tta_unfiltered_top100.zip")
    parser.add_argument("--f5-raw-zip", default=None)
    args = parser.parse_args()

    run_deep_comparison(args.f5_zip, args.base_zip, args.v23_zip, args.v23_tta_zip, args.f5_raw_zip)

if __name__ == "__main__":
    main()
