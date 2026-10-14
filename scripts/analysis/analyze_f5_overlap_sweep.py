#!/usr/bin/env python3
"""
Comprehensive Overlap Analysis: Sovereign V29 Cycle 5 vs Official F5 Baseline (58.105)

Evaluates:
- Overall box counts & empty image counts
- Exact Consensus (IoU >= 0.50, same class_id)
- High-precision Consensus (IoU >= 0.75, same class_id)
- Recall of F5 (consensus / F5_total)
- Precision relative to F5 (consensus / V29_total)
- Category conflicts (IoU >= 0.50, different class_id)
- Per-class breakdown with deep UAV (CID 10) dynamics
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

def parse_zip(zip_path, top_k=0, min_conf=0.0):
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
                        if sc >= min_conf:
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

def match_predictions(data_ref, data_cand, iou_thr=0.50):
    stems = sorted(set(data_ref.keys()) | set(data_cand.keys()))
    matched_same = 0
    matched_diff = 0
    total_ref = sum(len(v) for v in data_ref.values())
    total_cand = sum(len(v) for v in data_cand.values())
    ious = []
    
    per_class_ref = Counter()
    per_class_cand = Counter()
    per_class_matched = Counter()
    per_class_conflict = Counter()

    for stem in stems:
        boxes_ref = data_ref.get(stem, [])
        boxes_cand = data_cand.get(stem, [])
        for b in boxes_ref:
            per_class_ref[b[0]] += 1
        for b in boxes_cand:
            per_class_cand[b[0]] += 1

        used_cand = set()

        # Step 1: Match same class
        for ba in boxes_ref:
            b_xyxy = to_xyxy(*ba[1:5])
            best_iou = iou_thr
            best_bi = -1
            for bi, bb in enumerate(boxes_cand):
                if bi in used_cand or ba[0] != bb[0]:
                    continue
                bb_xyxy = to_xyxy(*bb[1:5])
                iou = iou_xyxy(b_xyxy, bb_xyxy)
                if iou >= best_iou:
                    best_iou = iou
                    best_bi = bi
            if best_bi >= 0:
                used_cand.add(best_bi)
                matched_same += 1
                per_class_matched[ba[0]] += 1
                ious.append(best_iou)

        # Step 2: Detect class conflicts (same box, different class)
        for ba in boxes_ref:
            b_xyxy = to_xyxy(*ba[1:5])
            best_iou = iou_thr
            best_bi = -1
            for bi, bb in enumerate(boxes_cand):
                if bi in used_cand:
                    continue
                bb_xyxy = to_xyxy(*bb[1:5])
                iou = iou_xyxy(b_xyxy, bb_xyxy)
                if iou >= best_iou:
                    best_iou = iou
                    best_bi = bi
            if best_bi >= 0:
                used_cand.add(best_bi)
                matched_diff += 1
                per_class_conflict[ba[0]] += 1

    return {
        "total_ref": total_ref,
        "total_cand": total_cand,
        "matched_same": matched_same,
        "matched_diff": matched_diff,
        "unique_ref": total_ref - matched_same - matched_diff,
        "unique_cand": total_cand - matched_same - matched_diff,
        "recall_ref": (matched_same / total_ref * 100.0) if total_ref > 0 else 0.0,
        "precision_cand": (matched_same / total_cand * 100.0) if total_cand > 0 else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "per_class_ref": per_class_ref,
        "per_class_cand": per_class_cand,
        "per_class_matched": per_class_matched,
        "per_class_conflict": per_class_conflict,
    }

def print_summary_table(f5_data, cand_dict):
    print("\n" + "=" * 110)
    print(" Sovereign V29 Cycle 5 Peak vs F5 (58.105) Multi-Threshold Consensus Sweep ".center(110, "="))
    print("=" * 110)
    header = f"{'Variant':<28} | {'Total':<7} | {'Consensus':<10} | {'F5 Recall':<10} | {'Cand Prec':<10} | {'Conflicts':<10} | {'UAV Recall':<12}"
    print(header)
    print("-" * 110)

    results = {}
    for name, data in cand_dict.items():
        res = match_predictions(f5_data, data, iou_thr=0.50)
        results[name] = res
        uav_f5 = res["per_class_ref"][10]
        uav_mat = res["per_class_matched"][10]
        uav_cand = res["per_class_cand"][10]
        uav_rec = (uav_mat / uav_f5 * 100.0) if uav_f5 > 0 else 0.0
        
        print(f"{name:<28} | {res['total_cand']:<7} | {res['matched_same']:<10} | {res['recall_ref']:>8.2f}% | {res['precision_cand']:>8.2f}% | {res['matched_diff']:<10} | {uav_mat}/{uav_f5} ({uav_rec:.1f}%) [Tot:{uav_cand}]")

    print("=" * 110)
    return results

def print_per_class_table(results, f5_data):
    print("\n" + "=" * 115)
    print(" Per-Class Exact Consensus (IoU >= 0.50) with F5 Benchmark Across Thresholds ".center(115, "="))
    print("=" * 115)
    names = list(results.keys())
    col_headers = "".join([f" | {name[:14]:<14}" for name in names])
    print(f"{'CID':<3} | {'Class Name':<12} | {'F5 Baseline':<11}" + col_headers)
    print("-" * 115)

    f5_counts = Counter()
    for v in f5_data.values():
        for b in v:
            f5_counts[b[0]] += 1

    for cid in range(12):
        cname = CLASS_NAMES[cid]
        f5_c = f5_counts[cid]
        cols = []
        for name in names:
            res = results[name]
            mat = res["per_class_matched"][cid]
            cand = res["per_class_cand"][cid]
            rec = (mat / f5_c * 100.0) if f5_c > 0 else 0.0
            cols.append(f"{mat}/{f5_c} ({rec:4.1f}%)")
        print(f"{cid:<3} | {cname:<12} | {f5_c:<11}" + "".join([f" | {col:<14}" for col in cols]))
    print("=" * 115)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f5-zip", default="work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip")
    parser.add_argument("--dir", default="work_dir/submissions/v29_cycle5_peak")
    args = parser.parse_args()

    assert os.path.exists(args.f5_zip), f"F5 zip not found: {args.f5_zip}"
    f5_data = parse_zip(args.f5_zip)
    print(f"[✔] Loaded F5 benchmark: {sum(len(v) for v in f5_data.values())} boxes across {len(f5_data)} images")

    # Find available packages
    packages = [
        ("Cap60 (45/50)", os.path.join(args.dir, "submission_sovereign_v29_c5_peak_cap60.zip")),
        ("Native (45/50)", os.path.join(args.dir, "submission_sovereign_v29_c5_peak_native_thr45_50.zip")),
        ("Uniform Thr45", os.path.join(args.dir, "submission_sovereign_v29_c5_peak_thr45.zip")),
        ("Uniform Thr35", os.path.join(args.dir, "submission_sovereign_v29_c5_peak_thr35.zip")),
        ("Uniform Thr30", os.path.join(args.dir, "submission_sovereign_v29_c5_peak_thr30.zip")),
        ("Uniform Thr25", os.path.join(args.dir, "submission_sovereign_v29_c5_peak_thr25.zip")),
    ]

    cand_dict = {}
    for label, path in packages:
        if os.path.exists(path):
            cand_dict[label] = parse_zip(path)
            print(f"[+] Loaded {label} from {path} ({sum(len(v) for v in cand_dict[label].values())} boxes)")

    if cand_dict:
        results = print_summary_table(f5_data, cand_dict)
        print_per_class_table(results, f5_data)
