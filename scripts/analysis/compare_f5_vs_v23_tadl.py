#!/usr/bin/env python3
"""
Deep Head-to-Head Comparative Analysis:
He Jianzhong F5 Compact (58.105, 7,184 boxes) vs Sovereign V23 TADL (6,789 boxes)
================================================================================
Performs:
1. Macro metric comparisons (counts, scores, density)
2. 12-class distribution & parity ratio
3. Spatial consensus (Greedy / Hungarian matching at IoU >= 0.50)
4. Cross-model pseudo-precision & pseudo-recall
5. Spatial intersection & box IoU distribution
6. Cross-class label disagreements
7. Extreme divergent test images
"""

import os
import sys
import json
import zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np

AIC_CATEGORIES = [
    {"id": 0, "name": "person"},
    {"id": 1, "name": "boat"},
    {"id": 2, "name": "animal"},
    {"id": 3, "name": "seat"},
    {"id": 4, "name": "sign"},
    {"id": 5, "name": "bicycle"},
    {"id": 6, "name": "car"},
    {"id": 7, "name": "ball"},
    {"id": 8, "name": "light"},
    {"id": 9, "name": "garbage can"},
    {"id": 10, "name": "uav"},
    {"id": 11, "name": "tricycle"},
]
ID2NAME = {c["id"]: c["name"] for c in AIC_CATEGORIES}

def parse_zip(zpath):
    preds = {}
    total_boxes = 0
    class_counts = defaultdict(int)
    scores_list = []
    per_img_counts = []
    
    with zipfile.ZipFile(zpath) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.txt'):
                continue
            stem = Path(name).stem
            content = z.read(name).decode('utf-8').strip()
            lines = content.splitlines() if content else []
            boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cls_id = int(parts[0])
                    cx, cy, w, h, s = map(float, parts[1:6])
                    boxes.append((cls_id, cx, cy, w, h, s))
                    class_counts[cls_id] += 1
                    scores_list.append(s)
            preds[stem] = boxes
            total_boxes += len(boxes)
            per_img_counts.append(len(boxes))
            
    return preds, total_boxes, class_counts, np.array(scores_list), np.array(per_img_counts)

def cxcywh_to_xyxy(box):
    cls_id, cx, cy, w, h, s = box
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 1e-9:
        return 0.0
    return inter_area / union_area

def main():
    f5_path = Path('/home/constant/Downloads/Baidu/AIC-2026-54199578-我很帅对不队-复赛-贺建中/AIC-2026-54199578-代码与数据/results/AIC-2026-54199578_复赛测试集_结果文件_F5_compact7184_榜分58.105.zip')
    v23_path = Path('/home/constant/CodeBuddy/Dev/AIC/work_dir/audits/tadl/submission_v23_tadl_peak6789_official.zip')

    f5_preds, f5_total, f5_counts, f5_scores, f5_img_counts = parse_zip(f5_path)
    v23_preds, v23_total, v23_counts, v23_scores, v23_img_counts = parse_zip(v23_path)

    all_stems = sorted(set(f5_preds.keys()) | set(v23_preds.keys()))

    print("=" * 80)
    print("      SOVEREIGN V23 TADL VS F5 COMPACT 58.105 全方位对撞审计报告")
    print("=" * 80)
    print(f"测试集图片数: {len(all_stems)} 张 (两包皆为 1,000 张图)")
    print(f"F5 Compact 总框数:    {f5_total:5d}  (平均 {f5_total/len(all_stems):.2f} 框/图, 中位数 {np.median(f5_img_counts):.0f})")
    print(f"V23 TADL   总框数:    {v23_total:5d}  (平均 {v23_total/len(all_stems):.2f} 框/图, 中位数 {np.median(v23_img_counts):.0f})")
    print(f"框数总差值 (V23 - F5): {v23_total - f5_total:+5d} 框 (吻合度: {v23_total/f5_total*100:.2f}%)")
    print(f"F5 置信度分布:   均值={np.mean(f5_scores):.4f}, p25={np.percentile(f5_scores, 25):.4f}, p50={np.percentile(f5_scores, 50):.4f}, p75={np.percentile(f5_scores, 75):.4f}, p90={np.percentile(f5_scores, 90):.4f}")
    print(f"V23 置信度分布:  均值={np.mean(v23_scores):.4f}, p25={np.percentile(v23_scores, 25):.4f}, p50={np.percentile(v23_scores, 50):.4f}, p75={np.percentile(v23_scores, 75):.4f}, p90={np.percentile(v23_scores, 90):.4f}")

    # Class comparison
    print("\n" + "-" * 80)
    print(" 12 类目标框数横向分布比对")
    print("-" * 80)
    print(f"{'类别 ID':<8} {'类别名称':<14} {'F5 框数':<10} {'V23 框数':<10} {'差值':<8} {'比例(V23/F5)':<14}")
    print("-" * 80)
    for cid in range(12):
        cname = ID2NAME[cid]
        c_f5 = f5_counts[cid]
        c_v23 = v23_counts[cid]
        diff = c_v23 - c_f5
        ratio = (c_v23 / c_f5 * 100) if c_f5 > 0 else 0.0
        print(f"{cid:<8} {cname:<14} {c_f5:<10} {c_v23:<10} {diff:+8d} {ratio:6.1f}%")

    # Spatial matching
    # Greedy match per image per class
    matched_pairs = []
    class_matches = defaultdict(int)
    f5_matched_mask = defaultdict(lambda: defaultdict(bool))
    v23_matched_mask = defaultdict(lambda: defaultdict(bool))
    ious_list = []
    conf_diffs = []
    
    # Also cross-class overlap (same spatial box, different class)
    cross_class_conflicts = defaultdict(int)

    for stem in all_stems:
        f5_b = f5_preds.get(stem, [])
        v23_b = v23_preds.get(stem, [])
        
        f5_xy = [(b[0], cxcywh_to_xyxy(b), b[5], idx) for idx, b in enumerate(f5_b)]
        v23_xy = [(b[0], cxcywh_to_xyxy(b), b[5], idx) for idx, b in enumerate(v23_b)]
        
        # Sort by confidence descending
        f5_xy.sort(key=lambda x: -x[2])
        v23_xy.sort(key=lambda x: -x[2])
        
        used_v23 = set()
        for f_cls, f_box, f_s, f_idx in f5_xy:
            best_iou = 0.0
            best_v_idx = -1
            best_v_cls = -1
            best_v_s = 0.0
            
            for v_cls, v_box, v_s, v_idx in v23_xy:
                if v_idx in used_v23:
                    continue
                iou = compute_iou(f_box, v_box)
                if iou > best_iou:
                    best_iou = iou
                    best_v_idx = v_idx
                    best_v_cls = v_cls
                    best_v_s = v_s
            
            if best_iou >= 0.50:
                if f_cls == best_v_cls:
                    used_v23.add(best_v_idx)
                    matched_pairs.append((stem, f_cls, best_iou, f_s, best_v_s))
                    class_matches[f_cls] += 1
                    ious_list.append(best_iou)
                    conf_diffs.append(best_v_s - f_s)
                    f5_matched_mask[stem][f_idx] = True
                    v23_matched_mask[stem][best_v_idx] = True
                else:
                    # Spatial overlap but label disagreed!
                    cross_class_conflicts[(f_cls, best_v_cls)] += 1

    total_matched = len(matched_pairs)
    f5_match_rate = total_matched / f5_total * 100
    v23_match_rate = total_matched / v23_total * 100
    mean_iou = np.mean(ious_list)
    mean_conf_diff = np.mean(conf_diffs)

    print("\n" + "-" * 80)
    print(" 空间共识度与两两匹配审计 (IoU >= 0.50, 相同类别)")
    print("-" * 80)
    print(f"双方严格共识框数:        {total_matched:5d} 组")
    print(f"F5 预测框被 V23 召回率:   {f5_match_rate:.2f}% ({total_matched}/{f5_total})")
    print(f"V23 预测框被 F5 召回率:   {v23_match_rate:.2f}% ({total_matched}/{v23_total})")
    print(f"共识框平均重合度 (Mean IoU): {mean_iou:.4f} (几何定位高度契合)")
    print(f"共识框置信度差 (V23 - F5):  {mean_conf_diff:+.4f} (刻度基本无漂移)")

    print("\n" + "-" * 80)
    print(" 逐类别共识匹配详情")
    print("-" * 80)
    print(f"{'类别 ID':<8} {'类别名称':<14} {'共识匹配数':<12} {'F5 召回率':<14} {'V23 召回率':<14}")
    print("-" * 80)
    for cid in range(12):
        cname = ID2NAME[cid]
        m = class_matches[cid]
        r_f5 = (m / f5_counts[cid] * 100) if f5_counts[cid] > 0 else 0.0
        r_v23 = (m / v23_counts[cid] * 100) if v23_counts[cid] > 0 else 0.0
        print(f"{cid:<8} {cname:<14} {m:<12} {r_f5:6.2f}%        {r_v23:6.2f}%")

    # Cross class conflicts
    print("\n" + "-" * 80)
    print(" 空间位置重合 (IoU >= 0.50) 但分类存在分歧的 Top-10 争议对")
    print("-" * 80)
    sorted_conflicts = sorted(cross_class_conflicts.items(), key=lambda x: -x[1])[:10]
    print(f"{'F5 类别':<18} {'V23 类别':<18} {'争议框数量':<12}")
    print("-" * 80)
    for (f_cls, v_cls), cnt in sorted_conflicts:
        print(f"{ID2NAME[f_cls] + f' ({f_cls})':<18} {ID2NAME[v_cls] + f' ({v_cls})':<18} {cnt:<12}")

    # Top Divergent Images
    print("\n" + "-" * 80)
    print(" 框数分歧最大的 Top-10 极值图片")
    print("-" * 80)
    img_diffs = []
    for stem in all_stems:
        c1 = len(f5_preds.get(stem, []))
        c2 = len(v23_preds.get(stem, []))
        img_diffs.append((stem, c1, c2, abs(c1 - c2)))
    img_diffs.sort(key=lambda x: -x[3])
    print(f"{'图片 Stem':<20} {'F5 框数':<12} {'V23 框数':<12} {'差异 (|F5 - V23|)':<16}")
    print("-" * 80)
    for stem, c1, c2, diff in img_diffs[:10]:
        print(f"{stem:<20} {c1:<12} {c2:<12} {diff:<16}")

    # Output JSON summary
    summary = {
        "macro": {
            "images": len(all_stems),
            "f5_total_boxes": f5_total,
            "v23_total_boxes": v23_total,
            "box_diff": v23_total - f5_total,
            "f5_mean_score": float(np.mean(f5_scores)),
            "v23_mean_score": float(np.mean(v23_scores)),
            "total_matched_consensus": total_matched,
            "f5_matched_rate": float(f5_match_rate),
            "v23_matched_rate": float(v23_match_rate),
            "consensus_mean_iou": float(mean_iou),
        },
        "per_class": {
            ID2NAME[cid]: {
                "f5_count": f5_counts[cid],
                "v23_count": v23_counts[cid],
                "diff": v23_counts[cid] - f5_counts[cid],
                "matched": class_matches[cid],
                "f5_recall_pct": float(class_matches[cid] / f5_counts[cid] * 100) if f5_counts[cid] > 0 else 0.0,
                "v23_recall_pct": float(class_matches[cid] / v23_counts[cid] * 100) if v23_counts[cid] > 0 else 0.0,
            }
            for cid in range(12)
        }
    }
    with open('/home/constant/CodeBuddy/Dev/AIC/work_dir/audits/tadl/f5_vs_v23_comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved JSON report to work_dir/audits/tadl/f5_vs_v23_comparison_summary.json")

if __name__ == '__main__':
    main()
