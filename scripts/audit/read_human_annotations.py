#!/usr/bin/env python3
"""
AIC 2026 Human Verification & Audit Annotation Reader & Downstream Analysis Tool
================================================================================
Reads human audit annotations exported from the Real-Time Multimodal Inspection Station.
Provides:
1. Executive statistical analysis (True Positives, False Positives, Missed GTs, Uncertain).
2. Per-model performance metrics (F5 Baseline vs Candidate Models precision on audited samples).
3. Per-class breakdown and false-alarm frequency analysis.
4. Export human-refined pseudo ground truth (`dataset/refined_ground_truth.json`).
5. Export false positive blacklist patterns for model post-processing.
6. Generate a comprehensive Markdown audit report.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path('/home/constant/CodeBuddy/Dev/AIC')
DEFAULT_AUDIT_JSON = ROOT_DIR / 'outputs' / 'audit' / 'aic_human_audit_annotations.json'
DEFAULT_OUTPUT_DIR = ROOT_DIR / 'outputs' / 'audit'

CLASS_NAMES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}
CLASS_ZH = {
    0: '人 (person)', 1: '船 (boat)', 2: '动物 (animal)', 3: '座椅 (seat)',
    4: '标牌 (sign)', 5: '单车 (bicycle)', 6: '汽车 (car)', 7: '球 (ball)',
    8: '路灯 (light)', 9: '垃圾桶 (garbage can)', 10: '无人机 (uav)', 11: '三轮车 (tricycle)'
}


def load_annotations(json_path):
    """Load and validate the audit annotations JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Audit file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check format: either direct list or dict with 'annotations' key
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and 'annotations' in data:
        items = data['annotations']
    elif isinstance(data, dict) and 'records' in data:
        items = data['records']
    else:
        # Dictionary keyed by boxId
        items = list(data.values())
    
    return items


def analyze_audit_records(items):
    """Compute comprehensive statistics across models and classes."""
    stats = {
        'total': len(items),
        'by_status': defaultdict(int),
        'by_model': defaultdict(lambda: defaultdict(int)),
        'by_class': defaultdict(lambda: defaultdict(int)),
        'by_role': defaultdict(lambda: defaultdict(int)),
        'stems_reviewed': set(),
        'fp_records': [],
        'tp_records': [],
        'missed_gt_records': [],
        'uncertain_records': []
    }

    for it in items:
        status = it.get('status', 'unknown')
        mid = it.get('model_id', 'unknown')
        cid = int(it.get('class_id', 0))
        stem = it.get('stem', '')
        role = it.get('role', 'unknown')
        
        stats['by_status'][status] += 1
        stats['by_model'][mid][status] += 1
        stats['by_class'][cid][status] += 1
        stats['by_role'][role][status] += 1
        if stem:
            stats['stems_reviewed'].add(stem)

        if status == 'false':
            stats['fp_records'].append(it)
        elif status == 'true':
            stats['tp_records'].append(it)
        elif status == 'missed':
            stats['missed_gt_records'].append(it)
        elif status == 'uncertain':
            stats['uncertain_records'].append(it)

    return stats


def generate_markdown_report(stats, output_path):
    """Generate an executive Markdown analysis report."""
    total = stats['total']
    n_tp = stats['by_status'].get('true', 0)
    n_fp = stats['by_status'].get('false', 0)
    n_missed = stats['by_status'].get('missed', 0)
    n_unc = stats['by_status'].get('uncertain', 0)

    tp_rate = (n_tp / total * 100) if total > 0 else 0
    fp_rate = (n_fp / total * 100) if total > 0 else 0

    lines = [
        "# AIC 2026 人工复审标定深度分析报告",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**复审样本覆盖**: 共有 `{len(stats['stems_reviewed'])}` 张图完成人工标定  ",
        f"**核验目标总数**: `{total}` 个目标框\n",
        "## 1. 总体核验结论 (Executive Summary)",
        "| 核验分类 | 数量 | 占比 | 战术解读 |",
        "| :--- | :---: | :---: | :--- |",
        f"| **真实有效目标 (True Positive)** | `{n_tp}` | `{tp_rate:.1f}%` | 确认为真实物理目标（含候选模型发现的真实暗弱目标） |",
        f"| **虚警误报伪框 (False Positive)** | `{n_fp}` | `{fp_rate:.1f}%` | 确认为误检虚警（反射、树冠、云纹、背景杂波） |",
        f"| **标杆漏报真值 (Missed GT)** | `{n_missed}` | `{(n_missed/total*100) if total else 0:.1f}%` | F5 标杆遗漏、由新模型成功检出的珍贵真值 |",
        f"| **存疑待复查 (Uncertain)** | `{n_unc}` | `{(n_unc/total*100) if total else 0:.1f}%` | 尺度极微或波段信号微弱，需进一步对比 |",
        "",
        "## 2. 各模型核验精度与虚警率对比 (Model Performance Breakdown)",
        "| 模型标识 | 核验总数 | 真框 (TP) | 伪框 (FP) | 漏标真值 | 人工核准精确率 (Precision) | 虚警率 (FAR) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for mid, mstats in sorted(stats['by_model'].items()):
        m_tot = sum(mstats.values())
        m_tp = mstats.get('true', 0)
        m_fp = mstats.get('false', 0)
        m_mgt = mstats.get('missed', 0)
        m_prec = (m_tp / (m_tp + m_fp) * 100) if (m_tp + m_fp) > 0 else 0.0
        m_far = (m_fp / m_tot * 100) if m_tot > 0 else 0.0
        lines.append(f"| **{mid.upper()}** | `{m_tot}` | `{m_tp}` | `{m_fp}` | `{m_mgt}` | **`{m_prec:.1f}%`** | `{m_far:.1f}%` |")

    lines.extend([
        "",
        "## 3. 12大类别核验分布 (Per-Class Verification Breakdown)",
        "| 类别 ID | 类别名称 | 核验总数 | 真框 (TP) | 伪框 (FP) | 漏标真值 | 人工真实精确率 | 虚警主要原因 / 备注 |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ])

    for cid in range(12):
        cstats = stats['by_class'][cid]
        c_tot = sum(cstats.values())
        if c_tot == 0:
            continue
        c_tp = cstats.get('true', 0)
        c_fp = cstats.get('false', 0)
        c_mgt = cstats.get('missed', 0)
        c_prec = (c_tp / (c_tp + c_fp) * 100) if (c_tp + c_fp) > 0 else 0.0
        
        # Collect sample notes for this class
        notes = [it.get('note', '') for it in stats['fp_records'] if int(it.get('class_id', -1)) == cid and it.get('note')]
        note_str = '; '.join(notes[:2]) if notes else "正常复审"
        lines.append(f"| `{cid}` | **{CLASS_ZH[cid]}** | `{c_tot}` | `{c_tp}` | `{c_fp}` | `{c_mgt}` | **`{c_prec:.1f}%`** | {note_str} |")

    lines.extend([
        "",
        "## 4. 战术改进建议 (Actionable Insights)",
        "1. **虚警黑名单过滤**: 针对人工标记为 `false` 的虚警，结合提取的尺度与置信度模式，可在后处理规则中增加对应抑制门限。",
        "2. **扩充真实评估集**: 标记为 `true` 与 `missed` 的目标可合并入高可信伪真值集，用于线下多模型集成与验证。",
        "3. **稀有类别 (球 / 无人机 / 动物) 重点保护**: 观察候选模型独有检出的真框，若大量为珍贵真值，可进一步调低此类别的独立 NMS 门限以拔高召回。"
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Generated Markdown Audit Report: {output_path}")


def export_refined_pseudo_ground_truth(tp_and_missed_records, output_path):
    """Export verified true positives and missed GTs as clean ground truth JSON."""
    gt_by_stem = defaultdict(list)
    for it in tp_and_missed_records:
        stem = it.get('stem')
        if not stem:
            continue
        gt_by_stem[stem].append({
            'class_id': int(it.get('class_id', 0)),
            'box': it.get('box', []),
            'source_model': it.get('model_id', ''),
            'verified_status': it.get('status', 'true'),
            'score': float(it.get('score', 1.0)),
            'note': it.get('note', '')
        })

    payload = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_verified_stems': len(gt_by_stem),
            'total_verified_boxes': len(tp_and_missed_records),
            'description': 'Human-verified high-confidence pseudo ground truth'
        },
        'ground_truth': gt_by_stem
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Exported Refined Pseudo Ground Truth: {output_path}")


def export_fp_blacklist_patterns(fp_records, output_path):
    """Analyze false positives and export patterns for filtering."""
    analysis = {
        'total_false_positives': len(fp_records),
        'by_class': defaultdict(int),
        'score_distribution': {'<0.50': 0, '0.50-0.70': 0, '>=0.70': 0},
        'small_boxes_count': 0,  # width or height < 0.02
        'fp_entries': []
    }

    for it in fp_records:
        cid = int(it.get('class_id', 0))
        analysis['by_class'][cid] += 1
        sc = float(it.get('score', 0.0))
        if sc < 0.50:
            analysis['score_distribution']['<0.50'] += 1
        elif sc < 0.70:
            analysis['score_distribution']['0.50-0.70'] += 1
        else:
            analysis['score_distribution']['>=0.70'] += 1

        box = it.get('box', [0, 0, 0, 0])
        if len(box) >= 4 and (box[2] < 0.02 or box[3] < 0.02):
            analysis['small_boxes_count'] += 1

        analysis['fp_entries'].append({
            'stem': it.get('stem'),
            'class_id': cid,
            'model_id': it.get('model_id'),
            'box': box,
            'score': sc,
            'note': it.get('note', '')
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"Exported False Positive Blacklist Patterns: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Read and analyze human audit annotations.")
    parser.add_argument('--input', type=str, default=str(DEFAULT_AUDIT_JSON), help="Path to exported human audit JSON")
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to save analysis reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"[!] Warning: Audit file not found at {input_path}")
        print("    If you just exported annotations from the web UI, please provide its path via --input <file.json>.")
        print("    Generating a demonstrative sample annotation structure for inspection...")
        sample_records = [
            {
                "id": "000009__v23__10_0.4521_0.3214",
                "stem": "000009",
                "image_name": "000009.jpg",
                "model_id": "v23",
                "model_name": "Sovereign V23",
                "class_id": 10,
                "class_name": "uav",
                "class_zh": "无人机 (uav)",
                "box": [0.4521, 0.3214, 0.0245, 0.0189],
                "score": 0.62,
                "role": "fp",
                "status": "true",
                "note": "两波长均有明显热源结构，确实为暗弱无人机，F5遗漏",
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "000009__v23__7_0.1245_0.8872",
                "stem": "000009",
                "image_name": "000009.jpg",
                "model_id": "v23",
                "class_id": 7,
                "class_name": "ball",
                "class_zh": "球 (ball)",
                "box": [0.1245, 0.8872, 0.0150, 0.0150],
                "score": 0.51,
                "role": "fp",
                "status": "false",
                "note": "地面杂波误报，无球体特征",
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "000012__f5__0_0.6120_0.4510",
                "stem": "000012",
                "image_name": "000012.jpg",
                "model_id": "f5",
                "class_id": 0,
                "class_name": "person",
                "class_zh": "人 (person)",
                "box": [0.6120, 0.4510, 0.0350, 0.0820],
                "score": 0.78,
                "role": "missed",
                "status": "true",
                "note": "标杆正确检出远距离行人",
                "timestamp": datetime.now().isoformat()
            }
        ]
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump({'annotations': sample_records}, f, ensure_ascii=False, indent=2)
        print(f"    Sample annotations created at: {input_path}")

    records = load_annotations(input_path)
    print(f"\nSuccessfully loaded {len(records)} audit annotations from: {input_path}")
    stats = analyze_audit_records(records)

    print("\n" + "="*60)
    print(f"📊 人工复审标定深度分析报告 (AUDIT REPORT)")
    print("="*60)
    print(f"已核验目标总数: {stats['total']}")
    print(f"  ✓ 真实有效目标 (TP): {stats['by_status'].get('true', 0)}")
    print(f"  ✗ 虚警误报伪框 (FP): {stats['by_status'].get('false', 0)}")
    print(f"  ★ 标杆漏报真值 (FN): {stats['by_status'].get('missed', 0)}")
    print(f"  ? 存疑待核查 (Uncertain): {stats['by_status'].get('uncertain', 0)}")
    print("-" * 60)
    print(f"覆盖图片数: {len(stats['stems_reviewed'])} 张")
    print("各模型标注分布:")
    for mid, mstats in sorted(stats['by_model'].items()):
        print(f"  [{mid.upper()}]: 真={mstats.get('true',0)}, 伪={mstats.get('false',0)}, 漏真={mstats.get('missed',0)}")
    print("="*60 + "\n")

    # Generate Reports & Artifacts
    report_md = Path(args.output_dir) / 'audit_summary_report.md'
    generate_markdown_report(stats, report_md)

    pseudo_gt_json = Path(args.output_dir) / 'human_refined_ground_truth.json'
    export_refined_pseudo_ground_truth(stats['tp_records'] + stats['missed_gt_records'], pseudo_gt_json)

    fp_patterns_json = Path(args.output_dir) / 'fp_blacklist_patterns.json'
    export_fp_blacklist_patterns(stats['fp_records'], fp_patterns_json)

    print("All audit analysis artifacts exported successfully!")


if __name__ == '__main__':
    main()
