#!/usr/bin/env python3
"""
AIC 2026 Bidirectional AI <-> Human Audit Data Bridge
=====================================================
1. AI -> Web System:
   Generates and exports standardized AI diagnostic case studies (e.g. Cross-category confusion,
   pure-missed targets, high-confidence discrepancies) to `outputs/analysis/audit/ai_focus_cases.json`.
   The web inspection system directly ingests this JSON, allowing users to jump to problem images
   and auto-zoom onto the exact discrepancy boxes diagnosed by AI!

2. Web System (Human Audits) -> AI:
   Reads human-verified annotations and notes from `outputs/analysis/audit/aic_human_audit_annotations.json`,
   compiles an AI Knowledge Digest (`outputs/analysis/audit/ai_knowledge_digest.json`), and provides
   a query interface so AI can automatically understand human findings, user comments, and rejected false alarms.
"""

import os
import sys
import json
import zipfile
import argparse
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path

ROOT_DIR = Path('/home/constant/CodeBuddy/Dev/AIC')
OUTPUT_DIR = ROOT_DIR / 'outputs'
AUDIT_DIR = OUTPUT_DIR / 'audit'

F5_ZIP = ROOT_DIR / 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip'
V26_ZIP = OUTPUT_DIR / 'submissions_v26/submission_sovereign_v26_rare45_common50_clean7241.zip'
V23_ZIP = Path('/home/constant/Downloads/submission_v23_flip_tta_thr45.zip')

CLASS_NAMES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}
CLASS_ZH = {
    0: '人 (person)', 1: '船 (boat)', 2: '动物 (animal)', 3: '座椅 (seat)',
    4: '标牌 (sign)', 5: '单车 (bicycle)', 6: '汽车 (car)', 7: '球 (ball)',
    8: '路灯 (light)', 9: '垃圾桶 (garbage can)', 10: '无人机 (uav)', 11: '三轮车 (tricycle)'
}
CLASS_ZH_SHORT = {
    0: '人', 1: '船', 2: '动物', 3: '座椅', 4: '标牌', 5: '单车',
    6: '汽车', 7: '球', 8: '路灯', 9: '垃圾桶', 10: '无人机', 11: '三轮车'
}

def load_predictions(zip_path):
    d = {}
    if not zip_path.exists():
        return d
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for f in zf.namelist():
            if not f.endswith('.txt'): continue
            stem = os.path.basename(f).rsplit('.', 1)[0]
            boxes = []
            for line in zf.read(f).decode('utf-8').strip().split('\n'):
                p = line.strip().split()
                if len(p) >= 6:
                    boxes.append([int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])])
            d[stem] = boxes
    return d

def compute_iou(b1, b2):
    x1_1, x2_1 = b1[1] - b1[3]/2, b1[1] + b1[3]/2
    y1_1, y2_1 = b1[2] - b1[4]/2, b1[2] + b1[4]/2
    x1_2, x2_2 = b2[1] - b2[3]/2, b2[1] + b2[3]/2
    y1_2, y2_2 = b2[2] - b2[4]/2, b2[2] + b2[4]/2
    ix = max(0.0, min(x2_1, x2_2) - max(x1_1, x1_2))
    iy = max(0.0, min(y2_1, y2_2) - max(y1_1, y1_2))
    inter = ix * iy
    union = b1[3]*b1[4] + b2[3]*b2[4] - inter
    return inter / union if union > 0 else 0.0

def generate_ai_cross_category_cases(f5_preds, cand_preds, cand_id='v26', min_iou=0.45, conf_thr=0.45):
    """
    Mine cross-category confusion cases where Candidate model and F5 point to the same object
    (IoU >= min_iou) but predict conflicting semantic classes.
    """
    cases = []
    confusion_pairs = []

    for stem, bf_list in sorted(f5_preds.items()):
        bg_list = cand_preds.get(stem, [])
        for f in bf_list:
            if f[5] < conf_thr: continue
            for g in bg_list:
                if g[5] < conf_thr: continue
                iou = compute_iou(f, g)
                if iou >= min_iou and f[0] != g[0]:
                    c_f5 = f[0]
                    c_cand = g[0]
                    confusion_pairs.append((c_f5, c_cand))

                    # Specific scenario diagnosis
                    scenario_notes = {
                        (5, 0): "车把/车架在红外低反射区被语义误激活为人身躯干，特征空间漂移",
                        (0, 5): "路边站立或行走的人被误识别为自行车，外形轮廓混淆",
                        (5, 3): "自行车后座/横梁被识别为椅子/长凳",
                        (3, 0): "长椅靠背被识别为人体躯干",
                        (0, 3): "蹲姿/坐姿行人被误判为座椅",
                        (4, 8): "标牌立杆与反光面被误激活为路灯",
                        (2, 1): "陆地动物误判为船只",
                        (6, 1): "路边泊车误判为船只",
                        (3, 2): "椅子被误判为动物"
                    }
                    diag = scenario_notes.get((c_f5, c_cand), f"在同一空间位置(IoU={iou:.2f})上发生跨类别语义错位")

                    cases.append({
                        "id": f"case_{stem}_{c_f5}to{c_cand}_{int(iou*100)}",
                        "stem": stem,
                        "title": f"{CLASS_ZH_SHORT[c_f5]} 误认成 {CLASS_ZH_SHORT[c_cand]}",
                        "category_confusion": f"{CLASS_NAMES[c_f5]} -> {CLASS_NAMES[c_cand]}",
                        "f5_class_id": c_f5,
                        "f5_class_zh": CLASS_ZH[c_f5],
                        "f5_score": round(f[5], 2),
                        "cand_class_id": c_cand,
                        "cand_class_zh": CLASS_ZH[c_cand],
                        "cand_score": round(g[5], 2),
                        "iou": round(iou, 3),
                        "focus_box": [round(g[1], 4), round(g[2], 4), round(g[3], 4), round(g[4], 4)],
                        "ai_diagnosis": diag,
                        "suggested_action": "请在核验卡片中核验真实属性：若F5正确点【真】，若候选多报点【伪】"
                    })

    # Sort cases by iou descending
    cases.sort(key=lambda x: x['iou'], reverse=True)

    counts = Counter(confusion_pairs)
    summary_stats = []
    for (c1, c2), num in counts.most_common(12):
        summary_stats.append({
            "pair": f"{CLASS_ZH_SHORT[c1]} -> {CLASS_ZH_SHORT[c2]}",
            "count": num,
            "f5_class": CLASS_NAMES[c1],
            "cand_class": CLASS_NAMES[c2]
        })

    package = {
        "version": "1.0",
        "case_id": f"ai_cross_category_confusion_{cand_id}",
        "title": f"{cand_id.upper()} 跨类别严重错位实测证据 (语义特征对撞诊断)",
        "author": "Antigravity AI Diagnostic Engine",
        "target_model": cand_id,
        "baseline_model": "f5",
        "created_at": datetime.now().isoformat(),
        "total_cases": len(cases),
        "confusion_ranking": summary_stats,
        "description": "全量1000张测试集对撞中提取的跨类别混淆错乱样本（自行车误认成人、长椅误认成人、标牌误认成路灯等），可一键导入审查。",
        "cases": cases
    }
    return package

def export_ai_cases_to_system(package, out_path):
    """Write AI cases to outputs/analysis/audit/ai_focus_cases.json for web ingestion."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(package, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(package['cases'])} AI Cases to: {out_path}")

def digest_human_audits_for_ai(audit_json_path, digest_out_path):
    """
    Read human audit annotations from web UI, aggregate user feedback,
    and output a comprehensive Knowledge Digest for AI to read and comprehend.
    """
    if not audit_json_path.exists():
        print(f"Audit file not found: {audit_json_path}")
        return None

    with open(audit_json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    items = raw if isinstance(raw, list) else raw.get('annotations', list(raw.values()))

    digest = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_human_audits": len(items),
            "stems_audited": list(set(it.get('stem', '') for it in items if it.get('stem'))),
        },
        "statistics": {
            "true_positives": sum(1 for it in items if it.get('status') == 'true'),
            "false_positives": sum(1 for it in items if it.get('status') == 'false'),
            "missed_ground_truths": sum(1 for it in items if it.get('status') == 'missed'),
            "uncertain": sum(1 for it in items if it.get('status') == 'uncertain')
        },
        "user_insights_by_class": defaultdict(lambda: {"true": 0, "false": 0, "missed": 0, "notes": []}),
        "confirmed_false_positives": [],
        "confirmed_true_positives": [],
        "key_user_notes": []
    }

    for it in items:
        cid = int(it.get('class_id', 0))
        st = it.get('status', 'unknown')
        note = it.get('note', '').strip()
        stem = it.get('stem', '')

        if st in digest['user_insights_by_class'][cid]:
            digest['user_insights_by_class'][cid][st] += 1
        if note:
            digest['user_insights_by_class'][cid]['notes'].append(note)
            digest['key_user_notes'].append({
                "stem": stem,
                "class_id": cid,
                "class_zh": CLASS_ZH.get(cid, str(cid)),
                "status": st,
                "note": note
            })

        if st == 'false':
            digest['confirmed_false_positives'].append({
                "stem": stem,
                "class_id": cid,
                "class_name": it.get('class_name', CLASS_NAMES.get(cid, '')),
                "model_id": it.get('model_id', ''),
                "score": it.get('score', 0),
                "box": it.get('box', []),
                "user_note": note
            })
        elif st in ('true', 'missed'):
            digest['confirmed_true_positives'].append({
                "stem": stem,
                "class_id": cid,
                "class_name": it.get('class_name', CLASS_NAMES.get(cid, '')),
                "model_id": it.get('model_id', ''),
                "is_missed_by_f5": (st == 'missed'),
                "score": it.get('score', 0),
                "box": it.get('box', []),
                "user_note": note
            })

    # Save digest
    digest_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(digest_out_path, 'w', encoding='utf-8') as f:
        json.dump(digest, f, ensure_ascii=False, indent=2)

    print(f"Generated AI Knowledge Digest: {digest_out_path}")
    print(f"  Total Reviewed: {digest['statistics']}")
    print(f"  User Notes Recorded: {len(digest['key_user_notes'])}")
    return digest

def main():
    parser = argparse.ArgumentParser(description="Bidirectional AI <-> Human Audit Bridge")
    parser.add_argument('--generate-cases', action='store_true', help="Mine and export AI diagnostic cases to outputs/analysis/audit/ai_focus_cases.json")
    parser.add_argument('--model', type=str, default='v26', help="Candidate model id (default: v26)")
    parser.add_argument('--digest-human-audits', action='store_true', help="Parse human audit feedback for AI comprehension")
    parser.add_argument('--query-sample', type=str, default='', help="Query audit and prediction status for a specific stem")
    args = parser.parse_args()

    ai_cases_path = AUDIT_DIR / 'ai_focus_cases.json'
    human_audit_path = AUDIT_DIR / 'aic_human_audit_annotations.json'
    ai_digest_path = AUDIT_DIR / 'ai_knowledge_digest.json'

    # Default action: run both if no flags specified
    if not (args.generate_cases or args.digest_human_audits or args.query_sample):
        args.generate_cases = True
        args.digest_human_audits = True

    if args.generate_cases:
        print(f"Mining AI discrepancy cases for {args.model.upper()} vs F5...")
        f5_preds = load_predictions(F5_ZIP)
        cand_zip = V26_ZIP if args.model == 'v26' else V23_ZIP
        cand_preds = load_predictions(cand_zip)
        pkg = generate_ai_cross_category_cases(f5_preds, cand_preds, cand_id=args.model)
        export_ai_cases_to_system(pkg, ai_cases_path)

    if args.digest_human_audits:
        print("Digesting human audit feedback for AI comprehension...")
        digest_human_audits_for_ai(human_audit_path, ai_digest_path)

    if args.query_sample:
        stem = args.query_sample.strip()
        print(f"\nQuerying Stem: {stem}")
        if ai_digest_path.exists():
            with open(ai_digest_path) as f:
                d = json.load(f)
            relevant_notes = [n for n in d.get('key_user_notes', []) if n.get('stem') == stem]
            print(f"User Audit Notes on {stem}: {relevant_notes}")

if __name__ == '__main__':
    main()
