#!/usr/bin/env python3
"""
Build Dedicated High-Performance Interactive Comparison Webpage:
"V23 遗漏 F5 真框 vs V23 独有多报伪框"
Outputs to:
  outputs/analysis/discrepancies_f5_vs_v23/index.html
  outputs/analysis/discrepancies_f5_vs_v23/v23_missed_vs_false_positives.html
  outputs/analysis/discrepancies_f5_vs_v23/discrepancy_report.json
"""

import os
import sys
import json
import zipfile
from collections import defaultdict, Counter

F5_DIR = '/home/constant/Downloads/AIC2026_phase2_predictions_F5seed20261002_RARE45_COMMON50_compact7212_20261005'
V23_ZIP = '/home/constant/Downloads/submission_v23_flip_tta_thr45.zip'
OUTPUT_DIR = '/home/constant/CodeBuddy/Dev/AIC/outputs/analysis/discrepancies_f5_vs_v23'

CLASS_NAMES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}
CLASS_ZH = {
    0: '人 (person)', 1: '船 (boat)', 2: '动物 (animal)', 3: '座椅 (seat)',
    4: '标牌 (sign)', 5: '单车 (bicycle)', 6: '汽车 (car)', 7: '球 (ball)',
    8: '路灯 (light)', 9: '垃圾桶 (garbage)', 10: '无人机 (uav)', 11: '三轮车 (tricycle)'
}
CLASS_ZH_SHORT = {
    0: '人', 1: '船', 2: '动物', 3: '座椅', 4: '标牌', 5: '单车',
    6: '汽车', 7: '球', 8: '路灯', 9: '垃圾桶', 10: '无人机', 11: '三轮车'
}
RARE_CLASSES = {1, 7, 9, 10, 11}

def load_f5():
    d = {}
    for fname in sorted(os.listdir(F5_DIR)):
        if not fname.endswith('.txt'): continue
        stem = fname.rsplit('.', 1)[0]
        with open(os.path.join(F5_DIR, fname)) as f:
            lines = [l.strip() for l in f if l.strip()]
        boxes = []
        for l in lines:
            p = l.split()
            cid = int(p[0])
            cx, cy, w, h, sc = map(float, p[1:])
            thr = 0.45 if cid in RARE_CLASSES else 0.50
            if sc >= thr:
                boxes.append((cid, cx, cy, w, h, sc))
        boxes.sort(key=lambda x: x[5], reverse=True)
        d[stem] = boxes[:60]
    return d

def load_v23():
    d = {}
    with zipfile.ZipFile(V23_ZIP) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith('.txt'): continue
            stem = name.rsplit('.', 1)[0]
            lines = zf.read(name).decode().strip().split('\n')
            boxes = []
            for l in lines:
                if not l.strip(): continue
                p = l.split()
                cid = int(p[0])
                cx, cy, w, h, sc = map(float, p[1:])
                boxes.append((cid, cx, cy, w, h, sc))
            boxes.sort(key=lambda x: x[5], reverse=True)
            d[stem] = boxes[:60]
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

def match_boxes(bf, bg):
    pairs_exact = []
    for i, b_f in enumerate(bf):
        for j, b_g in enumerate(bg):
            if b_f[0] == b_g[0]:
                iou = compute_iou(b_f, b_g)
                if iou >= 0.25:
                    pairs_exact.append((iou, i, j))
    pairs_exact.sort(key=lambda x: x[0], reverse=True)
    
    matched_f = {}
    matched_g = {}
    for iou, i, j in pairs_exact:
        if i not in matched_f and j not in matched_g:
            matched_f[i] = ('exact', j, iou)
            matched_g[j] = ('exact', i, iou)
            
    pairs_conflict = []
    for i, b_f in enumerate(bf):
        if i in matched_f: continue
        for j, b_g in enumerate(bg):
            if j in matched_g: continue
            if b_f[0] != b_g[0]:
                iou = compute_iou(b_f, b_g)
                if iou >= 0.30:
                    pairs_conflict.append((iou, i, j))
    pairs_conflict.sort(key=lambda x: x[0], reverse=True)
    
    for iou, i, j in pairs_conflict:
        if i not in matched_f and j not in matched_g:
            matched_f[i] = ('conflict', j, iou)
            matched_g[j] = ('conflict', i, iou)
            
    return matched_f, matched_g

def build_data():
    print("Loading predictions...")
    f5_data = load_f5()
    v23_data = load_v23()
    
    all_stems = sorted(f5_data.keys())
    print(f"Total test stems: {len(all_stems)}")
    
    items = []
    
    # Global class stats
    stats_cls_missed = Counter()
    stats_cls_fp = Counter()
    stats_cls_consensus = Counter()
    stats_cls_conflict = Counter()
    
    total_f5_boxes = 0
    total_v23_boxes = 0
    total_consensus_boxes = 0
    total_conflict_boxes = 0
    total_f5_only_boxes = 0
    total_v23_only_boxes = 0
    
    for stem in all_stems:
        bf = f5_data.get(stem, [])
        bg = v23_data.get(stem, [])
        total_f5_boxes += len(bf)
        total_v23_boxes += len(bg)
        
        mf, mg = match_boxes(bf, bg)
        
        num_conflicts = sum(1 for v in mf.values() if v[0] == 'conflict')
        num_exact = sum(1 for v in mf.values() if v[0] == 'exact')
        total_consensus_boxes += num_exact
        total_conflict_boxes += num_conflicts
        
        for i, b in enumerate(bf):
            if i in mf:
                if mf[i][0] == 'exact':
                    stats_cls_consensus[b[0]] += 1
                else:
                    stats_cls_conflict[b[0]] += 1
                    
        f5_only = []
        for i, b in enumerate(bf):
            if i not in mf:
                f5_only.append({
                    'cid': b[0],
                    'cname': CLASS_NAMES[b[0]],
                    'czh': CLASS_ZH_SHORT[b[0]],
                    'score': round(b[5], 3),
                    'box': [round(b[1], 4), round(b[2], 4), round(b[3], 4), round(b[4], 4)]
                })
                stats_cls_missed[b[0]] += 1
                total_f5_only_boxes += 1
                
        v23_only = []
        for j, b in enumerate(bg):
            if j not in mg:
                v23_only.append({
                    'cid': b[0],
                    'cname': CLASS_NAMES[b[0]],
                    'czh': CLASS_ZH_SHORT[b[0]],
                    'score': round(b[5], 3),
                    'box': [round(b[1], 4), round(b[2], 4), round(b[3], 4), round(b[4], 4)]
                })
                stats_cls_fp[b[0]] += 1
                total_v23_only_boxes += 1
                
        conflicts = []
        for i, v in mf.items():
            if v[0] == 'conflict':
                j, iou = v[1], v[2]
                conflicts.append({
                    'f5_cid': bf[i][0],
                    'f5_cname': CLASS_NAMES[bf[i][0]],
                    'f5_czh': CLASS_ZH_SHORT[bf[i][0]],
                    'f5_score': round(bf[i][5], 3),
                    'v23_cid': bg[j][0],
                    'v23_cname': CLASS_NAMES[bg[j][0]],
                    'v23_czh': CLASS_ZH_SHORT[bg[j][0]],
                    'v23_score': round(bg[j][5], 3),
                    'iou': round(iou, 2)
                })
                
        delta = len(f5_only) + len(v23_only)
        if num_conflicts == 0 and delta == 0:
            continue
            
        summary_parts = []
        if num_conflicts > 0:
            for c in conflicts:
                summary_parts.append(f"⚠️ 冲突: F5:{c['f5_czh']} ⚔️ V23:{c['v23_czh']} (IoU={c['iou']:.2f})")
        if len(f5_only) > 0:
            summary_parts.append(f"🟦 V23遗漏真框 {len(f5_only)}个")
        if len(v23_only) > 0:
            summary_parts.append(f"🟥 V23多报伪框 {len(v23_only)}个")
            
        max_f5_score = max([x['score'] for x in f5_only]) if f5_only else 0.0
        max_v23_score = max([x['score'] for x in v23_only]) if v23_only else 0.0
        
        has_f5 = len(f5_only) > 0
        has_v23 = len(v23_only) > 0
        is_pure_miss = has_f5 and not has_v23 and num_conflicts == 0
        is_pure_fp = has_v23 and not has_f5 and num_conflicts == 0
        is_mixed = has_f5 and has_v23
        
        items.append({
            'stem': stem,
            'f5_cnt': len(bf),
            'v23_cnt': len(bg),
            'exact_cnt': num_exact,
            'conf_cnt': num_conflicts,
            'delta': delta,
            'f5_only_cnt': len(f5_only),
            'v23_only_cnt': len(v23_only),
            'f5_only': f5_only,
            'v23_only': v23_only,
            'conflicts': conflicts,
            'f5_cids': sorted(list(set(x['cid'] for x in f5_only))),
            'v23_cids': sorted(list(set(x['cid'] for x in v23_only))),
            'has_f5': has_f5,
            'has_v23': has_v23,
            'is_pure_miss': is_pure_miss,
            'is_pure_fp': is_pure_fp,
            'is_mixed': is_mixed,
            'max_f5_score': max_f5_score,
            'max_v23_score': max_v23_score,
            'summary': " | ".join(summary_parts)
        })
        
    print(f"Total diff items: {len(items)}")
    
    global_meta = {
        'total_images': len(all_stems),
        'diff_images': len(items),
        'identical_images': len(all_stems) - len(items),
        'f5_total_boxes': total_f5_boxes,
        'v23_total_boxes': total_v23_boxes,
        'consensus_boxes': total_consensus_boxes,
        'conflict_boxes': total_conflict_boxes,
        'f5_only_boxes': total_f5_only_boxes,
        'v23_only_boxes': total_v23_only_boxes,
        'cls_missed': {cid: stats_cls_missed[cid] for cid in range(12)},
        'cls_fp': {cid: stats_cls_fp[cid] for cid in range(12)},
        'cls_consensus': {cid: stats_cls_consensus[cid] for cid in range(12)},
        'class_zh': CLASS_ZH,
        'class_zh_short': CLASS_ZH_SHORT,
    }
    
    return items, global_meta

def generate_html(items, global_meta):
    json_data = json.dumps({'items': items, 'meta': global_meta}, ensure_ascii=False)
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIC2026: V23 遗漏 F5 真框 vs V23 独有多报伪框 深度对比工作台</title>
    <style>
        :root {{
            --bg-dark: #0f1115;
            --bg-card: #181b21;
            --bg-card-hover: #212630;
            --border: #282e3b;
            --text-main: #f3f4f6;
            --text-dim: #94a3b8;
            --text-muted: #64748b;
            --f5-color: #06b6d4;      /* Electric Cyan */
            --f5-bg: rgba(6, 182, 212, 0.12);
            --v23-color: #f43f5e;     /* Crimson Pink/Red */
            --v23-bg: rgba(244, 63, 94, 0.12);
            --consensus-color: #10b981; /* Emerald */
            --conflict-color: #f59e0b;  /* Amber */
            --primary: #3b82f6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }}
        body {{ background: var(--bg-dark); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; overflow-x: hidden; }}
        
        /* Top Navigation Header */
        header {{
            background: #090a0d;
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 1000;
            backdrop-filter: blur(12px);
        }}
        .brand {{ display: flex; align-items: center; gap: 14px; }}
        .brand-badge {{
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: #fff;
            font-size: 11px;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }}
        .brand-title {{ font-size: 17px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }}
        
        /* KPI Metrics Dashboard Bar */
        .kpi-bar {{
            background: #12151b;
            border-bottom: 1px solid var(--border);
            padding: 10px 24px;
            display: flex;
            align-items: center;
            gap: 16px;
            overflow-x: auto;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            white-space: nowrap;
        }}
        .kpi-val {{ font-size: 16px; font-weight: 700; }}
        .kpi-lbl {{ font-size: 12px; color: var(--text-dim); }}
        
        /* Filter Controls */
        .controls-section {{
            background: #14171e;
            border-bottom: 1px solid var(--border);
            padding: 10px 24px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .tab-group {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
        .tab-btn {{
            background: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .tab-btn:hover {{ background: var(--bg-card-hover); border-color: #3b4252; }}
        .tab-btn.active {{ background: #2563eb; border-color: #3b82f6; color: #fff; box-shadow: 0 0 12px rgba(37,99,235,0.4); }}
        .tab-btn.tab-f5.active {{ background: #0891b2; border-color: var(--f5-color); box-shadow: 0 0 12px rgba(6,182,212,0.4); }}
        .tab-btn.tab-v23.active {{ background: #e11d48; border-color: var(--v23-color); box-shadow: 0 0 12px rgba(244,63,94,0.4); }}
        .tab-btn.tab-pure.active {{ background: #4f46e5; border-color: #6366f1; }}
        
        .sub-filters {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 12px; }}
        .filter-label {{ color: var(--text-muted); font-weight: 600; }}
        .chip {{
            background: #1e222a;
            border: 1px solid #2d3340;
            color: var(--text-dim);
            padding: 3px 9px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.15s;
        }}
        .chip:hover {{ color: #fff; border-color: #4b5563; }}
        .chip.active {{ background: #374151; color: #fff; border-color: #9ca3af; font-weight: 600; }}
        .search-box {{
            background: #1a1e26;
            border: 1px solid var(--border);
            color: #fff;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            width: 140px;
        }}
        .search-box:focus {{ outline: none; border-color: var(--primary); }}
        
        /* Main Workspace Layout */
        main {{ flex: 1; padding: 16px 24px; display: flex; flex-direction: column; align-items: center; gap: 14px; }}
        
        /* Image Info & Box Inspector Panel */
        .info-card {{
            width: 100%;
            max-width: 1920px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .info-row1 {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .stem-title {{ font-size: 18px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 12px; }}
        .nav-controls {{ display: flex; gap: 8px; align-items: center; }}
        .action-btn {{
            background: #2563eb;
            color: #fff;
            border: none;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s;
        }}
        .action-btn:hover {{ background: #1d4ed8; }}
        
        /* Box Breakdown Tags */
        .box-breakdown {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}
        @media (max-width: 1024px) {{ .box-breakdown {{ grid-template-columns: 1fr; }} }}
        
        .box-list-card {{
            background: #121418;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .box-list-title {{
            font-size: 13px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .box-tags-wrap {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            max-height: 100px;
            overflow-y: auto;
        }}
        .tag-f5 {{
            background: var(--f5-bg);
            border: 1px solid rgba(6,182,212,0.4);
            color: var(--f5-color);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}
        .tag-v23 {{
            background: var(--v23-bg);
            border: 1px solid rgba(244,63,94,0.4);
            color: var(--v23-color);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}
        .tag-conf {{
            background: rgba(245,158,11,0.12);
            border: 1px solid rgba(245,158,11,0.4);
            color: var(--conflict-color);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        /* Inspection Image Container */
        .stage-container {{
            width: 100%;
            max-width: 1920px;
            background: #08090b;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            box-shadow: 0 8px 30px rgba(0,0,0,0.5);
            position: relative;
        }}
        .stage-container img {{
            width: 100%;
            height: auto;
            display: block;
            object-fit: contain;
            cursor: zoom-in;
        }}
        
        /* Modal for Fullscreen Zoom */
        .zoom-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.92);
            z-index: 2000;
            display: none;
            justify-content: center;
            align-items: center;
            padding: 20px;
            cursor: zoom-out;
        }}
        .zoom-overlay img {{
            max-width: 98vw;
            max-height: 98vh;
            object-fit: contain;
            box-shadow: 0 0 40px rgba(0,0,0,0.8);
        }}
        
        /* Quick Navigation Thumbnails Carousel */
        .thumb-carousel {{
            width: 100%;
            max-width: 1920px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            overflow-x: auto;
            white-space: nowrap;
        }}
        .thumb-item {{
            display: inline-block;
            background: #121418;
            border: 1px solid #2d3340;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.15s;
            color: var(--text-dim);
        }}
        .thumb-item:hover {{ background: #222733; color: #fff; }}
        .thumb-item.active {{ background: #2563eb; color: #fff; border-color: #3b82f6; font-weight: 700; }}
        
        .footer-tips {{
            font-size: 12px;
            color: var(--text-muted);
            text-align: center;
            padding-bottom: 20px;
        }}
        kbd {{ background: #232732; border: 1px solid #3c4456; border-radius: 3px; padding: 1px 5px; font-size: 11px; }}
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="brand">
            <div class="brand-badge">AIC 2026 AUDIT</div>
            <div class="brand-title">
                <span>V23 遗漏 F5 真框</span>
                <span style="color:var(--text-muted);font-weight:400;">⚔️</span>
                <span>V23 独有多报伪框</span>
                <span style="font-size:13px;color:var(--text-dim);font-weight:400;">(对比工作台)</span>
            </div>
        </div>
        <div class="nav-controls">
            <button class="action-btn" onclick="prevImg()">◀ 上一张 (←)</button>
            <span id="curIndexLabel" style="font-size:13px;font-weight:700;min-width:70px;text-align:center;">1 / 0</span>
            <button class="action-btn" onclick="nextImg()">下一张 (→) ▶</button>
        </div>
    </header>

    <!-- Global KPI Statistics Bar -->
    <div class="kpi-bar">
        <div class="kpi-card" style="border-left: 3px solid #3b82f6;">
            <div>
                <div class="kpi-val" style="color:#60a5fa;" id="kpiTotalDiff">692 张</div>
                <div class="kpi-lbl">存在差异图像 (共1000张)</div>
            </div>
        </div>
        <div class="kpi-card" style="border-left: 3px solid var(--f5-color);">
            <div>
                <div class="kpi-val" style="color:var(--f5-color);">373 张 (748 框)</div>
                <div class="kpi-lbl">📉 V23 遗漏 F5 真框 (高置信≥0.75: 127框)</div>
            </div>
        </div>
        <div class="kpi-card" style="border-left: 3px solid var(--v23-color);">
            <div>
                <div class="kpi-val" style="color:var(--v23-color);">554 张 (1,559 框)</div>
                <div class="kpi-lbl">📈 V23 独有多报伪框 (高置信≥0.75: 822框)</div>
            </div>
        </div>
        <div class="kpi-card" style="border-left: 3px solid var(--consensus-color);">
            <div>
                <div class="kpi-val" style="color:var(--consensus-color);">5,748 框</div>
                <div class="kpi-lbl">🟩 双方强共识真框 (IoU≥0.25)</div>
            </div>
        </div>
        <div class="kpi-card" style="border-left: 3px solid var(--conflict-color);">
            <div>
                <div class="kpi-val" style="color:var(--conflict-color);">27 处</div>
                <div class="kpi-lbl">⚠️ 同物体类别冲突</div>
            </div>
        </div>
        <div class="kpi-card">
            <div>
                <div class="kpi-val" style="color:#a78bfa;">308 张</div>
                <div class="kpi-lbl">💯 双方100%完全无差异图片</div>
            </div>
        </div>
    </div>

    <!-- Interactive Filters & Tabs -->
    <div class="controls-section">
        <!-- Main Mode Tabs -->
        <div class="tab-group">
            <span class="filter-label">核心模式：</span>
            <button class="tab-btn tab-f5 active" onclick="setMainMode('f5_missed')">
                <span>📉 V23 遗漏 F5 真框</span>
                <span style="opacity:0.8;">(<span id="cntF5Missed">373</span>)</span>
            </button>
            <button class="tab-btn tab-v23" onclick="setMainMode('v23_fp')">
                <span>📈 V23 独有多报伪框</span>
                <span style="opacity:0.8;">(<span id="cntV23FP">554</span>)</span>
            </button>
            <button class="tab-btn tab-pure" onclick="setMainMode('pure_miss')">
                <span>🎯 纯遗漏无多报</span>
                <span style="opacity:0.8;">(<span id="cntPureMiss">134</span>)</span>
            </button>
            <button class="tab-btn tab-pure" onclick="setMainMode('pure_fp')">
                <span>🚨 纯多报无遗漏</span>
                <span style="opacity:0.8;">(<span id="cntPureFP">315</span>)</span>
            </button>
            <button class="tab-btn" onclick="setMainMode('mixed')">
                <span>⚖️ 既漏又多 (交织)</span>
                <span style="opacity:0.8;">(<span id="cntMixed">239</span>)</span>
            </button>
            <button class="tab-btn" onclick="setMainMode('conflict')" style="color:#fbbf24;">
                <span>⚠️ 类别冲突</span>
                <span style="opacity:0.8;">(<span id="cntConflict">27</span>)</span>
            </button>
            <button class="tab-btn" onclick="setMainMode('all')">
                <span>🌐 全量差异</span>
                <span style="opacity:0.8;">(<span id="cntAll">692</span>)</span>
            </button>
        </div>

        <!-- Class Secondary Filter Chips -->
        <div class="sub-filters">
            <span class="filter-label">类别精确筛选：</span>
            <span class="chip active" onclick="setClassFilter(-1)" id="chipAllCls">全部类别</span>
            <span class="chip" onclick="setClassFilter(0)">人 (<span id="clsCnt0">0</span>)</span>
            <span class="chip" onclick="setClassFilter(2)">动物 (<span id="clsCnt2">0</span>)</span>
            <span class="chip" onclick="setClassFilter(8)">路灯 (<span id="clsCnt8">0</span>)</span>
            <span class="chip" onclick="setClassFilter(6)">汽车 (<span id="clsCnt6">0</span>)</span>
            <span class="chip" onclick="setClassFilter(4)">标牌 (<span id="clsCnt4">0</span>)</span>
            <span class="chip" onclick="setClassFilter(3)">座椅 (<span id="clsCnt3">0</span>)</span>
            <span class="chip" onclick="setClassFilter(9)">垃圾桶 (<span id="clsCnt9">0</span>)</span>
            <span class="chip" onclick="setClassFilter(5)">单车 (<span id="clsCnt5">0</span>)</span>
            <span class="chip" onclick="setClassFilter(1)">船 (<span id="clsCnt1">0</span>)</span>
            <span class="chip" onclick="setClassFilter(10)">无人机 (<span id="clsCnt10">0</span>)</span>
            <span class="chip" onclick="setClassFilter(7)">球 (<span id="clsCnt7">0</span>)</span>
            <span class="chip" onclick="setClassFilter(11)">三轮车 (<span id="clsCnt11">0</span>)</span>
            
            <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
                <span class="filter-label">仅看高置信(≥0.75):</span>
                <input type="checkbox" id="chkHighConf" onchange="toggleHighConf()" style="cursor:pointer;" />
                <span class="filter-label" style="margin-left:6px;">搜索图号:</span>
                <input type="text" class="search-box" id="searchBox" placeholder="输入图号..." oninput="onSearchInput()" />
            </div>
        </div>
    </div>

    <!-- Active Image Workspace -->
    <main>
        <!-- Info Card -->
        <div class="info-card">
            <div class="info-row1">
                <div class="stem-title">
                    <span id="stemTitle">【图号 00000000】</span>
                    <span id="stemCountsBadge" style="font-size:13px;font-weight:500;color:var(--text-dim);"></span>
                </div>
                <div id="stemSummaryText" style="font-size:13px;color:#fbbf24;font-weight:600;"></div>
            </div>

            <!-- Detailed Box Lists -->
            <div class="box-breakdown">
                <!-- Missed True Boxes -->
                <div class="box-list-card" style="border-left: 3px solid var(--f5-color);">
                    <div class="box-list-title">
                        <span style="color:var(--f5-color);">📉 V23 遗漏的 F5 真框 (F5 独有检出)</span>
                        <span id="f5OnlyCountBadge" style="font-size:12px;color:var(--text-muted);">0 个</span>
                    </div>
                    <div class="box-tags-wrap" id="f5OnlyTags">
                        <span style="font-size:12px;color:var(--text-muted);">无遗漏真框</span>
                    </div>
                </div>

                <!-- Over-predicted False Positives -->
                <div class="box-list-card" style="border-left: 3px solid var(--v23-color);">
                    <div class="box-list-title">
                        <span style="color:var(--v23-color);">📈 V23 独有多报的伪框/虚警 (F5 未检出)</span>
                        <span id="v23OnlyCountBadge" style="font-size:12px;color:var(--text-muted);">0 个</span>
                    </div>
                    <div class="box-tags-wrap" id="v23OnlyTags">
                        <span style="font-size:12px;color:var(--text-muted);">无多报伪框</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- 3-Column Inspection Panel Image -->
        <div class="stage-container">
            <img id="mainStageImg" src="" alt="对比面板加载中..." onclick="openZoom()" />
        </div>

        <!-- Quick Carousel Thumbnails List -->
        <div class="thumb-carousel" id="thumbCarousel"></div>

        <div class="footer-tips">
            快捷键导航：键盘左右键 <kbd>←</kbd> <kbd>→</kbd> 快速翻页，<kbd>1</kbd> 遗漏真框，<kbd>2</kbd> 多报伪框，<kbd>3</kbd> 纯遗漏，<kbd>4</kbd> 纯多报，<kbd>5</kbd> 类别冲突。点击图片可放大查看。
        </div>
    </main>

    <!-- Zoom Modal -->
    <div class="zoom-overlay" id="zoomOverlay" onclick="closeZoom()">
        <img id="zoomImg" src="" alt="放大图" />
    </div>

    <!-- Data Injection -->
    <script>
        const DATA = {json_data};
        const items = DATA.items;
        const meta = DATA.meta;

        let curMainMode = 'f5_missed';
        let curClassId = -1;
        let onlyHighConf = false;
        let searchQuery = '';
        
        let filteredItems = [];
        let curIndex = 0;

        function applyFilters() {{
            filteredItems = items.filter(item => {{
                // Main mode filter
                if (curMainMode === 'f5_missed' && !item.has_f5) return false;
                if (curMainMode === 'v23_fp' && !item.has_v23) return false;
                if (curMainMode === 'pure_miss' && !item.is_pure_miss) return false;
                if (curMainMode === 'pure_fp' && !item.is_pure_fp) return false;
                if (curMainMode === 'mixed' && !item.is_mixed) return false;
                if (curMainMode === 'conflict' && item.conf_cnt === 0) return false;

                // Class filter
                if (curClassId !== -1) {{
                    if (curMainMode === 'f5_missed' || curMainMode === 'pure_miss') {{
                        if (!item.f5_cids.includes(curClassId)) return false;
                    }} else if (curMainMode === 'v23_fp' || curMainMode === 'pure_fp') {{
                        if (!item.v23_cids.includes(curClassId)) return false;
                    }} else {{
                        if (!item.f5_cids.includes(curClassId) && !item.v23_cids.includes(curClassId)) return false;
                    }}
                }}

                // High confidence filter
                if (onlyHighConf) {{
                    if (curMainMode === 'f5_missed' || curMainMode === 'pure_miss') {{
                        if (item.max_f5_score < 0.75) return false;
                    }} else if (curMainMode === 'v23_fp' || curMainMode === 'pure_fp') {{
                        if (item.max_v23_score < 0.75) return false;
                    }} else {{
                        if (item.max_f5_score < 0.75 && item.max_v23_score < 0.75) return false;
                    }}
                }}

                // Search query
                if (searchQuery.trim() !== '') {{
                    if (!item.stem.toLowerCase().includes(searchQuery.trim().toLowerCase())) return false;
                }}

                return true;
            }});

            curIndex = 0;
            updateUI();
            updateThumbCarousel();
        }}

        function setMainMode(mode) {{
            curMainMode = mode;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            if (mode === 'f5_missed') document.querySelector('.tab-f5').classList.add('active');
            else if (mode === 'v23_fp') document.querySelector('.tab-v23').classList.add('active');
            else if (mode === 'pure_miss') document.querySelectorAll('.tab-pure')[0].classList.add('active');
            else if (mode === 'pure_fp') document.querySelectorAll('.tab-pure')[1].classList.add('active');
            else if (mode === 'mixed') document.querySelectorAll('.tab-btn')[4].classList.add('active');
            else if (mode === 'conflict') document.querySelectorAll('.tab-btn')[5].classList.add('active');
            else if (mode === 'all') document.querySelectorAll('.tab-btn')[6].classList.add('active');
            
            updateClassCounts();
            applyFilters();
        }}

        function setClassFilter(cid) {{
            curClassId = cid;
            document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
            if (cid === -1) {{
                document.getElementById('chipAllCls').classList.add('active');
            }} else {{
                const chips = Array.from(document.querySelectorAll('.chip'));
                chips.forEach(c => {{
                    if (c.getAttribute('onclick') === `setClassFilter(${{cid}})`) c.classList.add('active');
                }});
            }}
            applyFilters();
        }}

        function toggleHighConf() {{
            onlyHighConf = document.getElementById('chkHighConf').checked;
            applyFilters();
        }}

        function onSearchInput() {{
            searchQuery = document.getElementById('searchBox').value;
            applyFilters();
        }}

        function updateClassCounts() {{
            for (let cid = 0; cid < 12; cid++) {{
                const el = document.getElementById(`clsCnt${{cid}}`);
                if (!el) continue;
                if (curMainMode === 'f5_missed' || curMainMode === 'pure_miss') {{
                    el.innerText = meta.cls_missed[cid] || 0;
                }} else if (curMainMode === 'v23_fp' || curMainMode === 'pure_fp') {{
                    el.innerText = meta.cls_fp[cid] || 0;
                }} else {{
                    const total = (meta.cls_missed[cid] || 0) + (meta.cls_fp[cid] || 0);
                    el.innerText = total;
                }}
            }}
        }}

        function updateUI() {{
            const total = filteredItems.length;
            const curLabel = document.getElementById('curIndexLabel');
            if (total === 0) {{
                curLabel.innerText = "0 / 0";
                document.getElementById('stemTitle').innerText = "没有符合筛选条件的样本";
                document.getElementById('stemCountsBadge').innerText = "";
                document.getElementById('stemSummaryText').innerText = "";
                document.getElementById('f5OnlyTags').innerHTML = '<span style="color:var(--text-muted);font-size:12px;">无</span>';
                document.getElementById('v23OnlyTags').innerHTML = '<span style="color:var(--text-muted);font-size:12px;">无</span>';
                document.getElementById('mainStageImg').src = "";
                return;
            }}

            curLabel.innerText = `${{curIndex + 1}} / ${{total}}`;
            const curItem = filteredItems[curIndex];

            document.getElementById('stemTitle').innerText = `【图号 ${{curItem.stem}}】`;
            document.getElementById('stemCountsBadge').innerText = 
                `F5标杆: ${{curItem.f5_cnt}}框 | V23预测: ${{curItem.v23_cnt}}框 | 共识: ${{curItem.exact_cnt}}框 | 冲突: ${{curItem.conf_cnt}}处`;
            document.getElementById('stemSummaryText').innerText = curItem.summary;

            // Render F5 Only tags (V23 missed)
            const f5Wrap = document.getElementById('f5OnlyTags');
            document.getElementById('f5OnlyCountBadge').innerText = `${{curItem.f5_only_cnt}} 个`;
            if (curItem.f5_only.length === 0) {{
                f5Wrap.innerHTML = '<span style="font-size:12px;color:var(--text-muted);">无遗漏真框 (V23完全检出)</span>';
            }} else {{
                f5Wrap.innerHTML = curItem.f5_only.map(b => 
                    `<span class="tag-f5">🔷 ${{b.czh}} (${{b.score.toFixed(2)}})</span>`
                ).join('');
            }}

            // Render V23 Only tags (V23 false positives)
            const v23Wrap = document.getElementById('v23OnlyTags');
            document.getElementById('v23OnlyCountBadge').innerText = `${{curItem.v23_only_cnt}} 个`;
            if (curItem.v23_only.length === 0) {{
                v23Wrap.innerHTML = '<span style="font-size:12px;color:var(--text-muted);">无多报伪框 (无多余虚警)</span>';
            }} else {{
                v23Wrap.innerHTML = curItem.v23_only.map(b => 
                    `<span class="tag-v23">🔴 ${{b.czh}} (${{b.score.toFixed(2)}})</span>`
                ).join('');
            }}

            // Main Image
            const imgPath = `all_discrepancies/${{curItem.stem}}.png`;
            document.getElementById('mainStageImg').src = imgPath;
        }}

        function updateThumbCarousel() {{
            const car = document.getElementById('thumbCarousel');
            if (filteredItems.length === 0) {{
                car.innerHTML = '<span style="color:var(--text-muted);font-size:12px;padding:4px;">暂无样本</span>';
                return;
            }}
            car.innerHTML = filteredItems.map((item, idx) => {{
                let badge = '';
                if (item.f5_only_cnt > 0 && item.v23_only_cnt > 0) badge = `📉${{item.f5_only_cnt}} 📈${{item.v23_only_cnt}}`;
                else if (item.f5_only_cnt > 0) badge = `📉-${{item.f5_only_cnt}}`;
                else if (item.v23_only_cnt > 0) badge = `📈+${{item.v23_only_cnt}}`;
                else badge = `⚠️${{item.conf_cnt}}`;
                
                return `<div class="thumb-item ${{idx === curIndex ? 'active' : ''}}" onclick="jumpTo(${{idx}})">
                    ${{item.stem}} <span style="font-size:10px;opacity:0.8;">[${{badge}}]</span>
                </div>`;
            }}).join('');
            
            // Auto scroll active thumb
            const activeEl = car.querySelector('.thumb-item.active');
            if (activeEl) activeEl.scrollIntoView({{ behavior: 'smooth', block: 'nearest', inline: 'center' }});
        }}

        function jumpTo(idx) {{
            if (idx >= 0 && idx < filteredItems.length) {{
                curIndex = idx;
                updateUI();
                document.querySelectorAll('.thumb-item').forEach((el, i) => {{
                    if (i === curIndex) el.classList.add('active');
                    else el.classList.remove('active');
                }});
            }}
        }}

        function prevImg() {{
            if (filteredItems.length === 0) return;
            curIndex = (curIndex - 1 + filteredItems.length) % filteredItems.length;
            updateUI();
            updateThumbCarousel();
        }}

        function nextImg() {{
            if (filteredItems.length === 0) return;
            curIndex = (curIndex + 1) % filteredItems.length;
            updateUI();
            updateThumbCarousel();
        }}

        function openZoom() {{
            if (filteredItems.length === 0) return;
            const src = document.getElementById('mainStageImg').src;
            document.getElementById('zoomImg').src = src;
            document.getElementById('zoomOverlay').style.display = 'flex';
        }}

        function closeZoom() {{
            document.getElementById('zoomOverlay').style.display = 'none';
        }}

        // Keyboard hotkeys
        window.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'INPUT') return;
            if (e.key === 'ArrowLeft') prevImg();
            else if (e.key === 'ArrowRight') nextImg();
            else if (e.key === 'Escape') closeZoom();
            else if (e.key === '1') setMainMode('f5_missed');
            else if (e.key === '2') setMainMode('v23_fp');
            else if (e.key === '3') setMainMode('pure_miss');
            else if (e.key === '4') setMainMode('pure_fp');
            else if (e.key === '5') setMainMode('conflict');
            else if (e.key === '0') setMainMode('all');
        }});

        // Init
        document.addEventListener('DOMContentLoaded', () => {{
            updateClassCounts();
            applyFilters();
        }});
    </script>
</body>
</html>
'''
    return html

def main():
    items, meta = build_data()
    
    # Save structured json
    json_path = os.path.join(OUTPUT_DIR, 'discrepancy_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'items': items, 'meta': meta}, f, ensure_ascii=False, indent=2)
    print(f"Saved rich metadata to: {json_path}")
    
    # Generate HTML
    html_content = generate_html(items, meta)
    
    # Write to index.html and compare_missed_vs_false_positives.html
    html_path = os.path.join(OUTPUT_DIR, 'index.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Updated interactive gallery: {html_path}")
    
    html_alias = os.path.join(OUTPUT_DIR, 'compare_missed_vs_false_positives.html')
    with open(html_alias, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved alias page: {html_alias}")

if __name__ == '__main__':
    main()
