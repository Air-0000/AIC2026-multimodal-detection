#!/usr/bin/env python3
"""
AIC 2026 Multimodal Object Detection Inspection & Comparison Workstation
=========================================================================
State-of-the-art Cyber-Scientific Workstation featuring:
1. Neural Super-Resolution Engine:
   - Real-ESRGAN 4x deep learning reconstruction on RTX 5060 Ti GPU.
   - Full 4x Ultra-HD canvas view (2560x1440 / 5120x2880).
   - Before/After Interactive Curtain Swipe Comparison (1x Raw vs 4x Neural SR).
   - HUD Circular Reticle Loupe with 4x Super-Resolution pixel rendering.
   - Real-time on-demand GPU inference endpoint integration (/api/generate_sr).
2. Advanced Visual & Aesthetic Design:
   - Obsidian glassmorphism frosted interface with neon accent glows.
   - Sci-Fi HUD canvas telemetry overlay (resolution, zoom ratio, box stats).
   - Visual segmented progress bar for human audit verification.
   - Sleek target inspector cards with confidence gauge bars and category badges.
3. Universal Interactive Model Comparison:
   - Dynamic real-time bipartite IoU matching (F5 Benchmark vs V26 / V25 / V24 / V23).
   - 12-class dynamic filter matrix with instant re-calculation.
   - High-performance Pan & Zoom engine (up to 1500% magnification).
   - Split-screen synchronized dual viewport mode.
4. Bidirectional AI Diagnosis & Human Audit System:
   - Integrated AI Focus Cases Hub with cross-category confusion taxonomy.
   - One-click jump & auto-zoom onto problem targets with AI diagnostic notes.
   - Human audit tagging (✓ True TP, ✗ False FP, ★ Missed FN, ? Uncertain, ↺ Clear).
   - Dual persistence: LocalStorage + Backend JSON auto-sync (`/api/audit/save`).
"""

import os
import sys
import json
import zipfile
from pathlib import Path

ROOT_DIR = '/home/constant/CodeBuddy/Dev/AIC'
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
VIS_DIR = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/visible')
IR_DIR = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/infrared')
VIS_SR_DIR = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/visible_sr4x')
IR_SR_DIR = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/infrared_sr4x')

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
CLASS_COLORS = {
    0: '#38bdf8',  # light blue (person)
    1: '#0ea5e9',  # cyan (boat)
    2: '#34d399',  # emerald (animal)
    3: '#fbbf24',  # amber (seat)
    4: '#f97316',  # orange (sign)
    5: '#a78bfa',  # purple (bicycle)
    6: '#60a5fa',  # blue (car)
    7: '#f43f5e',  # rose red (ball)
    8: '#facc15',  # yellow (light)
    9: '#94a3b8',  # slate (garbage)
    10: '#ec4899', # pink (uav)
    11: '#c084fc'  # violet (tricycle)
}

MODELS_CONFIG = [
    {
        'id': 'v26',
        'name': 'Sovereign V26 Golden',
        'score': '58.10+',
        'badge': '7,241 框 / Golden Decoupled',
        'path': os.path.join(OUTPUT_DIR, 'submissions_v26/submission_sovereign_v26_rare45_common50_clean7241.zip'),
        'color': '#10b981'
    },
    {
        'id': 'v25',
        'name': 'Sovereign V25 Compact',
        'score': '56.40+',
        'badge': '7,200 框 / Perfect Union',
        'path': os.path.join(OUTPUT_DIR, 'submissions_v25/submission_v25_compact.zip'),
        'color': '#a855f7'
    },
    {
        'id': 'v24',
        'name': 'Sovereign V24 Native',
        'score': '53.555',
        'badge': '6,711 框 / Native Compact',
        'path': os.path.join(OUTPUT_DIR, 'submissions_v24/submission_v24_native_compact.zip'),
        'color': '#3b82f6'
    },
    {
        'id': 'v23',
        'name': 'Sovereign V23 Flip TTA',
        'score': '53.00',
        'badge': '8,023 框 / Flip TTA',
        'path': '/home/constant/Downloads/submission_v23_flip_tta_thr45.zip',
        'color': '#f43f5e'
    }
]

BASELINE_CONFIG = {
    'id': 'f5',
    'name': 'F5 (57.266 黄金标杆)',
    'score': '57.266',
    'path': os.path.join(ROOT_DIR, 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip')
}

def parse_boxes_from_zip(zip_p):
    d = {}
    with zipfile.ZipFile(zip_p, 'r') as zf:
        for fname in sorted(zf.namelist()):
            if not fname.endswith('.txt'): continue
            stem = os.path.basename(fname).rsplit('.', 1)[0]
            content = zf.read(fname).decode('utf-8', errors='ignore').strip()
            boxes = []
            if content:
                for line in content.split('\n'):
                    p = line.strip().split()
                    if len(p) >= 6:
                        cid = int(p[0])
                        cx, cy, w, h, sc = map(float, p[1:6])
                        boxes.append([cid, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4), round(sc, 3)])
            boxes.sort(key=lambda x: x[5], reverse=True)
            d[stem] = boxes[:60]
    return d

print("Scanning test images...")
file_map = {}
for f in os.listdir(VIS_DIR):
    if f.endswith('.png') or f.endswith('.jpg'):
        stem = f.rsplit('.', 1)[0]
        file_map[stem] = f

all_stems = sorted(file_map.keys())
print(f"Total test images: {len(all_stems)}")

print("Loading predictions for Baseline and Candidate models...")
raw_predictions = {
    'f5': parse_boxes_from_zip(BASELINE_CONFIG['path'])
}
for m in MODELS_CONFIG:
    print(f"  Parsing {m['name']}...")
    raw_predictions[m['id']] = parse_boxes_from_zip(m['path'])

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

def quick_match(bf, bg, iou_exact=0.25, iou_conf=0.30):
    matched_f, matched_g = {}, {}
    pairs_exact = []
    for i, f in enumerate(bf):
        for j, g in enumerate(bg):
            if f[0] == g[0]:
                iou = compute_iou(f, g)
                if iou >= iou_exact:
                    pairs_exact.append((iou, i, j))
    pairs_exact.sort(key=lambda x: x[0], reverse=True)
    for iou, i, j in pairs_exact:
        if i not in matched_f and j not in matched_g:
            matched_f[i] = ('exact', j, iou)
            matched_g[j] = ('exact', i, iou)

    pairs_conf = []
    for i, f in enumerate(bf):
        if i in matched_f: continue
        for j, g in enumerate(bg):
            if j in matched_g or f[0] == g[0]: continue
            iou = compute_iou(f, g)
            if iou >= iou_conf:
                pairs_conf.append((iou, i, j))
    pairs_conf.sort(key=lambda x: x[0], reverse=True)
    for iou, i, j in pairs_conf:
        if i not in matched_f and j not in matched_g:
            matched_f[i] = ('conflict', j, iou)
            matched_g[j] = ('conflict', i, iou)

    n_exact = sum(1 for v in matched_f.values() if v[0] == 'exact')
    n_conf = sum(1 for v in matched_f.values() if v[0] == 'conflict')
    n_missed = len(bf) - len(matched_f)
    n_fp = len(bg) - len(matched_g)
    return n_exact, n_conf, n_missed, n_fp

print("Pre-calculating discrepancy summaries for all candidate models...")
diff_summaries = {}
for m in MODELS_CONFIG:
    mid = m['id']
    preds = raw_predictions[mid]
    f5_preds = raw_predictions['f5']
    diff_list = []
    for stem in all_stems:
        bf = [b for b in f5_preds.get(stem, []) if b[5] >= 0.45]
        bg = [b for b in preds.get(stem, []) if b[5] >= 0.45]
        n_exact, n_conf, n_missed, n_fp = quick_match(bf, bg)
        delta = len(bg) - len(bf)

        missed_cids = [b[0] for i, b in enumerate(bf)]
        fp_cids = [b[0] for j, b in enumerate(bg)]

        diff_list.append({
            'stem': stem,
            'f_cnt': len(bf),
            'c_cnt': len(bg),
            'exact': n_exact,
            'conf': n_conf,
            'missed': n_missed,
            'fp': n_fp,
            'delta': delta,
            'missed_cids': missed_cids,
            'fp_cids': fp_cids,
            'is_pure_miss': n_missed > 0 and n_fp == 0 and n_conf == 0,
            'is_pure_fp': n_fp > 0 and n_missed == 0 and n_conf == 0,
            'is_mixed': n_missed > 0 and n_fp > 0,
            'is_conf': n_conf > 0
        })
    diff_summaries[mid] = diff_list

# Load AI Diagnostic Cases Package if available
ai_cases_p = os.path.join(OUTPUT_DIR, 'audit/ai_focus_cases.json')
ai_cases_pkg = {}
if os.path.exists(ai_cases_p):
    try:
        with open(ai_cases_p, 'r', encoding='utf-8') as f:
            ai_cases_pkg = json.load(f)
        print(f"Loaded {len(ai_cases_pkg.get('cases', []))} AI Diagnostic Focus Cases.")
    except Exception as e:
        print(f"Warning: Failed to load AI cases: {e}")

# Load alignment summary if available
align_summary_p = os.path.join(OUTPUT_DIR, 'test_alignment_summary.json')
align_map = {}
if os.path.exists(align_summary_p):
    try:
        with open(align_summary_p, 'r', encoding='utf-8') as f:
            al_data = json.load(f)
            for item in al_data.get('details', []):
                stem = item.get('stem')
                if stem:
                    align_map[stem] = {
                        'global_applied': item.get('global_applied', False),
                        'dx': round(item.get('dx', 0.0), 2),
                        'dy': round(item.get('dy', 0.0), 2),
                        'angle': round(item.get('angle', 0.0), 2),
                        'delta_cc': round(item.get('delta_cc', 0.0), 3),
                        'targets': item.get('local_snapped_targets', 0)
                    }
        print(f"Loaded {len(align_map)} IR-RGB pixel alignment telemetry profiles.")
    except Exception as e:
        print(f"Warning: Failed to load alignment summary: {e}")

# Scan available SR files
sr_available = {}
if os.path.exists(VIS_SR_DIR):
    for f in os.listdir(VIS_SR_DIR):
        if f.endswith('.jpg') or f.endswith('.png'):
            s = f.rsplit('.', 1)[0]
            sr_available[s] = True
print(f"Detected {len(sr_available)} pre-generated Real-ESRGAN 4x SR images.")

master_payload = {
    'classes': CLASS_NAMES,
    'class_zh': CLASS_ZH,
    'class_zh_short': CLASS_ZH_SHORT,
    'class_colors': CLASS_COLORS,
    'file_map': file_map,
    'models': MODELS_CONFIG,
    'baseline': BASELINE_CONFIG,
    'diff_summaries': diff_summaries,
    'predictions': raw_predictions,
    'ai_case_package': ai_cases_pkg,
    'sr_available': sr_available,
    'align_map': align_map
}

print(f"Compiling Cyber-Scientific Multimodal Super-Resolution & Audit Workstation...")
json_data_str = json.dumps(master_payload, ensure_ascii=False)
print(f"Embedded payload size: {len(json_data_str) / 1024:.1f} KB")

# Load template from file or inject directly
TEMPLATE_PATH = os.path.join(ROOT_DIR, 'scripts/viz/workstation_template.html')

if os.path.exists(TEMPLATE_PATH):
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        html_src = f.read()
else:
    raise FileNotFoundError(f"Template not found at {TEMPLATE_PATH}")

compiled_html = html_src.replace('__PAYLOAD_PLACEHOLDER__', json_data_str)

# 1. Save to outputs/reports/index.html
index_p = os.path.join(OUTPUT_DIR, 'index.html')
with open(index_p, 'w', encoding='utf-8') as f:
    f.write(compiled_html)
print(f"Saved Real-Time Interactive Station to: {index_p}")

# 2. Save to root /index.html (for zero-config direct access via http://localhost:8088/)
root_index_p = os.path.join(ROOT_DIR, 'index.html')
with open(root_index_p, 'w', encoding='utf-8') as f:
    f.write(compiled_html)
print(f"Saved Root Index to: {root_index_p}")

# 3. Save to alias
alias_p = os.path.join(OUTPUT_DIR, 'universal_comparison_system.html')
with open(alias_p, 'w', encoding='utf-8') as f:
    f.write(compiled_html)
print(f"Saved Alias to: {alias_p}")
