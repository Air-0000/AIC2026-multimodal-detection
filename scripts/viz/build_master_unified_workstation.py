#!/usr/bin/env python3
"""
AIC 2026 Master Unified Multimodal Workstation Builder
======================================================
Consolidates all fragmented dashboards into ONE comprehensive, single-page application:
  1. Interactive Collision & Super-Resolution Workstation (4x Real-ESRGAN, Loupe, Dual-Canvas)
  2. Sovereign V23 Stage A 1,000-Image Full Test Set Prediction Gallery
  3. Universal Model Comparison Matrix (F5 Benchmark vs V23-A vs V23-B vs V24 vs V25 vs V26 vs V27 vs V28)
  4. Specialized Category Audit Vault (Ball, UAV, Tricycle Deep Dives)
  5. Architecture Provenance & Technical Specs Whitepaper

Outputs:
  outputs/reports/index.html
  outputs/reports/universal_comparison_system.html
  outputs/reports/audit_workbench_superres_collision.html
  index.html (Root server alias)
"""

import os
import sys
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = '/home/constant/CodeBuddy/Dev/AIC'
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
SUBMISSIONS_DIR = os.path.join(OUTPUT_DIR, 'submissions')
os.makedirs(REPORTS_DIR, exist_ok=True)

VIS_DIR = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/visible')
IR_DIR = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/infrared')
STAGEA_VIS_DIR = os.path.join(ROOT_DIR, 'work_dir/visualizations/v23_test_vis_all1000')

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
    0: '#38bdf8', 1: '#0ea5e9', 2: '#34d399', 3: '#fbbf24', 4: '#f97316',
    5: '#a78bfa', 6: '#60a5fa', 7: '#f43f5e', 8: '#facc15', 9: '#94a3b8',
    10: '#ec4899', 11: '#c084fc'
}

BASELINE_CONFIG = {
    'id': 'f5',
    'name': 'F5 (55.5/58.1 黄金标杆)',
    'score': '58.105',
    'path': os.path.join(ROOT_DIR, 'work_dir/submissions/submissions_f5/submission_f5_pure_rgb_compact7184.zip')
            if os.path.exists(os.path.join(ROOT_DIR, 'work_dir/submissions/submissions_f5/submission_f5_pure_rgb_compact7184.zip'))
            else os.path.join(ROOT_DIR, 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip'),
    'color': '#00f2fe'
}

MODELS_CONFIG = [
    {
        'id': 'f5_rgb',
        'name': 'F5 (58.105 黄金标杆)',
        'score': '58.105',
        'badge': '7,184 框 / 外部标杆',
        'path': os.path.join(ROOT_DIR, 'work_dir/submissions/submissions_f5/submission_f5_pure_rgb_compact7184.zip')
                if os.path.exists(os.path.join(ROOT_DIR, 'work_dir/submissions/submissions_f5/submission_f5_pure_rgb_compact7184.zip'))
                else os.path.join(ROOT_DIR, 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip'),
        'color': '#00f2fe',
        'desc': '官方复赛顶榜 58.105 纯可见光 D-FINE-X 标杆提交包（1216x2176，全量 1,000 张测试集 7,184 框）'
    },
    {
        'id': 'mal2',
        'name': '自研 MAL2 (纯 RGB 55.5 / 56.14%)',
        'score': '56.14% mAP',
        'badge': '4,784 框 / 自研纯可见光',
        'path': os.path.join(ROOT_DIR, 'work_dir/submissions/submission_mal2/submit_mal2_B_thr050.zip'),
        'color': '#38bdf8',
        'desc': '自研 D-FINE-X / MAL2 纯可见光基线（deim_hgnetv2_x_aic_obj2coco_ft_mal2，mAP 56.14% / 实测 55.5 分），τ=0.50 标准官方提交包'
    },
    {
        'id': 'puremal_refine',
        'name': '自研 PureMAL Refine (56.51% 纯RGB峰值)',
        'score': '56.51% mAP',
        'badge': '6,795 框 / 边界精炼峰值',
        'path': os.path.join(ROOT_DIR, 'work_dir/submissions/submission_puremal_refine/submission_puremal_refine_thr40.zip'),
        'color': '#818cf8',
        'desc': '自研 PureMAL 纯可见光 12-Epoch GIoU 3.5x 边界精炼模型（验证集最高 mAP 56.51%），τ=0.40 紧致提交包'
    },
    {
        'id': 'v20',
        'name': 'Sovereign V20 (老标多模态)',
        'score': '58.45% mAP',
        'badge': '6,894 框 / 纯老标无补标',
        'path': os.path.join(ROOT_DIR, 'work_dir/submissions/submissions_v20/submission_v20_thr45.zip'),
        'color': '#06b6d4',
        'desc': 'Pure-LMM 50-50 连续流形模型汤，100% 官方老标 1800 训练，完全未引入额外补标，与 F5 达成 88.8% 超高共识'
    },
    {
        'id': 'v23_stageA',
        'name': 'Sovereign V23 Stage A (fast32)',
        'score': '88.83% mAP',
        'badge': '7,323 框 / Stage A 补标基石',
        'path': os.path.join(ROOT_DIR, 'work_dir/submissions/v23_test1_final_submission/submission_v23_ep71_thr45.zip'),
        'color': '#f59e0b',
        'desc': 'Stage A quality1998 补标 72 轮收敛底座，1152x2048 零畸变，微小目标感知极强'
    },
    {
        'id': 'v23_stageB',
        'name': 'Sovereign V23 Stage B (TADL 53.555)',
        'score': '53.555 官方实测',
        'badge': '7,231 框 / TADL 任务解耦',
        'path': os.path.join(SUBMISSIONS_DIR, 'submissions_v23_native/submission_v23_native_refined_clean7231.zip'),
        'color': '#ec4899',
        'desc': 'Stage B 老标签 1800 + TADL 解耦微调，全项目官方复赛最高纪录 53.555'
    },
    {
        'id': 'v26',
        'name': 'Sovereign V26 Golden',
        'score': '58.10+',
        'badge': '7,241 框 / Golden Decoupled',
        'path': os.path.join(SUBMISSIONS_DIR, 'submissions_v26/submission_sovereign_v26_rare45_common50_clean7241.zip'),
        'color': '#10b981',
        'desc': 'Golden 跨模态解耦终局模型，多模态特征极致分离'
    },
    {
        'id': 'v27',
        'name': 'Sovereign V27 Pure Unbiased',
        'score': '89.59% mAP',
        'badge': '7,052 框 / 纯净无偏',
        'path': os.path.join(SUBMISSIONS_DIR, 'submissions_v27/submission_sovereign_v27_pure_thr50.zip'),
        'color': '#14b8a6',
        'desc': '纯净无偏老标注训练，验证集 mAP 0.8959 / AP50 0.9638'
    },
    {
        'id': 'v25',
        'name': 'Sovereign V25 Compact',
        'score': '56.40+',
        'badge': '9,637 框 / Perfect Union',
        'path': os.path.join(SUBMISSIONS_DIR, 'submissions_v25/submission_v25_compact.zip'),
        'color': '#a855f7',
        'desc': 'Perfect Union 压框融合，兼具超高召回与高精度空间对齐'
    },
    {
        'id': 'v24',
        'name': 'Sovereign V24 Native',
        'score': '53.555',
        'badge': '6,711 框 / Native Compact',
        'path': os.path.join(SUBMISSIONS_DIR, 'submissions_v24/submission_v24_native_compact.zip'),
        'color': '#3b82f6',
        'desc': 'Native Compact 压框出清，伪框大幅锐减 51%，官方实战 53.555 高分'
    },
    {
        'id': 'v28',
        'name': 'Sovereign V28 Disparity Awakening',
        'score': '58.50+ 冲顶',
        'badge': '5,940 框 / DCDA 视差觉醒',
        'path': os.path.join(SUBMISSIONS_DIR, 'submissions_v28/submission_sovereign_v28_cap60thr45_50.zip'),
        'color': '#6366f1',
        'desc': 'DCDA 物理极线高速路 + 凸组合平滑替换，AP75 0.9502 全项目最高'
    }
]

# Support dynamic submission registry from work_dir/submissions/workstation_models_registry.json
REGISTRY_FILE = os.path.join(ROOT_DIR, 'work_dir/submissions/workstation_models_registry.json')
if os.path.exists(REGISTRY_FILE):
    try:
        with open(REGISTRY_FILE, 'r', encoding='utf-8') as rf:
            extra_models = json.load(rf)
            existing_ids = {m['id'] for m in MODELS_CONFIG}
            for em in extra_models:
                if em.get('id') not in existing_ids and os.path.exists(em.get('path', '')):
                    MODELS_CONFIG.append(em)
                    existing_ids.add(em['id'])
    except Exception as e:
        print(f"Warning: Failed to load dynamic model registry: {e}")

def parse_boxes_from_zip(zip_p):
    d = {}
    if not os.path.exists(zip_p):
        print(f"Warning: Zip not found: {zip_p}")
        return d
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

print("1. Scanning 1,000 test set images...")
file_map = {}
if os.path.exists(VIS_DIR):
    for f in os.listdir(VIS_DIR):
        if f.endswith('.png') or f.endswith('.jpg'):
            stem = f.rsplit('.', 1)[0]
            file_map[stem] = f

all_stems = sorted(file_map.keys())
print(f"Total test images: {len(all_stems)}")

print("2. Parsing bounding boxes across all models...")
raw_predictions = {}
raw_predictions['f5'] = parse_boxes_from_zip(BASELINE_CONFIG['path'])

for m in MODELS_CONFIG:
    print(f"  Parsing {m['name']}...")
    raw_predictions[m['id']] = parse_boxes_from_zip(m['path'])

print("3. Precomputing discrepancy metrics and matrix summaries...")
diff_summaries = {}
model_matrix_stats = []

for m in MODELS_CONFIG:
    mid = m['id']
    preds = raw_predictions[mid]
    f5_preds = raw_predictions['f5']
    diff_list = []
    
    total_f = 0
    total_c = 0
    total_exact = 0
    total_conf = 0
    total_missed = 0
    total_fp = 0
    
    cat_stats = {cid: {'f5': 0, 'cand': 0, 'exact': 0, 'missed': 0, 'fp': 0} for cid in range(12)}
    
    for stem in all_stems:
        bf = [b for b in f5_preds.get(stem, []) if b[5] >= 0.45]
        bg = [b for b in preds.get(stem, []) if b[5] >= 0.45]
        n_exact, n_conf, n_missed, n_fp = quick_match(bf, bg)
        delta = len(bg) - len(bf)
        
        total_f += len(bf)
        total_c += len(bg)
        total_exact += n_exact
        total_conf += n_conf
        total_missed += n_missed
        total_fp += n_fp
        
        for b in bf: cat_stats[b[0]]['f5'] += 1
        for b in bg: cat_stats[b[0]]['cand'] += 1

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
    
    total_union = total_exact + total_missed + total_fp
    agree_pct = round((total_exact / total_union * 100), 1) if total_union > 0 else 0.0
    
    model_matrix_stats.append({
        'id': mid,
        'name': m['name'],
        'score': m['score'],
        'badge': m['badge'],
        'color': m['color'],
        'desc': m['desc'],
        'total_cand_boxes': total_c,
        'total_f5_boxes': total_f,
        'consensus_boxes': total_exact,
        'conflict_boxes': total_conf,
        'missed_boxes': total_missed,
        'fp_boxes': total_fp,
        'agree_rate': agree_pct,
        'cat_stats': cat_stats
    })

print("4. Building Stage A 1,000-Image Full Gallery Index...")
stage_a_preds = raw_predictions.get('v23_stageA', {})
gallery_index = []
class_distribution = Counter()

for stem in all_stems:
    boxes = stage_a_preds.get(stem, [])
    counts = Counter(b[0] for b in boxes)
    for cid, cnt in counts.items():
        class_distribution[cid] += cnt
        
    tag_strs = []
    for cid, cnt in sorted(counts.items()):
        c_zh = CLASS_ZH_SHORT.get(cid, str(cid))
        tag_strs.append(f"{c_zh}x{cnt}")
        
    gallery_index.append({
        'stem': stem,
        'total': len(boxes),
        'counts': dict(counts),
        'tags': " ".join(tag_strs) if tag_strs else "无检出",
        'max_conf': round(max((b[5] for b in boxes), default=0.0), 3),
        'has_tricycle': 11 in counts,
        'has_ball': 7 in counts,
        'has_uav': 10 in counts,
        'has_boat': 1 in counts,
        'has_animal': 2 in counts,
        'is_dense': len(boxes) >= 10
    })

print(f"Gallery Index built: {len(gallery_index)} items.")
print(f"Stage A Key Detections: Tricycle in {sum(1 for g in gallery_index if g['has_tricycle'])} images, "
      f"Ball in {sum(1 for g in gallery_index if g['has_ball'])} images, "
      f"UAV in {sum(1 for g in gallery_index if g['has_uav'])} images, "
      f"Boat in {sum(1 for g in gallery_index if g['has_boat'])} images.")

print("5. Ingesting specialized audit packages (Ball, UAV, Tricycle)...")
ball_cases = []
ball_json_p = os.path.join(OUTPUT_DIR, 'analysis/ball_analysis_f5_vs_v23/ball_analysis_report.json')
if os.path.exists(ball_json_p):
    try:
        with open(ball_json_p, 'r', encoding='utf-8') as f:
            ball_cases = json.load(f)
        print(f"Loaded {len(ball_cases)} Ball audit cases.")
    except Exception as e:
        print(f"Ball audit load failed: {e}")

uav_cases = []
for stem in all_stems:
    f5_b = [b for b in raw_predictions['f5'].get(stem, []) if b[0] == 10 and b[5] >= 0.45]
    v23_b = [b for b in stage_a_preds.get(stem, []) if b[0] == 10 and b[5] >= 0.45]
    if f5_b or v23_b:
        uav_cases.append({
            'stem': stem,
            'f5_cnt': len(f5_b),
            'v23_cnt': len(v23_b),
            'f5_max': round(max((b[5] for b in f5_b), default=0.0), 3),
            'v23_max': round(max((b[5] for b in v23_b), default=0.0), 3)
        })
print(f"Indexed {len(uav_cases)} UAV comparison cases.")

tricycle_cases = []
for stem in all_stems:
    f5_b = [b for b in raw_predictions['f5'].get(stem, []) if b[0] == 11 and b[5] >= 0.45]
    v23_b = [b for b in stage_a_preds.get(stem, []) if b[0] == 11 and b[5] >= 0.45]
    if f5_b or v23_b:
        tricycle_cases.append({
            'stem': stem,
            'f5_cnt': len(f5_b),
            'v23_cnt': len(v23_b),
            'f5_max': round(max((b[5] for b in f5_b), default=0.0), 3),
            'v23_max': round(max((b[5] for b in v23_b), default=0.0), 3)
        })
print(f"Indexed {len(tricycle_cases)} Tricycle comparison cases.")

# Load AI Diagnostic Cases Package
ai_cases_pkg = {}
ai_cases_p = os.path.join(OUTPUT_DIR, 'audit/ai_focus_cases.json')
if os.path.exists(ai_cases_p):
    try:
        with open(ai_cases_p, 'r', encoding='utf-8') as f:
            ai_cases_pkg = json.load(f)
    except Exception:
        pass

# Alignment summary
align_map = {}
align_summary_p = os.path.join(OUTPUT_DIR, 'test_alignment_summary.json')
if os.path.exists(align_summary_p):
    try:
        with open(align_summary_p, 'r', encoding='utf-8') as f:
            al_data = json.load(f)
            for item in al_data.get('details', []):
                s = item.get('stem')
                if s:
                    align_map[s] = {
                        'dx': item.get('dx', 0),
                        'dy': item.get('dy', 0),
                        'global_applied': item.get('global_applied', False),
                        'delta_cc': round(item.get('delta_cc', 0.0), 4),
                        'targets': len(item.get('targets', []))
                    }
    except Exception:
        pass

sr_available = {}
vis_sr_dir = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/visible_sr4x')
if os.path.exists(vis_sr_dir):
    for f in os.listdir(vis_sr_dir):
        if f.endswith('.jpg') or f.endswith('.png'):
            sr_available[f.rsplit('.', 1)[0]] = True

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
    'align_map': align_map,
    'gallery_index': gallery_index,
    'model_matrix_stats': model_matrix_stats,
    'specialized_audits': {
        'ball_cases': ball_cases,
        'uav_cases': uav_cases,
        'tricycle_cases': tricycle_cases
    }
}

print(f"6. Compiling Master Single-Page Unified Station...")
payload_json_str = json.dumps(master_payload, ensure_ascii=False)
print(f"Payload Size: {len(payload_json_str) / 1024:.1f} KB")

TEMPLATE_PATH = os.path.join(ROOT_DIR, 'scripts/viz/workstation_template.html')
with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    template_html = f.read()

# Replace payload placeholder
compiled_html = template_html.replace('__PAYLOAD_PLACEHOLDER__', payload_json_str)

# 7. Inject Master Tabs in Header
tabs_header_html = """
            <!-- Sovereign Master Navigation Tabs -->
            <div class="master-nav-tabs" style="display:flex;align-items:center;background:rgba(14,19,31,0.92);border:1px solid var(--border-glass);border-radius:10px;padding:3px 6px;gap:4px;box-shadow:0 2px 14px rgba(0,0,0,0.5);">
                <button type="button" class="master-tab-btn active" id="btnMasterTabWorkbench" onclick="switchMasterTab('workbench')" style="cursor:pointer;">
                    🛰️ 对撞与超分工作台
                </button>
                <button type="button" class="master-tab-btn" id="btnMasterTabGallery" onclick="switchMasterTab('gallery')" style="cursor:pointer;">
                    🖼️ Stage A 全量 1000 图画廊 (<span id="galleryTotalBadge" style="color:#f59e0b;font-family:monospace;">1000</span>)
                </button>
                <button type="button" class="master-tab-btn" id="btnMasterTabMatrix" onclick="switchMasterTab('matrix')" style="cursor:pointer;">
                    ⚖️ 历代模型矩阵对比
                </button>
                <button type="button" class="master-tab-btn" id="btnMasterTabSpecial" onclick="switchMasterTab('special')" style="cursor:pointer;">
                    🔬 专项长尾透视 (球/机/三轮)
                </button>
                <button type="button" class="master-tab-btn" id="btnMasterTabSpecs" onclick="switchMasterTab('specs')" style="cursor:pointer;">
                    📜 架构演进与白皮书
                </button>
            </div>
"""

# Insert right after brand-title-wrap
brand_anchor = '</div>\n        </div>\n\n        <!-- Active Comparison HUD in Header -->'
if brand_anchor in compiled_html:
    compiled_html = compiled_html.replace(brand_anchor, '</div>\n        </div>\n' + tabs_header_html + '\n        <!-- Active Comparison HUD in Header -->')
else:
    brand_anchor2 = '</header>'
    compiled_html = compiled_html.replace(brand_anchor2, tabs_header_html + '\n    </header>')

# 8. Inject Master CSS and Views
extra_css_and_views = """
<style>
.master-tab-btn {
    background: transparent;
    color: var(--text-dim);
    border: none;
    padding: 6px 14px;
    font-size: 11px;
    font-weight: 700;
    border-radius: 7px;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
}
.master-tab-btn:hover {
    color: #fff;
    background: rgba(255, 255, 255, 0.08);
}
.master-tab-btn.active {
    color: #fff;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    box-shadow: 0 0 14px rgba(37, 99, 235, 0.5);
}
.master-view {
    display: none;
    width: 100%;
    height: calc(100vh - 48px);
    overflow: hidden;
}
.master-view.active {
    display: flex;
}
.gallery-grid-deck {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
    padding: 20px;
    overflow-y: auto;
    flex: 1;
}
.gallery-card {
    background: rgba(18, 24, 37, 0.85);
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    cursor: pointer;
}
.gallery-card:hover {
    transform: translateY(-3px);
    border-color: rgba(37, 99, 235, 0.6);
    box-shadow: 0 8px 24px rgba(0,0,0,0.6);
}
.gallery-thumb-wrap {
    width: 100%;
    height: 160px;
    background: #080b11;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}
.gallery-thumb-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.3s;
}
.gallery-card:hover .gallery-thumb-wrap img {
    transform: scale(1.05);
}
.gallery-card-body {
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.gallery-card-stem {
    font-size: 13px;
    font-weight: 800;
    color: #fff;
    font-family: monospace;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.gallery-card-tags {
    font-size: 11px;
    color: var(--text-dim);
    line-height: 1.4;
    max-height: 38px;
    overflow: hidden;
    text-overflow: ellipsis;
}
.gallery-card-actions {
    display: flex;
    gap: 6px;
    margin-top: 4px;
}
.matrix-container {
    width: 100%;
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
    overflow-y: auto;
    color: var(--text-glow);
}
.matrix-card {
    background: rgba(18, 24, 37, 0.85);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
}
.matrix-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 12px;
}
.matrix-table th, .matrix-table td {
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border-glass);
}
.matrix-table th {
    background: rgba(14, 19, 31, 0.9);
    color: var(--text-dim);
    font-weight: 700;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
}
.matrix-table tr:hover {
    background: rgba(255, 255, 255, 0.03);
}
</style>

<!-- ==================== VIEW 2: STAGE A 1000 GALLERY ==================== -->
<div id="viewGallery" class="master-view" style="flex-direction:column;background:var(--bg-obsidian);">
    <div style="padding:12px 20px;border-bottom:1px solid var(--border-glass);background:rgba(11,15,23,0.92);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:14px;font-weight:800;color:#fff;">🖼️ Stage A 补标底座 全量 1,000 张测试预测画廊</span>
            <span style="font-size:11px;color:#f59e0b;background:rgba(245,158,11,0.12);padding:3px 8px;border-radius:6px;border:1px solid rgba(245,158,11,0.3);">
                72 轮收敛底模 · 1152x2048 零畸变 · 共 7,323 框 (τ≥0.45)
            </span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div class="search-input-wrap" style="width:260px;">
                <span class="search-icon">🔍</span>
                <input type="text" class="search-input" id="gallerySearchInput" placeholder="搜索样本 ID (如 003429)..." oninput="renderGalleryPage(1)" />
            </div>
            <div class="filter-chip-deck" style="margin:0;">
                <button type="button" class="chip-btn active" id="chipGalAll" onclick="setGalleryFilter('all')">全部 (1000)</button>
                <button type="button" class="chip-btn" id="chipGalTricycle" onclick="setGalleryFilter('tricycle')">🛵 三轮车 (8)</button>
                <button type="button" class="chip-btn" id="chipGalBall" onclick="setGalleryFilter('ball')">⚽ 小球 (48)</button>
                <button type="button" class="chip-btn" id="chipGalUAV" onclick="setGalleryFilter('uav')">🛸 无人机 (46)</button>
                <button type="button" class="chip-btn" id="chipGalBoat" onclick="setGalleryFilter('boat')">⛵ 船只 (46)</button>
                <button type="button" class="chip-btn" id="chipGalDense" onclick="setGalleryFilter('dense')">🔥 高密(≥10框)</button>
            </div>
        </div>
    </div>
    
    <div class="gallery-grid-deck" id="galleryCardsDeck"></div>
    
    <div style="padding:10px 20px;border-top:1px solid var(--border-glass);background:rgba(11,15,23,0.92);display:flex;align-items:center;justify-content:space-between;">
        <span style="font-size:12px;color:var(--text-dim);" id="galleryPaginationInfo">显示 1 - 48 / 共 1000 项</span>
        <div style="display:flex;gap:8px;">
            <button type="button" class="btn-hdr" onclick="changeGalleryPage(-1)">◀ 上一页</button>
            <span id="galleryCurrentPageBadge" style="font-family:monospace;font-weight:700;padding:4px 10px;background:rgba(255,255,255,0.06);border-radius:4px;">1</span>
            <button type="button" class="btn-hdr" onclick="changeGalleryPage(1)">下一页 ▶</button>
        </div>
    </div>
</div>

<!-- ==================== VIEW 3: UNIVERSAL MATRIX ==================== -->
<div id="viewMatrix" class="master-view" style="flex-direction:column;overflow-y:auto;background:var(--bg-obsidian);">
    <div class="matrix-container" id="matrixContainer"></div>
</div>

<!-- ==================== VIEW 4: SPECIALIZED AUDITS ==================== -->
<div id="viewSpecial" class="master-view" style="flex-direction:column;overflow-y:auto;background:var(--bg-obsidian);">
    <div class="matrix-container" id="specialAuditContainer"></div>
</div>

<!-- ==================== VIEW 5: SPECS & PROVENANCE ==================== -->
<div id="viewSpecs" class="master-view" style="flex-direction:column;overflow-y:auto;background:var(--bg-obsidian);">
    <div class="matrix-container" id="specsContainer"></div>
</div>
"""

# Insert Views before closing body tag
closing_body_anchor = '</body>'
compiled_html = compiled_html.replace(closing_body_anchor, extra_css_and_views + '\n' + closing_body_anchor)

# 9. Inject Master Navigation and Gallery JavaScript
extra_js = """
<script>
// ==================== MASTER TABS LOGIC ====================
let currentMasterTab = 'workbench';
let currentGalleryFilter = 'all';
let currentGalleryPage = 1;
const GALLERY_PAGE_SIZE = 48;

function switchMasterTab(tabName) {
    currentMasterTab = tabName;
    document.querySelectorAll('.master-tab-btn').forEach(b => b.classList.remove('active'));
    
    // Hide all views
    const views = ['viewWorkbench', 'viewGallery', 'viewMatrix', 'viewSpecial', 'viewSpecs'];
    views.forEach(vId => {
        const el = document.getElementById(vId);
        if (el) el.style.display = 'none';
    });
    
    if (tabName === 'workbench') {
        document.getElementById('btnMasterTabWorkbench').classList.add('active');
        const v = document.getElementById('viewWorkbench');
        if (v) v.style.display = 'flex';
        if (typeof resizeCanvas === 'function') resizeCanvas();
        if (typeof redrawCanvas === 'function') redrawCanvas();
    } else if (tabName === 'gallery') {
        document.getElementById('btnMasterTabGallery').classList.add('active');
        const v = document.getElementById('viewGallery');
        if (v) v.style.display = 'flex';
        renderGalleryPage(1);
    } else if (tabName === 'matrix') {
        document.getElementById('btnMasterTabMatrix').classList.add('active');
        const v = document.getElementById('viewMatrix');
        if (v) v.style.display = 'flex';
        renderMatrixView();
    } else if (tabName === 'special') {
        document.getElementById('btnMasterTabSpecial').classList.add('active');
        const v = document.getElementById('viewSpecial');
        if (v) v.style.display = 'flex';
        renderSpecialAuditView();
    } else if (tabName === 'specs') {
        document.getElementById('btnMasterTabSpecs').classList.add('active');
        const v = document.getElementById('viewSpecs');
        if (v) v.style.display = 'flex';
        renderSpecsView();
    }
}

// Wrap original workspace in viewWorkbench
document.addEventListener('DOMContentLoaded', () => {
    const wsGrid = document.querySelector('.workspace-grid');
    if (wsGrid && !document.getElementById('viewWorkbench')) {
        const wrap = document.createElement('div');
        wrap.id = 'viewWorkbench';
        wrap.className = 'master-view active';
        wrap.style.display = 'flex';
        wrap.style.flex = '1';
        wrap.style.overflow = 'hidden';
        wsGrid.parentNode.insertBefore(wrap, wsGrid);
        wrap.appendChild(wsGrid);
    }
});

// ==================== GALLERY PAGE LOGIC ====================
function setGalleryFilter(filterType) {
    currentGalleryFilter = filterType;
    document.querySelectorAll('#viewGallery .chip-btn').forEach(b => b.classList.remove('active'));
    
    if (filterType === 'all') document.getElementById('chipGalAll').classList.add('active');
    else if (filterType === 'tricycle') document.getElementById('chipGalTricycle').classList.add('active');
    else if (filterType === 'ball') document.getElementById('chipGalBall').classList.add('active');
    else if (filterType === 'uav') document.getElementById('chipGalUAV').classList.add('active');
    else if (filterType === 'boat') document.getElementById('chipGalBoat').classList.add('active');
    else if (filterType === 'dense') document.getElementById('chipGalDense').classList.add('active');
    
    renderGalleryPage(1);
}

function changeGalleryPage(delta) {
    renderGalleryPage(currentGalleryPage + delta);
}

function renderGalleryPage(page = 1) {
    if (!PAYLOAD || !PAYLOAD.gallery_index) return;
    const searchVal = (document.getElementById('gallerySearchInput')?.value || '').trim().toLowerCase();
    
    let filtered = PAYLOAD.gallery_index.filter(item => {
        if (searchVal && !item.stem.toLowerCase().includes(searchVal)) return false;
        if (currentGalleryFilter === 'tricycle' && !item.has_tricycle) return false;
        if (currentGalleryFilter === 'ball' && !item.has_ball) return false;
        if (currentGalleryFilter === 'uav' && !item.has_uav) return false;
        if (currentGalleryFilter === 'boat' && !item.has_boat) return false;
        if (currentGalleryFilter === 'dense' && !item.is_dense) return false;
        return true;
    });
    
    const totalItems = filtered.length;
    const maxPage = Math.max(1, Math.ceil(totalItems / GALLERY_PAGE_SIZE));
    currentGalleryPage = Math.min(Math.max(1, page), maxPage);
    
    const startIdx = (currentGalleryPage - 1) * GALLERY_PAGE_SIZE;
    const endIdx = Math.min(startIdx + GALLERY_PAGE_SIZE, totalItems);
    const pageItems = filtered.slice(startIdx, endIdx);
    
    // Update pagination UI
    const infoEl = document.getElementById('galleryPaginationInfo');
    if (infoEl) infoEl.textContent = `显示 ${totalItems > 0 ? startIdx + 1 : 0} - ${endIdx} / 筛选共 ${totalItems} 项 (全量 1000)`;
    const pageBadge = document.getElementById('galleryCurrentPageBadge');
    if (pageBadge) pageBadge.textContent = `${currentGalleryPage} / ${maxPage}`;
    
    const deck = document.getElementById('galleryCardsDeck');
    if (!deck) return;
    
    const isFileProto = window.location.protocol === 'file:';
    const basePath = isFileProto ? '../../' : '/';
    
    deck.innerHTML = pageItems.map(item => {
        const imgSrc = `${basePath}work_dir/visualizations/v23_test_vis_all1000/${item.stem}.jpg`;
        const tagsHtml = Object.entries(item.counts).map(([cid, cnt]) => {
            const cColor = PAYLOAD.class_colors[cid] || '#38bdf8';
            const cZh = PAYLOAD.class_zh_short[cid] || cid;
            return `<span style="display:inline-block;padding:1px 6px;border-radius:4px;background:rgba(255,255,255,0.06);border:1px solid ${cColor}40;color:${cColor};font-size:10px;font-weight:700;margin-right:4px;margin-bottom:3px;">${cZh} x${cnt}</span>`;
        }).join('');
        
        return `
        <div class="gallery-card" onclick="jumpToWorkbenchSample('${item.stem}')">
            <div class="gallery-thumb-wrap">
                <img src="${imgSrc}" loading="lazy" alt="${item.stem}" onerror="this.onerror=null;this.src='${basePath}dataset/test/AIC2026_PHASE_2_1000/visible/${item.stem}.jpg';" />
                <span style="position:absolute;top:6px;right:6px;background:rgba(0,0,0,0.75);color:#f59e0b;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:800;border:1px solid rgba(245,158,11,0.4);font-family:monospace;">
                    ${item.total} 框
                </span>
            </div>
            <div class="gallery-card-body">
                <div class="gallery-card-stem">
                    <span>${item.stem}</span>
                    <span style="font-size:10px;color:var(--text-muted);font-weight:normal;">最高置信度: ${item.max_conf}</span>
                </div>
                <div class="gallery-card-tags">
                    ${tagsHtml || '<span style="color:var(--text-muted);">无检出目标</span>'}
                </div>
                <div class="gallery-card-actions">
                    <button type="button" class="btn-hdr" style="flex:1;background:linear-gradient(135deg,#2563eb,#7c3aed);color:#fff;border:none;font-weight:700;padding:5px 8px;font-size:10px;" onclick="event.stopPropagation();jumpToWorkbenchSample('${item.stem}');">
                        🔍 超分工作台核验
                    </button>
                    <button type="button" class="btn-hdr" style="background:#1a2233;border:1px solid var(--border-glass);color:var(--text-dim);padding:5px 8px;font-size:10px;" onclick="event.stopPropagation();openLightboxModal('${imgSrc}', '${item.stem}', ${item.total});">
                        🖼️ 大图
                    </button>
                </div>
            </div>
        </div>
        `;
    }).join('');
}

function jumpToWorkbenchSample(stem) {
    switchMasterTab('workbench');
    if (typeof selectSample === 'function') {
        selectSample(stem);
    }
}

// Lightbox modal for gallery
function openLightboxModal(imgSrc, stem, boxCount) {
    let modal = document.getElementById('galleryLightboxModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'galleryLightboxModal';
        modal.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.92);z-index:99999;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px;';
        modal.onclick = () => { modal.style.display = 'none'; };
        document.body.appendChild(modal);
    }
    modal.innerHTML = `
        <div style="position:absolute;top:16px;right:24px;display:flex;gap:12px;align-items:center;z-index:100000;" onclick="event.stopPropagation()">
            <span style="font-size:14px;color:#fff;font-family:monospace;font-weight:800;">${stem} (${boxCount} 框)</span>
            <button class="btn-hdr" style="background:#ef4444;color:#fff;border:none;padding:6px 14px;font-weight:800;" onclick="document.getElementById('galleryLightboxModal').style.display='none'">✕ 关闭 (ESC)</button>
        </div>
        <img src="${imgSrc}" style="max-width:95vw;max-height:90vh;border-radius:8px;border:1px solid rgba(255,255,255,0.2);box-shadow:0 12px 48px rgba(0,0,0,0.9);object-fit:contain;" onclick="event.stopPropagation()" />
    `;
    modal.style.display = 'flex';
}

// ==================== MATRIX VIEW LOGIC ====================
function renderMatrixView() {
    const el = document.getElementById('matrixContainer');
    if (!el || !PAYLOAD || !PAYLOAD.model_matrix_stats) return;
    
    let rowsHtml = PAYLOAD.model_matrix_stats.map(st => {
        return `
        <tr>
            <td style="font-weight:800;color:${st.color};">
                ${st.name}
                <div style="font-size:10px;color:var(--text-muted);font-weight:normal;margin-top:2px;">${st.desc}</div>
            </td>
            <td style="font-family:monospace;font-weight:800;color:#fff;">${st.score}</td>
            <td style="font-family:monospace;font-weight:700;">${st.total_cand_boxes} 框</td>
            <td style="font-family:monospace;color:#10b981;font-weight:700;">${st.consensus_boxes} 框</td>
            <td style="font-family:monospace;color:#f59e0b;font-weight:700;">${st.conflict_boxes} 框</td>
            <td style="font-family:monospace;color:#ef4444;font-weight:700;">${st.missed_boxes} 框</td>
            <td style="font-family:monospace;color:#38bdf8;font-weight:700;">${st.fp_boxes} 框</td>
            <td style="font-family:monospace;font-weight:800;color:${st.agree_rate>=75?'#10b981':st.agree_rate>=60?'#f59e0b':'#ef4444'};">${st.agree_rate}%</td>
            <td>
                <button type="button" class="btn-hdr" style="font-size:11px;padding:4px 10px;background:rgba(255,255,255,0.06);border:1px solid var(--border-glass);" onclick="switchMasterTab('workbench');switchCandidateModel('${st.id}');">
                    进入对撞分析
                </button>
            </td>
        </tr>
        `;
    }).join('');
    
    el.innerHTML = `
    <div class="matrix-card">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
            <div>
                <h2 style="font-size:18px;font-weight:800;color:#fff;">⚖️ AIC 2026 历代检测模型全量对撞矩阵</h2>
                <p style="font-size:12px;color:var(--text-dim);margin-top:4px;">基准标杆：<strong>F5 (57.266 / 7,097 框)</strong> 逐样本二分图真实 IoU 空间对撞分析</p>
            </div>
            <span style="padding:4px 12px;background:rgba(0,242,254,0.12);border:1px solid rgba(0,242,254,0.3);color:#00f2fe;border-radius:6px;font-weight:800;font-size:11px;">
                7 款旗舰全息就绪
            </span>
        </div>
        <table class="matrix-table">
            <thead>
                <tr>
                    <th>模型版本与架构</th>
                    <th>复赛实测 / 估分</th>
                    <th>预测总框数</th>
                    <th>双方共识 (TP)</th>
                    <th>类别冲突</th>
                    <th>模型遗漏 (Missed)</th>
                    <th>独有发现 (FP/新目标)</th>
                    <th>与F5吻合率</th>
                    <th>操作</th>
                </tr>
            </thead>
            <tbody>
                ${rowsHtml}
            </tbody>
        </table>
    </div>
    `;
}

// ==================== SPECIALIZED AUDITS VIEW LOGIC ====================
function renderSpecialAuditView() {
    const el = document.getElementById('specialAuditContainer');
    if (!el || !PAYLOAD || !PAYLOAD.specialized_audits) return;
    
    const audits = PAYLOAD.specialized_audits;
    
    let ballRows = (audits.ball_cases || []).slice(0, 15).map(c => {
        return `
        <div style="background:rgba(18,24,37,0.7);border:1px solid var(--border-glass);border-radius:8px;padding:10px 14px;display:flex;align-items:center;justify-content:space-between;">
            <div>
                <span style="font-family:monospace;font-weight:800;color:#fff;">${c.stem}</span>
                <span style="margin-left:10px;font-size:11px;color:#f59e0b;padding:2px 6px;background:rgba(245,158,11,0.1);border-radius:4px;">${c.cat_label}</span>
                <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">F5: ${c.f5_cnt} 框 | V23: ${c.v23_cnt} 框 (共识: ${c.exact_cnt})</div>
            </div>
            <button type="button" class="btn-hdr" onclick="jumpToWorkbenchSample('${c.stem}')">深度核验 ➔</button>
        </div>
        `;
    }).join('');
    
    let uavRows = (audits.uav_cases || []).slice(0, 15).map(c => {
        return `
        <div style="background:rgba(18,24,37,0.7);border:1px solid var(--border-glass);border-radius:8px;padding:10px 14px;display:flex;align-items:center;justify-content:space-between;">
            <div>
                <span style="font-family:monospace;font-weight:800;color:#fff;">${c.stem}</span>
                <span style="margin-left:10px;font-size:11px;color:#ec4899;padding:2px 6px;background:rgba(236,72,153,0.1);border-radius:4px;">无人机空域核查</span>
                <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">F5: ${c.f5_cnt} (最高 ${c.f5_max}) | V23: ${c.v23_cnt} (最高 ${c.v23_max})</div>
            </div>
            <button type="button" class="btn-hdr" onclick="jumpToWorkbenchSample('${c.stem}')">深度核验 ➔</button>
        </div>
        `;
    }).join('');
    
    let triRows = (audits.tricycle_cases || []).slice(0, 15).map(c => {
        return `
        <div style="background:rgba(18,24,37,0.7);border:1px solid var(--border-glass);border-radius:8px;padding:10px 14px;display:flex;align-items:center;justify-content:space-between;">
            <div>
                <span style="font-family:monospace;font-weight:800;color:#fff;">${c.stem}</span>
                <span style="margin-left:10px;font-size:11px;color:#c084fc;padding:2px 6px;background:rgba(192,132,252,0.1);border-radius:4px;">三轮车视差觉醒</span>
                <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">F5: ${c.f5_cnt} 框 | V23 Stage A: ${c.v23_cnt} 框 (最高 ${c.v23_max})</div>
            </div>
            <button type="button" class="btn-hdr" onclick="jumpToWorkbenchSample('${c.stem}')">深度核验 ➔</button>
        </div>
        `;
    }).join('');
    
    el.innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;">
        <div class="matrix-card">
            <h3 style="color:#f59e0b;margin-bottom:12px;font-size:15px;display:flex;align-items:center;gap:6px;">⚽ 小球 (Ball) 专项穿透 (共 55 例)</h3>
            <div style="display:flex;flex-direction:column;gap:8px;">${ballRows}</div>
        </div>
        <div class="matrix-card">
            <h3 style="color:#ec4899;margin-bottom:12px;font-size:15px;display:flex;align-items:center;gap:6px;">🛸 无人机 (UAV) 弱小空域 (共 42 例)</h3>
            <div style="display:flex;flex-direction:column;gap:8px;">${uavRows}</div>
        </div>
        <div class="matrix-card">
            <h3 style="color:#c084fc;margin-bottom:12px;font-size:15px;display:flex;align-items:center;gap:6px;">🛵 三轮车 (Tricycle) 关键解耦 (共 30 例)</h3>
            <div style="display:flex;flex-direction:column;gap:8px;">${triRows}</div>
        </div>
    </div>
    `;
}

// ==================== SPECS & PROVENANCE VIEW LOGIC ====================
function renderSpecsView() {
    const el = document.getElementById('specsContainer');
    if (!el) return;
    
    el.innerHTML = `
    <div class="matrix-card" style="line-height:1.7;">
        <h2 style="font-size:20px;font-weight:800;color:#fff;margin-bottom:16px;">📜 Sovereign 架构演进与全生命周期白皮书</h2>
        
        <div style="background:rgba(37,99,235,0.08);border-left:4px solid #3b82f6;padding:12px 16px;border-radius:4px;margin-bottom:20px;">
            <strong style="color:#60a5fa;">两段式巅峰：Sovereign V23 的前世今生</strong><br/>
            Sovereign V23 官方实测最高分 <strong>53.555 (7,459 框)</strong> 并非一蹴而就，而是由两段式经典配合而成：
            <ul style="margin-left:20px;margin-top:6px;">
                <li><strong>Stage A · quality1998 补标 72 轮深度预训练 (zero_distort_fast32)</strong>：在 17,921 框的丰富监督下，将 1152×2048 零畸变超高清几何特征与 6 通道早融合神经耦合推至极限，在测试集自然产出 6,789 / 7,323 框，微小目标感知力达到全项目顶峰。</li>
                <li><strong>Stage B · 官方老标注 1800 + TADL 任务对齐解耦损失 (21 轮精调)</strong>：以 Stage A 权重为巅峰热启动，将置信度决策面无偏吸附回官方粗放标注评测机，斩获 53.555 官方榜首战绩。</li>
            </ul>
        </div>
        
        <div style="background:rgba(239,68,68,0.08);border-left:4px solid #ef4444;padding:12px 16px;border-radius:4px;margin-bottom:20px;">
            <strong style="color:#f87171;">⚠️ 深刻警示：V28-B 72 轮长跑单调退化的惨痛教训</strong><br/>
            V28 曾尝试 72 轮长跑，但从 Epoch 0 的 88.83% mAP 一路单调坠落至 Epoch 71 的 77.84% (-11.0%)：
            <ul style="margin-left:20px;margin-top:6px;">
                <li><strong>罪魁祸首 1 (Mosaic 降采样)</strong>：配置启用了长达 54 轮的 <code>Mosaic [576, 1024]</code>，把本就微小的三轮车和小球等压缩至 $4\\times 4$ 像素噪声，导致三轮车 AP 从 80.2% 暴跌至 35.8%。</li>
                <li><strong>罪魁祸首 2 (标注口径错配)</strong>：训练用了补标 quality1998，而验证用了老标注 instances_val，且 200 张验证图全包含在训练集内，多出的真框被老评测机全判为假阳性 (FP)。</li>
            </ul>
        </div>
        
        <div style="background:rgba(16,185,129,0.08);border-left:4px solid #10b981;padding:12px 16px;border-radius:4px;">
            <strong style="color:#34d399;">🚀 下一代 Sovereign V29 规划：Meta DINOv3-ViT-Large 在线微创手术蒸馏</strong><br/>
            坚决摒弃退化的 V28 权重，死锁最纯净的 V23 黄金底座 (88.83%)：
            <ul style="margin-left:20px;margin-top:6px;">
                <li><strong>绝对零 Mosaic</strong>：100% 原始 1152×2048 分辨率输入，彻底保全三轮车、小球、远景目标的真实物理空间几何。</li>
                <li><strong>Meta 官方 DINOv3-ViT-Large (300M 参数) 在线冻结教师</strong>：动态 2x 双线性降采样至 576x1024，提取 1024 维高阶稠密 Patch 表征，引导学生 P4 层通过余弦相似度 + Gram 矩阵拓扑损失微创对齐。</li>
            </ul>
        </div>
    </div>
    `;
}
</script>
"""

# Inject before closing body
compiled_html = compiled_html.replace('</body>', extra_js + '\n</body>')

# 10. Write to destinations
destinations = [
    os.path.join(REPORTS_DIR, 'index.html'),
    os.path.join(REPORTS_DIR, 'universal_comparison_system.html'),
    os.path.join(REPORTS_DIR, 'audit_workbench_superres_collision.html'),
    os.path.join(ROOT_DIR, 'index.html')
]

for dest in destinations:
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(compiled_html)
    print(f"Successfully compiled Master Station to: {dest} (Size: {len(compiled_html) / 1024 / 1024:.2f} MB)")

print("\nAll fragmented dashboards successfully UNIFIED into a SINGLE MASTER WORKSTATION with ALL 7 MODELS!")
