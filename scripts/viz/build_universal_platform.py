#!/usr/bin/env python3
"""
Build the Universal Multimodal Detection Comparison & Audit Station
===================================================================
Compiles all benchmark models through MultimodalComparator into a single,
fully dynamic Single Page Application (SPA). Zero iframes.
Outputs:
  outputs/reports/index.html
  outputs/reports/universal_comparison_system.html
  outputs/metrics/universal_benchmarks_data.json
"""

import os
import sys
import json
from collections import OrderedDict
from multimodal_comparator import MultimodalComparator

OUTPUT_DIR = '/home/constant/CodeBuddy/Dev/AIC/outputs'
DISCR_DIR = os.path.join(OUTPUT_DIR, 'visualizations_discrepancy')

MODELS_CONFIG = [
    {
        'id': 'v23',
        'name': 'Sovereign V23',
        'badge': '53.00分 (8023框)',
        'score': '53.00',
        'color': '#f43f5e',
        'desc': 'Flip TTA thr45 未经压框，召回率高但背景多报偏多',
        'baseline_path': 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip',
        'candidate_path': '/home/constant/Downloads/submission_v23_flip_tta_thr45.zip',
        'img_folder': 'discrepancies_f5_vs_v23/all_discrepancies'
    },
    {
        'id': 'v24',
        'name': 'Sovereign V24 Native',
        'badge': '53.555分 (6711框)',
        'score': '53.555',
        'color': '#3b82f6',
        'desc': 'Native Compact 压框出清，伪框大幅锐减 51%，官方实战 53.555 高分',
        'baseline_path': 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip',
        'candidate_path': 'outputs/submissions/submissions_v24/submission_v24_native_compact.zip',
        'img_folder': 'discrepancies_f5_vs_v24/all_discrepancies'
    },
    {
        'id': 'v25',
        'name': 'Sovereign V25 Compact',
        'badge': '56.40+ 稳态',
        'score': '56.40',
        'color': '#a855f7',
        'desc': 'Perfect Union 压框融合，兼具超高召回与高精度空间对齐',
        'baseline_path': 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip',
        'candidate_path': 'outputs/submissions/submissions_v25/submission_v25_compact.zip',
        'img_folder': 'discrepancies_f5_vs_v25/all_discrepancies'
    },
    {
        'id': 'v26',
        'name': 'Sovereign V26 Golden',
        'badge': '58.10+ 冲榜',
        'score': '58.10',
        'color': '#10b981',
        'desc': 'Golden 跨模态解耦终局模型，多模态特征极致分离',
        'baseline_path': 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip',
        'candidate_path': 'outputs/submissions/submissions_v26/submission_sovereign_v26_rare45_common50_clean7241.zip',
        'img_folder': 'discrepancies_f5_vs_v26/all_discrepancies'
    },
    {
        'id': 'v28',
        'name': 'Sovereign V28 Disparity Awakening',
        'badge': 'V28 物理视差觉醒 (DCDA 对齐)',
        'score': '58.50+ 冲顶',
        'color': '#6366f1',
        'desc': 'DCDA 物理极线高速路 + 凸组合平滑替换，边缘相似度 0.946，彻底根除双峰重叠与错位冲突',
        'baseline_path': 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip',
        'candidate_path': 'outputs/submissions/submissions_v28/submission_sovereign_v28_cap60thr45_50.zip',
        'img_folder': 'discrepancies_f5_vs_v28/all_discrepancies'
    }
]

print("Ingesting and auditing benchmark models...")
hub_data = {
    'models': {},
    'meta': {
        'baseline_name': 'F5 (57.266 黄金标杆)',
        'baseline_score': '57.266',
        'baseline_boxes': 7212,
        'model_list': []
    }
}

for m in MODELS_CONFIG:
    print(f"Auditing {m['name']}...")
    comp = MultimodalComparator(
        baseline_path=m['baseline_path'],
        candidate_path=m['candidate_path'],
        name_baseline='F5 (57.266)',
        name_candidate=m['name']
    )
    records, meta = comp.run_audit()
    hub_data['models'][m['id']] = {
        'id': m['id'],
        'name': m['name'],
        'badge': m['badge'],
        'score': m['score'],
        'color': m['color'],
        'desc': m['desc'],
        'img_folder': m['img_folder'],
        'meta': meta,
        'items': records
    }
    hub_data['meta']['model_list'].append({
        'id': m['id'],
        'name': m['name'],
        'badge': m['badge'],
        'score': m['score'],
        'color': m['color'],
        'diff_cnt': meta['diff_images'],
        'missed_cnt': meta['missed_boxes'],
        'fp_cnt': meta['fp_boxes'],
        'consensus_cnt': meta['consensus_boxes']
    })

# Scan CatA-CatE classic showcase
cat_descriptions = {
    'CatA_01': ('CatA (01)', '000351', 'F5 漏检 / Sovereign 独立高置信检出真实动物 (0.96)'),
    'CatA_02': ('CatA (02)', '000367', 'F5 漏检 / Sovereign 独立高置信检出真实动物 (0.94)'),
    'CatA_03': ('CatA (03)', '00000432', 'F5 漏检 / Sovereign 独立高置信检出微小无人机 (0.91)'),
    'CatA_04': ('CatA (04)', '002586', 'F5 漏检 / Sovereign 独立高置信检出标牌 (0.91)'),
    'CatA_05': ('CatA (05)', '00000297', 'F5 漏检 / Sovereign 独立高置信检出长椅座椅 (0.82)'),
    'CatA_06': ('CatA (06)', '00000299', 'F5 漏检 / Sovereign 独立高置信检出长椅座椅 (0.84)'),
    'CatA_07': ('CatA (07)', '00000302', 'F5 漏检 / Sovereign 独立高置信检出 4 个真目标'),
    'CatA_08': ('CatA (08)', '002507', 'F5 漏检 / Sovereign 独立高置信检出 3 个真目标'),
    'CatA_09': ('CatA (09)', '003331', 'F5 漏检 / Sovereign 独立高置信检出车辆 (0.84)'),
    'CatA_10': ('CatA (10)', '000008_009', 'F5 漏检 / Sovereign 独立高置信检出动物 (0.91)'),
    'CatB_01': ('CatB (01)', '000003', 'Sovereign 漏检 / F5 独立高置信检出路灯 (0.93)'),
    'CatB_02': ('CatB (02)', '00000422', 'Sovereign 漏检 / F5 独立检出无人机 (0.51)'),
    'CatB_03': ('CatB (03)', '003360', 'Sovereign 漏检 / F5 独立高置信检出标牌 (0.75)'),
    'CatB_04': ('CatB (04)', '003727', 'Sovereign 漏检 / F5 独立高置信检出动物 (0.81)'),
    'CatC_01': ('CatC (01)', '002808', '杂波治理：F5 地表长条矮墙密集误报座椅 vs Sovereign 纯净人群'),
    'CatC_02': ('CatC (02)', 'shuming96', '杂波治理：F5 夜景远景路灯噪声 vs Sovereign 纯净检出'),
    'CatC_03': ('CatC (03)', '000008_021', '杂波治理：F5 边缘破碎杂波 vs Sovereign 显著实体'),
    'CatC_04': ('CatC (04)', '003418', '杂波治理：F5 复杂街景交通杂波 vs Sovereign 清爽结构'),
    'CatC_05': ('CatC (05)', '000153', '杂波治理：F5 窗格外立面碎片标牌 vs Sovereign 实体'),
    'CatD_01': ('CatD (01)', '001539', '解耦能力：密集动物群分离与重叠分割'),
    'CatD_02': ('CatD (02)', '00000290', '解耦能力：复杂高召回场景无人机与座椅同时精细锁定'),
    'CatD_03': ('CatD (03)', '001016', '解耦能力：标牌与单车密集重叠群体的边界清晰划分'),
    'CatE_01': ('CatE (01)', '000020', '类别冲突：人 (person) ⚔️ 座椅 (seat) 高重合争议 (IoU=0.99)'),
    'CatE_02': ('CatE (02)', '000008', '类别冲突：动物 (animal) ⚔️ 人 (person) 争议 (IoU=0.85)'),
    'CatE_03': ('CatE (03)', '000004', '类别冲突：汽车 (car) ⚔️ 标牌 (sign) 争议 (IoU=0.93)'),
    'CatE_04': ('CatE (04)', '000003', '类别冲突：骑行者 (person) ⚔️ 自行车 (bicycle) 边界解耦'),
    'CatE_05': ('CatE (05)', '00000018', '类别冲突：标牌 (sign) ⚔️ 人 (person) 空间混淆')
}

gallery_items = []
if os.path.exists(DISCR_DIR):
    for f in sorted(os.listdir(DISCR_DIR)):
        if not f.endswith('.png'): continue
        prefix = f[:7]
        tag_info = cat_descriptions.get(prefix, (f[:4], 'unknown', f))
        gallery_items.append({
            'filename': f,
            'path': f'visualizations_discrepancy/{f}',
            'group': tag_info[0].split()[0],
            'badge': tag_info[0],
            'stem': tag_info[1],
            'desc': tag_info[2]
        })
hub_data['gallery'] = gallery_items

# Save unified data JSON
data_json_path = os.path.join(OUTPUT_DIR, 'universal_benchmarks_data.json')
with open(data_json_path, 'w', encoding='utf-8') as f:
    json.dump(hub_data, f, ensure_ascii=False)
print(f"Saved complete benchmark hub data to: {data_json_path}")

# Build Single-Page Application HTML
def build_spa_html(data_dict):
    json_str = json.dumps(data_dict, ensure_ascii=False)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIC 2026 多模态检测统一对比与审计系统</title>
    <style>
        :root {{
            --bg-dark: #0b0d11;
            --bg-header: #08090d;
            --bg-card: #151820;
            --bg-card-hover: #1d222d;
            --border: #232834;
            --border-light: #32394a;
            --text-main: #f1f5f9;
            --text-dim: #94a3b8;
            --text-muted: #64748b;
            --primary: #3b82f6;
            --f5-color: #06b6d4;
            --v23-color: #f43f5e;
            --consensus: #10b981;
            --amber: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }}
        body {{ background: var(--bg-dark); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; overflow-x: hidden; }}

        /* Top Executive Header */
        header {{
            background: var(--bg-header);
            border-bottom: 1px solid var(--border);
            padding: 10px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 1000;
            backdrop-filter: blur(12px);
        }}
        .brand-area {{ display: flex; align-items: center; gap: 12px; }}
        .brand-badge {{
            background: linear-gradient(135deg, #2563eb, #8b5cf6);
            color: #fff;
            font-size: 11px;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }}
        .brand-title {{ font-size: 16px; font-weight: 700; color: #fff; }}

        /* Model Switcher Pill Group */
        .model-switcher {{
            display: flex;
            background: #11141b;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 3px;
            gap: 4px;
        }}
        .model-pill {{
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-dim);
            padding: 5px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .model-pill:hover {{ color: #fff; background: rgba(255,255,255,0.03); }}
        .model-pill.active {{
            background: #2563eb;
            color: #fff;
            border-color: #3b82f6;
            box-shadow: 0 0 12px rgba(37,99,235,0.4);
        }}

        /* Functional Section Navigation */
        .section-tabs-bar {{
            background: #101319;
            border-bottom: 1px solid var(--border);
            padding: 0 24px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .sec-tab {{
            padding: 10px 16px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-dim);
            border-bottom: 2px solid transparent;
            cursor: pointer;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .sec-tab:hover {{ color: #fff; }}
        .sec-tab.active {{ color: #fff; border-bottom-color: var(--primary); background: rgba(59,130,246,0.06); }}

        /* KPI Dashboard Bar */
        .kpi-bar {{
            background: #12151b;
            border-bottom: 1px solid var(--border);
            padding: 10px 24px;
            display: flex;
            align-items: center;
            gap: 14px;
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
        .kpi-val {{ font-size: 16px; font-weight: 800; }}
        .kpi-lbl {{ font-size: 12px; color: var(--text-dim); }}

        /* Filter Controls */
        .controls-section {{
            background: #14171f;
            border-bottom: 1px solid var(--border);
            padding: 10px 24px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .tab-group {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
        .filter-btn {{
            background: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .filter-btn:hover {{ background: var(--bg-card-hover); border-color: var(--border-light); }}
        .filter-btn.active {{ background: #2563eb; border-color: #3b82f6; color: #fff; box-shadow: 0 0 10px rgba(37,99,235,0.4); }}
        .filter-btn.btn-missed.active {{ background: #0891b2; border-color: var(--f5-color); box-shadow: 0 0 10px rgba(6,182,212,0.4); }}
        .filter-btn.btn-fp.active {{ background: #e11d48; border-color: var(--v23-color); box-shadow: 0 0 10px rgba(244,63,94,0.4); }}

        .sub-filters {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 12px; }}
        .chip {{
            background: #1e222b;
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
            background: #1a1e27;
            border: 1px solid var(--border);
            color: #fff;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            width: 140px;
        }}

        /* Main Workspace Stage */
        main {{ flex: 1; padding: 16px 24px; display: flex; flex-direction: column; align-items: center; gap: 14px; }}
        
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
        .info-row1 {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
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

        .box-breakdown {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
        @media (max-width: 1024px) {{ .box-breakdown {{ grid-template-columns: 1fr; }} }}
        .box-list-card {{
            background: #121419;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .box-tags-wrap {{ display: flex; flex-wrap: wrap; gap: 6px; max-height: 100px; overflow-y: auto; }}
        .tag-missed {{
            background: rgba(6,182,212,0.12);
            border: 1px solid rgba(6,182,212,0.4);
            color: var(--f5-color);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .tag-fp {{
            background: rgba(244,63,94,0.12);
            border: 1px solid rgba(244,63,94,0.4);
            color: var(--v23-color);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}

        /* Image Display Stage */
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
            cursor: zoom-in;
        }}
        .stage-container img {{ width: 100%; height: auto; display: block; object-fit: contain; }}

        /* Thumb Carousel */
        .thumb-carousel {{
            width: 100%;
            max-width: 1920px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            align-items: center;
            gap: 8px;
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

        /* Gallery Showcase View */
        .gallery-view {{
            width: 100%;
            max-width: 1920px;
            display: none;
            flex-direction: column;
            gap: 16px;
        }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 16px;
        }}
        .gal-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .gal-card:hover {{ transform: translateY(-2px); border-color: var(--primary); }}
        .gal-card img {{ width: 100%; height: 200px; object-fit: cover; display: block; }}
        .gal-body {{ padding: 12px 14px; }}

        /* Matrix View */
        .matrix-view {{
            width: 100%;
            max-width: 1920px;
            display: none;
            flex-direction: column;
            gap: 16px;
        }}
        .matrix-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            font-size: 13px;
        }}
        .matrix-table th, .matrix-table td {{ padding: 12px 16px; border-bottom: 1px solid var(--border); text-align: left; }}
        .matrix-table th {{ background: #11141b; color: var(--text-dim); font-weight: 600; }}

        /* Lightbox */
        .lightbox {{
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
        .lightbox img {{ max-width: 98vw; max-height: 98vh; object-fit: contain; }}
        kbd {{ background: #232732; border: 1px solid #3c4456; border-radius: 3px; padding: 1px 5px; font-size: 11px; }}
    </style>
</head>
<body>

    <!-- Executive Header -->
    <header>
        <div class="brand-area">
            <div class="brand-badge">AIC 2026 AUDIT HUB</div>
            <div class="brand-title">AIC 2026 多模态检测统一对比与审计系统</div>
        </div>

        <!-- Universal Model Switcher -->
        <div class="model-switcher" id="modelSwitcher">
            <!-- Dynamically populated -->
        </div>

        <div class="nav-controls">
            <button class="action-btn" onclick="prevImg()">◀ 上一张 (←)</button>
            <span id="curIndexLabel" style="font-size:13px;font-weight:700;min-width:70px;text-align:center;">1 / 0</span>
            <button class="action-btn" onclick="nextImg()">下一张 (→) ▶</button>
        </div>
    </header>

    <!-- Top Section Tabs -->
    <div class="section-tabs-bar">
        <div class="sec-tab active" onclick="switchSection('inspector')">
            <span>🔍 逐样本深度对撞大盘</span>
        </div>
        <div class="sec-tab" onclick="switchSection('gallery')">
            <span>🖼️ 经典典型案卷画廊 (CatA ~ CatE)</span>
        </div>
        <div class="sec-tab" onclick="switchSection('matrix')">
            <span>📊 全局横向演进矩阵 (Cross-Model Matrix)</span>
        </div>
    </div>

    <!-- Inspector View Port -->
    <div id="inspectorSection" style="display:flex;flex-direction:column;width:100%;">
        <!-- KPI Metrics Bar -->
        <div class="kpi-bar">
            <div class="kpi-card" style="border-left: 3px solid #3b82f6;">
                <div>
                    <div class="kpi-val" style="color:#60a5fa;" id="kpiTotalDiff">0 张</div>
                    <div class="kpi-lbl">存在差异图像 (共1000张)</div>
                </div>
            </div>
            <div class="kpi-card" style="border-left: 3px solid var(--f5-color);">
                <div>
                    <div class="kpi-val" style="color:var(--f5-color);" id="kpiMissed">0 框</div>
                    <div class="kpi-lbl">📉 候选模型 遗漏 标杆真框</div>
                </div>
            </div>
            <div class="kpi-card" style="border-left: 3px solid var(--v23-color);">
                <div>
                    <div class="kpi-val" style="color:var(--v23-color);" id="kpiFP">0 框</div>
                    <div class="kpi-lbl">📈 候选模型 独有多报伪框</div>
                </div>
            </div>
            <div class="kpi-card" style="border-left: 3px solid var(--consensus);">
                <div>
                    <div class="kpi-val" style="color:var(--consensus);" id="kpiConsensus">0 框</div>
                    <div class="kpi-lbl">🟩 双方强共识检测框</div>
                </div>
            </div>
            <div class="kpi-card" style="border-left: 3px solid var(--amber);">
                <div>
                    <div class="kpi-val" style="color:var(--amber);" id="kpiConflict">0 处</div>
                    <div class="kpi-lbl">⚠️ 类别冲突</div>
                </div>
            </div>
        </div>

        <!-- Filter Controls -->
        <div class="controls-section">
            <div class="tab-group">
                <button class="filter-btn btn-missed active" onclick="setMainMode('missed')">
                    <span>📉 遗漏真框</span>
                    <span style="opacity:0.8;">(<span id="cntMissed">0</span>)</span>
                </button>
                <button class="filter-btn btn-fp" onclick="setMainMode('fp')">
                    <span>📈 多报伪框</span>
                    <span style="opacity:0.8;">(<span id="cntFP">0</span>)</span>
                </button>
                <button class="filter-btn" onclick="setMainMode('pure_miss')">
                    <span>🎯 纯遗漏</span>
                    <span style="opacity:0.8;">(<span id="cntPureMiss">0</span>)</span>
                </button>
                <button class="filter-btn" onclick="setMainMode('pure_fp')">
                    <span>🚨 纯多报</span>
                    <span style="opacity:0.8;">(<span id="cntPureFP">0</span>)</span>
                </button>
                <button class="filter-btn" onclick="setMainMode('mixed')">
                    <span>⚖️ 交织</span>
                    <span style="opacity:0.8;">(<span id="cntMixed">0</span>)</span>
                </button>
                <button class="filter-btn" onclick="setMainMode('conflict')" style="color:#fbbf24;">
                    <span>⚠️ 冲突</span>
                    <span style="opacity:0.8;">(<span id="cntConflict">0</span>)</span>
                </button>
                <button class="filter-btn" onclick="setMainMode('all')">
                    <span>🌐 全量差异</span>
                    <span style="opacity:0.8;">(<span id="cntAll">0</span>)</span>
                </button>
            </div>

            <div class="sub-filters">
                <span style="color:var(--text-muted);font-weight:600;">类别精确筛选：</span>
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
                    <span style="color:var(--text-muted);">高置信(≥0.75):</span>
                    <input type="checkbox" id="chkHigh" onchange="toggleHigh()" style="cursor:pointer;" />
                    <span style="color:var(--text-muted);margin-left:6px;">搜索:</span>
                    <input type="text" class="search-box" id="searchBox" placeholder="输入图号..." oninput="onSearch()" />
                </div>
            </div>
        </div>

        <main>
            <div class="info-card">
                <div class="info-row1">
                    <div class="stem-title">
                        <span id="stemTitle">【图号 00000000】</span>
                        <span id="stemCountsBadge" style="font-size:13px;font-weight:500;color:var(--text-dim);"></span>
                    </div>
                    <div id="stemSummaryText" style="font-size:13px;color:#fbbf24;font-weight:600;"></div>
                </div>

                <div class="box-breakdown">
                    <div class="box-list-card" style="border-left: 3px solid var(--f5-color);">
                        <div style="font-size:13px;font-weight:700;color:var(--f5-color);display:flex;justify-content:space-between;">
                            <span id="lblMissedTitle">📉 候选模型 遗漏的 标杆真框</span>
                            <span id="badgeMissedCount" style="font-size:12px;color:var(--text-muted);">0 个</span>
                        </div>
                        <div class="box-tags-wrap" id="missedTags"></div>
                    </div>

                    <div class="box-list-card" style="border-left: 3px solid var(--v23-color);">
                        <div style="font-size:13px;font-weight:700;color:var(--v23-color);display:flex;justify-content:space-between;">
                            <span id="lblFPTitle">📈 候选模型 独有多报的伪框/虚警</span>
                            <span id="badgeFPCount" style="font-size:12px;color:var(--text-muted);">0 个</span>
                        </div>
                        <div class="box-tags-wrap" id="fpTags"></div>
                    </div>
                </div>
            </div>

            <div class="stage-container" onclick="openLightbox(document.getElementById('mainStageImg').src)">
                <img id="mainStageImg" src="" alt="面板加载中..." />
            </div>

            <div class="thumb-carousel" id="thumbCarousel"></div>

            <div style="font-size:12px;color:var(--text-muted);text-align:center;padding-bottom:10px;">
                快捷键：键盘左右键 <kbd>←</kbd> <kbd>→</kbd> 快速翻页，<kbd>1</kbd> 遗漏真框，<kbd>2</kbd> 多报伪框，<kbd>3</kbd> 纯遗漏，<kbd>4</kbd> 纯多报，<kbd>5</kbd> 类别冲突。点击图片可全屏放大。
            </div>
        </main>
    </div>

    <!-- Gallery View Port -->
    <div class="gallery-view" id="gallerySection" style="padding:20px 24px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
            <div style="display:flex;gap:6px;">
                <span class="chip active" onclick="filterGallery('ALL')">全部 (27)</span>
                <span class="chip" onclick="filterGallery('CatA')">CatA: 标杆漏检真实体 (10)</span>
                <span class="chip" onclick="filterGallery('CatB')">CatB: 候选模型漏检 (4)</span>
                <span class="chip" onclick="filterGallery('CatC')">CatC: 背景杂波治理 (5)</span>
                <span class="chip" onclick="filterGallery('CatD')">CatD: 密集重叠解耦 (3)</span>
                <span class="chip" onclick="filterGallery('CatE')">CatE: 类别语义冲突 (5)</span>
            </div>
            <div style="font-size:12px;color:var(--text-muted);">点击卡片查看原图超高清面板</div>
        </div>
        <div class="gallery-grid" id="galleryGrid"></div>
    </div>

    <!-- Matrix View Port -->
    <div class="matrix-view" id="matrixSection" style="padding:20px 24px;">
        <div style="font-size:16px;font-weight:700;margin-bottom:8px;">📊 全模型横向演进对撞矩阵 (F5 标杆锚定)</div>
        <table class="matrix-table">
            <thead>
                <tr>
                    <th>模型版本</th>
                    <th>官方/实测分</th>
                    <th>差异图像数</th>
                    <th>📉 遗漏标杆真框数</th>
                    <th>📈 独有多报伪框数</th>
                    <th>🟩 双方强共识框数</th>
                    <th>演进特征与物理诊断</th>
                </tr>
            </thead>
            <tbody id="matrixTableBody"></tbody>
        </table>
    </div>

    <!-- Lightbox Modal -->
    <div class="lightbox" id="lightboxModal" onclick="closeLightbox()">
        <img id="lightboxImg" src="" alt="放大图" />
    </div>

    <script>
        const HUB = {json_str};

        let curModelId = 'v23';
        let curMainMode = 'missed';
        let curClassId = -1;
        let onlyHighConf = false;
        let searchQuery = '';

        let filteredItems = [];
        let curIndex = 0;

        // Init Model Switcher
        function initModelSwitcher() {{
            const sw = document.getElementById('modelSwitcher');
            sw.innerHTML = HUB.meta.model_list.map(m => 
                `<button class="model-pill ${{m.id === curModelId ? 'active' : ''}}" onclick="switchModel('${{m.id}}')">
                    <span>${{m.name}}</span>
                    <span style="font-size:11px;opacity:0.8;">[${{m.score}}]</span>
                </button>`
            ).join('');
        }}

        function switchModel(modelId) {{
            curModelId = modelId;
            document.querySelectorAll('.model-pill').forEach((p, idx) => {{
                if (HUB.meta.model_list[idx].id === modelId) p.classList.add('active');
                else p.classList.remove('active');
            }});
            updateModelKPIs();
            applyFilters();
        }}

        function switchSection(secKey) {{
            document.querySelectorAll('.sec-tab').forEach((tab, idx) => {{
                const keys = ['inspector', 'gallery', 'matrix'];
                if (keys[idx] === secKey) tab.classList.add('active');
                else tab.classList.remove('active');
            }});

            document.getElementById('inspectorSection').style.display = secKey === 'inspector' ? 'flex' : 'none';
            document.getElementById('gallerySection').style.display = secKey === 'gallery' ? 'block' : 'none';
            document.getElementById('matrixSection').style.display = secKey === 'matrix' ? 'block' : 'none';

            if (secKey === 'gallery') renderGallery('ALL');
            if (secKey === 'matrix') renderMatrix();
        }}

        function updateModelKPIs() {{
            const m = HUB.models[curModelId];
            document.getElementById('kpiTotalDiff').innerText = `${{m.meta.diff_images}} 张`;
            document.getElementById('kpiMissed').innerText = `${{m.meta.missed_boxes}} 框`;
            document.getElementById('kpiFP').innerText = `${{m.meta.fp_boxes}} 框`;
            document.getElementById('kpiConsensus').innerText = `${{m.meta.consensus_boxes}} 框`;
            document.getElementById('kpiConflict').innerText = `${{m.meta.conflict_boxes}} 处`;

            document.getElementById('cntMissed').innerText = m.items.filter(x => x.has_missed).length;
            document.getElementById('cntFP').innerText = m.items.filter(x => x.has_fp).length;
            document.getElementById('cntPureMiss').innerText = m.items.filter(x => x.is_pure_miss).length;
            document.getElementById('cntPureFP').innerText = m.items.filter(x => x.is_pure_fp).length;
            document.getElementById('cntMixed').innerText = m.items.filter(x => x.is_mixed).length;
            document.getElementById('cntConflict').innerText = m.items.filter(x => x.conf_cnt > 0).length;
            document.getElementById('cntAll').innerText = m.items.length;

            document.getElementById('lblMissedTitle').innerText = `📉 ${{m.name}} 遗漏的 F5 真框`;
            document.getElementById('lblFPTitle').innerText = `📈 ${{m.name}} 独有多报的伪框/虚警`;

            for (let cid = 0; cid < 12; cid++) {{
                const el = document.getElementById(`clsCnt${{cid}}`);
                if (!el) continue;
                if (curMainMode === 'missed' || curMainMode === 'pure_miss') {{
                    el.innerText = m.meta.cls_missed[cid] || 0;
                }} else if (curMainMode === 'fp' || curMainMode === 'pure_fp') {{
                    el.innerText = m.meta.cls_fp[cid] || 0;
                }} else {{
                    el.innerText = (m.meta.cls_missed[cid] || 0) + (m.meta.cls_fp[cid] || 0);
                }}
            }}
        }}

        function applyFilters() {{
            const m = HUB.models[curModelId];
            filteredItems = m.items.filter(it => {{
                if (curMainMode === 'missed' && !it.has_missed) return false;
                if (curMainMode === 'fp' && !it.has_fp) return false;
                if (curMainMode === 'pure_miss' && !it.is_pure_miss) return false;
                if (curMainMode === 'pure_fp' && !it.is_pure_fp) return false;
                if (curMainMode === 'mixed' && !it.is_mixed) return false;
                if (curMainMode === 'conflict' && it.conf_cnt === 0) return false;

                if (curClassId !== -1) {{
                    if (curMainMode === 'missed' || curMainMode === 'pure_miss') {{
                        if (!it.missed_cids.includes(curClassId)) return false;
                    }} else if (curMainMode === 'fp' || curMainMode === 'pure_fp') {{
                        if (!it.fp_cids.includes(curClassId)) return false;
                    }} else {{
                        if (!it.missed_cids.includes(curClassId) && !it.fp_cids.includes(curClassId)) return false;
                    }}
                }}

                if (onlyHighConf) {{
                    if (curMainMode === 'missed' || curMainMode === 'pure_miss') {{
                        if (it.max_miss_score < 0.75) return false;
                    }} else if (curMainMode === 'fp' || curMainMode === 'pure_fp') {{
                        if (it.max_fp_score < 0.75) return false;
                    }} else {{
                        if (it.max_miss_score < 0.75 && it.max_fp_score < 0.75) return false;
                    }}
                }}

                if (searchQuery.trim() !== '') {{
                    if (!it.stem.toLowerCase().includes(searchQuery.trim().toLowerCase())) return false;
                }}

                return true;
            }});

            curIndex = 0;
            renderInspector();
            renderCarousel();
        }}

        function renderInspector() {{
            const m = HUB.models[curModelId];
            const total = filteredItems.length;
            const curLabel = document.getElementById('curIndexLabel');
            if (total === 0) {{
                curLabel.innerText = "0 / 0";
                document.getElementById('stemTitle').innerText = "无符合筛选条件的样本";
                document.getElementById('stemCountsBadge').innerText = "";
                document.getElementById('stemSummaryText').innerText = "";
                document.getElementById('missedTags').innerHTML = '<span style="color:#64748b;font-size:12px;">无</span>';
                document.getElementById('fpTags').innerHTML = '<span style="color:#64748b;font-size:12px;">无</span>';
                document.getElementById('mainStageImg').src = "";
                return;
            }}

            curLabel.innerText = `${{curIndex + 1}} / ${{total}}`;
            const curItem = filteredItems[curIndex];

            document.getElementById('stemTitle').innerText = `【图号 ${{curItem.stem}}】`;
            document.getElementById('stemCountsBadge').innerText = 
                `F5标杆: ${{curItem.a_cnt}}框 | ${{m.name}}: ${{curItem.b_cnt}}框 | 共识: ${{curItem.exact_cnt}}框 | 冲突: ${{curItem.conf_cnt}}处`;
            document.getElementById('stemSummaryText').innerText = curItem.summary;

            document.getElementById('badgeMissedCount').innerText = `${{curItem.missed_cnt}} 个`;
            document.getElementById('missedTags').innerHTML = curItem.missed_boxes.map(b => 
                `<span class="tag-missed">🔷 ${{b.czh}} (${{b.score.toFixed(2)}})</span>`
            ).join('') || '<span style="color:#64748b;font-size:12px;">无遗漏真框</span>';

            document.getElementById('badgeFPCount').innerText = `${{curItem.fp_cnt}} 个`;
            document.getElementById('fpTags').innerHTML = curItem.fp_boxes.map(b => 
                `<span class="tag-fp">🔴 ${{b.czh}} (${{b.score.toFixed(2)}})</span>`
            ).join('') || '<span style="color:#64748b;font-size:12px;">无多报伪框</span>';

            document.getElementById('mainStageImg').src = `${{m.img_folder}}/${{curItem.stem}}.png`;
        }}

        function renderCarousel() {{
            const car = document.getElementById('thumbCarousel');
            if (filteredItems.length === 0) {{
                car.innerHTML = '<span style="color:#64748b;font-size:12px;">无样本</span>';
                return;
            }}
            car.innerHTML = filteredItems.map((item, idx) => {{
                let badge = '';
                if (item.missed_cnt > 0 && item.fp_cnt > 0) badge = `📉${{item.missed_cnt}} 📈${{item.fp_cnt}}`;
                else if (item.missed_cnt > 0) badge = `📉-${{item.missed_cnt}}`;
                else if (item.fp_cnt > 0) badge = `📈+${{item.fp_cnt}}`;
                else badge = `⚠️${{item.conf_cnt}}`;

                return `<div class="thumb-item ${{idx === curIndex ? 'active' : ''}}" onclick="jumpTo(${{idx}})">
                    ${{item.stem}} <span style="font-size:10px;opacity:0.8;">[${{badge}}]</span>
                </div>`;
            }}).join('');

            const activeEl = car.querySelector('.thumb-item.active');
            if (activeEl) activeEl.scrollIntoView({{ behavior: 'smooth', block: 'nearest', inline: 'center' }});
        }}

        function jumpTo(idx) {{
            if (idx >= 0 && idx < filteredItems.length) {{
                curIndex = idx;
                renderInspector();
                document.querySelectorAll('.thumb-item').forEach((el, i) => {{
                    if (i === curIndex) el.classList.add('active');
                    else el.classList.remove('active');
                }});
            }}
        }}

        function setMainMode(mode) {{
            curMainMode = mode;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            updateModelKPIs();
            applyFilters();
        }}

        function setClassFilter(cid) {{
            curClassId = cid;
            document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            applyFilters();
        }}

        function toggleHigh() {{
            onlyHighConf = document.getElementById('chkHigh').checked;
            applyFilters();
        }}

        function onSearch() {{
            searchQuery = document.getElementById('searchBox').value;
            applyFilters();
        }}

        function prevImg() {{
            if (filteredItems.length === 0) return;
            curIndex = (curIndex - 1 + filteredItems.length) % filteredItems.length;
            renderInspector();
            renderCarousel();
        }}

        function nextImg() {{
            if (filteredItems.length === 0) return;
            curIndex = (curIndex + 1) % filteredItems.length;
            renderInspector();
            renderCarousel();
        }}

        function openLightbox(src) {{
            if (!src) return;
            document.getElementById('lightboxImg').src = src;
            document.getElementById('lightboxModal').style.display = 'flex';
        }}

        function closeLightbox() {{
            document.getElementById('lightboxModal').style.display = 'none';
        }}

        // Render Gallery
        function renderGallery(filterGroup) {{
            const grid = document.getElementById('galleryGrid');
            const items = HUB.gallery.filter(it => filterGroup === 'ALL' || it.group === filterGroup);
            grid.innerHTML = items.map(it => 
                `<div class="gal-card" onclick="openLightbox('${{it.path}}')">
                    <img src="${{it.path}}" alt="${{it.stem}}" loading="lazy" />
                    <div class="gal-body">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                            <span style="font-weight:700;font-size:13px;">图号 ${{it.stem}}</span>
                            <span style="font-size:11px;font-weight:700;padding:2px 6px;border-radius:4px;background:#2563eb;color:#fff;">${{it.badge}}</span>
                        </div>
                        <div style="font-size:12px;color:var(--text-dim);line-height:1.4;">${{it.desc}}</div>
                    </div>
                </div>`
            ).join('');
        }}

        function filterGallery(group) {{
            document.querySelectorAll('#gallerySection .chip').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            renderGallery(group);
        }}

        // Render Matrix Table
        function renderMatrix() {{
            const tb = document.getElementById('matrixTableBody');
            tb.innerHTML = HUB.meta.model_list.map(m => {{
                const mod = HUB.models[m.id];
                return `<tr>
                    <td style="font-weight:700;color:${{m.color}};">${{m.name}}</td>
                    <td><span style="font-weight:700;font-family:monospace;">${{m.score}}</span></td>
                    <td><span style="font-family:monospace;">${{m.diff_cnt}} 张</span></td>
                    <td><span style="color:var(--f5-color);font-weight:700;font-family:monospace;">${{m.missed_cnt}} 框</span></td>
                    <td><span style="color:var(--v23-color);font-weight:700;font-family:monospace;">${{m.fp_cnt}} 框</span></td>
                    <td><span style="color:var(--consensus);font-weight:700;font-family:monospace;">${{m.consensus_cnt}} 框</span></td>
                    <td style="color:var(--text-dim);font-size:12px;">${{mod.desc}}</td>
                </tr>`;
            }}).join('');
        }}

        // Keybindings
        window.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'INPUT') return;
            if (e.key === 'ArrowLeft') prevImg();
            else if (e.key === 'ArrowRight') nextImg();
            else if (e.key === 'Escape') closeLightbox();
            else if (e.key === '1') setMainMode('missed');
            else if (e.key === '2') setMainMode('fp');
            else if (e.key === '3') setMainMode('pure_miss');
            else if (e.key === '4') setMainMode('pure_fp');
            else if (e.key === '5') setMainMode('conflict');
            else if (e.key === '0') setMainMode('all');
        }});

        // Init
        document.addEventListener('DOMContentLoaded', () => {{
            initModelSwitcher();
            updateModelKPIs();
            applyFilters();
        }});
    </script>
</body>
</html>
'''
    return html

html_content = build_spa_html(hub_data)
spa_path = os.path.join(OUTPUT_DIR, 'index.html')
with open(spa_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"Successfully generated Universal SPA Station: {spa_path}")

spa_alias = os.path.join(OUTPUT_DIR, 'universal_comparison_system.html')
with open(spa_alias, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"Successfully generated Universal SPA Alias: {spa_alias}")
