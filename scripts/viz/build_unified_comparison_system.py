#!/usr/bin/env python3
"""
Build Unified Multimodal Audit & Comparison Master Portal:
"AIC 2026 多模态检测统一对比与审计系统"
Outputs:
  outputs/reports/index.html
  outputs/reports/unified_comparison_system.html
"""

import os
import json

OUTPUT_DIR = '/home/constant/CodeBuddy/Dev/AIC/outputs'
DISCR_DIR = os.path.join(OUTPUT_DIR, 'visualizations_discrepancy')

# Scan CatA-CatE images
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

print(f"Loaded {len(gallery_items)} gallery showcase panels.")

html_template = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIC 2026 多模态检测统一对比与审计系统 (Unified Audit Station)</title>
    <style>
        :root {{
            --bg-dark: #0a0c10;
            --bg-header: #07080a;
            --bg-sidebar: #101319;
            --bg-card: #161a22;
            --bg-card-hover: #1e2430;
            --border: #232936;
            --border-light: #323b4e;
            --text-main: #f1f5f9;
            --text-dim: #94a3b8;
            --text-muted: #64748b;
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.4);
            --f5-color: #06b6d4;
            --v23-color: #f43f5e;
            --consensus: #10b981;
            --amber: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }}
        body {{ background: var(--bg-dark); color: var(--text-main); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}

        /* Top Executive Header */
        header {{
            background: var(--bg-header);
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
            z-index: 100;
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
        
        .score-chips {{ display: flex; gap: 8px; align-items: center; }}
        .score-chip {{
            background: #131720;
            border: 1px solid var(--border);
            padding: 3px 9px;
            border-radius: 4px;
            font-size: 11px;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .score-chip .lbl {{ color: var(--text-dim); }}
        .score-chip .val {{ font-weight: 700; }}
        
        .header-actions {{ display: flex; gap: 8px; align-items: center; }}
        .hdr-btn {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-dim);
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .hdr-btn:hover {{ background: var(--bg-card-hover); color: #fff; border-color: var(--primary); }}

        /* Primary Navigation Tabs Bar */
        .nav-tabs-bar {{
            background: var(--bg-sidebar);
            border-bottom: 1px solid var(--border);
            padding: 0 16px;
            display: flex;
            align-items: center;
            gap: 4px;
            flex-shrink: 0;
            overflow-x: auto;
        }}
        .nav-tab {{
            padding: 10px 16px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-dim);
            border-bottom: 2px solid transparent;
            cursor: pointer;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            gap: 8px;
            white-space: nowrap;
        }}
        .nav-tab:hover {{ color: #fff; background: rgba(255,255,255,0.02); }}
        .nav-tab.active {{
            color: #fff;
            border-bottom-color: var(--primary);
            background: rgba(59, 130, 246, 0.08);
        }}
        .tab-badge {{
            font-size: 11px;
            padding: 1px 6px;
            border-radius: 10px;
            font-weight: 700;
        }}
        .badge-cyan {{ background: rgba(6,182,212,0.18); color: var(--f5-color); border: 1px solid rgba(6,182,212,0.3); }}
        .badge-rose {{ background: rgba(244,63,94,0.18); color: var(--v23-color); border: 1px solid rgba(244,63,94,0.3); }}
        .badge-amber {{ background: rgba(245,158,11,0.18); color: var(--amber); border: 1px solid rgba(245,158,11,0.3); }}
        .badge-emerald {{ background: rgba(16,185,129,0.18); color: var(--consensus); border: 1px solid rgba(16,185,129,0.3); }}

        /* Sub-navigation bar */
        .sub-nav-bar {{
            background: #0d0f14;
            border-bottom: 1px solid var(--border);
            padding: 7px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
        }}
        .sub-tab-group {{ display: flex; gap: 6px; align-items: center; overflow-x: auto; }}
        .sub-tab-btn {{
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-dim);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.15s;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .sub-tab-btn:hover {{ color: #fff; background: var(--bg-card); }}
        .sub-tab-btn.active {{
            background: var(--bg-card);
            border-color: var(--border-light);
            color: #fff;
            font-weight: 600;
        }}
        .sub-info {{ font-size: 12px; color: var(--text-muted); display: flex; align-items: center; gap: 10px; }}

        /* Main Workspace Stage */
        .stage-area {{
            flex: 1;
            position: relative;
            overflow: hidden;
            background: #000;
        }}
        .view-frame {{
            width: 100%;
            height: 100%;
            border: none;
            display: block;
        }}

        /* Built-in CatA-CatE Gallery View */
        .gallery-view {{
            width: 100%;
            height: 100%;
            overflow-y: auto;
            padding: 20px 24px;
            background: var(--bg-dark);
            display: none;
        }}
        .gallery-header {{
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .gallery-filters {{ display: flex; gap: 6px; }}
        .gal-chip {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-dim);
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            font-weight: 600;
        }}
        .gal-chip.active {{ background: #2563eb; color: #fff; border-color: #3b82f6; }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 16px;
        }}
        .gallery-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .gallery-card:hover {{
            transform: translateY(-2px);
            border-color: var(--primary);
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        }}
        .card-img-wrap {{
            width: 100%;
            height: 200px;
            background: #050608;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .card-img-wrap img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s;
        }}
        .gallery-card:hover .card-img-wrap img {{ transform: scale(1.03); }}
        .card-body {{ padding: 12px 14px; display: flex; flex-direction: column; gap: 6px; }}
        .card-title-row {{ display: flex; justify-content: space-between; align-items: center; }}
        .card-badge {{ font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 4px; }}
        .badge-cata {{ background: rgba(6,182,212,0.2); color: var(--f5-color); border: 1px solid rgba(6,182,212,0.4); }}
        .badge-catb {{ background: rgba(244,63,94,0.2); color: var(--v23-color); border: 1px solid rgba(244,63,94,0.4); }}
        .badge-catc {{ background: rgba(16,185,129,0.2); color: var(--consensus); border: 1px solid rgba(16,185,129,0.4); }}
        .badge-catd {{ background: rgba(139,92,246,0.2); color: #a78bfa; border: 1px solid rgba(139,92,246,0.4); }}
        .badge-cate {{ background: rgba(245,158,11,0.2); color: var(--amber); border: 1px solid rgba(245,158,11,0.4); }}
        .card-stem {{ font-size: 14px; font-weight: 700; color: #fff; }}
        .card-desc {{ font-size: 12px; color: var(--text-dim); line-height: 1.4; }}

        /* Lightbox modal */
        .lightbox {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.92);
            z-index: 1000;
            display: none;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 24px;
        }}
        .lightbox img {{
            max-width: 96vw;
            max-height: 88vh;
            object-fit: contain;
            box-shadow: 0 0 50px rgba(0,0,0,0.9);
            border-radius: 4px;
        }}
        .lightbox-close {{
            position: absolute;
            top: 20px;
            right: 24px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: #fff;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            font-weight: 700;
        }}
        .lightbox-info {{
            margin-top: 12px;
            font-size: 14px;
            color: #fff;
            font-weight: 600;
            text-align: center;
        }}
    </style>
</head>
<body>

    <!-- Executive Header -->
    <header>
        <div class="brand-area">
            <div class="brand-badge">AIC 2026 AUDIT HUB</div>
            <div class="brand-title">AIC 2026 多模态检测统一对比与审计系统</div>
        </div>
        <div class="score-chips">
            <div class="score-chip"><span class="lbl">F5 标杆:</span><span class="val" style="color:var(--f5-color);">57.266</span></div>
            <div class="score-chip"><span class="lbl">V23:</span><span class="val" style="color:var(--v23-color);">53.00</span></div>
            <div class="score-chip"><span class="lbl">V24 Compact:</span><span class="val" style="color:#60a5fa;">53.555</span></div>
            <div class="score-chip"><span class="lbl">V25:</span><span class="val" style="color:#a78bfa;">56.40+</span></div>
            <div class="score-chip"><span class="lbl">V26 Golden:</span><span class="val" style="color:var(--consensus);">58.10+</span></div>
        </div>
        <div class="header-actions">
            <a class="hdr-btn" id="openExternalBtn" href="discrepancies_f5_vs_v23/index.html" target="_blank">
                <span>新标签打开 ↗</span>
            </a>
            <button class="hdr-btn" onclick="toggleFullscreen()">
                <span id="fsText">全屏模式</span>
            </button>
        </div>
    </header>

    <!-- Master Navigation Tabs -->
    <div class="nav-tabs-bar">
        <!-- Tab 1: V23 Core -->
        <div class="nav-tab active" onclick="switchMainTab('v23')">
            <span>👑 V23 深度对比核心 (遗漏真框 ⚔️ 多报伪框)</span>
            <span class="tab-badge badge-cyan">692 图</span>
        </div>

        <!-- Tab 2: Version Evolution -->
        <div class="nav-tab" onclick="switchMainTab('evolution')">
            <span>🚀 多版本横向演进 (V24 / V25 / V26)</span>
            <span class="tab-badge badge-rose">3 组对撞</span>
        </div>

        <!-- Tab 3: Rare Class Deep Dives -->
        <div class="nav-tab" onclick="switchMainTab('rare')">
            <span>🎯 长尾目标专项审查 (球 / 无人机)</span>
            <span class="tab-badge badge-amber">高微裁切片</span>
        </div>

        <!-- Tab 4: Classic Cases Gallery -->
        <div class="nav-tab" onclick="switchMainTab('gallery')">
            <span>🖼️ 经典典型案卷画廊 (CatA ~ CatE)</span>
            <span class="tab-badge badge-emerald">27 组切片</span>
        </div>

        <!-- Tab 5: Benchmark & Metrics -->
        <div class="nav-tab" onclick="switchMainTab('metrics')">
            <span>📊 全局指标体系与报告</span>
            <span class="tab-badge badge-cyan">mAP / Stage C</span>
        </div>
    </div>

    <!-- Sub Navigation Bar -->
    <div class="sub-nav-bar">
        <div class="sub-tab-group" id="subTabContainer">
            <!-- Dynamically populated based on active main tab -->
        </div>
        <div class="sub-info" id="subInfoText">
            <span>正在浏览：V23 深度对比核心 (F5 57.266 vs V23 53.00)</span>
        </div>
    </div>

    <!-- Stage Area -->
    <div class="stage-area">
        <!-- Main Iframe Host -->
        <iframe class="view-frame" id="mainIframe" src="discrepancies_f5_vs_v23/index.html"></iframe>

        <!-- Built-in CatA-CatE Showcase Gallery -->
        <div class="gallery-view" id="galleryView">
            <div class="gallery-header">
                <div class="gallery-filters">
                    <span class="gal-chip active" onclick="filterGallery('ALL')">全部 (27)</span>
                    <span class="gal-chip" onclick="filterGallery('CatA')">CatA: F5漏检真目标 (10)</span>
                    <span class="gal-chip" onclick="filterGallery('CatB')">CatB: Sovereign漏检 (4)</span>
                    <span class="gal-chip" onclick="filterGallery('CatC')">CatC: 背景杂波治理 (5)</span>
                    <span class="gal-chip" onclick="filterGallery('CatD')">CatD: 密集重叠解耦 (3)</span>
                    <span class="gal-chip" onclick="filterGallery('CatE')">CatE: 类别语义冲突 (5)</span>
                </div>
                <div style="font-size:12px;color:var(--text-muted);">
                    点击卡片查看原图超高清面板
                </div>
            </div>
            <div class="gallery-grid" id="galleryGrid">
                <!-- Gallery Cards rendered via JS -->
            </div>
        </div>
    </div>

    <!-- Lightbox Modal -->
    <div class="lightbox" id="lightbox" onclick="closeLightbox()">
        <button class="lightbox-close" onclick="closeLightbox()">✕ 关闭</button>
        <img id="lightboxImg" src="" alt="超高清面板" />
        <div class="lightbox-info" id="lightboxInfo"></div>
    </div>

    <script>
        const GALLERY_DATA = {json.dumps(gallery_items, ensure_ascii=False)};

        const MODULE_CONFIG = {{
            'v23': {{
                'name': '👑 V23 深度对比核心',
                'subtabs': [
                    {{ 'title': 'V23 全量深度对撞 (692图 · 真框遗漏 vs 多报伪框)', 'url': 'discrepancies_f5_vs_v23/index.html', 'desc': '包含 748 遗漏真框、1559 多报伪框、12 类精确过滤与键盘快捷流转' }}
                ]
            }},
            'evolution': {{
                'name': '🚀 多版本横向演进',
                'subtabs': [
                    {{ 'title': 'F5 vs V24 Native (664 图)', 'url': 'discrepancies_f5_vs_v24/index.html', 'desc': 'Sovereign V24 Native Compact (6,711 框) 实测对撞' }},
                    {{ 'title': 'F5 vs V25 Compact (660 图)', 'url': 'discrepancies_f5_vs_v25/index.html', 'desc': 'Sovereign V25 结构性纠偏与 Compact 压框对撞' }},
                    {{ 'title': 'F5 vs V26 Golden (655 图)', 'url': 'discrepancies_f5_vs_v26/index.html', 'desc': 'Sovereign V26 Golden 终局版本对撞' }}
                ]
            }},
            'rare': {{
                'name': '🎯 长尾目标专项审查',
                'subtabs': [
                    {{ 'title': '类别 7: 球 (Ball) 深度审查 (@ 0.45)', 'url': 'ball_analysis_f5_vs_v23/index.html', 'desc': '全图三栏 + 局部高分辨率直接微裁切片' }},
                    {{ 'title': '类别 7: 球 (Ball) 高阈值 (@ 0.70)', 'url': 'ball_analysis_f5_vs_v23_thr70/index.html', 'desc': '0.70 严格置信度下球体虚警出清状态' }},
                    {{ 'title': '类别 10: 无人机 (UAV) 深度审查 (@ 0.45)', 'url': 'uav_analysis_f5_vs_v23/index.html', 'desc': '全图三栏 + 局部无人机微裁切片' }},
                    {{ 'title': '类别 10: 无人机 (UAV) 高阈值 (@ 0.60)', 'url': 'uav_analysis_f5_vs_v23_thr60/index.html', 'desc': '0.60 严格置信度下无人机对撞' }}
                ]
            }},
            'gallery': {{
                'name': '🖼️ 经典典型案卷画廊',
                'isGallery': true,
                'subtabs': [
                    {{ 'title': 'CatA ~ CatE 典型案卷画廊 (27 组精细切片)', 'url': '#gallery', 'desc': '27 组核心典型对撞案卷高清原图展示' }}
                ]
            }},
            'metrics': {{
                'name': '📊 全局指标体系与报告',
                'subtabs': [
                    {{ 'title': '全模型 mAP 横向对撞大盘', 'url': 'model_map_comparison.html', 'desc': '全系列模型在验证集上的 mAP50 与 mAP50-95 演进折线雷达' }},
                    {{ 'title': 'Stage C 阶段综合评测报告', 'url': 'stageC_report.html', 'desc': 'RGB/IR 多模态特征融合攻坚评测终局' }},
                    {{ 'title': 'ViT-L 800 分辨率演进曲线', 'url': 'vitl800_map_curve.html', 'desc': 'ViT-L 800 分辨率学习率与收敛曲线' }}
                ]
            }}
        }};

        let curMainTab = 'v23';
        let curSubIndex = 0;

        function switchMainTab(tabKey) {{
            curMainTab = tabKey;
            curSubIndex = 0;
            
            // Update tab UI
            document.querySelectorAll('.nav-tab').forEach((tab, idx) => {{
                const keys = ['v23', 'evolution', 'rare', 'gallery', 'metrics'];
                if (keys[idx] === tabKey) tab.classList.add('active');
                else tab.classList.remove('active');
            }});

            renderSubTabs();
            loadActiveModule();
        }}

        function switchSubTab(idx) {{
            curSubIndex = idx;
            document.querySelectorAll('.sub-tab-btn').forEach((btn, i) => {{
                if (i === idx) btn.classList.add('active');
                else btn.classList.remove('active');
            }});
            loadActiveModule();
        }}

        function renderSubTabs() {{
            const conf = MODULE_CONFIG[curMainTab];
            const container = document.getElementById('subTabContainer');
            container.innerHTML = conf.subtabs.map((sub, idx) => 
                `<button class="sub-tab-btn ${{idx === curSubIndex ? 'active' : ''}}" onclick="switchSubTab(${{idx}})">
                    ${{sub.title}}
                </button>`
            ).join('');
        }}

        function loadActiveModule() {{
            const conf = MODULE_CONFIG[curMainTab];
            const sub = conf.subtabs[curSubIndex];
            const iframe = document.getElementById('mainIframe');
            const galleryView = document.getElementById('galleryView');
            const extBtn = document.getElementById('openExternalBtn');
            const infoText = document.getElementById('subInfoText');

            infoText.innerHTML = `<span><strong>${{conf.name}}</strong> › ${{sub.title}}</span>`;

            if (conf.isGallery) {{
                iframe.style.display = 'none';
                galleryView.style.display = 'block';
                extBtn.style.display = 'none';
                renderGalleryGrid('ALL');
            }} else {{
                galleryView.style.display = 'none';
                iframe.style.display = 'block';
                extBtn.style.display = 'flex';
                extBtn.href = sub.url;
                iframe.src = sub.url;
            }}
        }}

        function renderGalleryGrid(catFilter) {{
            const grid = document.getElementById('galleryGrid');
            const items = GALLERY_DATA.filter(it => catFilter === 'ALL' || it.group === catFilter);
            grid.innerHTML = items.map(it => {{
                let bClass = 'badge-cata';
                if (it.group === 'CatB') bClass = 'badge-catb';
                else if (it.group === 'CatC') bClass = 'badge-catc';
                else if (it.group === 'CatD') bClass = 'badge-catd';
                else if (it.group === 'CatE') bClass = 'badge-cate';

                return `<div class="gallery-card" onclick="openLightbox('${{it.path}}', '${{it.badge}}: ${{it.desc}}')">
                    <div class="card-img-wrap">
                        <img src="${{it.path}}" alt="${{it.stem}}" loading="lazy" />
                    </div>
                    <div class="card-body">
                        <div class="card-title-row">
                            <span class="card-stem">图号 ${{it.stem}}</span>
                            <span class="card-badge ${{bClass}}">${{it.badge}}</span>
                        </div>
                        <div class="card-desc">${{it.desc}}</div>
                    </div>
                </div>`;
            }}).join('');
        }}

        function filterGallery(group) {{
            document.querySelectorAll('.gal-chip').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            renderGalleryGrid(group);
        }}

        function openLightbox(src, desc) {{
            document.getElementById('lightboxImg').src = src;
            document.getElementById('lightboxInfo').innerText = desc;
            document.getElementById('lightbox').style.display = 'flex';
        }}

        function closeLightbox() {{
            document.getElementById('lightbox').style.display = 'none';
        }}

        function toggleFullscreen() {{
            if (!document.fullscreenElement) {{
                document.documentElement.requestFullscreen();
                document.getElementById('fsText').innerText = '退出全屏';
            }} else {{
                if (document.exitFullscreen) {{
                    document.exitFullscreen();
                    document.getElementById('fsText').innerText = '全屏模式';
                }}
            }}
        }}

        // Keybindings
        window.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeLightbox();
            else if (e.key === '1') switchMainTab('v23');
            else if (e.key === '2') switchMainTab('evolution');
            else if (e.key === '3') switchMainTab('rare');
            else if (e.key === '4') switchMainTab('gallery');
            else if (e.key === '5') switchMainTab('metrics');
            else if (e.key === 'f' || e.key === 'F') toggleFullscreen();
        }});

        // Init
        document.addEventListener('DOMContentLoaded', () => {{
            renderSubTabs();
            loadActiveModule();
        }});
    </script>
</body>
</html>
'''

with open(os.path.join(OUTPUT_DIR, 'index.html'), 'w', encoding='utf-8') as f:
    f.write(html_template)
print(f"Built Master Portal: {os.path.join(OUTPUT_DIR, 'index.html')}")

with open(os.path.join(OUTPUT_DIR, 'unified_comparison_system.html'), 'w', encoding='utf-8') as f:
    f.write(html_template)
print(f"Built Master Portal Alias: {os.path.join(OUTPUT_DIR, 'unified_comparison_system.html')}")
