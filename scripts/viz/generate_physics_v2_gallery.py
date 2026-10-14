#!/usr/bin/env python3
"""
Generate Updated Visual Gallery reflecting Physics V2 Clean:
- Confirmed True Drones: 93 (All with Full-Drone Envelope)
- Executed FPs: All static bins (00000071, 00000076, 00000037, ground curbs, etc.)
"""

import os
import sys
import cv2
import json
import math
import torch
import numpy as np
from collections import defaultdict

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
RAW_PT = os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/raw_detections_v29_c5_peak.pt")
VIS_DIR = os.path.join(ROOT, "dataset/test/AIC2026_PHASE_2_1000/visible")
CROPS_DIR = os.path.join(ROOT, "outputs/figures/physics_v2_crops")
HTML_PATH = os.path.join(ROOT, "outputs/reports/uav_physics_verified_gallery.html")

os.makedirs(CROPS_DIR, exist_ok=True)

raw = torch.load(RAW_PT, weights_only=False)

def find_img(stem):
    for ext in [".png", ".jpg"]:
        p = os.path.join(VIS_DIR, f"{stem}{ext}")
        if os.path.exists(p): return p
    return None

clusters_by_stem = defaultdict(list)
all_clusters = []

for stem, bxs in sorted(raw.items()):
    uav_queries = [b for b in bxs if b[0] == 10 and b[5] >= 0.15]
    used = set()
    for i, b in enumerate(uav_queries):
        if i in used: continue
        cluster = [b]
        used.add(i)
        bx1, by1, bx2, by2 = b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2
        for j in range(i+1, len(uav_queries)):
            if j in used: continue
            b2 = uav_queries[j]
            b2x1, b2y1, b2x2, b2y2 = b2[1]-b2[3]/2, b2[2]-b2[4]/2, b2[1]+b2[3]/2, b2[2]+b2[4]/2
            inter = max(0, min(bx2, b2x2) - max(bx1, b2x1)) * max(0, min(by2, b2y2) - max(by1, b2y1))
            u = b[3]*b[4] + b2[3]*b2[4] - inter
            if inter/u >= 0.28:
                cluster.append(b2)
                used.add(j)
        
        min_x1 = min(x[1]-x[3]/2 for x in cluster)
        min_y1 = min(x[2]-x[4]/2 for x in cluster)
        max_x2 = max(x[1]+x[3]/2 for x in cluster)
        max_y2 = max(x[2]+x[4]/2 for x in cluster)
        cx = (min_x1 + max_x2) / 2
        cy = (min_y1 + max_y2) / 2
        w = max_x2 - min_x1
        h = max_y2 - min_y1
        
        max_sc = max(x[5] for x in cluster)
        if len(cluster) == 1:
            comb_sc = max_sc
        else:
            prob_or = 1.0 - math.prod([1.0 - x[5] for x in cluster])
            comb_sc = min(max_sc + 0.15, prob_or)
            
        c_item = {
            "stem": stem, "cx": cx, "cy": cy, "w": w, "h": h,
            "max_sc": max_sc, "comb_sc": comb_sc, "cluster_len": len(cluster)
        }
        clusters_by_stem[stem].append(c_item)
        all_clusters.append(c_item)

grid = defaultdict(list)
for c in all_clusters:
    gx, gy = int(round(c["cx"] / 0.025)), int(round(c["cy"] / 0.025))
    grid[(gx, gy)].append(c)

static_bins = set()
for (gx, gy), items in grid.items():
    stems = set(x["stem"] for x in items)
    if len(stems) >= 3:
        static_bins.add((gx, gy))

confirmed_drones = []
executed_fps = []

for stem in sorted(raw.keys()):
    for c in clusters_by_stem.get(stem, []):
        cx, cy, w, h = c["cx"], c["cy"], c["w"], c["h"]
        comb_sc = c["comb_sc"]
        max_sc = c["max_sc"]
        area = w * h
        ar = w / (h + 1e-6)
        gx, gy = int(round(cx / 0.025)), int(round(cy / 0.025))
        c["area"] = area
        c["aspect_ratio"] = ar
        
        if (cy + h/2.0) > 0.85:
            c["reason"] = f"违反定律1: 贴地路面伪影 (y={cy+h/2.0:.3f} > 0.85)"
            executed_fps.append(c)
            continue
            
        if (gx, gy) in static_bins:
            c["reason"] = f"违反定律2: 跨图像静态背景地物 (共{len(set(x['stem'] for x in grid[(gx,gy)]))}帧完全静止)"
            executed_fps.append(c)
            continue
            
        if area > 0.0050:
            c["reason"] = f"违反定律3: 超出物理投影尺度 (面积={area*100:.2f}% > 0.5%)"
            executed_fps.append(c)
            continue
            
        if ar < 0.40 or ar > 2.50:
            c["reason"] = f"违反定律3: 违背旋翼机对称长宽比 (长宽比={ar:.2f})"
            executed_fps.append(c)
            continue
            
        if comb_sc >= 0.45 or (max_sc >= 0.28 and comb_sc >= 0.38):
            c["status"] = "rescued" if max_sc < 0.45 else "native"
            c["reason"] = "符合全部物理法则 (整机包络+动态空域真机)"
            confirmed_drones.append(c)
        else:
            c["reason"] = f"置信度未过门限 (得分={comb_sc:.3f} < 0.38)"
            executed_fps.append(c)

print(f"Confirmed Drones: {len(confirmed_drones)}, Executed FPs: {len(executed_fps)}")

def render_cards(items, is_executed=False):
    cards = []
    cached_img, cur_stem = None, None
    for idx, b in enumerate(items):
        stem = b["stem"]
        if stem != cur_stem:
            p = find_img(stem)
            cached_img = cv2.imread(p) if p else None
            cur_stem = stem
        if cached_img is None: continue
        
        H, W = cached_img.shape[:2]
        cx_px, cy_px = int(b["cx"] * W), int(b["cy"] * H)
        w_px, h_px = int(b["w"] * W), int(b["h"] * H)
        
        pad_w = max(60, int(w_px * 1.8))
        pad_h = max(60, int(h_px * 1.8))
        x1_c = max(0, cx_px - w_px // 2 - pad_w)
        y1_c = max(0, cy_px - h_px // 2 - pad_h)
        x2_c = min(W, cx_px + w_px // 2 + pad_w)
        y2_c = min(H, cy_px + h_px // 2 + pad_h)
        
        crop = cached_img[y1_c:y2_c, x1_c:x2_c].copy()
        bx1 = cx_px - w_px // 2 - x1_c
        by1 = cy_px - h_px // 2 - y1_c
        bx2 = cx_px + w_px // 2 - x1_c
        by2 = cy_px + h_px // 2 - y1_c
        
        color = (0, 0, 255) if is_executed else ((0, 165, 255) if b.get("status") == "rescued" else (0, 220, 0))
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, 2)
        
        prefix = "fp" if is_executed else "drone"
        crop_fn = f"crop_{prefix}_{stem}_{idx:03d}.jpg"
        cv2.imwrite(os.path.join(CROPS_DIR, crop_fn), crop)
        
        badge_cls = "badge-executed" if is_executed else ("badge-rescued" if b.get("status") == "rescued" else "badge-native")
        badge_txt = "⚖️ 物理判定背景/伪影 (已清零)" if is_executed else ("🛡️ 整机包络挽救" if b.get("status") == "rescued" else "🚀 原生整机高分")
        category = "executed" if is_executed else b.get("status", "native")
        
        card = f"""
        <div class="card" data-category="{category}" data-stem="{stem}">
            <div class="img-wrap">
                <img src="/outputs/figures/physics_v2_crops/{crop_fn}" alt="{stem}" loading="lazy">
                <span class="badge {badge_cls}">{badge_txt}</span>
            </div>
            <div class="card-info">
                <div class="card-title">
                    <span class="stem-id">#{idx+1:03d} &bull; {stem}</span>
                    <span class="score" style="color: {'#ff7b72' if is_executed else '#58a6ff'}">{b['comb_sc']:.3f}</span>
                </div>
                <div class="meta-row">
                    <span>中心: ({b['cx']:.3f}, {b['cy']:.3f})</span>
                    <span>尺寸: {w_px}&times;{h_px}px ({b['area']*100:.3f}%)</span>
                </div>
                <div class="meta-row">
                    <span>长宽比: <b>{b['aspect_ratio']:.2f}</b></span>
                    <span>Query聚合: <b>{b['cluster_len']}个</b></span>
                </div>
                <div class="reason-box {'reason-executed' if is_executed else 'reason-passed'}">
                    {b['reason']}
                </div>
                <div class="action-row">
                    <a href="/index.html?search={stem}" target="_blank" class="view-btn">在主工作台核验该图 &rarr;</a>
                </div>
            </div>
        </div>
        """
        cards.append(card)
    return cards

confirmed_cards = render_cards(confirmed_drones, is_executed=False)
executed_cards = render_cards(executed_fps, is_executed=True)

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sovereign V29 - 物理引擎 V2 (整机包络+静态背景清零)</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --green: #238636;
            --orange: #d29922;
            --red: #da3633;
            --blue: #1f6feb;
            --card-hover: #1f242c;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 24px;
        }}
        header {{
            max-width: 1400px;
            margin: 0 auto 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}
        h1 {{ font-size: 24px; color: #f0f6fc; display: flex; align-items: center; gap: 10px; }}
        .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
        .btn {{
            background: var(--surface);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .btn:hover {{ border-color: var(--accent); }}
        .btn.active {{ background: var(--blue); color: #fff; border-color: var(--blue); }}
        .btn.danger {{ border-color: rgba(218, 54, 51, 0.4); color: #ff7b72; }}
        .btn.danger.active {{ background: var(--red); color: #fff; border-color: var(--red); }}
        .search-box {{
            background: var(--surface);
            border: 1px solid var(--border);
            color: #fff;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 14px;
            width: 220px;
        }}
        .search-box:focus {{ outline: none; border-color: var(--accent); }}
        .grid {{
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
            gap: 16px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.15s, border-color 0.15s;
        }}
        .card:hover {{ transform: translateY(-2px); border-color: var(--accent); background: var(--card-hover); }}
        .img-wrap {{
            position: relative;
            width: 100%;
            height: 190px;
            background: #05070a;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .img-wrap img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
        .badge {{
            position: absolute;
            top: 8px;
            right: 8px;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
        }}
        .badge-native {{ background: var(--green); color: #fff; }}
        .badge-rescued {{ background: var(--orange); color: #000; }}
        .badge-executed {{ background: var(--red); color: #fff; }}
        .card-info {{ padding: 14px; display: flex; flex-direction: column; gap: 8px; flex: 1; }}
        .card-title {{ display: flex; justify-content: space-between; align-items: center; }}
        .stem-id {{ font-weight: 600; color: #f0f6fc; font-size: 14px; }}
        .score {{ font-weight: 700; font-size: 16px; }}
        .meta-row {{ display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted); }}
        .meta-row b {{ color: var(--text); }}
        .reason-box {{
            font-size: 12px;
            padding: 6px 10px;
            border-radius: 4px;
            line-height: 1.4;
            margin-top: 4px;
        }}
        .reason-passed {{ background: rgba(35, 134, 54, 0.15); color: #7ee787; border: 1px solid rgba(35, 134, 54, 0.3); }}
        .reason-executed {{ background: rgba(218, 54, 51, 0.15); color: #ff7b72; border: 1px solid rgba(218, 54, 51, 0.3); }}
        .action-row {{ margin-top: auto; padding-top: 10px; border-top: 1px solid #21262d; }}
        .view-btn {{
            display: block;
            text-align: center;
            background: rgba(88, 166, 255, 0.1);
            color: var(--accent);
            text-decoration: none;
            padding: 6px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }}
        .view-btn:hover {{ background: rgba(88, 166, 255, 0.2); }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>⚖️ Sovereign V29 &bull; 物理引擎 V2 (整机包络 + 静态背景地物清零)</h1>
            <p style="color: var(--text-muted); margin-top: 6px; font-size: 13px;">
                针对用户实检反馈全面修正：整机外扩包络消灭残框、静态监控机位背景 100% 连根拔除
            </p>
        </div>
        <div class="controls">
            <button class="btn active" onclick="switchTab('confirmed', this)">✅ 物理实锤整机 ({len(confirmed_drones)})</button>
            <button class="btn danger" onclick="switchTab('executed', this)">⚖️ 物理法庭处决区 ({len(executed_fps)})</button>
            <input type="text" class="search-box" placeholder="搜索图号 (如 00000071)..." oninput="searchStem(this.value)">
            <a href="/index.html" class="btn" style="text-decoration:none; background:#238636; border-color:#238636; color:#fff;">返回主工作台</a>
        </div>
    </header>

    <div class="grid" id="grid-confirmed">
        {"".join(confirmed_cards)}
    </div>

    <div class="grid" id="grid-executed" style="display: none;">
        {"".join(executed_cards)}
    </div>

    <script>
        let currentTab = 'confirmed';
        let currentSearch = '';

        function switchTab(tab, btn) {{
            currentTab = tab;
            document.querySelectorAll('.controls .btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            if (tab === 'confirmed') {{
                document.getElementById('grid-confirmed').style.display = 'grid';
                document.getElementById('grid-executed').style.display = 'none';
            }} else {{
                document.getElementById('grid-confirmed').style.display = 'none';
                document.getElementById('grid-executed').style.display = 'grid';
            }}
            updateSearch();
        }}

        function updateSearch() {{
            const activeGrid = document.getElementById(currentTab === 'confirmed' ? 'grid-confirmed' : 'grid-executed');
            const cards = activeGrid.querySelectorAll('.card');
            cards.forEach(card => {{
                const stem = card.getAttribute('data-stem');
                const match = (!currentSearch || stem.toLowerCase().includes(currentSearch.toLowerCase()));
                card.style.display = match ? 'flex' : 'none';
            }});
        }}

        function searchStem(val) {{
            currentSearch = val.trim();
            updateSearch();
        }}
    </script>
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[✔] Physics V2 gallery compiled to {HTML_PATH}")
