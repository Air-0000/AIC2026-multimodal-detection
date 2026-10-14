#!/usr/bin/env python3
"""
Generate comprehensive visual discrepancy inspection panels between
F5 (57.266 Gold Standard) and Sovereign V24 Native Compact (6,711 boxes).

Directory Output Structure:
outputs/analysis/discrepancies_f5_vs_v24/
├── 01_category_conflicts/     # 22 images: Same object, different class labels
├── 02_large_diff_count/       # Images with >= 5 box differences
├── 03_medium_diff_count/      # Images with 2-4 box differences
├── 04_single_box_diff/        # Images with 1 box difference
├── all_discrepancies/         # All 664 images with differences
├── index.html                 # Interactive Web Gallery with keyboard navigation
└── discrepancy_report.json    # Full structured discrepancy metadata
"""

import os
import sys
import json
import zipfile
import shutil
import concurrent.futures
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}
RARE_CLASSES = {1, 7, 9, 10, 11}

# Colors (RGB)
C_EXACT    = (46, 204, 113)    # Emerald Green (Consensus)
C_CONF     = (255, 140, 0)     # Bright Orange (Conflict)
C_F5_ONLY  = (0, 195, 255)     # Electric Cyan (F5 Only)
C_V24_ONLY = (255, 45, 85)     # Crimson / Magenta (V24 Only)
C_BG       = (20, 22, 26)      # Dark Background
C_BANNER   = (12, 14, 18)      # Header
C_CARD     = (28, 32, 38)
C_TEXT     = (240, 245, 250)
C_MUTED    = (160, 168, 180)

# Paths
F5_DIR = '/home/constant/Downloads/AIC2026_phase2_predictions_F5seed20261002_RARE45_COMMON50_compact7212_20261005'
V24_ZIP = '/home/constant/CodeBuddy/Dev/AIC/outputs/submissions/submissions_v24/submission_v24_native_compact.zip'
VIS_DIR = '/home/constant/CodeBuddy/Dev/AIC/dataset/test/AIC2026_PHASE_2_1000/visible'
IR_DIR  = '/home/constant/CodeBuddy/Dev/AIC/dataset/test/AIC2026_PHASE_2_1000/infrared'

OUTPUT_ROOT = '/home/constant/CodeBuddy/Dev/AIC/outputs/analysis/discrepancies_f5_vs_v24'
DIR_CONFLICTS = os.path.join(OUTPUT_ROOT, '01_category_conflicts')
DIR_LARGE     = os.path.join(OUTPUT_ROOT, '02_large_diff_count')
DIR_MEDIUM    = os.path.join(OUTPUT_ROOT, '03_medium_diff_count')
DIR_SINGLE    = os.path.join(OUTPUT_ROOT, '04_single_box_diff')
DIR_ALL       = os.path.join(OUTPUT_ROOT, 'all_discrepancies')

FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'


def load_f5_predictions(path):
    d = {}
    for fname in sorted(os.listdir(path)):
        if not fname.endswith('.txt'): continue
        stem = fname.rsplit('.', 1)[0]
        with open(os.path.join(path, fname)) as f:
            lines = [l.strip() for l in f if l.strip()]
        boxes = []
        for l in lines:
            parts = l.split()
            cid = int(parts[0])
            cx, cy, w, h, sc = map(float, parts[1:])
            thr = 0.45 if cid in RARE_CLASSES else 0.50
            if sc >= thr:
                boxes.append((cid, cx, cy, w, h, sc))
        boxes.sort(key=lambda x: x[5], reverse=True)
        d[stem] = boxes[:60]
    return d


def load_v24_predictions(path):
    d = {}
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith('.txt'): continue
            stem = name.rsplit('.', 1)[0]
            content = zf.read(name).decode('utf-8').strip()
            boxes = []
            for line in content.split('\n'):
                if not line.strip(): continue
                parts = line.split()
                cid = int(parts[0])
                cx, cy, w, h, sc = map(float, parts[1:])
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


def render_image_panel(item):
    stem, bf, bg, matched_f, matched_g = item
    
    vis_path = os.path.join(VIS_DIR, f"{stem}.jpg")
    ir_path = os.path.join(IR_DIR, f"{stem}.jpg")
    if not os.path.exists(vis_path):
        vis_path = os.path.join(VIS_DIR, f"{stem}.png")
    if not os.path.exists(ir_path):
        ir_path = os.path.join(IR_DIR, f"{stem}.png")
        
    img_vis = Image.open(vis_path).convert('RGB')
    img_ir = Image.open(ir_path).convert('RGB')
    
    W_PANEL = 800
    H_PANEL = 450
    img_vis_r = img_vis.resize((W_PANEL, H_PANEL), Image.Resampling.LANCZOS)
    img_ir_r = img_ir.resize((W_PANEL, H_PANEL), Image.Resampling.LANCZOS)
    
    font_title = ImageFont.truetype(FONT_PATH, 22)
    font_sub = ImageFont.truetype(FONT_PATH, 15)
    font_badge = ImageFont.truetype(FONT_PATH, 13)
    font_box = ImageFont.truetype(FONT_PATH, 13)
    
    MARGIN = 16
    PAD = 14
    CANVAS_W = MARGIN * 2 + W_PANEL * 3 + PAD * 2
    HEADER_H = 100
    TITLE_BAR_H = 36
    CANVAS_H = HEADER_H + TITLE_BAR_H + H_PANEL + MARGIN * 2
    
    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), C_BG)
    draw = ImageDraw.Draw(canvas)
    
    draw.rectangle([0, 0, CANVAS_W, HEADER_H], fill=C_BANNER)
    draw.line([(0, HEADER_H), (CANVAS_W, HEADER_H)], fill=(45, 50, 60), width=1)
    
    num_conflicts = sum(1 for v in matched_f.values() if v[0] == 'conflict')
    num_exact = sum(1 for v in matched_f.values() if v[0] == 'exact')
    num_f_only = len(bf) - len(matched_f)
    num_g_only = len(bg) - len(matched_g)
    
    title_str = f"【图号 {stem}】 F5 (57.266) vs V24 (Native 6711) 对比面板"
    sub_str1 = f"框数对比: F5标杆 (57.266) 共 {len(bf)} 框  |  V24 Native 共 {len(bg)} 框  |  共识一致: {num_exact} 框"
    draw.text((MARGIN, 16), title_str, fill=C_TEXT, font=font_title)
    draw.text((MARGIN, 48), sub_str1, fill=C_MUTED, font=font_sub)
    
    diff_items = []
    if num_conflicts > 0:
        for i, v in matched_f.items():
            if v[0] == 'conflict':
                j, iou = v[1], v[2]
                diff_items.append(f"[类别冲突] F5:{CLASS_NAMES[bf[i][0]]}({bf[i][5]:.2f}) vs V24:{CLASS_NAMES[bg[j][0]]}({bg[j][5]:.2f}) IoU={iou:.2f}")
    if num_f_only > 0:
        f_only_names = [f"{CLASS_NAMES[bf[i][0]]}({bf[i][5]:.2f})" for i in range(len(bf)) if i not in matched_f]
        diff_items.append(f"[F5独有 {num_f_only}框]: " + ", ".join(f_only_names[:5]) + (f" 等+{len(f_only_names)-5}个" if len(f_only_names)>5 else ""))
    if num_g_only > 0:
        g_only_names = [f"{CLASS_NAMES[bg[j][0]]}({bg[j][5]:.2f})" for j in range(len(bg)) if j not in matched_g]
        diff_items.append(f"[V24独有 {num_g_only}框]: " + ", ".join(g_only_names[:5]) + (f" 等+{len(g_only_names)-5}个" if len(g_only_names)>5 else ""))
        
    diff_text = "  |  ".join(diff_items)
    draw.text((MARGIN, 72), diff_text, fill=(255, 200, 80), font=font_sub)
    
    # Col 1: Reference
    x_col1 = MARGIN
    y_panels = HEADER_H + TITLE_BAR_H
    draw.rectangle([x_col1, HEADER_H + 4, x_col1 + W_PANEL, HEADER_H + TITLE_BAR_H - 2], fill=C_CARD)
    draw.text((x_col1 + 10, HEADER_H + 8), "物理参考：可见光原图 (右上角为红外热成像)", fill=(120, 220, 255), font=font_sub)
    
    panel1 = img_vis_r.copy()
    w_thumb = 240
    h_thumb = int(w_thumb * H_PANEL / W_PANEL)
    ir_thumb = img_ir.resize((w_thumb, h_thumb), Image.Resampling.LANCZOS)
    panel1.paste(ir_thumb, (W_PANEL - w_thumb - 10, 10))
    d1 = ImageDraw.Draw(panel1)
    d1.rectangle([W_PANEL - w_thumb - 10, 10, W_PANEL - 10, 10 + h_thumb], outline=(255, 255, 255), width=2)
    d1.rectangle([W_PANEL - w_thumb - 10, 10, W_PANEL - w_thumb + 85, 30], fill=(0, 0, 0))
    d1.text((W_PANEL - w_thumb - 6, 12), "IR 热成像", fill=(255, 215, 0), font=font_badge)
    canvas.paste(panel1, (x_col1, y_panels))
    
    def draw_boxes_on_img(base_img, boxes, match_dict, is_f5=True):
        panel = base_img.copy()
        draw_p = ImageDraw.Draw(panel)
        for idx, b in enumerate(boxes):
            cid, cx, cy, bw, bh, sc = b
            x1 = (cx - bw/2.0) * W_PANEL
            y1 = (cy - bh/2.0) * H_PANEL
            x2 = (cx + bw/2.0) * W_PANEL
            y2 = (cy + bh/2.0) * H_PANEL
            
            if idx in match_dict:
                m_type = match_dict[idx][0]
                if m_type == 'exact':
                    color = C_EXACT
                    tag = f"{CLASS_NAMES[cid]} {sc:.2f}"
                else: # conflict
                    color = C_CONF
                    tag = f"[冲突] {CLASS_NAMES[cid]} {sc:.2f}"
            else:
                color = C_F5_ONLY if is_f5 else C_V24_ONLY
                tag = f"[{'F5独有' if is_f5 else 'V24独有'}] {CLASS_NAMES[cid]} {sc:.2f}"
                
            draw_p.rectangle([x1, y1, x2, y2], outline=color, width=3)
            bbox = font_box.getbbox(tag)
            tw = bbox[2] - bbox[0] + 8
            th = bbox[3] - bbox[1] + 6
            by1 = max(0, y1 - th)
            draw_p.rectangle([x1, by1, x1 + tw, by1 + th], fill=color)
            draw_p.text((x1 + 4, by1 + 2), tag, fill=(0, 0, 0) if color in [C_CONF, C_F5_ONLY, C_EXACT] else (255, 255, 255), font=font_box)
        return panel
        
    # Col 2: F5
    x_col2 = x_col1 + W_PANEL + PAD
    draw.rectangle([x_col2, HEADER_H + 4, x_col2 + W_PANEL, HEADER_H + TITLE_BAR_H - 2], fill=C_CARD)
    draw.text((x_col2 + 10, HEADER_H + 8), f"F5 标杆 (57.266) - 预测 {len(bf)} 框  [绿色=共识, 橙色=类别冲突, 青色=F5独有]", fill=(0, 220, 255), font=font_sub)
    panel2 = draw_boxes_on_img(img_vis_r, bf, matched_f, is_f5=True)
    canvas.paste(panel2, (x_col2, y_panels))
    
    # Col 3: V24
    x_col3 = x_col2 + W_PANEL + PAD
    draw.rectangle([x_col3, HEADER_H + 4, x_col3 + W_PANEL, HEADER_H + TITLE_BAR_H - 2], fill=C_CARD)
    draw.text((x_col3 + 10, HEADER_H + 8), f"Sovereign V24 Native - 预测 {len(bg)} 框  [绿色=共识, 橙色=类别冲突, 洋红=V24独有]", fill=(255, 80, 120), font=font_sub)
    panel3 = draw_boxes_on_img(img_vis_r, bg, matched_g, is_f5=False)
    canvas.paste(panel3, (x_col3, y_panels))
    
    all_out_path = os.path.join(DIR_ALL, f"{stem}.png")
    canvas.save(all_out_path, quality=92)
    
    delta = num_f_only + num_g_only
    if num_conflicts > 0:
        conf_details = []
        for i, v in matched_f.items():
            if v[0] == 'conflict':
                j = v[1]
                conf_details.append(f"{CLASS_NAMES[bf[i][0]]}_vs_{CLASS_NAMES[bg[j][0]]}")
        conf_tag = "_".join(conf_details[:2])
        target_path = os.path.join(DIR_CONFLICTS, f"{stem}_conflict_{conf_tag}.png")
        shutil.copyfile(all_out_path, target_path)
    elif delta >= 5:
        target_path = os.path.join(DIR_LARGE, f"{stem}_diff_{delta}boxes.png")
        shutil.copyfile(all_out_path, target_path)
    elif delta >= 2:
        target_path = os.path.join(DIR_MEDIUM, f"{stem}_diff_{delta}boxes.png")
        shutil.copyfile(all_out_path, target_path)
    elif delta == 1:
        target_path = os.path.join(DIR_SINGLE, f"{stem}_diff_1box.png")
        shutil.copyfile(all_out_path, target_path)
        
    return stem


def generate_html_viewer(report_data):
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIC2026 - F5 (57.266) vs V24 Native 差异样本审阅器</title>
    <style>
        :root {
            --bg-dark: #121418;
            --bg-card: #1c1f26;
            --border: #2d3340;
            --text-main: #f0f3f8;
            --text-dim: #9aa5b8;
            --primary: #3b82f6;
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #10b981;
            --cyan: #06b6d4;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Noto Sans CJK SC", sans-serif; }
        body { background: var(--bg-dark); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }
        header { background: #0b0d10; border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }
        .logo-title { display: flex; align-items: center; gap: 12px; }
        .title { font-size: 18px; font-weight: 700; color: #fff; }
        .stats-badge { background: #232732; border: 1px solid var(--border); padding: 4px 10px; border-radius: 6px; font-size: 13px; color: var(--text-dim); }
        .controls { display: flex; gap: 8px; align-items: center; }
        button { background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.15s; }
        button:hover { background: #282d38; border-color: var(--primary); }
        button.active { background: var(--primary); border-color: var(--primary); color: #fff; }
        .filter-group { display: flex; gap: 6px; }
        .nav-btn { background: #2563eb; color: #fff; border: none; font-weight: 600; padding: 6px 16px; }
        .nav-btn:hover { background: #1d4ed8; }
        main { flex: 1; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        .info-panel { width: 100%; max-width: 1920px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 20px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
        .img-meta { font-size: 15px; font-weight: 600; }
        .img-diff-summary { font-size: 14px; color: var(--warning); font-weight: 500; }
        .image-container { width: 100%; max-width: 1920px; display: flex; justify-content: center; background: #08090b; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
        .image-container img { width: 100%; height: auto; display: block; object-fit: contain; }
        .keyboard-hint { margin-top: 12px; font-size: 12px; color: var(--text-dim); text-align: center; }
        kbd { background: #232732; border: 1px solid #3c4456; border-radius: 4px; padding: 2px 6px; font-size: 11px; }
    </style>
</head>
<body>
    <header>
        <div class="logo-title">
            <div class="title">AIC2026: F5 (57.266) vs V24 (Native 6711) 审阅器</div>
            <div class="stats-badge" id="statsBadge">加载中...</div>
        </div>
        <div class="controls">
            <div class="filter-group">
                <button class="active" onclick="setFilter('all')">全部差异 (<span id="cntAll">0</span>)</button>
                <button onclick="setFilter('conflict')" style="color:#fb923c;">⚠️ 类别冲突 (<span id="cntConf">0</span>)</button>
                <button onclick="setFilter('large')" style="color:#f87171;">重大差异(Δ≥5) (<span id="cntLarge">0</span>)</button>
                <button onclick="setFilter('medium')">中度差异(Δ2~4) (<span id="cntMed">0</span>)</button>
                <button onclick="setFilter('single')">单一微差(Δ1) (<span id="cntSingle">0</span>)</button>
            </div>
            <div style="width: 1px; height: 24px; background: var(--border); margin: 0 4px;"></div>
            <button class="nav-btn" onclick="prevImg()">◀ 上一张 (←)</button>
            <span id="curIndex" style="font-size: 13px; font-weight: 600; min-width: 60px; text-align: center;">1 / 0</span>
            <button class="nav-btn" onclick="nextImg()">下一张 (→) ▶</button>
        </div>
    </header>

    <main>
        <div class="info-panel">
            <div class="img-meta" id="imgMeta">载入中...</div>
            <div class="img-diff-summary" id="imgDiffSummary"></div>
        </div>
        <div class="image-container">
            <img id="mainImg" src="" alt="对比面板" />
        </div>
        <div class="keyboard-hint">
            快捷键提示：键盘左右键 <kbd>←</kbd> <kbd>→</kbd> 快速切换图片，<kbd>1</kbd> 查看全量，<kbd>2</kbd> 类别冲突(仅22张)，<kbd>3</kbd> 重大差异
        </div>
    </main>

    <script>
        const report = DATA_PLACEHOLDER;
        let currentFilter = 'all';
        let filteredList = [];
        let currentIndex = 0;

        function init() {
            document.getElementById('cntAll').innerText = report.all.length;
            document.getElementById('cntConf').innerText = report.conflict.length;
            document.getElementById('cntLarge').innerText = report.large.length;
            document.getElementById('cntMed').innerText = report.medium.length;
            document.getElementById('cntSingle').innerText = report.single.length;
            setFilter('conflict');
        }

        function setFilter(type) {
            currentFilter = type;
            document.querySelectorAll('.filter-group button').forEach(b => b.classList.remove('active'));
            if(type === 'all') document.querySelectorAll('.filter-group button')[0].classList.add('active');
            if(type === 'conflict') document.querySelectorAll('.filter-group button')[1].classList.add('active');
            if(type === 'large') document.querySelectorAll('.filter-group button')[2].classList.add('active');
            if(type === 'medium') document.querySelectorAll('.filter-group button')[3].classList.add('active');
            if(type === 'single') document.querySelectorAll('.filter-group button')[4].classList.add('active');

            filteredList = report[type];
            currentIndex = 0;
            renderCurrent();
        }

        function renderCurrent() {
            if(filteredList.length === 0) return;
            const item = filteredList[currentIndex];
            document.getElementById('curIndex').innerText = `${currentIndex + 1} / ${filteredList.length}`;
            document.getElementById('statsBadge').innerText = `当前类别: ${currentFilter} (共 ${filteredList.length} 张)`;
            document.getElementById('imgMeta').innerText = `【图号 ${item.stem}】 F5预测: ${item.f5_cnt} 框 | V24预测: ${item.v24_cnt} 框 | 共识一致: ${item.exact_cnt} 框`;
            document.getElementById('imgDiffSummary').innerText = item.summary;
            document.getElementById('mainImg').src = 'all_discrepancies/' + item.stem + '.png';
        }

        function nextImg() {
            if(currentIndex < filteredList.length - 1) {
                currentIndex++;
                renderCurrent();
            }
        }

        function prevImg() {
            if(currentIndex > 0) {
                currentIndex--;
                renderCurrent();
            }
        }

        window.addEventListener('keydown', (e) => {
            if(e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') nextImg();
            if(e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') prevImg();
            if(e.key === '1') setFilter('all');
            if(e.key === '2') setFilter('conflict');
            if(e.key === '3') setFilter('large');
            if(e.key === '4') setFilter('medium');
            if(e.key === '5') setFilter('single');
        });

        init();
    </script>
</body>
</html>
"""
    return html_content


def main():
    print("==================================================================")
    print(" AIC2026: F5 (57.266) vs V24 (6,711 Native) Discrepancy Generator")
    print("==================================================================")
    
    for p in [DIR_CONFLICTS, DIR_LARGE, DIR_MEDIUM, DIR_SINGLE, DIR_ALL]:
        os.makedirs(p, exist_ok=True)
        for f in os.listdir(p):
            fp = os.path.join(p, f)
            if os.path.isfile(fp): os.remove(fp)
            
    print("Loading predictions...")
    f5_data = load_f5_predictions(F5_DIR)
    v24_data = load_v24_predictions(V24_ZIP)
    
    all_stems = sorted(f5_data.keys())
    print(f"Total test images: {len(all_stems)}")
    
    tasks = []
    identical_count = 0
    report_dict = {
        'all': [],
        'conflict': [],
        'large': [],
        'medium': [],
        'single': []
    }
    
    for stem in all_stems:
        bf = f5_data.get(stem, [])
        bg = v24_data.get(stem, [])
        matched_f, matched_g = match_boxes(bf, bg)
        
        num_conflicts = sum(1 for v in matched_f.values() if v[0] == 'conflict')
        num_exact = sum(1 for v in matched_f.values() if v[0] == 'exact')
        num_f_only = len(bf) - len(matched_f)
        num_g_only = len(bg) - len(matched_g)
        delta = num_f_only + num_g_only
        
        if num_conflicts == 0 and delta == 0:
            identical_count += 1
            continue
            
        tasks.append((stem, bf, bg, matched_f, matched_g))
        
        summary_parts = []
        if num_conflicts > 0:
            for i, v in matched_f.items():
                if v[0] == 'conflict':
                    j, iou = v[1], v[2]
                    summary_parts.append(f"⚠️ 冲突: F5:{CLASS_NAMES[bf[i][0]]} ⚔️ V24:{CLASS_NAMES[bg[j][0]]} (IoU={iou:.2f})")
        if num_f_only > 0:
            summary_parts.append(f"🟦 F5独有 {num_f_only}框")
        if num_g_only > 0:
            summary_parts.append(f"🟥 V24独有 {num_g_only}框")
        summary_str = " | ".join(summary_parts)
        
        item_meta = {
            'stem': stem,
            'f5_cnt': len(bf),
            'v24_cnt': len(bg),
            'exact_cnt': num_exact,
            'conf_cnt': num_conflicts,
            'delta': delta,
            'summary': summary_str
        }
        report_dict['all'].append(item_meta)
        if num_conflicts > 0:
            report_dict['conflict'].append(item_meta)
        elif delta >= 5:
            report_dict['large'].append(item_meta)
        elif delta >= 2:
            report_dict['medium'].append(item_meta)
        elif delta == 1:
            report_dict['single'].append(item_meta)
            
    print(f"100% Identical images (No diff): {identical_count}")
    print(f"Total images with differences to render: {len(tasks)}")
    print(f"  - 01_category_conflicts: {len(report_dict['conflict'])} images")
    print(f"  - 02_large_diff_count  : {len(report_dict['large'])} images")
    print(f"  - 03_medium_diff_count : {len(report_dict['medium'])} images")
    print(f"  - 04_single_box_diff   : {len(report_dict['single'])} images")
    
    print(f"\nRendering {len(tasks)} inspection panels using 10 CPU workers...")
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(render_image_panel, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            f.result()
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f"  Progress: {completed}/{len(tasks)} panels rendered ({completed*100.0/len(tasks):.1f}%)")
                
    report_json_path = os.path.join(OUTPUT_ROOT, 'discrepancy_report.json')
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    print(f"\nSaved structured metadata to: {report_json_path}")
    
    html_template = generate_html_viewer(report_dict)
    final_html = html_template.replace('DATA_PLACEHOLDER', json.dumps(report_dict, ensure_ascii=False))
    html_path = os.path.join(OUTPUT_ROOT, 'index.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"Saved interactive gallery viewer to: {html_path}")
    print("\nGeneration finished successfully!")


if __name__ == '__main__':
    main()
