#!/usr/bin/env python3
"""
Generate comprehensive visual discrepancy and comparison suite specifically for category 10: uav (无人机).
F5 (57.266) vs V23 (53.00分).
Supports configurable confidence threshold (e.g. --thr 0.45, --thr 0.60).
Rule: ONLY annotate category 10 (uav)! No other categories.
Includes:
- Section 1: Full-scene comparison (Col 1 Raw, Col 2 F5, Col 3 V23)
- Section 2: High-resolution zoom crops centered directly on UAV predictions
- Interactive web gallery index.html with filtering & keyboard navigation
"""

import os
import sys
import json
import zipfile
import shutil
import argparse
import concurrent.futures
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# Colors
C_EXACT    = (46, 204, 113)    # Emerald Green (Consensus)
C_F5_ONLY  = (0, 195, 255)     # Electric Cyan (F5 Only)
C_V23_ONLY = (255, 45, 85)     # Crimson / Magenta (V23 Only)
C_BG       = (16, 18, 22)      # Dark Background
C_BANNER   = (10, 12, 16)      # Header
C_CARD     = (26, 30, 36)
C_TEXT     = (240, 245, 250)
C_MUTED    = (160, 168, 180)

FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
font_title = ImageFont.truetype(FONT_PATH, 24)
font_sub = ImageFont.truetype(FONT_PATH, 16)
font_sec = ImageFont.truetype(FONT_PATH, 15)
font_box = ImageFont.truetype(FONT_PATH, 12)
font_badge = ImageFont.truetype(FONT_PATH, 11)

F5_DIR = '/home/constant/Downloads/AIC2026_phase2_predictions_F5seed20261002_RARE45_COMMON50_compact7212_20261005'
V23_ZIP = '/home/constant/Downloads/submission_v23_flip_tta_thr45.zip'
VIS_DIR = '/home/constant/CodeBuddy/Dev/AIC/dataset/test/AIC2026_PHASE_2_1000/visible'
IR_DIR  = '/home/constant/CodeBuddy/Dev/AIC/dataset/test/AIC2026_PHASE_2_1000/infrared'


def get_image_path(dir_p, stem):
    p1 = os.path.join(dir_p, f'{stem}.png')
    if os.path.exists(p1): return p1
    p2 = os.path.join(dir_p, f'{stem}.jpg')
    if os.path.exists(p2): return p2
    return None


def compute_iou(b1, b2):
    x1_1, x2_1 = b1[0] - b1[2]/2, b1[0] + b1[2]/2
    y1_1, y2_1 = b1[1] - b1[3]/2, b1[1] + b1[3]/2
    x1_2, x2_2 = b2[0] - b2[2]/2, b2[0] + b2[2]/2
    y1_2, y2_2 = b2[1] - b2[3]/2, b2[1] + b2[3]/2
    ix = max(0.0, min(x2_1, x2_2) - max(x1_1, x1_2))
    iy = max(0.0, min(y2_1, y2_2) - max(y1_1, y1_2))
    inter = ix * iy
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


def load_uavs(thr=0.45):
    f5_uavs = defaultdict(list)
    for f in sorted(os.listdir(F5_DIR)):
        if not f.endswith('.txt'): continue
        stem = f.rsplit('.', 1)[0]
        with open(os.path.join(F5_DIR, f)) as fp:
            for l in fp:
                p = l.split()
                if int(p[0]) == 10 and float(p[5]) >= thr:
                    f5_uavs[stem].append([float(x) for x in p[1:]])
        f5_uavs[stem].sort(key=lambda x: x[4], reverse=True)

    v23_uavs = defaultdict(list)
    with zipfile.ZipFile(V23_ZIP) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith('.txt'): continue
            stem = name.rsplit('.', 1)[0]
            for l in zf.read(name).decode().split('\n'):
                if not l.strip(): continue
                p = l.split()
                if int(p[0]) == 10 and float(p[5]) >= thr:
                    v23_uavs[stem].append([float(x) for x in p[1:]])
            v23_uavs[stem].sort(key=lambda x: x[4], reverse=True)

    f5_real = {k: v for k, v in f5_uavs.items() if len(v) > 0}
    v23_real = {k: v for k, v in v23_uavs.items() if len(v) > 0}
    all_stems = sorted(set(f5_real.keys()) | set(v23_real.keys()))
    return f5_real, v23_real, all_stems


def draw_uavs_on_crop(base_crop, boxes, match_set, is_f5, crop_rect):
    rx1, ry1, rx2, ry2 = crop_rect
    cw = rx2 - rx1
    ch = ry2 - ry1
    crop_w, crop_h = base_crop.size
    
    panel = base_crop.copy()
    draw_p = ImageDraw.Draw(panel)
    
    for idx, b in enumerate(boxes):
        cx, cy, bw, bh, sc = b
        bx1 = cx - bw/2.0
        by1 = cy - bh/2.0
        bx2 = cx + bw/2.0
        by2 = cy + bh/2.0
        
        if bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2:
            continue
            
        px1 = (bx1 - rx1) / cw * crop_w
        py1 = (by1 - ry1) / ch * crop_h
        px2 = (bx2 - rx1) / cw * crop_w
        py2 = (by2 - ry1) / ch * crop_h
        
        is_consensus = (idx in match_set)
        if is_consensus:
            color = C_EXACT
            line_w = 2
            tag = f"uav {sc:.2f}"
        else:
            color = C_F5_ONLY if is_f5 else C_V23_ONLY
            line_w = 3
            tag = f"★uav {sc:.2f}"
            
        draw_p.rectangle([px1, py1, px2, py2], outline=color, width=line_w)
        
        bbox = font_box.getbbox(tag)
        tw = bbox[2] - bbox[0] + 6
        th = bbox[3] - bbox[1] + 4
        by_label = py1 - th if py1 - th > 0 else py2 + 1
        
        draw_p.rectangle([px1, by_label, px1 + tw, by_label + th], fill=(15, 17, 22))
        draw_p.rectangle([px1, by_label, px1 + tw, by_label + th], outline=color, width=1)
        draw_p.text((px1 + 3, by_label + 1), tag, fill=color, font=font_box)
        
    return panel


def generate_panel_for_stem(stem, f5_list, v23_list, output_root, thr):
    dir_consensus  = os.path.join(output_root, '01_consensus')
    dir_diff_count = os.path.join(output_root, '02_diff_count')
    dir_f5_only    = os.path.join(output_root, '03_f5_only')
    dir_v23_only   = os.path.join(output_root, '04_v23_only')
    dir_all        = os.path.join(output_root, 'all_uav_panels')

    p_vis = get_image_path(VIS_DIR, stem)
    p_ir  = get_image_path(IR_DIR, stem)
    if not p_vis or not p_ir:
        return None
        
    img_vis = Image.open(p_vis).convert('RGB')
    img_ir  = Image.open(p_ir).convert('RGB')
    W_orig, H_orig = img_vis.size
    
    # Match F5 vs V23 UAVs
    p_matches = []
    for i, bf in enumerate(f5_list):
        for j, bv in enumerate(v23_list):
            iou = compute_iou(bf, bv)
            if iou >= 0.20:
                p_matches.append((iou, i, j))
    p_matches.sort(key=lambda x: x[0], reverse=True)
    
    mf_set = set()
    mv_set = set()
    for iou, i, j in p_matches:
        if i not in mf_set and j not in mv_set:
            mf_set.add(i)
            mv_set.add(j)
            
    n_f5 = len(f5_list)
    n_v23 = len(v23_list)
    n_exact = len(mf_set)
    
    if n_f5 > 0 and n_v23 == 0:
        cat_type = 'f5_only'
        cat_label = '仅F5报无人机 (F5 Only)'
        target_dir = dir_f5_only
    elif n_f5 == 0 and n_v23 > 0:
        cat_type = 'v23_only'
        cat_label = '仅V23报无人机 (V23 Only)'
        target_dir = dir_v23_only
    elif n_f5 == n_v23 and n_exact == n_f5:
        cat_type = 'consensus'
        cat_label = '完全共识 (Exact Consensus)'
        target_dir = dir_consensus
    else:
        cat_type = 'diff_count'
        cat_label = '框数/定位差异 (Count Mismatch)'
        target_dir = dir_diff_count
        
    all_boxes = f5_list + v23_list
    zoom_windows = []
    if all_boxes:
        min_x = min(b[0] - b[2]/2 for b in all_boxes)
        max_x = max(b[0] + b[2]/2 for b in all_boxes)
        min_y = min(b[1] - b[3]/2 for b in all_boxes)
        max_y = max(b[1] + b[3]/2 for b in all_boxes)
        
        span_x = max_x - min_x
        span_y = max_y - min_y
        
        pad_x = max(0.08, span_x * 0.3)
        pad_y = max(0.08, span_y * 0.3)
        zx1 = max(0.0, min_x - pad_x)
        zy1 = max(0.0, min_y - pad_y)
        zx2 = min(1.0, max_x + pad_x)
        zy2 = min(1.0, max_y + pad_y)
        zoom_windows.append(('【无人机超清特写镜头】', (zx1, zy1, zx2, zy2)))
    
    CANVAS_W = 2400
    HEADER_H = 105
    SEC_TITLE_H = 38
    FULL_H = 460
    ZOOM_H = 420 if zoom_windows else 0
    MARGIN = 16
    PAD = 14
    
    total_h = HEADER_H + MARGIN + (SEC_TITLE_H + FULL_H + MARGIN)
    if zoom_windows:
        total_h += (SEC_TITLE_H + ZOOM_H + MARGIN)
        
    canvas = Image.new('RGB', (CANVAS_W, total_h), C_BG)
    draw = ImageDraw.Draw(canvas)
    
    # Header
    draw.rectangle([0, 0, CANVAS_W, HEADER_H], fill=C_BANNER)
    draw.line([(0, HEADER_H), (CANVAS_W, HEADER_H)], fill=(45, 50, 60), width=1)
    draw.text((MARGIN, 14), f"【图号 {stem}】 类别 10: uav (无人机) 专项审查对比面板 (阈值 ≥ {thr:.2f})", fill=C_TEXT, font=font_title)
    
    status_desc = f"类型判定: {cat_label}  |  F5机数: {n_f5} 框  |  V23机数: {n_v23} 框  |  达成共识: {n_exact} 框  |  F5独有: {n_f5 - n_exact} 框  |  V23独有: {n_v23 - n_exact} 框"
    draw.text((MARGIN, 50), status_desc, fill=C_MUTED, font=font_sub)
    draw.text((MARGIN, 76), f"💡 专项审查规则: 过滤阈值 ≥ {thr:.2f}；仅标注类别 10 (uav)；第一列纯净原图，第二/三列极细线框+浮动角标，零遮挡！", fill=(100, 255, 160), font=font_sub)
    
    W_COL = (CANVAS_W - MARGIN*2 - PAD*2) // 3
    curr_y = HEADER_H + MARGIN
    
    # Section 1: Full scene
    draw.rectangle([MARGIN, curr_y, CANVAS_W - MARGIN, curr_y + SEC_TITLE_H], fill=C_CARD)
    draw.text((MARGIN + 12, curr_y + 8), f"【全景全局视角】 原图尺寸 {W_orig}x{H_orig}  —  左: 纯净物理原图(+IR热像) | 中: F5标杆预测 | 右: V23预测", fill=(255, 215, 100), font=font_sec)
    curr_y += SEC_TITLE_H + 4
    
    full_vis_r = img_vis.resize((W_COL, FULL_H), Image.Resampling.LANCZOS)
    full_ir_r  = img_ir.resize((W_COL, FULL_H), Image.Resampling.LANCZOS)
    
    x1 = MARGIN
    x2 = x1 + W_COL + PAD
    x3 = x2 + W_COL + PAD
    
    # Col 1: Raw
    c1 = full_vis_r.copy()
    w_th = 150
    h_th = int(w_th * FULL_H / W_COL)
    ir_th = full_ir_r.resize((w_th, h_th), Image.Resampling.LANCZOS)
    c1.paste(ir_th, (W_COL - w_th - 6, 6))
    d_c1 = ImageDraw.Draw(c1)
    d_c1.rectangle([W_COL - w_th - 6, 6, W_COL - 6, 6 + h_th], outline=(255, 255, 255), width=2)
    d_c1.rectangle([W_COL - w_th - 6, 6, W_COL - w_th + 65, 22], fill=(0, 0, 0))
    d_c1.text((W_COL - w_th - 3, 7), "IR热成像", fill=(255, 215, 0), font=font_badge)
    d_c1.rectangle([6, 6, 180, 24], fill=(0, 0, 0, 180))
    d_c1.text((10, 8), "物理原图 (只审无人机·零遮挡)", fill=(100, 220, 255), font=font_badge)
    canvas.paste(c1, (x1, curr_y))
    
    # Col 2: F5
    c2 = draw_uavs_on_crop(full_vis_r, f5_list, mf_set, is_f5=True, crop_rect=(0, 0, 1, 1))
    d_c2 = ImageDraw.Draw(c2)
    d_c2.rectangle([6, 6, 180, 24], fill=(0, 0, 0, 180))
    d_c2.text((10, 8), f"F5 预测 ({n_f5}机, 青色=F5独有)", fill=(0, 220, 255), font=font_badge)
    canvas.paste(c2, (x2, curr_y))
    
    # Col 3: V23
    c3 = draw_uavs_on_crop(full_vis_r, v23_list, mv_set, is_f5=False, crop_rect=(0, 0, 1, 1))
    d_c3 = ImageDraw.Draw(c3)
    d_c3.rectangle([6, 6, 180, 24], fill=(0, 0, 0, 180))
    d_c3.text((10, 8), f"V23 预测 ({n_v23}机, 洋红=V23独有)", fill=(255, 80, 120), font=font_badge)
    canvas.paste(c3, (x3, curr_y))
    
    curr_y += FULL_H + MARGIN
    
    # Section 2: Zoom Crop
    if zoom_windows:
        z_title, z_rect = zoom_windows[0]
        draw.rectangle([MARGIN, curr_y, CANVAS_W - MARGIN, curr_y + SEC_TITLE_H], fill=C_CARD)
        draw.text((MARGIN + 12, curr_y + 8), f"{z_title} (区域 x: {z_rect[0]:.2f}~{z_rect[2]:.2f}, y: {z_rect[1]:.2f}~{z_rect[3]:.2f})  —  高倍率无损放大验真", fill=(255, 215, 100), font=font_sec)
        curr_y += SEC_TITLE_H + 4
        
        px1, py1 = int(z_rect[0]*W_orig), int(z_rect[1]*H_orig)
        px2, py2 = int(z_rect[2]*W_orig), int(z_rect[3]*H_orig)
        crop_vis = img_vis.crop((px1, py1, px2, py2))
        crop_ir  = img_ir.crop((px1, py1, px2, py2))
        
        crop_vis_r = crop_vis.resize((W_COL, ZOOM_H), Image.Resampling.LANCZOS)
        crop_ir_r  = crop_ir.resize((W_COL, ZOOM_H), Image.Resampling.LANCZOS)
        
        # Col 1: Raw Zoom
        cz1 = crop_vis_r.copy()
        w_th = 150
        h_th = int(w_th * ZOOM_H / W_COL)
        ir_z_th = crop_ir_r.resize((w_th, h_th), Image.Resampling.LANCZOS)
        cz1.paste(ir_z_th, (W_COL - w_th - 6, 6))
        d_cz1 = ImageDraw.Draw(cz1)
        d_cz1.rectangle([W_COL - w_th - 6, 6, W_COL - 6, 6 + h_th], outline=(255, 255, 255), width=2)
        d_cz1.rectangle([W_COL - w_th - 6, 6, W_COL - w_th + 65, 22], fill=(0, 0, 0))
        d_cz1.text((W_COL - w_th - 3, 7), "IR热成像特写", fill=(255, 215, 0), font=font_badge)
        d_cz1.rectangle([6, 6, 180, 24], fill=(0, 0, 0, 180))
        d_cz1.text((10, 8), "物理原图特写 (零遮挡)", fill=(100, 220, 255), font=font_badge)
        canvas.paste(cz1, (x1, curr_y))
        
        # Col 2: F5 Zoom
        cz2 = draw_uavs_on_crop(crop_vis_r, f5_list, mf_set, is_f5=True, crop_rect=z_rect)
        d_cz2 = ImageDraw.Draw(cz2)
        d_cz2.rectangle([6, 6, 180, 24], fill=(0, 0, 0, 180))
        d_cz2.text((10, 8), "F5 特写预测", fill=(0, 220, 255), font=font_badge)
        canvas.paste(cz2, (x2, curr_y))
        
        # Col 3: V23 Zoom
        cz3 = draw_uavs_on_crop(crop_vis_r, v23_list, mv_set, is_f5=False, crop_rect=z_rect)
        d_cz3 = ImageDraw.Draw(cz3)
        d_cz3.rectangle([6, 6, 180, 24], fill=(0, 0, 0, 180))
        d_cz3.text((10, 8), "V23 特写预测", fill=(255, 80, 120), font=font_badge)
        canvas.paste(cz3, (x3, curr_y))
        
    fn = f"uav_panel_{stem}.png"
    p_all = os.path.join(dir_all, fn)
    p_cat = os.path.join(target_dir, fn)
    
    canvas.save(p_all, quality=92)
    shutil.copyfile(p_all, p_cat)
    
    meta_item = {
        'stem': stem,
        'cat_type': cat_type,
        'cat_label': cat_label,
        'f5_cnt': n_f5,
        'v23_cnt': n_v23,
        'exact_cnt': n_exact,
        'f5_only_cnt': n_f5 - n_exact,
        'v23_only_cnt': n_v23 - n_exact,
        'filename': fn,
        'rel_path': f"all_uav_panels/{fn}",
        'cat_rel_path': f"{os.path.basename(target_dir)}/{fn}",
        'f5_boxes': [{'cx': b[0], 'cy': b[1], 'w': b[2], 'h': b[3], 'score': b[4]} for b in f5_list],
        'v23_boxes': [{'cx': b[0], 'cy': b[1], 'w': b[2], 'h': b[3], 'score': b[4]} for b in v23_list]
    }
    return meta_item


def build_html_viewer(metadata_list, output_root, thr):
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIC2026 - 类别 10: uav (无人机) 专项审查大盘 (F5 vs V23)</title>
    <style>
        :root {{
            --bg-dark: #101216;
            --bg-card: #1a1e24;
            --border: #2a303c;
            --text-main: #f0f4f8;
            --text-dim: #9aa4b2;
            --primary: #3b82f6;
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #10b981;
            --cyan: #06b6d4;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Noto Sans CJK SC", sans-serif; }}
        body {{ background: var(--bg-dark); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }}
        header {{ background: #0a0c10; border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }}
        .logo-title {{ display: flex; align-items: center; gap: 12px; }}
        .title {{ font-size: 18px; font-weight: 700; color: #fff; }}
        .stats-badge {{ background: #232732; border: 1px solid var(--border); padding: 4px 10px; border-radius: 6px; font-size: 13px; color: var(--text-dim); }}
        .controls {{ display: flex; gap: 8px; align-items: center; }}
        button {{ background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.15s; }}
        button:hover {{ background: #282d38; border-color: var(--primary); }}
        button.active {{ background: var(--primary); border-color: var(--primary); color: #fff; }}
        .filter-group {{ display: flex; gap: 6px; }}
        .nav-btn {{ background: #2563eb; color: #fff; border: none; font-weight: 600; padding: 6px 16px; }}
        .nav-btn:hover {{ background: #1d4ed8; }}
        main {{ flex: 1; padding: 20px; display: flex; flex-direction: column; align-items: center; }}
        .info-panel {{ width: 100%; max-width: 2400px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 20px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
        .img-meta {{ font-size: 15px; font-weight: 600; }}
        .img-diff-summary {{ font-size: 14px; color: var(--warning); font-weight: 500; }}
        .search-select {{ background: #121418; color: #fff; border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; font-size: 13px; }}
        .image-container {{ width: 100%; max-width: 2400px; display: flex; justify-content: center; background: #08090b; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
        .image-container img {{ width: 100%; height: auto; display: block; object-fit: contain; }}
        .keyboard-hint {{ margin-top: 12px; font-size: 12px; color: var(--text-dim); text-align: center; }}
        kbd {{ background: #232732; border: 1px solid #3c4456; border-radius: 4px; padding: 2px 6px; font-size: 11px; }}
    </style>
</head>
<body>
    <header>
        <div class="logo-title">
            <span class="title">🛸 AIC2026 - 类别 10: uav (无人机) 专项审查大盘 (阈值 ≥ {thr:.2f})</span>
            <span class="stats-badge" id="statsBadge">加载中...</span>
        </div>
        <div class="controls">
            <div class="filter-group">
                <button class="active" onclick="setFilter('all')">全部无人机图 ({len(metadata_list)})</button>
                <button onclick="setFilter('consensus')">完全共识</button>
                <button onclick="setFilter('v23_only')">仅V23有无人机</button>
                <button onclick="setFilter('f5_only')">仅F5有无人机</button>
                <button onclick="setFilter('diff_count')">框数/定位差异</button>
            </div>
            <select class="search-select" id="imageSelect" onchange="jumpToImage(this.value)"></select>
            <button class="nav-btn" onclick="prevImage()">上一张 (←)</button>
            <button class="nav-btn" onclick="nextImage()">下一张 (→)</button>
        </div>
    </header>

    <main>
        <div class="info-panel">
            <div class="img-meta" id="imgMeta">正在加载...</div>
            <div class="img-diff-summary" id="imgDiffSummary"></div>
            <div style="font-size: 13px; color: var(--text-dim);" id="imgIndex"></div>
        </div>

        <div class="image-container">
            <img id="mainImage" src="" alt="无人机专项对比面板加载中...">
        </div>

        <div class="keyboard-hint">
            快捷键提示：使用键盘 <kbd>←</kbd> <kbd>→</kbd> 键快速切换上一张/下一张图片
        </div>
    </main>

    <script>
        const metadata = __METADATA_PLACEHOLDER__;
        let currentFilter = 'all';
        let filteredList = [...metadata];
        let currentIndex = 0;

        function updateFilterCounts() {{
            const counts = {{ all: metadata.length, consensus: 0, v23_only: 0, f5_only: 0, diff_count: 0 }};
            metadata.forEach(item => {{
                if (counts[item.cat_type] !== undefined) counts[item.cat_type]++;
            }});
            document.getElementById('statsBadge').textContent = 
                `阈值 ≥ {thr:.2f} | 共 ${{metadata.length}} 张图 (共识: ${{counts.consensus}} | V23独有: ${{counts.v23_only}} | F5独有: ${{counts.f5_only}} | 差异: ${{counts.diff_count}})`;
        }}

        function setFilter(filter) {{
            currentFilter = filter;
            document.querySelectorAll('.filter-group button').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');

            if (filter === 'all') {{
                filteredList = [...metadata];
            }} else {{
                filteredList = metadata.filter(item => item.cat_type === filter);
            }}
            currentIndex = 0;
            updateSelectOptions();
            renderCurrent();
        }}

        function updateSelectOptions() {{
            const sel = document.getElementById('imageSelect');
            sel.innerHTML = '';
            filteredList.forEach((item, idx) => {{
                const opt = document.createElement('option');
                opt.value = idx;
                opt.textContent = `${{idx + 1}}. ${{item.stem}} (F5:${{item.f5_cnt}} | V23:${{item.v23_cnt}})`;
                sel.appendChild(opt);
            }});
        }}

        function jumpToImage(idx) {{
            currentIndex = parseInt(idx);
            renderCurrent();
        }}

        function renderCurrent() {{
            if (filteredList.length === 0) return;
            const item = filteredList[currentIndex];
            document.getElementById('mainImage').src = item.rel_path;
            document.getElementById('imageSelect').value = currentIndex;
            
            let badgeColor = '#10b981';
            if (item.cat_type === 'v23_only') badgeColor = '#ef4444';
            if (item.cat_type === 'f5_only') badgeColor = '#06b6d4';
            if (item.cat_type === 'diff_count') badgeColor = '#f59e0b';

            document.getElementById('imgMeta').innerHTML = 
                `【图号 ${{item.stem}}】 <span style="background:${{badgeColor}}22; color:${{badgeColor}}; border:1px solid ${{badgeColor}}; padding:2px 8px; border-radius:4px; margin-left:8px;">${{item.cat_label}}</span>`;
                
            let diffText = `F5: ${{item.f5_cnt}} 机 | V23: ${{item.v23_cnt}} 机 | 共识: ${{item.exact_cnt}} 机`;
            if (item.f5_only_cnt > 0) diffText += ` | <span style="color:var(--cyan)">F5独有: ${{item.f5_only_cnt}}</span>`;
            if (item.v23_only_cnt > 0) diffText += ` | <span style="color:var(--danger)">V23独有: ${{item.v23_only_cnt}}</span>`;
            document.getElementById('imgDiffSummary').innerHTML = diffText;
            
            document.getElementById('imgIndex').textContent = `当前第 ${{currentIndex + 1}} / ${{filteredList.length}} 张 (过滤类别: ${{currentFilter}})`;
        }}

        function prevImage() {{
            if (filteredList.length === 0) return;
            currentIndex = (currentIndex - 1 + filteredList.length) % filteredList.length;
            renderCurrent();
        }}

        function nextImage() {{
            if (filteredList.length === 0) return;
            currentIndex = (currentIndex + 1) % filteredList.length;
            renderCurrent();
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') prevImage();
            else if (e.key === 'ArrowRight') nextImage();
        }});

        updateFilterCounts();
        updateSelectOptions();
        renderCurrent();
    </script>
</body>
</html>
"""
    html_content = html_content.replace('__METADATA_PLACEHOLDER__', json.dumps(metadata_list, ensure_ascii=False, indent=2))
    with open(os.path.join(output_root, 'index.html'), 'w', encoding='utf-8') as fp:
        fp.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Generate UAV (Cat 10) comparison gallery")
    parser.add_argument('--thr', type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument('--out-dir', type=str, default='outputs/analysis/uav_analysis_f5_vs_v23', help="Output root directory")
    parser.add_argument('--workers', type=int, default=16, help="Worker threads")
    args = parser.parse_args()

    thr = args.thr
    output_root = args.out_dir
    os.makedirs(output_root, exist_ok=True)
    for sub in ['01_consensus', '02_diff_count', '03_f5_only', '04_v23_only', 'all_uav_panels']:
        os.makedirs(os.path.join(output_root, sub), exist_ok=True)

    print(f"=== Loading Category 10: UAV Predictions (Threshold >= {thr:.2f}) ===")
    f5_uavs, v23_uavs, all_stems = load_uavs(thr=thr)
    print(f"Total images with UAV: {len(all_stems)}")
    print(f"  F5 images: {len(f5_uavs)} (total boxes: {sum(len(v) for v in f5_uavs.values())})")
    print(f"  V23 images: {len(v23_uavs)} (total boxes: {sum(len(v) for v in v23_uavs.values())})")

    results = []
    print(f"=== Generating UAV Panels with {args.workers} Threads ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(generate_panel_for_stem, stem, f5_uavs.get(stem, []), v23_uavs.get(stem, []), output_root, thr): stem
            for stem in all_stems
        }
        for future in concurrent.futures.as_completed(future_map):
            stem = future_map[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"Error processing {stem}: {e}")

    # Sort results
    # Priority: diff_count -> v23_only -> f5_only -> consensus
    prio_map = {'diff_count': 0, 'v23_only': 1, 'f5_only': 2, 'consensus': 3}
    results.sort(key=lambda x: (prio_map.get(x['cat_type'], 9), x['stem']))

    print(f"=== Building Interactive HTML Web Viewer in {output_root} ===")
    build_html_viewer(results, output_root, thr)
    print(f"✅ Finished! Open file://{os.path.abspath(output_root)}/index.html to view results.")


if __name__ == '__main__':
    main()
