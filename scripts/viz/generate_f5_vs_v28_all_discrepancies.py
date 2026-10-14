#!/usr/bin/env python3
"""
Generate Comprehensive Visual Discrepancy Inspection Panels between
F5 (58.105 Official Gold Benchmark) and Sovereign V28 (Disparity Awakening Cap60).

Directory Output Structure:
outputs/analysis/discrepancies_f5_vs_v28/
├── 01_category_conflicts/     # Images with same object, conflicting class labels
├── 02_large_diff_count/       # Images with >= 5 box differences
├── 03_medium_diff_count/      # Images with 2-4 box differences
├── 04_single_box_diff/        # Images with 1 box difference
├── all_discrepancies/         # All images with visual differences (00000018.png, etc.)
├── index.html                 # Interactive Web Gallery with keyboard navigation & filters
└── discrepancy_report.json    # Structured metadata for universal comparison platform
"""

import os
import sys
import json
import zipfile
import shutil
import concurrent.futures
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

ROOT = '/home/constant/CodeBuddy/Dev/AIC'
CLASS_NAMES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}
RARE_CLASSES = {1, 7, 9, 10, 11}

# Colors (RGB)
C_EXACT    = (46, 204, 113)    # Emerald Green (Consensus)
C_CONF     = (255, 140, 0)     # Bright Orange (Conflict)
C_F5_ONLY  = (0, 195, 255)     # Electric Cyan (F5 Only)
C_V28_ONLY = (255, 45, 85)     # Crimson / Magenta (V28 Only)
C_BG       = (20, 22, 26)      # Dark Background
C_BANNER   = (12, 14, 18)      # Header
C_CARD     = (28, 32, 38)
C_TEXT     = (240, 245, 250)
C_MUTED    = (160, 168, 180)

# Paths
F5_ZIP  = os.path.join(ROOT, 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip')
V28_ZIP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'outputs/submissions/submissions_v28/submission_sovereign_v28_3d_consolidated.zip')
VIS_DIR = os.path.join(ROOT, 'dataset/test/AIC2026_PHASE_2_1000/visible')
IR_DIR  = os.path.join(ROOT, 'dataset/test/AIC2026_PHASE_2_1000/infrared')

OUTPUT_ROOT = os.path.join(ROOT, 'outputs/analysis/discrepancies_f5_vs_v28')
DIR_CONFLICTS = os.path.join(OUTPUT_ROOT, '01_category_conflicts')
DIR_LARGE     = os.path.join(OUTPUT_ROOT, '02_large_diff_count')
DIR_MEDIUM    = os.path.join(OUTPUT_ROOT, '03_medium_diff_count')
DIR_SINGLE    = os.path.join(OUTPUT_ROOT, '04_single_box_diff')
DIR_ALL       = os.path.join(OUTPUT_ROOT, 'all_discrepancies')

FONT_CANDS = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
]
FONT_PATH = next((f for f in FONT_CANDS if os.path.exists(f)), None)


def load_submission_zip(zip_path):
    d = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith('.txt'):
                continue
            stem = os.path.splitext(os.path.basename(name))[0]
            content = zf.read(name).decode('utf-8').strip()
            boxes = []
            if content:
                for line in content.replace('\r\n', '\n').split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        cid = int(parts[0])
                        cx, cy, w, h, sc = map(float, parts[1:6])
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
    # 1. Exact consensus pool (same class, IoU >= 0.25)
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
            
    # 2. Category conflict pool (diff class, IoU >= 0.30)
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
    
    font_title = ImageFont.truetype(FONT_PATH, 22) if FONT_PATH else ImageFont.load_default()
    font_sub = ImageFont.truetype(FONT_PATH, 15) if FONT_PATH else ImageFont.load_default()
    font_badge = ImageFont.truetype(FONT_PATH, 13) if FONT_PATH else ImageFont.load_default()
    font_box = ImageFont.truetype(FONT_PATH, 13) if FONT_PATH else ImageFont.load_default()
    
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
    
    title_str = f"【图号 {stem}】 F5 (58.105) vs Sovereign V28 (Disparity Awakening) 差异深度核查"
    sub_str1 = f"框数统计: F5 标杆共 {len(bf)} 框  |  V28 觉醒版共 {len(bg)} 框  |  共识一致: {num_exact} 框  |  类别冲突: {num_conflicts} 处"
    draw.text((MARGIN, 16), title_str, fill=C_TEXT, font=font_title)
    draw.text((MARGIN, 48), sub_str1, fill=C_MUTED, font=font_sub)
    
    diff_items = []
    if num_conflicts > 0:
        for i, v in matched_f.items():
            if v[0] == 'conflict':
                j, iou = v[1], v[2]
                diff_items.append(f"[类别冲突] F5:{CLASS_NAMES[bf[i][0]]}({bf[i][5]:.2f}) vs V28:{CLASS_NAMES[bg[j][0]]}({bg[j][5]:.2f}) IoU={iou:.2f}")
    if num_f_only > 0:
        f_only_names = [f"{CLASS_NAMES[bf[i][0]]}({bf[i][5]:.2f})" for i in range(len(bf)) if i not in matched_f]
        diff_items.append(f"[F5 独有 {num_f_only}框]: " + ", ".join(f_only_names[:5]) + (f" 等+{len(f_only_names)-5}个" if len(f_only_names)>5 else ""))
    if num_g_only > 0:
        g_only_names = [f"{CLASS_NAMES[bg[j][0]]}({bg[j][5]:.2f})" for j in range(len(bg)) if j not in matched_g]
        diff_items.append(f"[V28 独有 {num_g_only}框]: " + ", ".join(g_only_names[:5]) + (f" 等+{len(g_only_names)-5}个" if len(g_only_names)>5 else ""))
        
    diff_text = "  |  ".join(diff_items)
    draw.text((MARGIN, 72), diff_text, fill=(255, 200, 80), font=font_sub)
    
    # Col 1: Reference
    x_col1 = MARGIN
    y_panels = HEADER_H + TITLE_BAR_H
    draw.rectangle([x_col1, HEADER_H + 4, x_col1 + W_PANEL, HEADER_H + TITLE_BAR_H - 2], fill=C_CARD)
    draw.text((x_col1 + 10, HEADER_H + 8), "物理参考：可见光 RGB 原图 (右上角为红外热成像)", fill=(120, 220, 255), font=font_sub)
    
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
                else:
                    color = C_CONF
                    partner_idx = match_dict[idx][1]
                    partner_b = (bg if is_f5 else bf)[partner_idx]
                    tag = f"!冲突: {CLASS_NAMES[cid]} vs {CLASS_NAMES[partner_b[0]]}"
            else:
                color = C_F5_ONLY if is_f5 else C_V28_ONLY
                tag = f"{CLASS_NAMES[cid]} {sc:.2f} ({'F5独有' if is_f5 else 'V28独有'})"
                
            draw_p.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Tag banner
            bbox = font_box.getbbox(tag) if hasattr(font_box, 'getbbox') else (0, 0, len(tag)*8, 14)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            ty = max(0, y1 - th - 4)
            draw_p.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
            draw_p.text((x1 + 3, ty + 1), tag, fill=(0, 0, 0) if color == C_EXACT else (255, 255, 255), font=font_box)
            
        return panel

    # Col 2: F5 Benchmark
    x_col2 = MARGIN + W_PANEL + PAD
    draw.rectangle([x_col2, HEADER_H + 4, x_col2 + W_PANEL, HEADER_H + TITLE_BAR_H - 2], fill=C_CARD)
    draw.text((x_col2 + 10, HEADER_H + 8), f"F5 官方标杆预测 (58.105 分) — 共 {len(bf)} 框", fill=(0, 195, 255), font=font_sub)
    panel2 = draw_boxes_on_img(img_vis_r, bf, matched_f, is_f5=True)
    canvas.paste(panel2, (x_col2, y_panels))
    
    # Col 3: Sovereign V28
    x_col3 = MARGIN + (W_PANEL + PAD) * 2
    draw.rectangle([x_col3, HEADER_H + 4, x_col3 + W_PANEL, HEADER_H + TITLE_BAR_H - 2], fill=C_CARD)
    draw.text((x_col3 + 10, HEADER_H + 8), f"Sovereign V28 物理视差觉醒 — 共 {len(bg)} 框", fill=(255, 45, 85), font=font_sub)
    panel3 = draw_boxes_on_img(img_vis_r, bg, matched_g, is_f5=False)
    canvas.paste(panel3, (x_col3, y_panels))
    
    out_file = os.path.join(DIR_ALL, f"{stem}.png")
    canvas.save(out_file, quality=92)
    
    # Symlink to categorized folders
    total_diff = num_conflicts + num_f_only + num_g_only
    if num_conflicts > 0:
        os.symlink(os.path.relpath(out_file, DIR_CONFLICTS), os.path.join(DIR_CONFLICTS, f"{stem}.png"))
    if total_diff >= 5:
        os.symlink(os.path.relpath(out_file, DIR_LARGE), os.path.join(DIR_LARGE, f"{stem}.png"))
    elif total_diff in (2, 3, 4):
        os.symlink(os.path.relpath(out_file, DIR_MEDIUM), os.path.join(DIR_MEDIUM, f"{stem}.png"))
    elif total_diff == 1:
        os.symlink(os.path.relpath(out_file, DIR_SINGLE), os.path.join(DIR_SINGLE, f"{stem}.png"))
        
    return {
        'stem': stem,
        'f5_boxes': len(bf),
        'v28_boxes': len(bg),
        'exact_consensus': num_exact,
        'conflicts': num_conflicts,
        'f5_only': num_f_only,
        'v28_only': num_g_only,
        'total_diff': total_diff,
        'img_path': f"all_discrepancies/{stem}.png"
    }


def main():
    print("=" * 80)
    print(" F5 vs Sovereign V28 Comprehensive Visual Discrepancy Generation ".center(80, "="))
    print("=" * 80)
    
    for d in [DIR_CONFLICTS, DIR_LARGE, DIR_MEDIUM, DIR_SINGLE, DIR_ALL]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        
    print(f"[+] Loading F5 predictions from {F5_ZIP}...")
    f5_data = load_submission_zip(F5_ZIP)
    print(f"[+] Loading Sovereign V28 predictions from {V28_ZIP}...")
    v28_data = load_submission_zip(V28_ZIP)
    
    stems = sorted(set(list(f5_data.keys()) + list(v28_data.keys())))
    print(f"[+] Total images to audit: {len(stems)}")
    
    items_to_render = []
    discrepancy_records = []
    
    for stem in stems:
        bf = f5_data.get(stem, [])
        bg = v28_data.get(stem, [])
        mf, mg = match_boxes(bf, bg)
        num_conflicts = sum(1 for v in mf.values() if v[0] == 'conflict')
        num_exact = sum(1 for v in mf.values() if v[0] == 'exact')
        num_f_only = len(bf) - len(mf)
        num_g_only = len(bg) - len(mg)
        
        if num_conflicts > 0 or num_f_only > 0 or num_g_only > 0:
            items_to_render.append((stem, bf, bg, mf, mg))
            
    print(f"[*] Found {len(items_to_render)} images with discrepancies. Commencing parallel rendering...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(render_image_panel, items_to_render):
            discrepancy_records.append(res)
            
    print(f"[✔] Rendered {len(discrepancy_records)} discrepancy panels!")
    
    # Save structured JSON
    json_path = os.path.join(OUTPUT_ROOT, 'discrepancy_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(discrepancy_records, f, indent=2, ensure_ascii=False)
        
    print(f"[✔] Saved discrepancy report to {json_path}")


if __name__ == '__main__':
    main()
