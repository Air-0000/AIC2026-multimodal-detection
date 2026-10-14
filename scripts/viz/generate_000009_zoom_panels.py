#!/usr/bin/env python3
"""
Generate comprehensive high-resolution zoomed inspection panels for image 000009_020_00000244
comparing F5 (57.266) vs V23 (53.00分).
"""

import os
import zipfile
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}
RARE_CLASSES = {1, 7, 9, 10, 11}

# Colors
C_EXACT    = (46, 204, 113)    # Emerald Green (Consensus)
C_CONF     = (255, 140, 0)     # Bright Orange (Conflict)
C_F5_ONLY  = (0, 195, 255)     # Electric Cyan (F5 Only)
C_V23_ONLY = (255, 45, 85)     # Crimson / Magenta (V23 Only)
C_BG       = (16, 18, 22)
C_BANNER   = (10, 12, 16)
C_CARD     = (26, 30, 36)
C_TEXT     = (240, 245, 250)
C_MUTED    = (160, 168, 180)

FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
font_title = ImageFont.truetype(FONT_PATH, 24)
font_sub = ImageFont.truetype(FONT_PATH, 16)
font_card = ImageFont.truetype(FONT_PATH, 15)
font_box = ImageFont.truetype(FONT_PATH, 13)
font_badge = ImageFont.truetype(FONT_PATH, 12)

stem = '000009_020_00000244'
p_vis = f'dataset/test/AIC2026_PHASE_2_1000/visible/{stem}.png'
p_ir = f'dataset/test/AIC2026_PHASE_2_1000/infrared/{stem}.png'

img_vis = Image.open(p_vis).convert('RGB')
img_ir = Image.open(p_ir).convert('RGB')
W_orig, H_orig = img_vis.size

# Load F5
with open(f'/home/constant/Downloads/AIC2026_phase2_predictions_F5seed20261002_RARE45_COMMON50_compact7212_20261005/{stem}.txt') as f:
    f5_boxes = []
    for l in f:
        if not l.strip(): continue
        p = l.split()
        cid = int(p[0])
        cx, cy, w, h, sc = map(float, p[1:])
        thr = 0.45 if cid in RARE_CLASSES else 0.50
        if sc >= thr: f5_boxes.append((cid, cx, cy, w, h, sc))
f5_boxes.sort(key=lambda x: x[5], reverse=True)

# Load V23
with zipfile.ZipFile('/home/constant/Downloads/submission_v23_flip_tta_thr45.zip') as zf:
    v23_content = zf.read(f'{stem}.txt').decode()
    v23_boxes = []
    for l in v23_content.split('\n'):
        if not l.strip(): continue
        p = l.split()
        cid = int(p[0])
        cx, cy, w, h, sc = map(float, p[1:])
        v23_boxes.append((cid, cx, cy, w, h, sc))
v23_boxes.sort(key=lambda x: x[5], reverse=True)

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

# Matching
p_exact = []
for i, bf in enumerate(f5_boxes):
    for j, bv in enumerate(v23_boxes):
        if bf[0] == bv[0]:
            iou = compute_iou(bf, bv)
            if iou >= 0.25: p_exact.append((iou, i, j))
p_exact.sort(key=lambda x: x[0], reverse=True)
mf = {}
mv = {}
for iou, i, j in p_exact:
    if i not in mf and j not in mv:
        mf[i] = ('exact', j, iou)
        mv[j] = ('exact', i, iou)

p_conf = []
for i, bf in enumerate(f5_boxes):
    if i in mf: continue
    for j, bv in enumerate(v23_boxes):
        if j in mv: continue
        if bf[0] != bv[0]:
            iou = compute_iou(bf, bv)
            if iou >= 0.30: p_conf.append((iou, i, j))
p_conf.sort(key=lambda x: x[0], reverse=True)
for iou, i, j in p_conf:
    if i not in mf and j not in mv:
        mf[i] = ('conflict', j, iou)
        mv[j] = ('conflict', i, iou)

def draw_boxes_on_crop(base_crop, boxes, match_dict, is_f5, crop_rect):
    rx1, ry1, rx2, ry2 = crop_rect
    cw = rx2 - rx1
    ch = ry2 - ry1
    crop_w, crop_h = base_crop.size
    
    panel = base_crop.copy()
    draw_p = ImageDraw.Draw(panel)
    
    for idx, b in enumerate(boxes):
        cid, cx, cy, bw, bh, sc = b
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
        
        if idx in match_dict:
            m_type = match_dict[idx][0]
            if m_type == 'exact':
                color = C_EXACT
                tag = f"{CLASS_NAMES[cid]} {sc:.2f}"
            else:
                color = C_CONF
                tag = f"[冲突] {CLASS_NAMES[cid]} {sc:.2f}"
        else:
            color = C_F5_ONLY if is_f5 else C_V23_ONLY
            tag = f"[{'F5独有' if is_f5 else 'V23独有'}] {CLASS_NAMES[cid]} {sc:.2f}"
            
        draw_p.rectangle([px1, py1, px2, py2], outline=color, width=3)
        bbox = font_box.getbbox(tag)
        tw = bbox[2] - bbox[0] + 8
        th = bbox[3] - bbox[1] + 6
        by_label = max(0, py1 - th)
        draw_p.rectangle([px1, by_label, px1 + tw, by_label + th], fill=color)
        draw_p.text((px1 + 4, by_label + 2), tag, fill=(0, 0, 0) if color in [C_CONF, C_F5_ONLY, C_EXACT] else (255, 255, 255), font=font_box)
        
    return panel


CANVAS_W = 2400
HEADER_H = 110
SECTION_H = 40
MARGIN = 16
PAD = 14

# Section 1: Panoramic Audience Strip (x: 0.12~0.92, y: 0.05~0.22)
roi_rect = (0.12, 0.05, 0.92, 0.22)
rx1, ry1, rx2, ry2 = int(0.12*W_orig), int(0.05*H_orig), int(0.92*W_orig), int(0.22*H_orig)
crop_roi_vis = img_vis.crop((rx1, ry1, rx2, ry2))
crop_roi_ir = img_ir.crop((rx1, ry1, rx2, ry2))

W_ROI_PANEL = (CANVAS_W - MARGIN*2 - PAD*2) // 3
H_ROI_PANEL = 260

roi_vis_r = crop_roi_vis.resize((W_ROI_PANEL, H_ROI_PANEL), Image.Resampling.LANCZOS)
roi_ir_r = crop_roi_ir.resize((W_ROI_PANEL, H_ROI_PANEL), Image.Resampling.LANCZOS)

# Section 2: 3 High-Power Zoom Crops
focus_crops = [
    {
        'title': '局部一【看台左侧与柱旁行人】：left crowd (x: 0.28~0.42, y: 0.07~0.18)',
        'rect': (0.28, 0.07, 0.42, 0.18),
        'desc': 'F5独有检出 person 0.61 (柱旁行人) | V23独有检出 person 0.47 (暗光微弱行人)。两模型达成4人高度共识。'
    },
    {
        'title': '局部二【看台中右部密集坐席】：middle-right seating (x: 0.52~0.68, y: 0.06~0.18)',
        'rect': (0.52, 0.06, 0.68, 0.18),
        'desc': 'V23独有召回2名紧密并排坐姿行人: person 0.68 & person 0.67！F5此处严重漏检并坐观众。'
    },
    {
        'title': '局部三【看台最右侧座椅与边缘观众】：far right seats & crowd (x: 0.70~0.92, y: 0.06~0.22)',
        'rect': (0.70, 0.06, 0.92, 0.22),
        'desc': 'F5独有检出台阶长椅 seat 0.63 | V23独有召回极远处边缘行人 person 0.56。共识长凳 seat (F5:0.62, V23:0.90)。'
    }
]

H_FOCUS_ROW = 300
CANVAS_H = HEADER_H + SECTION_H + H_ROI_PANEL + MARGIN + (SECTION_H + H_FOCUS_ROW + MARGIN) * 3 + MARGIN

canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), C_BG)
draw = ImageDraw.Draw(canvas)

# Header
draw.rectangle([0, 0, CANVAS_W, HEADER_H], fill=C_BANNER)
draw.line([(0, HEADER_H), (CANVAS_W, HEADER_H)], fill=(45, 50, 60), width=1)
draw.text((MARGIN, 16), f"【图号 {stem}】 1080P 密集人群深度放大对比面板 (F5 标杆 vs V23 官方版)", fill=C_TEXT, font=font_title)
draw.text((MARGIN, 52), f"框数全景: F5 (57.266) 共 17 框  |  V23 (53分版) 共 19 框  |  完全共识: 15 框  |  类别冲突: 0 框  |  F5独有: 2 框  |  V23独有: 4 框", fill=C_MUTED, font=font_sub)
draw.text((MARGIN, 78), "✅ 核心特征: 零类别冲突！15个共识目标高度严密重合 (IoU 0.77~0.98) | V23独有召回中右侧2名坐姿观众 (0.68, 0.67) | F5独有检出1条长椅", fill=(100, 255, 160), font=font_sub)

curr_y = HEADER_H + MARGIN

# Panoramic Audience Strip
draw.rectangle([MARGIN, curr_y, CANVAS_W - MARGIN, curr_y + SECTION_H], fill=C_CARD)
draw.text((MARGIN + 12, curr_y + 8), "【全景密集观众台阶横幅放大 (ROI: X 12%~92%, Y 5%~22%)】", fill=(100, 220, 255), font=font_card)
curr_y += SECTION_H + 4

x1 = MARGIN
x2 = x1 + W_ROI_PANEL + PAD
x3 = x2 + W_ROI_PANEL + PAD

p1 = roi_vis_r.copy()
w_ir = 220
h_ir = int(w_ir * H_ROI_PANEL / W_ROI_PANEL)
ir_th = roi_ir_r.resize((w_ir, h_ir), Image.Resampling.LANCZOS)
p1.paste(ir_th, (W_ROI_PANEL - w_ir - 8, 8))
d_p1 = ImageDraw.Draw(p1)
d_p1.rectangle([W_ROI_PANEL - w_ir - 8, 8, W_ROI_PANEL - 8, 8 + h_ir], outline=(255, 255, 255), width=2)
d_p1.rectangle([W_ROI_PANEL - w_ir - 8, 8, W_ROI_PANEL - w_ir + 80, 26], fill=(0, 0, 0))
d_p1.text((W_ROI_PANEL - w_ir - 4, 10), "IR 热成像", fill=(255, 215, 0), font=font_badge)
canvas.paste(p1, (x1, curr_y))

p2 = draw_boxes_on_crop(roi_vis_r, f5_boxes, mf, is_f5=True, crop_rect=roi_rect)
canvas.paste(p2, (x2, curr_y))

p3 = draw_boxes_on_crop(roi_vis_r, v23_boxes, mv, is_f5=False, crop_rect=roi_rect)
canvas.paste(p3, (x3, curr_y))

curr_y += H_ROI_PANEL + MARGIN

# 3 Focus Crops
for f_idx, fc in enumerate(focus_crops):
    draw.rectangle([MARGIN, curr_y, CANVAS_W - MARGIN, curr_y + SECTION_H], fill=C_CARD)
    draw.text((MARGIN + 12, curr_y + 8), fc['title'] + "  —  " + fc['desc'], fill=(255, 215, 100), font=font_card)
    curr_y += SECTION_H + 4
    
    fx1, fy1, fx2, fy2 = fc['rect']
    px1, py1, px2, py2 = int(fx1*W_orig), int(fy1*H_orig), int(fx2*W_orig), int(fy2*H_orig)
    crop_vis = img_vis.crop((px1, py1, px2, py2))
    crop_ir = img_ir.crop((px1, py1, px2, py2))
    
    W_F_PANEL = (CANVAS_W - MARGIN*2 - PAD*2) // 3
    crop_vis_r = crop_vis.resize((W_F_PANEL, H_FOCUS_ROW), Image.Resampling.LANCZOS)
    crop_ir_r = crop_ir.resize((W_F_PANEL, H_FOCUS_ROW), Image.Resampling.LANCZOS)
    
    f_p1 = crop_vis_r.copy()
    w_th = 160
    h_th = int(w_th * H_FOCUS_ROW / W_F_PANEL)
    ir_th_f = crop_ir_r.resize((w_th, h_th), Image.Resampling.LANCZOS)
    f_p1.paste(ir_th_f, (W_F_PANEL - w_th - 6, 6))
    d_f1 = ImageDraw.Draw(f_p1)
    d_f1.rectangle([W_F_PANEL - w_th - 6, 6, W_F_PANEL - 6, 6 + h_th], outline=(255, 255, 255), width=2)
    canvas.paste(f_p1, (x1, curr_y))
    
    f_p2 = draw_boxes_on_crop(crop_vis_r, f5_boxes, mf, is_f5=True, crop_rect=fc['rect'])
    canvas.paste(f_p2, (x2, curr_y))
    
    f_p3 = draw_boxes_on_crop(crop_vis_r, v23_boxes, mv, is_f5=False, crop_rect=fc['rect'])
    canvas.paste(f_p3, (x3, curr_y))
    
    curr_y += H_FOCUS_ROW + MARGIN

out_artifact = '/home/constant/.gemini/antigravity/brain/9ade8979-579e-41ff-bbad-391b8a3785b5/zoom_audit_000009_020_00000244.png'
out_disk = '/home/constant/CodeBuddy/Dev/AIC/outputs/analysis/discrepancies_f5_vs_v23/zoom_000009_020_00000244_deepdive.png'

canvas.save(out_artifact, quality=95)
canvas.save(out_disk, quality=95)
print('Successfully saved zoom panel to:')
print('  Artifact:', out_artifact)
print('  Disk    :', out_disk)
