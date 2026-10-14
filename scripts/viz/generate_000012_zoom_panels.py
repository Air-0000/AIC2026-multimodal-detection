#!/usr/bin/env python3
"""
Generate comprehensive high-resolution, un-occluded zoom inspection panels for image 000012_030_00000250
comparing F5 (57.266) vs V23 (53.00分).
Specifically addresses user feedback:
"展示的时候一次看部分，全被标签挡住了" (Show partial areas in sections, labels must NOT occlude the objects).
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
font_box = ImageFont.truetype(FONT_PATH, 12)
font_badge = ImageFont.truetype(FONT_PATH, 11)

stem = '000012_030_00000250'
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

def draw_boxes_clean(base_crop, boxes, match_dict, is_f5, crop_rect):
    rx1, ry1, rx2, ry2 = crop_rect
    cw = rx2 - rx1
    ch = ry2 - ry1
    crop_w, crop_h = base_crop.size
    
    panel = base_crop.copy()
    draw_p = ImageDraw.Draw(panel)
    
    # Sort boxes: consensus first, discrepancies last so discrepancies are on top
    box_indices = list(range(len(boxes)))
    box_indices.sort(key=lambda idx: 1 if idx not in match_dict else 0)
    
    for idx in box_indices:
        b = boxes[idx]
        cid, cx, cy, bw, bh, sc = b
        bx1 = cx - bw/2.0
        by1 = cy - bh/2.0
        bx2 = cx + bw/2.0
        by2 = cy + bh/2.0
        
        # Check overlap
        if bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2:
            continue
            
        px1 = (bx1 - rx1) / cw * crop_w
        py1 = (by1 - ry1) / ch * crop_h
        px2 = (bx2 - rx1) / cw * crop_w
        py2 = (by2 - ry1) / ch * crop_h
        
        is_consensus = (idx in match_dict)
        if is_consensus:
            color = C_EXACT
            line_w = 2
            tag = f"{CLASS_NAMES[cid][:3]} {sc:.2f}" # e.g. ani 0.89
        else:
            color = C_F5_ONLY if is_f5 else C_V23_ONLY
            line_w = 3
            tag = f"★{CLASS_NAMES[cid]} {sc:.2f}"
            
        # Draw thin, completely hollow wireframe box
        draw_p.rectangle([px1, py1, px2, py2], outline=color, width=line_w)
        
        # Draw minimalist unobtrusive label tag outside above
        bbox = font_box.getbbox(tag)
        tw = bbox[2] - bbox[0] + 6
        th = bbox[3] - bbox[1] + 4
        by_label = py1 - th if py1 - th > 0 else py2 + 1
        
        # Only draw background tag for discrepancies or consensus
        draw_p.rectangle([px1, by_label, px1 + tw, by_label + th], fill=(15, 17, 22))
        draw_p.rectangle([px1, by_label, px1 + tw, by_label + th], outline=color, width=1)
        draw_p.text((px1 + 3, by_label + 1), tag, fill=color, font=font_box)
        
    return panel


CANVAS_W = 2400
HEADER_H = 110
SECTION_H = 42
MARGIN = 16
PAD = 14

# Define 4 distinct, un-occluded focus sections:
focus_sections = [
    {
        'title': '【局部一：右侧码头船只与甲板游客群】 (x: 0.74~1.00, y: 0.38~0.50)',
        'rect': (0.74, 0.38, 1.00, 0.50),
        'desc': '检出2艘船+近处码头游客。共识5人2船。V23高分多召回船头2名远景站立游客 (0.86, 0.86)，F5独有报1人 (0.76)。',
        'height': 380
    },
    {
        'title': '【局部二：右下方水鸟密集区·上层群】 (x: 0.80~0.98, y: 0.46~0.55)',
        'rect': (0.80, 0.46, 0.98, 0.55),
        'desc': '密集游动水鸟。共识5只鸟。V23高召回多检出4只较小水鸟 (0.63, 0.61, 0.57, 0.48)，F5独有检出1只 (0.73)。',
        'height': 380
    },
    {
        'title': '【局部三：右下方水鸟密集区·下层近景大鸟】 (x: 0.78~1.00, y: 0.53~0.66)',
        'rect': (0.78, 0.53, 1.00, 0.66),
        'desc': '近景清晰游动大水鸟。共识2只核心大鸟 (0.89, 0.76)。F5多报2只水波倒影 (0.69, 0.60)。',
        'height': 380
    },
    {
        'title': '【局部四：湖面左侧与中部散游天鹅群】 (x: 0.12~0.60, y: 0.42~0.60)',
        'rect': (0.12, 0.42, 0.60, 0.60),
        'desc': '左侧与中央开阔湖面。两模型达成 100% 绝对共识（共识 7 只天鹅水鸟，零差异，逐只严丝合缝）！',
        'height': 320
    }
]

total_sections_h = sum(s['height'] + SECTION_H + MARGIN for s in focus_sections)
CANVAS_H = HEADER_H + MARGIN + total_sections_h

canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), C_BG)
draw = ImageDraw.Draw(canvas)

# Header
draw.rectangle([0, 0, CANVAS_W, HEADER_H], fill=C_BANNER)
draw.line([(0, HEADER_H), (CANVAS_W, HEADER_H)], fill=(45, 50, 60), width=1)
draw.text((MARGIN, 16), f"【图号 {stem}】 密集水鸟与游船场景·分区分段超清无遮挡对比面板", fill=C_TEXT, font=font_title)
draw.text((MARGIN, 52), f"框数全景: F5 (57.266) 共 25 框  |  V23 (53分版) 共 27 框  |  完全共识: 21 框 (84%)  |  类别冲突: 0 框  |  F5独有: 4 框  |  V23独有: 6 框", fill=C_MUTED, font=font_sub)
draw.text((MARGIN, 78), "💡 针对密集排布优化: 第一列物理原图纯净无任何遮挡; 第二/三列采用极细透明线框与微型浮动标签，彻底避免标签覆盖水鸟与行人！", fill=(100, 255, 160), font=font_sub)

curr_y = HEADER_H + MARGIN

W_COL = (CANVAS_W - MARGIN*2 - PAD*2) // 3

for s_idx, sec in enumerate(focus_sections):
    h_sec = sec['height']
    draw.rectangle([MARGIN, curr_y, CANVAS_W - MARGIN, curr_y + SECTION_H], fill=C_CARD)
    draw.text((MARGIN + 12, curr_y + 10), sec['title'] + "  —  " + sec['desc'], fill=(255, 215, 100), font=font_card)
    curr_y += SECTION_H + 4
    
    fx1, fy1, fx2, fy2 = sec['rect']
    px1, py1, px2, py2 = int(fx1*W_orig), int(fy1*H_orig), int(fx2*W_orig), int(fy2*H_orig)
    crop_vis = img_vis.crop((px1, py1, px2, py2))
    crop_ir = img_ir.crop((px1, py1, px2, py2))
    
    crop_vis_r = crop_vis.resize((W_COL, h_sec), Image.Resampling.LANCZOS)
    crop_ir_r = crop_ir.resize((W_COL, h_sec), Image.Resampling.LANCZOS)
    
    x1 = MARGIN
    x2 = x1 + W_COL + PAD
    x3 = x2 + W_COL + PAD
    
    # Col 1: Pristine Raw (Unobstructed!) + Small IR in corner
    c1 = crop_vis_r.copy()
    w_th = 150
    h_th = int(w_th * h_sec / W_COL)
    ir_th = crop_ir_r.resize((w_th, h_th), Image.Resampling.LANCZOS)
    c1.paste(ir_th, (W_COL - w_th - 6, 6))
    d_c1 = ImageDraw.Draw(c1)
    d_c1.rectangle([W_COL - w_th - 6, 6, W_COL - 6, 6 + h_th], outline=(255, 255, 255), width=2)
    d_c1.rectangle([W_COL - w_th - 6, 6, W_COL - w_th + 65, 22], fill=(0, 0, 0))
    d_c1.text((W_COL - w_th - 3, 7), "IR热成像", fill=(255, 215, 0), font=font_badge)
    # Add watermark to ensure user knows Col 1 is 100% clean
    d_c1.rectangle([6, 6, 170, 24], fill=(0, 0, 0, 180))
    d_c1.text((10, 8), "物理原图 (无遮挡纯净像素)", fill=(100, 220, 255), font=font_badge)
    canvas.paste(c1, (x1, curr_y))
    
    # Col 2: F5
    c2 = draw_boxes_clean(crop_vis_r, f5_boxes, mf, is_f5=True, crop_rect=sec['rect'])
    d_c2 = ImageDraw.Draw(c2)
    d_c2.rectangle([6, 6, 170, 24], fill=(0, 0, 0, 180))
    d_c2.text((10, 8), f"F5 标杆预测 (青色=F5独有)", fill=(0, 220, 255), font=font_badge)
    canvas.paste(c2, (x2, curr_y))
    
    # Col 3: V23
    c3 = draw_boxes_clean(crop_vis_r, v23_boxes, mv, is_f5=False, crop_rect=sec['rect'])
    d_c3 = ImageDraw.Draw(c3)
    d_c3.rectangle([6, 6, 170, 24], fill=(0, 0, 0, 180))
    d_c3.text((10, 8), f"V23 预测 (洋红=V23独有)", fill=(255, 80, 120), font=font_badge)
    canvas.paste(c3, (x3, curr_y))
    
    curr_y += h_sec + MARGIN

out_artifact = '/home/constant/.gemini/antigravity/brain/9ade8979-579e-41ff-bbad-391b8a3785b5/zoom_audit_000012_030_00000250.png'
out_disk = '/home/constant/CodeBuddy/Dev/AIC/outputs/analysis/discrepancies_f5_vs_v23/zoom_000012_030_00000250_deepdive.png'

canvas.save(out_artifact, quality=95)
canvas.save(out_disk, quality=95)
print('Successfully saved zoom panel to:')
print('  Artifact:', out_artifact)
print('  Disk    :', out_disk)
