#!/usr/bin/env python3
"""
Generate High-Resolution Discrepancy & Zoom Inspection Panels (V24 vs F5)
========================================================================
Draws:
  - Sovereign V24 predictions in Crimson Red (#FF2A2A)
  - F5 Benchmark predictions in Cyan/Blue (#00D4FF)
  - Full image overview + High-resolution zoomed insets on key discrepancy ROIs.
"""

import os
import sys
import glob
import zipfile
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path("/home/constant/CodeBuddy/Dev/AIC")
DATA_DIR = ROOT / "dataset/test/AIC2026_PHASE_2_1000/visible"
F5_ZIP = ROOT / "work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip"
V24_ZIP = ROOT / "outputs/submissions/submissions_v24/submission_v24_native_compact.zip"
OUT_DIR = ROOT / "outputs/analysis/visualizations_discrepancy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}

# Colors (RGB)
COLOR_V24 = (230, 40, 40)       # Crimson Red
HEX_V24 = "#E62828"
COLOR_F5 = (0, 180, 230)        # Electric Cyan
HEX_F5 = "#00B4E6"
COLOR_CONSENSUS = (46, 204, 113) # Emerald Green
HEX_CONSENSUS = "#2ECC71"
COLOR_ROI = (255, 215, 0)       # Gold
HEX_ROI = "#FFD700"

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'LXGW WenKai', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def read_zip(p):
    res = {}
    with zipfile.ZipFile(p, 'r') as zf:
        for name in zf.namelist():
            if not name.endswith('.txt'):
                continue
            stem = Path(name).stem
            lines = [l.strip() for l in zf.read(name).decode('utf-8').strip().split('\n') if l.strip()]
            boxes = []
            for l in lines:
                parts = l.split()
                if len(parts) >= 6:
                    boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])))
            res[stem] = boxes
    return res


def load_image(stem):
    for ext in ('.jpg', '.png', '.jpeg'):
        p = DATA_DIR / f"{stem}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"Image not found for stem {stem} in {DATA_DIR}")


def box_iou(b1, b2):
    x1_1, y1_1 = b1[1] - b1[3]/2, b1[2] - b1[4]/2
    x2_1, y2_1 = b1[1] + b1[3]/2, b1[2] + b1[4]/2
    x1_2, y1_2 = b2[1] - b2[3]/2, b2[2] - b2[4]/2
    x2_2, y2_2 = b2[1] + b2[3]/2, b2[2] + b2[4]/2
    ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
    ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    a2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = a1 + a2 - inter
    return inter / union if union > 1e-9 else 0.0


def determine_rois(boxes_v24, boxes_f5, W, H, max_rois=2):
    """
    Finds the bounding boxes with the highest discrepancy to zoom in on.
    Returns list of (crop_x1, crop_y1, crop_x2, crop_y2, desc) in pixels.
    """
    # 1. Look for un-matched boxes
    unmatched_v24 = []
    for b_v in boxes_v24:
        matched = any(box_iou(b_v, b_f) >= 0.50 for b_f in boxes_f5)
        if not matched:
            unmatched_v24.append((b_v, "V24 独立检出目标 (F5 遗漏)"))

    unmatched_f5 = []
    for b_f in boxes_f5:
        matched = any(box_iou(b_f, b_v) >= 0.50 for b_v in boxes_v24)
        if not matched:
            unmatched_f5.append((b_f, "F5 独有预测目标 (疑似虚警/噪点)"))

    # 2. Look for class confusion
    confused = []
    for b_v in boxes_v24:
        for b_f in boxes_f5:
            if box_iou(b_v, b_f) >= 0.50 and b_v[0] != b_f[0]:
                confused.append((b_v, f"类别分歧: V24={CLASS_NAMES[b_v[0]]} vs F5={CLASS_NAMES[b_f[0]]}"))

    candidates = confused + unmatched_v24 + unmatched_f5
    if not candidates:
        # Fallback: take highest confidence box
        all_b = boxes_v24 or boxes_f5
        if all_b:
            sorted_b = sorted(all_b, key=lambda x: -x[5])
            candidates = [(sorted_b[0], "高置信度共识目标细节")]

    rois = []
    for item in candidates[:max_rois]:
        b, desc = item
        cid, cx, cy, bw, bh, sc = b
        px_cx = cx * W
        px_cy = cy * H
        px_w = bw * W
        px_h = bh * H

        # Generous padding around the target for visual context (at least 2x box size or 180px)
        pad_w = max(px_w * 0.8, 120.0)
        pad_h = max(px_h * 0.8, 120.0)

        x1 = max(0, int(px_cx - px_w / 2.0 - pad_w))
        y1 = max(0, int(px_cy - px_h / 2.0 - pad_h))
        x2 = min(W, int(px_cx + px_w / 2.0 + pad_w))
        y2 = min(H, int(px_cy + px_h / 2.0 + pad_h))

        # Check if overlaps heavily with existing ROI
        if any(abs(x1 - r[0]) < 100 and abs(y1 - r[1]) < 100 for r in rois):
            continue
        rois.append((x1, y1, x2, y2, desc))

    return rois


def render_panel(stem, category_title, case_desc, out_filename):
    img = load_image(stem)
    W, H = img.size

    b_v24 = v24_data.get(stem, [])
    b_f5 = f5_data.get(stem, [])

    rois = determine_rois(b_v24, b_f5, W, H, max_rois=2)

    # Setup matplotlib figure
    # Left: Big Overview (2 rows or 1 big col)
    # Right: 2 Zoomed Insets
    n_rois = len(rois)
    fig = plt.figure(figsize=(20, 10.5), dpi=220)
    
    # Grid layout: 2 rows, 2 columns (Left col spans 2 rows)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0],
                          wspace=0.12, hspace=0.18, left=0.04, right=0.97, top=0.91, bottom=0.05)

    ax_main = fig.add_subplot(gs[:, 0])

    # 1. Main Overview
    ax_main.imshow(img)
    ax_main.set_title(f"全局全景视图 (全幅 1920×1080) — 红色: Sovereign V24 ({len(b_v24)}框) | 青色: F5 ({len(b_f5)}框)",
                      fontsize=12, fontweight='bold', pad=10)
    ax_main.axis('off')

    # Draw F5 boxes first (Cyan)
    for b in b_f5:
        cid, cx, cy, bw, bh, sc = b
        x1 = (cx - bw/2.0) * W
        y1 = (cy - bh/2.0) * H
        w = bw * W
        h = bh * H
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2.0, edgecolor=HEX_F5, facecolor='none', linestyle='-')
        ax_main.add_patch(rect)
        # Small label
        ax_main.text(x1, max(12, y1 - 4), f"F5:{CLASS_NAMES[cid]} {sc:.2f}",
                     fontsize=7, fontweight='bold', color='black',
                     bbox=dict(boxstyle='square,pad=0.15', facecolor=HEX_F5, edgecolor='none', alpha=0.85))

    # Draw V24 boxes on top (Red)
    for b in b_v24:
        cid, cx, cy, bw, bh, sc = b
        x1 = (cx - bw/2.0) * W
        y1 = (cy - bh/2.0) * H
        w = bw * W
        h = bh * H
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2.5, edgecolor=HEX_V24, facecolor='none', linestyle='-')
        ax_main.add_patch(rect)
        ax_main.text(x1 + w, min(H - 4, y1 + h + 10), f"V24:{CLASS_NAMES[cid]} {sc:.2f}",
                     fontsize=7.5, fontweight='bold', color='white',
                     bbox=dict(boxstyle='square,pad=0.15', facecolor=HEX_V24, edgecolor='none', alpha=0.92))

    # Draw ROI indicator boxes on main
    for i, (rx1, ry1, rx2, ry2, rdesc) in enumerate(rois):
        roi_rect = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1,
                                     linewidth=2.8, edgecolor=HEX_ROI, facecolor='none', linestyle='--')
        ax_main.add_patch(roi_rect)
        ax_main.text(rx1 + 8, ry1 + 22, f"放大区域 #{i+1}",
                     fontsize=9, fontweight='bold', color='black',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor=HEX_ROI, edgecolor='black', alpha=0.95))

    # 2. Right Zoomed Insets
    for i in range(2):
        ax_zoom = fig.add_subplot(gs[i, 1])
        if i < len(rois):
            rx1, ry1, rx2, ry2, rdesc = rois[i]
            crop_img = img.crop((rx1, ry1, rx2, ry2))
            crop_w = rx2 - rx1
            crop_h = ry2 - ry1

            ax_zoom.imshow(crop_img)
            ax_zoom.set_title(f"【高精细节放大 #{i+1}】: {rdesc}",
                             fontsize=11, fontweight='bold', color='#1D3557', pad=8)
            ax_zoom.axis('off')

            # Draw local boxes in this crop
            # F5 (Cyan)
            for b in b_f5:
                cid, cx, cy, bw, bh, sc = b
                bx1 = (cx - bw/2.0) * W - rx1
                by1 = (cy - bh/2.0) * H - ry1
                bw_px = bw * W
                bh_px = bh * H
                if bx1 + bw_px > 0 and bx1 < crop_w and by1 + bh_px > 0 and by1 < crop_h:
                    rect = patches.Rectangle((bx1, by1), bw_px, bh_px, linewidth=2.8,
                                             edgecolor=HEX_F5, facecolor='none', linestyle='--')
                    ax_zoom.add_patch(rect)
                    ax_zoom.text(bx1, max(15, by1 - 5), f"F5: {CLASS_NAMES[cid]} {sc:.2f}",
                                 fontsize=8.5, fontweight='bold', color='black',
                                 bbox=dict(boxstyle='square,pad=0.2', facecolor=HEX_F5, edgecolor='black', alpha=0.9))

            # V24 (Red)
            for b in b_v24:
                cid, cx, cy, bw, bh, sc = b
                bx1 = (cx - bw/2.0) * W - rx1
                by1 = (cy - bh/2.0) * H - ry1
                bw_px = bw * W
                bh_px = bh * H
                if bx1 + bw_px > 0 and bx1 < crop_w and by1 + bh_px > 0 and by1 < crop_h:
                    rect = patches.Rectangle((bx1, by1), bw_px, bh_px, linewidth=3.2,
                                             edgecolor=HEX_V24, facecolor='none', linestyle='-')
                    ax_zoom.add_patch(rect)
                    ax_zoom.text(bx1, min(crop_h - 6, by1 + bh_px + 14), f"V24: {CLASS_NAMES[cid]} {sc:.2f}",
                                 fontsize=9, fontweight='bold', color='white',
                                 bbox=dict(boxstyle='square,pad=0.2', facecolor=HEX_V24, edgecolor='black', alpha=0.95))
        else:
            # Placeholder if only 1 ROI
            ax_zoom.axis('off')
            ax_zoom.text(0.5, 0.5, "（该图像仅包含单处显著差异区域）",
                         fontsize=12, color='#6C757D', ha='center', va='center')

    # Figure Master Title
    fig.suptitle(f"{category_title} | 图像样本: {stem}\n{case_desc}",
                 fontsize=14, fontweight='bold', color='#111111', y=0.97)

    save_path = OUT_DIR / out_filename
    plt.savefig(save_path, bbox_inches='tight', dpi=220)
    plt.close()
    print(f"[+] Rendered: {out_filename}")
    return save_path


print("=" * 80)
print(" Loading Submissions for Visual Discrepancy Generation ".center(80, "="))
print("=" * 80)
f5_data = read_zip(F5_ZIP)
v24_data = read_zip(V24_ZIP)

# -------------------------------------------------------------------------
# Define all key case studies to generate
# -------------------------------------------------------------------------
CASES = [
    # ---------------- Category A: F5 Empty -> V24 Saved -----------------
    ("000351", "【类别 A: F5 全空漏检 / V24 稳健挽救】",
     "F5 检出 0 框 (全空漏检) | V24 独立捕获正中超高分动物 animal (置信度 0.963)",
     "CatA_01_000351_f5_empty_v24_animal_096.png"),

    ("000367", "【类别 A: F5 全空漏检 / V24 稳健挽救】",
     "F5 检出 0 框 (全空漏检) | V24 稳健捕获清晰野生动物 animal (置信度 0.941)",
     "CatA_02_000367_f5_empty_v24_animal_094.png"),

    ("00000432", "【类别 A: F5 全空漏检 / V24 稳健挽救】",
     "F5 检出 0 框 (全空漏检) | V24 成功召回低空微小无人机 uav (置信度 0.912)",
     "CatA_03_00000432_f5_empty_v24_uav_091.png"),

    ("002586", "【类别 A: F5 全空漏检 / V24 稳健挽救】",
     "F5 检出 0 框 (全空漏检) | V24 准确捕获清晰道路指示牌 sign (置信度 0.910)",
     "CatA_04_002586_f5_empty_v24_sign_091.png"),

    ("00000297", "【类别 A: F5 全空漏检 / V24 稳健挽救】",
     "F5 检出 0 框 (全空漏检) | V24 双框稳健检出园区长椅 seat (置信度 0.824, 0.723)",
     "CatA_05_00000297_f5_empty_v24_seat_082.png"),

    ("00000299", "【类别 A: F5 全空漏检 / V24 稳健挽救】",
     "F5 检出 0 框 (全空漏检) | V24 双框捕获同系列长椅 seat (置信度 0.842, 0.781)",
     "CatA_06_00000299_f5_empty_v24_seat_084.png"),

    ("00000302", "【类别 A: F5 全空漏检 / V24 稳健挽救】",
     "F5 检出 0 框 (全空漏检) | V24 一举检出 4 个目标: 2x seat (0.83, 0.63), uav (0.62), ball (0.46)",
     "CatA_07_00000302_f5_empty_v24_4boxes.png"),

    ("002507", "【类别 A: F5 全空漏检 / V24 稳健挽救】",
     "F5 检出 0 框 (全空漏检) | V24 检出 3 个街景目标: 2x sign (0.85, 0.74), person (0.69)",
     "CatA_08_002507_f5_empty_v24_3boxes.png"),

    ("003331", "【类别 A: F5 全空漏检 / V24 稳健挽救】",
     "F5 检出 0 框 (全空漏检) | V24 捕获远景车辆 car (置信度 0.844)",
     "CatA_09_003331_f5_empty_v24_car_084.png"),

    ("000008_009_00000301", "【类别 A: F5 全空漏检 / V24 稳健挽救】",
     "F5 检出 0 框 (全空漏检) | V24 检出高置信度动物 animal (置信度 0.910)",
     "CatA_10_000008_009_f5_empty_v24_animal_091.png"),

    # ---------------- Category B: V24 Empty -> F5 Positive --------------
    ("000003_018_00000333", "【类别 B: V24 全空 / F5 孤立有检出】",
     "V24 检出 0 框 (判定为无真实前景) | F5 预测 1 个高分灯具 light (置信度 0.932)",
     "CatB_01_000003_v24_empty_f5_light_093.png"),

    ("00000422", "【类别 B: V24 全空 / F5 孤立有检出】",
     "V24 检出 0 框 | F5 预测 1 个临界低分弱无人机 uav (置信度 0.512，疑似噪点)",
     "CatB_02_00000422_v24_empty_f5_uav_051.png"),

    ("003360", "【类别 B: V24 全空 / F5 孤立有检出】",
     "V24 检出 0 框 | F5 预测 1 个标志牌 sign (置信度 0.753)",
     "CatB_03_003360_v24_empty_f5_sign_075.png"),

    ("003727", "【类别 B: V24 全空 / F5 孤立有检出】",
     "V24 检出 0 框 | F5 预测 1 个动物 animal (置信度 0.814)",
     "CatB_04_003727_v24_empty_f5_animal_081.png"),

    # ---------------- Category C: F5 Noise / Hallucination -> V24 Clean -
    ("002808", "【类别 C: F5 严重地表长椅幻觉与人群混叠 vs V24 纯净精细】",
     "F5 检出 9 框 (含整幅地面超巨型虚假长椅 w=0.998) | V24 检出 15 框 (消除地面幻觉，清晰解耦 8 个独立行人 + 6 自行车)",
     "CatC_01_002808_f5_ground_seat_vs_v24_clean_crowd.png"),

    ("shuming_96_00000120", "【类别 C: F5 背景光斑噪点虚警 vs V24 纯净聚焦】",
     "F5 检出 16 框 (含 8 处低分 light 背景光斑与行人重叠) | V24 收敛为 6 框 (纯净锁定 5 个核心行人与 1 张长椅)",
     "CatC_02_shuming96_f5_light_noise_vs_v24_clean.png"),

    ("000008_021_00000139", "【类别 C: F5 远景杂斑虚警过多 vs V24 高信噪比】",
     "F5 检出 19 框 (含大量 0.52~0.68 远景人影碎片) | V24 收敛为 6 框 (全部为主体动物与高价值目标，最高达 0.967)",
     "CatC_03_000008_021_f5_clutter_vs_v24_salient.png"),

    ("003418", "【类别 C: F5 细碎交通标志与路灯噪点 vs V24 纯净过滤】",
     "F5 检出 24 框 (含大量 0.51~0.66 低分边缘噪点) | V24 收敛为 15 框 (滤除 9 处树丛杂光与模糊招牌)",
     "CatC_04_003418_f5_traffic_clutter_vs_v24.png"),

    ("000153", "【类别 C: F5 远景细碎标志虚警 vs V24 纯净刚体】",
     "F5 检出 17 框 (含 8 个标志牌与 8 个路灯) | V24 收敛为 9 框 (稳定保留前景车辆与高清晰度灯具)",
     "CatC_05_000153_f5_fragment_signs_vs_v24.png"),

    # ---------------- Category D: V24 Small-Object / Cluster Decomposition
    ("001539", "【类别 D: 重叠目标解耦拆分对比】",
     "F5 检出 5 框 (将中心 2 只粘连重叠动物合为单个巨框 w=0.23) | V24 检出 10 框 (精准解耦为 2 只独立动物，并召回 5 只远景小羊)",
     "CatD_01_001539_animal_cluster_separation.png"),

    ("00000290", "【类别 D: 复合小目标高灵敏度召回】",
     "F5 仅检出 2 框 (漏检长椅与副无人机) | V24 检出 8 框 (完整捕获 3x uav, 2x seat, 2x car, 1x ball)",
     "CatD_02_00000290_v24_high_recall_uav_seat.png"),

    ("001016", "【类别 D: 复杂街道密集标志与非机动车解耦】",
     "F5 检出 7 框 (多处远景标志漏检) | V24 检出 13 框 (高频分支清晰解析密集标志群与并排自行车)",
     "CatD_03_001016_v24_sign_bike_cluster.png"),

    # ---------------- Category E: Class Confusion (IoU >= 0.50) ---------
    ("002014", "【类别 E: 语义判定分歧 (骑手 vs 自行车)】",
     "F5 偏向车辆外包围框判定为 bicycle | V24 准确识别上方骑行主体为 person",
     "CatE_01_002014_bicycle_vs_person.png"),

    ("001712", "【类别 E: 语义判定分歧 (远景低矮人影 vs 动物)】",
     "F5 判定为 animal | V24 判定为 person (弯腰农作人员)",
     "CatE_02_001712_animal_vs_person.png"),
]

rendered_files = []
for item in CASES:
    stem, cat_title, desc, fname = item
    try:
        p = render_panel(stem, cat_title, desc, fname)
        rendered_files.append((item, p))
    except Exception as e:
        print(f"[!] Error rendering {stem}: {e}")

print(f"\n[✔] Successfully generated {len(rendered_files)} detailed inspection panels in {OUT_DIR}")
