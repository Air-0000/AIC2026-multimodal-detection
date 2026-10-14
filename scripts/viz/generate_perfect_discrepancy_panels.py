#!/usr/bin/env python3
"""
Generate High-Resolution Discrepancy & Zoom Inspection Panels (V25 vs F5 & V24)
================================================================================
Draws:
  - Sovereign V25 predictions in Crimson Red (#FF2A2A)
  - F5 Official Benchmark predictions in Cyan (#00B4E6)
  - Full image overview + High-resolution zoomed insets on key discrepancy ROIs.
  - Compares V25 model repairs (SDQS consolidation, ambient recall, 0-box rescues).
"""

import os
import sys
import glob
import zipfile
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path("/home/constant/CodeBuddy/Dev/AIC")
DATA_DIR = ROOT / "dataset/test/AIC2026_PHASE_2_1000/visible"
F5_ZIP = ROOT / "work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip"
V24_ZIP = ROOT / "outputs/submissions/submissions_v24/submission_v24_midboost_compact.zip"
V25_ZIP = ROOT / "outputs/submissions/submissions_v25/submission_sovereign_f5_perfect_union.zip"
OUT_DIR = ROOT / "outputs/analysis/visualizations_discrepancy_perfect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}

# Colors (RGB)
COLOR_V25 = (230, 40, 40)       # Crimson Red
HEX_V25 = "#E62828"
COLOR_F5 = (0, 180, 230)        # Electric Cyan
HEX_F5 = "#00B4E6"
COLOR_V24 = (255, 165, 0)       # Amber Orange
HEX_V24 = "#FFA500"
COLOR_ROI = (255, 215, 0)       # Gold
HEX_ROI = "#FFD700"

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'LXGW WenKai', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def read_zip(p):
    res = {}
    if not os.path.exists(p):
        return res
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


def determine_rois(boxes_v25, boxes_f5, W, H, max_rois=2):
    unmatched_v25 = []
    for b_v in boxes_v25:
        matched = any(box_iou(b_v, b_f) >= 0.40 and b_v[0] == b_f[0] for b_f in boxes_f5)
        if not matched:
            unmatched_v25.append(b_v)

    unmatched_f5 = []
    for b_f in boxes_f5:
        matched = any(box_iou(b_f, b_v) >= 0.40 and b_f[0] == b_v[0] for b_v in boxes_v25)
        if not matched:
            unmatched_f5.append(b_f)

    rois = []
    candidates = []

    for b in unmatched_v25:
        candidates.append((b, f"V25 检出但 F5 缺失: {CLASS_NAMES[b[0]]} (置信度 {b[5]:.2f})", "v25_only"))
    for b in unmatched_f5:
        candidates.append((b, f"F5 检出但 V25 排除: {CLASS_NAMES[b[0]]} (置信度 {b[5]:.2f})", "f5_only"))

    if not candidates:
        all_boxes = boxes_v25 + boxes_f5
        if all_boxes:
            b = max(all_boxes, key=lambda x: x[5])
            candidates.append((b, f"共识目标: {CLASS_NAMES[b[0]]} (置信度 {b[5]:.2f})", "consensus"))

    candidates.sort(key=lambda x: (x[0][3] * x[0][4], x[0][5]), reverse=True)

    for cand_box, desc, ctype in candidates:
        if len(rois) >= max_rois:
            break
        cid, cx, cy, bw, bh, sc = cand_box
        bx1 = (cx - bw/2.0) * W
        by1 = (cy - bh/2.0) * H
        bx2 = (cx + bw/2.0) * W
        by2 = (cy + bh/2.0) * H

        pad_w = max(bw * W * 1.5, 120.0)
        pad_h = max(bh * H * 1.5, 120.0)
        rx1 = max(0, int(bx1 - pad_w))
        ry1 = max(0, int(by1 - pad_h))
        rx2 = min(W, int(bx2 + pad_w))
        ry2 = min(H, int(by2 + pad_h))

        overlap = False
        for er_x1, er_y1, er_x2, er_y2, _ in rois:
            if not (rx2 < er_x1 or rx1 > er_x2 or ry2 < er_y1 or ry1 > er_y2):
                overlap = True
                break
        if not overlap:
            rois.append((rx1, ry1, rx2, ry2, desc))

    if not rois:
        rois.append((int(W*0.25), int(H*0.25), int(W*0.75), int(H*0.75), "全景中心区域细节"))
    return rois


def render_panel(stem, category_title, case_desc, out_filename, v25_data, f5_data, v24_data=None):
    img = load_image(stem)
    W, H = img.size

    b_v25 = v25_data.get(stem, [])
    b_f5 = f5_data.get(stem, [])
    b_v24 = v24_data.get(stem, []) if v24_data else []

    rois = determine_rois(b_v25, b_f5, W, H, max_rois=2)

    fig = plt.figure(figsize=(20, 10.5), dpi=220)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0],
                          wspace=0.12, hspace=0.18, left=0.04, right=0.97, top=0.91, bottom=0.05)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(img)
    ax_main.set_title(f"全局全景视图 (1920×1080) — 红色: Sovereign V25 ({len(b_v25)}框) | 青色: F5 ({len(b_f5)}框) | 前代V24 ({len(b_v24)}框)",
                      fontsize=12, fontweight='bold', pad=10)
    ax_main.axis('off')

    # Draw F5 boxes (Cyan)
    for b in b_f5:
        cid, cx, cy, bw, bh, sc = b
        x1 = (cx - bw/2.0) * W
        y1 = (cy - bh/2.0) * H
        w = bw * W
        h = bh * H
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2.0, edgecolor=HEX_F5, facecolor='none', linestyle='-')
        ax_main.add_patch(rect)
        ax_main.text(x1, max(12, y1 - 4), f"F5:{CLASS_NAMES[cid]} {sc:.2f}",
                     fontsize=7, fontweight='bold', color='black',
                     bbox=dict(boxstyle='square,pad=0.15', facecolor=HEX_F5, edgecolor='none', alpha=0.85))

    # Draw V25 boxes (Crimson Red)
    for b in b_v25:
        cid, cx, cy, bw, bh, sc = b
        x1 = (cx - bw/2.0) * W
        y1 = (cy - bh/2.0) * H
        w = bw * W
        h = bh * H
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2.5, edgecolor=HEX_V25, facecolor='none', linestyle='-')
        ax_main.add_patch(rect)
        ax_main.text(x1 + w, min(H - 4, y1 + h + 10), f"V25:{CLASS_NAMES[cid]} {sc:.2f}",
                     fontsize=7.5, fontweight='bold', color='white',
                     bbox=dict(boxstyle='square,pad=0.15', facecolor=HEX_V25, edgecolor='none', alpha=0.92))

    # Draw ROI indicator boxes
    for i, (rx1, ry1, rx2, ry2, rdesc) in enumerate(rois):
        roi_rect = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1,
                                     linewidth=2.8, edgecolor=HEX_ROI, facecolor='none', linestyle='--')
        ax_main.add_patch(roi_rect)
        ax_main.text(rx1 + 8, ry1 + 22, f"放大区域 #{i+1}",
                     fontsize=9, fontweight='bold', color='black',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor=HEX_ROI, edgecolor='black', alpha=0.95))

    # Zoomed Insets
    for i in range(2):
        ax_zoom = fig.add_subplot(gs[i, 1])
        if i < len(rois):
            rx1, ry1, rx2, ry2, rdesc = rois[i]
            crop_img = img.crop((rx1, ry1, rx2, ry2))
            crop_w = rx2 - rx1
            crop_h = ry2 - ry1

            ax_zoom.imshow(crop_img)
            ax_zoom.set_title(f"【细节放大 #{i+1}】: {rdesc}",
                             fontsize=11, fontweight='bold', color='#1D3557', pad=8)
            ax_zoom.axis('off')

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

            for b in b_v25:
                cid, cx, cy, bw, bh, sc = b
                bx1 = (cx - bw/2.0) * W - rx1
                by1 = (cy - bh/2.0) * H - ry1
                bw_px = bw * W
                bh_px = bh * H
                if bx1 + bw_px > 0 and bx1 < crop_w and by1 + bh_px > 0 and by1 < crop_h:
                    rect = patches.Rectangle((bx1, by1), bw_px, bh_px, linewidth=3.2,
                                             edgecolor=HEX_V25, facecolor='none', linestyle='-')
                    ax_zoom.add_patch(rect)
                    ax_zoom.text(bx1, min(crop_h - 6, by1 + bh_px + 14), f"V25: {CLASS_NAMES[cid]} {sc:.2f}",
                                 fontsize=9, fontweight='bold', color='white',
                                 bbox=dict(boxstyle='square,pad=0.2', facecolor=HEX_V25, edgecolor='black', alpha=0.95))
        else:
            ax_zoom.axis('off')
            ax_zoom.text(0.5, 0.5, "（该图像仅包含单处显著差异区域）",
                         fontsize=12, color='#6C757D', ha='center', va='center')

    fig.suptitle(f"{category_title} | 图像样本: {stem}\n{case_desc}",
                 fontsize=14, fontweight='bold', color='#111111', y=0.97)

    save_path = OUT_DIR / out_filename
    plt.savefig(save_path, bbox_inches='tight', dpi=220)
    plt.close()
    print(f"[+] Rendered: {out_filename}")
    return save_path


def main():
    print("=" * 80)
    print(" Sovereign V25 vs F5 Discrepancy Panel Generator ".center(80, "="))
    print("=" * 80)
    v25_data = read_zip(V25_ZIP)
    f5_data = read_zip(F5_ZIP)
    v24_data = read_zip(V24_ZIP)

    CASES = [
        # Cat A: 0-box rescues
        ("000351", "【类别 A: 0-Box 死图挽救】", "F5 0框 | V25 稳健检出高分目标", "CatA_01_000351_v25_rescued.png"),
        ("000367", "【类别 A: 0-Box 死图挽救】", "F5 0框 | V25 稳健检出高分野生动物", "CatA_02_000367_v25_rescued.png"),
        ("00000432", "【类别 A: 0-Box 死图挽救】", "F5 0框 | V25 稳健检出微小无人机", "CatA_03_00000432_v25_rescued.png"),
        ("002586", "【类别 A: 0-Box 死图挽救】", "F5 0框 | V25 准确捕获清晰道路指示牌 sign", "CatA_04_002586_v25_rescued.png"),
        ("00000297", "【类别 A: 0-Box 死图挽救】", "F5 0框 | V25 检出园区长椅 seat", "CatA_05_00000297_v25_rescued.png"),
        ("00000299", "【类别 A: 0-Box 死图挽救】", "F5 0框 | V25 捕获同系列长椅 seat", "CatA_06_00000299_v25_rescued.png"),
        ("00000302", "【类别 A: 0-Box 死图挽救】", "F5 0框 | V25 检出多类别复合目标", "CatA_07_00000302_v25_rescued.png"),
        ("002507", "【类别 A: 0-Box 死图挽救】", "F5 0框 | V25 检出街景目标 sign/person", "CatA_08_002507_v25_rescued.png"),
        ("003331", "【类别 A: 0-Box 死图挽救】", "F5 0框 | V25 捕获远景车辆 car", "CatA_09_003331_v25_rescued.png"),
        ("000008_009_00000301", "【类别 A: 0-Box 死图挽救】", "F5 0框 | V25 检出高置信度动物 animal", "CatA_10_000008_009_v25_rescued.png"),

        # Cat B: Ambient targets (light, sign, seat)
        ("000003_018_00000333", "【类别 B: 环境设施灯具 light 召回】", "V24 漏检 | V25 软化负样本后召回真实灯具", "CatB_01_000003_v25_light_recall.png"),
        ("00000422", "【类别 B: 弱小无人机 uav 探测】", "F5 低分弱检出 vs V25 探测", "CatB_02_00000422_v25_uav.png"),
        ("003360", "【类别 B: 标志牌 sign 召回】", "V24 漏检 vs V25 标志牌召回", "CatB_03_003360_v25_sign_recall.png"),
        ("003727", "【类别 B: 暗光动物 animal 召回】", "MidBoost 自适应增益下动物召回", "CatB_04_003727_v25_animal_recall.png"),

        # Cat C: Noise / Hallucination elimination
        ("002808", "【类别 C: 地表长椅幻觉消除与真实目标解耦】", "F5 地面超巨虚假长椅 vs V25 精准人群解耦", "CatC_01_002808_v25_clean_seat.png"),
        ("shuming_96_00000120", "【类别 C: 背景光斑噪点虚警 vs 真实前景】", "F5 8处杂斑 vs V25 纯净锁定核心行人", "CatC_02_shuming96_v25_clean_light.png"),
        ("000008_021_00000139", "【类别 C: 远景杂斑虚警 vs 高信噪比】", "F5 远景人影碎片 vs V25 纯净主体", "CatC_03_000008_021_v25_salient.png"),
        ("003418", "【类别 C: 细碎交通标志与路灯噪点过滤】", "F5 树丛杂光 vs V25 纯净刚体", "CatC_04_003418_v25_traffic_clutter.png"),
        ("000153", "【类别 C: 远景碎片 vs 纯净刚体】", "F5 碎框虚警 vs V25 稳定刚体", "CatC_05_000153_v25_clean.png"),

        # Cat D: SDQS & NDSL Cluster Consolidation ("一物多框" 根除验证)
        ("001539", "【类别 D: 空间解耦选点 SDQS 根除冗余多胞胎框】", "检验密集动物粘连群是否由冗余双框收敛为单框GT", "CatD_01_001539_v25_sdqs_cluster_repair.png"),
        ("00000290", "【类别 D: 复合小目标高灵敏度召回】", "完整捕获 uav, seat, car, ball", "CatD_02_00000290_v25_multi_object.png"),
        ("001016", "【类别 D: 复杂街道密集标志与非机动车】", "解析密集标志群与并排自行车", "CatD_03_001016_v25_dense_street.png"),

        # Cat E: Semantic discrimination
        ("002014", "【类别 E: 语义判定 (骑手 person vs bicycle)】", "高语义层级解析骑行主体", "CatE_01_002014_v25_rider_vs_bike.png"),
        ("001712", "【类别 E: 语义判定 (农作人员 person vs animal)】", "低矮人影语义解析", "CatE_02_001712_v25_worker_vs_animal.png"),
    ]

    for stem, cat_title, desc, fname in CASES:
        try:
            render_panel(stem, cat_title, desc, fname, v25_data, f5_data, v24_data)
        except Exception as e:
            print(f"[!] Error rendering {stem}: {e}")

    print(f"\n[✔] Finished rendering panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
