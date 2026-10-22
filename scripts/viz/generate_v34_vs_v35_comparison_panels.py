#!/usr/bin/env python3
"""
Generate High-Resolution Visual Comparison Panels: V34 (No SDQS) vs V35 (With SDQS)
=====================================================================================
Generates side-by-side comparison images with local zoom crops comparing V34 Native
against V35 Native on the Phase 2 test set case images.
"""

import os
import sys
import glob
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_root = Path("/home/constant/CodeBuddy/Dev/AIC")
artifact_dir = Path("/home/constant/.gemini/antigravity/brain/4649f9d3-7539-431e-bbce-b6b49ee452ce")
out_dir = _root / "outputs/figures/v34_vs_v35_comparisons"
out_dir.mkdir(parents=True, exist_ok=True)
artifact_dir.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

# Distinct BGR colors
CLASS_COLORS = {
    0: (0, 165, 255),    # person: orange
    1: (255, 191, 0),    # boat: deep sky blue
    2: (0, 255, 127),    # animal: spring green
    3: (238, 130, 238),  # seat: violet
    4: (255, 255, 0),    # sign: cyan
    5: (0, 255, 255),    # bicycle: yellow
    6: (255, 0, 0),      # car: blue
    7: (0, 0, 255),      # ball: red
    8: (255, 20, 147),   # light: deep pink
    9: (128, 0, 128),    # garbage can: purple
    10: (0, 215, 255),   # uav: gold
    11: (50, 205, 50),   # tricycle: lime green
}


def parse_zip(zip_path):
    res = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith('.txt'):
                stem = name[:-4]
                lines = zf.read(name).decode('utf-8').strip().split('\n')
                boxes = []
                for l in lines:
                    p = l.strip().split()
                    if len(p) >= 6:
                        boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
                res[stem] = boxes
    return res


def box_iou(b1, b2):
    x1_a, y1_a, x2_a, y2_a = b1[0] - b1[2]/2, b1[1] - b1[3]/2, b1[0] + b1[2]/2, b1[1] + b1[3]/2
    x1_b, y1_b, x2_b, y2_b = b2[0] - b2[2]/2, b2[1] - b2[3]/2, b2[0] + b2[2]/2, b2[1] + b2[3]/2
    ix1, iy1 = max(x1_a, x1_b), max(y1_a, y1_b)
    ix2, iy2 = min(x2_a, x2_b), min(y2_a, y2_b)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


def draw_boxes(img_bgr, boxes, highlight_dups=True):
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]

    # Find duplicates
    dup_indices = set()
    dup_pairs = []
    if highlight_dups:
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                if boxes[i][0] == boxes[j][0]:
                    iou = box_iou(boxes[i][1:5], boxes[j][1:5])
                    if iou >= 0.60:
                        dup_indices.add(i)
                        dup_indices.add(j)
                        dup_pairs.append((i, j, iou))

    for idx, b in enumerate(boxes):
        cid, cx, cy, bw, bh, sc = b
        x1 = int(round((cx - bw/2.0) * w))
        y1 = int(round((cy - bh/2.0) * h))
        x2 = int(round((cx + bw/2.0) * w))
        y2 = int(round((cy + bh/2.0) * h))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)

        is_dup = idx in dup_indices
        if is_dup:
            color = (0, 0, 255) # Bright red for duplicate boxes
            thick = 3
        else:
            color = CLASS_COLORS.get(cid, (0, 255, 0))
            thick = 2

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)

        # Label background
        cname = CLASS_NAMES.get(cid, str(cid))
        lbl_text = f"{cname} {sc:.2f}"
        if is_dup:
            lbl_text += " [DUP]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        (lw, lh), baseline = cv2.getTextSize(lbl_text, font, scale, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - lh - 6)), (x1 + lw + 4, max(lh + 6, y1)), color, -1)
        text_color = (255, 255, 255) if is_dup else (0, 0, 0)
        cv2.putText(canvas, lbl_text, (x1 + 2, max(lh + 2, y1 - 3)), font, scale, text_color, 1, cv2.LINE_AA)

    return canvas, dup_pairs


def make_comparison_panel(stem, v34_boxes, v35_boxes, vis_path):
    img = cv2.imread(str(vis_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    # Find duplicate pair in V34 to zoom into
    best_pair = None
    best_iou = 0.0
    for i in range(len(v34_boxes)):
        for j in range(i+1, len(v34_boxes)):
            if v34_boxes[i][0] == v34_boxes[j][0]:
                iou = box_iou(v34_boxes[i][1:5], v34_boxes[j][1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_pair = (v34_boxes[i], v34_boxes[j])

    # Draw full images
    vis_v34, dups_v34 = draw_boxes(img, v34_boxes, highlight_dups=True)
    vis_v35, dups_v35 = draw_boxes(img, v35_boxes, highlight_dups=True)

    # Header bars
    bar_h = 56
    hdr_v34 = np.zeros((bar_h, w, 3), dtype=np.uint8)
    hdr_v34[:] = (20, 20, 180) # Reddish banner
    title_v34 = f"V34 Native (No SDQS): {len(v34_boxes)} boxes, {len(dups_v34)} duplicate pairs"
    cv2.putText(hdr_v34, title_v34, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    panel_v34 = np.vstack([hdr_v34, vis_v34])

    hdr_v35 = np.zeros((bar_h, w, 3), dtype=np.uint8)
    hdr_v35[:] = (20, 140, 40) if len(dups_v35) == 0 else (20, 100, 140)
    title_v35 = f"V35 Distinct-Query (With SDQS): {len(v35_boxes)} boxes, {len(dups_v35)} duplicate pairs"
    cv2.putText(hdr_v35, title_v35, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    panel_v35 = np.vstack([hdr_v35, vis_v35])

    # Resize full views for side-by-side display (each width=1280)
    target_w = 1280
    scale = target_w / w
    target_h = int(round(panel_v34.shape[0] * scale))
    panel_v34_res = cv2.resize(panel_v34, (target_w, target_h), interpolation=cv2.INTER_AREA)
    panel_v35_res = cv2.resize(panel_v35, (target_w, target_h), interpolation=cv2.INTER_AREA)

    top_row = np.hstack([panel_v34_res, panel_v35_res])

    # Local crop zoom comparison
    if best_pair is not None:
        b1, b2 = best_pair
        x1_a, y1_a, x2_a, y2_a = b1[1] - b1[3]/2, b1[2] - b1[4]/2, b1[1] + b1[3]/2, b1[2] + b1[4]/2
        x1_b, y1_b, x2_b, y2_b = b2[1] - b2[3]/2, b2[2] - b2[4]/2, b2[1] + b2[3]/2, b2[2] + b2[4]/2
        ux1 = max(0.0, min(x1_a, x1_b) - 0.05)
        uy1 = max(0.0, min(y1_a, y1_b) - 0.05)
        ux2 = min(1.0, max(x2_a, x2_b) + 0.05)
        uy2 = min(1.0, max(y2_a, y2_b) + 0.05)

        px1, py1 = int(round(ux1 * w)), int(round(uy1 * h))
        px2, py2 = int(round(ux2 * w)), int(round(uy2 * h))

        crop_v34 = vis_v34[py1:py2, px1:px2]
        crop_v35 = vis_v35[py1:py2, px1:px2]

        crop_target_h = 420
        c_scale = crop_target_h / max(1, crop_v34.shape[0])
        crop_target_w = int(round(crop_v34.shape[1] * c_scale))

        crop_v34_res = cv2.resize(crop_v34, (crop_target_w, crop_target_h), interpolation=cv2.INTER_LINEAR)
        crop_v35_res = cv2.resize(crop_v35, (crop_target_w, crop_target_h), interpolation=cv2.INTER_LINEAR)

        chdr_v34 = np.zeros((40, crop_target_w, 3), dtype=np.uint8)
        chdr_v34[:] = (40, 40, 180)
        cv2.putText(chdr_v34, f"V34 ZOOM: Duplicate Pair (IoU={best_iou:.3f})", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cpanel_v34 = np.vstack([chdr_v34, crop_v34_res])

        chdr_v35 = np.zeros((40, crop_target_w, 3), dtype=np.uint8)
        chdr_v35[:] = (20, 140, 30) if len(dups_v35) == 0 else (20, 100, 140)
        zoom_v35_title = "V35 ZOOM: Single Clean Box" if len(dups_v35) == 0 else "V35 ZOOM: SDQS Area"
        cv2.putText(chdr_v35, zoom_v35_title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cpanel_v35 = np.vstack([chdr_v35, crop_v35_res])

        crop_combo = np.hstack([cpanel_v34, cpanel_v35])

        total_w = top_row.shape[1]
        if crop_combo.shape[1] > total_w:
            cc_scale = total_w / crop_combo.shape[1]
            crop_combo = cv2.resize(crop_combo, (total_w, int(round(crop_combo.shape[0] * cc_scale))), interpolation=cv2.INTER_AREA)
        else:
            pad_total = total_w - crop_combo.shape[1]
            if pad_total > 0:
                pad_l = pad_total // 2
                pad_r = pad_total - pad_l
                crop_combo = cv2.copyMakeBorder(crop_combo, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(30, 30, 30))

        full_panel = np.vstack([top_row, crop_combo])
    else:
        full_panel = top_row

    return full_panel, best_iou, len(dups_v34), len(dups_v35)


def main():
    z34_path = _root / "work_dir/submissions/sovereign_v34_clean_from_scratch/submission_sovereign_v34_native.zip"
    z35_path = _root / "work_dir/submissions/sovereign_v35_distinct_queries/submission_sovereign_v35_native.zip"

    assert z34_path.exists(), f"V34 submission not found: {z34_path}"
    assert z35_path.exists(), f"V35 submission not found: {z35_path}"

    d34 = parse_zip(z34_path)
    d35 = parse_zip(z35_path)

    cases = [
        "000023_014_00000088",
        "003116",
        "001932",
        "002034",
        "000003_028_00000163",
        "000011_030_00000001",
    ]

    for stem in cases:
        vis_path = _root / f"dataset/test/AIC2026_PHASE_2_1000/visible/{stem}.png"
        if not vis_path.exists():
            vis_path = _root / f"dataset/test/AIC2026_PHASE_2_1000/visible/{stem}.jpg"
        if not vis_path.exists():
            print(f"[-] Image not found: {stem}")
            continue

        b34 = d34.get(stem, [])
        b35 = d35.get(stem, [])

        panel, iou, d34_cnt, d35_cnt = make_comparison_panel(stem, b34, b35, vis_path)
        if panel is not None:
            out_file = out_dir / f"comparison_{stem}.jpg"
            cv2.imwrite(str(out_file), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
            art_file = artifact_dir / f"comparison_{stem}.jpg"
            cv2.imwrite(str(art_file), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"[+] Rendered {stem}: V34({len(b34)}b, {d34_cnt}dups) vs V35({len(b35)}b, {d35_cnt}dups), max IoU={iou:.3f} -> {art_file.name}")


if __name__ == '__main__':
    main()
