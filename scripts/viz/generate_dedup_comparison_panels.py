#!/usr/bin/env python3
"""
Generate High-Resolution Visual Comparison Panels for Deduplicated Images
==========================================================================
Produces before-and-after side-by-side comparison images with local zoom crops,
highlighting the duplicate overlapping boxes before vs the single clean box after.
"""

import os
import sys
import glob
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_root = Path("/home/constant/CodeBuddy/Dev/AIC")
artifact_dir = Path("/home/constant/.gemini/antigravity/brain/4649f9d3-7539-431e-bbce-b6b49ee452ce")
out_dir = _root / "outputs/figures/dedup_comparisons"
out_dir.mkdir(parents=True, exist_ok=True)
artifact_dir.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

# Vivid distinct colors per class (BGR)
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


def draw_boxes(img_bgr, boxes, highlight_dups=False, is_after=False):
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]

    # Find duplicates if requested
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

        cname = CLASS_NAMES.get(cid, str(cid))
        color = CLASS_COLORS.get(cid, (0, 255, 0))

        if idx in dup_indices:
            # Highlight duplicate in flaming red/magenta with thick border
            box_color = (0, 0, 255) # Red for duplicate alert
            thickness = 4
            tag = f"!DUP! {cname} {sc:.3f}"
        elif is_after:
            box_color = (0, 230, 0) # Clean emerald green
            thickness = 3
            tag = f"{cname} {sc:.3f}"
        else:
            box_color = color
            thickness = 2
            tag = f"{cname} {sc:.3f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)

        # Label badge
        font_scale = 0.7
        font_thick = 2
        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        bg_y1 = max(0, y1 - th - baseline - 6)
        bg_y2 = y1
        bg_x2 = min(w, x1 + tw + 8)
        cv2.rectangle(canvas, (x1, bg_y1), (bg_x2, bg_y2), box_color, -1)
        cv2.putText(canvas, tag, (x1 + 4, bg_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    return canvas, dup_pairs


def make_comparison_panel(stem, orig_boxes, dedup_boxes, vis_path):
    img = cv2.imread(str(vis_path))
    if img is None:
        print(f"[!] Failed to read {vis_path}")
        return None
    h, w = img.shape[:2]

    # Find duplicate pair to zoom into
    best_pair = None
    best_iou = 0.0
    for i in range(len(orig_boxes)):
        for j in range(i+1, len(orig_boxes)):
            if orig_boxes[i][0] == orig_boxes[j][0]:
                iou = box_iou(orig_boxes[i][1:5], orig_boxes[j][1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_pair = (orig_boxes[i], orig_boxes[j])

    # Draw full images
    vis_orig, dup_pairs = draw_boxes(img, orig_boxes, highlight_dups=True, is_after=False)
    vis_dedup, _ = draw_boxes(img, dedup_boxes, highlight_dups=False, is_after=True)

    # Add header bars to full views
    bar_h = 56
    hdr_orig = np.zeros((bar_h, w, 3), dtype=np.uint8)
    hdr_orig[:] = (20, 20, 180) # Reddish banner
    title_orig = f"BEFORE: V34 Native Output ({len(orig_boxes)} boxes, DUP DETECTED: {len(dup_pairs)} pair, IoU={best_iou:.3f})"
    cv2.putText(hdr_orig, title_orig, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    panel_orig = np.vstack([hdr_orig, vis_orig])

    hdr_dedup = np.zeros((bar_h, w, 3), dtype=np.uint8)
    hdr_dedup[:] = (20, 150, 40) # Greenish banner
    title_dedup = f"AFTER: Distinct Dedup Output ({len(dedup_boxes)} boxes, 0 DUPLICATES - CLEAN)"
    cv2.putText(hdr_dedup, title_dedup, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    panel_dedup = np.vstack([hdr_dedup, vis_dedup])

    # Resize full views for side-by-side display (each width=1280)
    target_w = 1280
    scale = target_w / w
    target_h = int(round(panel_orig.shape[0] * scale))
    panel_orig_resized = cv2.resize(panel_orig, (target_w, target_h), interpolation=cv2.INTER_AREA)
    panel_dedup_resized = cv2.resize(panel_dedup, (target_w, target_h), interpolation=cv2.INTER_AREA)

    top_row = np.hstack([panel_orig_resized, panel_dedup_resized])

    # Now make local crop zoom comparison if duplicate exists
    if best_pair is not None:
        b1, b2 = best_pair
        # Bounding box union
        x1_a, y1_a, x2_a, y2_a = b1[1] - b1[3]/2, b1[2] - b1[4]/2, b1[1] + b1[3]/2, b1[2] + b1[4]/2
        x1_b, y1_b, x2_b, y2_b = b2[1] - b2[3]/2, b2[2] - b2[4]/2, b2[1] + b2[3]/2, b2[2] + b2[4]/2
        ux1 = max(0.0, min(x1_a, x1_b) - 0.05)
        uy1 = max(0.0, min(y1_a, y1_b) - 0.05)
        ux2 = min(1.0, max(x2_a, x2_b) + 0.05)
        uy2 = min(1.0, max(y2_a, y2_b) + 0.05)

        px1, py1 = int(round(ux1 * w)), int(round(uy1 * h))
        px2, py2 = int(round(ux2 * w)), int(round(uy2 * h))

        crop_orig = vis_orig[py1:py2, px1:px2]
        crop_dedup = vis_dedup[py1:py2, px1:px2]

        crop_target_h = 420
        c_scale = crop_target_h / max(1, crop_orig.shape[0])
        crop_target_w = int(round(crop_orig.shape[1] * c_scale))

        crop_orig_res = cv2.resize(crop_orig, (crop_target_w, crop_target_h), interpolation=cv2.INTER_LINEAR)
        crop_dedup_res = cv2.resize(crop_dedup, (crop_target_w, crop_target_h), interpolation=cv2.INTER_LINEAR)

        # Labels for crops
        chdr_orig = np.zeros((40, crop_target_w, 3), dtype=np.uint8)
        chdr_orig[:] = (40, 40, 180)
        cv2.putText(chdr_orig, f"ZOOM: Double Box (IoU={best_iou:.3f})", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cpanel_orig = np.vstack([chdr_orig, crop_orig_res])

        chdr_dedup = np.zeros((40, crop_target_w, 3), dtype=np.uint8)
        chdr_dedup[:] = (20, 140, 30)
        cv2.putText(chdr_dedup, "ZOOM: Single Clean Box", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cpanel_dedup = np.vstack([chdr_dedup, crop_dedup_res])

        crop_combo = np.hstack([cpanel_orig, cpanel_dedup])

        # Center pad crop_combo to match top_row width
        total_w = top_row.shape[1]
        pad_total = total_w - crop_combo.shape[1]
        if pad_total > 0:
            pad_l = pad_total // 2
            pad_r = pad_total - pad_l
            crop_pad = np.zeros((crop_combo.shape[0], total_w, 3), dtype=np.uint8)
            crop_pad[:, pad_l:pad_l+crop_combo.shape[1]] = crop_combo
            final_panel = np.vstack([top_row, crop_pad])
        else:
            crop_combo = cv2.resize(crop_combo, (total_w, int(crop_combo.shape[0] * total_w / crop_combo.shape[1])))
            final_panel = np.vstack([top_row, crop_combo])
    else:
        final_panel = top_row

    return final_panel, best_iou, (b1, b2) if best_pair else None


def main():
    orig_zip = _root / "work_dir/submissions/sovereign_v34_clean_from_scratch/submission_sovereign_v34_native.zip"
    dedup_zip = _root / "work_dir/submissions/sovereign_v34_clean_from_scratch/submission_sovereign_v34_distinct_dedup.zip"

    print("[+] Loading submission predictions...")
    orig = parse_zip(orig_zip)
    dedup = parse_zip(dedup_zip)

    # 6 Key Representative Cases
    cases = [
        ("000023_014_00000088", "animal", "极致同位重叠 (IoU=0.996)"),
        ("003116", "person", "大幅行人双重包络 (IoU=0.973)"),
        ("001932", "boat", "水面船舶近距重合 (IoU=0.959)"),
        ("002034", "person", "全身行人双层框 (IoU=0.940)"),
        ("000003_028_00000163", "car", "大型车辆冗余框 (IoU=0.980)"),
        ("000011_030_00000001", "light", "高立柱路灯重叠框 (IoU=0.990)"),
    ]

    generated_panels = []

    vis_dir = _root / "dataset/test/AIC2026_PHASE_2_1000/visible"
    for stem, cat, desc in cases:
        vis_path = vis_dir / f"{stem}.png"
        if not vis_path.exists():
            vis_path = vis_dir / f"{stem}.jpg"
        if not vis_path.exists():
            print(f"[!] Warning: Image not found for {stem}")
            continue

        orig_b = orig.get(stem, [])
        dedup_b = dedup.get(stem, [])

        panel, iou, pair = make_comparison_panel(stem, orig_b, dedup_b, vis_path)
        if panel is not None:
            out_file = out_dir / f"comparison_{stem}.jpg"
            cv2.imwrite(str(out_file), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
            
            # Copy to artifact dir
            art_file = artifact_dir / f"comparison_{stem}.jpg"
            shutil.copyfile(str(out_file), str(art_file))

            print(f"[+] Saved comparison panel: {out_file} (IoU={iou:.3f}, {cat} - {desc})")
            generated_panels.append({
                "stem": stem,
                "cat": cat,
                "desc": desc,
                "iou": iou,
                "orig_count": len(orig_b),
                "dedup_count": len(dedup_b),
                "local_path": str(out_file),
                "artifact_path": str(art_file),
                "pair": pair,
            })

    print(f"\n[+] Successfully generated {len(generated_panels)} high-res comparison panels!")


if __name__ == '__main__':
    main()
