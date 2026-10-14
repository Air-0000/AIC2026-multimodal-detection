#!/usr/bin/env python3
"""
Generate Clear Large-Image Comparisons with Clean Originals and Target Crops
=============================================================================
For AIC2026 test set inspection:
- Top: High-resolution clean original image (100% untouched, zero boxes)
- Middle: Model detection result image (with class colors, boxes, confidences)
- Bottom: High-magnification zoomed-in crops of key objects to reveal what was actually framed!
"""

import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image

CLASS_INFO = {
    0: ("person", (0, 255, 0)),         # Green
    1: ("boat", (255, 200, 0)),         # Teal
    2: ("animal", (0, 165, 255)),       # Orange
    3: ("seat", (139, 69, 19)),         # Brown
    4: ("sign", (0, 140, 255)),         # Dark Orange
    5: ("bicycle", (255, 255, 0)),      # Cyan
    6: ("car", (255, 60, 60)),          # Blue
    7: ("ball", (0, 0, 255)),           # Red
    8: ("light", (0, 215, 255)),        # Gold/Yellow
    9: ("garbage can", (200, 50, 200)), # Purple
    10: ("uav", (255, 0, 255)),         # Magenta/Hot Pink
    11: ("tricycle", (50, 205, 50)),    # Lime Green
}


def load_boxes(txt_path, w, h, min_conf=0.15):
    if not os.path.exists(txt_path):
        return []
    boxes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh, conf = map(float, parts[1:6])
            except ValueError:
                continue
            if conf < min_conf:
                continue
            x1 = int(round((cx - bw / 2.0) * w))
            y1 = int(round((cy - bh / 2.0) * h))
            x2 = int(round((cx + bw / 2.0) * w))
            y2 = int(round((cy + bh / 2.0) * h))
            boxes.append({
                "cid": cid,
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
            })
    boxes.sort(key=lambda b: -b["conf"])

    # Class-wise NMS (IoU >= 0.55) to eliminate duplicate boxes on the same object
    kept = []
    for b in boxes:
        is_dup = False
        for kb in kept:
            if b["cid"] == kb["cid"]:
                b1, b2 = b["bbox"], kb["bbox"]
                xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
                if inter / max(union, 1e-6) >= 0.55:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(b)
    return kept


def draw_clean_annotated(orig_bgr, boxes, title=""):
    canvas = orig_bgr.copy()
    h, w = canvas.shape[:2]

    # Header
    hdr_h = 42
    hdr = np.zeros((hdr_h, w, 3), dtype=np.uint8)
    hdr[:] = (25, 25, 25)
    cv2.putText(hdr, title, (16, 29), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA)

    for b in boxes:
        cid = b["cid"]
        x1, y1, x2, y2 = b["bbox"]
        conf = b["conf"]
        cname, color = CLASS_INFO.get(cid, (f"cls{cid}", (200, 200, 200)))

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        thickness = 3 if cid in (7, 10) else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        badge_text = f"{cname} {conf:.2f}"
        font_scale = 0.50
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        by1 = max(0, y1 - th - 6)
        by2 = y1
        bx1 = x1
        bx2 = min(w, x1 + tw + 6)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, -1)
        text_color = (0, 0, 0) if (color[0]*0.114 + color[1]*0.587 + color[2]*0.299) > 120 else (255, 255, 255)
        cv2.putText(canvas, badge_text, (bx1 + 3, by2 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

    return np.vstack([hdr, canvas])


def extract_crops_panel(orig_bgr, boxes, total_width=1280, max_crops=4):
    """
    Extracts zoomed-in crops of top/interesting objects and tiles them horizontally.
    """
    h, w = orig_bgr.shape[:2]
    if not boxes:
        return None

    # Pick representative crops (prefer UAV, Ball, or top scoring targets)
    selected = []
    # 1. UAV or Ball first
    for b in boxes:
        if b["cid"] in (7, 10) and len(selected) < 2:
            selected.append(b)
    # 2. Add other distinct classes
    seen_cids = {b["cid"] for b in selected}
    for b in boxes:
        if len(selected) >= max_crops:
            break
        if b["cid"] not in seen_cids:
            selected.append(b)
            seen_cids.add(b["cid"])
    # 3. Fill up to max_crops
    for b in boxes:
        if len(selected) >= max_crops:
            break
        if b not in selected:
            selected.append(b)

    crop_h = 240
    crop_w = total_width // max_crops
    crops_imgs = []

    for idx, b in enumerate(selected):
        cid = b["cid"]
        x1, y1, x2, y2 = b["bbox"]
        cname, color = CLASS_INFO.get(cid, ("obj", (200, 200, 200)))
        conf = b["conf"]

        # Expand box by 25% for surrounding context
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        exp_w = int(max(bw * 1.5, 40))
        exp_h = int(max(bh * 1.5, 40))

        ex1 = max(0, cx - exp_w // 2)
        ey1 = max(0, cy - exp_h // 2)
        ex2 = min(w, cx + exp_w // 2)
        ey2 = min(h, cy + exp_h // 2)

        crop = orig_bgr[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            continue

        # Draw red border on target in crop
        tx1 = max(0, x1 - ex1)
        ty1 = max(0, y1 - ey1)
        tx2 = min(crop.shape[1], x2 - ex1)
        ty2 = min(crop.shape[0], y2 - ey1)
        crop_annot = crop.copy()
        cv2.rectangle(crop_annot, (tx1, ty1), (tx2, ty2), color, 2)

        # Resize to standard tile size
        crop_tile = cv2.resize(crop_annot, (crop_w, crop_h - 40), interpolation=cv2.INTER_LANCZOS4)

        # Label banner below crop
        banner = np.zeros((40, crop_w, 3), dtype=np.uint8)
        banner[:] = (40, 40, 40)
        desc = f"#{idx+1} {cname} {conf:.2f} ({bw}x{bh}px)"
        cv2.putText(banner, desc, (8, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)
        tile_full = np.vstack([crop_tile, banner])
        crops_imgs.append(tile_full)

    if not crops_imgs:
        return None

    # Pad if fewer than max_crops
    while len(crops_imgs) < max_crops:
        empty = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        crops_imgs.append(empty)

    crop_row = np.hstack(crops_imgs)
    # Add title for crop row
    title_bar = np.zeros((32, total_width, 3), dtype=np.uint8)
    title_bar[:] = (20, 20, 20)
    cv2.putText(title_bar, "KEY OBJECT CROPS ZOOMED-IN (局部高倍放大特写，看清框内真实物体)", (16, 22), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([title_bar, crop_row])


def build_large_comparison(stem: str, img_dir: str, sub_dir: str, out_dir: str, target_width: int = 1440):
    img_path = os.path.join(img_dir, f"{stem}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(img_dir, f"{stem}.png")
    if not os.path.exists(img_path):
        print(f"Not found: {stem}")
        return None

    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        return None
    H, W = orig_bgr.shape[:2]

    txt_path = os.path.join(sub_dir, f"{stem}.txt")
    boxes = load_boxes(txt_path, W, H, min_conf=0.15)

    # 1. Clean Original Image Header & Canvas
    hdr_orig = np.zeros((42, W, 3), dtype=np.uint8)
    hdr_orig[:] = (25, 25, 25)
    cv2.putText(hdr_orig, f"[1] CLEAN ORIGINAL IMAGE (100% 干净原图，无任何框干扰，方便肉眼看清场景细节) - {stem}", (16, 29), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
    p_orig = np.vstack([hdr_orig, orig_bgr])

    # 2. Annotated Image
    p_annot = draw_clean_annotated(orig_bgr, boxes, title=f"[2] V18 MODEL DETECTIONS (模型目标检测与定位，共检测到 {len(boxes)} 个有效目标) - {stem}")

    # Scale both to target_width
    target_h = int(round(p_orig.shape[0] * (target_width / W)))
    p_orig_scaled = cv2.resize(p_orig, (target_width, target_h), interpolation=cv2.INTER_LANCZOS4)
    p_annot_scaled = cv2.resize(p_annot, (target_width, target_h), interpolation=cv2.INTER_LANCZOS4)

    # 3. Zoomed-in Crops
    p_crops = extract_crops_panel(orig_bgr, boxes, total_width=target_width, max_crops=4)

    parts = [p_orig_scaled, p_annot_scaled]
    if p_crops is not None:
        parts.append(p_crops)

    full_composite = np.vstack(parts)
    out_file = os.path.join(out_dir, f"detail_{stem}.jpg")
    cv2.imwrite(out_file, full_composite, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"[+] Saved High-Res Detail Comparison: {out_file} ({full_composite.shape[1]}x{full_composite.shape[0]})")
    return out_file


def main():
    root = "/home/constant/CodeBuddy/Dev/AIC"
    img_dir = os.path.join(root, "dataset/test/AIC2026_PHASE_2_1000/visible")
    sub_dir = os.path.join(root, "work_dir/submissions/submission_puremal_aligned_top35/predictions")
    out_dir = os.path.join(root, "work_dir/audits/vis_test_predictions")
    artifact_dir = "/home/constant/.gemini/antigravity/brain/0e8de2bf-ddee-429b-bd88-8d0434ba0e1d"
    os.makedirs(out_dir, exist_ok=True)

    test_stems = [
        "00000010",  # Plaza scene with park bench vs UAV
        "001097",    # Fisherman by lake with round container/ball
        "002239",    # Zoo bears and observation deck
        "003153",    # Flamingo flock in greenhouse
        "00000020",  # Complex lighting / poles
        "003740",    # Primate rock mountain with waterfall
    ]

    import shutil
    for s in test_stems:
        f = build_large_comparison(s, img_dir, sub_dir, out_dir, target_width=1440)
        if f:
            shutil.copy2(f, os.path.join(artifact_dir, os.path.basename(f)))


if __name__ == "__main__":
    main()
