#!/usr/bin/env python3
"""
Test Set Detections Visualizer for AIC2026 Multimodal Object Detection
======================================================================
Renders detection boxes on test set images comparing:
1. V18 原始预测 (Raw Unfiltered - showing multi-label phantoms & dense queries)
2. 0.20 置信度截断 (Threshold 0.20 - showing current remote 53.8-score predictions)
3. 精准对齐去重 (Clean Deduplicated Top-35 - showing true objects cleanly)
"""

import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image

# 12 Classes and specific distinctive colors (BGR)
CLASS_INFO = {
    0: ("person", (0, 255, 0)),         # Green
    1: ("boat", (255, 200, 0)),         # Teal / Cyan
    2: ("animal", (0, 165, 255)),       # Orange
    3: ("seat", (139, 69, 19)),         # Brown
    4: ("sign", (0, 140, 255)),         # Dark Orange
    5: ("bicycle", (255, 255, 0)),      # Cyan
    6: ("car", (255, 50, 50)),          # Dodger Blue
    7: ("ball", (0, 0, 255)),           # Red
    8: ("light", (0, 215, 255)),        # Gold / Yellow
    9: ("garbage can", (200, 50, 200)), # Purple
    10: ("uav", (255, 0, 255)),         # Hot Pink / Magenta
    11: ("tricycle", (50, 205, 50)),    # Lime Green
}


def load_txt_boxes(txt_path, img_w, img_h, min_conf=0.0):
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
                cx, cy, w, h, conf = map(float, parts[1:6])
            except ValueError:
                continue
            if conf < min_conf:
                continue

            x1 = int(round((cx - w / 2.0) * img_w))
            y1 = int(round((cy - h / 2.0) * img_h))
            x2 = int(round((cx + w / 2.0) * img_w))
            y2 = int(round((cy + h / 2.0) * img_h))
            boxes.append({
                "cid": cid,
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
            })
    # Sort descending
    boxes.sort(key=lambda b: -b["conf"])
    return boxes


def draw_boxes(image_bgr, boxes, title=""):
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    # Draw header bar
    header_h = 36
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    cv2.putText(header, title, (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw boxes
    for b in boxes:
        cid = b["cid"]
        x1, y1, x2, y2 = b["bbox"]
        conf = b["conf"]

        cname, color = CLASS_INFO.get(cid, (f"cls{cid}", (200, 200, 200)))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        # Thick border for UAV and Ball to make phantoms instantly visible
        thickness = 3 if cid in (7, 10) else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Label badge
        label_text = f"{cname} {conf:.2f}"
        font_scale = 0.45
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        bg_y1 = max(0, y1 - th - 6)
        bg_y2 = y1
        bg_x1 = x1
        bg_x2 = min(w, x1 + tw + 6)

        cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        # Choose text color (black or white for contrast)
        text_color = (0, 0, 0) if (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299) > 120 else (255, 255, 255)
        cv2.putText(canvas, label_text, (bg_x1 + 3, bg_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

    # Combine header and canvas
    return np.vstack([header, canvas])


def render_comparison_triplet(
    stem: str,
    test_img_dir: str,
    raw_sub_dir: str,
    clean_sub_dir: str,
    out_dir: str,
):
    img_path = os.path.join(test_img_dir, f"{stem}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(test_img_dir, f"{stem}.png")
    if not os.path.exists(img_path):
        print(f"[!] Test image not found: {stem}")
        return None

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    H, W = img_bgr.shape[:2]

    raw_txt = os.path.join(raw_sub_dir, f"{stem}.txt")
    clean_txt = os.path.join(clean_sub_dir, f"{stem}.txt")

    # 1. Raw All
    boxes_raw = load_txt_boxes(raw_txt, W, H, min_conf=0.0)
    # Count special classes in raw
    uav_raw = sum(1 for b in boxes_raw if b["cid"] == 10)
    ball_raw = sum(1 for b in boxes_raw if b["cid"] == 7)
    title_raw = f"[1] V18 Raw Unfiltered ({len(boxes_raw)} boxes, UAV:{uav_raw}, Ball:{ball_raw})"
    rendered_raw = draw_boxes(img_bgr, boxes_raw, title=title_raw)

    # 2. Raw Filtered 0.20
    boxes_thr20 = load_txt_boxes(raw_txt, W, H, min_conf=0.20)
    uav_t20 = sum(1 for b in boxes_thr20 if b["cid"] == 10)
    ball_t20 = sum(1 for b in boxes_thr20 if b["cid"] == 7)
    title_t20 = f"[2] Threshold 0.20 ({len(boxes_thr20)} boxes, UAV:{uav_t20}, Ball:{ball_t20})"
    rendered_t20 = draw_boxes(img_bgr, boxes_thr20, title=title_t20)

    # 3. Clean Deduplicated Top-35
    boxes_clean = load_txt_boxes(clean_txt, W, H, min_conf=0.0)
    uav_cln = sum(1 for b in boxes_clean if b["cid"] == 10)
    ball_cln = sum(1 for b in boxes_clean if b["cid"] == 7)
    title_clean = f"[3] Clean Top-35 Aligned ({len(boxes_clean)} boxes, UAV:{uav_cln}, Ball:{ball_cln})"
    rendered_clean = draw_boxes(img_bgr, boxes_clean, title=title_clean)

    # Stack side-by-side or vertical
    # For good inspection on screen, stack 3 panels vertically or side-by-side
    # If W=640, side-by-side of 3 is 1920 pixels wide! Perfect 1080p width!
    target_w = 640
    target_h = int(round(H * (target_w / W)))
    p1 = cv2.resize(rendered_raw, (target_w, target_h + 36))
    p2 = cv2.resize(rendered_t20, (target_w, target_h + 36))
    p3 = cv2.resize(rendered_clean, (target_w, target_h + 36))

    side_by_side = np.hstack([p1, p2, p3])
    out_file = os.path.join(out_dir, f"compare_{stem}.jpg")
    cv2.imwrite(out_file, side_by_side, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[+] Saved comparison: {out_file}")
    return out_file


def main():
    root = "/home/constant/CodeBuddy/Dev/AIC"
    test_img_dir = os.path.join(root, "dataset/test/AIC2026_PHASE_2_1000/visible")
    raw_sub_dir = os.path.join(root, "work_dir/submissions/submission_sovereign_v18_orthogonal_soup")
    clean_sub_dir = os.path.join(root, "work_dir/submissions/submission_puremal_aligned_top35/predictions")
    out_dir = os.path.join(root, "work_dir/audits/vis_test_predictions")
    artifact_dir = "/home/constant/.gemini/antigravity/brain/0e8de2bf-ddee-429b-bd88-8d0434ba0e1d"
    os.makedirs(out_dir, exist_ok=True)

    # Select representative stems:
    # 00000010: Street scene with 8 UAV phantom predictions on the road
    # 00000017: Nighttime scene with dark road and phantom ball
    # 00000018: Urban scene with multiple pedestrians and cars
    # 00000020: Water reflection and night streetlights
    # 00000037: Dense traffic scene
    # 00000062: High-contrast night infrared scene
    stems = ["00000010", "00000017", "00000018", "00000020", "00000037", "00000062"]

    saved_files = []
    for s in stems:
        f = render_comparison_triplet(s, test_img_dir, raw_sub_dir, clean_sub_dir, out_dir)
        if f:
            saved_files.append(f)
            # Also copy to artifact directory for presentation
            import shutil
            shutil.copy2(f, os.path.join(artifact_dir, os.path.basename(f)))

    print(f"\nRendered {len(saved_files)} comparison panels to {out_dir} and {artifact_dir}")


if __name__ == "__main__":
    main()
