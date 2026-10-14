#!/usr/bin/env python3
"""
Visualize thr45 test predictions on test visible images and generate artifacts.
"""

import os
import glob
import zipfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARTIFACT_DIR = "/home/constant/.gemini/antigravity/brain/46ad3bec-cfa4-4605-895d-0a4627a337ed"
ZIP_PATH = "/home/constant/CodeBuddy/Dev/AIC/work_dir/submissions/submissions_v22_cycle1/submission_v22_cycle1_thr45.zip"
TEST_VIS_DIR = "/home/constant/CodeBuddy/Dev/AIC/dataset/test/AIC2026_PHASE_2_1000/visible"

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

# Distinct vivid RGB colors for 12 classes
CLASS_COLORS = {
    0: (0, 220, 255),    # person: cyan-yellow
    1: (0, 165, 255),    # boat: orange
    2: (0, 255, 0),      # animal: green
    3: (255, 191, 0),    # seat: sky blue
    4: (0, 0, 255),      # sign: bright red
    5: (255, 0, 255),    # bicycle: magenta
    6: (255, 255, 0),    # car: bright cyan
    7: (0, 255, 255),    # ball: yellow
    8: (180, 105, 255),  # light: pink/purple
    9: (50, 205, 50),    # garbage can: lime green
    10: (255, 100, 100), # uav: blue-violet
    11: (200, 200, 0),   # tricycle: teal
}

SELECTED_STEMS = [
    ("003429", "多目标城市场景 (行人/车辆/指示牌/路灯/自行车)"),
    ("000021_015_00000073", "复杂弱目标与三轮车 (小球/三轮车/指示牌/行人)"),
    ("003268", "水域微光场景 (水面船只与岸边反光解耦)"),
    ("00000010", "高难开阔空域 (低空微小无人机/车辆/行人)"),
    ("000153", "夜间暗光高密街区 (路灯/车辆/立柱/行人密集流)"),
    ("hehe_46_00000008", "微小地砖小球与长椅 (小球/座椅/垃圾桶/暗区目标)"),
]


def render_image_predictions(stem, title_text, zf):
    # Find image file
    matches = glob.glob(os.path.join(TEST_VIS_DIR, f"{stem}.*"))
    if not matches:
        print(f"Image not found: {stem}")
        return None
    img_path = matches[0]

    # Read image (RGB)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    H, W = img_bgr.shape[:2]

    # Read predictions from zip
    txt_name = f"{stem}.txt"
    if txt_name not in zf.namelist():
        lines = []
    else:
        lines = [ln.strip() for ln in zf.read(txt_name).decode().splitlines() if ln.strip()]

    # Parse boxes
    boxes = []
    for ln in lines:
        parts = ln.split()
        cid = int(parts[0])
        cx, cy, bw, bh, conf = map(float, parts[1:6])
        x1 = max(0, min(W - 1, int((cx - bw / 2.0) * W)))
        y1 = max(0, min(H - 1, int((cy - bh / 2.0) * H)))
        x2 = max(0, min(W - 1, int((cx + bw / 2.0) * W)))
        y2 = max(0, min(H - 1, int((cy + bh / 2.0) * H)))
        boxes.append((cid, x1, y1, x2, y2, conf))

    # Draw on canvas
    canvas = img_bgr.copy()
    for cid, x1, y1, x2, y2, conf in boxes:
        color = CLASS_COLORS.get(cid, (0, 255, 0))
        cname = CLASS_NAMES.get(cid, str(cid))
        label_text = f"{cname} {conf:.2f}"

        # Draw box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Draw label badge background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        bg_y1 = max(0, y1 - th - 6)
        bg_y2 = y1
        bg_x1 = x1
        bg_x2 = min(W, x1 + tw + 6)

        cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        # Contrast text color (black for light, white for dark)
        text_color = (0, 0, 0)
        cv2.putText(canvas, label_text, (bg_x1 + 3, bg_y2 - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Top title bar on image
    title_bar_h = 32
    title_bar = np.zeros((title_bar_h, W, 3), dtype=np.uint8)
    title_bar[:] = (30, 30, 30)
    info_str = f"[{stem}] {title_text} | Total: {len(boxes)} boxes (tau>=0.45)"
    cv2.putText(title_bar, info_str, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    combined = np.vstack([title_bar, canvas])
    return combined, stem, len(boxes)


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        rendered_list = []
        for stem, desc in SELECTED_STEMS:
            res = render_image_predictions(stem, desc, zf)
            if res is not None:
                img_out, stem, n_boxes = res
                out_path = os.path.join(ARTIFACT_DIR, f"thr45_test_{stem}.png")
                cv2.imwrite(out_path, img_out)
                print(f"Saved: {out_path} ({n_boxes} boxes)")
                # Resize for 2x3 composite
                target_w, target_h = 640, 392  # 360 + 32 title
                resized = cv2.resize(img_out, (target_w, target_h), interpolation=cv2.INTER_AREA)
                rendered_list.append(resized)

    # Build 2x3 composite gallery
    if len(rendered_list) == 6:
        row1 = np.hstack([rendered_list[0], rendered_list[1]])
        row2 = np.hstack([rendered_list[2], rendered_list[3]])
        row3 = np.hstack([rendered_list[4], rendered_list[5]])
        gallery = np.vstack([row1, row2, row3])
        gallery_path = os.path.join(ARTIFACT_DIR, "thr45_test_predictions_gallery.png")
        cv2.imwrite(gallery_path, gallery)
        print(f"Saved gallery composite: {gallery_path} (size: {gallery.shape[1]}x{gallery.shape[0]})")


if __name__ == "__main__":
    main()
