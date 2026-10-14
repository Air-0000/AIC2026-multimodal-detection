#!/usr/bin/env python3
"""
Render Sovereign V22 thr45 predictions onto test visible images at 100% native resolution.
Saves standalone image files into dataset/test/AIC2026_PHASE_2_1000/annotated_visible_thr45/
without modifying or overwriting original visible images.
"""

import os
import glob
import zipfile
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np

TEST_DIR = "/home/constant/CodeBuddy/Dev/AIC/dataset/test/AIC2026_PHASE_2_1000"
VIS_DIR = os.path.join(TEST_DIR, "visible")
OUT_DIR = os.path.join(TEST_DIR, "annotated_visible_thr45")
ZIP_PATH = "/home/constant/CodeBuddy/Dev/AIC/work_dir/submissions/submissions_v22_cycle1/submission_v22_cycle1_thr45.zip"

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

# Vivid distinct BGR colors for OpenCV drawing
CLASS_COLORS = {
    0: (0, 220, 255),    # person: yellow-cyan
    1: (0, 140, 255),    # boat: deep orange
    2: (0, 255, 0),      # animal: bright green
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


def process_one_image(item):
    img_path, lines = item
    fname = os.path.basename(img_path)
    out_path = os.path.join(OUT_DIR, fname)

    img = cv2.imread(img_path)
    if img is None:
        return 0
    H, W = img.shape[:2]

    # Draw boxes if predictions exist
    for ln in lines:
        parts = ln.split()
        if len(parts) < 6:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh, conf = map(float, parts[1:6])

        x1 = max(0, min(W - 1, int((cx - bw / 2.0) * W)))
        y1 = max(0, min(H - 1, int((cy - bh / 2.0) * H)))
        x2 = max(0, min(W - 1, int((cx + bw / 2.0) * W)))
        y2 = max(0, min(H - 1, int((cy + bh / 2.0) * H)))

        color = CLASS_COLORS.get(cid, (0, 255, 0))
        cname = CLASS_NAMES.get(cid, str(cid))
        label_text = f"{cname} {conf:.2f}"

        # Draw box (2px border)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        bg_y1 = max(0, y1 - th - 5)
        bg_y2 = y1
        bg_x1 = x1
        bg_x2 = min(W, x1 + tw + 4)

        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(img, label_text, (bg_x1 + 2, bg_y2 - 3), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Save standalone full-resolution image
    cv2.imwrite(out_path, img)
    return len(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print(" 渲染 0.45 黄金门控预测至测试集原生全清大图 ".center(70, "="))
    print("=" * 70)
    print(f"原图目录: {VIS_DIR}")
    print(f"输出目录: {OUT_DIR}")
    print(f"输入预测: {ZIP_PATH}")

    # Read all predictions from zip into memory
    preds = {}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in zf.namelist():
            stem = os.path.splitext(name)[0]
            lines = [ln.strip() for ln in zf.read(name).decode().splitlines() if ln.strip()]
            preds[stem] = lines

    # Match with visible images
    img_files = sorted(glob.glob(os.path.join(VIS_DIR, "*.*")))
    tasks = []
    for f in img_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        lines = preds.get(stem, [])
        tasks.append((f, lines))

    print(f"发现 {len(tasks)} 张测试图，正在并发渲染原生全尺寸图像...")
    with ProcessPoolExecutor(max_workers=16) as pool:
        box_counts = list(pool.map(process_one_image, tasks))

    total_boxes = sum(box_counts)
    print("=" * 70)
    print(f"渲染完成！共 {len(tasks)} 张原生独立高清大图已全部输出至:")
    print(f"👉 {OUT_DIR}")
    print(f"总标注框数: {total_boxes} 框")
    print("=" * 70)


if __name__ == "__main__":
    main()
