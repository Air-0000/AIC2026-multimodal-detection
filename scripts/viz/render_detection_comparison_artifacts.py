#!/usr/bin/env python3
"""
Render Detection Comparison: 54.28 Baseline vs V23 Native Top-100
"""

import os
import cv2
import zipfile
import numpy as np

OUT_DIR = "/home/constant/.gemini/antigravity/brain/46ad3bec-cfa4-4605-895d-0a4627a337ed"

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage", 10: "uav", 11: "tricycle",
}

COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 200, 255), (180, 100, 255),
    (255, 255, 0), (255, 0, 255), (0, 165, 255), (0, 255, 128),
    (255, 200, 0), (128, 128, 255), (0, 255, 255), (200, 50, 200)
]

def draw_boxes(img, lines, min_score=0.10, title=""):
    canvas = img.copy()
    H, W = canvas.shape[:2]
    cnt = 0
    for l in lines:
        parts = l.strip().split()
        if len(parts) < 6: continue
        cid = int(parts[0])
        cx, cy, bw, bh, sc = map(float, parts[1:6])
        if sc < min_score: continue
        cnt += 1
        x1 = int((cx - bw / 2.0) * W)
        y1 = int((cy - bh / 2.0) * H)
        x2 = int((cx + bw / 2.0) * W)
        y2 = int((cy + bh / 2.0) * H)
        color = COLORS[cid % len(COLORS)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cid]}:{sc:.2f}"
        cv2.rectangle(canvas, (x1, max(0, y1 - 18)), (x1 + len(label) * 9, max(18, y1)), color, -1)
        cv2.putText(canvas, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Title header
    cv2.rectangle(canvas, (0, 0), (W, 36), (20, 20, 20), -1)
    cv2.putText(canvas, f"{title} (Displaying sc >= {min_score:.2f}: {cnt} boxes)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas

with zipfile.ZipFile("work_dir/submissions/submission_clean_single/submission_clean_single.zip") as zf_base, \
     zipfile.ZipFile("work_dir/submissions/v23_unfiltered_top100/submission_v23_native_unfiltered_top100.zip") as zf_v23:
    for stem in ["00000018", "00000010"]:
        img_p = f"work_dir/audits/vis_qc_samples/test_visible/{stem}.jpg"
        vis = cv2.imread(img_p)
        b_lines = [l.strip() for l in zf_base.read(f"{stem}.txt").decode().split("\r\n") if l.strip()][:100]
        v_lines = [l.strip() for l in zf_v23.read(f"{stem}.txt").decode().split("\r\n") if l.strip()][:100]

        p_base = draw_boxes(vis, b_lines, min_score=0.20, title=f"54.28 Baseline ({stem})")
        p_v23 = draw_boxes(vis, v_lines, min_score=0.20, title=f"V23 Multimodal Native Top-100 ({stem})")

        comp = np.vstack([p_base, p_v23])
        out_p = os.path.join(OUT_DIR, f"det_comp_{stem}.jpg")
        cv2.imwrite(out_p, comp, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"[+] Saved detection comparison artifact: {out_p}")
