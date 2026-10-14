#!/usr/bin/env python3
"""
Draw detection comparison side-by-side:
Buggy Submission vs Fixed Submission on Test Set images
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CAT_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle"
}

COLORS = {
    0: (0, 255, 0),     # person - green
    1: (255, 0, 0),     # boat - blue
    2: (255, 128, 0),   # animal
    3: (128, 0, 255),   # seat
    4: (0, 0, 255),     # sign - red
    5: (255, 255, 0),   # bicycle - cyan
    6: (0, 255, 255),   # car - yellow
    7: (255, 0, 255),   # ball - magenta
    8: (255, 200, 200), # light
    9: (100, 200, 100), # garbage can
    10: (50, 50, 255),  # uav
    11: (200, 100, 50)  # tricycle
}

def load_boxes(txt_path, W, H, conf_thresh=0.25):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                c = int(parts[0])
                cx, cy, bw, bh, conf = map(float, parts[1:6])
                if conf < conf_thresh:
                    continue
                x1 = int((cx - bw / 2.0) * W)
                y1 = int((cy - bh / 2.0) * H)
                x2 = int((cx + bw / 2.0) * W)
                y2 = int((cy + bh / 2.0) * H)
                boxes.append((c, conf, max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
    return boxes

def draw_boxes(img, boxes):
    canvas = img.copy()
    for c, conf, x1, y1, x2, y2 in boxes:
        color = COLORS.get(c, (0, 255, 0))
        label = f"{CAT_NAMES.get(c, str(c))} {conf:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(0, y1)), color, -1)
        cv2.putText(canvas, label, (x1 + 2, max(th, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas

def main():
    test_root = "/root/autodl-tmp/AIC/dataset/test/AIC2026_PHASE_2_1000/visible"
    old_dir = "/root/autodl-tmp/AIC/work_dir/submission_v10_nops"
    fix_dir = "/root/autodl-tmp/AIC/work_dir/submission_v10_fixed"
    out_dir = "/root/autodl-tmp/AIC/work_dir/vis_artifacts"
    os.makedirs(out_dir, exist_ok=True)

    stems = ["00000018", "001672"]

    for stem in stems:
        img_p = os.path.join(test_root, stem + ".jpg")
        if not os.path.exists(img_p): img_p = os.path.join(test_root, stem + ".png")
        if not os.path.exists(img_p): continue

        img = cv2.imread(img_p)
        H, W = img.shape[:2]

        old_boxes = load_boxes(os.path.join(old_dir, stem + ".txt"), W, H, conf_thresh=0.25)
        fix_boxes = load_boxes(os.path.join(fix_dir, stem + ".txt"), W, H, conf_thresh=0.25)

        img_old = draw_boxes(img, old_boxes)
        img_fix = draw_boxes(img, fix_boxes)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
        ax1.imshow(cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"[Old Buggy & Unaligned Setup] - {len(old_boxes)} Boxes (conf >= 0.25)", fontsize=14, color="darkred")
        ax1.axis("off")

        ax2.imshow(cv2.cvtColor(img_fix, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"[Fixed LBG & Calibrated Setup] - {len(fix_boxes)} Boxes (conf >= 0.25)", fontsize=14, color="darkgreen")
        ax2.axis("off")

        fig.suptitle(f"Test Set Prediction Comparison: Stem {stem}", fontsize=18, fontweight="bold")
        plt.tight_layout()
        save_p = os.path.join(out_dir, f"test_pred_comparison_{stem}.png")
        plt.savefig(save_p, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_p}")

if __name__ == "__main__":
    main()
