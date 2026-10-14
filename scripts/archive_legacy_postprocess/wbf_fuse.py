"""
Weighted Boxes Fusion (WBF) of two submission dirs:
    DEIMv2-L  (work_dir/submission_deimv2)
    RF-DETR   (work_dir/submission_rgb)

Each txt line:  <cls> <cx> <cy> <w> <h> <conf>  (normalized)
WBF clusters boxes per class using IoU > iou_thr and averages them
weighted by confidence. Output matches the official format (CRLF, <=100).

Usage:
    python3 wbf_fuse.py [A_DIR] [B_DIR] [OUT_DIR] [IOU_THR]
"""
import os
import sys
import glob

import numpy as np

A_DIR = sys.argv[1] if len(sys.argv) > 1 else "work_dir/submission_deimv2"
B_DIR = sys.argv[2] if len(sys.argv) > 2 else "work_dir/submission_rgb"
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "work_dir/submission_wbf"
IOU_THR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.55
LINESEP = "\r\n"
MAX_BOXES = 100


def load_txt(path):
    """Return list of (cls, cx, cy, w, h, conf)."""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 6:
                continue
            cls, cx, cy, w, h, conf = parts
            boxes.append((int(cls), float(cx), float(cy), float(w), float(h), float(conf)))
    return boxes


def iou(b1, b2):
    """IoU between two [cx,cy,w,h] normalized boxes."""
    x1a, y1a = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
    x2a, y2a = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    x1b, y1b = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
    x2b, y2b = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (b1[2] * b1[3]) + (b2[2] * b2[3]) - inter
    return inter / ua if ua > 0 else 0.0


def wbf_boxes(all_boxes, iou_thr):
    """all_boxes: list of (cls, cx, cy, w, h, conf) from ALL models (any count).
    Standard WBF: cluster by IoU, weighted-average position, boost fused conf.
    Returns fused list of (cls, cx, cy, w, h, conf)."""
    # sort by confidence desc
    all_boxes = sorted(all_boxes, key=lambda b: b[5], reverse=True)
    fused = []
    # cluster per class
    classes = sorted(set(b[0] for b in all_boxes))
    for cls in classes:
        members = [b for b in all_boxes if b[0] == cls]
        clusters = []  # each cluster = list of boxes
        for box in members:
            placed = False
            for cl in clusters:
                # match against cluster's best (first) box
                if iou(box[1:5], cl[0][1:5]) >= iou_thr:
                    cl.append(box)
                    placed = True
                    break
            if not placed:
                clusters.append([box])
        for cl in clusters:
            confs = np.array([b[5] for b in cl])
            ws = confs / confs.sum()
            cx = sum(b[1] * w for b, w in zip(cl, ws))
            cy = sum(b[2] * w for b, w in zip(cl, ws))
            bw = sum(b[3] * w for b, w in zip(cl, ws))
            bh = sum(b[4] * w for b, w in zip(cl, ws))
            # boosted confidence (WBF heuristic)
            c = np.clip(confs.sum() / (len(cl) ** 0.5), 0, 1)
            fused.append((cls, cx, cy, bw, bh, float(c)))
    return fused


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files_a = {os.path.splitext(os.path.basename(p))[0]: p
               for p in glob.glob(os.path.join(A_DIR, "*.txt"))}
    files_b = {os.path.splitext(os.path.basename(p))[0]: p
               for p in glob.glob(os.path.join(B_DIR, "*.txt"))}
    stems = sorted(set(files_a) & set(files_b))
    print(f"[wbf] {len(stems)} images fused from {os.path.basename(A_DIR)} "
          f"({len(files_a)}) + {os.path.basename(B_DIR)} ({len(files_b)}) iou_thr={IOU_THR}")

    total_boxes = 0
    for stem in stems:
        boxes = load_txt(files_a[stem]) + load_txt(files_b[stem])
        fused = wbf_boxes(boxes, IOU_THR)
        fused = sorted(fused, key=lambda b: b[5], reverse=True)[:MAX_BOXES]
        lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {c:.4f}"
                 for cls, cx, cy, w, h, c in fused]
        with open(os.path.join(OUT_DIR, f"{stem}.txt"), "w", newline="") as f:
            f.write(LINESEP.join(lines))
        total_boxes += len(lines)

    print(f"[wbf] done: {total_boxes} boxes -> {OUT_DIR}")


if __name__ == "__main__":
    main()
