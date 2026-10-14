"""
Run inference with a trained RGB RF-DETR model and write submission TXTs.

For each image in dataset/test/visible, write a same-named .txt with lines:
    <class_id> <cx> <cy> <w> <h> <confidence>
Normalized coords, at most 100 boxes per image (top by confidence).

Usage:
    python3 predict_rgb.py [CKPT_DIR] [OUTPUT_DIR]
Defaults: CKPT_DIR=work_dir/rgb_baseline, OUTPUT_DIR=work_dir/submission
"""
import os
import sys
import glob
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
CKPT_DIR = sys.argv[1] if len(sys.argv) > 1 else "work_dir/rgb_baseline"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "work_dir/submission"
# Confidence threshold applied AFTER NMS. Lower => more recall but more FP.
THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
# Match the official example format exactly: CRLF (Windows-style) line endings.
LINESEP = "\r\n"

TEST_VISIBLE = os.path.join(ROOT, "dataset/test/visible")
os.makedirs(OUT_DIR, exist_ok=True)

cands = [
    os.path.join(ROOT, CKPT_DIR, "checkpoint_best_ema.pth"),
    os.path.join(ROOT, CKPT_DIR, "checkpoint_best_regular.pth"),
    os.path.join(ROOT, CKPT_DIR, "checkpoint_best.pth"),
    os.path.join(ROOT, CKPT_DIR, "checkpoint_last.pth"),
]
ckpt = next((c for c in cands if os.path.exists(c)), None)
assert ckpt, f"no checkpoint found in {CKPT_DIR}"
print(f"[predict] using checkpoint: {ckpt}")

model = RFDETRBase(pretrain_weights=ckpt, resolution=560, num_classes=12)

imgs = sorted(glob.glob(os.path.join(TEST_VISIBLE, "*")))
print(f"[predict] {len(imgs)} test images")

for imp in imgs:
    stem = os.path.splitext(os.path.basename(imp))[0]
    im = Image.open(imp).convert("RGB")
    W, H = im.size
    det = model.predict(im, threshold=THRESH)
    xyxy = np.asarray(det.xyxy, dtype=float)
    conf = np.asarray(det.confidence, dtype=float)
    cid = np.asarray(det.class_id, dtype=int)

    lines = []
    if len(conf) > 0:
        order = np.argsort(-conf)
        xyxy, conf, cid = xyxy[order], conf[order], cid[order]
        for k in range(min(len(conf), 100)):
            x1, y1, x2, y2 = xyxy[k]
            cx = min(max((x1 + x2) / 2 / W, 0.0), 1.0)
            cy = min(max((y1 + y2) / 2 / H, 0.0), 1.0)
            w = min(max((x2 - x1) / W, 0.0), 1.0)
            h = min(max((y2 - y1) / H, 0.0), 1.0)
            lines.append(f"{int(cid[k])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {float(conf[k]):.4f}")

    # Always write a file (empty file when no detection) -> satisfies the
    # "must submit an empty TXT even if nothing detected" rule.
    with open(os.path.join(OUT_DIR, stem + ".txt"), "w", newline="") as f:
        if lines:
            f.write(LINESEP.join(lines) + LINESEP)

print(f"[predict] wrote {len(imgs)} files -> {OUT_DIR}")
