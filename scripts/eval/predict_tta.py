"""DEIMv2-L test-set prediction with TTA (hflip x multiscale) + WBF fusion.

Views: scales [640, 800, 960] x flips [False, True] = 6 forwards per image.
WBF clusters boxes across views and averages (standard, equal-weight).

Usage:
    /home/constant/miniconda3/envs/deimv2/bin/python predict_tta.py [OUT_DIR] [THRESH] [SCALES]
Defaults: OUT_DIR=work_dir/submission_tta, THRESH=0.3, SCALES=640,800,960
"""
import os
import sys
import glob

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "work_dir/submission_tta")
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
SCALES = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "640,800,960").split(",")]
FLIPS = [False, True]
LINESEP = "\r\n"
MAX_BOXES = 100

DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
sys.path.insert(0, DEIMV2_ROOT)
os.chdir(DEIMV2_ROOT)
from engine.core import YAMLConfig

TEST_VISIBLE = os.path.join(ROOT, "dataset/test/visible")
os.makedirs(OUT_DIR, exist_ok=True)

cfg = YAMLConfig("configs/deimv2/deimv2_dinov3_l_aic.yml")
ck = torch.load(os.path.join(ROOT, "vendor/DEIMv2/outputs/deimv2_dinov3_l_aic/best_stg2.pth"),
                map_location="cpu")
sd = ck["ema"]["module"] if "ema" in ck else ck["model"]
cfg.model.load_state_dict(sd, strict=False)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig):
        return self.postprocessor(self.model(images), orig)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
infer = Model().to(device).eval()
print(f"[tta] DEIMv2-L loaded, scales={SCALES} flips={FLIPS} thresh={THRESH} device={device}")


def iou(b1, b2):
    x1a, y1a, x2a, y2a = b1
    x1b, y1b, x2b, y2b = b2
    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b) - inter
    return inter / ua if ua > 0 else 0.0


def wbf(all_boxes, iou_thr=0.55):
    """all_boxes: list of (cls, [x1,y1,x2,y2], conf). Standard WBF."""
    all_boxes = sorted(all_boxes, key=lambda b: b[2], reverse=True)
    fused = []
    for cls in sorted(set(b[0] for b in all_boxes)):
        members = [b for b in all_boxes if b[0] == cls]
        clusters = []
        for box in members:
            placed = False
            for cl in clusters:
                if iou(box[1], cl[0][1]) >= iou_thr:
                    cl.append(box)
                    placed = True
                    break
            if not placed:
                clusters.append([box])
        for cl in clusters:
            confs = np.array([b[2] for b in cl])
            ws = confs / confs.sum()
            cx = sum(b[1][0] * w for b, w in zip(cl, ws))
            cy = sum(b[1][1] * w for b, w in zip(cl, ws))
            x2 = sum(b[1][2] * w for b, w in zip(cl, ws))
            y2 = sum(b[1][3] * w for b, w in zip(cl, ws))
            c = np.clip(confs.sum() / (len(cl) ** 0.5), 0, 1)
            fused.append((cls, [cx, cy, x2, y2], float(c)))
    return fused


imgs = sorted(glob.glob(os.path.join(TEST_VISIBLE, "*")))
print(f"[tta] {len(imgs)} test images")

transforms_cache = {}
for size in SCALES:
    transforms_cache[size] = T.Compose([
        T.Resize((size, size)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

with torch.no_grad():
    for i, imp in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(imp))[0]
        im_pil = Image.open(imp).convert("RGB")
        w, h = im_pil.size
        orig = torch.tensor([[w, h]]).to(device)

        views = []
        for size in SCALES:
            tr = transforms_cache[size]
            for flip in FLIPS:
                t = tr(im_pil)
                if flip:
                    t = torch.flip(t, dims=[2])
                labels, boxes, scores = infer(t.unsqueeze(0).to(device), orig)
                boxes = boxes[0].clone()
                if flip:
                    x1, x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
                    boxes[:, 0] = w - x2
                    boxes[:, 2] = w - x1
                keep = scores[0] > THRESH
                for k in torch.nonzero(keep).flatten():
                    views.append((int(labels[0][k].item()),
                                  [float(boxes[k, 0]), float(boxes[k, 1]),
                                   float(boxes[k, 2]), float(boxes[k, 3])],
                                  float(scores[0][k].item())))

        fused = wbf(views)
        fused = sorted(fused, key=lambda b: b[2], reverse=True)[:MAX_BOXES]
        lines = []
        for cls, (x1, y1, x2, y2), c in fused:
            cx = min(max((x1 + x2) / 2 / w, 0.0), 1.0)
            cy = min(max((y1 + y2) / 2 / h, 0.0), 1.0)
            bw = min(max((x2 - x1) / w, 0.0), 1.0)
            bh = min(max((y2 - y1) / h, 0.0), 1.0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {c:.4f}")
        with open(os.path.join(OUT_DIR, f"{stem}.txt"), "w", newline="") as f:
            f.write(LINESEP.join(lines))
        if (i + 1) % 100 == 0:
            print(f"[tta] {i + 1}/{len(imgs)}")

print(f"[tta] done -> {OUT_DIR}")
