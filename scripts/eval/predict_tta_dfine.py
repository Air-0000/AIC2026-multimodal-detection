"""D-FINE-X (dfine_x_obj2coco_ft) 第二测试集 (PHASE_2_1000) TTA 推理.

单模型 TTA（多尺度 + 水平翻转）+ WBF 融合, 输出竞赛提交 txt（每图一个文件）.
与 predict_tta.py 的区别:
  * 宿主 = dfine HGNetv2-X, config = deim_hgnetv2_x_aic_obj2coco_ft.yml
  * 输入归一化 = /255 (RT-DETR/D-FINE 系), **不能**用 ImageNet mean/std
  * 权重取 ckpt['ema']['module'] (val 评的就是 EMA)

Usage:
  cd vendor/DEIMv2 && PYTHONPATH=../../scripts/lib python ../../scripts/eval/predict_tta_dfine.py [OUT_DIR] [THRESH] [SCALES]
"""
import os
import sys

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "work_dir/submission_tta_dfine")
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
SCALES = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "576,640,768").split(",")]
# argv[4] = "noflip" 时关闭翻转 (单视图, 用于不 TTA 的提交包)
FLIPS = [False] if (len(sys.argv) > 4 and sys.argv[4] == "noflip") else [False, True]
LINESEP = "\r\n"
MAX_BOXES = 100
CKPT = os.path.join(ROOT, "vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_obj2coco_ft/best_stg2.pth")

DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
for p in (DEIMV2_ROOT, os.path.join(ROOT, "scripts", "lib")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(DEIMV2_ROOT)

import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from engine.core import YAMLConfig

TEST_VISIBLE = os.path.join(ROOT, "dataset/test/AIC2026_PHASE_2_1000/visible")
os.makedirs(OUT_DIR, exist_ok=True)

cfg = YAMLConfig("configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft.yml")
ck = torch.load(CKPT, map_location="cpu")
sd = ck["ema"]["module"] if "ema" in ck else ck["model"]
missing, unexpected = cfg.model.load_state_dict(sd, strict=False)
print(f"[tta-dfine] load: missing={len(missing)} unexpected={len(unexpected)}")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        # deploy 会按 eval_spatial_size=640 缓存 anchors/valid_mask, 多尺度 TTA
        # 时 576/768 视图 token 数对不上 (8400 vs 6804) 直接崩; 置 None 让
        # decoder 按实际 spatial_shapes 现算 (dfine_decoder._get_decoder_input)
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "eval_spatial_size"):
            self.model.decoder.eval_spatial_size = None
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig):
        return self.postprocessor(self.model(images), orig)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
infer = Model().to(device).eval()
print(f"[tta-dfine] D-FINE-X loaded, scales={SCALES} flips={FLIPS} thresh={THRESH} device={device}")


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
            c = float(confs.mean())  # 标准WBF: 均值; sum/√n 在6视图下66%饱和到1.0
            fused.append((cls, [cx, cy, x2, y2], float(c)))
    return fused


imgs = sorted(glob.glob(os.path.join(TEST_VISIBLE, "*")))
print(f"[tta-dfine] {len(imgs)} test images")

transforms_cache = {}
for size in SCALES:
    transforms_cache[size] = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),          # [0,1] —— D-FINE 系不做 ImageNet 归一化
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
            print(f"[tta-dfine] {i + 1}/{len(imgs)}", flush=True)

print(f"[tta-dfine] done -> {OUT_DIR}")
