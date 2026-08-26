"""DEIMv2-L validation-split prediction with TTA (hflip x multiscale) + WBF."""
import sys, os, time
sys.path.insert(0, "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")
os.chdir("/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")
import torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from engine.core import YAMLConfig

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
cfg = YAMLConfig("configs/deimv2/deimv2_dinov3_l_aic.yml")
ck = torch.load(f"{ROOT}/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic/best_stg2.pth", map_location="cpu")
sd = ck["ema"]["module"] if "ema" in ck else ck["model"]
cfg.model.load_state_dict(sd, strict=False)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig):
        return self.postprocessor(self.model(images), orig)


infer = Model().cuda().eval()
SCALES = [640, 800, 960]
THRESH = 0.3
FLIPS = [False, True]
OUT = f"{ROOT}/work_dir/val_pred_tta"
os.makedirs(OUT, exist_ok=True)
IMG = f"{ROOT}/work_dir/deimv2_dataset/valid/images"
names = sorted(os.listdir(IMG))
trs = {s: T.Compose([T.Resize((s, s)), T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
       for s in SCALES}


def iou(b1, b2):
    x1a, y1a, x2a, y2a = b1
    x1b, y1b, x2b, y2b = b2
    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b) - inter
    return inter / ua if ua > 0 else 0


def wbf(allb, iou_thr=0.55):
    allb = sorted(allb, key=lambda b: b[2], reverse=True)
    fused = []
    for cls in sorted(set(b[0] for b in allb)):
        mem = [b for b in allb if b[0] == cls]
        clusters = []
        for box in mem:
            for cl in clusters:
                if iou(box[1], cl[0][1]) >= iou_thr:
                    cl.append(box)
                    break
            else:
                clusters.append([box])
        for cl in clusters:
            cf = np.array([b[2] for b in cl])
            ws = cf / cf.sum()
            cx = sum(b[1][0] * w for b, w in zip(cl, ws))
            cy = sum(b[1][1] * w for b, w in zip(cl, ws))
            x2 = sum(b[1][2] * w for b, w in zip(cl, ws))
            y2 = sum(b[1][3] * w for b, w in zip(cl, ws))
            c = float(np.clip(cf.sum() / len(cl) ** 0.5, 0, 1))
            fused.append((cls, [cx, cy, x2, y2], c))
    return fused


t0 = time.time()
with torch.no_grad():
    for i, fname in enumerate(names):
        im = Image.open(os.path.join(IMG, fname)).convert("RGB")
        w, h = im.size
        orig = torch.tensor([[w, h]]).cuda()
        views = []
        for s in SCALES:
            for flip in FLIPS:
                t = trs[s](im)
                if flip:
                    t = torch.flip(t, dims=[2])
                labels, boxes, scores = infer(t.unsqueeze(0).cuda(), orig)
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
        fused = sorted(fused, key=lambda b: b[2], reverse=True)[:100]
        lines = []
        for cls, (x1, y1, x2, y2), c in fused:
            cx = min(max((x1 + x2) / 2 / w, 0), 1)
            cy = min(max((y1 + y2) / 2 / h, 0), 1)
            bw = min(max((x2 - x1) / w, 0), 1)
            bh = min(max((y2 - y1) / h, 0), 1)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {c:.4f}")
        with open(os.path.join(OUT, fname.rsplit(".", 1)[0] + ".txt"), "w", newline="") as f:
            f.write("\r\n".join(lines))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(names)}", flush=True)
print(f"done {time.time() - t0:.0f}s", flush=True)
