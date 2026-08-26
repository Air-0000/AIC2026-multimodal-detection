"""Predict on the AIC VALIDATION split (200 imgs) with a given model, write txts.

Usage:
    python3 predict_val.py deimv2 [OUT_DIR] [THRESH]      # needs deimv2 env
    python3 predict_val.py rgb    [OUT_DIR] [THRESH]
"""
import os
import sys
import glob
import json

import numpy as np
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "deimv2"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else f"work_dir/val_pred_{MODEL}"
if not os.path.isabs(OUT_DIR):
    OUT_DIR = os.path.join(ROOT, OUT_DIR)
THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
LINESEP = "\r\n"

ANN = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
with open(ANN) as f:
    val_ann = json.load(f)
names = sorted(i["file_name"] for i in val_ann["images"])
IMG_DIR = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/images")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[val-pred] {MODEL}: {len(names)} images -> {OUT_DIR}, thresh={THRESH}")

if MODEL == "deimv2":
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
    sys.path.insert(0, DEIMV2_ROOT)
    os.chdir(DEIMV2_ROOT)
    from engine.core import YAMLConfig
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
    infer = Model().cuda().eval()
    tr = T.Compose([T.Resize((640, 640)), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    with torch.no_grad():
        for i, fname in enumerate(names):
            im = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
            w, h = im.size
            im_data = tr(im).unsqueeze(0).cuda()
            orig = torch.tensor([[w, h]]).cuda()
            labels, boxes, scores = infer(im_data, orig)
            labels, boxes, scores = labels[0], boxes[0], scores[0]
            keep = scores > THRESH
            cx = ((boxes[:, 0] + boxes[:, 2]) / 2)[keep]
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2)[keep]
            bw = (boxes[:, 2] - boxes[:, 0])[keep]
            bh = (boxes[:, 3] - boxes[:, 1])[keep]
            conf = scores[keep]
            cid = labels[keep].long()
            lines = []
            if len(conf) > 0:
                order = np.argsort(-conf.cpu().numpy())
                cid = cid.cpu().numpy()[order]
                cx = cx.cpu().numpy()[order] / w
                cy = cy.cpu().numpy()[order] / h
                bw = np.clip(bw.cpu().numpy()[order] / w, 0, 1)
                bh = np.clip(bh.cpu().numpy()[order] / h, 0, 1)
                conf = conf.cpu().numpy()[order]
                for k in range(min(len(conf), 100)):
                    lines.append(f"{cid[k]} {cx[k]:.6f} {cy[k]:.6f} {bw[k]:.6f} {bh[k]:.6f} {conf[k]:.4f}")
            with open(os.path.join(OUT_DIR, fname.rsplit(".", 1)[0] + ".txt"), "w", newline="") as f:
                f.write(LINESEP.join(lines))
            if (i + 1) % 50 == 0:
                print(f"[val-pred] {i + 1}/{len(names)}")
else:
    from rfdetr import RFDETRBase
    cands = [os.path.join(ROOT, "work_dir/rgb_baseline", c) for c in
             ["checkpoint_best_ema.pth", "checkpoint_best_regular.pth"]]
    ckpt = next((c for c in cands if os.path.exists(c)), None)
    model = RFDETRBase(pretrain_weights=ckpt, resolution=560, num_classes=12)
    for i, fname in enumerate(names):
        im = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
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
                cx = min(max((x1 + x2) / 2 / W, 0), 1)
                cy = min(max((y1 + y2) / 2 / H, 0), 1)
                bw = min(max((x2 - x1) / W, 0), 1)
                bh = min(max((y2 - y1) / H, 0), 1)
                lines.append(f"{cid[k]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf[k]:.4f}")
        with open(os.path.join(OUT_DIR, fname.rsplit(".", 1)[0] + ".txt"), "w", newline="") as f:
            f.write(LINESEP.join(lines))
        if (i + 1) % 50 == 0:
            print(f"[val-pred] {i + 1}/{len(names)}")

print(f"[val-pred] done -> {OUT_DIR}")
