"""
Run inference with the trained DEIMv2-L model and write submission TXTs
matching the official AIC format:
    <class_id> <cx> <cy> <w> <h> <confidence>
Normalized coords, at most 100 boxes per image (top by confidence), CRLF lines.

Usage:
    /home/constant/miniconda3/envs/deimv2/bin/python predict_deimv2.py [CKPT] [OUTPUT_DIR] [THRESH] [SIZE]
Defaults:
    CKPT      = vendor/DEIMv2/outputs/deimv2_dinov3_l_aic/best_stg2.pth
    OUTPUT    = work_dir/submission_deimv2
    THRESH    = 0.3
    SIZE      = 640
"""
import os
import sys
import glob

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
sys.path.insert(0, DEIMV2_ROOT)
os.chdir(DEIMV2_ROOT)  # config/ckpts paths are relative to DEIMv2
from engine.core import YAMLConfig
CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(ROOT, "vendor/DEIMv2/outputs/deimv2_dinov3_l_aic/best_stg2.pth")
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "work_dir/submission_deimv2")
THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 640
LINESEP = "\r\n"

TEST_VISIBLE = os.path.join(ROOT, "dataset/test/visible")
os.makedirs(OUT_DIR, exist_ok=True)

cfg = YAMLConfig("configs/deimv2/deimv2_dinov3_l_aic.yml")
# load into train-mode model first (matches official torch_inf.py)
model = cfg.model
ck = torch.load(CKPT, map_location="cpu")
if "ema" in ck:
    sd = ck["ema"]["module"]
elif "model" in ck:
    sd = ck["model"]
else:
    sd = ck
model.load_state_dict(sd, strict=False)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
infer_model = Model().to(device)
infer_model.eval()
print(f"[predict] DEIMv2-L loaded from {CKPT} (EMA), device={device}")

transforms = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

imgs = sorted(glob.glob(os.path.join(TEST_VISIBLE, "*")))
print(f"[predict] {len(imgs)} test images")

with torch.no_grad():
    for i, imp in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(imp))[0]
        im_pil = Image.open(imp).convert("RGB")
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(im_pil).unsqueeze(0).to(device)

        labels, boxes, scores = infer_model(im_data, orig_size)
        labels, boxes, scores = labels[0], boxes[0], scores[0]

        # postprocessor already returns boxes in ORIGINAL pixel coords
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        keep = scores > THRESH
        cx = ((x1 + x2) / 2)[keep]
        cy = ((y1 + y2) / 2)[keep]
        bw = (x2 - x1)[keep]
        bh = (y2 - y1)[keep]
        conf = scores[keep]
        cid = labels[keep].long()

        lines = []
        if len(conf) > 0:
            order = np.argsort(-conf.cpu().numpy())
            cid = cid.cpu().numpy()[order]
            cx = cx.cpu().numpy()[order]
            cy = cy.cpu().numpy()[order]
            bw = bw.cpu().numpy()[order]
            bh = bh.cpu().numpy()[order]
            conf = conf.cpu().numpy()[order]
            for k in range(min(len(conf), 100)):
                # clamp to [0,1]
                cxx = min(max(cx[k] / w, 0.0), 1.0)
                cyy = min(max(cy[k] / h, 0.0), 1.0)
                bww = min(max(bw[k] / w, 0.0), 1.0)
                bhh = min(max(bh[k] / h, 0.0), 1.0)
                lines.append(f"{cid[k]} {cxx:.6f} {cyy:.6f} {bww:.6f} {bhh:.6f} {conf[k]:.4f}")

        with open(os.path.join(OUT_DIR, f"{stem}.txt"), "w", newline="") as f:
            f.write(LINESEP.join(lines))

        if (i + 1) % 100 == 0:
            print(f"[predict] {i + 1}/{len(imgs)} done, total boxes: {len(lines)}")

print(f"[predict] done -> {OUT_DIR}")
