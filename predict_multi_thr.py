"""Generate test-set predictions at multiple confidence thresholds in one pass."""
import sys, os, glob, time
sys.path.insert(0, "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")
os.chdir("/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")
import torch, torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
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
THRESHS = [0.4, 0.5, 0.6, 0.7]
OUTS = {t: f"{ROOT}/work_dir/submission_deimv2_thr{t}" for t in THRESHS}
for d in OUTS.values():
    os.makedirs(d, exist_ok=True)
TEST = f"{ROOT}/dataset/test/visible"
imgs = sorted(glob.glob(os.path.join(TEST, "*")))
tr = T.Compose([T.Resize((640, 640)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

t0 = time.time()
with torch.no_grad():
    for i, imp in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(imp))[0]
        im = Image.open(imp).convert("RGB")
        w, h = im.size
        orig = torch.tensor([[w, h]]).cuda()
        labels, boxes, scores = infer(tr(im).unsqueeze(0).cuda(), orig)
        labels, boxes, scores = labels[0], boxes[0], scores[0]
        for t in THRESHS:
            keep = scores > t
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
            with open(os.path.join(OUTS[t], f"{stem}.txt"), "w", newline="") as f:
                f.write("\r\n".join(lines))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(imgs)} {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
