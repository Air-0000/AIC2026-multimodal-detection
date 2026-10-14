"""Per-class confidence threshold sweep on AIC validation split.

1. Run DEIMv2-L once, keep all detections with conf > 0.1 (high recall).
2. For each class, sweep its threshold while others stay at 0.3, record AP.
3. Report per-class best threshold and the combined-optimum mAP.
"""
import sys, os, json
sys.path.insert(0, "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")
os.chdir("/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")
import torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from engine.core import YAMLConfig
import faster_coco_eval
faster_coco_eval.init_as_pycocotools()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
IMG = f"{ROOT}/work_dir/deimv2_dataset/valid/images"
ANN = f"{ROOT}/work_dir/deimv2_dataset/valid/annotations/instances_val.json"
names = sorted(os.listdir(IMG))
tr = T.Compose([T.Resize((640, 640)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
coco_gt = COCO(ANN)
cats = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}
name2id = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}

# collect ALL dets > 0.1 (raw pixel boxes)
raw = []  # list of dicts with score
with torch.no_grad():
    for i, fname in enumerate(names):
        im = Image.open(os.path.join(IMG, fname)).convert("RGB")
        w, h = im.size
        orig = torch.tensor([[w, h]]).cuda()
        labels, boxes, scores = infer(tr(im).unsqueeze(0).cuda(), orig)
        labels, boxes, scores = labels[0], boxes[0], scores[0]
        keep = scores > 0.1
        img_id = name2id[fname]
        for k in torch.nonzero(keep).flatten():
            # postprocessor returns xyxy (pixel) -> convert to xywh for COCOeval
            x1, y1, x2, y2 = [float(v) for v in boxes[k].tolist()]
            raw.append({"image_id": img_id, "category_id": int(labels[k].item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(scores[k].item())})
        if (i + 1) % 50 == 0:
            print(f"infer {i + 1}/{len(names)}", flush=True)
print(f"raw dets (conf>0.1): {len(raw)}", flush=True)


def eval_with_threshs(thr_dict):
    dets = [d for d in raw if d["score"] >= thr_dict.get(d["category_id"], 0.3)]
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats[0]


# baseline: all 0.3
base = eval_with_threshs({})
print(f"baseline all-0.3: mAP={base:.4f}", flush=True)

# per-class sweep
best_per_class = {}
for cid in sorted(cats):
    nm = cats[cid]
    best_ap, best_t = -1, 0.3
    for t in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
        thr = {cid: t}
        ap = eval_with_threshs(thr)
        if ap > best_ap:
            best_ap, best_t = ap, t
    best_per_class[cid] = best_t
    print(f"  {nm:12s}: best_t={best_t}  mAP(单类调)={best_ap:.4f}  (vs base {base:.4f})", flush=True)

# combined: apply each class's best threshold together
combined = eval_with_threshs(best_per_class)
print(f"\ncombined per-class best: mAP={combined:.4f}  (baseline {base:.4f})", flush=True)
print("best thresholds:", {cats[c]: t for c, t in sorted(best_per_class.items())}, flush=True)
