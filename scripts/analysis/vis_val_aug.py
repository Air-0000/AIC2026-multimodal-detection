"""Validation-set test of the midboost (multi-modal) augmentation effect.

For every validation image we build the full 6-ch input (RGB + IR-heat + raw-IR
+ depth) and run the trained detector in THREE modes:
    DAY-FULL   : daytime RGB + real aux        (upper bound)
    NIGHT-FULL : nightified RGB + real aux      (midboost at night)
    NIGHT-NOAUG: nightified RGB + ZERO aux       (= RGB-only, "no augmentation")

GT comes from instances_val.json (COCO). We report COCO mAP for each mode and
draw a 3-col figure (Day-full / Night-full / Night-noaug) on 3 GT-rich images,
with GT (red dashed) and predictions (green) overlaid. This isolates the value
of the IR/depth augmentation when the visible channel goes dark.
"""
import os, sys, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
import numpy as np
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from faster_coco_eval import COCO, COCOeval_faster as COCOeval

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
sys.path.insert(0, DEIMV2_ROOT)
os.chdir(DEIMV2_ROOT)
from engine.core import YAMLConfig

CFG = "configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost.yml"
CK = os.path.join(ROOT, "vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost/best_stg1.pth")

IMG_DIR = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/images")
HEAT_DIR = os.path.join(ROOT, "work_dir/deimv2_heat/valid")
RAWIR_DIR = os.path.join(ROOT, "work_dir/deimv2_ir/valid")
DEPTH_DIR = os.path.join(ROOT, "dataset/train/depth")
ANN = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
LEVEL = 0.05
THR = 0.3

MEAN = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)

cats = json.load(open(ANN))["categories"]
SORTED = sorted(cats, key=lambda c: c["id"])
CAT_LIST = [c["name"] for c in SORTED]          # index -> name
CAT_ID = [c["id"] for c in SORTED]               # DEIM label -> coco category_id
NAME2ID = {c["name"]: c["id"] for c in cats}


def label_name(l):
    try:
        return CAT_LIST[int(l)]
    except Exception:
        return str(l)


def _resolve(folder, name):
    p = os.path.join(folder, os.path.splitext(name)[0] + ".png")
    return p if os.path.exists(p) else os.path.join(folder, name)


def read_ch(folder, name, n_ch):
    im = Image.open(_resolve(folder, name))
    if n_ch == 1:
        im = im.convert("L")
        return T.functional.pil_to_tensor(im).float() / 255.0
    im = im.convert("RGB")
    return T.functional.pil_to_tensor(im).float() / 255.0


def read_depth(folder, name):
    arr = np.array(Image.open(_resolve(folder, name)).convert("L")).astype("float32")
    arr = arr / 19999.0 if arr.max() > 1000.0 else arr / 255.0
    return torch.from_numpy(arr.clip(0.0, 1.0))[None]


def build_six(name):
    rgb = Image.open(os.path.join(IMG_DIR, name)).convert("RGB")
    W, H = rgb.size
    rgb_t = T.functional.pil_to_tensor(rgb).float() / 255.0
    heat = T.functional.resize(read_ch(HEAT_DIR, name, 1), (H, W))
    raw = T.functional.resize(read_ch(RAWIR_DIR, name, 1), (H, W))
    dep = T.functional.resize(read_depth(DEPTH_DIR, name), (H, W))
    x = torch.cat([rgb_t, heat, raw, dep], dim=0)
    return x, rgb_t, W, H


def nightify(x, level, seed=42):
    g = torch.Generator().manual_seed(seed)
    rgb, aux = x[:3], x[3:]
    h, w = rgb.shape[-2:]
    hh, ww = max(h // 8, 4), max(w // 8, 4)
    yy = torch.linspace(0.0, 1.0, hh).view(hh, 1)
    xx = torch.linspace(0.0, 1.0, ww).view(1, ww)
    f = F.interpolate(torch.randn(1, 1, 4, 4, generator=g), size=(hh, ww), mode="bilinear", align_corners=False)[0, 0]
    f = 1.0 + 0.22 * f
    r = ((yy - 0.5) ** 2 + (xx - 0.5) ** 2).sqrt() * 1.4142
    f = f - 0.45 * (r - 0.4).clamp_min(0.0)
    for _ in range(3):
        by = torch.rand(1, generator=g).item(); bx = torch.rand(1, generator=g).item()
        rad = float(torch.rand(1, generator=g).item()) * (0.25 - 0.08) + 0.08
        amp = float(torch.rand(1, generator=g).item()) * (2.2 - 0.6) + 0.6
        f = f + amp * torch.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2.0 * rad * rad)))
    f = (f.clamp(0.15, 3.5) / f.mean())
    f = F.interpolate(f.view(1, 1, hh, ww), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    xN = rgb * level * f.view(1, h, w)
    gain = (torch.rand(3, generator=g) * (1.15 - 0.85) + 0.85) * torch.tensor([0.92, 0.98, 1.12])
    xN = (xN * gain.view(3, 1, 1)).clamp_min(0.0)
    shot = (xN / 70.0).sqrt()
    xN = (xN + torch.randn_like(xN) * (shot + 0.012)).clamp(0.0, 1.0)
    return torch.cat([xN, aux], dim=0)


def prep(x):
    return (T.functional.resize(x, (640, 640)) - MEAN) / STD


def load_model():
    cfg = YAMLConfig(CFG)
    sd = torch.load(CK, map_location="cpu")
    sd = sd["ema"]["module"] if "ema" in sd else sd["model"]
    cfg.model.load_state_dict(sd, strict=False)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()
        def forward(self, images, orig):
            return self.post(self.model(images), orig)
    return M().cuda().eval()


@torch.no_grad()
def infer(model, x, W, H):
    out = model(prep(x).unsqueeze(0).cuda(), torch.tensor([[W, H]]).cuda())
    labels, boxes, scores = out
    keep = scores[0] > THR
    return [(int(l), b.tolist(), float(s)) for l, b, s in zip(labels[0][keep], boxes[0][keep], scores[0][keep])]


def gt_for(name):
    ann = json.load(open(ANN))
    imid = next(i["id"] for i in ann["images"] if i["file_name"] == name)
    gts = []
    for a in ann["annotations"]:
        if a["image_id"] == imid:
            x, y, w, h = a["bbox"]
            gts.append((label_name(a["category_id"]), [x, y, x + w, y + h]))
    return imid, gts


def run_eval(model):
    ann = json.load(open(ANN))
    id2name = {i["id"]: i["file_name"] for i in ann["images"]}
    res = {"day": [], "night": [], "noaug": []}
    per_img = []
    for im in ann["images"]:
        name = im["file_name"]
        # aux availability
        if not (os.path.exists(_resolve(HEAT_DIR, name)) and os.path.exists(_resolve(RAWIR_DIR, name)) and os.path.exists(_resolve(DEPTH_DIR, name))):
            continue
        x, rgb_t, W, H = build_six(name)
        xN = nightify(x, LEVEL, seed=42)
        x0 = torch.cat([xN[:3], torch.zeros_like(x[3:])], dim=0)
        pd = infer(model, x, W, H)
        pn = infer(model, xN, W, H)
        p0 = infer(model, x0, W, H)
        imid = im["id"]
        for mode, plist in (("day", pd), ("night", pn), ("noaug", p0)):
            per_img.append((mode, imid, len(plist)))
            for lab, (x1, y1, x2, y2), sc in plist:
                res[mode].append(dict(image_id=imid, category_id=CAT_ID[lab],
                                       bbox=[x1, y1, x2 - x1, y2 - y1], score=sc))
    cocoGt = COCO(ANN)
    out = {}
    for mode in ("day", "night", "noaug"):
        cocoDt = cocoGt.loadRes(res[mode])
        E = COCOeval(cocoGt, cocoDt, "bbox")
        E.evaluate(); E.accumulate(); E.summarize()
        out[mode] = E.stats[0]
    return out, per_img


def pick_three():
    ann = json.load(open(ANN))
    cnt = {}
    for a in ann["annotations"]:
        cnt[a["image_id"]] = cnt.get(a["image_id"], 0) + 1
    id2name = {i["id"]: i["file_name"] for i in ann["images"]}
    chosen = []
    for imid in sorted(cnt, key=cnt.get, reverse=True):
        name = id2name[imid]
        if cnt[imid] >= 20 and os.path.exists(_resolve(HEAT_DIR, name)) and os.path.exists(_resolve(RAWIR_DIR, name)) and os.path.exists(_resolve(DEPTH_DIR, name)):
            chosen.append(name)
        if len(chosen) == 3:
            break
    return chosen


def draw(ax, rgb_np, preds, gts, title):
    ax.imshow(rgb_np.clip(0, 1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    for nm, (x1, y1, x2, y2) in gts:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linestyle="--", linewidth=1.1))
    for lab, (x1, y1, x2, y2), sc in preds:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1.3))
        ax.text(x1, y1 - 2, f"{lab} {sc:.2f}", color="lime", fontsize=5.5,
                bbox=dict(facecolor="black", alpha=0.4, pad=0.3))


def make_figure(model, names):
    fig, ax = plt.subplots(len(names), 3, figsize=(15, 5 * len(names)))
    cols = ["Day (RGB+IR+depth)", f"Night lvl={LEVEL} (RGB+IR+depth)", "Night (RGB only, NO aug)"]
    for ri, name in enumerate(names):
        x, rgb_t, W, H = build_six(name)
        xN = nightify(x, LEVEL, seed=42)
        x0 = torch.cat([xN[:3], torch.zeros_like(x[3:])], dim=0)
        pd = infer(model, x, W, H)
        pn = infer(model, xN, W, H)
        p0 = infer(model, x0, W, H)
        imid, gts = gt_for(name)
        ims = [rgb_t.permute(1, 2, 0).numpy(), xN[:3].permute(1, 2, 0).numpy(), x0[:3].permute(1, 2, 0).numpy()]
        pls = [pd, pn, p0]
        for ci in range(3):
            draw(ax[ri, ci], ims[ci], pls[ci], gts, f"{cols[ci]}\n{name}  pred={len(pls[ci])} gt={len(gts)}")
    fig.suptitle("Validation set — green=pred, red=GT (dashed). Augmentation = IR-heat+raw-IR+depth.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG_DIR, "vis_val_aug.png")
    fig.savefig(out, dpi=105)
    print(f"[fig] saved -> {out}")


def main():
    model = load_model()
    print("[eval] running full validation mAP (day / night / night-noaug) ...")
    ap, per_img = run_eval(model)
    print(f"  mAP  DAY-FULL   = {ap['day']:.3f}")
    print(f"  mAP  NIGHT-FULL = {ap['night']:.3f}   (midboost at night)")
    print(f"  mAP  NIGHT-NOAUG= {ap['noaug']:.3f}   (RGB only, no augmentation)")
    gain = ap["night"] - ap["noaug"]
    print(f"  augmentation gain at night (NIGHT-FULL - NIGHT-NOAUG) = {gain:+.3f}")
    print(f"  night drop vs day (DAY - NIGHT) = {ap['day']-ap['night']:+.3f}")
    names = pick_three()
    print(f"[fig] picks: {names}")
    make_figure(model, names)


if __name__ == "__main__":
    main()
