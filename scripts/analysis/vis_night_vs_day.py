"""Single-image qualitative test of the trained midboost detector.

For one validation scene we build the FULL 6-channel input the model expects
(RGB + IR-heat + raw-IR + depth), then run it twice:
  * NIGHT  : RGB darkened with the project's own RandomNight5ch math (level=0.05)
  * DNTD   : original (daytime) RGB
IR / depth aux are identical for both, so this isolates how the detector behaves
when only the visible channel goes dark. Predicted boxes (green) and GT boxes
(red dashed) are overlaid for direct comparison.

Usage:
  /home/constant/miniconda3/envs/deimv2/bin/python vis_night_vs_day.py [FILE_NAME] [NIGHT_LEVEL]
"""
import os, sys, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

MEAN = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)

CATS = {c["id"]: c["name"] for c in json.load(open(ANN))["categories"]}
# DEIM label index -> category (sorted by id, 0-based)
CAT_LIST = [c["name"] for c in sorted(json.load(open(ANN))["categories"], key=lambda c: c["id"])]
def label_name(l):
    try:
        return CAT_LIST[int(l)]
    except Exception:
        return str(l)


def read_ch(folder, name, n_ch):
    p_png = os.path.join(folder, os.path.splitext(name)[0] + ".png")
    p_jpg = os.path.join(folder, name)
    path = p_png if os.path.exists(p_png) else p_jpg
    im = Image.open(path)
    if n_ch == 1:
        im = im.convert("L")
        return T.functional.pil_to_tensor(im).float() / 255.0
    im = im.convert("RGB")
    t = T.functional.pil_to_tensor(im).float() / 255.0
    return t.mean(dim=0, keepdim=True)


def read_depth(folder, name):
    base = os.path.splitext(name)[0]
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(folder, base + ext)
        if os.path.exists(p):
            arr = np.array(Image.open(p)).astype("float32") if False else None
            from PIL import Image
            arr = np.array(Image.open(p)).astype("float32")
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            arr = arr / 19999.0 if arr.max() > 1000.0 else arr / 255.0
            return torch.from_numpy(arr.clip(0.0, 1.0))[None]
    raise FileNotFoundError(os.path.join(folder, base + ".(png|jpg)"))


import numpy as np


def nightify(x, level, n_lamps=3, seed=0):
    """Faithful re-implementation of RandomNight5ch.transform (RGB only, aux pass-through)."""
    g = torch.Generator().manual_seed(seed)
    rgb = x[:3]
    aux = x[3:]
    h, w = rgb.shape[-2:]
    hh, ww = max(h // 8, 4), max(w // 8, 4)
    yy = torch.linspace(0.0, 1.0, hh).view(hh, 1)
    xx = torch.linspace(0.0, 1.0, ww).view(1, ww)
    coarse = torch.randn(1, 1, 4, 4, generator=g)
    f = F.interpolate(coarse, size=(hh, ww), mode="bilinear", align_corners=False)[0, 0]
    f = 1.0 + 0.22 * f
    r = ((yy - 0.5) ** 2 + (xx - 0.5) ** 2).sqrt() * 1.4142
    f = f - 0.45 * (r - 0.4).clamp_min(0.0)
    for _ in range(n_lamps):
        by = torch.rand(1, generator=g).item()
        bx = torch.rand(1, generator=g).item()
        rad = float(torch.rand(1, generator=g).item()) * (0.25 - 0.08) + 0.08
        amp = float(torch.rand(1, generator=g).item()) * (2.2 - 0.6) + 0.6
        f = f + amp * torch.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2.0 * rad * rad)))
    f = f.clamp(0.15, 3.5)
    f = f / f.mean()
    f = F.interpolate(f.view(1, 1, hh, ww), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    xN = rgb * level * f.view(1, h, w)
    gain = (torch.rand(3, generator=g) * (1.15 - 0.85) + 0.85) * torch.tensor([0.92, 0.98, 1.12])
    gain = gain.view(3, 1, 1)
    xN = xN * gain
    xN = xN.clamp_min(0.0)
    shot = (xN / 70.0).sqrt()
    xN = xN + torch.randn_like(xN) * (shot + 0.012)
    xN = xN.clamp(0.0, 1.0)
    return torch.cat([xN, aux], dim=0)


def build_six(name):
    rgb = Image.open(os.path.join(IMG_DIR, name)).convert("RGB")
    W, H = rgb.size
    rgb_t = T.functional.pil_to_tensor(rgb).float() / 255.0
    heat = read_ch(HEAT_DIR, name, 1)
    raw = read_ch(RAWIR_DIR, name, 1)
    dep = read_depth(DEPTH_DIR, name)
    heat = T.functional.resize(heat, (H, W))
    raw = T.functional.resize(raw, (H, W))
    dep = T.functional.resize(dep, (H, W))
    x = torch.cat([rgb_t, heat, raw, dep], dim=0)  # [6,H,W] in [0,1]
    return x, rgb_t, W, H


def prep(x):
    x = T.functional.resize(x, (640, 640))
    return (x - MEAN) / STD


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
def infer(model, x, W, H, thr=0.3):
    out = model(prep(x).unsqueeze(0).cuda(), torch.tensor([[W, H]]).cuda())
    labels, boxes, scores = out
    labels, boxes, scores = labels[0], boxes[0], scores[0]
    keep = scores > thr
    return labels[keep].cpu(), boxes[keep].cpu(), scores[keep].cpu()


def gt_boxes(name):
    ann = json.load(open(ANN))
    imid = next(i["id"] for i in ann["images"] if i["file_name"] == name)
    res = []
    for a in ann["annotations"]:
        if a["image_id"] == imid:
            x, y, w, h = a["bbox"]
            res.append((label_name(a["category_id"]), [x, y, x + w, y + h]))
    return res


def draw(ax, rgb_np, preds, gts, title):
    ax.imshow(rgb_np)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    for name, (x1, y1, x2, y2) in gts:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                    edgecolor="red", linestyle="--", linewidth=1.2))
    for lab, (x1, y1, x2, y2), sc in preds:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                    edgecolor="lime", linewidth=1.4))
        ax.text(x1, y1 - 2, f"{lab} {sc:.2f}", color="lime", fontsize=6,
                bbox=dict(facecolor="black", alpha=0.4, pad=0.5))


def pick_name():
    ann = json.load(open(ANN))
    cnt = {}
    for a in ann["annotations"]:
        cnt[a["image_id"]] = cnt.get(a["image_id"], 0) + 1
    ims = {i["id"]: i["file_name"] for i in ann["images"]}
    for imid in sorted(cnt, key=cnt.get, reverse=True):
        n = ims[imid]
        ok = (os.path.exists(os.path.join(IMG_DIR, n)) and
              os.path.exists(os.path.join(HEAT_DIR, os.path.splitext(n)[0] + ".png")) and
              os.path.exists(os.path.join(RAWIR_DIR, os.path.splitext(n)[0] + ".png")))
        if ok:
            return n
    return sorted(ims.values())[0]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else pick_name()
    level = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    print(f"[vis] image={name} night_level={level}")

    model = load_model()
    x, rgb_t, W, H = build_six(name)

    x_night = nightify(x, level, n_lamps=3, seed=42)

    lab_n, box_n, sc_n = infer(model, x_night, W, H)
    lab_d, box_d, sc_d = infer(model, x, W, H)
    preds_n = [(label_name(l), b.tolist(), float(s)) for l, b, s in zip(lab_n, box_n, sc_n)]
    preds_d = [(label_name(l), b.tolist(), float(s)) for l, b, s in zip(lab_d, box_d, sc_d)]
    gts = gt_boxes(name)
    print(f"[vis] night preds={len(preds_n)} day preds={len(preds_d)} gt={len(gts)}")

    night_rgb = x_night[:3].permute(1, 2, 0).numpy()
    day_rgb = rgb_t.permute(1, 2, 0).numpy()
    heat = x[3:4].permute(1, 2, 0).numpy().squeeze()
    dep = x[5:6].permute(1, 2, 0).numpy().squeeze()

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    draw(ax[0, 0], night_rgb, preds_n, gts, f"NIGHT (lvl={level})  pred={len(preds_n)}")
    draw(ax[0, 1], day_rgb, preds_d, gts, f"DNTD / daytime  pred={len(preds_d)}")
    ax[1, 0].imshow(heat, cmap="inferno"); ax[1, 0].set_title("IR heat (aux, shared)", fontsize=9); ax[1, 0].axis("off")
    ax[1, 1].imshow(dep, cmap="magma"); ax[1, 1].set_title("depth (aux, shared)", fontsize=9); ax[1, 1].axis("off")
    fig.suptitle(f"{name}   green=pred  red=GT   (IR/depth aux identical for both)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG_DIR, "vis_night_vs_day.png")
    fig.savefig(out, dpi=110)
    print(f"[vis] saved -> {out}")


if __name__ == "__main__":
    main()
