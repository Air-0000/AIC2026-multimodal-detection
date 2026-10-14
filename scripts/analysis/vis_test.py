"""Test-set qualitative check of the trained midboost detector (no GT for test).

For ONE real test image we run the full 6-channel model twice:
  * NIGHT : visible channel darkened with the project's RandomNight5ch (level)
  * DNTD  : original (daytime) visible channel
IR-heat / raw-IR / depth aux are identical for both (real test-set aux).
Only predictions are drawn.

Usage:
  python vis_test.py [TEST_IMAGE_NAME] [NIGHT_LEVEL]
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

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
sys.path.insert(0, DEIMV2_ROOT)
os.chdir(DEIMV2_ROOT)
from engine.core import YAMLConfig

CFG = "configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost.yml"
CK = os.path.join(ROOT, "vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost/best_stg1.pth")

VIS_DIR = os.path.join(ROOT, "dataset/test/visible")
HEAT_DIR = os.path.join(ROOT, "work_dir/deimv2_heat/test")
RAWIR_DIR = os.path.join(ROOT, "work_dir/deimv2_ir/test")
DEPTH_DIR = os.path.join(ROOT, "dataset/test/depth")
ANN = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")

MEAN = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)
CAT_LIST = [c["name"] for c in sorted(json.load(open(ANN))["categories"], key=lambda c: c["id"])]
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
    rgb = Image.open(_resolve(VIS_DIR, name)).convert("RGB")
    W, H = rgb.size
    rgb_t = T.functional.pil_to_tensor(rgb).float() / 255.0
    heat = T.functional.resize(read_ch(HEAT_DIR, name, 1), (H, W))
    raw = T.functional.resize(read_ch(RAWIR_DIR, name, 1), (H, W))
    dep = T.functional.resize(read_depth(DEPTH_DIR, name), (H, W))
    x = torch.cat([rgb_t, heat, raw, dep], dim=0)
    return x, rgb_t, W, H


def nightify(x, level, n_lamps=3, seed=42):
    g = torch.Generator().manual_seed(seed)
    rgb, aux = x[:3], x[3:]
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
    f = f.clamp(0.15, 3.5) / f.mean()
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
def infer(model, x, W, H, thr=0.3):
    out = model(prep(x).unsqueeze(0).cuda(), torch.tensor([[W, H]]).cuda())
    labels, boxes, scores = out
    keep = scores[0] > thr
    return [(label_name(l), b.tolist(), float(s)) for l, b, s in zip(labels[0][keep], boxes[0][keep], scores[0][keep])]


def draw(ax, rgb_np, preds, title):
    ax.imshow(rgb_np)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    for lab, (x1, y1, x2, y2), sc in preds:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1.4))
        ax.text(x1, y1 - 2, f"{lab} {sc:.2f}", color="lime", fontsize=6,
                bbox=dict(facecolor="black", alpha=0.4, pad=0.5))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "shuming_856_00000062.png"
    level = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    print(f"[vis_test] image={name} night_level={level}")

    model = load_model()
    x, rgb_t, W, H = build_six(name)
    x_night = nightify(x, level)
    pn = infer(model, x_night, W, H)
    pd = infer(model, x, W, H)
    print(f"[vis_test] night preds={len(pn)}  dntd/day preds={len(pd)}")

    heat = x[3:4].permute(1, 2, 0).numpy().squeeze()
    dep = x[5:6].permute(1, 2, 0).numpy().squeeze()
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    draw(ax[0, 0], x_night[:3].permute(1, 2, 0).numpy(), pn, f"NIGHT (lvl={level})  {name}  pred={len(pn)}")
    draw(ax[0, 1], rgb_t.permute(1, 2, 0).numpy(), pd, f"DNTD / daytime  {name}  pred={len(pd)}")
    ax[1, 0].imshow(heat, cmap="inferno"); ax[1, 0].set_title("IR heat (aux, shared)", fontsize=9); ax[1, 0].axis("off")
    ax[1, 1].imshow(dep, cmap="magma"); ax[1, 1].set_title("depth (aux, shared)", fontsize=9); ax[1, 1].axis("off")
    fig.suptitle("Test set — green=pred (no GT). Same scene nightified vs original.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG_DIR, "vis_test.png")
    fig.savefig(out, dpi=110)
    print(f"[vis_test] saved -> {out}")


if __name__ == "__main__":
    main()
