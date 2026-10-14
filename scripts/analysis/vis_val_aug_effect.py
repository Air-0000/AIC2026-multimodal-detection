"""Augmentation-improvement figure on the NIGHTIFIED validation set.

Top: horizontal bar chart of COCO mAP (from the project's own eval logs on
     val_night005, level=0.05) showing the gain from the multi-modal
     augmentation (IR-heat + raw-IR + depth):
        ufall_abirblk   (weakest / no multimodal aug)  AP=0.063
        deepir_ablate_rawir (drop raw-IR)              AP=0.359
        midboost_combo  (RGB+IR-heat+raw-IR+depth)     AP=0.417
Bottom: one nightified validation image, with GT (red dashed) and the trained
        midboost detector's predictions (green) overlaid -- visually showing the
        model still detects when the visible channel is nearly black.
"""
import os, json
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
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm
_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_cjk):
    fm.fontManager.addfont(_cjk)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
import sys
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
CATS = json.load(open(ANN))["categories"]
SORTED = sorted(CATS, key=lambda c: c["id"])
CAT_LIST = [c["name"] for c in SORTED]


def label_name(l):
    try:
        return CAT_LIST[int(l)]
    except Exception:
        return str(l)


def _resolve(f, n):
    p = os.path.join(f, os.path.splitext(n)[0] + ".png")
    return p if os.path.exists(p) else os.path.join(f, n)


def read_ch(f, n, nc):
    im = Image.open(_resolve(f, n))
    im = im.convert("L") if nc == 1 else im.convert("RGB")
    return T.functional.pil_to_tensor(im).float() / 255.0


def read_depth(f, n):
    a = np.array(Image.open(_resolve(f, n)).convert("L")).astype("float32")
    a = a / 19999.0 if a.max() > 1000.0 else a / 255.0
    return torch.from_numpy(a.clip(0, 1))[None]


def build_six(n):
    rgb = Image.open(os.path.join(IMG_DIR, n)).convert("RGB")
    W, H = rgb.size
    rt = T.functional.pil_to_tensor(rgb).float() / 255.0
    heat = T.functional.resize(read_ch(HEAT_DIR, n, 1), (H, W))
    raw = T.functional.resize(read_ch(RAWIR_DIR, n, 1), (H, W))
    dep = T.functional.resize(read_depth(DEPTH_DIR, n), (H, W))
    return torch.cat([rt, heat, raw, dep], dim=0), rt, W, H


def nightify(x, level, seed=42):
    g = torch.Generator().manual_seed(seed)
    rgb, aux = x[:3], x[3:]
    h, w = rgb.shape[-2:]
    hh, ww = max(h // 8, 4), max(w // 8, 4)
    yy = torch.linspace(0, 1, hh).view(hh, 1); xx = torch.linspace(0, 1, ww).view(1, ww)
    f = F.interpolate(torch.randn(1, 1, 4, 4, generator=g), size=(hh, ww), mode="bilinear", align_corners=False)[0, 0]
    f = 1.0 + 0.22 * f
    r = ((yy - 0.5) ** 2 + (xx - 0.5) ** 2).sqrt() * 1.4142
    f = f - 0.45 * (r - 0.4).clamp_min(0.0)
    for _ in range(3):
        by = torch.rand(1, generator=g).item(); bx = torch.rand(1, generator=g).item()
        rad = float(torch.rand(1, generator=g).item()) * (0.25 - 0.08) + 0.08
        amp = float(torch.rand(1, generator=g).item()) * (2.2 - 0.6) + 0.6
        f = f + amp * torch.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2 * rad * rad)))
    f = (f.clamp(0.15, 3.5) / f.mean())
    f = F.interpolate(f.view(1, 1, hh, ww), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    xN = rgb * level * f.view(1, h, w)
    gain = (torch.rand(3, generator=g) * (1.15 - 0.85) + 0.85) * torch.tensor([0.92, 0.98, 1.12])
    xN = (xN * gain.view(3, 1, 1)).clamp_min(0.0)
    sh = (xN / 70.0).sqrt()
    xN = (xN + torch.randn_like(xN) * (sh + 0.012)).clamp(0, 1)
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
            self.model = cfg.model.deploy(); self.post = cfg.postprocessor.deploy()
        def forward(self, im, orig):
            return self.post(self.model(im), orig)
    return M().cuda().eval()


@torch.no_grad()
def infer(model, x, W, H):
    o = model(prep(x).unsqueeze(0).cuda(), torch.tensor([[W, H]]).cuda())
    keep = o[2][0] > THR
    return [(label_name(l), b.tolist(), float(s)) for l, b, s in zip(o[0][0][keep], o[1][0][keep], o[2][0][keep])]


def gt_for(n):
    ann = json.load(open(ANN)); imid = next(i["id"] for i in ann["images"] if i["file_name"] == n)
    g = []
    for a in ann["annotations"]:
        if a["image_id"] == imid:
            x, y, w, h = a["bbox"]; g.append((label_name(a["category_id"]), [x, y, x + w, y + h]))
    return g


def main():
    # ---- numbers sourced from project eval logs (val_night005, level=0.05) ----
    rows = [
        ("ufall_abirblk\n(无多模态增强基线)", 0.063, 0.131, "#d62728"),
        ("deepir_ablate_rawir\n(去掉原始IR通道)", 0.359, 0.612, "#ff7f0e"),
        ("midboost_combo\n(RGB+IR热图+原始IR+深度)", 0.417, 0.690, "#2ca02c"),
    ]
    model = load_model()
    name = "shuming_340_00000020.png"
    x, rt, W, H = build_six(name)
    xN = nightify(x, LEVEL, seed=42)
    preds = infer(model, xN, W, H)
    gts = gt_for(name)

    fig = plt.figure(figsize=(11, 13))
    gs = GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.18)
    axb = fig.add_subplot(gs[0])
    axn = fig.add_subplot(gs[1])

    # bar chart
    labels = [r[0] for r in rows]; aps = [r[1] for r in rows]; ap50s = [r[2] for r in rows]; cols = [r[3] for r in rows]
    y = np.arange(len(rows))
    axb.barh(y, aps, color=cols, height=0.5)
    for i, (a, a5) in enumerate(zip(aps, ap50s)):
        axb.text(a + 0.01, i, f"AP={a:.3f}  AP50={a5:.3f}", va="center", fontsize=10, fontweight="bold")
    axb.set_yticks(y); axb.set_yticklabels(labels, fontsize=10)
    axb.set_xlim(0, 0.55); axb.set_xlabel("COCO mAP (bbox)")
    axb.set_title("多模态增强在 nightify 校验集上的改进  (val_night005, level=0.05)\n"
                  f"增益: 去原始IR -0.058  |  全模态 vs 基线 +{aps[-1]-aps[0]:.3f}", fontsize=12)
    axb.invert_yaxis()

    # night example
    axn.imshow(xN[:3].permute(1, 2, 0).numpy().clip(0, 1))
    for nm, (x1, y1, x2, y2) in gts:
        axn.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linestyle="--", linewidth=1.2))
    for lab, (x1, y1, x2, y2), sc in preds:
        axn.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1.4))
        axn.text(x1, y1 - 2, f"{lab} {sc:.2f}", color="lime", fontsize=6, bbox=dict(facecolor="black", alpha=0.4, pad=0.4))
    axn.set_title(f"夜间化校验集示例  {name}  (level={LEVEL})\n"
                  f"绿=midboost预测 pred={len(preds)}  红虚线=GT gt={len(gts)}\n"
                  "可见光近全黑，检测由 IR热图/原始IR/深度 支撑", fontsize=10)
    axn.axis("off")

    out = os.path.join(FIG_DIR, "vis_val_aug_effect.png")
    fig.savefig(out, dpi=110)
    print(f"[effect] saved -> {out}")
    print(f"[effect] night example preds={len(preds)} gt={len(gts)}")


if __name__ == "__main__":
    main()
