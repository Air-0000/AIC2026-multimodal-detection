#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把同一批原图在 10 个 nightify 亮度档下成像, 拼成网格图, 直观看数据增强效果.
使用 config 中真实的 RandomNight5ch (photon_max=70, read_sigma=0.012, lamps=3).
"""
import sys, os, glob
import torch
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
IMG_DIR = f"{ROOT}/work_dir/deimv2_dataset/train/images"
OUT = os.path.join(FIG_DIR, "nightify_levels.png")

sys.path.insert(0, f"{ROOT}/vendor/DEIMv2")
from engine.data.transforms._transforms import RandomNight5ch

# 评测协议对应的曝光档 (名称, level)
LEVELS = [
    ("day",      1.0),
    ("night50",  0.5),
    ("night25",  0.25),
    ("night12",  0.12),
    ("night05",  0.05),
    ("night01",  0.01),
    ("night005", 0.005),
    ("night002", 0.002),
    ("night001", 0.001),
    ("night0005",0.0005),
]

SAMPLES = 4
RESIZE = 256

def main():
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*")))[:SAMPLES]
    paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"选了 {len(paths)} 张原图")

    tf = RandomNight5ch(p=1.0, level=(0.0005, 1.0), photon_max=70.0,
                        read_sigma=0.012, log_level=True, lamps=(3, 4),
                        color_shift=False)
    to_t = T.Compose([T.Resize((RESIZE, RESIZE)), T.ToTensor()])

    grid = []  # [sample][level] -> [3,h,w] tensor
    for si, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        base = to_t(img)                       # [3,h,w] in [0,1]
        inpt = torch.cat([base, torch.zeros(1, RESIZE, RESIZE)], dim=0)  # [4,h,w]
        row = []
        for li, (name, lvl) in enumerate(LEVELS):
            torch.manual_seed(1000 + si * 100 + li)  # 每格可复现
            params = {"apply": True, "level": float(lvl),
                      "lamps": 3, "gain": torch.ones(3, 1, 1)}
            out = tf.transform(inpt, params)     # [4,h,w]
            row.append(out[:3].clamp(0, 1))
        grid.append(row)

    # 拼图
    ncol = len(LEVELS)
    fig, axes = plt.subplots(len(paths), ncol, figsize=(ncol * 2.0, len(paths) * 2.0))
    if len(paths) == 1:
        axes = [axes]
    for si in range(len(paths)):
        for li, (name, _) in enumerate(LEVELS):
            ax = axes[si][li]
            im = grid[si][li].permute(1, 2, 0).numpy()
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if si == 0:
                ax.set_title(name, fontsize=9)
            if li == 0:
                ax.set_ylabel(f"img{si}", fontsize=8)
    fig.suptitle("RandomNight5ch across exposure levels (RGB channels)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=130)
    print(f"已保存: {OUT}")

if __name__ == "__main__":
    main()
