#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DINOv3STAsRDTTrack 多模态融合架构图 (RGB 主流 + IR/Depth 作为 prompt 注入)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR

OUT = os.path.join(FIG_DIR, "arch_rdttrack.png")

fig, ax = plt.subplots(figsize=(13.5, 9))
ax.set_xlim(0, 13.5)
ax.set_ylim(-0.3, 10.2)
ax.axis("off")


def box(x, y, w, h, text, fc="#eef3fb", ec="#4472c4", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=1.7, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=4, linespacing=1.45)


def arr(x1, y1, x2, y2, label="", color="#444", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.6, color=color,
                                 zorder=3, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2 + 0.06, label,
                fontsize=8, color="#555", zorder=5)


# ---- row 1: input ----
box(4.3, 9.0, 4.9, 0.85, "INPUT  6ch\nRGB(3) + heat(1) + rawIR(1) + depth(1)",
    fc="#dfe6f2", ec="#2f5597", fs=9.5)
arr(5.3, 9.0, 1.9, 8.15)
arr(6.75, 9.0, 6.5, 8.15)
arr(8.2, 9.0, 11.1, 8.15)

# ---- row 2: encoders ----
box(0.3, 7.05, 3.3, 1.1, "RGB (3ch)\nrgb_dinov3  vits16plus\npretrained / FROZEN\nlast 3 blocks unfrozen",
    fc="#e8f0fe", ec="#2f5597")
box(4.85, 7.05, 3.3, 1.1, "IR (2ch: heat + raw)\nir_dinov3\nfull DINOv3",
    fc="#fff2e0", ec="#c47b1a")
box(9.4, 7.05, 3.3, 1.1, "Depth (1ch)\ndepth_dinov3\nfull DINOv3",
    fc="#f3e8f8", ec="#7b3fa0")

# ---- row 3: feature maps ----
box(0.3, 5.3, 3.3, 0.85, "rgb_feats\nc2 / c3 / c4", fc="#e8f0fe", ec="#2f5597")
box(4.85, 5.3, 3.3, 0.85, "ir_norms  (BatchNorm)", fc="#fff2e0", ec="#c47b1a")
box(9.4, 5.3, 3.3, 0.85, "depth_norms  (BatchNorm)", fc="#f3e8f8", ec="#7b3fa0")
arr(1.95, 7.05, 1.95, 6.15)
arr(6.5, 7.05, 6.5, 6.15)
arr(11.05, 7.05, 11.05, 6.15)

# ---- row 4: orthogonal fusion ----
box(5.1, 3.85, 6.4, 0.95,
    "dt_ort   ORTHOGONAL PROJECTION\nremove IR<->Depth redundancy, keep complementary only\n(*** RGB is NOT touched here ***)",
    fc="#fff9e0", ec="#b8860b", fs=8.6)
arr(6.5, 5.3, 6.9, 4.8)
arr(11.05, 5.3, 10.4, 4.8)

# ---- row 5: prompt block ----
box(4.4, 2.4, 3.9, 0.95, "PromptBlock (Fovea)\ncat[ rgb , dt ]  ->  learned residual",
    fc="#e6f6ec", ec="#2e7d4f")
arr(8.3, 3.85, 7.4, 3.35)
arr(1.95, 5.3, 1.95, 2.85, rad=0.28, color="#2f5597")   # rgb_feats -> (down to add)
arr(1.95, 2.85, 4.4, 2.85, "rgb", color="#2f5597")

# ---- row 6: residual add ----
box(0.3, 0.95, 3.3, 0.95, "fused = rgb + prompt\n(c2/c3)     c4 = pure RGB",
    fc="#e6f6ec", ec="#2e7d4f")
arr(6.35, 2.4, 3.6, 1.5)
arr(1.95, 5.3, 1.95, 1.9, rad=0.0, color="#2f5597")

# ---- sta branch ----
box(9.3, 2.4, 3.5, 0.95, "sta(RGB) -> detail\nspatial prior module",
    fc="#f0f0f0", ec="#666")
arr(1.95, 7.05, 11.05, 3.35, rad=-0.30, color="#888")
arr(11.05, 2.4, 11.05, 1.95, color="#888")

# ---- output ----
box(4.0, -0.25, 6.2, 0.9,
    "concat[ fused , detail ]  ->  1x1 conv  ->  BN\nOUTPUT: 3 scales x 256ch  ->  detection head",
    fc="#dfe6f2", ec="#2f5597", fs=9)
arr(3.6, 0.95, 6.0, 0.68)
arr(11.05, 2.4, 8.4, 0.68, rad=0.25, color="#888")

ax.set_title("DINOv3STAsRDTTrack  -  RGB main stream + IR/Depth prompt injection",
             fontsize=12.5, pad=10)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved:", OUT)
