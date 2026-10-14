#!/usr/bin/env python3
"""Same-epoch mAP: ViT-S/640 baseline (midboost_combo) vs ViT-L/800 (vitl800_clean).

Reads the per-epoch records each training run appends to
vendor/DEIMv2/outputs/<exp>/log.txt (JSON lines) and plots test AP + train lr,
so "does a bigger backbone help" is judged at equal epochs instead of comparing
a 6-epoch run against a 50-epoch one.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTS = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs"
FIG = "/home/constant/CodeBuddy/Dev/AIC/outputs/figures/backbone_same_epoch_map.png"

RUNS = [
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo",
     "midboost_combo (ViT-S/640, RGB lr 1e-4)", "#4C72B0"),
    ("deimv2_dinov3_l_aic_vitl800_clean",
     "vitl800_clean (ViT-L/800, RGB lr capped to 8.9e-6 by LLRD)", "#C44E52"),
]


def load(exp):
    p = os.path.join(OUTS, exp, "log.txt")
    rows = []
    if not os.path.exists(p):
        return rows
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ap = d.get("test_coco_eval_bbox")
            if not ap:
                continue
            rows.append((d.get("epoch", -1), float(ap[0]), float(d.get("train_lr", 0.0))))
    return rows


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for exp, label, color in RUNS:
        rows = load(exp)
        if not rows:
            print("no data:", exp)
            continue
        rows.sort()
        ep = [r[0] for r in rows]
        ap = [r[1] for r in rows]
        ax1.plot(ep, ap, "-", color=color, marker="o", ms=3.5, lw=1.8, label=label)
        ax2.plot(ep, [r[2] for r in rows], "-", color=color, marker="o", ms=3, lw=1.5,
                 label=label)
        print(f"{exp[:40]:42s} epochs 0..{max(ep)}  max AP={max(ap):.4f} "
              f"last AP={ap[-1]:.4f}")

    ax1.set_ylabel("val mAP @ IoU=0.50:0.95")
    ax1.set_title("Same-epoch mAP: does the bigger backbone help?\n"
                  "(vitl800_clean stopped at epoch 6 — repeatedly paused for eval jobs)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.axhline(0.5222, color="#4C72B0", ls=":", lw=1.2)
    ax1.text(0.4, 0.5235, "combo best 0.5222 @ep49", fontsize=8, color="#4C72B0")

    ax2.set_yscale("log")
    ax2.set_ylabel("train lr (first param group)")
    ax2.set_xlabel("epoch")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.axhline(1e-4, color="#55A868", ls=":", lw=1.2)
    ax2.text(0.4, 1.05e-4, "intended RGB lr = 1e-4", fontsize=8, color="#55A868")

    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print("saved", FIG)


if __name__ == "__main__":
    main()
