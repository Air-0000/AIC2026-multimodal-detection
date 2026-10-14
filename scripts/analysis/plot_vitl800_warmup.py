#!/usr/bin/env python3
"""Why did vitl800_clean jump to ~0.513 within 6 epochs?

Answer hint: it did NOT "learn detection from scratch" in 6 epochs. Only ~0.8% of
the weights (the cross-modal injection path) started random, and those are tiny
conv/prompt modules sitting on top of an already-trained DINOv3 ViT-L + a decoder
inherited from a converged detector. So epochs 0-6 are a *repair* phase where the
random fusion stops corrupting the frozen backbone, and the detector snaps back to
its pre-trained competence.

Panels:
  1) AP / AP50 per epoch  -> shows the repair curve, AP50 recovering first
  2) box/giou/fgl losses  -> shows WHAT converged (localisation, not semantics)
  3) train lr             -> shows the ViT-L backbone was nearly frozen meanwhile
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTS = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs"
FIG = "/home/constant/CodeBuddy/Dev/AIC/outputs/figures/vitl800_clean_warmup_why.png"

VITL = "deimv2_dinov3_l_aic_vitl800_clean"
COMBO = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo"


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
            rows.append(d)
    rows.sort(key=lambda d: d.get("epoch", -1))
    return rows


def main():
    vitl = load(VITL)
    combo = load(COMBO)

    print("=== vitl800_clean per epoch ===")
    print(f"{'ep':>3} {'AP':>7} {'AP50':>7} {'loss':>7} {'mal':>6} {'bbox':>6} "
          f"{'giou':>6} {'fgl':>6} {'lr':>10}")
    for d in vitl:
        ap = d["test_coco_eval_bbox"]
        print(f"{d.get('epoch', -1):>3} {ap[0]:>7.4f} {ap[1]:>7.4f} "
              f"{d.get('train_loss', 0):>7.2f} {d.get('train_loss_mal', 0):>6.3f} "
              f"{d.get('train_loss_bbox', 0):>6.3f} {d.get('train_loss_giou', 0):>6.3f} "
              f"{d.get('train_loss_fgl', 0):>6.3f} {d.get('train_lr', 0):>10.3e}")

    print(f"\ncombo epochs={len(combo)}  best AP={max(r['test_coco_eval_bbox'][0] for r in combo):.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11))
    ax1, ax2, ax3 = axes

    ep = [d.get("epoch", -1) for d in vitl]
    ap = [d["test_coco_eval_bbox"][0] for d in vitl]
    ap50 = [d["test_coco_eval_bbox"][1] for d in vitl]

    # ---- panel 1: AP / AP50 ----
    ax1.plot(ep, ap, "-o", color="#C44E52", ms=5, lw=2, label="vitl800_clean AP @.50:.95")
    ax1.plot(ep, ap50, "-s", color="#DD8452", ms=4.5, lw=1.8, label="vitl800_clean AP50")
    cep = [d.get("epoch", -1) for d in combo if d.get("epoch", -1) <= max(ep)]
    cap = [d["test_coco_eval_bbox"][0] for d in combo if d.get("epoch", -1) <= max(ep)]
    ax1.plot(cep, cap, "--^", color="#4C72B0", ms=4, lw=1.5,
             label="midboost_combo AP @.50:.95 (ViT-S/640)")
    combobest = max(r["test_coco_eval_bbox"][0] for r in combo)
    ax1.axhline(combobest, color="#4C72B0", ls=":", lw=1.3)
    ax1.text(0.15, combobest + 0.006, f"combo best {combobest:.4f} @ep49",
             fontsize=8.5, color="#4C72B0")

    # annotate the three phases
    ax1.annotate("random fusion injects noise\ninto a strong frozen backbone",
                 xy=(0, ap[0]), xytext=(0.7, 0.20), fontsize=8.5, color="#C44E52",
                 arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1))
    ax1.annotate(f"repair done: AP {ap[-1]:.3f}\n= pretrained competence restored",
                 xy=(ep[-1], ap[-1]), xytext=(ep[-1] - 4.3, ap[-1] - 0.085),
                 fontsize=8.5, color="#C44E52",
                 arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1))

    ax1.set_ylabel("val mAP")
    ax1.set_title("vitl800_clean: epochs 0-6 are a REPAIR phase, not learning-from-scratch\n"
                  "(only 0.8% of params / 2.9M -- the cross-modal path -- started random)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8.5, loc="lower right")

    # ---- panel 2: which loss component moved ----
    for key, color, lab in [("train_loss_mal", "#937860", "mal (classification)"),
                            ("train_loss_bbox", "#4C72B0", "bbox"),
                            ("train_loss_giou", "#C44E52", "giou"),
                            ("train_loss_fgl", "#55A868", "fgl (fine-grained loc)")]:
        ax2.plot(ep, [d.get(key, float("nan")) for d in vitl], "-o", color=color,
                 ms=3.5, lw=1.6, label=lab)
    ax2.set_ylabel("train loss component")
    ax2.set_xlabel("epoch")
    ax2.set_title("What actually converged: localisation heads, not semantics "
                  "(bbox 0.83->0.18, giou 1.83->0.73)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8.5, ncol=4, loc="upper right")

    # ---- panel 3: lr ----
    ax3.plot(ep, [d.get("train_lr", 0) for d in vitl], "-o", color="#C44E52", ms=4,
             lw=1.8, label="vitl800_clean  ViT-L backbone lr (LLRD capped)")
    cep_all = [d.get("epoch", -1) for d in combo if d.get("epoch", -1) <= max(ep)]
    ax3.plot(cep_all, [d.get("train_lr", 0) for d in combo if d.get("epoch", -1) <= max(ep)],
             "--^", color="#4C72B0", ms=4, lw=1.5, label="midboost_combo  ViT-S backbone lr")
    ax3.axhline(1e-4, color="#55A868", ls=":", lw=1.3)
    ax3.text(0.15, 1.08e-4, "intended RGB lr = 1e-4 (yml value, silently ignored)",
             fontsize=8.5, color="#55A868")
    ax3.set_yscale("log")
    ax3.set_ylabel("lr")
    ax3.set_xlabel("epoch")
    ax3.set_title("Meanwhile the new big backbone was nearly frozen: "
                  "peak 8.87e-6 instead of 1e-4")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print("\nsaved", FIG)


if __name__ == "__main__":
    main()
