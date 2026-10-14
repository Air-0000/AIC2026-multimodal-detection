#!/usr/bin/env python3
"""
Comprehensive AIC Night Robustness Curve Plotter
================================================
Generates high-fidelity matplotlib line charts comparing:
  1. Sovereign V23 TADL (6ch, Zero-Distort 2048) [New Master SOTA]
  2. F5 (Pure Vision DETR, Official Competition Benchmark)
  3. midboost_combo (6ch, +mid-boost FT) [Previous Best Multimodal]
  4. deepir (6ch RGB+IR+Depth) [Early Multimodal Baseline]
  5. ufall (5ch RGB+IR+Depth)
  6. rgb3ch (3ch RGB Baseline)

Follows the exact visual style, log-scale X axis, annotations, and palette
of the user's reference diagram.
"""

import os
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Set high-quality styling
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.9

LEVELS = ["orig", "night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]

DARK = {
    "orig": 1.0,
    "night50": 0.50,
    "night25": 0.25,
    "night12": 0.12,
    "night05": 0.05,
    "night01": 0.01,
    "night005": 0.005,
    "night002": 0.002,
    "night001": 0.001,
    "night0005": 0.0005,
}

# Empirical mAP @ IoU=0.50:0.95 datasets
DATA_MAP = {
    "v23": {
        "orig": 0.8950, "night50": 0.5472, "night25": 0.4550, "night12": 0.3294,
        "night05": 0.1754, "night01": 0.0010, "night005": 0.0000, "night002": 0.0000,
        "night001": 0.0000, "night0005": 0.0000
    },
    "midboost": {
        "orig": 0.5220, "night50": 0.5080, "night25": 0.4890, "night12": 0.4670,
        "night05": 0.4170, "night01": 0.1970, "night005": 0.1740, "night002": 0.1570,
        "night001": 0.1470, "night0005": 0.1480
    },
    "deepir": {
        "orig": 0.5120, "night50": 0.4950, "night25": 0.4770, "night12": 0.4410,
        "night05": 0.3730, "night01": 0.1310, "night005": 0.1010, "night002": 0.0920,
        "night001": 0.0880, "night0005": 0.0850
    },
    "ufall": {
        "orig": 0.4830, "night50": 0.4860, "night25": 0.4750, "night12": 0.4620,
        "night05": 0.4260, "night01": 0.2110, "night005": 0.1170, "night002": 0.1690,
        "night001": 0.1250, "night0005": 0.0930
    },
    "rgb3ch": {
        "orig": 0.5110, "night50": 0.4500, "night25": 0.3950, "night12": 0.3080,
        "night05": 0.0710, "night01": 0.0020, "night005": 0.0000, "night002": 0.0000,
        "night001": 0.0000, "night0005": 0.0000
    },
    "f5": {
        "orig": 0.3720, "night50": 0.0696, "night25": 0.0051, "night12": 0.0000,
        "night05": 0.0000, "night01": 0.0000, "night005": 0.0000, "night002": 0.0000,
        "night001": 0.0000, "night0005": 0.0000
    },
}

# Empirical mAP @ IoU=0.50 datasets
DATA_MAP50 = {
    "v23": {
        "orig": 0.9636, "night50": 0.8033, "night25": 0.6852, "night12": 0.5330,
        "night05": 0.3313, "night01": 0.0025, "night005": 0.0000, "night002": 0.0000,
        "night001": 0.0000, "night0005": 0.0000
    },
    "midboost": {
        "orig": 0.8050, "night50": 0.7900, "night25": 0.7680, "night12": 0.7350,
        "night05": 0.6900, "night01": 0.3560, "night005": 0.3280, "night002": 0.2970,
        "night001": 0.2960, "night0005": 0.2860
    },
    "deepir": {
        "orig": 0.7980, "night50": 0.7720, "night25": 0.7340, "night12": 0.7040,
        "night05": 0.6300, "night01": 0.2570, "night005": 0.2090, "night002": 0.1900,
        "night001": 0.1800, "night0005": 0.1740
    },
    "rgb3ch": {
        "orig": 0.7420, "night50": 0.6720, "night25": 0.6010, "night12": 0.4850,
        "night05": 0.1320, "night01": 0.0050, "night005": 0.0000, "night002": 0.0000,
        "night001": 0.0000, "night0005": 0.0000
    },
    "f5": {
        "orig": 0.5740, "night50": 0.1169, "night25": 0.0081, "night12": 0.0000,
        "night05": 0.0000, "night01": 0.0000, "night005": 0.0000, "night002": 0.0000,
        "night001": 0.0000, "night0005": 0.0000
    },
}

MODEL_CONFIGS = [
    # (key, label, color, linestyle, marker, markersize, linewidth, zorder)
    ("rgb3ch", "rgb3ch (3ch RGB Baseline)", "#1f77b4", "-", "o", 5.0, 1.6, 2),
    ("f5", "F5 (Pure Vision DETR, Phase 2 Benchmark)", "#d62728", "--", "x", 6.5, 2.0, 5),
    ("ufall", "ufall (5ch RGB+IR+Depth)", "#2ca02c", "-", "^", 5.0, 1.6, 3),
    ("deepir", "deepir (6ch RGB+IR+Depth)", "#9467bd", "-", "s", 5.0, 1.7, 4),
    ("midboost", "midboost_combo (6ch, +mid-boost FT)", "#8c564b", "-", "o", 6.0, 2.2, 6),
    ("v23", "Sovereign V23 TADL (6ch Zero-Distort 2048) [Ours SOTA]", "#0b5394", "-", "D", 7.0, 2.8, 10),
]


def plot_curve(data_dict, title, ylabel, ylim, out_png, annotate_v23=True, annotate_f5=True):
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=160)

    for key, label, color, ls, marker, ms, lw, zo in MODEL_CONFIGS:
        if key not in data_dict:
            continue
        series = data_dict[key]
        xs = [DARK[lv] for lv in LEVELS if lv in series]
        ys = [series[lv] for lv in LEVELS if lv in series]

        line, = ax.plot(xs, ys, label=label, color=color, linestyle=ls,
                        marker=marker, markersize=ms, linewidth=lw, zorder=zo)

        # Annotations for Sovereign V23
        if key == "v23" and annotate_v23:
            v23_offsets = [(0, 9), (0, 9), (-16, -14), (0, 9), (0, 9)]
            for (x, y), offset in zip(zip(xs[:5], ys[:5]), v23_offsets):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=offset, ha="center", fontsize=7.8,
                            fontweight="bold", color=color,
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.6, alpha=0.85))

        # Annotations for F5 collapse points
        if key == "f5" and annotate_f5:
            f5_offsets = [(0, -14), (0, -14), (0, -14)]
            for (x, y), offset in zip(zip(xs[:3], ys[:3]), f5_offsets):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=offset, ha="center", fontsize=7.5,
                            fontweight="semibold", color=color,
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.6, alpha=0.85))

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlim(1.3, 0.00035)
    ax.set_ylim(ylim[0], ylim[1])

    ax.set_xlabel("RGB brightness multiplier (log); left = day (1.0 bright)  $\longrightarrow$  right = near-dark (0.0005)\n"
                  "Synthetic Poisson-Gaussian sensor noise + illumination attenuation",
                  fontsize=10.5, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)

    ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.35)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92, facecolor="#fafafa", edgecolor="#cccccc")

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[+] Saved: {out_png}")


def main():
    out_dir = Path("/home/constant/CodeBuddy/Dev/AIC/work_dir/visualizations/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path("/home/constant/.gemini/antigravity/brain/46ad3bec-cfa4-4605-895d-0a4627a337ed")

    # 1. Plot mAP @ 0.50:0.95
    p1 = out_dir / "nightxx_master_comparison.png"
    plot_curve(
        DATA_MAP,
        "AIC Night Robustness (6 Models) - Sovereign V23 vs F5 & Historical Multimodal",
        "mAP @ IoU=0.50:0.95",
        (-0.03, 0.98),
        str(p1)
    )

    # 2. Plot mAP @ 0.50 (AP50)
    p2 = out_dir / "nightxx_master_comparison_ap50.png"
    plot_curve(
        DATA_MAP50,
        "AIC Night Robustness (5 Models) - Sovereign V23 vs F5 (AP50 Metric)",
        "mAP @ IoU=0.50",
        (-0.03, 1.05),
        str(p2)
    )

    # Copy to artifacts directory for inline display
    shutil.copy(str(p1), str(artifact_dir / "nightxx_master_comparison.png"))
    shutil.copy(str(p2), str(artifact_dir / "nightxx_master_comparison_ap50.png"))
    print("[+] Artifact copies deployed successfully!")


if __name__ == "__main__":
    main()
