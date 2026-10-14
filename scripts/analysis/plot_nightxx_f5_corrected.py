#!/usr/bin/env python3
"""
Master Comparison Plotter: Sovereign V24 TADL vs F5 Corrected vs Historical Baselines
=====================================================================================
Plots the full degradation curve (mAP@50:95 and mAP50) across 11 levels of synthetic night:
  1. Sovereign V24 TADL (Night Robust, 14 Epochs) [NEW Master SOTA]
  2. Sovereign V23 TADL (Zero-Distort, 72 Epochs)
  3. F5 Corrected (Tri-Modal Residual Fusion, RAW Top-100)
  4. F5 Corrected (Tri-Modal Residual Fusion, COMPACT Peak)
  5. midboost_combo (6ch, Dedicated Night Fine-Tuning)
  6. deepir (6ch, Standard RGB+IR+Depth)
  7. rgb3ch (3ch Pure Visible Baseline)
"""

import os
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/constant/CodeBuddy/Dev/AIC")
FIG_DIR = ROOT / "work_dir/visualizations/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/home/constant/.gemini/antigravity/brain/46ad3bec-cfa4-4605-895d-0a4627a337ed")

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.9

LEVELS = ["orig", "night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005", "black"]

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
    "black": 0.0001,
}

# Historical Empirical mAP @ IoU=0.50:0.95 datasets
DATA_MAP = {
    "v23": {
        "orig": 0.8950, "night50": 0.5472, "night25": 0.4550, "night12": 0.3294,
        "night05": 0.1754, "night01": 0.0010, "night005": 0.0000, "night002": 0.0000,
        "night001": 0.0000, "night0005": 0.0000, "black": 0.0000,
    },
    "midboost": {
        "orig": 0.5220, "night50": 0.5080, "night25": 0.4890, "night12": 0.4670,
        "night05": 0.4170, "night01": 0.1970, "night005": 0.1740, "night002": 0.1570,
        "night001": 0.1470, "night0005": 0.1480, "black": 0.1420,
    },
    "deepir": {
        "orig": 0.5120, "night50": 0.4950, "night25": 0.4770, "night12": 0.4410,
        "night05": 0.3730, "night01": 0.1310, "night005": 0.1010, "night002": 0.0920,
        "night001": 0.0880, "night0005": 0.0850, "black": 0.0810,
    },
    "rgb3ch": {
        "orig": 0.5110, "night50": 0.4500, "night25": 0.3950, "night12": 0.3080,
        "night05": 0.0710, "night01": 0.0020, "night005": 0.0000, "night002": 0.0000,
        "night001": 0.0000, "night0005": 0.0000, "black": 0.0000,
    },
}

DATA_MAP50 = {
    "v23": {
        "orig": 0.9636, "night50": 0.8033, "night25": 0.6852, "night12": 0.5330,
        "night05": 0.3313, "night01": 0.0025, "night005": 0.0000, "night002": 0.0000,
        "night001": 0.0000, "night0005": 0.0000, "black": 0.0000,
    },
    "midboost": {
        "orig": 0.8050, "night50": 0.7900, "night25": 0.7680, "night12": 0.7350,
        "night05": 0.6900, "night01": 0.3560, "night005": 0.3280, "night002": 0.2970,
        "night001": 0.2960, "night0005": 0.2860, "black": 0.2780,
    },
    "deepir": {
        "orig": 0.7980, "night50": 0.7720, "night25": 0.7340, "night12": 0.7040,
        "night05": 0.6300, "night01": 0.2570, "night005": 0.2090, "night002": 0.1900,
        "night001": 0.1800, "night0005": 0.1740, "black": 0.1680,
    },
    "rgb3ch": {
        "orig": 0.7420, "night50": 0.6720, "night25": 0.6010, "night12": 0.4850,
        "night05": 0.1320, "night01": 0.0050, "night005": 0.0000, "night002": 0.0000,
        "night001": 0.0000, "night0005": 0.0000, "black": 0.0000,
    },
}


def load_json_data():
    # 1. Load F5 Corrected
    f5_path = ROOT / "outputs/analysis/nightxx_eval/f5_corrected_nightxx_results.json"
    f5_raw_map, f5_raw_map50 = {}, {}
    f5_comp_map, f5_comp_map50 = {}, {}
    if f5_path.exists():
        with open(f5_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("results", []):
            lvl = entry["level"]
            f5_raw_map[lvl] = entry["raw"]["mAP_50_95"]
            f5_raw_map50[lvl] = entry["raw"]["mAP_50"]
            f5_comp_map[lvl] = entry["compact"]["mAP_50_95"]
            f5_comp_map50[lvl] = entry["compact"]["mAP_50"]

    # 2. Load V24 TADL
    v24_path = ROOT / "outputs/analysis/nightxx_eval/nightxx_v24_tadl_curve.json"
    v24_map, v24_map50 = {}, {}
    if v24_path.exists():
        with open(v24_path, "r", encoding="utf-8") as f:
            v24_data = json.load(f)
        for entry in v24_data:
            lvl = entry["level"]
            v24_map[lvl] = entry["mAP"]
            v24_map50[lvl] = entry["mAP50"]

    return (f5_raw_map, f5_raw_map50), (f5_comp_map, f5_comp_map50), (v24_map, v24_map50)


def plot_comparison():
    (f5_raw_map, f5_raw_map50), (f5_comp_map, f5_comp_map50), (v24_map, v24_map50) = load_json_data()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8.0), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")

    # Plot 1: mAP@50:95
    ax1 = axes[0]
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("AIC Synthetic Night Robustness Spectrum (mAP@50:95)", fontsize=15, fontweight="bold", pad=12)
    ax1.set_xlabel("Illumination / Degrade Level (Relative Brightness)", fontsize=12, labelpad=8)
    ax1.set_ylabel("COCO mAP @ IoU=0.50:0.95", fontsize=12, labelpad=8)
    ax1.set_ylim(-0.02, 0.95)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

    # Plot 2: mAP50
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("AIC Synthetic Night Robustness Spectrum (mAP@50)", fontsize=15, fontweight="bold", pad=12)
    ax2.set_xlabel("Illumination / Degrade Level (Relative Brightness)", fontsize=12, labelpad=8)
    ax2.set_ylabel("COCO mAP @ IoU=0.50", fontsize=12, labelpad=8)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

    x_indices = list(range(len(LEVELS)))

    # Historical Models config: (key, label, color, marker, linestyle, linewidth, zorder)
    models_config = [
        ("v23", "Sovereign V23 TADL (Baseline 6ch)", "#FF7F0E", "s", "--", 1.8, 6),
        ("midboost", "midboost_combo (6ch Night FT)", "#2CA02C", "^", "-.", 1.8, 5),
        ("rgb3ch", "rgb3ch (Visible Baseline)", "#7F7F7F", "x", ":", 1.5, 3),
    ]

    for key, label, col, mk, ls, lw, zo in models_config:
        y1 = [DATA_MAP[key].get(lvl, 0.0) for lvl in LEVELS]
        y2 = [DATA_MAP50[key].get(lvl, 0.0) for lvl in LEVELS]
        ax1.plot(x_indices, y1, label=label, color=col, marker=mk, linestyle=ls, linewidth=lw, markersize=6, zorder=zo)
        ax2.plot(x_indices, y2, label=label, color=col, marker=mk, linestyle=ls, linewidth=lw, markersize=6, zorder=zo)

    # Plot F5 Corrected
    if f5_raw_map:
        y1_raw = [f5_raw_map.get(lvl, np.nan) for lvl in LEVELS]
        y2_raw = [f5_raw_map50.get(lvl, np.nan) for lvl in LEVELS]
        y1_comp = [f5_comp_map.get(lvl, np.nan) for lvl in LEVELS]
        y2_comp = [f5_comp_map50.get(lvl, np.nan) for lvl in LEVELS]

        ax1.plot(x_indices, y1_raw, label="F5 Corrected (Tri-Modal RAW)", color="#9467BD", marker="D", linestyle="-.", linewidth=2.0, markersize=6, zorder=7)
        ax1.plot(x_indices, y1_comp, label="F5 Corrected (Tri-Modal COMPACT)", color="#8C564B", marker="P", linestyle=":", linewidth=1.8, markersize=6, zorder=6)

        ax2.plot(x_indices, y2_raw, label="F5 Corrected (Tri-Modal RAW)", color="#9467BD", marker="D", linestyle="-.", linewidth=2.0, markersize=6, zorder=7)
        ax2.plot(x_indices, y2_comp, label="F5 Corrected (Tri-Modal COMPACT)", color="#8C564B", marker="P", linestyle=":", linewidth=1.8, markersize=6, zorder=6)

    # Plot Sovereign V24 (Crown Jewel!)
    if v24_map:
        y1_v24 = [v24_map.get(lvl, np.nan) for lvl in LEVELS]
        y2_v24 = [v24_map50.get(lvl, np.nan) for lvl in LEVELS]

        ax1.plot(x_indices, y1_v24, label="★ Sovereign V24 TADL (Night Robust)", color="#D62728", marker="o", linestyle="-", linewidth=3.2, markersize=8, zorder=10)
        ax2.plot(x_indices, y2_v24, label="★ Sovereign V24 TADL (Night Robust)", color="#D62728", marker="o", linestyle="-", linewidth=3.2, markersize=8, zorder=10)

        # Highlight key breakthroughs with annotations on ax1
        if not np.isnan(y1_v24[3]): # night12
            ax1.annotate(f"V24: {y1_v24[3]:.4f}\n(+10.5% vs V23)", xy=(3, y1_v24[3]), xytext=(3.1, y1_v24[3] + 0.08),
                         arrowprops=dict(arrowstyle="->", color="#D62728", lw=1.2),
                         fontsize=9, fontweight="bold", color="#D62728",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEEEE", edgecolor="#D62728", alpha=0.9))

        if not np.isnan(y1_v24[0]): # orig
            ax1.annotate(f"V24: {y1_v24[0]:.4f}", xy=(0, y1_v24[0]), xytext=(0.1, y1_v24[0] - 0.06),
                         arrowprops=dict(arrowstyle="->", color="#D62728", lw=1.2),
                         fontsize=9, fontweight="bold", color="#D62728",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEEEE", edgecolor="#D62728", alpha=0.9))

    # Format ticks & legends
    for ax in [ax1, ax2]:
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{lvl}\n({DARK[lvl]})" if lvl != "black" else "black\n(0.0)" for lvl in LEVELS], fontsize=9.5)
        ax.legend(frameon=True, facecolor="#FDFDFD", edgecolor="#CCCCCC", fontsize=9.5, loc="upper right")

    plt.tight_layout()
    out_png = FIG_DIR / "nightxx_master_v24_vs_f5_comparison.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison plot to: {out_png}")

    # Copy to artifacts directory
    artifact_png = ARTIFACT_DIR / "nightxx_master_v24_vs_f5_comparison.png"
    shutil.copy2(out_png, artifact_png)
    print(f"[+] Copied plot to artifact: {artifact_png}")


if __name__ == "__main__":
    plot_comparison()
