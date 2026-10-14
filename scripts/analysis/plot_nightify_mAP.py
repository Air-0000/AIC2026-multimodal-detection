#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""各 nightify 档位上的 mAP 对比图 (双面板: AP@0.5 + COCO mAP@0.5:0.95).
严格区分 Average Precision / Recall: 只取 "Average Precision (AP)" 且 "area= all" 的行.
自动包含已存在变体 (base/L2/L3/L4/combo...).
"""
import os, re, glob
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/constant/CodeBuddy/Dev/AIC/logs"
OUT = os.path.join(FIG_DIR, "nightify_mAP.png")

BRACKETS = [
    ("day", 1.0), ("night50", 0.5), ("night25", 0.25), ("night12", 0.12),
    ("night05", 0.05), ("night01", 0.01), ("night005", 0.005),
    ("night002", 0.002), ("night001", 0.001), ("night0005", 0.0005),
]
ORDER = [b[0] for b in BRACKETS]
LEVEL = {b[0]: b[1] for b in BRACKETS}

VARIANTS = [
    ("base",  "midboost_eval"),
    ("L2",    "midboost_l2_eval"),
    ("L3",    "midboost_l3_eval"),
    ("L4",    "midboost_l4_eval"),
    ("combo", "midboost_combo_eval"),
]

# 只取 "Average Precision (AP)" 且 area=all 的那一行; 锚定行尾 "] = <数值>"
RE_AP50  = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s+\|\s+area=\s+all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)")
RE_AP5095 = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s+\|\s+area=\s+all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)")

def parse(path):
    ap50 = ap5095 = None
    with open(path) as f:
        txt = f.read()
    m = RE_AP50.search(txt);  ap50 = float(m.group(1)) if m else None
    m = RE_AP5095.search(txt); ap5095 = float(m.group(1)) if m else None
    return ap50, ap5095

def main():
    series = {}
    for label, pref in VARIANTS:
        d = {}
        for br in ORDER:
            p = os.path.join(W, f"{pref}_{br}.log")
            if os.path.exists(p):
                a50, a5095 = parse(p)
                if a50 is not None:
                    d[br] = (a50, a5095)
        if d:
            series[label] = d
    if not series:
        print("未找到任何变体评测日志")
        return

    xs = [LEVEL[b] for b in ORDER]
    styles = {
        "base":  ("#888888", "--", "o"),
        "L2":    ("#1f77b4", "-",  "s"),
        "L3":    ("#2ca02c", "-",  "^"),
        "L4":    ("#ff7f0e", "-",  "D"),
        "combo": ("#d62728", "-",  "*"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    titles = ["mAP @ IoU=0.50", "COCO mAP @ IoU=0.50:0.95"]
    for ci, key in enumerate([0, 1]):
        ax = axes[ci]
        for label, d in series.items():
            ys = [d[b][key] for b in ORDER]
            color, ls, mk = styles.get(label, ("#000000", "-", "o"))
            ax.plot(xs, ys, marker=mk, ls=ls, color=color, lw=2, ms=7,
                    label=label, zorder=3 if label == "combo" else 2)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(ORDER, rotation=30, ha="right")
        ax.set_xlabel("nightify exposure level (log scale)")
        ax.set_ylabel(titles[ci])
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(titles[ci])
        ax.legend(loc="lower left")
    fig.suptitle("mAP across nightify levels (base vs L2 vs L3 vs L4"
                 + (" vs combo" if "combo" in series else "") + ")")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=130)
    print("已保存:", OUT)
    print("包含变体:", list(series.keys()))

if __name__ == "__main__":
    main()
