#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""各 nightify 档位上的 mAP 增幅对比: 每个杠杆相对 base 的 Δ (AP@0.5 与 COCO mAP@0.5:0.95).
基准 = base (midboost). 自动包含已存在变体 (L2/L3/L4/combo...).
"""
import os, re
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/constant/CodeBuddy/Dev/AIC/logs"
OUT = os.path.join(FIG_DIR, "nightify_gain.png")

BRACKETS = [
    ("day", 1.0), ("night50", 0.5), ("night25", 0.25), ("night12", 0.12),
    ("night05", 0.05), ("night01", 0.01), ("night005", 0.005),
    ("night002", 0.002), ("night001", 0.001), ("night0005", 0.0005),
]
ORDER = [b[0] for b in BRACKETS]
LEVEL = {b[0]: b[1] for b in BRACKETS}

BASE_PREF = "midboost_eval"
VARIANTS = [
    ("L2", "midboost_l2_eval"),
    ("L3", "midboost_l3_eval"),
    ("L4", "midboost_l4_eval"),
    ("combo", "midboost_combo_eval"),
]

RE_AP50 = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s+\|\s+area=\s+all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)")
RE_AP5095 = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s+\|\s+area=\s+all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)")

def parse(path):
    with open(path) as f:
        txt = f.read()
    def g(rx):
        m = rx.search(txt); return float(m.group(1)) if m else None
    return g(RE_AP50), g(RE_AP5095)

def series(pref):
    d = {}
    for br in ORDER:
        p = os.path.join(W, f"{pref}_{br}.log")
        if not os.path.exists(p):
            # combo 系列的 log 文件名不带 _eval (midboost_combo_night50.log)
            alt = pref[:-5] if pref.endswith("_eval") else pref
            p2 = os.path.join(W, f"{alt}_{br}.log")
            if os.path.exists(p2):
                p = p2
        if os.path.exists(p):
            d[br] = parse(p)
    return d

def main():
    base = series(BASE_PREF)
    if not base:
        print("base 评测日志缺失")
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    titles = ["\u0394 mAP @ IoU=0.50 (vs base)", "\u0394 COCO mAP @ IoU=0.50:0.95 (vs base)"]
    styles = {"L2": ("#1f77b4", "-", "s"), "L3": ("#2ca02c", "-", "^"),
              "L4": ("#ff7f0e", "-", "D"), "combo": ("#d62728", "-", "*")}
    xs = [LEVEL[b] for b in ORDER]
    for ci, key in enumerate([0, 1]):
        ax = axes[ci]
        for label, pref in VARIANTS:
            sv = series(pref)
            if not sv:
                continue
            ys = []
            ok = True
            for b in ORDER:
                if b in base and b in sv and base[b][key] is not None and sv[b][key] is not None:
                    ys.append(sv[b][key] - base[b][key])
                else:
                    ys.append(None); ok = False
            if not ok and all(v is None for v in ys):
                continue
            color, ls, mk = styles.get(label, ("#000000", "-", "o"))
            ax.plot(xs, ys, marker=mk, ls=ls, color=color, lw=2, ms=7, label=label)
        ax.axhline(0.0, color="#888", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(ORDER, rotation=30, ha="right")
        ax.set_xlabel("nightify exposure level (log scale)")
        ax.set_ylabel(titles[ci])
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(titles[ci])
        ax.legend(loc="upper left")
    fig.suptitle("Gain over base across nightify levels")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=130)
    print("已保存:", OUT)

if __name__ == "__main__":
    main()
