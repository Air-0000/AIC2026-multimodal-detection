#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可加性检验 + L4U3 对照。

灰度柱 = 三个单杠杆(L2/L3/L4)相对 base(v1)的增益之和 (理想可加上限)
红柱   = COMBO(L2+L3+L4) 实际增益
蓝柱   = L4U3(L4+L3, 不含 L2) 实际增益
绿柱   = 最强单杠杆 (不叠加的对照)
同时打印各变体相对 ufall 5ch 基线的残差。
"""
import os, re
import matplotlib
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = "/home/constant/CodeBuddy/Dev/AIC/logs"
OUT = os.path.join(FIG_DIR, "midboost_additivity.png")

LEVELS = ["day", "night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]

PREFS = {
    "base": "midboost_eval",
    "L2": "midboost_l2_eval",
    "L3": "midboost_l3_eval",
    "L4": "midboost_l4_eval",
    "L4U3": "midboost_l4u3_eval",
    "COMBO": "midboost_combo",
    "ufall": "ufall",
}
# ufall 5ch 基线 (与其 plot 脚本一致; day 来自 ufall_orig_eval.log)
UFALL = {"day": 0.483, "night50": 0.486, "night25": 0.475, "night12": 0.462,
         "night05": 0.426, "night01": 0.211, "night005": 0.117,
         "night002": 0.169, "night001": 0.125, "night0005": 0.093}

RE_AP = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s+\|\s+"
                   r"area=\s+all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)")


def read(pref, lv):
    paths = [os.path.join(W, f"{pref}_{lv}.log")]
    if pref == "ufall":
        paths = [os.path.join(W, f"ufall_{'orig' if lv == 'day' else lv}_eval.log")]
    elif pref == "midboost_combo":
        paths.append(os.path.join(W, f"midboost_combo_{lv}.log"))
    for p in paths:
        if os.path.exists(p):
            m = RE_AP.search(open(p, errors="ignore").read())
            if m:
                return float(m.group(1))
    return None


def main():
    ap = {k: {lv: read(p, lv) for lv in LEVELS} for k, p in PREFS.items()}
    keys = ("L2", "L3", "L4", "L4U3", "COMBO")
    d = {k: {lv: (ap[k][lv] - ap["base"][lv]) for lv in LEVELS
             if ap[k][lv] is not None and ap["base"][lv] is not None} for k in keys}

    print(f"{'level':10s} {'ufall':>7s} | {'L2':>6s} {'L3':>6s} {'L4':>6s} | "
          f"{'SUM':>6s} {'COMBO':>6s} {'L4U3':>6s} | {'L4U3-L4':>8s} {'L4U3-ufall':>10s}")
    xs, sumv, combov, u3v, bestv = [], [], [], [], []
    for lv in LEVELS:
        if lv not in d.get("L4U3", {}) and lv not in d.get("COMBO", {}):
            continue
        parts = [d[k].get(lv) for k in ("L2", "L3", "L4")]
        parts = [p for p in parts if p is not None]
        s, b = (sum(parts), max(parts)) if parts else (None, None)
        c, u3 = d["COMBO"].get(lv), d["L4U3"].get(lv)
        u = UFALL.get(lv)
        xs.append(lv); sumv.append(s or 0); combov.append(c or 0)
        u3v.append(u3 or 0); bestv.append(b or 0)
        f = lambda v: f"{v:+6.3f}" if v is not None else "     -"
        # 注意: d[] 存的是相对 base 的 Δ; 与 ufall 比较必须用绝对 AP
        dl = f"{u3 - d['L4'][lv]:+8.3f}" if (u3 is not None and lv in d["L4"]) else "       -"
        a3 = ap["L4U3"][lv]
        du = f"{a3 - u:+10.3f}" if (a3 is not None and u is not None) else "         -"
        print(f"{lv:10s} {u if u else 0:7.3f} | " + " ".join(f(d[k].get(lv)) for k in ("L2", "L3", "L4")) +
              f" | {f(s)} {f(c)} {f(u3)} | {dl} {du}")

    n = len(xs); idx = np.arange(n); w = 0.2
    plt.figure(figsize=(13, 5.8))
    plt.bar(idx - 1.5 * w, sumv, w, color="#7f7f7f", label="SUM(L2+L3+L4) if additive")
    plt.bar(idx - 0.5 * w, combov, w, color="#d62728", label="COMBO (L2+L3+L4)")
    plt.bar(idx + 0.5 * w, u3v, w, color="#1f77b4", label="L4U3 (L4+L3, no L2)")
    plt.bar(idx + 1.5 * w, bestv, w, color="#2ca02c", label="best single lever")
    plt.axhline(0.0, color="#444", lw=1)
    plt.xticks(idx, xs, rotation=25, ha="right")
    plt.ylabel("Δ mAP @ IoU=0.50:0.95 (vs midboost v1)")
    plt.title("Additivity: SUM vs COMBO vs L4U3 vs best single")
    plt.legend(fontsize=9); plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT, dpi=130)
    print("saved:", OUT)
    r = [c / s for c, s in zip(combov, sumv) if s]
    print(f"\nmean COMBO/SUM = {np.mean(r):.0%}   mean(COMBO-best1) = "
          f"{np.mean([c - b for c, b in zip(combov, bestv)]):+.3f}   "
          f"mean(L4U3-best1) = {np.mean([u - b for u, b in zip(u3v, bestv)]):+.3f}")


if __name__ == "__main__":
    main()
