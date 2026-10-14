import os, re
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/constant/CodeBuddy/Dev/AIC/logs"
PAT = re.compile(r"IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)")
DARK = {"day": 1.0, "night50": 0.50, "night25": 0.25, "night12": 0.12,
        "night05": 0.05, "night01": 0.01, "night005": 0.005, "night002": 0.002,
        "night001": 0.001, "night0005": 0.0005}
LEVELS = ["day", "night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]
UFALL = {"day": 0.483, "night50": 0.486, "night25": 0.475, "night12": 0.462,
         "night05": 0.426, "night01": 0.211, "night005": 0.117}


def read_map(fname):
    if not os.path.exists(fname):
        return None
    m = PAT.search(open(fname, errors="ignore").read())
    return float(m.group(1)) if m else None


def load(prefix):
    return {lv: read_map(os.path.join(BASE, f"{prefix}_{lv if lv != 'day' else 'orig'}.log")) for lv in LEVELS}


base = load("deepir_eval")
v1 = load("midboost_eval")
l2 = load("midboost_l2_eval")
l3 = load("midboost_l3_eval")
l4 = load("midboost_l4_eval")
xs = [DARK[l] for l in LEVELS]


def series(d, color, label, style="-", marker="o"):
    pts = [(DARK[l], d[l]) for l in LEVELS if d[l] is not None]
    if pts:
        plt.plot([p[0] for p in pts], [p[1] for p in pts], style, color=color,
                 marker=marker, linewidth=2, label=label)


plt.figure(figsize=(11, 6.2))
series(base, "gray", "deepir base (6ch)", "--", "s")
series(v1, "C3", "mid-boost v1 (uniform mid)", "-", "^")
series(l2, "C4", "L2 level-weighted + bright refill", "-", "D")
series(l3, "C1", "L3 RGB higher LR + capacity", "-", "P")
series(l4, "C2", "L4 stronger photometric aug", "-", "X")
up = [(DARK[l], UFALL[l]) for l in UFALL]
plt.plot([p[0] for p in up], [p[1] for p in up], "k", marker="o", linewidth=2,
         label="ufall 5ch baseline")
plt.xscale("log"); plt.gca().invert_xaxis()
plt.xlabel("RGB brightness multiplier (log); left=bright -> right=near-dark")
plt.ylabel("mAP @ IoU=0.50:0.95")
plt.title("Mid-boost ablations: L2 / L3 / L4 vs v1 vs ufall")
plt.legend(fontsize=8.5); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "midboost_all.png"), dpi=130); plt.close()
print("saved", os.path.join(FIG_DIR, "midboost_all.png"))

print("\nlevel     ufall   base    v1     L2     L3     L4")
for lv in LEVELS:
    def f(v): return f"{v:.3f}" if v is not None else "  -  "
    u = UFALL.get(lv)
    print(f"{lv:9s} {f(u)}  {f(base[lv])}  {f(v1[lv])}  {f(l2[lv])}  {f(l3[lv])}  {f(l4[lv])}")

print("\n--- residual to ufall (only where ufall known); best of {v1,L2,L3,L4} in parens ---")
for lv in ["night50", "night25", "night12", "night05", "night01", "night005"]:
    u = UFALL.get(lv)
    row = []
    best = None
    for name, d in [("base", base), ("v1", v1), ("L2", l2), ("L3", l3), ("L4", l4)]:
        val = d[lv]
        if u and val is not None:
            r = val - u
            row.append(f"{name}={r:+.3f}")
            if best is None or r > best[1]:
                best = (name, r)
    print(f"{lv:9s} ufall={u:.3f} " + " ".join(row) + (f"  -> best={best[0]}{best[1]:+.3f}" if best else ""))
