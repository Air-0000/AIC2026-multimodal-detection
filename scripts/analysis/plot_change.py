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
# only the night levels show the change we care about
LEVELS = ["night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]
UFALL = {"night50": 0.486, "night25": 0.475, "night12": 0.462,
         "night05": 0.426, "night01": 0.211, "night005": 0.117}


def read_map(fname):
    if not os.path.exists(fname):
        return None
    m = PAT.search(open(fname, errors="ignore").read())
    return float(m.group(1)) if m else None


def load(prefix):
    return {lv: read_map(os.path.join(BASE, f"{prefix}_{lv if lv != 'day' else 'orig'}.log")) for lv in LEVELS}


base = load("deepir_eval")
mid = load("midboost_eval")
labels = [l.replace("night", "") for l in LEVELS]
x = np.arange(len(LEVELS))

b = np.array([base[l] for l in LEVELS], float)
m = np.array([mid[l] for l in LEVELS], float)
u = np.array([UFALL.get(l) for l in LEVELS], float)  # nan where unknown
delta = m - b
resid = u - m  # positive => still below ufall

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- left: 3 curves ---
xs = [DARK[l] for l in LEVELS]
ax1.plot(xs, b, "--s", color="gray", lw=2, ms=7, label="deepir base (6ch)")
ax1.plot(xs, m, "-^", color="C3", lw=2, ms=7, label="mid-boost FT")
up = [(DARK[l], UFALL[l]) for l in UFALL]
ax1.plot([p[0] for p in up], [p[1] for p in up], "k-o", lw=2, ms=6, label="ufall 5ch (ref)")
ax1.set_xscale("log"); ax1.invert_xaxis()
ax1.set_xlabel("RGB brightness (log); left=bright → right=dark")
ax1.set_ylabel("mAP @ IoU=0.50:0.95")
ax1.set_title("Curves: base vs mid-boost vs ufall")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# --- right: change bars ---
colors = ["C2" if d > 0 else "C1" for d in delta]
ax2.bar(x - 0.2, delta, width=0.4, color=colors, label="Δ mid-boost − base")
# residual to ufall where known
xm = x[~np.isnan(resid)]
ax2.bar(xm + 0.2, resid[~np.isnan(resid)], width=0.4, color="C0",
        alpha=0.7, label="ufall − mid-boost (still-to-close)")
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax2.set_xlabel("level (brightness multiplier)")
ax2.set_ylabel("mAP delta")
ax2.set_title("Change from base (green=gain) & residual to ufall")
ax2.legend(fontsize=9); ax2.grid(True, axis="y", alpha=0.3)
for xi, d in zip(x, delta):
    ax2.annotate(f"{d:+.3f}", (xi - 0.2, d), textcoords="offset points",
                 xytext=(0, 4 if d >= 0 else -10), ha="center", fontsize=7)

plt.tight_layout()
out = os.path.join(FIG_DIR, "midboost_change.png")
plt.savefig(out, dpi=130); plt.close()
print("saved", out)

# also refresh the 3-curve figure
import importlib.util
print("\n".join([f"{l:9s} base={b[i]:.3f} mid={m[i]:.3f} Δ={delta[i]:+.3f}" for i, l in enumerate(LEVELS)]))
