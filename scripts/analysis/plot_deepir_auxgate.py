import os, re
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/constant/CodeBuddy/Dev/AIC/logs"
PAT = re.compile(r"IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)")
PAT50 = re.compile(r"IoU=0\.50\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)")

DARK = {"day": 1.0, "night50": 0.50, "night25": 0.25, "night12": 0.12,
        "night05": 0.05, "night01": 0.01, "night005": 0.005, "night002": 0.002,
        "night001": 0.001, "night0005": 0.0005}
# public levels where both deepir and ufall exist
COMMON = ["night50", "night25", "night12", "night05", "night01", "night005", "night002", "night001", "night0005"]
ALLV = ["day"] + COMMON


def read_map(fname):
    if not os.path.exists(fname):
        return None, None
    s = open(fname, errors="ignore").read()
    m = PAT.search(s); a = PAT50.search(s)
    return (float(m.group(1)) if m else None, float(a.group(1)) if a else None)


# baselines
deepir_base = {lv: read_map(os.path.join(BASE, f"deepir_eval_{lv if lv!='day' else 'orig'}.log")) for lv in ALLV}
ufall = {lv: read_map(os.path.join(BASE, f"ufall_eval_{lv}.log")) for lv in COMMON}

# gates
gates = {}
for tag in ["g1", "g2", "g3", "g4"]:
    gates[tag] = {lv: read_map(os.path.join(BASE, f"deepir_gate_{tag}_{lv if lv!='day' else 'orig'}.log")) for lv in ALLV}

GTAG = {"g1": "g1 gain10 bias.12 cap1.4", "g2": "g2 gain10 bias.25 cap1.4",
        "g3": "g3 gain16 bias.12 cap1.6", "g4": "g4 gain6 bias.18 cap1.3"}


def draw(metric, title, ylab, path):
    plt.figure(figsize=(10.5, 6))
    xs = [DARK[l] for l in ALLV]
    # ufall (9 common levels)
    ux = [DARK[l] for l in COMMON]
    uy = [ufall[l][metric] for l in COMMON]
    up = [(x, y) for x, y in zip(ux, uy) if y is not None]
    plt.plot([p[0] for p in up], [p[1] for p in up], "k--o", linewidth=2, markersize=6,
             label="ufall 5ch (RGB+IR+rawDepth) baseline")
    # deepir base
    by = [deepir_base[l][metric] for l in ALLV]
    bp = [(x, y) for x, y in zip(xs, by) if y is not None]
    plt.plot([p[0] for p in bp], [p[1] for p in bp], "gray", marker="s", linewidth=1.6,
             label="deepir 6ch base (no aux gate)")
    # gates
    cmap = {"g1": "C3", "g2": "C1", "g3": "C2", "g4": "C4"}
    for tag in ["g1", "g2", "g3", "g4"]:
        gy = [gates[tag][l][metric] for l in ALLV]
        gp = [(x, y) for x, y in zip(xs, gy) if y is not None]
        if gp:
            plt.plot([p[0] for p in gp], [p[1] for p in gp], color=cmap[tag], marker="^",
                     linewidth=1.8, markersize=4, label=f"deepir+{GTAG[tag]}")
    plt.xscale("log"); plt.gca().invert_xaxis()
    plt.xlabel("RGB brightness multiplier (log); left=bright(day) -> right=near-dark")
    plt.ylabel(ylab); plt.title(title)
    plt.legend(fontsize=8, loc="upper right"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print("saved", path)


draw(0, "Brightness-adaptive aux gate: deepir vs ufall (mAP@IoU=0.50:0.95)",
     "mAP @ IoU=0.50:0.95", os.path.join(FIG_DIR, "deepir_auxgate.png"))
draw(1, "Brightness-adaptive aux gate: deepir vs ufall (AP50)",
     "mAP @ IoU=0.50", os.path.join(FIG_DIR, "deepir_auxgate_ap50.png"))

# summary table on common levels
print("\nlevel     ufall   deepir_base  g1     g2     g3     g4")
for lv in COMMON:
    def f(v): return f"{v[0]:.3f}" if v and v[0] is not None else "  -  "
    print(f"{lv:9s} {f(ufall[lv])}  {f(deepir_base[lv])}  {f(gates['g1'][lv])}  "
          f"{f(gates['g2'][lv])}  {f(gates['g3'][lv])}  {f(gates['g4'][lv])}")

# best gate at each level vs ufall
print("\n--- per-level: best deepir gate vs ufall ---")
for lv in COMMON:
    vals = {t: gates[t][lv][0] for t in ["g1", "g2", "g3", "g4"] if gates[t][lv][0] is not None}
    ub = ufall[lv][0]
    if ub is not None and vals:
        best = max(vals, key=vals.get)
        print(f"{lv:9s} ufall={ub:.3f}  best_gate={best}({vals[best]:.3f})  "
              f"{'DEEPIR_WINS' if vals[best] > ub else 'still behind'}")
