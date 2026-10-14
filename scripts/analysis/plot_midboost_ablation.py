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

# x positions (brightness multiplier); day on the left, near-dark on the right
DARK = {"day": 1.0, "night50": 0.50, "night25": 0.25, "night12": 0.12,
        "night05": 0.05, "night01": 0.01, "night005": 0.005, "night002": 0.002,
        "night001": 0.001, "night0005": 0.0005}
LEVELS = ["day", "night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]
ABL = ["rawir", "heat", "depth"]
ACOL = {"rawir": "C0", "heat": "C1", "depth": "C2"}
ALAB = {"rawir": "midboost -rawIR (5ch)", "heat": "midboost -heatIR (5ch)",
        "depth": "midboost -depth (5ch)"}


def read_map(fname):
    if not os.path.exists(fname):
        return None, None
    s = open(fname, errors="ignore").read()
    m = PAT.search(s); a = PAT50.search(s)
    return (float(m.group(1)) if m else None, float(a.group(1)) if a else None)


# full midboost_combo reference: day->midboost_combo_day, nightXX->midboost_combo_nightXX
full = {}
for lv in LEVELS:
    src = "day" if lv == "day" else lv
    full[lv] = read_map(os.path.join(BASE, f"midboost_combo_{src}.log"))

# ablations
abl = {ab: {} for ab in ABL}
for ab in ABL:
    for lv in LEVELS:
        abl[ab][lv] = read_map(os.path.join(BASE, f"midboost_combo_ablate_{ab}_{lv}.log"))


def draw(metric, title, ylabel, path):
    plt.figure(figsize=(9.5, 5.5))
    # full
    xs = [DARK[l] for l in LEVELS]
    ys = [full[l][metric] for l in LEVELS]
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    plt.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="black",
             marker="o", linewidth=2.2, label="midboost_combo 6ch (full: RGB+heat+rawIR+depth)")
    for ab in ABL:
        ys2 = [abl[ab][l][metric] for l in LEVELS]
        p2 = [(x, y) for x, y in zip(xs, ys2) if y is not None]
        if p2:
            plt.plot([p[0] for p in p2], [p[1] for p in p2], "--",
                     color=ACOL[ab], marker="s", markersize=5, linewidth=1.6,
                     label=ALAB[ab])
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("RGB brightness multiplier (log); left=day(1.0 bright) -> right=near-dark(0.0005)\n"
               "solid=all channels; dashed=one channel filled with its mean (occlusion)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print("saved", path)


draw(0, "midboost_combo (fully integrated) channel ablation - mAP@IoU=0.50:0.95",
     "mAP @ IoU=0.50:0.95", os.path.join(FIG_DIR, "midboost_ablation.png"))
draw(1, "midboost_combo (fully integrated) channel ablation - AP50",
     "mAP @ IoU=0.50", os.path.join(FIG_DIR, "midboost_ablation_ap50.png"))

# console summary: per-level drop vs full
print("\nlevel     full   -rawIR  -heat   -depth")
for lv in LEVELS:
    f = full[lv][0]
    r = abl["rawir"][lv][0]; h = abl["heat"][lv][0]; d = abl["depth"][lv][0]
    def fmt(v): return f"{v:.3f}" if v is not None else "  -  "
    print(f"{lv:9s} {fmt(f)}  {fmt(r)}  {fmt(h)}  {fmt(d)}")
