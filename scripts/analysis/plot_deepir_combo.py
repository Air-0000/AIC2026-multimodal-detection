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
LEVELS = ["day", "night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]


def read_map(fname):
    if not os.path.exists(fname):
        return None, None
    s = open(fname, errors="ignore").read()
    m = PAT.search(s); a = PAT50.search(s)
    return (float(m.group(1)) if m else None, float(a.group(1)) if a else None)


full = {}
for lv in LEVELS:
    src = "orig" if lv == "day" else lv
    full[lv] = read_map(os.path.join(BASE, f"deepir_eval_{src}.log"))

keep = {}
for k in ["heat", "rawir", "depth", "rgb"]:
    keep[k] = {}
    for lv in LEVELS:
        keep[k][lv] = read_map(os.path.join(BASE, f"deepir_keep_{k}_{lv}.log"))

# color / legend
KCOL = {"heat": "C3", "rawir": "C0", "depth": "C2", "rgb": "gray"}
KLAB = {"heat": "keep heat only (RGB+heat)", "rawir": "keep rawIR only (RGB+rawIR)",
        "depth": "keep depth only (RGB+depth)", "rgb": "RGB only (no aux)"}


def draw(metric, title, ylabel, path):
    plt.figure(figsize=(10, 5.8))
    xs = [DARK[l] for l in LEVELS]
    # full
    ys = [full[l][metric] for l in LEVELS]
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    plt.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="black",
             marker="o", linewidth=2.4, label="deepir 6ch full (RGB+heat+rawIR+depth)")
    for k in ["heat", "rawir", "depth", "rgb"]:
        ys2 = [keep[k][l][metric] for l in LEVELS]
        p2 = [(x, y) for x, y in zip(xs, ys2) if y is not None]
        if p2:
            ls = "-" if k == "heat" else "--"
            plt.plot([p[0] for p in p2], [p[1] for p in p2], ls, color=KCOL[k],
                     marker="^" if k == "heat" else "s", markersize=5, linewidth=1.7,
                     label=KLAB[k])
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("RGB brightness multiplier (log); left=day(1.0 bright) -> right=near-dark(0.0005)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8.5, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print("saved", path)


draw(0, "deepir modal decomposition - mAP@IoU=0.50:0.95 (single-modality vs full)",
     "mAP @ IoU=0.50:0.95", os.path.join(FIG_DIR, "deepir_combo.png"))
draw(1, "deepir modal decomposition - AP50 (single-modality vs full)",
     "mAP @ IoU=0.50", os.path.join(FIG_DIR, "deepir_combo_ap50.png"))

print("\nlevel     full   heat   rawir  depth  rgb")
for lv in LEVELS:
    f = full[lv][0]
    h = keep["heat"][lv][0]; r = keep["rawir"][lv][0]
    d = keep["depth"][lv][0]; g = keep["rgb"][lv][0]
    def fmt(v): return f"{v:.3f}" if v is not None else "  -  "
    print(f"{lv:9s} {fmt(f)}  {fmt(h)}  {fmt(r)}  {fmt(d)}  {fmt(g)}")

# synergy note at darkest: full - sum(independent?) -> just report full vs best single
print("\n--- night0005 分解 ---")
lv = "night0005"
print(f"full={full[lv][0]:.3f}  heat={keep['heat'][lv][0]:.3f}  "
      f"rawir={keep['rawir'][lv][0]:.3f}  depth={keep['depth'][lv][0]:.3f}  "
      f"rgb={keep['rgb'][lv][0]:.3f}")
