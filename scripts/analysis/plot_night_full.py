import os, re
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/night_full"
LEVELS = ["day", "night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]
DNDT_LEVELS = ["night50", "night25", "night12", "night05", "night01",
               "night005", "night002", "night001", "night0005"]
DARK = {"day": 1.0, "night50": 0.50, "night25": 0.25, "night12": 0.12, "nightify": 0.11,
        "night05": 0.05, "night01": 0.01, "night005": 0.005, "night002": 0.002,
        "night001": 0.001, "night0005": 0.0005}
PAT = re.compile(r"IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)")
PAT50 = re.compile(r"IoU=0\.50\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)")
MODELS = ["rgb3ch", "nightsim", "fused", "ufall", "deepir"]
LABELS = {"rgb3ch": "rgb3ch (3ch RGB)", "nightsim": "nightsim (3ch RGB+inorm)",
          "fused": "fused (3ch RGB)", "ufall": "ufall (5ch RGB+IR+Depth)",
          "deepir": "deepir (6ch RGB+IR+Depth)"}
BASE = "/home/constant/CodeBuddy/Dev/AIC/logs"

# synthetic (Poisson-Gaussian) curves
data, data50 = {m: {} for m in MODELS}, {m: {} for m in MODELS}
for m in MODELS:
    for lv in LEVELS:
        f = os.path.join(OUT, f"{m}_{lv}.log")
        if not os.path.exists(f):
            continue
        s = open(f, errors="ignore").read()
        mm = PAT.search(s)
        if mm:
            data[m][lv] = float(mm.group(1))
        mm = PAT50.search(s)
        if mm:
            data50[m][lv] = float(mm.group(1))

# DNDT physical-night curves (same x positions as synthetic dark levels)
dndt, dndt50 = {m: {} for m in MODELS}, {m: {} for m in MODELS}
dn, dn50 = {}, {}   # deepir nightify (stylized) — standalone star, NOT part of dashed polyline
for m in MODELS:
    for lv in DNDT_LEVELS:
        f = os.path.join(OUT, f"{m}_dndt_{lv}.log")
        if not os.path.exists(f):
            continue
        s = open(f, errors="ignore").read()
        mm = PAT.search(s)
        if mm:
            dndt[m][lv] = float(mm.group(1))
        mm = PAT50.search(s)
        if mm:
            dndt50[m][lv] = float(mm.group(1))

# deepir (6ch) — synthetic night levels + day, read from work_dir/deepir_eval_<lv>.log
# "day" in LEVELS maps to the deepir daylight log (orig); nightify -> dashed (DNDT physical)
DEEPIR_SRC = {"day": "orig"}   # level in LEVELS -> log suffix
for lv in LEVELS:
    src = DEEPIR_SRC.get(lv, lv)
    f = os.path.join(BASE, f"deepir_eval_{src}.log")
    if not os.path.exists(f):
        continue
    s = open(f, errors="ignore").read()
    mm = PAT.search(s)
    if mm:
        data["deepir"][lv] = float(mm.group(1))
    mm = PAT50.search(s)
    if mm:
        data50["deepir"][lv] = float(mm.group(1))
# deepir nightify (stylized night) — standalone star at x=0.11, NOT connected to dashed polyline
f = os.path.join(BASE, "deepir_eval_nightify.log")
if os.path.exists(f):
    s = open(f, errors="ignore").read()
    mm = PAT.search(s)
    if mm:
        dn["deepir"] = float(mm.group(1))
    mm = PAT50.search(s)
    if mm:
        dn50["deepir"] = float(mm.group(1))
# deepir DNDT physical-night (dashed, 9 levels) — read from work_dir/deepir_dndt_eval_<lv>.log
DEEPIR_DNDT_LV = ["night50", "night25", "night12", "night05", "night01",
                  "night005", "night002", "night001", "night0005"]
for lv in DEEPIR_DNDT_LV:
    f = os.path.join(BASE, f"deepir_dndt_eval_{lv}.log")
    if not os.path.exists(f):
        continue
    s = open(f, errors="ignore").read()
    mm = PAT.search(s)
    if mm:
        dndt["deepir"][lv] = float(mm.group(1))
    mm = PAT50.search(s)
    if mm:
        dndt50["deepir"][lv] = float(mm.group(1))


def draw(d, dd, title, ylabel, path):
    plt.figure(figsize=(9.5, 5.5))
    for m in MODELS:
        # synthetic (solid)
        xs = [DARK[l] for l in LEVELS]
        ys = [d[m].get(lv) for lv in LEVELS]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        line, = plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=LABELS[m])
        if m == "deepir":
            deepir_color = line.get_color()
        # DNDT physical (dashed, same color)
        xs2 = [DARK[l] for l in DNDT_LEVELS]
        ys2 = [dd[m].get(lv) for lv in DNDT_LEVELS]
        p2 = [(x, y) for x, y in zip(xs2, ys2) if y is not None]
        if p2:
            plt.plot([p[0] for p in p2], [p[1] for p in p2], linestyle="--", marker="s",
                     markersize=5, linewidth=1.4, color=line.get_color(), zorder=4)
    plt.plot([], [], linestyle="--", color="gray", label="DNDT physical night (dashed)")
    # deepir nightify (stylized) — standalone star at x=0.11, NOT connected to the polyline
    if "deepir" in dn and dn.get("deepir") is not None:
        plt.scatter([DARK["nightify"]], [dn["deepir"]], marker="*", s=180,
                    color=deepir_color, zorder=7, label="deepir nightify (stylized)")
        plt.scatter([DARK["nightify"]], [dn50.get("deepir")], marker="*", s=180,
                    color=deepir_color, zorder=7)
    plt.xscale("log")
    plt.gca().invert_xaxis()   # brighter (day=1.0) on the left, near-dark on the right
    plt.xlabel("RGB brightness multiplier (log); left=day(1.0 bright) -> right=near-dark(0.0005)\n"
               "solid = synthetic Poisson-Gaussian night; dashed = DNDT physical night; * = deepir nightify(stylized)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print("saved", path)


draw(data, dndt, "AIC night robustness (5 models) - synthetic vs DNDT physical night",
     "mAP @ IoU=0.50:0.95", os.path.join(FIG_DIR, "night_full_4models.png"))
draw(data50, dndt50, "AIC night robustness (5 models) - synthetic vs DNDT physical night (AP50)",
     "mAP @ IoU=0.50", os.path.join(FIG_DIR, "night_full_4models_ap50.png"))
for m in MODELS:
    print(m, "synth=", {lv: data[m].get(lv) for lv in LEVELS})
    print(m, "dndt =", {lv: dndt[m].get(lv) for lv in DNDT_LEVELS})
