"""Same figure/style as night_full_4models.png, but adds the current best model
(midboost_combo = deepir 6ch + mid-boost fine-tune).

Curves: mAP vs RGB brightness multiplier (log, day=1.0 -> near-dark=0.0005) on the
AIC validation set.  solid = synthetic Poisson-Gaussian night, dashed = DNDT
physical night, * = deepir nightify (stylized).  IR/depth are untouched, so any
gap between the RGB-only models and the multimodal ones is attributable to the
auxiliary modalities.
"""
import os, re
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/night_full"
BASE = "/home/constant/CodeBuddy/Dev/AIC/logs"
LEVELS = ["day", "night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]
DNDT_LEVELS = ["night50", "night25", "night12", "night05", "night01",
               "night005", "night002", "night001", "night0005"]
DARK = {"day": 1.0, "night50": 0.50, "night25": 0.25, "night12": 0.12, "nightify": 0.11,
        "night05": 0.05, "night01": 0.01, "night005": 0.005, "night002": 0.002,
        "night001": 0.001, "night0005": 0.0005}
PAT = re.compile(r"IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)")
PAT50 = re.compile(r"IoU=0\.50\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)")

# order matters for legend; midboost added last as the highlighted model
MODELS = ["rgb3ch", "nightsim", "fused", "ufall", "deepir", "midboost"]
LABELS = {"rgb3ch": "rgb3ch (3ch RGB)", "nightsim": "nightsim (3ch RGB+inorm)",
          "fused": "fused (3ch RGB)", "ufall": "ufall (5ch RGB+IR+Depth)",
          "deepir": "deepir (6ch RGB+IR+Depth)",
          "midboost": "midboost_combo (6ch, +mid-boost FT)  [ours]"}


def read(fname):
    if not os.path.exists(fname):
        return None, None
    s = open(fname, errors="ignore").read()
    m = PAT.search(s); m50 = PAT50.search(s)
    return (float(m.group(1)) if m else None, float(m50.group(1)) if m50 else None)


data, data50 = {m: {} for m in MODELS}, {m: {} for m in MODELS}
dndt, dndt50 = {m: {} for m in MODELS}, {m: {} for m in MODELS}

for m in MODELS:
    # ---- synthetic night (solid) ----
    for lv in LEVELS:
        if m == "deepir":
            src = "orig" if lv == "day" else lv
            v, v50 = read(os.path.join(BASE, f"deepir_eval_{src}.log"))
        elif m == "midboost":
            src = "day" if lv == "day" else lv
            v, v50 = read(os.path.join(BASE, f"midboost_combo_{src}.log"))
        else:
            v, v50 = read(os.path.join(OUT, f"{m}_{lv}.log"))
        if v is not None:
            data[m][lv] = v
        if v50 is not None:
            data50[m][lv] = v50
    # ---- DNDT physical night (dashed) ----
    for lv in DNDT_LEVELS:
        if m == "deepir":
            v, v50 = read(os.path.join(BASE, f"deepir_dndt_eval_{lv}.log"))
        elif m == "midboost":
            v, v50 = read(os.path.join(BASE, f"midboost_combo_dndt_{lv}.log"))  # may not exist
        else:
            v, v50 = read(os.path.join(OUT, f"{m}_dndt_{lv}.log"))
        if v is not None:
            dndt[m][lv] = v
        if v50 is not None:
            dndt50[m][lv] = v50

# deepir nightify (stylized) standalone star
dn, dn50 = {}, {}
v, v50 = read(os.path.join(BASE, "deepir_eval_nightify.log"))
if v is not None:
    dn["deepir"] = v; dn50["deepir"] = v50
# midboost_combo nightify (stylized) standalone star
dn_mb, dn50_mb = {}, {}
v, v50 = read(os.path.join(BASE, "midboost_combo_nightify.log"))
if v is not None:
    dn_mb["midboost"] = v; dn50_mb["midboost"] = v50


def draw(d, dd, dn_, dn50_, dn_mb_, dn50_mb_, title, ylabel, path):
    plt.figure(figsize=(9.5, 5.5))
    deepir_color = None
    for m in MODELS:
        xs = [DARK[l] for l in LEVELS]
        ys = [d[m].get(lv) for lv in LEVELS]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        lw = 2.4 if m == "midboost" else 1.6
        z = 6 if m == "midboost" else 3
        line, = plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                         markersize=(6 if m == "midboost" else 4.5), linewidth=lw,
                         zorder=z, label=LABELS[m])
        if m == "deepir":
            deepir_color = line.get_color()
        if m == "midboost":
            midboost_color = line.get_color()
        if m == "midboost":
            for x, y in pts:
                plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                             xytext=(0, 6), ha="center", fontsize=6.5, color=line.get_color())
        xs2 = [DARK[l] for l in DNDT_LEVELS]
        ys2 = [dd[m].get(lv) for lv in DNDT_LEVELS]
        p2 = [(x, y) for x, y in zip(xs2, ys2) if y is not None]
        if p2:
            plt.plot([p[0] for p in p2], [p[1] for p in p2], linestyle="--", marker="s",
                     markersize=5, linewidth=1.4, color=line.get_color(), zorder=4)
    plt.plot([], [], linestyle="--", color="gray", label="DNDT physical night (dashed)")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("RGB brightness multiplier (log); left=day(1.0 bright) -> right=near-dark(0.0005)\n"
               "solid = synthetic Poisson-Gaussian night; dashed = DNDT physical night")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print("saved", path)


draw(data, dndt, dn, dn50, dn_mb, dn50_mb,
     "AIC night robustness (6 models) - synthetic vs DNDT physical night",
     "mAP @ IoU=0.50:0.95", os.path.join(FIG_DIR, "night_full_6models_midboost.png"))
draw(data50, dndt50, dn, dn50, dn_mb, dn50_mb,
     "AIC night robustness (6 models) - synthetic vs DNDT physical night (AP50)",
     "mAP @ IoU=0.50", os.path.join(FIG_DIR, "night_full_6models_midboost_ap50.png"))

print("\n=== midboost_combo (ours) vs deepir ===")
for lv in LEVELS:
    mb = data["midboost"].get(lv); dp = data["deepir"].get(lv)
    f = lambda v: f"{v:.3f}" if v is not None else "  -  "
    d = f"{mb-dp:+.3f}" if (mb is not None and dp is not None) else "   -"
    print(f"{lv:9s} deepir={f(dp)}  midboost={f(mb)}  delta={d}")
