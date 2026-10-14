import re
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

SRC = "/home/constant/CodeBuddy/Dev/AIC/docs/training_records_mAP.md"  # fallback
SRC = "/home/constant/CodeBuddy/Dev/AIC/docs/training_records_mAP.md"
OUT = os.path.join(FIG_DIR, "map_distribution.png")

MOD_EN = {
    "CSD": "CSD",
    "IR(Thermal)": "IR(heat)",
    "RGB(3ch)": "RGB(3ch)",
    "RGB+Thermal(多模态)": "RGB+Thermal",
    "diffuse IR(4ch)": "diffuse IR",
    "fused(多模态)": "fused",
    "night-sim(4ch)": "night-sim",
    "other": "other",
    "smoke-test": "smoke",
    "step-pipeline": "step",
}

rows = []
with open(SRC) as f:
    for line in f:
        if not line.strip().startswith("|"):
            continue
        if "---" in line or "mAP" in line.split("|")[1]:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        try:
            m = float(cells[4])
            a50 = float(cells[5])
        except ValueError:
            continue
        rows.append((MOD_EN.get(cells[0], cells[0]), cells[3], m, a50))

assert rows, "no rows parsed"
mods = [r[0] for r in rows]
maps = np.array([r[2] for r in rows])
ap50s = np.array([r[3] for r in rows])
names = [r[1] for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

# 1) mAP histogram
axes[0].hist(maps, bins=12, color="steelblue", edgecolor="white")
mean_m, med_m = maps.mean(), np.median(maps)
axes[0].axvline(mean_m, color="red", ls="--", lw=1.5, label=f"mean={mean_m:.3f}")
axes[0].axvline(med_m, color="green", ls="--", lw=1.5, label=f"median={med_m:.3f}")
axes[0].set_title(f"mAP distribution (n={len(rows)} trained models)")
axes[0].set_xlabel("mAP @ IoU=0.50:0.95")
axes[0].set_ylabel("count")
axes[0].legend(fontsize=8)

# 2) boxplot by modality
groups = defaultdict(list)
for mm in rows:
    groups[mm[0]].append(mm[2])
order = sorted(groups.keys(), key=lambda k: -np.mean(groups[k]))
data = [groups[k] for k in order]
bp = axes[1].boxplot(data, tick_labels=[f"{k}\n(n={len(groups[k])})" for k in order],
                     patch_artist=True, showmeans=True)
for patch in bp["boxes"]:
    patch.set_facecolor("lightsteelblue")
for m in bp["means"]:
    m.set_markerfacecolor("red")
    m.set_markeredgecolor("red")
axes[1].set_title("mAP by modality")
axes[1].set_ylabel("mAP")
plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right", fontsize=8)

# 3) scatter mAP vs AP50
axes[2].scatter(maps, ap50s, c="darkorange", s=45, edgecolor="k", zorder=3)
# annotate extremes
for mm in rows:
    if mm[2] < 0.22 or mm[2] > 0.50:
        axes[2].annotate(mm[1][:18], (mm[2], mm[3]), fontsize=6,
                         xytext=(3, 3), textcoords="offset points")
lim = [0, 0.86]
axes[2].plot(lim, lim, "k--", alpha=0.35, label="mAP = AP50")
axes[2].set_xlim(lim)
axes[2].set_ylim(lim)
axes[2].set_title("mAP vs AP50")
axes[2].set_xlabel("mAP")
axes[2].set_ylabel("AP50")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT, dpi=120)
print(f"saved {OUT}  n={len(rows)}  mean_mAP={mean_m:.3f} median={med_m:.3f} "
      f"mean_AP50={ap50s.mean():.3f}")
print("best:", max(rows, key=lambda r: r[2])[1], max(maps))
print("worst:", min(rows, key=lambda r: r[2])[1], min(maps))
