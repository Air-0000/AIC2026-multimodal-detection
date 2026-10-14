import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# order: easiest (orig) -> hardest (night0005)
conds = ["orig", "night50", "night25", "night12", "night05",
         "night01", "night005", "night002", "night001", "night0005"]
ap_all =  [0.503, 0.485, 0.472, 0.454, 0.407, 0.241, 0.208, 0.182, 0.192, 0.191]
ap50    =  [0.841, 0.827, 0.814, 0.790, 0.721, 0.489, 0.432, 0.401, 0.406, 0.402]

# night level = brightness multiplier (smaller = darker). Label darkness under x.
dark_label = {
    "orig":"clean", "night50":"50x", "night25":"25x", "night12":"12x",
    "night05":"5x", "night01":"1x", "night005":"0.5x", "night002":"0.2x",
    "night001":"0.1x", "night0005":"0.05x",
}

x = np.arange(len(conds))
w = 0.38

fig, ax = plt.subplots(figsize=(12.5, 6.2))
b1 = ax.bar(x - w/2, ap_all, w, label="AP@.5:.95", color="#2563eb", edgecolor="white")
b2 = ax.bar(x + w/2, ap50, w, label="AP50", color="#f59e0b", edgecolor="white")

# orig baseline reference line
ax.axhline(0.503, color="#2563eb", ls="--", lw=1.1, alpha=0.6)
ax.text(len(conds)-0.45, 0.503 + 0.012, "orig AP@.5:.95 = 0.503",
        color="#2563eb", fontsize=8.5, ha="right")

# mark the collapse zone (severe low-light)
ax.axvspan(4.5, len(conds)-0.5, color="#ef4444", alpha=0.07)
ax.text(7.5, 0.93, "severe low-light collapse zone\n(AP@.5:.95 ≈ 0.18–0.24)",
        color="#dc2626", fontsize=9, ha="center", va="top", weight="bold")

ax.set_ylabel("mAP", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title("vitl800 (best) — nightxx differential inference\n"
             "no-TTA, single-scale 800  |  left = clean daylight → right = darkest",
             fontsize=12, weight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n({dark_label[c]})" for c in conds], fontsize=9)
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", ls=":", alpha=0.4)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# value labels
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", (rect.get_x()+rect.get_width()/2, h),
                    ha="center", va="bottom", fontsize=7.2, color="#374151")

fig.text(0.13, 0.02,
         "Takeaway: robust under mild-moderate darkness (night05→orig: 0.41–0.50); collapses on severe "
         "low-light (night0005/001/002 ≈ 0.18–0.19). Polluted config = stronger aug + LLRD-capped lr + fusion from scratch.",
         fontsize=8.5, color="#7a8699")

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = "/home/constant/CodeBuddy/Dev/AIC/outputs/figures/vitl800_night_compare.png"
plt.savefig(out, dpi=130)
print("SAVED", out)
