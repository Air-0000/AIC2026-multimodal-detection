import re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

vitl = {
    "orig": 0.503, "night50": 0.485, "night25": 0.472, "night12": 0.454,
    "night05": 0.407, "night01": 0.241, "night005": 0.208, "night002": 0.182,
    "night001": 0.192, "night0005": 0.191,
}

LOGDIR = "/home/constant/CodeBuddy/Dev/AIC/logs"
# require AP (not AR): anchor on "Average Precision  (AP)"
pat = re.compile(r"Average Precision\s+\(AP\) @\[\s*IoU=0\.50:0\.95 \| area=\s*all \| maxDets=100 \] = ([0-9.]+)")
cond_file = {"orig": "day", "night0005": "night0005", "night001": "night001",
             "night002": "night002", "night005": "night005", "night01": "night01",
             "night05": "night05", "night12": "night12", "night25": "night25",
             "night50": "night50"}
l4u3 = {}
for cond, fname in cond_file.items():
    p = os.path.join(LOGDIR, f"midboost_l4u3_eval_{fname}.log")
    v = None
    for line in open(p, "r", errors="ignore"):
        m = pat.search(line)
        if m:
            v = float(m.group(1)); break
    l4u3[cond] = v
    print(f"l4u3 {cond}: {v}")

order = ["orig", "night50", "night25", "night12", "night05",
         "night01", "night005", "night002", "night001", "night0005"]
dark = {"orig":"clean","night50":"50x","night25":"25x","night12":"12x",
        "night05":"5x","night01":"1x","night005":"0.5x","night002":"0.2x",
        "night001":"0.1x","night0005":"0.05x"}

x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(13, 6.4))
b1 = ax.bar(x - w/2, [vitl[c] for c in order], w, label="vitl800 (best, 800-eval)", color="#2563eb", edgecolor="white")
b2 = ax.bar(x + w/2, [l4u3[c] for c in order], w, label="l4u3 (fine-tuned, 640-eval)", color="#16a34a", edgecolor="white")
ax.axhline(vitl["orig"], color="#2563eb", ls="--", lw=1.0, alpha=0.55)
ax.axhline(l4u3["orig"], color="#16a34a", ls="--", lw=1.0, alpha=0.55)
ax.text(len(order)-0.45, vitl["orig"]+0.012, f"vitl800 orig={vitl['orig']:.3f}", color="#2563eb", fontsize=8.2, ha="right")
ax.text(len(order)-0.45, l4u3["orig"]+0.012, f"l4u3 orig={l4u3['orig']:.3f}", color="#16a34a", fontsize=8.2, ha="right")
ax.axvspan(4.5, len(order)-0.5, color="#ef4444", alpha=0.07)
ax.text(7.5, 0.95, "severe low-light collapse zone\n(AP@.5:.95 ≈ 0.18–0.30)", color="#dc2626", fontsize=9, ha="center", va="top", weight="bold")
ax.set_ylabel("AP@.5:.95 (no-TTA)", fontsize=11); ax.set_ylim(0, 1.0)
ax.set_title("vitl800 (best)  vs  l4u3 (fine-tuned) — nightxx differential inference\n"
             "left = clean daylight → right = darkest  |  vitl800 @800-eval, l4u3 @640-eval (native resolutions)",
             fontsize=11.5, weight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"{c}\n({dark[c]})" for c in order], fontsize=9)
ax.legend(loc="upper right", fontsize=9.5); ax.grid(axis="y", ls=":", alpha=0.4)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for bars in (b1,b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", (rect.get_x()+rect.get_width()/2, h), ha="center", va="bottom", fontsize=6.6, color="#374151")
fig.text(0.13, 0.02,
         "FIXED: l4u3 numbers now use AP@.5:.95 (area=all), not AR. Caveat: l4u3 is a multi-variable fine-tune; eval resolutions differ (800 vs 640). Not a clean backbone comparison.",
         fontsize=8, color="#7a8699")
plt.tight_layout(rect=[0,0.04,1,1])
out = "/home/constant/CodeBuddy/Dev/AIC/outputs/figures/vitl800_vs_l4u3_night_compare.png"
plt.savefig(out, dpi=130); print("SAVED", out)
