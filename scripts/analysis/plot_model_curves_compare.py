import json
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"

data = json.load(open("/home/constant/CodeBuddy/Dev/AIC/outputs/metrics/model_curves_compare.json"))

styles = {
    "D-FINE-X 微调 (best 0.5550@47)": dict(color="#e6194b", lw_thin=1.6, z=10),
    "Stage A 纯RGB ViT-L/800 scratch": dict(color="#4363d8", lw_thin=0.9),
    "Stage B 模态注入": dict(color="#3cb44b", lw_thin=0.9),
    "Stage C 融合微调": dict(color="#911eb4", lw_thin=1.1),
    "combo 基线 (50ep)": dict(color="#f58231", lw_thin=0.9, ls="--"),
    "l4u3 调优成品": dict(color="#000000", lw_thin=1.0, ls="--"),
    "vitl16_l4like 640 scratch": dict(color="#999999", lw_thin=0.8, ls=":"),
}

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.55, 1]})

ax = axes[0]
for label, vals in data.items():
    st = styles[label]
    running = []
    best = -1
    for v in vals:
        best = max(best, v)
        running.append(best)
    ax.plot(range(len(vals)), [v * 100 for v in running], color=st["color"], lw=st.get("lw_thin", 1.1),
            ls=st.get("ls", "-"), label=label, zorder=st.get("z", 5), alpha=0.9)
ax.axhline(52.41, color="#f58231", lw=0.6, alpha=0.4)
ax.set_xlabel("epoch"); ax.set_ylabel("val200 mAP@50-95 (%)")
ax.set_title("各模型 val200 mAP 曲线（累计最优）")
ax.grid(alpha=0.2); ax.legend(fontsize=8.5, loc="lower right")

ax2 = axes[1]
labels, bests, lasts = [], [], []
for label, vals in data.items():
    labels.append(label.replace(" (进行中, ep20/50)", "*"))
    bests.append(max(vals) * 100)
    lasts.append(vals[-1] * 100)
y = range(len(labels))
colors = ["#e6194b" if "D-FINE" in l else "#7f9fd4" for l in labels]
ax2.barh(list(y), bests, color=colors, alpha=0.85, height=0.62)
for i, (b, l) in enumerate(zip(bests, lasts)):
    ax2.text(b + 0.08, i, f"{b:.2f}", va="center", fontsize=9)
ax2.set_yticks(list(y)); ax2.set_yticklabels(labels, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlim(40, 56)
ax2.set_xlabel("best val200 mAP@50-95 (%)")
ax2.set_title("最优值对比（红=D-FINE-X 当前）")
ax2.grid(alpha=0.25, axis="x")

fig.suptitle("D-FINE-X 微调线 vs 已有模型 · val200 mAP@50-95（dfine=best_stg2 ep47）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "/home/constant/CodeBuddy/Dev/AIC/outputs/figures/model_curves_compare.png"
fig.savefig(out, dpi=140)
print("saved", out)
