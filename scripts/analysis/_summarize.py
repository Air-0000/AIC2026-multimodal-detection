import re, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR
W = "/home/constant/CodeBuddy/Dev/AIC/logs"
# 暗 -> 亮 顺序
levels = ["night0005","night001","night002","night005","night01","night05",
          "night12","night25","night50","nightify","orig"]
data = {}
for lv in levels:
    log = f"{W}/deepir_eval_{lv}.log"
    if not os.path.exists(log):
        data[lv] = (None, None); continue
    t = open(log, encoding="utf-8", errors="ignore").read()
    m = re.search(r"IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)", t)
    m50 = re.search(r"IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)", t)
    data[lv] = (float(m.group(1)) if m else None, float(m50.group(1)) if m50 else None)

# 写汇总
lines = ["level        AP@0.5:0.95   AP@0.50"]
for lv in levels:
    ap, ap50 = data[lv]
    lines.append(f"{lv:12s} {('%.4f'%ap) if ap is not None else '  --  ':>12s}   {('%0.4f'%ap50) if ap50 is not None else '  --  '}")
out = "\n".join(lines)
print(out)
open(os.path.join(LOG_TEXT_DIR, "deepir_eval_matrix_summary.txt"),"w").write(out+"\n")

# 画图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
xs = list(range(len(levels)))
aps = [data[lv][0] for lv in levels]
ap50s = [data[lv][1] for lv in levels]
fig, ax = plt.subplots(figsize=(11,5.2))
ax.plot(xs, aps, "o-", color="#c0392b", label="mAP@0.5:0.95", lw=2, ms=7)
ax.plot(xs, ap50s, "s--", color="#2980b9", label="AP@0.50", lw=2, ms=6)
for x, v in zip(xs, aps):
    if v is not None: ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0,8), ha="center", fontsize=8, color="#c0392b")
for x, v in zip(xs, ap50s):
    if v is not None: ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0,-14), ha="center", fontsize=8, color="#2980b9")
ax.set_xticks(xs); ax.set_xticklabels(levels, rotation=35, ha="right")
ax.set_ylim(0, 1.0); ax.set_ylabel("AP")
ax.set_title("deepir (DEIMv2-X 6ch) road valid — mAP vs night level\nckpt: best_stg2.pth")
ax.grid(alpha=0.3); ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "deepir_night_mAP.png"), dpi=130)
print("saved", os.path.join(FIG_DIR, "deepir_night_mAP.png"))
