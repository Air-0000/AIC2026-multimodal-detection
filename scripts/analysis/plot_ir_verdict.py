#!/usr/bin/env python
"""IR 贡献的最终结论图 —— 三面板，带样本量约束标注。

结论：IR 在正常光照下对任何类别都几乎零收益；它的价值全部集中在极暗档位，
且主要由 heat(热成像) 通道提供。长尾类别样本极少，其 per-class AP 不可用于结论。
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

for _f in ("LXGW WenKai", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "AR PL UKai CN"):
    try:
        matplotlib.font_manager.findfont(
            matplotlib.font_manager.FontProperties(family=_f), fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f] + list(plt.rcParams["font.sans-serif"])
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

AIC = "/home/constant/CodeBuddy/Dev/AIC"
LOGDIR = os.path.join(AIC, "logs")
OUT = os.path.join(AIC, "outputs")

CATS = ["person", "boat", "animal", "seat", "sign", "bicycle", "car",
        "ball", "light", "garbage can", "uav", "tricycle"]
# 验证集真实标注数（200 图 / 1523 框）——长尾类别 AP 不可信
NANN = {"person": 574, "animal": 352, "car": 156, "light": 135, "sign": 76,
        "bicycle": 74, "seat": 73, "garbage can": 35, "uav": 19, "boat": 14,
        "ball": 13, "tricycle": 2}
ABL = ["heat", "rawir", "depth"]
COL = {"heat": "#C44E52", "rawir": "#DD8452", "depth": "#55A868"}
LAB = {"heat": "遮挡 heat(热成像)", "rawir": "遮挡 rawIR(原始红外)",
       "depth": "遮挡 depth(深度)"}
COND = ["day", "night50", "night25", "night12", "night05",
        "night01", "night005", "night002", "night001", "night0005"]
LBL = {"day": "白天", "night50": "0.50", "night25": "0.25", "night12": "0.12",
       "night05": "0.05", "night01": "0.01", "night005": "0.005",
       "night002": "0.002", "night001": "0.001", "night0005": "0.0005"}
DARKX = {"day": 1.0, "night50": 0.50, "night25": 0.25, "night12": 0.12,
         "night05": 0.05, "night01": 0.01, "night005": 0.005,
         "night002": 0.002, "night001": 0.001, "night0005": 0.0005}
TOT = re.compile(r"Average Precision.*?IoU=0\.50:0\.95.*?area=\s+all"
                 r".*?maxDets=100\s*\]\s*=\s*([0-9.]+)")


def parse(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    m = re.search(r"per-category.*?\n((?:\s+\S.*\n){3,20})", txt)
    aps = {}
    if m:
        for ln in m.group(1).splitlines():
            mm = re.match(r"\s+(.+?)\s+AP=([0-9.]+)\s+AP50=([0-9.]+)", ln)
            if mm:
                aps[mm.group(1).strip()] = float(mm.group(2))
    t = TOT.search(txt)
    return aps, (float(t.group(1)) if t else None)


full = {c: parse(os.path.join(LOGDIR, f"midboost_combo_{c}.log")) for c in COND}
abl = {ab: {c: parse(os.path.join(LOGDIR, f"midboost_combo_ablate_{ab}_{c}.log"))
            for c in COND} for ab in ABL}

fig = plt.figure(figsize=(16.5, 5.5))

# ---------- A: mAP vs 亮度 ----------
ax = fig.add_subplot(1, 3, 1)
xs = [DARKX[c] for c in COND]
ax.plot(xs, [full[c][1] for c in COND], "-o", color="black", lw=2.6, ms=7,
        label="完整模型 (6ch)", zorder=5)
for ab in ABL:
    ax.plot(xs, [abl[ab][c][1] for c in COND], "--s", color=COL[ab], lw=1.8, ms=5,
            alpha=0.9, label=LAB[ab])
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("RGB 亮度倍数（左=白天 1.0 → 右=极暗 0.0005）", fontsize=11)
ax.set_ylabel("mAP @ IoU=0.50:0.95", fontsize=11)
ax.set_title("IR 的作用只在「极暗区」出现\n同一权重、同一 val，仅推理时遮挡一个通道",
             fontsize=12, fontweight="bold")
ax.axvspan(0.0004, 0.02, color="#C44E52", alpha=0.07)
ax.text(0.006, 0.02, "极暗区\nIR 决定成败", fontsize=9.5, color="#8B0000",
        ha="center", fontweight="bold")
ax.annotate("白天：四条线几乎重合\nIR 贡献 ≤ +0.014", xy=(1.0, 0.515),
            xytext=(0.30, 0.33), fontsize=9.5, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.3))
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.3)

# ---------- B: 白天逐类别 ----------
ax2 = fig.add_subplot(1, 3, 2)
cats = [c for c in CATS if c in full["day"][0]]
gain = [full["day"][0][c] - abl["heat"]["day"][0].get(c, full["day"][0][c]) for c in cats]
order = sorted(range(len(cats)), key=lambda i: gain[i])
cats = [cats[i] for i in order]
gain = [gain[i] for i in order]
cols = ["#BBBBBB" if NANN[c] < 40 else ("#4C72B0" if gain[i] >= 0 else "#C44E52")
        for i, c in enumerate(cats)]
y = range(len(cats))
ax2.barh(list(y), gain, color=cols, height=0.68)
for i, c in enumerate(cats):
    ax2.text(gain[i] + (0.0015 if gain[i] >= 0 else -0.0015), i,
             f"n={NANN[c]}", va="center",
             ha="left" if gain[i] >= 0 else "right", fontsize=7.6,
             color="#666666")
ax2.set_yticks(list(y))
ax2.set_yticklabels(cats, fontsize=9.5)
ax2.axvline(0, color="k", lw=1.0)
ax2.set_xlabel("白天 IR 带来的 AP 增量（完整 − 遮挡 heat）", fontsize=10.5)
ax2.set_title("白天：任何类别都没有实质收益\n灰色 = 标注 <40，AP 不可信", fontsize=12,
              fontweight="bold")
ax2.set_xlim(-0.035, 0.115)
ax2.grid(alpha=0.3, axis="x")

# ---------- C: person / car 随光照 ----------
ax3 = fig.add_subplot(1, 3, 3)
for cat, ccol in (("person", "#C44E52"), ("car", "#4C72B0")):
    pf = [full[c][0].get(cat, 0) for c in COND]
    pa = [abl["heat"][c][0].get(cat, 0) for c in COND]
    ax3.plot(xs, pf, "-o", color=ccol, lw=2.4, ms=6, label=f"{cat} — 完整")
    ax3.plot(xs, pa, "--s", color=ccol, lw=1.7, ms=5, alpha=0.5,
             label=f"{cat} — 无 IR")
    ax3.fill_between(xs, pa, pf, color=ccol, alpha=0.10)
ax3.set_xscale("log")
ax3.invert_xaxis()
ax3.axvspan(0.0004, 0.02, color="#C44E52", alpha=0.07)
ax3.annotate("person 白天 +0.010\n(574 个标注 → 基本等于没提升)", xy=(1.0, 0.467),
             xytext=(0.055, 0.53), fontsize=9, color="#8B0000",
             arrowprops=dict(arrowstyle="->", color="#8B0000", lw=1.3))
ax3.annotate("极暗 +0.177", xy=(0.001, 0.10), xytext=(0.09, 0.16), fontsize=9,
             color="#8B0000", arrowprops=dict(arrowstyle="->", color="#8B0000", lw=1.3))
ax3.set_xlabel("RGB 亮度倍数", fontsize=11)
ax3.set_ylabel("AP @ IoU=0.50:0.95", fontsize=11)
ax3.set_title("person / car：白天平，暗光断崖\n阴影 = IR 的贡献", fontsize=12,
              fontweight="bold")
ax3.legend(fontsize=9, loc="upper right")
ax3.grid(alpha=0.3)

plt.tight_layout()
os.makedirs(OUT, exist_ok=True)
dst = os.path.join(OUT, "ir_contribution_person.png")
plt.savefig(dst, dpi=135, bbox_inches="tight")
print("写出:", dst)
