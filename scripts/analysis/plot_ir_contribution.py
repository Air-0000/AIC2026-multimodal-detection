#!/usr/bin/env python
"""IR(热成像) 到底给哪些类别、在什么光照下带来了 AP —— 用 combo 的消融实验量化。

对照来源 (同一份权重, 只改 IR 输入):
    midboost_combo_{cond}.log              完整模型 (RGB + IR + Depth)
    midboost_combo_ablate_heat_{cond}.log  消融 heat (IR 通路失效)
差值 = IR 的贡献。

产出: outputs/figures/ir_contribution_person.png
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体 (宿主机上 DejaVu Sans 无 CJK 字形, 会渲染成方块)
for f in ("LXGW WenKai", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "AR PL UKai CN"):
    try:
        matplotlib.font_manager.findfont(
            matplotlib.font_manager.FontProperties(family=f), fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f] + list(plt.rcParams["font.sans-serif"])
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

AIC = "/home/constant/CodeBuddy/Dev/AIC"
LOGDIR = os.path.join(AIC, "logs")
OUT = os.path.join(AIC, "outputs")

CATS = ["person", "boat", "animal", "seat", "sign", "bicycle", "car",
        "ball", "light", "garbage can", "uav", "tricycle"]

# 由亮到暗
CONDS = ["day", "night50", "night25", "night12", "night05",
         "night01", "night005", "night002", "night001", "night0005"]
LABEL = {"day": "day", "night50": "0.50", "night25": "0.25", "night12": "0.12",
         "night05": "0.05", "night01": "0.01", "night005": "0.005",
         "night002": "0.002", "night001": "0.001", "night0005": "0.0005"}


def parse(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    m = re.search(r"per-category.*?\n((?:\s+\S.*\n){3,20})", txt)
    if not m:
        return None
    aps = {}
    for ln in m.group(1).splitlines():
        mm = re.match(r"\s+(.+?)\s+AP=([0-9.]+)\s+AP50=([0-9.]+)", ln)
        if mm:
            aps[mm.group(1).strip()] = float(mm.group(2))
    tot = re.search(r"Average Precision.*?IoU=0\.50:0\.95.*?area=\s+all"
                    r".*?maxDets=100\s*\]\s*=\s*([0-9.]+)", txt)
    return aps, (float(tot.group(1)) if tot else None)


def series(prefix):
    """返回 {cond: (aps, mAP)}"""
    out = {}
    for c in CONDS:
        r = parse(os.path.join(LOGDIR, f"{prefix}_{c}.log"))
        if r:
            out[c] = r
    return out


def main():
    full = series("midboost_combo")
    ablate = series("midboost_combo_ablate_heat")
    if not full or not ablate:
        print("数据不足:", len(full), len(ablate))
        return 1

    common = [c for c in CONDS if c in full and c in ablate]
    print(f"可配对条件: {len(common)}\n")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    # ---- 左: person 类 AP, 完整 vs 消融 IR ----
    ax = axes[0]
    x = range(len(common))
    pf = [full[c][0].get("person", 0) for c in common]
    pa = [ablate[c][0].get("person", 0) for c in common]
    mf = [full[c][1] for c in common]
    ma = [ablate[c][1] for c in common]
    ax.plot(x, pf, "-o", color="#C44E52", lw=2.2, ms=7, label="person — 完整模型 (含 IR)")
    ax.plot(x, pa, "--s", color="#C44E52", lw=1.8, ms=6, alpha=0.55,
            label="person — 消融 IR")
    ax.plot(x, mf, "-^", color="#4C72B0", lw=2.0, ms=6, label="mAP — 完整模型")
    ax.plot(x, ma, "--v", color="#4C72B0", lw=1.6, ms=5, alpha=0.55, label="mAP — 消融 IR")
    ax.fill_between(x, pa, pf, color="#C44E52", alpha=0.13)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[c] for c in common])
    ax.set_xlabel("光照强度 (1.0 = 原始亮度)", fontsize=11)
    ax.set_ylabel("AP @ IoU=0.50:0.95", fontsize=11)
    ax.set_title("IR 的贡献随光照变暗而急剧放大\n(midboost-combo, 同一权重只改 IR 输入)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # 标注白天差值
    d_day = pf[0] - pa[0]
    ax.annotate(f"白天 IR 贡献\n仅 +{d_day:.3f}", xy=(0, (pf[0] + pa[0]) / 2),
                xytext=(1.1, 0.62), fontsize=9.5, color="#8B0000",
                arrowprops=dict(arrowstyle="->", color="#8B0000", lw=1.4))
    i_dark = len(common) - 2
    d_dark = pf[i_dark] - pa[i_dark]
    ax.annotate(f"极暗 IR 贡献\n+{d_dark:.3f}", xy=(i_dark, (pf[i_dark] + pa[i_dark]) / 2),
                xytext=(i_dark - 3.4, 0.30), fontsize=9.5, color="#8B0000",
                arrowprops=dict(arrowstyle="->", color="#8B0000", lw=1.4))

    # ---- 右: 各类别 IR 贡献, 白天 vs 极暗 ----
    ax2 = axes[1]
    day_c, dark_c = "day", "night001"
    cats = [c for c in CATS if c in full[day_c][0] and c in ablate[day_c][0]]
    gain_day = [full[day_c][0][c] - ablate[day_c][0][c] for c in cats]
    gain_dark = [full[dark_c][0][c] - ablate[dark_c][0][c] for c in cats]
    y = range(len(cats))
    h = 0.38
    ax2.barh([i + h / 2 for i in y], gain_day, height=h, color="#8DA0CB",
             label=f"白天 (IR 贡献)")
    ax2.barh([i - h / 2 for i in y], gain_dark, height=h, color="#C44E52",
             label=f"极暗 0.001 (IR 贡献)")
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(cats, fontsize=10)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel("IR 带来的 AP 增量 (完整 − 消融 IR)", fontsize=11)
    ax2.set_title("白天几乎零收益，极暗才有决定性收益\n每个类别都是这个模式",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9.5, loc="lower right")
    ax2.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "ir_contribution_person.png")
    plt.savefig(dst, dpi=140, bbox_inches="tight")
    print(f"写出: {dst}\n")

    # ---- 数字表 ----
    print(f"{'条件':>10s} {'mAP完整':>8s} {'mAP无IR':>8s} {'ΔmAP':>7s} "
          f"{'person完整':>10s} {'person无IR':>10s} {'Δperson':>8s}")
    print("-" * 66)
    for c in common:
        print(f"{LABEL[c]:>10s} {full[c][1]:8.4f} {ablate[c][1]:8.4f} "
              f"{full[c][1]-ablate[c][1]:+7.4f} "
              f"{full[c][0].get('person',0):10.4f} "
              f"{ablate[c][0].get('person',0):10.4f} "
              f"{full[c][0].get('person',0)-ablate[c][0].get('person',0):+8.4f}")

    print("\n=== 白天各类别 IR 贡献 ===")
    for c in cats:
        g = full[day_c][0][c] - ablate[day_c][0][c]
        print(f"  {c:14s} {full[day_c][0][c]:.3f} - {ablate[day_c][0][c]:.3f} = {g:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
