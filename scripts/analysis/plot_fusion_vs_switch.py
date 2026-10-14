#!/usr/bin/env python
"""可视化：「硬切换」还是「真融合」—— 完整模型 vs 五个单模态专家，随光照变化。

产出 outputs/figures/fusion_vs_switch.png
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# CJK 字体（宿主机 DejaVu Sans 无中文字形）
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

COND = ["day", "night50", "night25", "night12", "night05",
        "night01", "night005", "night002", "night001", "night0005"]
BRIGHT = {"day": 1.0, "night50": 0.50, "night25": 0.25, "night12": 0.12,
          "night05": 0.05, "night01": 0.01, "night005": 0.005,
          "night002": 0.002, "night001": 0.001, "night0005": 0.0005}

TOT = re.compile(r"Average Precision.*?IoU=0\.50:0\.95.*?area=\s+all"
                 r".*?maxDets=100\s*\]\s*=\s*([0-9.]+)")
CAT_ROW = re.compile(r"\s+(.+?)\s+AP=([0-9.]+)\s+AP50=([0-9.]+)")

MODES = [
    ("full", "deepir_eval"),
    ("rgb", "deepir_keep_rgb"),
    ("ir", "deepir_only_ir"),
    ("heat", "deepir_only_heat"),
    ("rawir", "deepir_only_rawir"),
    ("depth", "deepir_only_depth"),
]
SINGLE = ["rgb", "ir", "heat", "rawir", "depth"]
STYLE = {
    "full":  dict(color="#111111", lw=2.6, marker="o", ms=7, ls="-"),
    "rgb":   dict(color="#4C72B0", lw=2.0, marker="s", ms=6, ls="--"),
    "ir":    dict(color="#C44E52", lw=2.0, marker="^", ms=6, ls="-"),
    "heat":  dict(color="#DD8452", lw=1.6, marker="v", ms=5, ls=":"),
    "rawir": dict(color="#55A868", lw=1.6, marker="D", ms=5, ls=":"),
    "depth": dict(color="#8172B3", lw=1.6, marker="P", ms=5, ls=":"),
}
NAME = {"full": "完整模型 (RGB+h+r+d)", "rgb": "仅 RGB", "ir": "仅 IR (h+r)",
        "heat": "仅 heat", "rawir": "仅 rawIR", "depth": "仅 depth"}


def parse(prefix, cond):
    cands = ([f"{prefix}_day.log", f"{prefix}_orig.log"] if cond == "day"
             else [f"{prefix}_{cond}.log"])
    for fn in cands:
        p = os.path.join(LOGDIR, fn)
        if not os.path.exists(p):
            continue
        txt = open(p, errors="ignore").read()
        t = TOT.search(txt)
        if not t:
            continue
        aps = {}
        m = re.search(r"per-category.*?\n((?:\s+\S.*\n){3,20})", txt)
        if m:
            for ln in m.group(1).splitlines():
                mm = CAT_ROW.match(ln)
                if mm:
                    aps[mm.group(1).strip()] = float(mm.group(2))
        return float(t.group(1)), aps
    return None


def main():
    data = {k: {c: parse(p, c) for c in COND} for k, p in MODES}
    have = [c for c in COND if all(data[k][c] for k, _ in MODES)]
    print(f"可配对 {len(have)}/10: {have}")
    if len(have) < 3:
        print("数据不足")
        return 1

    x = [BRIGHT[c] for c in have]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

    # ---------- 面板 A: 专家曲线 ----------
    ax = axes[0]
    for k, _ in MODES:
        ys = [data[k][c][0] for c in have]
        ax.plot(x, ys, label=NAME[k], **STYLE[k])
    # 切换上限
    ceil_v = [max(data[k][c][0] for k in SINGLE) for c in have]
    ax.plot(x, ceil_v, color="gray", lw=1.8, ls="-.", marker="x", ms=6,
            label="切换上限 max(单模态专家)")
    full_v = [data["full"][c][0] for c in have]
    ax.fill_between(x, ceil_v, full_v, color="#C44E52", alpha=0.18,
                    label="融合余量 FULL − max(专家)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("RGB 亮度倍数（左=白天 1.0 → 右=极暗 0.0005）", fontsize=11)
    ax.set_ylabel("mAP @ IoU=0.50:0.95", fontsize=11)
    ax.set_title("完整模型始终高于所有单模态专家\n高出部分 = 真正的互补融合",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="center left")
    ax.grid(alpha=0.3)

    # ---------- 面板 B: 融合余量 ----------
    ax2 = axes[1]
    mar = [f - b for f, b in zip(full_v, ceil_v)]
    idx = range(len(have))
    ax2.bar(idx, mar, color=["#C44E52" if m > 0.02 else "#BBBBBB" for m in mar])
    ax2.set_xticks(list(idx))
    ax2.set_xticklabels([("白天" if c == "day" else f"{BRIGHT[c]:g}") for c in have],
                        fontsize=9)
    ax2.axhline(0, color="k", lw=0.9)
    ax2.axhline(0.02, color="gray", lw=1, ls="--")
    ax2.text(0.05, 0.021, "0.02（显著性参考线）", fontsize=9, color="gray")
    ax2.set_xlabel("RGB 亮度倍数", fontsize=11)
    ax2.set_ylabel("融合余量 = FULL − max(单模态专家)", fontsize=11)
    ax2.set_title("融合余量随变暗单调放大\n⇒ 越暗越依赖「组合」而非「切换」",
                  fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    # ---------- 面板 C: person ----------
    ax3 = axes[2]
    for k in ["full", "rgb", "ir", "heat"]:
        ys = [data[k][c][1].get("person", 0) for c in have]
        st = dict(STYLE[k])
        st["ms"] = 6
        ax3.plot(x, ys, label=NAME[k], **st)
    pceil = [max(data[k][c][1].get("person", 0) for k in SINGLE) for c in have]
    pfull = [data["full"][c][1].get("person", 0) for c in have]
    ax3.fill_between(x, pceil, pfull, color="#4C72B0", alpha=0.18,
                     label="融合余量")
    ax3.set_xscale("log")
    ax3.invert_xaxis()
    ax3.set_xlabel("RGB 亮度倍数", fontsize=11)
    ax3.set_ylabel("person AP @ IoU=0.50:0.95", fontsize=11)
    ax3.set_title("person 类同样：白天靠 RGB，极暗靠 IR\n但完整模型始终 > 任一单模态",
                  fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "fusion_vs_switch.png")
    plt.savefig(dst, dpi=135, bbox_inches="tight")
    print("写出:", dst)

    print(f"\n{'亮度':>8s} {'FULL':>7s} {'max专家':>8s} {'余量':>8s} {'F/专家':>7s}  最佳专家")
    for c, f, b in zip(have, full_v, ceil_v):
        S = {k: data[k][c][0] for k in SINGLE}
        bk = max(S, key=S.get)
        print(f"{BRIGHT[c]:8g} {f:7.4f} {b:8.4f} {f - b:+8.4f} {f / b:7.3f}  {bk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
