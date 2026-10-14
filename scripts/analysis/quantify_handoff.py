#!/usr/bin/env python
"""检验假设：模型是否只是"随亮度从纯 RGB 平滑切换到纯 IR"，而非真正融合。

数据来源（同一权重 _x_deepir/best_stg2.pth + 同一 val 配置，只改推理输入）：
    deepir_eval_{cond}.log            完整 6ch              -> FULL
    deepir_keep_rgb_{cond}.log        只留 RGB(遮挡 aux×3)   -> R
    deepir_keep_heat_{cond}.log       只留 heat             -> H
    deepir_keep_rawir_{cond}.log      只留 rawIR            -> IR
    deepir_keep_depth_{cond}.log      只留 depth            -> D

判据：
    若 F ≈ max(单模态)        -> 只是"按可靠性切换专家"（handoff / 选择器）
    若 F >  max(单模态) 有明显余量 -> 存在真正的互补融合（fusion）
    complementarity = F - max(R, H, IR, D)
"""
import os
import re
import sys

AIC = "/home/constant/CodeBuddy/Dev/AIC"
LOGDIR = os.path.join(AIC, "logs")

COND = ["day", "night50", "night25", "night12", "night05",
        "night01", "night005", "night002", "night001", "night0005"]
LABEL = {"day": "1.0(白天)", "night50": "0.50", "night25": "0.25", "night12": "0.12",
         "night05": "0.05", "night01": "0.01", "night005": "0.005",
         "night002": "0.002", "night001": "0.001", "night0005": "0.0005"}

TOT = re.compile(r"Average Precision.*?IoU=0\.50:0\.95.*?area=\s+all"
                 r".*?maxDets=100\s*\]\s*=\s*([0-9.]+)")
CAT_ROW = re.compile(r"\s+(.+?)\s+AP=([0-9.]+)\s+AP50=([0-9.]+)")

MODES = [("full", "deepir_eval"), ("rgb", "deepir_keep_rgb"),
         ("heat", "deepir_keep_heat"), ("rawir", "deepir_keep_rawir"),
         ("depth", "deepir_keep_depth")]


def parse(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    t = TOT.search(txt)
    if not t:
        return None
    aps = {}
    m = re.search(r"per-category.*?\n((?:\s+\S.*\n){3,20})", txt)
    if m:
        for ln in m.group(1).splitlines():
            mm = CAT_ROW.match(ln)
            if mm:
                aps[mm.group(1).strip()] = float(mm.group(2))
    return float(t.group(1)), aps


def main():
    data = {}
    for name, pref in MODES:
        data[name] = {}
        for c in COND:
            f = f"{pref}_{c}.log" if c != "day" else f"{pref}_orig.log"
            data[name][c] = parse(os.path.join(LOGDIR, f))

    have = [c for c in COND if all(data[m][c] for m, _ in MODES)]
    print(f"可配对条件: {len(have)}/10   权重: _x_deepir/best_stg2.pth\n")

    # ---------------- 主表 ----------------
    print("=" * 92)
    print("【主表】完整模型 vs 单模态专家 (mAP@IoU=0.50:0.95)")
    print("=" * 92)
    hdr = (f"{'亮度':>10s} {'FULL':>7s} | {'R(仅RGB)':>9s} {'H(仅heat)':>10s} "
           f"{'IR(仅raw)':>10s} {'D(仅深度)':>10s} | {'最佳单模态':>10s} "
           f"{'融合余量':>9s} {'F/最佳':>7s}")
    print(hdr)
    print("-" * 92)
    rows = []
    for c in have:
        F = data["full"][c][0]
        R = data["rgb"][c][0]
        H = data["heat"][c][0]
        I = data["rawir"][c][0]
        D = data["depth"][c][0]
        best = max(R, H, I, D)
        best_name = ["R", "H", "IR", "D"][[R, H, I, D].index(best)]
        margin = F - best
        rows.append((c, F, R, H, I, D, best, best_name, margin))
        print(f"{LABEL[c]:>10s} {F:7.4f} | {R:9.4f} {H:10.4f} {I:10.4f} {D:10.4f} | "
              f"{best:10.4f} {margin:+9.4f} {F / best:7.3f}   (最佳={best_name})")

    print("-" * 92)
    mF = sum(r[1] for r in rows) / len(rows)
    mB = sum(r[6] for r in rows) / len(rows)
    print(f"{'均值':>10s} {mF:7.4f} | {'':9s} {'':10s} {'':10s} {'':10s} | "
          f"{mB:10.4f} {mF - mB:+9.4f} {mF / mB:7.3f}")

    # ---------------- 切换点 ----------------
    print()
    print("=" * 92)
    print("【切换点】RGB 专家 vs IR 专家，谁更强（交叉点 = 模式切换的位置）")
    print("=" * 92)
    print(f"{'亮度':>10s} {'仅RGB':>9s} {'仅heat':>9s} {'差值(R-H)':>11s}  胜者")
    print("-" * 92)
    for r in rows:
        c, F, R, H = r[0], r[1], r[2], r[3]
        win = "RGB" if R > H else "heat"
        print(f"{LABEL[c]:>10s} {R:9.4f} {H:9.4f} {R - H:+11.4f}  {win}")

    # ---------------- 关键判据 ----------------
    print()
    print("=" * 92)
    print("【判据】融合余量 = FULL − 最佳单模态")
    print("=" * 92)
    pos = [r for r in rows if r[8] > 0.02]
    print(f"  融合余量 > 0.02 (真正互补) 的档位: {len(pos)}/10")
    for r in pos:
        print(f"    {LABEL[r[0]]:>10s}  FULL {r[1]:.4f} vs 最佳单模态 {r[6]:.4f} ({r[7]}) "
              f"=> +{r[8]:.4f}")
    mx = max(rows, key=lambda r: r[8])
    print(f"  最大融合余量: {mx[8]:+.4f} @ 亮度 {LABEL[mx[0]]}")
    print(f"  平均融合余量: {mF - mB:+.4f}，FULL/最佳单模态 = {mF / mB:.3f}")

    # ---------------- person 类同指标 ----------------
    print()
    print("=" * 92)
    print("【person 类】同样的分解")
    print("=" * 92)
    print(f"{'亮度':>10s} {'FULL':>7s} {'仅RGB':>7s} {'仅heat':>7s} {'最佳':>7s} "
          f"{'融合余量':>9s}")
    print("-" * 92)
    for r in rows:
        c = r[0]
        pF = data["full"][c][1].get("person", 0)
        pR = data["rgb"][c][1].get("person", 0)
        pH = data["heat"][c][1].get("person", 0)
        pb = max(pR, pH)
        print(f"{LABEL[c]:>10s} {pF:7.3f} {pR:7.3f} {pH:7.3f} {pb:7.3f} {pF - pb:+9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
