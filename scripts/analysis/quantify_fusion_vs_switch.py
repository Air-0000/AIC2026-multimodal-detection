#!/usr/bin/env python
"""判定：模型是否只是"随亮度从纯 RGB 平滑切换到纯 IR"，而非真正融合。

数据（同一权重 _x_deepir/best_stg2.pth + 同一 val 配置，仅改推理输入通道）：

  模式                        ABLATE                    日志前缀
  full      (RGB+h+R+d)       无                        deepir_eval_{orig|night*}
  rgb       (仅 RGB)          heat,rawir,depth          deepir_keep_rgb_{day|night*}
  rgb_heat  (RGB+h)           rawir,depth               deepir_keep_heat_*
  rgb_rawir (RGB+r)           heat,depth                deepir_keep_rawir_*
  rgb_depth (RGB+d)           heat,rawir                deepir_keep_depth_*
  ir        (h+r, 无 RGB)     rgb,depth                 deepir_only_ir_*
  heat      (仅 h)            rgb,rawir,depth           deepir_only_heat_*
  rawir     (仅 r)            rgb,heat,depth            deepir_only_rawir_*
  depth     (仅 d)            rgb,heat,rawir            deepir_only_depth_*

判据：F_switch = max(所有单模态专家)。F 明显高于它 => 存在真正的互补融合。
"""
import os
import re
import sys

AIC = "/home/constant/CodeBuddy/Dev/AIC"
LOGDIR = os.path.join(AIC, "logs")

COND = ["day", "night50", "night25", "night12", "night05",
        "night01", "night005", "night002", "night001", "night0005"]
LABEL = {"day": "1.0白天", "night50": "0.50", "night25": "0.25", "night12": "0.12",
         "night05": "0.05", "night01": "0.01", "night005": "0.005",
         "night002": "0.002", "night001": "0.001", "night0005": "0.0005"}

TOT = re.compile(r"Average Precision.*?IoU=0\.50:0\.95.*?area=\s+all"
                 r".*?maxDets=100\s*\]\s*=\s*([0-9.]+)")
CAT_ROW = re.compile(r"\s+(.+?)\s+AP=([0-9.]+)\s+AP50=([0-9.]+)")

# (key, log prefix)  —— full 的 day 档文件名是 _orig，其余都是 _day
MODES = [
    ("full",      "deepir_eval"),
    ("rgb",       "deepir_keep_rgb"),
    ("rgb_heat",  "deepir_keep_heat"),
    ("rgb_rawir", "deepir_keep_rawir"),
    ("rgb_depth", "deepir_keep_depth"),
    ("ir",        "deepir_only_ir"),
    ("heat",      "deepir_only_heat"),
    ("rawir",     "deepir_only_rawir"),
    ("depth",     "deepir_only_depth"),
]
SINGLE = ["rgb", "ir", "heat", "rawir", "depth"]


def parse(prefix, cond):
    if cond == "day":
        cands = [f"{prefix}_day.log", f"{prefix}_orig.log"]
    else:
        cands = [f"{prefix}_{cond}.log"]
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

    missing = [(k, c) for k in data for c in COND if data[k][c] is None]
    if missing:
        print("缺失档位:", missing[:12], f"... 共 {len(missing)}")
        print()

    have = [c for c in COND if all(data[k][c] for k, _ in MODES)]
    print(f"可配对: {len(have)}/10   权重 _x_deepir/best_stg2.pth\n")

    # ---------------- 表 1: full vs 单模态专家 ----------------
    print("=" * 100)
    print("【表1】完整模型 vs 五个「单模态专家」")
    print("=" * 100)
    print(f"{'亮度':>8s} {'FULL':>7s} | {'仅RGB':>7s} {'仅IR(hr)':>8s} {'仅heat':>7s} "
          f"{'仅rawIR':>7s} {'仅depth':>7s} | {'最佳专家':>8s} {'切换上限':>8s} "
          f"{'融合余量':>8s} {'F/上限':>7s}")
    print("-" * 100)
    rows = []
    for c in have:
        F = data["full"][c][0]
        S = {k: data[k][c][0] for k in SINGLE}
        best_k = max(S, key=S.get)
        best = S[best_k]
        margin = F - best
        rows.append((c, F, S, best, margin))
        print(f"{LABEL[c]:>8s} {F:7.4f} | {S['rgb']:7.4f} {S['ir']:8.4f} "
              f"{S['heat']:7.4f} {S['rawir']:7.4f} {S['depth']:7.4f} | "
              f"{best:8.4f} {best:8.4f} {margin:+8.4f} {F / best:7.3f}  ({best_k})")

    # ---------------- 表 2: RGB 之上逐路增量 ----------------
    print()
    print("=" * 100)
    print("【表2】以纯 RGB 为基准，逐路加辅助模态的增量")
    print("=" * 100)
    print(f"{'亮度':>8s} {'RGB':>7s} {'+heat':>7s} {'+rawIR':>7s} {'+depth':>7s} "
          f"{'全部(=FULL)':>11s} | {'Δ(+heat)':>9s} {'Δ(+raw)':>8s} {'Δ(+dep)':>8s} "
          f"{'Δ(全)':>7s}")
    print("-" * 100)
    for c in have:
        R = data["rgb"][c][0]
        print(f"{LABEL[c]:>8s} {R:7.4f} {data['rgb_heat'][c][0]:7.4f} "
              f"{data['rgb_rawir'][c][0]:7.4f} {data['rgb_depth'][c][0]:7.4f} "
              f"{data['full'][c][0]:11.4f} | "
              f"{data['rgb_heat'][c][0] - R:+9.4f} {data['rgb_rawir'][c][0] - R:+8.4f} "
              f"{data['rgb_depth'][c][0] - R:+8.4f} {data['full'][c][0] - R:+7.4f}")

    # ---------------- 表 3: person ----------------
    print()
    print("=" * 100)
    print("【表3】person 类")
    print("=" * 100)
    print(f"{'亮度':>8s} {'FULL':>7s} {'仅RGB':>7s} {'仅IR(hr)':>8s} {'仅heat':>7s} "
          f"{'最佳':>7s} {'融合余量':>8s}")
    print("-" * 100)
    for c in have:
        F = data["full"][c][1].get("person", 0)
        S = {k: data[k][c][1].get("person", 0) for k in SINGLE}
        best = max(S.values())
        print(f"{LABEL[c]:>8s} {F:7.3f} {S['rgb']:7.3f} {S['ir']:8.3f} "
              f"{S['heat']:7.3f} {best:7.3f} {F - best:+8.3f}")

    # ---------------- 判据 ----------------
    print()
    print("=" * 100)
    print("【判据】若模型只是「按可靠性切换专家」=> FULL ≈ max(专家)；")
    print("       若 FULL 持续高于 max(专家) => 存在互补融合。")
    print("=" * 100)
    mar = [r[4] for r in rows]
    mean_F = sum(r[1] for r in rows) / len(rows)
    mean_B = sum(r[3] for r in rows) / len(rows)
    print(f"  平均融合余量 = {mean_F - mean_B:+.4f}   （FULL 均值 {mean_F:.4f} / 专家均值 {mean_B:.4f}）")
    print(f"  融合余量为正的档位数: {sum(1 for v in mar if v > 0)}/{len(mar)}")
    print(f"  融合余量 > 0.02 的档位数: {sum(1 for v in mar if v > 0.02)}/{len(mar)}")
    print()
    print(f"  {'亮度':>8s} {'FULL':>7s} {'最佳专家':>9s} {'余量':>8s} {'F/专家':>8s}")
    print("  " + "-" * 44)
    for c, F, S, best, margin in rows:
        bk = max(S, key=S.get)
        print(f"  {LABEL[c]:>8s} {F:7.4f} {best:9.4f} {margin:+8.4f} {F / best:8.3f}  ({bk})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
