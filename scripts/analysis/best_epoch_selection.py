#!/usr/bin/env python3
"""训练轨迹侧的降噪：对比「看读数」的去噪 vs 「改权重」的去噪。

核心问题：在 val mAP 序列上 argmax 选 best epoch，到底是在挑真实的好权重，
还是在追噪声？用**去趋势 + 奇偶留出**来无循环地检验。

  1. 对 mAP 序列 x_t 去掉线性趋势得残差 r_t（训练本就有上升趋势，不去会假阳性）
  2. 在奇数位的残差上 argmax -> 位置 p，对应 epoch e*
  3. 在**偶数位**（留出集）找离 e* 最近的 epoch，取其残差 r_test
  4. 若 argmax 追的是真信号 => r_test 应明显为正；纯噪声 => r_test ≈ 0

同时量化几种可选做法：
  * MA-k 滑动平均后再 argmax（零成本后处理）
  * 用 EMA/SWA 在**权重空间**做同样的事（需真跑，本脚本只能给出上界）

用法: python scripts/analysis/best_epoch_selection.py A:exp1 B:exp2 ...
"""
import argparse
import json
import math
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "vendor", "DEIMv2", "outputs")
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))
from dump_epoch_series import series  # noqa: E402


def var(x):
    m = sum(x) / len(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1) if len(x) > 1 else 0.0


def sd_noise(x):
    """趋势稳健的逐点不可解释抖动 = sqrt(Var(一阶差分)/2)。"""
    if len(x) < 3:
        return float("nan")
    d = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    return math.sqrt(var(d) / 2.0)


def movavg(x, k):
    if len(x) < k:
        return list(x)
    out = [sum(x[max(0, i - k // 2): min(len(x), i + k // 2 + 1)]) /
           len(x[max(0, i - k // 2): min(len(x), i + k // 2 + 1)])
           for i in range(len(x))]
    return out


def detrend(x):
    """去掉线性趋势，返回残差（对照用；对有阶段结构的训练曲线不合适）。"""
    n = len(x)
    if n < 3:
        return list(x)
    t = list(range(n))
    mt = sum(t) / n
    mx = sum(x) / n
    num = sum((t[i] - mt) * (x[i] - mx) for i in range(n))
    den = sum((t[i] - mt) ** 2 for i in range(n))
    b = num / den if den else 0.0
    a = mx - b * mt
    return [x[i] - (a + b * t[i]) for i in range(n)]


def detrend_ma(x, k=9):
    """高通：减去中心滑动均值。

    线性去趋势对「先快升后平台」的训练曲线不成立——端点残差巨大，会让 argmax
    恒选首/尾。这里改用中心 MA 作为局部趋势，残差才是真正的逐 epoch 抖动。
    k 应为奇数；n 小时自动缩小。
    """
    n = len(x)
    if n < 3:
        return list(x)
    k = min(k if k % 2 else k + 1, n - (1 - n % 2))
    k = max(k, 3)
    trend = movavg(x, k)
    return [x[i] - trend[i] for i in range(n)]


def holdout_test(vals, epochs, key, nperm=4000, seed=0, margin=0):
    """奇偶留出检验。key(lst)->argmax index。返回 (留出值, 随机零假设均值/sd)

    margin: 两端各排除的点数——中心 MA 在边缘是退化窗口，残差不可信，
            不排除的话 argmax 恒选首/尾，这是纯粹的边缘伪影。
    """
    n = len(vals)
    if n < 6:
        return None
    even_idx = [i for i in range(0, n, 2)]        # 选择集（含位置 0）
    odd_idx = [i for i in range(1, n, 2)]         # 留出集
    trim = lambda L: [i for i in L if margin <= i < n - margin]  # noqa: E731
    odd, even = trim(even_idx), trim(odd_idx)
    if len(odd) < 3 or len(even) < 3:
        return None
    p = key([vals[i] for i in odd])
    e_star = epochs[odd[p]]
    # 留出集里离 e_star 最近的 epoch
    j = min(even, key=lambda i: abs(epochs[i] - e_star))
    observed = vals[j]
    rng = random.Random(seed)
    null = []
    for _ in range(nperm):
        pi = list(odd)
        rng.shuffle(pi)
        pe = epochs[pi[rng.randrange(len(pi))]]
        pj = min(even, key=lambda i: abs(epochs[i] - pe))
        null.append(vals[pj])
    m = sum(null) / len(null)
    s = math.sqrt(var(null)) if len(null) > 1 else 0.0
    z = (observed - m) / s if s > 0 else float("nan")
    return dict(epoch_selected=e_star, leaveout_epoch=epochs[j],
                observed=observed, null_mean=m, null_sd=s, z=z,
                pct=float(sum(1 for v in null if v < observed) / len(null)))


def argmax_idx(v):
    return max(range(len(v)), key=lambda i: v[i])


def analyze(tag, exp, out):
    rows = series(exp)
    if not rows:
        print("== %s : 无 log" % tag)
        return None
    rows = [r for r in rows if r["ap"] is not None]
    if len(rows) < 6:
        print("== %s : 点数不足 (%d)" % (tag, len(rows)))
        return None
    # 去重：同一 epoch 可能被 resume 写了两次，取最后一次
    seen = {}
    for r in rows:
        seen[r["epoch"]] = r
    rows = [seen[e] for e in sorted(seen)]
    ep = [r["epoch"] for r in rows]
    x = [r["ap"] for r in rows]

    print("\n" + "=" * 78)
    print("%s  (%s)   epoch %d-%d   n=%d   best=%.4f@ep%d"
          % (tag, exp, ep[0], ep[-1], len(x), max(x), ep[argmax_idx(x)]))
    print("=" * 78)

    r = detrend(x)
    res = dict(tag=tag, exp=exp, epochs=ep, ap=x, best_epoch=ep[argmax_idx(x)],
               best=max(x))
    print("%-22s %10s %10s" % ("处理方式", "sd_noise(pt)", "降噪倍数"))
    s0 = sd_noise(x)
    print("%-22s %10.2f %10s" % ("原始读数", s0 * 100, "1.00x"))
    res["sd_raw"] = s0
    for k in (3, 5):
        sk = sd_noise(movavg(x, k))
        print("%-22s %10.2f %10s" % ("MA-%d 滑动平均" % k, sk * 100,
                                     "%.2fx" % (s0 / sk) if sk else "?"))
        res["sd_ma%d" % k] = sk

    print("\n--- 留出检验：奇偶分半，在奇数位上选，在偶数位上验证 ---")
    print("%-26s %8s %11s %10s %8s %8s"
          % ("策略", "选中ep", "留出值(pt)", "随机基线", "z", "百分位"))
    rh = detrend_ma(x, 9)
    print("  (高通残差峰值 ±%.2f pt；趋势用中心 MA-9)"
          % (max(abs(v) for v in rh) * 100))
    strategies = [("argmax(原始)", rh),
                  ("argmax(MA3)", detrend_ma(movavg(x, 3), 9)),
                  ("argmax(MA5)", detrend_ma(movavg(x, 5), 9))]
    res["strategies"] = {}
    for si, (name, vals) in enumerate(strategies):
        o = holdout_test(vals, ep, argmax_idx, margin=4 if si else 2)
        if o is None:
            continue
        res["strategies"][name] = o
        flag = "  <-- 追噪声" if o["pct"] < 0.90 else ""
        print("%-26s %8d %11.2f %10.2f %8.2f %7.1f%%%s"
              % (name, o["epoch_selected"], o["observed"] * 100,
                 o["null_mean"] * 100, o["z"], o["pct"] * 100, flag))
    ol = holdout_test(detrend(x), ep, argmax_idx)
    if ol:
        res["strategies"]["对照:线性去趋势"] = ol
        print("%-26s %8d %11.2f %10.2f %8.2f %7.1f%%  (端点残差污染)"
              % ("对照:线性去趋势", ol["epoch_selected"], ol["observed"] * 100,
                 ol["null_mean"] * 100, ol["z"], ol["pct"] * 100))
    res["resid_hp"] = rh
    out[tag] = res
    return res


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", nargs="*", default=None,
                     help="TAG:exp_name ...")
    ap_.add_argument("--json", default=os.path.join(ROOT, "outputs",
                                                    "best_epoch_selection.json"))
    a = ap_.parse_args()
    if a.stage:
        pairs = []
        for s in a.stage:
            if ":" in s:
                t, e = s.split(":", 1)
            else:
                t, e = s, s
            pairs.append((t, e))
    else:
        pairs = [("A", "vitl800_scratch_stageA_rgb"),
                 ("B", "vitl800_scratch_stageB_mod"),
                 ("C", "vitl800_scratch_stageC_fuse")]

    out = {}
    for t, e in pairs:
        analyze(t, e, out)

    print("\n" + "=" * 78)
    print("判读：z < 1.5 或百分位 < 90%  => argmax 选出的 epoch 在留出集上"
          "没有超额，说明它在追噪声。")
    print("      MA-k 能把「读数」降噪 %s，但那只是报告口径；"
          "真正让模型更强的是在权重空间做平均（EMA / SWA / soup）。"
          % "/".join("%.1fx" % (out[t]["sd_raw"] / out[t].get("sd_ma3", 1))
                     for t in out if out[t].get("sd_ma3")))
    print("=" * 78)

    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved ->", a.json)


if __name__ == "__main__":
    main()
