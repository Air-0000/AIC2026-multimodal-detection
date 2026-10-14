#!/usr/bin/env python3
"""
wobble_whitenoise_test.py -- 判定「逐 epoch mAP 起伏」是评估噪声还是训练动态（lr 问题）

方法（对每个 stage 的 flat-lr 区段）:
  设观测序列 x_t = f(t) + e_t,  f 为真实权重轨迹, e 为评估噪声.
  一阶差分 d_t = x_t - x_{t-1}.
  若 f 在局部近似线性, 则 Var(d) = 2 * Var(e).
    R = Var(d)/2 / Var(x) 可解释为「观测方差中噪声占比」:
      R ~ 1  -> 观测方差几乎全是评估噪声, 权重轨迹看不出起伏
      R << 1 -> 权重轨迹本身在起伏 (真信号)
  若 e 为白噪声, d 为 MA(1) 且 lag-1 自相关 rho1(d) = -0.5.
      rho1 ~ -0.5 -> level 序列是「常值 + 白噪声」
      rho1 ~  0   -> level 序列是随机游走 / 平滑漂移 (真动态主导)

内部对照: train_loss 是 150 iter 的均值, 天然平滑.
  若 loss 的 R << 1 而 mAP 的 R ~ 1, 说明训练过程平滑、抖动来自评估端.

跨模型对照: A(RGB-only) / B(RGB冻结+模态注入) / C(联合微调) 三者的
  可训练参数、梯度来源、lr 制度完全不同. 若三者 R 都 ~1, 则抖动是
  评测集(val200)的属性, 与 lr 无关.
"""
import json
import math
import os
import sys

LOG_ROOT = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs"
PC_JSON = "/home/constant/CodeBuddy/Dev/AIC/outputs/metrics/perclass_variance.json"

STAGES = [
    ("A  RGB-only",        "vitl800_scratch_stageA_rgb"),
    ("B  RGB冻结+注入",     "vitl800_scratch_stageB_mod"),
    ("C  联合微调",         "vitl800_scratch_stageC_fuse"),
]

BASELINE = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo"

# 参照基线 midboost_combo：手工指定窗口（前者=flat lr 段，后者=cosine/收敛尾段）
JOBS = [
    (BASELINE, "baseline combo  flat lr(1e-4) 段", 3, 24),
    (BASELINE, "baseline combo  尾段(cosine+收敛)", 25, 49),
]


def load(name):
    p = os.path.join(LOG_ROOT, name, "log.txt")
    if not os.path.exists(p):
        return None
    rows = []
    for line in open(p):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if "epoch" not in o:
            continue
        s = o.get("test_coco_eval_bbox")
        ap = s[0] if isinstance(s, (list, tuple)) and len(s) >= 1 else None
        rows.append(dict(epoch=o["epoch"], lr=o.get("train_lr"),
                         loss=o.get("train_loss"), ap=ap))
    # 去重（resume 会重写同一 epoch 的行）：保留最后一条
    seen = {}
    for r in rows:
        seen[r["epoch"]] = r
    return [seen[k] for k in sorted(seen)]


def flat_window(rows, rel_tol=0.02):
    """返回 lr 近似恒定的最长连续 epoch 区间（下标）"""
    best = (0, -1)
    i = 0
    n = len(rows)
    while i < n:
        if rows[i]["lr"] is None:
            i += 1
            continue
        j = i
        while (j + 1 < n and rows[j + 1]["lr"] is not None
               and abs(rows[j + 1]["lr"] - rows[i]["lr"]) <= rel_tol * rows[i]["lr"]
               and rows[j + 1]["epoch"] == rows[j]["epoch"] + 1):
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j + 1
    return best


def var(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def rho1(xs):
    n = len(xs)
    if n < 3:
        return float("nan")
    m = sum(xs) / n
    den = sum((x - m) ** 2 for x in xs)
    num = sum((xs[i] - m) * (xs[i + 1] - m) for i in range(n - 1))
    return num / den


def pearson(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    ca = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return ca / (va * vb) if va and vb else float("nan")


def report(tag, x, scale=None, unit=""):
    """x: 序列. 返回 dict.
    sd_noise = sqrt(Var(d)/2)  <- 趋势稳健的「逐点不可解释抖动」估计
                 (假设真实轨迹局部线性时, Var(d)=2*Var(noise))
    R        = sd_noise^2 / Var(x)  <- 噪声占总观测方差的比 (仅当窗口内无趋势时有意义)
    """
    if x is None or len(x) < 4:
        return None
    d = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    v_x, v_d = var(x), var(d)
    sd_n = math.sqrt(v_d / 2.0)
    # 局部线性趋势的稳健估计：相邻三点中点的二阶差分
    R = (sd_n ** 2) / v_x if v_x > 0 else float("nan")
    r = rho1(d)
    n = len(d)
    se = math.sqrt(0.75 / n)  # H0: MA(1) theta=-1 时 rho1 的渐近标准误
    z = (r + 0.5) / se
    s = scale if scale else 1.0
    print("  %-24s n=%2d  sd=%7s  sd_noise=%7s  R=%5.2f  rho1(d)=%+5.2f  z=%+5.2f"
          % (tag, len(x),
             ("%.4f" % (math.sqrt(v_x) * s)) + unit,
             ("%.4f" % (sd_n * s)) + unit, R, r, z))
    return dict(n=len(x), sd=math.sqrt(v_x), sd_noise=sd_n, R=R, rho1=r, z=z, se=se)


def predicted_sigma_map():
    """用 per-class bootstrap 方差合成单模型 mAP 预测噪声 sd"""
    if not os.path.exists(PC_JSON):
        return None, None
    d = json.load(open(PC_JSON))
    pc = d.get("per_class") or {}
    sds = []
    for k, v in pc.items():
        if isinstance(v, dict):
            s = v.get("boot_sd") or v.get("sd")
        else:
            s = None
        if isinstance(s, (int, float)):
            sds.append(s)
    if not sds:
        return None, d
    K = 12.0
    return math.sqrt(sum((s / K) ** 2 for s in sds)), d


def analyze(tag, name, i, j, rows=None):
    """对 rows[i..j] 做同一套统计"""
    if rows is None:
        rows = load(name)
    if not rows:
        print("%s: 无 log" % tag)
        return None
    by_ep = {r["epoch"]: r for r in rows}
    w = [by_ep[e] for e in range(i, j + 1) if e in by_ep]
    if len(w) < 4:
        print("%s: 窗口点不足" % tag)
        return None
    ap = [r["ap"] for r in w if r["ap"] is not None]
    lo = [r["loss"] for r in w if r["loss"] is not None]
    lo2 = lo[1:] if len(lo) > 4 else lo
    print("\n%s   ep%d-ep%d (%d 点, lr=%s)"
          % (tag, w[0]["epoch"], w[-1]["epoch"], len(w),
             ("%.4e" % w[0]["lr"]) if w[0]["lr"] else "?"))
    s_ap = report("mAP@50-95 (val200)", ap, scale=100.0, unit="pt")
    s_lo = report("train_loss", lo2, scale=1.0, unit="")
    dap = [ap[k + 1] - ap[k] for k in range(len(ap) - 1)]
    dlo = [lo[k + 1] - lo[k] for k in range(len(lo) - 1)]
    print("  corr(dmAP, dLoss) = %+.2f" % pearson(dap, dlo))
    return s_ap, (max(ap) - min(ap)) * 100, w


def analyze_auto(tag, name):
    rows = load(name)
    if not rows:
        print("%s: 无 log" % tag)
        return None
    i, j = flat_window(rows)
    return analyze(tag, name, rows[i]["epoch"], rows[j]["epoch"], rows)


def ema_table(tag, iters_per_epoch, total_epochs, epochs=(3, 10, 18, 20, 22, 25, 30, 35, 40, 49, 50)):
    """DEIM 的 ModelEMA: decay_fn(x)=0.9999*(1-exp(-x/1000)), x=迭代数, 单位「迭代」"""
    print("\n  %s  (iter/epoch=%d, 共 %d iters)" % (tag, iters_per_epoch,
                                                   iters_per_epoch * total_epochs))
    print("    %6s %9s %9s %8s %11s" % ("epoch", "iters", "decay", "tau(iter)", "tau(epoch)"))
    for e in epochs:
        if e > total_epochs:
            continue
        x = e * iters_per_epoch
        d = 0.9999 * (1 - math.exp(-x / 1000.0))
        tau = 1.0 / (1 - d)
        print("    %6d %9d %9.5f %8.1f %11.2f"
              % (e, x, d, tau, tau / iters_per_epoch))


def main():
    print("=" * 104)
    print("逐 epoch 起伏的性质：白噪声检验")
    print("=" * 104)
    print("sd_noise = sqrt(Var(d)/2)，逐点不可解释抖动（趋势稳健）;  mAP 以 pt 计")
    print("R        = 噪声方差 / 观测方差 (1.00 = 全是噪声；窗口内需无趋势才有意义)")
    print("rho1     = 一阶差分 lag-1 自相关 (纯白噪声 = -0.50；平滑轨迹 = 0)")
    print("-" * 104)

    summary = []
    for tag, name in STAGES:
        r = analyze_auto(tag, name)
        if r:
            summary.append((tag, r[0]["sd_noise"] * 100, r[1]))
    for name, tag, i, j in JOBS:
        r = analyze(tag, name, i, j)
        if r:
            summary.append((tag, r[0]["sd_noise"] * 100, r[1]))

    print("\n" + "=" * 104)
    pred, raw = predicted_sigma_map()
    if pred:
        print("per-class bootstrap 合成的「单个固定模型 + 换一批评测图」mAP sd = %.2f pt" % (pred * 100))
    print("汇总（趋势稳健的逐点抖动 / 窗口实测极差）:")
    for tag, sdn, rng in summary:
        print("   %-38s sd_noise = %5.2f pt   极差 = %6.2f pt" % (tag, sdn, rng))
    print("=" * 104)
    print("★ 为什么参照基线的后段看起来「不波」？—— EMA 时长随迭代数增长（warmups=1000 单位是迭代）")
    ema_table("midboost_combo (ViT-S+, bs10, 50ep)", 180, 50)
    ema_table("Stage C         (ViT-L@800, bs12, 40ep)", 150, 40)
    print("=" * 104)


if __name__ == "__main__":
    main()
