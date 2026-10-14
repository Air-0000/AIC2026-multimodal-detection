#!/usr/bin/env python3
"""把三阶段（A/B/C）的训练「路径」画到一张图上：逐 10-iter loss + 逐 epoch mAP + lr。

数据来源：
  outputs/<exp>/summary/*tfevents*  -> Loss/total, Lr/pg_0 （det_engine.py:140，每 10 iter 一条）
  outputs/<exp>/log.txt             -> 逐 epoch train_loss / test_coco_eval_bbox

注意：逐 iter 的 loss 跳动很大（26~43），因为 deim_criterion 用 batch 内 GT 框总数
(num_boxes) 归一化，每个 batch 的框数不同。原始点只用来显示「抖动带宽」，趋势看滑动平均。

运行环境：需要 tensorboard，用 deimv2 conda 解释器。
用法:
  python scripts/analysis/plot_train_path.py --out outputs/train_path.png \
      --stage A:vitl800_scratch_stageA_rgb B:vitl800_scratch_stageB_mod C:vitl800_scratch_stageC_fuse
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "vendor", "DEIMv2", "outputs")


def read_tb(exp, tag):
    from tensorboard.backend.event_processing import event_accumulator
    d = os.path.join(OUT, exp, "summary")
    if not glob.glob(os.path.join(d, "events.out.tfevents.*")):
        return []
    acc = event_accumulator.EventAccumulator(d, size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return []
    pts = [(e.step, e.value) for e in acc.Scalars(tag)]
    # 事件目录里可能有多次重启的事件文件，同 step 会有重复 -> 后写覆盖
    dd = {}
    for s, v in pts:
        dd[s] = v
    return sorted(dd.items())


def read_log(exp):
    p = os.path.join(OUT, exp, "log.txt")
    rows = []
    if not os.path.exists(p):
        return rows
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        b = d.get("test_coco_eval_bbox") or [None, None]
        rows.append(dict(epoch=int(d["epoch"]), loss=float(d["train_loss"]),
                         lr=float(d["train_lr"]), ap=b[0], ap50=b[1]))
    return rows


def ma(xs, w):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i + 1 - lo))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", required=True, help="TAG:exp_name")
    ap.add_argument("--out", default="outputs/train_path.png")
    ap.add_argument("--ma", type=int, default=12, help="滑动平均窗口（以 10-iter 点为单位）")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    stages = []
    for s in a.stage:
        tag, exp = s.split(":", 1)
        tb = read_tb(exp, "Loss/total")
        tbr = read_tb(exp, "Lr/pg_0")
        log = read_log(exp)
        stages.append(dict(tag=tag, exp=exp, tb=tb, tbr=dict(tbr), log=log))
        print("%s  %-40s TB点=%4d  epoch=%2d" % (tag, exp, len(tb), len(log)))

    # ---- 拼接全局 step / global epoch ----
    off = 0
    ep_offset = 0
    fig, ax = plt.subplots(3, 1, figsize=(12.5, 10.5), sharex=False)
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    bounds, bounds_ep = [], []
    for i, st in enumerate(stages):
        c = colors[i % len(colors)]
        tb = st["tb"]
        if tb:
            xs = [s + off for s, _ in tb]
            ys = [v for _, v in tb]
            ax[0].plot(xs, ys, "-", color=c, lw=0.5, alpha=0.30)
            ax[0].plot(xs, ma(ys, a.ma), "-", color=c, lw=2.0, label="%s (MA%d)" % (st["tag"], a.ma))
            span = tb[-1][0] - tb[0][0]
        else:
            span = 0
        n_ep = (max(r["epoch"] for r in st["log"]) + 1) if st["log"] else 0
        bounds.append((off, span, st["tag"]))
        bounds_ep.append((ep_offset, n_ep, st["tag"]))
        off += span
        ep_offset += n_ep
    for i, (o, sp, tag) in enumerate(bounds[:-1]):
        ax[0].axvline(o + sp, color="k", ls=":", lw=1)
    for i, (o, ne, tag) in enumerate(bounds_ep[:-1]):
        ax[1].axvline(o + ne, color="k", ls=":", lw=1)
    for i, (o, sp, tag) in enumerate(bounds):
        ax[0].text(o + sp * 0.5, ax[0].get_ylim()[1], tag, ha="center", va="top", fontsize=11, color="k")
    for i, (o, ne, tag) in enumerate(bounds_ep):
        ax[1].text(o + ne * 0.5, 0.5275, tag, ha="center", va="top", fontsize=11, color="k")
    ax[0].set_xlabel("global step (10-iter metric points)")
    ax[0].set_ylabel("train_loss (total)")
    ax[0].set_title("A→B→C 训练路径：逐 10-iter loss（细线=原始，粗线=滑动平均）")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].grid(alpha=0.25)

    ep_off2 = 0
    for i, st in enumerate(stages):
        c = colors[i % len(colors)]
        e = [ep_off2 + r["epoch"] for r in st["log"]]
        v = [r["ap"] for r in st["log"]]
        ax[1].plot(e, v, "-o", color=c, ms=3.5, lw=1.4, label=st["tag"])
        if v and any(x is not None for x in v):
            bi = max(range(len(v)), key=lambda k: (v[k] if v[k] is not None else -1))
            ax[1].annotate("best %.4f@ep%d" % (v[bi], st["log"][bi]["epoch"]), (e[bi], v[bi]),
                           textcoords="offset points", xytext=(6, 6), fontsize=8, color=c)
        ep_off2 += (max(r["epoch"] for r in st["log"]) + 1) if st["log"] else 0
    ax[1].set_xlabel("global epoch")
    ax[1].set_ylabel("val200 mAP@50-95")
    ax[1].set_ylim(0.44, 0.53)
    ax[1].set_title("val mAP（放大到 0.44–0.53；Stage A 前 4 个 epoch 为 0.0001/0.0250/0.0971/0.1806，超出下界）\n"
                    "val200 的 σ≈2.1pt ⇒ 逐点差异多为测量噪声，别按单点挑模型")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].grid(alpha=0.25)

    for i, st in enumerate(stages):
        c = colors[i % len(colors)]
        e = [r["epoch"] for r in st["log"]]
        v = [r["lr"] for r in st["log"]]
        ax[2].semilogy(e, v, "-o", color=c, ms=3, lw=1.4, label="%s (log.txt lr)" % st["tag"])
        if st["tbr"]:
            xs = [s for s, _ in sorted(st["tbr"].items())]
            ys = [st["tbr"][s] for s in xs]
            ax[2].semilogy(xs, ys, "-", color=c, lw=0.6, alpha=0.45)
    ax[2].set_xlabel("epoch（各阶段各自计数）")
    ax[2].set_ylabel("lr (param_groups[0])")
    ax[2].set_title("lr 调度：flatcosine = 爬升 → 平台 → 余弦衰减 → 触底")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.25, which="both")

    fig.tight_layout()
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("写出 %s" % out)


if __name__ == "__main__":
    main()
