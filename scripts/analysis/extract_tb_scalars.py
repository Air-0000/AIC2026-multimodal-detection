#!/usr/bin/env python3
"""从 outputs/<exp>/summary/*tfevents* 抽取逐 10-iteration 的训练标量。

框架侧：engine/solver/det_engine.py:140  `if global_step % 10 == 0`
  -> Loss/total, Lr/pg_j (j=0..N), Loss/<分量>
所以一条 40ep 的 run 会得到 ~600 个点（150 iter/ep），远比 log.txt 的 40 点密。

运行环境：需要 tensorboard（用 deimv2 conda 解释器，不要用沙箱 venv）。

用法: python extract_tb_scalars.py <exp_name> [--tag Loss/total] [--json out.json]
"""
import argparse
import glob
import json
import os
import sys

from tensorboard.backend.event_processing import event_accumulator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "vendor", "DEIMv2", "outputs")


def load_scalars(exp):
    d = os.path.join(OUT, exp, "summary")
    files = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
    if not files:
        return {}, files
    acc = event_accumulator.EventAccumulator(
        d, size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    data = {}
    for t in tags:
        data[t] = [(e.step, e.value) for e in acc.Scalars(t)]
    return data, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--json", default=None)
    ap.add_argument("--tags", nargs="*", default=["Loss/total"])
    a = ap.parse_args()

    data, files = load_scalars(a.exp)
    print("tfevents 文件 %d 个" % len(files))
    for f in files:
        print("   %s  %.1f KB" % (os.path.basename(f), os.path.getsize(f) / 1024))
    if not data:
        print("无标量")
        return
    print("\n可用 tag（前 20）: %s" % sorted(data.keys())[:20])
    for t in a.tags:
        if t not in data:
            print("\n[%s] 不存在" % t)
            continue
        s = sorted(data[t])
        print("\n[%s] n=%d  step %d..%d" % (t, len(s), s[0][0], s[-1][0]))
        print("  first 5: %s" % [(k, round(v, 5)) for k, v in s[:5]])
        print("  last  5: %s" % [(k, round(v, 5)) for k, v in s[-5:]])

    if a.json:
        out = {t: sorted(v) for t, v in data.items() if t in a.tags}
        with open(a.json, "w") as fh:
            json.dump(out, fh)
        print("\n写出 %s" % a.json)


if __name__ == "__main__":
    main()
