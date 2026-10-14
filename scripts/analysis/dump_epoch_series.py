#!/usr/bin/env python3
"""从 outputs/<exp>/log.txt 抽取逐 epoch 序列（epoch, lr, loss, mAP, AP50）并打印。

用法: python dump_epoch_series.py <exp_name> [<exp_name> ...]
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "vendor", "DEIMv2", "outputs")


def series(exp):
    p = os.path.join(OUT, exp, "log.txt")
    if not os.path.exists(p):
        return None
    rows = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        b = d.get("test_coco_eval_bbox") or [None, None]
        rows.append(dict(epoch=int(d["epoch"]), lr=float(d["train_lr"]),
                         loss=float(d["train_loss"]),
                         ap=float(b[0]) if b[0] is not None else None,
                         ap50=float(b[1]) if b[1] is not None else None,
                         mal=float(d.get("train_loss_mal", 0.0))))
    return rows


def main():
    for exp in sys.argv[1:]:
        rows = series(exp)
        if not rows:
            print("== %s : 无 log" % exp)
            continue
        print("=" * 92)
        print("%s   (%d 个 epoch 记录, epoch %d..%d)"
              % (exp, len(rows), rows[0]["epoch"], rows[-1]["epoch"]))
        print("%5s %11s %10s %8s %8s %8s" % ("ep", "lr", "loss", "mAP", "AP50", "loss_mal"))
        best = max((r for r in rows if r["ap"] is not None), key=lambda r: r["ap"])
        for r in rows:
            flag = "  <== BEST" if r is best else ""
            print("%5d %11.4e %10.4f %8.4f %8.4f %8.4f%s"
                  % (r["epoch"], r["lr"], r["loss"],
                     r["ap"] if r["ap"] is not None else float("nan"),
                     r["ap50"] if r["ap50"] is not None else float("nan"),
                     r["mal"], flag))
        print("best = %.4f @ ep%d   末轮 = %.4f" % (best["ap"], best["epoch"], rows[-1]["ap"]))


if __name__ == "__main__":
    main()
