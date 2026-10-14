#!/usr/bin/env python3
"""探针：确认 outputs/<exp>/log.txt 的行粒度（逐 iter / 逐 epoch），并抽取 1D 训练路径。

用法: python probe_log_granularity.py <exp_name> [<exp_name> ...]
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "vendor", "DEIMv2", "outputs")


def probe(exp):
    p = os.path.join(OUT, exp, "log.txt")
    if not os.path.exists(p):
        print("%-42s 无 log.txt" % exp)
        return
    lines = [l for l in open(p).read().strip().split("\n") if l.strip()]
    print("=" * 100)
    print("%s   %d 行   %.1f KB" % (exp, len(lines), os.path.getsize(p) / 1024))
    idx = [0, 1, 2, len(lines) - 1]
    for i in idx:
        try:
            d = json.loads(lines[i])
        except Exception:
            print("  [%d] 非 JSON: %s" % (i, lines[i][:100]))
            continue
        keys = ["epoch", "iter", "start_epoch", "train_lr", "train_loss",
                "train_loss_mal", "train_loss_cls", "test_coco_eval_bbox"]
        got = {k: d[k] for k in keys if k in d}
        if "test_coco_eval_bbox" in got:
            v = got["test_coco_eval_bbox"]
            got["test_coco_eval_bbox"] = "[mAP=%.4f AP50=%.4f]" % (v[0], v[1]) if isinstance(v, list) else v
        print("  [%d] %s" % (i, got))
    print("  全部键: %s" % sorted(json.loads(lines[-1]).keys()))


if __name__ == "__main__":
    for e in sys.argv[1:]:
        probe(e)
