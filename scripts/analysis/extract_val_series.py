#!/usr/bin/env python3
"""Extract the per-epoch val mAP series from a DEIMv2 training log.txt.

Used to check what a recorded MODEL_RECORDS number actually corresponds to
(which epoch, and whether best_stg1 really holds the best value).
"""
import json
import os
import sys

ROOT = "/home/constant/CodeBuddy/Dev/AIC"


def series(path):
    rows = []
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        s = o.get("test_coco_eval_bbox")
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            rows.append((o.get("epoch"), s[0], s[1]))
    return rows


if __name__ == "__main__":
    for name in sys.argv[1:]:
        p = name if os.path.isabs(name) else os.path.join(ROOT, "vendor/DEIMv2/outputs", name, "log.txt")
        rows = series(p)
        print(f"### {name}   val_rows={len(rows)}")
        for ep, ap, ap50 in rows:
            print(f"   epoch={ep}  AP={ap:.4f}  AP50={ap50:.4f}")
        if rows:
            b = max(rows, key=lambda r: r[1])
            print(f"   BEST: epoch={b[0]} AP={b[1]:.4f} AP50={b[2]:.4f}")
        print()
