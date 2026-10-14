#!/usr/bin/env python3
"""Extract per-epoch train_lr (and loss) from a DEIMv2 log.txt for diagnosing lr schedule."""
import json
import sys
import os

LOG_ROOT = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs"


def main():
    name = sys.argv[1]
    p = os.path.join(LOG_ROOT, name, "log.txt")
    if not os.path.exists(p):
        print("no log:", p)
        return
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
        rows.append(o)

    print("### %s  rows=%d" % (name, len(rows)))
    print("%6s %12s %10s %8s %8s" % ("epoch", "lr", "loss", "AP", "AP50"))
    for o in rows:
        ep = o.get("epoch")
        lr = o.get("train_lr")
        loss = o.get("train_loss")
        s = o.get("test_coco_eval_bbox")
        ap = ap50 = None
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            ap, ap50 = s[0], s[1]
        print(
            "%6s %12s %10s %8s %8s"
            % (
                ep,
                ("%.3e" % lr) if isinstance(lr, (int, float)) else lr,
                ("%.3f" % loss) if isinstance(loss, (int, float)) else loss,
                ("%.4f" % ap) if ap is not None else "-",
                ("%.4f" % ap50) if ap50 is not None else "-",
            )
        )


if __name__ == "__main__":
    main()
