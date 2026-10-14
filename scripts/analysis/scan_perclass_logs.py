#!/usr/bin/env python
"""扫描 logs/ 里所有含 per-category AP 的评测日志, 按 (模型, 条件, 消融方式) 归类。

目的: 回答"IR/热成像到底给哪些类别带来了 AP 提升"。

评测脚本会在日志里打印
    per-category (area=all, maxDets=100):
      person   AP=... AP50=...
我们需要同时知道这次评测**用的是哪个配置 / 哪个 checkpoint / 有没有做消融**,
否则无法配对成对照。消融信息通常在 cfg 行里的 'config': 字段或环境变量里。
"""
import glob
import os
import re

LOGDIR = "/home/constant/CodeBuddy/Dev/AIC/logs"
CATS = ["person", "boat", "animal", "seat", "sign", "bicycle", "car",
        "ball", "light", "garbage can", "uav", "tricycle"]


def parse(path):
    txt = open(path, errors="ignore").read()
    if "per-category" not in txt:
        return None
    m = re.search(r"per-category.*?\n((?:\s+\S.*\n){3,20})", txt)
    if not m:
        return None
    aps = {}
    for ln in m.group(1).splitlines():
        mm = re.match(r"\s+(.+?)\s+AP=([0-9.]+)\s+AP50=([0-9.]+)", ln)
        if mm:
            aps[mm.group(1).strip()] = (float(mm.group(2)), float(mm.group(3)))
    if not aps:
        return None
    # 整体 AP
    mall = re.search(r"Average Precision.*?IoU=0\.50:0\.95.*?area=\s+all.*?maxDets=100\s*\]?\s*=\s*([0-9.]+)", txt)
    # 配置名 / ckpt / 条件
    cfg = re.search(r"'config':\s*'([^']+)'", txt)
    ck = re.search(r"'resume':\s*'([^']+)'", txt)
    tot = re.search(r"Average Precision.*?IoU=0\.50:0\.95.*?area=\s+all.*?maxDets=100\s*\]\s*=\s*([0-9.]+)", txt)
    return dict(path=os.path.basename(path), aps=aps,
                mAP=float(tot.group(1)) if tot else None,
                cfg=cfg.group(1).split("/")[-1] if cfg else "",
                ckpt=ck.group(1).split("/")[-2] if ck else "")


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(LOGDIR, "*.log"))):
        r = parse(p)
        if r:
            rows.append(r)
    print(f"找到 {len(rows)} 个含 per-category 结果的日志\n")
    # 只列 combo / l4u3 / vitl800 相关
    key = ("combo", "l4u3", "vitl800", "ablate", "pureir", "keep_rgb", "deepir")
    sel = [r for r in rows if any(k in r["path"] for k in key)]
    hdr = f"{'log':58s} {'mAP':>7s} " + " ".join(f"{c[:6]:>7s}" for c in CATS)
    print(hdr)
    print("-" * len(hdr))
    for r in sel:
        cells = []
        for c in CATS:
            v = r["aps"].get(c)
            cells.append(f"{v[0]:7.3f}" if v else "      -")
        print(f"{r['path'][:58]:58s} {r['mAP'] if r['mAP'] else 0:7.4f} " + " ".join(cells))
    print()
    print("== 所有含 per-category 的日志 ==")
    for r in rows:
        print(f"  {r['path']:62s} mAP={r['mAP']}  cfg={r['cfg'][:60]}")


if __name__ == "__main__":
    main()
