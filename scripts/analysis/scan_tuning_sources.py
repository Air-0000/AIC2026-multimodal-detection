#!/usr/bin/env python3
"""Scan all output dirs for the 'tuning' (-t) value recorded at train start.

DEIMv2 dumps the full cfg at launch (train log / log.txt). We extract the
first occurrence of 'tuning' per run to map which runs were warm-started
from another detector vs started from an official init / random heads.
"""
import os, re, json

OUT = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs"
res = {}
for d in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, d)
    if not os.path.isdir(p):
        continue
    val = None
    # search args.yml first, then log.txt
    for fn in ("args.yml", "log.txt"):
        fp = os.path.join(p, fn)
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                txt = f.read(400000)
        except Exception:
            continue
        m = re.search(r"'tuning':\s*('([^']*)'|None|\"([^\"]*)\")", txt)
        if m:
            val = m.group(2) or m.group(3) or "None"
            break
    if val is None:
        # try weights_path presence in config snapshot? fallback
        val = "?"
    res[d] = val

print(json.dumps(res, indent=0, ensure_ascii=False))
