#!/usr/bin/env python3
"""Scan AIC logs/*.log for 'tuning' value to map warm-start sources."""
import os, re, glob

LOGS = "/home/constant/CodeBuddy/Dev/AIC/logs"
pat = re.compile(r"'tuning':\s*('([^']*)'|None|\"([^\"]*)\")")
res = {}
for fp in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
    name = os.path.basename(fp)
    if "eval" in name or "night" in name and "train" not in name:
        # still scan; cheap enough
        pass
    try:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            txt = f.read(600000)
    except Exception:
        continue
    m = pat.search(txt)
    if m:
        val = m.group(2) or m.group(3) or "None"
        res[name] = val

for k, v in sorted(res.items()):
    print(f"{k}\t{v}")
