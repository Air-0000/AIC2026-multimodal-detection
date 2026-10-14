"""Print the day->night brightness curve for the midboost family from MODEL_RECORDS.md.

MODEL_RECORDS.md is the single source of truth for experiment results; this just
pivots its per-bucket table (log | bucket | AP | AP50) into a model x bucket grid
so the "which model is best OVERALL, not just at day" question can be answered.
"""
import collections
import os
import re
import sys

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
ORDER = ["orig", "day", "night50", "night25", "night12", "night01", "night005"]


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else "midboost"
    rows = collections.defaultdict(dict)
    pat = re.compile(
        r"\|\s*`?([^`|]*%s[^`|]*?)`?\s*\|\s*([A-Za-z0-9]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|" % filt)
    for line in open(os.path.join(ROOT, "MODEL_RECORDS.md")):
        m = pat.search(line)
        if not m:
            continue
        log, bucket, ap, ap50 = m.groups()
        name = re.sub(r"_eval_.*$", "", log)
        name = re.sub(r"_(day|night\d+|nightify)$", "", name)
        rows[name][bucket] = (float(ap), float(ap50))

    buckets = [b for b in ORDER if any(b in rows[n] for n in rows)]
    print(f"{'model':40s}" + "".join(f"{b:>15s}" for b in buckets) + f"{'MEAN':>15s}")
    for n in sorted(rows):
        vals = [rows[n].get(b) for b in buckets]
        line = f"{n:40s}"
        for v in vals:
            line += f"{(f'{v[0]:.4f}/{v[1]:.3f}' if v else '-'):>15s}"
        have = [v[0] for v in vals if v]
        line += f"{(sum(have)/len(have) if have else 0):>15.4f}"
        print(line)
    print("\n(format: AP / AP50)")


if __name__ == "__main__":
    main()
