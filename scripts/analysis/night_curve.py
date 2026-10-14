"""Night-robustness curve report.

Objective metric: mean AP over the four RGB-light levels
    dark50 (0.50 dusk) / dark25 (0.25 twilight) / dark12 (0.12 night) / dark05 (0.05 deep night)
evaluated on AIC val with IR/Depth untouched. Higher is better; the four-level
mean is the single number to maximise.

Reads every <tag>_<level>.log under the given dirs (default /tmp/night_robust,
/tmp/llvip640 is a single-level real-night set and is reported separately).
"""
import os
import re
import sys
import glob

LEVELS = ["dark50", "dark25", "dark12", "dark05"]
PAT = re.compile(r"IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)")
AP50 = re.compile(r"IoU=0\.50\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)")


def read_ap(path):
    try:
        s = open(path, errors="ignore").read()
    except Exception:
        return None, None
    a = PAT.search(s)
    b = AP50.search(s)
    return (float(a.group(1)) if a else None), (float(b.group(1)) if b else None)


def collect(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        name = os.path.basename(f)[:-4]
        for lv in LEVELS:
            if name.endswith("_" + lv):
                tag = name[: -len(lv) - 1]
                rows.setdefault(tag, {})[lv] = read_ap(f)[0]
    return rows


def main():
    dirs = sys.argv[1:] or ["/tmp/night_robust"]
    for d in dirs:
        rows = collect(d)
        if not rows:
            continue
        print("=" * 78)
        print(f"night-robustness curve  ({d})")
        print("=" * 78)
        print(f"{'model':<16}{'dark50':>9}{'dark25':>9}{'dark12':>9}{'dark05':>9}{'MEAN':>9}")
        print("-" * 78)
        scored = []
        for tag, m in rows.items():
            vals = [m.get(lv) for lv in LEVELS]
            got = [v for v in vals if v is not None]
            mean = sum(got) / len(got) if got else float("nan")
            scored.append((mean, tag))
            print(f"{tag:<16}" + "".join(
                (f"{v:>9.3f}" if v is not None else f"{'-':>9}") for v in vals)
                + f"{mean:>9.4f}")
        print("-" * 78)
        for mean, tag in sorted(scored, reverse=True):
            print(f"  {tag:<16} mean={mean:.4f}")
        print()


if __name__ == "__main__":
    main()
