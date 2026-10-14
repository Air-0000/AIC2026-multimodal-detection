"""Report both night protocols side by side.

  darkXX  -- the ORIGINAL protocol: RGB x constant. Gameable: a per-image
             instance normalisation is mathematically invariant to it.
  nightXX -- prep_night_real.py: non-uniform lamp lighting + Poisson-Gaussian
             noise + colour shift + attenuation. Cannot be gamed that way.

Usage:  python night_curve2.py [dir_or_dirs...]     (default /tmp/night_robust /tmp/night_real)
"""
import os
import re
import sys
import glob

PAT = re.compile(r"IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)")
PROTO = [("dark", ["dark50", "dark25", "dark12", "dark05"]),
         ("night", ["night50", "night25", "night12", "night05"])]


def read_ap(path):
    try:
        s = open(path, errors="ignore").read()
    except Exception:
        return None
    m = PAT.search(s)
    return float(m.group(1)) if m else None


def collect(dirs, levels):
    """tag -> {level: ap}, scanning every dir."""
    rows = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.log")):
            name = os.path.basename(f)[:-4]
            for lv in levels:
                if name.endswith("_" + lv):
                    tag = name[: -len(lv) - 1]
                    if rows.get(tag, {}).get(lv) is None:
                        rows.setdefault(tag, {})[lv] = read_ap(f)
    return rows


def table(title, rows, levels):
    if not rows:
        return {}
    print("=" * 74)
    print(title)
    print("=" * 74)
    hdr = f"{'model':<12}" + "".join(f"{lv.replace('night','n').replace('dark','d'):>9}" for lv in levels)
    print(hdr + f"{'MEAN':>9}")
    print("-" * 74)
    means = {}
    for tag in sorted(rows):
        vals = [rows[tag].get(lv) for lv in levels]
        got = [v for v in vals if v is not None]
        mean = sum(got) / len(got) if got else float("nan")
        means[tag] = mean
        print(f"{tag:<12}" + "".join((f"{v:>9.3f}" if v is not None else f"{'-':>9}") for v in vals)
              + f"{mean:>9.4f}")
    print("-" * 74)
    for tag, m in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"  {tag:<12} MEAN={m:.4f}")
    print()
    return means


def main():
    dirs = sys.argv[1:] or ["/tmp/night_robust", "/tmp/night_real"]
    out = {}
    for name, levels in PROTO:
        rows = collect(dirs, levels)
        means = table(f"{name} protocol  ({' / '.join(levels)})", rows, levels)
        if name == "night":
            table(f"{name} protocol ranked", rows, levels)
        out[name] = means
    if out.get("dark") and out.get("night"):
        print("=" * 74)
        print("protocol gap: models that win on the OLD protocol but lose on the realistic one")
        print("=" * 74)
        common = set(out["dark"]) & set(out["night"])
        for tag in sorted(common, key=lambda t: -(out["dark"][t] - out["night"][t])):
            print(f"  {tag:<12} dark={out['dark'][tag]:.4f}  night={out['night'][tag]:.4f}  "
                  f"delta={out['dark'][tag] - out['night'][tag]:+.4f}")


if __name__ == "__main__":
    main()
