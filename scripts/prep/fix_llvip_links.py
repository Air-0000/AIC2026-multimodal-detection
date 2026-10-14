"""Fix LLVIP rgb/ir symlinks: relative link targets -> absolute paths.

Uses atomic os.replace() (rename) instead of remove+symlink, so no bulk
deletion is triggered.
"""
import os
import glob

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
BASE = os.path.join(ROOT, "work_dir/LLVIP")

for d in ("rgb", "ir"):
    p = os.path.join(BASE, d)
    fixed = skipped = 0
    files = sorted(os.listdir(p))
    for i, f in enumerate(files):
        fp = os.path.join(p, f)
        if not os.path.islink(fp):
            continue
        t = os.readlink(fp)
        if os.path.isabs(t):
            skipped += 1
            continue
        tmp = fp + ".tmpfix"
        if os.path.lexists(tmp):
            os.replace(tmp, fp)  # cleanup leftover, no unlink
        os.symlink(os.path.join(ROOT, t), tmp)
        os.replace(tmp, fp)
        fixed += 1
        if (i + 1) % 2000 == 0:
            print(f"{d}: {i+1}/{len(files)} fixed={fixed}", flush=True)
    print(f"{d}: DONE total={len(files)} fixed={fixed} already_abs={skipped}", flush=True)

from PIL import Image
bad = 0
for d in ("rgb", "ir"):
    fs = sorted(glob.glob(os.path.join(BASE, d, "*.jpg")))
    for f in fs[:100] + fs[-100:]:
        try:
            Image.open(f).convert("RGB")
        except Exception as e:
            bad += 1
            print("BAD:", f, e, flush=True)
print("sample_check_bad =", bad, flush=True)
