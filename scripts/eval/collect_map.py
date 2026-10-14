"""Collect best mAP from every experiment we have run (DEIMv2 outputs + /tmp logs)."""
import os
import re
import glob

OUT = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs"
PAT = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95 \| area=\s*all[^\]]*\]\s*=\s*([0-9.]+)")
AP50 = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s+\| area=\s*all[^\]]*\]\s*=\s*([0-9.]+)")
BEST = re.compile(r"best_stat: \{'epoch': (\d+), 'coco_eval_bbox': ([0-9.]+)\}")


def scan_file(path):
    try:
        s = open(path, errors="ignore").read()
    except Exception:
        return None, None
    aps = [float(x) for x in PAT.findall(s)]
    bs = BEST.findall(s)
    if bs:
        e, v = max(bs, key=lambda t: float(t[1]))
        return float(v), int(e)
    return (max(aps) if aps else None), None


print(f"{'experiment':<52} {'best mAP':>9} {'ep':>5}")
print("-" * 70)
rows = []
for d in sorted(glob.glob(os.path.join(OUT, "*"))):
    if not os.path.isdir(d):
        continue
    v, e = scan_file(os.path.join(d, "log.txt"))
    name = os.path.basename(d)
    if v is None:
        print(f"{name:<52} {'-':>9} {'-':>5}")
    else:
        print(f"{name:<52} {v:>9.4f} {str(e) if e is not None else '-':>5}")
    rows.append((name, v))

print()
print("=== /tmp 下的评测日志（LLVIP / AIC val） ===")
for f in sorted(glob.glob("/tmp/*.log")):
    v, e = scan_file(f)
    if v is None:
        continue
    aps = PAT.findall(open(f, errors="ignore").read())
    print(f"{os.path.basename(f):<24} best={v:.4f}  (AP 次数={len(aps)})")
