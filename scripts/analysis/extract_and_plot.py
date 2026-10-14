import os, re, glob, json
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

OUT_DIR = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs"
WORK_DIR = "/home/constant/CodeBuddy/Dev/AIC/work_dir"
FIG = os.path.join(FIG_DIR, "map_distribution_full.png")
CSV = "/home/constant/CodeBuddy/Dev/AIC/work_dir/scores_extracted.csv"

RE_EVAL_MAP = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95.*?= ([\d.]+)")
RE_EVAL_AP50 = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s.*?= ([\d.]+)")
RE_LOG_MAP = re.compile(r'test_coco_eval_bbox["\']?\s*:\s*\[([0-9.,\s]+)\]')

BRIGHT = ["orig", "night50", "night25", "night12", "night05", "night01", "night005",
          "night002", "night001", "night0005"]
CURVE_BASES = ["ufall(5ch)", "RGB(3ch)", "cssa", "envgate"]


def base_of(fn):
    s = fn.lower()
    if "rgb_base" in s:
        return "RGB(3ch)"
    if "csd_real" in s:
        return "csd_real"
    if "ufall" in s:
        return "ufall(5ch)"
    if "cssa" in s:
        return "cssa"
    if "envgate" in s:
        return "envgate"
    if "llvip" in s:
        return "llvip"
    if "flir" in s:
        return "flir"
    return os.path.basename(fn).split("_")[0]


def valset_of(fn):
    s = fn.lower()
    for k in ("abbothblk", "abirblk", "abdepblk", "abrgbblk"):
        if k in s:
            return "modal_blk"
    for k in ("night005", "night01", "night05", "night12", "night25", "night50",
              "night002", "night001", "night0005"):
        if k in s:
            return k
    if "orig" in s:
        return "orig"
    if "real" in s:
        return "orig"
    if "smoke" in s:  # ufall smoke == night005
        return "night005"
    if "llvip" in s or "flir" in s:
        return "cross"
    return "other"


def last_match(pat, text):
    m = pat.findall(text)
    return float(m[-1]) if m else None


records = []  # (source, base, valset, mAP, AP50)

# 1) training logs -> AIC-day validation mAP
for lp in glob.glob(os.path.join(OUT_DIR, "*", "log.txt")):
    name = os.path.basename(os.path.dirname(lp))
    try:
        txt = open(lp, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    ms = RE_LOG_MAP.findall(txt)
    if not ms:
        continue
    nums = [float(x) for x in ms[-1].split(",")]
    if len(nums) >= 2:
        records.append(("train-log", name, "AIC-day", nums[0], nums[1]))

# 2) eval logs -> brightness / cross / modal-ablation
for lp in glob.glob(os.path.join(WORK_DIR, "*_eval*.log")) + \
        glob.glob(os.path.join(WORK_DIR, "llvip_eval_result.log")) + \
        glob.glob(os.path.join(WORK_DIR, "night005_eval_launch.log")):
    fn = os.path.basename(lp)
    try:
        txt = open(lp, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    m = last_match(RE_EVAL_MAP, txt)
    a50 = last_match(RE_EVAL_AP50, txt)
    if m is None:
        continue
    src = "eval-best" if os.path.basename(lp).startswith("evalbest") else "eval-log"
    records.append((src, base_of(fn), valset_of(fn), m, a50 or 0.0))

assert records, "no records"
print(f"total records: {len(records)}  "
      f"(train-log={sum(1 for r in records if r[0]=='train-log')}, "
      f"eval-log={sum(1 for r in records if r[0]=='eval-log')})")

# save csv
with open(CSV, "w") as f:
    f.write("source,base,valset,mAP,AP50\n")
    for r in records:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f}\n")

allmap = [r[3] for r in records]
train_map = [r[3] for r in records if r[0] == "train-log"]
eval_map = [r[3] for r in records if r[0] == "eval-log"]

# brightness curves (standard eval batches only, exclude eval-best re-evals)
bright = defaultdict(lambda: defaultdict(list))
for r in records:
    if r[0] == "eval-log" and r[1] in CURVE_BASES and r[2] in BRIGHT:
        bright[r[1]][r[2]].append(r[3])

print("--- eval-log records used in brightness curves ---")
for base in CURVE_BASES:
    if base in bright:
        for v in BRIGHT:
            if bright[base][v]:
                print(f"  {base:12} {v:9} -> {[round(x,3) for x in bright[base][v]]}")

x = np.arange(len(BRIGHT))
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

# A) full histogram
axes[0].hist([train_map, eval_map], bins=14,
             label=[f"train-log AIC-day (n={len(train_map)})",
                    f"eval-log brightness/cross (n={len(eval_map)})"],
             color=["steelblue", "darkorange"], edgecolor="white")
mm, md = np.mean(allmap), np.median(allmap)
axes[0].axvline(mm, color="red", ls="--", label=f"mean={mm:.3f}")
axes[0].axvline(md, color="green", ls="--", label=f"median={md:.3f}")
axes[0].set_title(f"Full mAP distribution (n={len(allmap)})")
axes[0].set_xlabel("mAP @ IoU=0.50:0.95")
axes[0].set_ylabel("count")
axes[0].legend(fontsize=8)

# B) brightness degradation curves
for base in CURVE_BASES:
    if base not in bright:
        continue
    ys = [np.mean(bright[base][v]) if bright[base][v] else np.nan for v in BRIGHT]
    axes[1].plot(x, ys, marker="o", label=f"{base} (n={sum(len(v) for v in bright[base].values())})")
axes[1].set_xticks(x)
axes[1].set_xticklabels(BRIGHT, rotation=30)
axes[1].set_title("Brightness degradation: mAP vs val brightness\n(left=bright/orig -> right=near-dark)")
axes[1].set_ylabel("mAP")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# C) boxplot by brightness valset
groups = defaultdict(list)
for r in records:
    if r[2] in BRIGHT:
        groups[r[2]].append(r[3])
order = [v for v in BRIGHT if groups[v]]
data = [groups[v] for v in order]
bp = axes[2].boxplot(data, tick_labels=[f"{v}\n(n={len(groups[v])})" for v in order],
                     patch_artist=True, showmeans=True, showfliers=True)
for patch in bp["boxes"]:
    patch.set_facecolor("lightgreen")
for m in bp["means"]:
    m.set_markerfacecolor("red")
    m.set_markeredgecolor("red")
axes[2].set_title("mAP distribution by validation brightness")
axes[2].set_ylabel("mAP")
plt.setp(axes[2].get_xticklabels(), rotation=30, ha="right", fontsize=8)

plt.tight_layout()
plt.savefig(FIG, dpi=120)
print(f"saved {FIG}")
print("full mean={:.3f} median={:.3f}".format(mm, md))
print("brightness curve (base -> orig..night005):")
for base in CURVE_BASES:
    if base in bright:
        seq = [f"{v}={np.mean(bright[base][v]):.3f}" if bright[base][v] else f"{v}=NA"
               for v in BRIGHT]
        print("  ", base, "  ".join(seq))
