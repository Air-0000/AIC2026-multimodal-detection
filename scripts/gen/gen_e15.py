"""E15: DepthPrior -- make depth a PRIOR, not a fused feature.

Single-variable on top of ufall (the 5ch baseline that loses to RGB-only):
  * DLS + DLW at train: the box/cls losses are weighted by the GT depth, so
    far/small targets (where RGB is unreliable at night and depth is the dominant
    geometric cue) contribute more. Zero backbone change.
  * DCT at infer: far predictions get a score boost so they clear the global
    confidence threshold at night (auto-off when depth is blacked, i.e. ablk).

Generated:
    deimv2_..._rnight_ufall_depthrior.yml            (train, 24 ep quick run)
    deimv2_..._rnight_ufall_depthrior_val_*.yml       (all ufall levels + ablk)
    deimv2_..._rnight_ufall_depthrior_bdd_*.yml
Warm-started from ufall best_stg2 in run_e15.sh.
"""
import os
import glob
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"

SRC_TRAIN = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall"
DST_TRAIN = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_depthrior"
OLD_OUT = "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall"
NEW_OUT = "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_depthrior"

# ---------------------------------------------------------------- train config
t = open(os.path.join(CFG, f"{SRC_TRAIN}.yml")).read()
assert t.count(OLD_OUT + "\n") == 1, "train output_dir not unique"
t = t.replace(OLD_OUT + "\n", NEW_OUT + "\n")

anchor = "  matcher:\n    matcher_change_epoch: 50\n"
assert t.count(anchor) == 1, "criterion matcher anchor not unique"
t = t.replace(anchor, anchor +
    "  use_depth_prior: True\n"
    "  depth_bins: [0.33, 0.66]\n"
    "  depth_weights: [1.0, 2.0, 4.0]\n"
    "  depth_invert: True\n")

assert "PostProcessor:" not in t
t += ("\nPostProcessor:\n"
      "  depth_cond_enable: True\n"
      "  depth_cond_far: 0.5\n"
      "  depth_cond_boost: 1.5\n"
      "  depth_invert: True\n")

# short run to confirm the method fast
t = t.replace("epoches: 68\n", "epoches: 24\n")
t = t.replace("flat_epoch: 34\n", "flat_epoch: 12\n")
t.replace("no_aug_epoch: 8\n", "no_aug_epoch: 4\n")

open(os.path.join(CFG, f"{DST_TRAIN}.yml"), "w").write(t)
c = yaml.safe_load(open(os.path.join(CFG, f"{DST_TRAIN}.yml")))
assert c["DEIMCriterion"]["use_depth_prior"] is True, c["DEIMCriterion"]
assert c["output_dir"] == NEW_OUT, c["output_dir"]
assert c["epoches"] == 24, c["epoches"]
print("wrote", DST_TRAIN + ".yml", " epoches=", c["epoches"])

# ---------------------------------------------------------------- val configs
POST = ("PostProcessor:\n"
        "  depth_cond_enable: True\n"
        "  depth_cond_far: 0.5\n"
        "  depth_cond_boost: 1.5\n"
        "  depth_invert: True\n")

val_files = (sorted(glob.glob(os.path.join(CFG, f"{SRC_TRAIN}_val_*.yml")))
             + sorted(glob.glob(os.path.join(CFG, f"{SRC_TRAIN}_bdd_*.yml"))))
n = 0
for src in val_files:
    txt = open(src).read()
    base = os.path.basename(src)
    assert txt.count(OLD_OUT) == 1, f"{base}: output_dir not unique"
    txt = txt.replace(OLD_OUT, NEW_OUT)          # unique eval output_dir
    assert "PostProcessor:" not in txt, base
    txt = txt + "\n" + POST
    dst = os.path.join(CFG, base.replace("ufall_val_", "ufall_depthrior_val_")
                                .replace("ufall_bdd_", "ufall_depthrior_bdd_"))
    open(dst, "w").write(txt)
    n += 1
print(f"wrote {n} val/bdd configs (depth_cond_enable=True)")
print("OK")
