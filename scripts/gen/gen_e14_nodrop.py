"""E14: eliminate modality dropout, so the fusion cannot learn to ignore aux.

Diagnosis chain that motivates it
---------------------------------
1. The multimodal line loses to RGB-only on every axis, even after E11 gave it
   the same treatment as the RGB winner (full 22.6M fine-tune + input_norm):
   ufall 0.4347/0.4055 vs inorm 0.5130/0.4567.
2. Per-class (AIC val, night05): every class with enough boxes is LOST --
   person 0.376->0.330, animal 0.429->0.383, car 0.491->0.443, ... The only
   "wins" are tricycle (2 boxes) and uav (17 boxes), i.e. noise.
3. So the aux signal is not being used. Is it even there? A direct measurement of
   the heat residual inside GT boxes (/tmp/heat_signal_check.py) says yes:

       class      boxes   heat_in  heat_bg  in/bg
       person       567     57.66    34.99   1.65
       animal       350     43.02    27.47   1.57
       car          136     39.47    29.24   1.35
       seat          73     35.94    28.63   1.26
       ALL         1481     45.66    31.64   1.44     (45% of boxes at SNR>1.5)

   Note the two classes with the strongest thermal signature (person, animal)
   are exactly the two where RGB-only beats multimodal by the widest margin.

So: signal exists, fusion does not extract it, and carrying the dead branches
actively costs accuracy. The most likely culprit is the modality dropout --
`ir_drop_prob: 0.15` / `depth_drop_prob: 0.15` zero the aux 15% of the time, and
with only 1700 training images and a 1.44x aux SNR the optimal policy is to
output ~0 from the fusion and ignore it. E14 removes the dropout entirely and
warm-starts from E11, so "does the fusion use aux when it has no choice" becomes
a single-variable question.

Reading the result:
  E14 person AP rises above inorm's -> dropout was the blocker, fusion is fixable
  E14 stays where ufall was          -> the fusion design, not the schedule, is
                                        the problem, and the line should be dropped.

Generated:
    deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_nodrop.yml
    ..._bdd_{night,daytime}.yml
AIC per-level val configs are reused verbatim from ufall: architecture is
identical (dropout only acts at train time) and they already carry
`input_norm: instance`, which is required for a correct evaluation.
"""
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
SRC = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall"
DST = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_nodrop"

IR = "  ir_drop_prob: 0.15\n"
DEP = "  depth_drop_prob: 0.15\n"

# ---- training config
t = open(os.path.join(CFG, f"{SRC}.yml")).read()
assert t.count(IR) == 1 and t.count(DEP) == 1, "dropout lines not unique"
assert "input_norm: instance" in t, "ufall base must carry input_norm"
assert t.count(f"./outputs/{SRC}\n") == 1, "output_dir not unique"
t = t.replace(IR, "  ir_drop_prob: 0.0\n")
t = t.replace(DEP, "  depth_drop_prob: 0.0\n")
t = t.replace(f"./outputs/{SRC}\n", f"./outputs/{DST}\n")
open(os.path.join(CFG, f"{DST}.yml"), "w").write(t)

c = yaml.safe_load(open(os.path.join(CFG, f"{DST}.yml")))
b = c["DINOv3STAsRDTTrack"]
assert b["ir_drop_prob"] == 0.0 and b["depth_drop_prob"] == 0.0, b
assert b["input_norm"] == "instance" and b["unfreeze_rgb_stages"] == 12, b
print(f"wrote {DST}.yml   ir_drop=0 depth_drop=0 input_norm=instance unfreeze=12")

# ---- BDD real-night configs
for split in ["night", "daytime"]:
    s = f"{SRC}_bdd_{split}"
    t = open(os.path.join(CFG, f"{s}.yml")).read()
    assert t.count(IR) == 1 and t.count(DEP) == 1, f"{split}: dropout not unique"
    assert t.count(f"./outputs/{s}\n") == 1, f"{split}: output_dir not unique"
    t = t.replace(IR, "  ir_drop_prob: 0.0\n").replace(DEP, "  depth_drop_prob: 0.0\n")
    t = t.replace(f"./outputs/{s}\n", f"./outputs/{DST}_bdd_{split}\n")
    open(os.path.join(CFG, f"{DST}_bdd_{split}.yml"), "w").write(t)
    cc = yaml.safe_load(open(os.path.join(CFG, f"{DST}_bdd_{split}.yml")))
    assert cc["DINOv3STAsRDTTrack"]["ir_drop_prob"] == 0.0
    assert cc["DINOv3STAsRDTTrack"]["input_norm"] == "instance"
    print(f"wrote {DST}_bdd_{split}.yml")
print("OK  (AIC val configs reused from ufall)")
