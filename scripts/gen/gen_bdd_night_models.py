"""BDD100K eval configs for every night-trained model (the REAL-night column).

Motivation
----------
prep_night_real.py's own docstring records that the darkXX protocol is a single
global multiply -- exactly what a per-image instance normalisation is invariant
to -- so inorm's darkXX 0.5130 is a tautology, and on real night (BDD100K) inorm
is *worse* than the plain rgb3ch baseline (AP50 0.274 vs 0.284). Verified in
/tmp/bdd_night/inorm_night.log vs rgb3ch.log.

BDD100K night (3929 imgs, 63458 boxes, 5 AIC-overlapping classes: person, sign,
bicycle, car, light) is the only REAL night data we have, so every model we
consider submitting needs a number on it. Right now only rgb3ch, nightsim,
inorm and the frozen prompt model have one.

BDD has no infrared and no depth, so the 5ch checkpoints are probed with
all-black aux (data_aux/BDD100K/{ir_null,depth_null}) -- i.e. this measures them
with the aux modalities *absent*. That is the honest deployment number for a
multimodal model whenever the extra sensors are unavailable, and it is directly
comparable to the RGB-only runs on the same images.

Both splits are emitted for each model:
    BDD-daytime -> pure cross-dataset transfer (AIC -> BDD), daytime
    BDD-night   -> the number that matters
    night/daytime ratio -> the day->night drop, free of the cross-dataset gap

Generated (5ch, backbone matched to the checkpoint; aux fed all-black):
    deimv2_dinov3_l_aic_rdttrack_heat_night_log         nightlog  unfreeze 2
    deimv2_dinov3_l_aic_rdttrack_heat_night_log_inorm   nli (E9)  unfreeze 2 + input_norm
    deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight  rnight(E10) unfreeze 2
    deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall
                                                        ufall (E11) unfreeze 12 + input_norm
Generated (3ch):
    deimv2_dinov3_l_aic_nightsim_inorm_rnight           irn (E12) DINOv3STAs + input_norm
"""
import os

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
MM_TPL = "deimv2_dinov3_l_aic_rdttrack_heat_bdd_night.yml"      # 5ch, black aux
RGB_TPL = "deimv2_dinov3_l_aic_nightsim_inorm_bdd_night.yml"    # 3ch + input_norm
SPLITS = ["night", "daytime"]

# (stem, input_norm or None) -- 5ch family
MM = [
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log", None),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_inorm", "instance"),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight", None),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall", "instance"),
]
RGB = [
    ("deimv2_dinov3_l_aic_nightsim_inorm_rnight", "instance"),
]

OLD_OUT_MM = "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_bdd_night"
ANCHOR = "  conv_inplane: 32\n  hidden_dim: 224\n"

mm_tpl = open(os.path.join(CFG, MM_TPL)).read()
assert mm_tpl.count(OLD_OUT_MM + "\n") == 1, "MM template output_dir not unique"
assert mm_tpl.count(ANCHOR) == 1, "MM template backbone anchor not unique"
assert "ir_null" in mm_tpl and "depth_null" in mm_tpl, "black-aux paths missing"

for stem, inorm in MM:
    for split in SPLITS:
        txt = mm_tpl.replace(OLD_OUT_MM + "\n", f"./outputs/{stem}_bdd_{split}\n")
        txt = txt.replace("data_aux/BDD100K/instances_bdd_night.json",
                          f"data_aux/BDD100K/instances_bdd_{split}.json")
        if inorm:
            txt = txt.replace(ANCHOR, ANCHOR + f"  input_norm: {inorm}\n")
        dst = os.path.join(CFG, f"{stem}_bdd_{split}.yml")
        open(dst, "w").write(txt)
        # verify
        c = open(dst).read()
        assert f"./outputs/{stem}_bdd_{split}" in c
        assert f"instances_bdd_{split}.json" in c
        assert ("input_norm: instance" in c) == bool(inorm)
        print(f"wrote {os.path.basename(dst):<66} input_norm={inorm or 'none'}")

OLD_OUT_RGB = "./outputs/deimv2_dinov3_l_aic_nightsim_inorm_bdd_night"
rgb_tpl = open(os.path.join(CFG, RGB_TPL)).read()
assert rgb_tpl.count(OLD_OUT_RGB + "\n") == 1, "RGB template output_dir not unique"

for stem, inorm in RGB:
    for split in SPLITS:
        txt = rgb_tpl.replace(OLD_OUT_RGB + "\n", f"./outputs/{stem}_bdd_{split}\n")
        txt = txt.replace("data_aux/BDD100K/instances_bdd_night.json",
                          f"data_aux/BDD100K/instances_bdd_{split}.json")
        dst = os.path.join(CFG, f"{stem}_bdd_{split}.yml")
        open(dst, "w").write(txt)
        c = open(dst).read()
        assert f"./outputs/{stem}_bdd_{split}" in c
        assert f"instances_bdd_{split}.json" in c
        assert "input_norm: instance" in c, "3ch template must keep input_norm"
        print(f"wrote {os.path.basename(dst):<66} input_norm={inorm or 'none'}")

print("OK")
