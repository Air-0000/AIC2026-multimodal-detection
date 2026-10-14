"""E11 / E12: close the two confounds the E1-E10 sweep left open.

Diagnosis (from the E9/E10 logs):
  * inorm (RGB-only, E5) is the best model on every axis
      in-domain AP 0.5277 | darkXX MEAN 0.5130 | nightXX MEAN 0.4567
  * every 5ch model is worse, and the logs show why the comparison was never
    fair: `[RDTTrack] trainable params: 1.3M / total 22.9M (RGB backbone
    frozen=True)` -- the entire 21.6M-param DINOv3 vision stream is frozen
    (`multimodal_backbones.py:951-952`), and `unfreeze_rgb_stages: 2` only
    reopens the last 2 of 12 blocks (4.8M).
  * inorm instead runs `finetune: True`, i.e. all 22.9M params train.
So "single-modal beats multimodal" was really "22.9M trainable beats 4.8M".

E11 = rnight + input_norm: instance + unfreeze_rgb_stages: 12
    Everything the RGB line has that the 5ch line does not, on top of the best
    multimodal recipe (rnight). This is the last fair shot for fusion: if a
    fully-trainable RGB stream still cannot profit from IR/Depth, the aux
    modalities carry no usable signal for this dataset.
    Warm-started from rnight (aux stems already trained).

E12 = inorm + RandomNight5ch (the REALISTIC night augmentation)
    inorm's one remaining weakness is the realistic protocol: it loses 0.056
    going darkXX -> nightXX (0.5130 -> 0.4567), by far the largest gap among the
    top models, because it was only trained with the *global* attenuation of
    RandomRGBDegrade5ch. RandomNight5ch adds lamp lighting, Poisson-Gaussian
    sensor noise and a night white-balance shift -- exactly what nightXX
    evaluates. Fine-tuned from the inorm checkpoint so the darkXX capability
    is kept.

Both are single-variable vs their parent config.
"""
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"


def write(name, txt):
    dst = os.path.join(CFG, name)
    open(dst, "w").write(txt)
    print("wrote", dst)
    return dst


# ---------------------------------------------------------------- E11
E11_SRC = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight.yml"
E11_DST = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall.yml"
E11_OUT_OLD = "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight"
E11_OUT_NEW = "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall"

txt = open(os.path.join(CFG, E11_SRC)).read()
assert txt.count(E11_OUT_OLD + "\n") == 1, "E11 output_dir not unique"
assert txt.count("\n  unfreeze_rgb_stages: 2\n") == 1, "E11 unfreeze line not unique"

txt = txt.replace(E11_OUT_OLD + "\n", E11_OUT_NEW + "\n")
# unfreeze ALL 12 DINOv3 blocks (code clamps to len(blocks)); + input_norm so the
# RGB stream gets the same illumination invariance the RGB-only winner uses.
txt = txt.replace(
    "\n  unfreeze_rgb_stages: 2\n",
    "\n  unfreeze_rgb_stages: 12     # E11: ALL blocks (was 2/12) -- 22.9M trainable\n"
    "  input_norm: instance        # E11: same RGB illumination norm as inorm\n",
)
write(E11_DST, txt)

c = yaml.safe_load(open(os.path.join(CFG, E11_DST)))
b = c["DINOv3STAsRDTTrack"]
assert b["unfreeze_rgb_stages"] == 12 and b["input_norm"] == "instance", b
assert c["output_dir"] == E11_OUT_NEW
print("  E11 backbone:", {k: b[k] for k in
                          ("ir_drop_prob", "depth_drop_prob",
                           "unfreeze_rgb_stages", "input_norm")})


# ---------------------------------------------------------------- E12
E12_SRC = "deimv2_dinov3_l_aic_nightsim_inorm.yml"
E12_DST = "deimv2_dinov3_l_aic_nightsim_inorm_rnight.yml"
E12_OUT_OLD = "./outputs/deimv2_dinov3_l_aic_nightsim_inorm"
E12_OUT_NEW = "./outputs/deimv2_dinov3_l_aic_nightsim_inorm_rnight"
OLD_OP = ("        - {type: RandomRGBDegrade5ch, p: 0.5, brightness: [0.05, 0.80], "
          "contrast: [0.30, 0.90], noise_std: [0.02, 0.12], log_brightness: true}")
NEW_OP = ("        - {type: RandomNight5ch, p: 0.5, level: [0.005, 0.80], "
          "photon_max: 70.0, read_sigma: 0.012, log_level: true, lamps: [1, 4]}")

txt = open(os.path.join(CFG, E12_SRC)).read()
assert txt.count(E12_OUT_OLD + "\n") == 1, "E12 output_dir not unique"
assert txt.count(OLD_OP) == 1, "E12 degrade op not unique"
txt = txt.replace(E12_OUT_OLD + "\n", E12_OUT_NEW + "\n")
txt = txt.replace(OLD_OP, NEW_OP)
write(E12_DST, txt)

c = yaml.safe_load(open(os.path.join(CFG, E12_DST)))
ops = [o.get("type") for o in c["train_dataloader"]["dataset"]["transforms"]["ops"]]
assert "RandomNight5ch" in ops and "RandomRGBDegrade5ch" not in ops, ops
assert c["output_dir"] == E12_OUT_NEW
assert c["DINOv3STAs"]["input_norm"] == "instance"
assert c["DINOv3STAs"]["finetune"] is True
print("  E12 ops:", ops)
print("  E12 backbone: DINOv3STAs, finetune=True (full 22.9M trainable), input_norm=instance")
