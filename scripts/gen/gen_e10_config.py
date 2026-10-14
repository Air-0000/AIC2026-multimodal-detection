"""E10 = nightlog recipe but with the REALISTIC night augmentation.

Diagnosis that motivates it: the nightXX protocol (non-uniform lamp light +
Poisson-Gaussian noise + colour shift) dropped every model hard, and the model
trained with the *photometric* augmentation (nightlog) still lost 0.023 going
dark->night, i.e. it never saw lamp lighting or noise-limited inputs. The RGB
degradation used in training only scales brightness.

RandomNight5ch reproduces the evaluation degradation at train time (randomised,
log-uniform level), so this should transfer to nightXX far better than
RandomRGBDegrade5ch does.

Same base as nightlog (RDTTrack prompt + unfreeze 2), so the augmentation is the
only variable.
"""
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
BASE = "deimv2_dinov3_l_aic_rdttrack_heat_night_log.yml"
OUT_BASE = "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log"
OLD_OP = ("        - {type: RandomRGBDegrade5ch, p: 0.5, brightness: [0.05, 0.80], "
          "contrast: [0.30, 0.90], noise_std: [0.02, 0.12], log_brightness: true}")
NEW_OP = ("        - {type: RandomNight5ch, p: 0.5, level: [0.005, 0.80], "
          "photon_max: 70.0, read_sigma: 0.012, log_level: true, lamps: [1, 4]}")

txt = open(os.path.join(CFG, BASE)).read()
assert txt.count(OLD_OP) == 1, "degrade op not uniquely found"
txt = txt.replace(OUT_BASE, f"{OUT_BASE}_rnight")
txt = txt.replace(OLD_OP, NEW_OP)

dst = os.path.join(CFG, "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight.yml")
open(dst, "w").write(txt)

c = yaml.safe_load(open(dst))
ops = [o.get("type") for o in c["train_dataloader"]["dataset"]["transforms"]["ops"]]
assert "RandomNight5ch" in ops and "RandomRGBDegrade5ch" not in ops, ops
assert ops[ops.index("Normalize") - 1] == "RandomNight5ch"
print("wrote", dst)
print("  ops:", ops)
print("  output_dir:", c["output_dir"])
