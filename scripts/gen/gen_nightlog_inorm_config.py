"""E9 = nightlog + input_norm: instance -- merge the two winners.

On the realistic night protocol (non-uniform lamp light + Poisson-Gaussian
noise + colour shift + attenuation):
    inorm    (RGB-only)  MEAN 0.4567   <- wins on architecture (brightness invariance)
    nightlog (multimodal) MEAN 0.4060  <- wins on fusion + real augmentation,
                                          and is the most protocol-stable model
                                          (dark 0.4285 -> night 0.4060, -0.023)

The two gains are independent: aux-modality injection vs invariance of the RGB
representation. This config adds `input_norm: instance` to the multimodal
RDTTrack backbone (implemented in DINOv3STAsRDTTrack._normalize_rgb, applied to
the first 3 channels only) on top of the nightlog recipe.
"""
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
BASE = "deimv2_dinov3_l_aic_rdttrack_heat_night_log.yml"
OUT_BASE = "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log"

txt = open(os.path.join(CFG, BASE)).read()
assert txt.count("  unfreeze_rgb_stages: 2") == 1

txt = txt.replace(OUT_BASE, f"{OUT_BASE}_inorm")
txt = txt.replace("  unfreeze_rgb_stages: 2",
                  "  unfreeze_rgb_stages: 2\n"
                  "  input_norm: instance      # per-image RGB illumination normalisation")

dst = os.path.join(CFG, "deimv2_dinov3_l_aic_rdttrack_heat_night_log_inorm.yml")
open(dst, "w").write(txt)

c = yaml.safe_load(open(dst))
bb = c["DINOv3STAsRDTTrack"]
assert bb["input_norm"] == "instance", bb.get("input_norm")
assert bb["unfreeze_rgb_stages"] == 2
print("wrote", dst)
print("  backbone input_norm:", bb["input_norm"], "| unfreeze:", bb["unfreeze_rgb_stages"])
print("  output_dir:", c["output_dir"])
