"""Derive BDD100K-night eval configs that add a test-time IlluminationNorm.

Zero-retrain probe: how much of the RGB baseline's night drop is pure
photometric drift (night input statistics vs the daytime ImageNet Normalize it
was trained with)? Sweeps gamma (shadow lift) and the target luma mean.
"""
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
BASE = "deimv2_dinov3_l_aic_rgb3ch_bdd_night.yml"
# MUST land after ConvertPILImage (input is still a PIL image before that, and
# IlluminationNorm only handles [0,1] tensors -> it would silently no-op).
ANCHOR = ("        - {type: Normalize, mean: [0.485, 0.456, 0.406], "
          "std: [0.229, 0.224, 0.225]}")

# tag, gamma, target_mean, standardize
VARIANTS = [
    ("illum_g10_m45", 1.0, 0.45, False),   # pure brightness rescale
    ("illum_g07_m45", 0.7, 0.45, False),   # mild shadow lift
    ("illum_g05_m45", 0.5, 0.45, False),   # strong shadow lift
    ("illum_g07_m45_std", 0.7, 0.45, True) # + per-image standardisation
]

base = open(os.path.join(CFG, BASE)).read()
assert base.count(ANCHOR) == 1, "val anchor not found/uniquely matched"

for tag, g, m, std in VARIANTS:
    op = (f"        - {{type: IlluminationNorm, target_mean: {m}, gamma: {g}, "
          f"standardize: {str(std).lower()}}}\n")
    txt = base.replace(ANCHOR, op + ANCHOR)
    txt = txt.replace("./outputs/deimv2_dinov3_l_aic_rgb3ch_bdd_night",
                      f"./outputs/deimv2_dinov3_l_aic_rgb3ch_bdd_night__{tag}")
    dst = os.path.join(CFG, BASE.replace("_bdd_night.yml", f"_bdd_night_{tag}.yml"))
    open(dst, "w").write(txt)

    # verify
    c = yaml.safe_load(open(dst))
    ops = c["val_dataloader"]["dataset"]["transforms"]["ops"]
    got = [o for o in ops if o and o.get("type") == "IlluminationNorm"]
    assert len(got) == 1, (dst, got)
    idx = [i for i, o in enumerate(ops) if o and o.get("type") == "Normalize"][0]
    assert [o.get("type") for o in ops][idx - 1] == "IlluminationNorm"
    print(f"{os.path.basename(dst):<58} {got[0]}")
print("OK")
