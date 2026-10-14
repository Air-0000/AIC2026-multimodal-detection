"""Derive BDD100K eval configs for the RGB-only model family (night + daytime).

Why daytime too: our BDD-night zero-shot AP50 (0.284) mixes two effects --
the AIC->BDD cross-dataset gap AND the day->night gap. Evaluating the same model
on BDD **daytime** (5258 imgs) separates them:

    BDD-day AP50            = pure cross-dataset transfer, daytime
    BDD-night / BDD-day     = our day->night drop, same dataset/classes
    published Lower-Bound   = BDD-day-trained -> BDD-night (day->night only)

Models covered:
    deimv2_dinov3_l_aic_rgb3ch              (baseline)
    deimv2_dinov3_l_aic_nightsim            (E4, day->night augmented)
    deimv2_dinov3_l_aic_nightsim_inorm      (E5, E4 + backbone input_norm)
"""
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
TPL = "deimv2_dinov3_l_aic_rgb3ch_bdd_night.yml"
tpl = open(os.path.join(CFG, TPL)).read()

MODELS = [
    ("deimv2_dinov3_l_aic_rgb3ch", None),
    ("deimv2_dinov3_l_aic_nightsim", None),
    ("deimv2_dinov3_l_aic_nightsim_inorm", "instance"),
    ("deimv2_dinov3_l_aic_nightcut", None),
]
SPLITS = ["night", "daytime"]

for model, inorm in MODELS:
    for split in SPLITS:
        txt = tpl
        txt = txt.replace("./outputs/deimv2_dinov3_l_aic_rgb3ch_bdd_night",
                          f"./outputs/{model}_bdd_{split}")
        txt = txt.replace("data_aux/BDD100K/instances_bdd_night.json",
                          f"data_aux/BDD100K/instances_bdd_{split}.json")
        if inorm:
            assert "  conv_inplane: 32\n  hidden_dim: 224" in txt
            txt = txt.replace("  conv_inplane: 32\n  hidden_dim: 224",
                              f"  conv_inplane: 32\n  hidden_dim: 224\n  input_norm: {inorm}")
        dst = os.path.join(CFG, f"{model}_bdd_{split}.yml")
        open(dst, "w").write(txt)

        c = yaml.safe_load(open(dst))
        vd = c["val_dataloader"]["dataset"]
        print(f"{os.path.basename(dst):<52} ann={os.path.basename(vd['ann_file'])} "
              f"input_norm={c['DINOv3STAs'].get('input_norm','none')}")
print("OK")
