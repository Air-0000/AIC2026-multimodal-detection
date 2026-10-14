#!/usr/bin/env python3
"""Generate per-level val configs for vitl16_l4like_640 family.

Per-level val configs differ ONLY in val_dataloader.dataset.img_folder
(verified by diff against vitl800_rdttrack_deepir_aug_val_*). We clone a
base config and patch img_folder + output_dir per level.

Bases: full fusion (l4like_640) and RGB-only (l4like_640_rgbonly).
"""
import os

CFG_DIR = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
ORIG_IMG = "/home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_dataset/valid/images"
LEVELS = ["night50", "night25", "night12", "night05",
          "night01", "night005", "night002", "night001", "night0005"]
BASES = [
    "deimv2_dinov3_l_aic_vitl16_l4like_640",
    "deimv2_dinov3_l_aic_vitl16_l4like_640_rgbonly",
    "deimv2_dinov3_l_aic_vitl16_640_rgbonly_xstyle",
]

for base in BASES:
    src = os.path.join(CFG_DIR, base + ".yml")
    BASE = open(src, encoding="utf-8").read()

    # day eval config (orig images)
    day = BASE.replace(
        f"output_dir: ./outputs/{base.split('_aic_')[-1]}_scratch",
        f"output_dir: ./outputs/{base.split('_aic_')[-1]}_eval/orig")
    out = os.path.join(CFG_DIR, base + "_val_orig.yml")
    with open(out, "w", encoding="utf-8") as f:
        f.write(day)
    print("wrote", os.path.basename(out))

    for lv in LEVELS:
        txt = BASE.replace(
            f"img_folder: {ORIG_IMG}",
            f"img_folder: /home/constant/CodeBuddy/Dev/AIC/data_aux/aic_valid_rgb_{lv}")
        txt = txt.replace(
            f"output_dir: ./outputs/{base.split('_aic_')[-1]}_scratch",
            f"output_dir: ./outputs/{base.split('_aic_')[-1]}_eval/{lv}")
        if ORIG_IMG not in BASE or txt == BASE:
            raise SystemExit(f"img_folder/output_dir replace FAILED for {base} {lv}")
        out = os.path.join(CFG_DIR, base + f"_val_{lv}.yml")
        with open(out, "w", encoding="utf-8") as f:
            f.write(txt)
        print("wrote", os.path.basename(out))
