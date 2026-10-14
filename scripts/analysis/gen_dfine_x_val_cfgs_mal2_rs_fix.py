#!/usr/bin/env python3
"""Regenerate mal2 + rs val configs WITH the ops override fix.

cpoff 老坑: include 的 aic_detection*.yml 里 ops 为 None, PIL 直进 collate
=> "Image object is not subscriptable", 10 档评测全 rc=1。
"""
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
BASES = ["deim_hgnetv2_x_aic_obj2coco_ft_mal2",
         "deim_hgnetv2_x_aic_obj2coco_ft_rs"]
LEVELS = ["orig", "night50", "night25", "night12", "night05", "night005",
          "night01", "night002", "night001", "night0005"]

for BASE in BASES:
    for lvl in LEVELS:
        ds = "../dataset/aic_detection.yml" if lvl == "orig" else \
             f"../dataset/aic_detection_{lvl}.yml"
        # ⚠ 必须覆写 val ops (cpoff 老坑), 否则 PIL 直进 collate 崩
        text = (
            f"__include__: [\n"
            f"  '{BASE}.yml',\n"
            f"  '{ds}',\n"
            f"]\n\n"
            f"output_dir: ./outputs/{BASE}_val_{lvl}\n\n"
            f"# ops 覆写: aic_detection*.yml 的 ops:None 会让 PIL 直进 collate 崩\n"
            f"val_dataloader:\n"
            f"  total_batch_size: 4\n"
            f"  dataset:\n"
            f"    transforms:\n"
            f"      ops:\n"
            f"        - {{type: Resize, size: [640, 640]}}\n"
            f"        - {{type: ConvertPILImage, dtype: \"float32\", scale: True}}\n"
        )
        path = os.path.join(ROOT, f"{BASE}_val_{lvl}.yml")
        with open(path, "w") as f:
            f.write(text)
        print("wrote", os.path.basename(path))
