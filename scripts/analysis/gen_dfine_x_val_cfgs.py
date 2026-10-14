#!/usr/bin/env python3
"""Generate val configs (day + 9 night levels) for deim_hgnetv2_x_aic_obj2coco_ft.

Each val config simply includes the training config (for architecture) and the
matching dataset config (aic_detection.yml for day, aic_detection_nightXX.yml
for dark levels). Later include wins, so the dataset block overrides paths.
"""
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
BASE = "deim_hgnetv2_x_aic_obj2coco_ft"
LEVELS = ["orig", "night50", "night25", "night12", "night05", "night005",
          "night01", "night002", "night001", "night0005"]

def main():
    for lvl in LEVELS:
        ds = "../dataset/aic_detection.yml" if lvl == "orig" else \
             f"../dataset/aic_detection_{lvl}.yml"
        text = (
            f"__include__: [\n"
            f"  '{BASE}.yml',\n"
            f"  '{ds}',\n"
            f"]\n\n"
            f"output_dir: ./outputs/{BASE}_val_{lvl}\n"
        )
        path = os.path.join(ROOT, f"{BASE}_val_{lvl}.yml")
        with open(path, "w") as f:
            f.write(text)
        print("wrote", path)

if __name__ == "__main__":
    main()
