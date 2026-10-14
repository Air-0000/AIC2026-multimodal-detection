#!/usr/bin/env python3
"""Generate val configs (day + 9 night levels) for the OFFLINE-CP ft run.

Each val config includes the cpoff training config (architecture) and the
matching dataset config; later include wins, so val paths stay the standard
aic_detection(_nightXX).yml (offline CP only changes the TRAIN split).
"""
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
BASE = "deim_hgnetv2_x_aic_obj2coco_ft_cpoff"
LEVELS = ["orig", "night50", "night25", "night12", "night05", "night005",
          "night01", "night002", "night001", "night0005"]

for lvl in LEVELS:
    ds = "../dataset/aic_detection.yml" if lvl == "orig" else \
         f"../dataset/aic_detection_{lvl}.yml"
    # ⚠ 必须覆写 val ops: include 的 aic_detection*.yml 里 ops 为 None,
    # 会让 PIL Image 直接进 collate => "Image object is not subscriptable".
    # dcn/_dcncp 的 val yml 都带这段, 首版 cpoff 生成器漏了 => 10 档评测全 rc=1。
    text = (
        f"__include__: [\n"
        f"  '{BASE}.yml',\n"
        f"  '{ds}',\n"
        f"]\n\n"
        f"output_dir: ./outputs/{BASE}_val_{lvl}\n\n"
        f"# 本文件 body 键覆盖 __include__; aic_detection*.yml 的 ops:None 会让 PIL 直进\n"
        f"# collate 崩 (\"Image object is not subscriptable\"), 这里再覆写一次\n"
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
    print("wrote", path)
