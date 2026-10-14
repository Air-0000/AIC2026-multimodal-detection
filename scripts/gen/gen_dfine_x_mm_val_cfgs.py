#!/usr/bin/env python3
"""Generate the 10 eval configs (day + 9 night levels) for the D-FINE mm line."""
import io
import os

CFG_DIR = '/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2'
BASE = 'deim_hgnetv2_x_aic_obj2coco_ft_dcnmm.yml'
PREFIX = 'deim_hgnetv2_x_aic_obj2coco_ft_dcnmm_val_'
NIGHT = ['50', '25', '12', '05', '01', '005', '002', '001', '0005']

TPL = """__include__: [
  '{base}',
]

output_dir: ./outputs/{stem}_val_{tag}

# night 档: RGB 换暗化副本, IR/热图不变 (日/夜同 IR, 数据集语义与 RGB 线评测一致)
val_dataloader:
  dataset:
    img_folder: /home/constant/CodeBuddy/Dev/AIC/data_aux/aic_valid_rgb_{tag}
"""

# day (orig) config
with io.open(os.path.join(CFG_DIR, PREFIX + 'orig.yml'), 'w', encoding='utf-8') as f:
    f.write("__include__: [\n  '%s',\n]\n\noutput_dir: ./outputs/%sval_orig\n" % (BASE, PREFIX))
    print(PREFIX + 'orig.yml')

for lv in NIGHT:
    tag = 'night' + lv
    name = PREFIX + tag + '.yml'
    with io.open(os.path.join(CFG_DIR, name), 'w', encoding='utf-8') as f:
        f.write(TPL.format(base=BASE, stem=PREFIX.rstrip('_'), tag=tag))
    print(name)
