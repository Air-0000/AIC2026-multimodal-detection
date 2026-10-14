#!/usr/bin/env python3
"""Generate the 10 eval configs (Day orig + 9 night levels) for HGNetv2ViTDualStream."""
import io
import os

CFG_DIR = '/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2'
BASE = 'deim_hgnetv2_vit_dualstream_aic.yml'
PREFIX = 'deim_hgnetv2_vit_dualstream_val_'
NIGHT = ['50', '25', '12', '05', '01', '005', '002', '001', '0005']

TPL = """__include__: [
  '{base}',
]

output_dir: ./outputs/{stem}_val_{tag}

val_dataloader:
  dataset:
    img_folder: /home/constant/CodeBuddy/Dev/AIC/data_aux/aic_valid_rgb_{tag}
"""

# Day (orig) config
with io.open(os.path.join(CFG_DIR, PREFIX + 'orig.yml'), 'w', encoding='utf-8') as f:
    f.write("__include__: [\n  '%s',\n]\n\noutput_dir: ./outputs/%sval_orig\n" % (BASE, PREFIX))
    print(PREFIX + 'orig.yml')

for lv in NIGHT:
    tag = 'night' + lv
    name = PREFIX + tag + '.yml'
    with io.open(os.path.join(CFG_DIR, name), 'w', encoding='utf-8') as f:
        f.write(TPL.format(base=BASE, stem=PREFIX.rstrip('_'), tag=tag))
    print(name)

print("Generated all 10 evaluation YAML configs successfully!")
