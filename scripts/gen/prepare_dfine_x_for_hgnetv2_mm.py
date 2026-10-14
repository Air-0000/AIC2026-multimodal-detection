#!/usr/bin/env python3
"""Remap dfine_x_obj2coco.pth backbone keys for HGNetv2MultiModal tuning.

The multimodal backbone nests the stock HGNetv2 under `rgb_net.`, so every
`backbone.<rest>` key in the RGB checkpoint must become
`backbone.rgb_net.<rest>`. Encoder/decoder keys pass through unchanged.
Output: ckpts/dfine_x_obj2coco_mm.pth (structure: {'model': sd}).
"""
import torch

SRC = '/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/ckpts/dfine_x_obj2coco.pth'
DST = '/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/ckpts/dfine_x_obj2coco_mm.pth'

s = torch.load(SRC, map_location='cpu', weights_only=False)
sd = s['model'] if 'model' in s else s
out = {}
n_bb = 0
for k, v in sd.items():
    if k.startswith('backbone.'):
        out['backbone.rgb_net.' + k[len('backbone.'):]] = v
        n_bb += 1
    else:
        out[k] = v
torch.save({'model': out}, DST)
print(f'remapped {n_bb}/{len(sd)} backbone keys -> {DST}')
