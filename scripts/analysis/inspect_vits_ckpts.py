"""Inspect local vits16-family checkpoints: param count, FFN type, key prefixes.

Goal: decide whether any of them is a *clean* (non-AIC-trained) weight set that
can serve the aux streams (ir_dinov3 / depth_dinov3).
"""
import os
import torch

AIC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.path.join(AIC_ROOT, 'vendor', 'DEIMv2')

FILES = [
    'ckpts/dinov3/dinov3_vits16_from_coco_backbone.pth',
    'ckpts/deimv2_l_coco/dinov3_vits16_stem.pth',
    'outputs/deimv2_dinov3_l_aic/dinov3_vits16_backbone.pth',
]


def strip(ck):
    for k in ('model', 'ema'):
        if isinstance(ck, dict) and k in ck:
            return strip(ck[k])
    if isinstance(ck, dict) and 'module' in ck and isinstance(ck['module'], dict):
        return strip(ck['module'])
    if isinstance(ck, dict) and 'state_dict' in ck:
        return strip(ck['state_dict'])
    return ck


for rel in FILES:
    p = os.path.join(V, rel)
    print('=' * 78)
    print(p)
    if not os.path.exists(p):
        print('  (missing)')
        continue
    sd = strip(torch.load(p, map_location='cpu'))
    n = sum(t.numel() for t in sd.values() if hasattr(t, 'numel'))
    print(f'  keys={len(sd)}  params={n/1e6:.2f}M')

    keys = list(sd.keys())
    pref = keys[0].split('.')[0]
    print(f'  first keys: {keys[:4]}')

    swiglu = [k for k in keys if '.mlp.w1' in k or '.mlp.w2' in k or '.mlp.w3' in k]
    mlp = [k for k in keys if '.mlp.fc1' in k or '.mlp.fc2' in k]
    nblk = len({k.split('.')[1] for k in keys if k.startswith('blocks.') or '.blocks.' in k})
    nblk2 = len({k.split('blocks.')[1].split('.')[0] for k in keys if 'blocks.' in k})
    print(f'  ffn: swiglu={len(swiglu)} mlp={len(mlp)}   blocks={nblk2}')
    ed = [k for k in keys if 'patch_embed.proj.weight' in k]
    for k in ed[:2]:
        print(f'  {k}: {tuple(sd[k].shape)}')
    # does it look like an aux head (extra convs / interaction) or just a backbone?
    odd = {k.split('.')[0] for k in keys}
    print(f'  top-level prefixes: {sorted(odd)[:12]}')
print('=' * 78)
