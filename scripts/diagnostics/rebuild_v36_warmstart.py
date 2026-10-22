#!/usr/bin/env python3
"""Rebuild a migrated warm-start on this host from the frozen checkpoint + stage-1 template."""
import hashlib, json, sys
import torch

NR = '/home/ailab/ssd/bigdata/hejianzhong/AIC_V36_INJ_20261019'
R = '/home/ailab/ssd/bigdata/hejianzhong/AIC_V29'
sys.path[:0] = [NR + '/vendor_DEIMv2', NR + '/scripts/lib']

from engine.core import YAMLConfig
from injection_stage_migration import load_migrated_state

FAMS = ('ir_proj', 'ir_align', 'occlusion_gate', 'coord_lift', 'ir_fuse', 'ir_res', 'bilateral_pool')


def canonical(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k]
        if not hasattr(t, 'shape'):
            continue
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.contiguous().to(torch.float32).numpy().tobytes() if t.is_floating_point()
                 else t.contiguous().numpy().tobytes())
    return h.hexdigest()


def rebuild(seed):
    stages = [1, 2]
    cfgp = f'{NR}/vendor_DEIMv2/configs/deimv2/v36_injection_warmstart_20261019/seed{seed}_p3p4.yml'
    cfg = YAMLConfig(cfgp, HGNetv2MultiModal={'inject_stages': stages})
    model = cfg.model
    tpl = torch.load(f'{NR}/warmstarts/stage1_template_seed{seed}.pth', map_location='cpu',
                     weights_only=False)
    tpl = tpl.get('model', tpl)
    fresh = {k: v for k, v in tpl.items() if k.startswith('backbone.')
             and any(f'.{fam}.0.' in k for fam in FAMS)}
    model.load_state_dict(fresh, strict=False)
    src = torch.load(f'{R}/outputs/checkpoints/v36_mechanism_20261018/v36_frozen.pth',
                     map_location='cpu', mmap=True, weights_only=False)['model']
    load_migrated_state(model, src, [2, 3], stages)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    out = f'{NR}/warmstarts/seed{seed}_p3p4.pth'
    torch.save({'model': sd, 'provenance': {
        'rebuilt_on': 'ailab40', 'source': 'v36_frozen.pth', 'source_stages': [2, 3],
        'target_stages': stages, 'template': f'stage1_template_seed{seed}.pth',
        'note': 'rebuilt locally from frozen checkpoint plus shipped stage-1 template'}}, out)
    print(f'seed{seed}: canonical={canonical(sd)} tensors={len(sd)} -> {out}')


if __name__ == '__main__':
    for s in (42, 43):
        rebuild(s)
