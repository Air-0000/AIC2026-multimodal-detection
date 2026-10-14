"""Dry-run verification: build the model from a Stage config and load the clean
warm-start weights exactly the way `-t` does, WITHOUT training or writing
checkpoints. Prints which keys are missed / shape-unmatched.

Expected result:
  matched   = all three DINOv3 streams
  missed    = encoder / decoder / convs / aux_proj / sta / prompt / dt_ort
              (random init -- by design)
  unmatched = []   (a non-empty list means the warm-start is broken)
"""
import os
import sys

AIC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(AIC_ROOT, 'vendor', 'DEIMv2')
sys.path.insert(0, os.path.join(AIC_ROOT, 'scripts', 'lib'))
sys.path.insert(0, REPO)

import torch  # noqa: E402
from engine.core.yaml_config import YAMLConfig  # noqa: E402

STAGE = sys.argv[1] if len(sys.argv) > 1 else 'A'
CKPT = os.path.join(REPO, 'ckpts/dinov3/vitl16_rdttrack_scratch_full.pth')
CFG = os.path.join(REPO, 'configs', 'deimv2',
                   f'deimv2_dinov3_l_aic_vitl800_scratch_stage{STAGE}_'
                   f'{"rgb" if STAGE == "A" else "mod" if STAGE == "B" else "fuse"}.yml')

cfg = YAMLConfig(CFG)
model = cfg.model
state = torch.load(CKPT, map_location='cpu')
pre = state['ema']['module'] if 'ema' in state else state['model']

cur = model.state_dict()
missed, unmatched, matched = [], [], {}
groups = {}
for k, v in cur.items():
    if k in pre:
        if v.shape == pre[k].shape:
            matched[k] = pre[k]
            g = '.'.join(k.split('.')[:2])
            groups[g] = groups.get(g, 0) + v.numel()
        else:
            unmatched.append(k)
    else:
        missed.append(k)

print(f'config : {os.path.basename(CFG)}')
print(f'ckpt   : {os.path.basename(CKPT)}')
print(f'model  : {sum(v.numel() for v in cur.values())/1e6:.2f}M params')
print()
print('--- LOADED FROM CLEAN OFFICIAL WEIGHTS ---')
tot_m = 0
for g, n in sorted(groups.items(), key=lambda x: -x[1]):
    print(f'  {g:26s} {n/1e6:8.2f}M')
    tot_m += n
print(f'  {"TOTAL matched":26s} {tot_m/1e6:8.2f}M')
print()
print('--- MISSED (random init, expected: fusion + head) ---')
bym = {}
for k in missed:
    g = '.'.join(k.split('.')[:2])
    bym[g] = bym.get(g, 0) + cur[k].numel()
tot_r = 0
for g, n in sorted(bym.items(), key=lambda x: -x[1]):
    print(f'  {g:26s} {n/1e6:8.2f}M  ({len([k for k in missed if g in k])} keys)')
    tot_r += n
print(f'  {"TOTAL random-init":26s} {tot_r/1e6:8.2f}M  ({len(missed)} keys)')
print(f'  coverage = {len(matched)/(len(matched)+len(missed))*100:.1f}% of keys')
print()
print(f'--- UNMATCHED (shape mismatch) = {len(unmatched)} ---')
for k in unmatched[:10]:
    print(f'  {k}: model {tuple(cur[k].shape)} vs ckpt {tuple(pre[k].shape)}')
print()
print('VERDICT:', 'OK - aux streams loaded from official weights'
      if (not unmatched and 'backbone.ir_dinov3' in groups
          and 'backbone.depth_dinov3' in groups) else 'PROBLEM')
