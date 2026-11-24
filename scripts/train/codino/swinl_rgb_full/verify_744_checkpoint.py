"""Verify every original detector tensor after training and record checkpoint identity."""
import argparse, hashlib, json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--base',default='/autodl-fs/data/pretrained_models/teammate_s0918/epoch_12.pth');a=p.parse_args()
old=torch.load(a.base,map_location='cpu',weights_only=False)['state_dict']
new=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['state_dict']
branch={k:v for k,v in new.items() if k.startswith('data_preprocessor.fuse.')}
rest={k:v for k,v in new.items() if k not in branch}
assert set(old)==set(rest),'detector key mismatch'
assert all(torch.equal(v,rest[k]) for k,v in old.items()),'detector tensors changed'
assert all(torch.isfinite(v).all() for v in branch.values()),'non-finite branch'
assert branch['data_preprocessor.fuse.net.8.weight'].abs().sum()>0,'branch untrained'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
out=dict(detector_tensors_identical=len(rest),branch_parameters=sum(v.numel() for v in branch.values()),checkpoint_sha256=sha(a.checkpoint),base_sha256=sha(a.base))
Path(a.checkpoint+'.frozen_verified.json').write_text(json.dumps(out,indent=2))
print(out)
