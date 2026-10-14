import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import YAMLConfig
from engine.solver import TASKS

cfg = YAMLConfig('configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x.yml')
solver = TASKS[cfg.yaml_cfg['task']](cfg)
solver._setup()
model = solver.model
sd = torch.load('ckpts/deimv2_x_for_rdttrack.pth', map_location='cpu')
msd = model.state_dict()
filtered = {k: v for k, v in sd.items() if k in msd and tuple(v.shape) == tuple(msd[k].shape)}
print('loading', len(filtered), 'of', len(sd), 'keys')
model.load_state_dict(filtered, strict=False)
model.cuda()
model.train()  # batch stats, like real training

dl = cfg.train_dataloader
N = 60
nan_batches = 0
maxabs_list = []
it = iter(dl)
for bi in range(N):
    samples, targets = next(it)
    s_full = (samples.tensors if not isinstance(samples, torch.Tensor) else samples).to('cuda')
    ti = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(s_full, targets=ti)
    pb = out['pred_boxes']
    has_nan = bool(torch.isnan(pb).any() or torch.isinf(pb).any())
    if has_nan:
        nan_batches += 1
        print(f'batch{bi}: NaN/Inf in pred_boxes  max|pb|={float(pb.abs().max()):.3f}')
    else:
        maxabs_list.append(float(pb.abs().max()))
    torch.cuda.empty_cache()

print(f'=== {N} batches: {nan_batches} NaN batches ({100*nan_batches/N:.1f}%) ===')
if maxabs_list:
    print(f'max|pb| across clean batches: min={min(maxabs_list):.3f} max={max(maxabs_list):.3f} mean={sum(maxabs_list)/len(maxabs_list):.3f}')
