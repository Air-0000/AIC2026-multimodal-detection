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
print('loading', len(filtered), 'of', len(sd), 'keys; skipped', len(sd) - len(filtered))
model.load_state_dict(filtered, strict=False)
model.cuda()

dl = cfg.train_dataloader
it = iter(dl)
for bi in range(5):
    samples, targets = next(it)
    s_full = (samples.tensors if not isinstance(samples, torch.Tensor) else samples).to('cuda')
    ti = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
    # EVAL (running stats)
    model.eval()
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(s_full)
    pb = out.get('pred_boxes')
    print(f'batch{bi} EVAL full: nan={bool(torch.isnan(pb).any()) if pb is not None else "n/a"}')
    # TRAIN no_grad (batch stats, full bs8)
    model.train()
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(s_full, targets=ti)
    pb = out['pred_boxes']
    print(f'batch{bi} TRAIN-no_grad bs8: nan={bool(torch.isnan(pb).any())} inf={bool(torch.isinf(pb).any())} max|pb|={float(pb.abs().max()):.3f}')
    torch.cuda.empty_cache()
