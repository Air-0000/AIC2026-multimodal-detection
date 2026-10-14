"""Single-step forward+loss smoke test for E15 (DepthPrior).

Validates the whole new path without a long training:
  dataset __getitem__ -> target['depth'] -> collate -> model -> criterion
  with depth-weighted losses. Exits non-zero on any error.
"""
import os
import torch

os.chdir("/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")
from engine.core import YAMLConfig

CFG = "configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_depthrior.yml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = YAMLConfig(CFG)
model = cfg.model.to(DEVICE); model.train()
criterion = cfg.criterion.to(DEVICE); criterion.train()

dl = cfg.train_dataloader
it = iter(dl)
samples, targets = next(it)
samples = samples.to(DEVICE)
targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

assert "depth" in targets[0], "target['depth'] missing after collate!"
print("depth channel samples[:,4] (normalized): min %.3f max %.3f mean %.3f"
      % (float(samples[:, 4].min()), float(samples[:, 4].max()), float(samples[:, 4].mean())))
all_d = torch.cat([t["depth"] for t in targets if "depth" in t])
print("target depth: n=%d min %.3f max %.3f mean %.3f zeros=%d (of %d)"
      % (all_d.numel(), float(all_d.min()), float(all_d.max()), float(all_d.mean()),
         int((all_d == 0).sum()), all_d.numel()))

outputs = model(samples, targets=targets)
print("has pred_logits_dn:", 'pred_logits_dn' in outputs,
      "| pred_logits", tuple(outputs['pred_logits'].shape))
loss_dict = criterion(outputs, targets, epoch=0)
bad = {k: float(v) for k, v in loss_dict.items() if not torch.isfinite(v)}
print("loss keys:", list(loss_dict.keys()))
print("loss sample:", {k: round(float(v), 4) for k, v in loss_dict.items()})
assert not bad, f"non-finite losses: {bad}"
print("SMOKE OK")
