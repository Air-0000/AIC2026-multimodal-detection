"""Standalone DCT validation: run the postprocessor on a real val batch whose
depth channel is PRESENT (the config that previously crashed grid_sample).
Confirms the integer-indexing DCT no longer triggers a CUDA scatter/gather assert.
"""
import os
import torch

os.chdir("/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")
from engine.core import YAMLConfig

CFG = "configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_depthrior_val_dark05.yml"
cfg = YAMLConfig(CFG)
model = cfg.model.cuda().eval()
post = cfg.postprocessor
print("postprocessor.depth_cond_enable =", post.depth_cond_enable)
dl = cfg.val_dataloader
samples, targets = next(iter(dl))
samples = samples.cuda()
with torch.no_grad():
    outputs = model(samples)
    orig = torch.stack([t["orig_size"] for t in targets]).cuda()
    results = post(outputs, orig, samples)
print("DCT OK on depth-present batch: %d images, boxes[0] = %s"
      % (len(results), tuple(results[0]["boxes"].shape)))
print("DCT VALIDATED")
