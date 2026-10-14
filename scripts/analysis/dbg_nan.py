import os, sys
ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor", "DEIMv2"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
os.chdir(os.path.join(ROOT, "vendor", "DEIMv2"))

import torch
from engine.misc import dist_utils
from engine.solver import TASKS
import engine.backbone
from engine.core import YAMLConfig

cfg = YAMLConfig("configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3.yml")
model = cfg.model
bb = model.backbone
bb.train()
bb = bb.cuda()

x = torch.rand(1, 6, 800, 800, device="cuda")
with torch.no_grad():
    feats = bb(x)
for i, f in enumerate(feats):
    print(f"feat{i}: shape={tuple(f.shape)} finite={torch.isfinite(f).all().item()} "
          f"min={f.min().item():.4f} max={f.max().item():.4f}")

# 也查 SE/DCN 单模块数值
with torch.no_grad():
    for i, (se, dcn) in enumerate(zip(bb.aux_se, bb.aux_dcn)):
        c = dcn.offset_conv.weight.shape[1] // 2
        a = torch.randn(1, c, 16, 16, device="cuda")
        r = torch.randn(1, c, 16, 16, device="cuda")
        d = dcn(a, r)
        s = se(d)
        print(f"scale{i} c={c}: dcn finite={torch.isfinite(d).all().item()} se finite={torch.isfinite(s).all().item()}")
