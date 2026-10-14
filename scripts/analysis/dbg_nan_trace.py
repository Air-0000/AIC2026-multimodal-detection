import os, sys
ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor", "DEIMv2"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
os.chdir(os.path.join(ROOT, "vendor", "DEIMv2"))

import torch
import torch.nn as nn
from engine.misc import dist_utils
from engine.solver import TASKS
import engine.backbone
from engine.core import YAMLConfig

cfg = YAMLConfig("configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3.yml")
model = cfg.model
bb = model.backbone
bb.train().cuda()

hits = []
hooks = []
def mk(name):
    def h(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t) and not torch.isfinite(t).all():
            hits.append(name)
    return h
for name, mod in bb.named_modules():
    if list(mod.children()):  # 只看大块
        hooks.append(mod.register_forward_hook(mk(name)))

x = torch.rand(1, 6, 800, 800, device="cuda")
with torch.no_grad():
    try:
        bb(x)
    except Exception as e:
        print("fwd err:", e)
for h in hooks: h.remove()
print("first NaN modules (order of registration):", hits[:8])
