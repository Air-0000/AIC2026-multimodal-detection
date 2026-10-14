"""Smoke test: DINOv3STAsRDTTrack aux_se/aux_dcn switches (l4u3_sedcn config).

Verifies, on CUDA, with the REAL YAMLConfig (so yml kwargs wiring is tested too):
  1. model builds; SE/DCN modules exist; DCN path is EXACT identity at init
     (zero-init residual); SE is the only perturbing module.
  2. forward/backward works at the train resolution; grads reach the new params.
"""
import os
import sys

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor", "DEIMv2"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
os.chdir(os.path.join(ROOT, "vendor", "DEIMv2"))

import torch
from engine.misc import dist_utils  # noqa: F401  (train.py 同款 import 链)
from engine.solver import TASKS  # noqa: F401  触发 solver 链注册
import engine.backbone  # noqa: F401  显式触发 multimodal_backbones @register（否则 global_cfg 缺 _name）
from engine.core import YAMLConfig

cfg = YAMLConfig("configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3_sedcn.yml")
model = cfg.model
model.train()

bb = model.backbone
assert bb.aux_se is not None and bb.aux_dcn is not None, "switches not wired from yml"
n_se = sum(p.numel() for p in bb.aux_se.parameters())
n_dcn = sum(p.numel() for p in bb.aux_dcn.parameters())
print(f"[smoke] SE params {n_se/1e3:.1f}K, DCN params {n_dcn/1e6:.2f}M")

# identity-at-init check for the DCN residual path
with torch.no_grad():
    c_in = bb.aux_dcn[0].offset_conv.weight.shape[1] // 2  # 实际融合尺度通道数
    aux = torch.randn(1, c_in, 16, 16).cuda()
    rgb = torch.randn(1, c_in, 16, 16).cuda()
    m = bb.aux_dcn[0].cuda()
    out = m(aux, rgb)
    assert torch.equal(out, aux), "DCN not identity at init!"
    print(f"[smoke] DCN identity at init: OK (c_in={c_in})")

# full forward/backward at train resolution
dev = torch.device("cuda")
model = model.to(dev)
x = torch.rand(2, 6, 800, 800, device=dev)  # RGB3 + heat1 + rawIR1 + depth1
feats = model.backbone(x)
print(f"[smoke] backbone out: {[tuple(f.shape) for f in feats]}")

loss = sum(f.sum() for f in feats)
loss.backward()
g_se = sum(p.grad.abs().sum().item() for p in bb.aux_se.parameters() if p.grad is not None)
g_dcn = sum(p.grad.abs().sum().item() for p in bb.aux_dcn.parameters() if p.grad is not None)
print(f"[smoke] grad sums: SE={g_se:.3f} DCN={g_dcn:.3f}")
assert g_se > 0 and g_dcn > 0, "no grad reached new modules"
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[smoke] ALL OK (model trainable {n_train/1e6:.1f}M)")
