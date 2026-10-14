#!/usr/bin/env python3
"""Smoke test for DCNv2 residual (HGNetv2.dcn_stages).
Checks: (1) exact identity at init, (2) ckpt compatibility, (3) param debt,
(4) gradients reach the DCN path (DCN is in backbone, backward on feats OK).
"""
import os
import sys

import torch

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor", "DEIMv2"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))

from engine.backbone import hgnetv2 as H  # noqa: F401  (registers schema)

# --- 1. identity at init (independent of config plumbing) ---
m = H.DCNResidual(1024).eval()
x = torch.rand(1, 1024, 40, 40)
with torch.no_grad():
    diff = (m(x) - x).abs().max().item()
print(f"[smoke] DCNResidual identity max diff = {diff:.2e}")
assert diff == 0.0

# --- 2. build the real backbone from the new config ---
import engine  # noqa: F401
import torch.distributed as dist
from engine.core import YAMLConfig

if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)  # HGNetv2 构造期 get_rank

cfg = YAMLConfig(os.path.join(ROOT, "vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcn.yml"))
bb = cfg.model.backbone  # may be under model.emas etc; fall back below
print("[smoke] dcn_stages =", bb.dcn_stages)

n_dcn = sum(p.numel() for n_, p in bb.named_parameters() if "dcn_res" in n_)
n_all = sum(p.numel() for p in bb.parameters())
print(f"[smoke] dcn_res params = {n_dcn/1e6:.2f}M  (backbone total {n_all/1e6:.1f}M)")

# ckpt compatibility: all dcn keys must be NEW (missed), nothing else disturbed
sd = torch.load(os.path.join(ROOT, "vendor/DEIMv2/ckpts/dfine_x_obj2coco.pth"), map_location="cpu")
sd = sd.get("model", sd)
bb_keys = set(bb.state_dict().keys())
ck_keys = {k[9:] if k.startswith("backbone.") else k for k in sd.keys()}
new_keys = [k for k in bb_keys if k not in ck_keys]
dcn_new = [k for k in new_keys if "dcn_res" in k]
other_new = [k for k in new_keys if "dcn_res" not in k]
print(f"[smoke] ckpt: dcn new keys = {len(dcn_new)}, OTHER new keys = {len(other_new)}")
assert len(other_new) == 0, other_new[:8]
zero_ok = all((bb.state_dict()[k] == 0).all().item() for k in dcn_new
              if k.endswith("expand.weight"))
print(f"[smoke] expand zero-init: {zero_ok}")

# --- 3. forward + grad (fp32, eval off) ---
bb = bb.cuda().train()
x = torch.rand(2, 3, 640, 640, device="cuda")
feats = bb(x)
print("[smoke] feat shapes:", [tuple(f.shape) for f in feats])
loss = sum(f.float().pow(2).mean() for f in feats)
loss.backward()
g = sum(p.grad.abs().sum().item() for k_, p in bb.named_parameters()
        if "dcn_res" in k_ and p.grad is not None)
print(f"[smoke] dcn_res grad sum = {g:.3f} (must be > 0 and finite)")
assert g > 0 and torch.isfinite(torch.tensor(g))
print("[smoke] ALL OK")
