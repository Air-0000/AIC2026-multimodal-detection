"""Smoke test: PKIBlock bottleneck switch on D-FINE-X finetune host.

Verifies on CUDA:
  1. PKIVGGBlock is numerically IDENTICAL to VGGBlock at init (zero-init fuse).
  2. Model builds from the _pki.yml config; pki modules exist; ckpt matched-key
     count is the same as the baseline config (only new pki keys are extra).
  3. Forward/backward runs; grads reach the pki params.
"""
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
from engine.deim.hybrid_encoder import VGGBlock, PKIVGGBlock

# --- 1. block-level equivalence at init ---
torch.manual_seed(0)
for ch in (128, 256):
    v = VGGBlock(ch, ch, act="silu").eval()
    p = PKIVGGBlock(ch, ch, act="silu").eval()
    # copy identical submodule states (conv1/conv2)
    p.conv1.load_state_dict(v.conv1.state_dict())
    p.conv2.load_state_dict(v.conv2.state_dict())
    assert torch.all(p.pki_fuse.weight == 0), "pki_fuse not zero-init"
    x = torch.randn(2, ch, 32, 32)
    with torch.no_grad():
        d = (v(x) - p(x)).abs().max().item()
    print(f"[smoke] ch={ch}: max|VGG-PKI| = {d:.2e}")
    assert d == 0.0, "PKI not identity at init!"

# --- 2. model build + ckpt match ---
dist_utils.setup_distributed() if hasattr(dist_utils, "setup_distributed") else None
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29531",
                                         world_size=1, rank=0)

cfg = YAMLConfig("configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_pki.yml")
model = cfg.model
named = dict(model.named_parameters())
n_pki = sum(p.numel() for k, p in named.items() if "pki_" in k)
print(f"[smoke] pki params: {n_pki/1e6:.2f}M")

obj = torch.load("ckpts/dfine_x_obj2coco.pth", map_location="cpu", weights_only=False)
sd = obj.get("model", obj)
msd = model.state_dict()
matched = [k for k in sd if k in msd and sd[k].shape == msd[k].shape]
new_keys = [k for k in msd if k not in sd]
print(f"[smoke] ckpt matched={len(matched)} new(pki)={len(new_keys)}")
assert all("pki_" in k for k in new_keys), f"unexpected new keys: {new_keys[:5]}"

# --- 3. forward/backward ---
model = model.cuda().train()
x = torch.rand(2, 3, 640, 640, device="cuda")
enc_out = model.encoder(model.backbone(x))
loss = sum((o.sum() if torch.is_tensor(o) else sum(t.sum() for t in o if torch.is_tensor(t)))
           for o in (enc_out if isinstance(enc_out, (list, tuple)) else [enc_out]))
loss.backward()
g = sum(p.grad.abs().sum().item() for k, p in model.named_parameters()
        if p.grad is not None and "pki_" in k)
print(f"[smoke] pki grad sum: {g:.4f}")
assert g > 0, "no grad reached pki params"
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[smoke] ALL OK (trainable {n_train/1e6:.1f}M)")
