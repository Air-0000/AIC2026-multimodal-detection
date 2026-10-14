"""Smoke test: dfine_x_obj2coco.pth -> vendored DEIMv2 engine, AIC ft config.

Checks (run on host, deimv2 env, cwd=vendor/DEIMv2):
1. ckpt structure (top-level keys, state dict source)
2. key match stats vs model built from deim_hgnetv2_x_aic_obj2coco_ft.yml
3. cls head shape mismatch is confined to class-dependent params
4. one forward+backward on dummy 3ch 640 batch
"""
import os
import sys
import torch

# HGNetv2.__init__ calls torch.distributed.get_rank() unconditionally
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)

CKPT = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/ckpts/dfine_x_obj2coco.pth"
CFG = "configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft.yml"

# --- 1. ckpt structure ---
obj = torch.load(CKPT, map_location="cpu", weights_only=False)
print("[ckpt] top-level:", list(obj.keys())[:8] if isinstance(obj, dict) else type(obj))
sd = obj.get("model", obj) if isinstance(obj, dict) else obj
if not isinstance(sd, dict) or not all(isinstance(v, torch.Tensor) for v in sd.values()):
    # maybe nested
    for k in ("model", "state_dict", "ema"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            sd = obj[k]
            print(f"[ckpt] using obj['{k}']")
            break
print("[ckpt] n_tensors:", len(sd))

# --- 2. build model from config ---
sys.path.insert(0, ".")
from engine.core.yaml_config import YAMLConfig  # noqa: E402

cfg = YAMLConfig(CFG)
model = cfg.model
msd = model.state_dict()
print("[model] n_tensors:", len(msd), " params(M):",
      sum(p.numel() for p in model.parameters()) / 1e6)

ckpt_keys, model_keys = set(sd.keys()), set(msd.keys())
matched = ckpt_keys & model_keys
only_ckpt = ckpt_keys - model_keys
only_model = model_keys - ckpt_keys
print(f"[match] matched={len(matched)} only_in_ckpt={len(only_ckpt)} only_in_model={len(only_model)}")

# class-dependent params (cls head) are EXPECTED to mismatch in shape
shape_mismatch = []
for k in sorted(matched):
    if sd[k].shape != msd[k].shape:
        shape_mismatch.append((k, tuple(sd[k].shape), tuple(msd[k].shape)))
print("[shape] mismatched:", len(shape_mismatch))
for k, a, b in shape_mismatch:
    print("   ", k, a, "->", b)
print("[ckpt-only keys sample]:", sorted(only_ckpt)[:12])

# --- 3. sanity: decoder cls head is the only shape mismatch family ---
fam = {k.split(".")[0] + "." + k.split(".")[1] for k, _, _ in shape_mismatch}
print("[shape] mismatch families:", sorted(fam)[:8])

# --- 4. forward/backward ---
tgt = {
    "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.6, 0.1, 0.1]]),
    "labels": torch.tensor([3, 7]),
    "orig_size": torch.tensor([[640, 640]]),
    "size": torch.tensor([[640, 640]])}
targets = [tgt, tgt]  # must match batch size 2
model = model.train()
x = torch.randn(2, 3, 640, 640)
out = model(x, targets)          # train-mode decoder needs targets (CDN denoising)
loss_dict = cfg.criterion(out, targets)  # criterion lives on the config object
loss = sum(loss_dict.values())
loss.backward()
print("[fwd/bwd] OK, loss =", float(loss))
print("SMOKE_OK")
