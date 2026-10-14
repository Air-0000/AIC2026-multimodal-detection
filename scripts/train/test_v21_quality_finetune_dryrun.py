#!/usr/bin/env python3
import os
import sys

base_dir = "/home/ailab/ssd/bigdata/liuzhengyang"
deim_dir = os.path.join(base_dir, "deimv2")
for p in [deim_dir, os.path.join(base_dir, "scripts/lib")]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from engine.core import YAMLConfig

print("Python:", sys.executable)
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

cfg_path = os.path.join(deim_dir, "configs/deimv2/sovereign_v21_quality_finetune_ailab40.yml")
print(f"Loading config from {cfg_path} ...")
cfg = YAMLConfig(cfg_path)
print("Config loaded successfully!")

train_dl = cfg.train_dataloader
val_dl = cfg.val_dataloader

print("Train dataloader batches:", len(train_dl))
print("Val dataloader batches:", len(val_dl))

# Load one batch from train_dl
sample, target = next(iter(train_dl))
print("Train Sample shape:", sample.shape, "dtype:", sample.dtype)
print("Train Targets count:", len(target))
print("Train Sample target keys:", list(target[0].keys()))

# Load one batch from val_dl
v_sample, v_target = next(iter(val_dl))
print("Val Sample shape:", v_sample.shape, "dtype:", v_sample.dtype)
print("Val Targets count:", len(v_target))

# Load model and weights
model = cfg.model.cuda()
ck_path = os.path.join(base_dir, "weights/sovereign_v21_apex_soup_581.pth")
print(f"Loading weights from {ck_path} ...")
ck = torch.load(ck_path, map_location="cpu")
w = ck["ema"]["module"] if "ema" in ck and "module" in ck["ema"] else ck.get("model", ck)
missing, unexpected = model.load_state_dict(w, strict=False)
print(f"Weights loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")

# Run 1 forward + backward step
criterion = cfg.criterion.cuda()
optimizer = cfg.optimizer

model.train()
sample = sample.cuda()
target = [{k: v.cuda() for k, v in t.items()} for t in target]

out = model(sample, target)
loss_dict = criterion(out, target)
loss = sum(loss_dict.values())
print("Loss values:", {k: round(v.item(), 4) for k, v in loss_dict.items()}, "Total loss:", round(loss.item(), 4))

loss.backward()
print("Backward pass succeeded! Gradient norm for first param:", model.backbone.rgb_net.stem.stem1.conv.weight.grad.norm().item())
optimizer.step()
optimizer.zero_grad()

print("ALL PRE-FLIGHT CHECKS PASSED 100%!")
