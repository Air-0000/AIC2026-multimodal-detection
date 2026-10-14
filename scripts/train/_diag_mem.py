"""一次性诊断: 分段打印显存, 确认 probe_mem.py 的 forward/backward 是否真的执行。"""
import os, sys

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIM = os.path.join(ROOT, "vendor/DEIMv2")
sys.path.insert(0, DEIM)
sys.path.insert(0, os.path.join(ROOT, "scripts/lib"))
os.chdir(DEIM)

import torch
from engine.core import YAMLConfig

MB = 2 ** 20
cfg_path = sys.argv[1]
bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1

def mem(tag):
    a = torch.cuda.memory_allocated() / MB
    r = torch.cuda.memory_reserved() / MB
    p = torch.cuda.max_memory_allocated() / MB
    print("[%s] alloc=%.0f MB reserved=%.0f MB peak=%.0f MB" % (tag, a, r, p))

cfg = YAMLConfig(cfg_path)
print("cfg:", os.path.basename(cfg_path))
print("train size:", cfg.yaml_cfg.get("eval_spatial_size"))

model = cfg.model
criterion = cfg.criterion
optimizer = cfg.optimizer
ema = getattr(cfg, "ema", None)
scaler = getattr(cfg, "scaler", None)
print("ema:", ema is not None, "| scaler:", scaler is not None)

torch.cuda.empty_cache()
mem("00 baseline")

model.to("cuda"); criterion.to("cuda")
if ema is not None:
    ema.to("cuda")
mem("01 model+crit+ema on cuda")

dl = cfg.train_dataloader
samples, targets = next(iter(dl))
print("batch:", tuple(samples.shape), "dtype", samples.dtype,
      "| n_targets", len(targets))

s = samples[:bs].to("cuda", non_blocking=True)
t = [{k: v.to("cuda") for k, v in tg.items()} for tg in targets[:bs]]
mem("02 batch on cuda")

torch.cuda.reset_peak_memory_stats()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=True):
    out = model(s, targets=t)
mem("03 after forward")
print("   out keys:", list(out.keys()) if isinstance(out, dict) else type(out))

with torch.autocast(device_type="cuda", enabled=False):
    loss_dict = criterion(out, t, epoch=0, step=0, global_step=0, epoch_step=1)
loss = sum(loss_dict.values())
print("   loss = %.4f | keys = %s" % (loss.item(), list(loss_dict.keys())))
mem("04 after criterion")

optimizer.zero_grad(set_to_none=True)
if scaler is not None:
    scaler.scale(loss).backward()
else:
    loss.backward()
mem("05 after backward")

has_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
print("   params with grad:", has_grad)

if scaler is not None:
    scaler.step(optimizer); scaler.update()
else:
    optimizer.step()
mem("06 after optimizer.step")

if ema is not None:
    ema.update(model)
mem("07 after ema.update")

print("=" * 60)
print("PEAK TOTAL: %.0f MB (%.2f GiB)" % (torch.cuda.max_memory_allocated() / MB,
                                          torch.cuda.max_memory_allocated() / 2 ** 30))
