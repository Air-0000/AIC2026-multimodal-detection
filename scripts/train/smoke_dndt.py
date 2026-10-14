"""Smoke test for the DNDT self-training loop: run ~3 batches of train_one_epoch_dndt
to catch shape / logic / integration errors before launching a full multi-hour run.
"""
import os, sys, traceback, itertools
sys.path.insert(0, "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")

from torch.cuda.amp.grad_scaler import GradScaler
from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS
from engine.solver.dndt_solver import train_one_epoch_dndt

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2/deimv2_dinov3_l_aic_nightsim_inorm_dndt_uda.yml"
INIT = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_nightsim_inorm/best_stg1.pth"

cfg = YAMLConfig(CFG, tuning=INIT)
print("task =", cfg.yaml_cfg.get("task"))
solver = TASKS[cfg.yaml_cfg["task"]](cfg)
solver.train()
solver.self_lr_scheduler = False
solver.lr_scheduler = None
solver.lr_warmup_scheduler = None
solver.scaler = GradScaler()  # enable BF16 AMP path (matches real run, lower mem)
solver.target_dataloader = dist_utils.warp_loader(solver.cfg.build_dataloader('target_dataloader'), shuffle=True)

# pull a few real batches from each loader
print("building 3 src + 3 tgt batches ...")
src = [next(iter(solver.train_dataloader)) for _ in range(3)]
tgt = [next(iter(solver.target_dataloader)) for _ in range(3)]

d = getattr(cfg, "dndt", None) or cfg.yaml_cfg.get("dndt") or {}
d["num_classes"] = int(getattr(solver.criterion, "num_classes", 12))
print("dndt hyperparams:", d)


class _Lim:
    def __init__(self, batches):
        self.b = batches

    def __len__(self):
        return len(self.b)

    def __iter__(self):
        return iter(self.b)


try:
    stats = train_one_epoch_dndt(
        solver.self_lr_scheduler, solver.lr_scheduler, solver.model, solver.criterion,
        solver.postprocessor, _Lim(src), itertools.cycle(_Lim(tgt)), solver.optimizer,
        solver.device, 0, max_norm=getattr(solver.cfg, "clip_max_norm", 0.1),
        ema=solver.ema, scaler=solver.scaler, lr_warmup_scheduler=solver.lr_warmup_scheduler, d=d)
    print("SMOKE OK", stats)
except Exception:
    traceback.print_exc()
    sys.exit(1)
