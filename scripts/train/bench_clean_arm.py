#!/usr/bin/env python3
"""Measure time per training step and peak GPU memory of one clean-baseline arm, saving nothing.

Builds the solver exactly as train.py does for a fresh run (COCO tuning, seed, FP32, TF32 off, cuDNN off
only on Volta) and runs the real train_one_epoch for --iters steps of epoch 0. The optimizer steps happen
in memory only; checkpoints, logs and TensorBoard files go to a temporary folder that is deleted.
Prints JSON (first --warmup steps excluded): median seconds per step and its compute / loader-wait parts,
peak memory, and the projected hours for the full schedule excluding per-epoch validation.
Usage (repo root): CUDA_VISIBLE_DEVICES=0 python scripts/train/bench_clean_arm.py --config configs/deimv2/aic_clean_baseline_v1.yml
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT/'vendor/DEIMv2'


class Timed:
    """Loader view: keeps len() (LR schedule), stops after n steps, records per-step and loader-wait times."""

    def __init__(self, loader, n):
        self.loader, self.n, self.step_ends, self.waits = loader, n, [], []

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        import torch
        it = iter(self.loader)
        for i in range(self.n):
            torch.cuda.synchronize()
            t = time.time()
            if self.step_ends:
                self.step_ends[-1] = t - self.step_ends[-1]
            batch = next(it)
            self.waits.append(time.time() - t)
            self.step_ends.append(time.time())
            yield batch
        torch.cuda.synchronize()
        self.step_ends[-1] = time.time() - self.step_ends[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='relative to vendor/DEIMv2')
    p.add_argument('--iters', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    a = p.parse_args()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # as run_clean_arm.sh
    sys.path[:0] = [str(FW), str(ROOT/'scripts/lib')]
    os.chdir(FW)
    import torch
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.optim.lr_scheduler import FlatCosineLRScheduler
    from engine.solver import TASKS
    from engine.solver.det_engine import train_one_epoch

    with tempfile.TemporaryDirectory(prefix='bench_') as tmp:
        cfg = YAMLConfig(a.config, output_dir=f'{tmp}/out', summary_dir=f'{tmp}/tb')
        cfg.seed = 0 if cfg.seed is None else cfg.seed
        dist_utils.setup_distributed(0, 'builtin', seed=cfg.seed, strict=cfg.clean_baseline)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if torch.cuda.get_device_capability() == (7, 0):
            torch.backends.cudnn.enabled = False
        solver = TASKS[cfg.yaml_cfg['task']](cfg)
        solver.train()
        lr = FlatCosineLRScheduler(solver.optimizer, cfg.lr_gamma, len(solver.train_dataloader), cfg.epoches,
                                   cfg.warmup_iter, cfg.flat_epoch, cfg.no_aug_epoch)
        solver.train_dataloader.set_epoch(0)
        torch.cuda.reset_peak_memory_stats()
        timed = Timed(solver.train_dataloader, a.iters)
        train_one_epoch(True, lr, solver.model, solver.criterion, timed, solver.optimizer, solver.device, 0,
                        max_norm=cfg.clip_max_norm, print_freq=10, ema=solver.ema, scaler=solver.scaler, writer=None,
                        freeze_bn=cfg.freeze_bn, accum_steps=cfg.accum_steps, stop_on_nonfinite=True)
        compute, wait = timed.step_ends[a.warmup:], timed.waits[a.warmup:]
        per_step = statistics.median(c + w for c, w in zip(compute, wait))
        report = dict(config=a.config, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      cudnn_enabled=torch.backends.cudnn.enabled, iters=a.iters, warmup=a.warmup,
                      sec_per_step_median=round(per_step, 3), compute_median=round(statistics.median(compute), 3),
                      loader_wait_median=round(statistics.median(wait), 3),
                      peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), steps_per_epoch=len(solver.train_dataloader),
                      projected_hours_excl_validation=round(per_step * len(solver.train_dataloader) * cfg.epoches / 3600, 1))
    print('BENCH ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
