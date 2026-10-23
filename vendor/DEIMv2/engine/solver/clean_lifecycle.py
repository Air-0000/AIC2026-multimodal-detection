"""Monotonic training lifecycle for the corrected baseline (opt-in).

Never restores a best checkpoint mid-run. Saves the measured EMA and RNG state
after validation; final.pth always identifies the configured final epoch.
"""
import json
import os
from pathlib import Path

import torch

from ..misc import dist_utils
from ..optim.lr_scheduler import FlatCosineLRScheduler
from .det_engine import train_one_epoch, evaluate


def atomic_checkpoint(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def fit_clean(solver):
    solver.train()
    cfg = solver.cfg
    if (cfg.use_amp and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8) or not cfg.stop_on_nonfinite or not cfg.strict_resume:
        raise ValueError('Clean lifecycle requires FP32 on legacy SM < 8.0, stop_on_nonfinite and strict_resume')
    solver.self_lr_scheduler = True
    solver.lr_scheduler = FlatCosineLRScheduler(
        solver.optimizer, cfg.lr_gamma, len(solver.train_dataloader), cfg.epoches,
        cfg.warmup_iter, cfg.flat_epoch, cfg.no_aug_epoch)
    solver.best_stat = getattr(solver, 'best_stat', {})
    for epoch in range(solver.last_epoch + 1, cfg.epoches):
        solver.train_dataloader.set_epoch(epoch)
        if dist_utils.is_dist_available_and_initialized():
            solver.train_dataloader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            True, solver.lr_scheduler, solver.model, solver.criterion,
            solver.train_dataloader, solver.optimizer, solver.device, epoch,
            max_norm=cfg.clip_max_norm, print_freq=cfg.print_freq,
            ema=solver.ema, scaler=solver.scaler, writer=solver.writer,
            freeze_bn=cfg.freeze_bn, accum_steps=cfg.accum_steps,
            stop_on_nonfinite=True)
        module = solver.ema.module if solver.ema else solver.model
        test_stats, coco_evaluator = evaluate(
            module, solver.criterion, solver.postprocessor, solver.val_dataloader,
            solver.evaluator, solver.device, stop_on_nonfinite=True)
        solver.last_epoch = epoch
        score = test_stats['coco_eval_bbox'][0]
        improved = score > solver.best_stat.get('mAP', float('-inf'))
        if improved:
            solver.best_stat = dict(mAP=score, epoch=epoch)
        if dist_utils.is_main_process():
            state = solver.state_dict()
            state['validation_metrics'] = test_stats
            state['effective_seed'] = cfg.seed
            atomic_checkpoint(state, solver.output_dir / 'last.pth')
            if (epoch + 1) % cfg.checkpoint_freq == 0 or epoch == cfg.epoches - 1:
                atomic_checkpoint(state, solver.output_dir / f'checkpoint{epoch:04d}.pth')
            if improved:
                atomic_checkpoint(state, solver.output_dir / 'best.pth')
            if epoch == cfg.epoches - 1:
                # A hardlink avoids a second multi-GB serialization. Atomic
                # replacement of last.pth on a future run cannot alter final.pth.
                temporary = solver.output_dir / 'final.pth.tmp'
                if temporary.exists():
                    temporary.unlink()
                os.link(solver.output_dir / 'last.pth', temporary)
                os.replace(temporary, solver.output_dir / 'final.pth')
            with (solver.output_dir / 'log.txt').open('a') as stream:
                row = {**{f'train_{k}': v for k, v in train_stats.items()},
                       **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch,
                       'seed': cfg.seed, 'best_stat': solver.best_stat}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            (solver.output_dir / 'eval').mkdir(exist_ok=True)
            atomic_checkpoint(coco_evaluator.coco_eval['bbox'].eval,
                              solver.output_dir / 'eval/latest.pth')
