"""
DEIM: Online Cross-Modal Knowledge Distillation Engine
Teacher: Frozen Pure RGB 56.71% Model
Student: Sovereign Multimodal Model (RGB + Aligned IR + LBG IR + 16-bit Depth)
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from .det_engine import evaluate


def train_one_epoch_distill(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                            data_loader: Iterable, optimizer: torch.optim.Optimizer,
                            device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()

    teacher_model = kwargs.get('teacher_model', None)
    if teacher_model is not None:
        teacher_model.eval()

    if kwargs.get('freeze_bn', False):
        from torch.nn import BatchNorm2d, SyncBatchNorm
        _m = model.module if hasattr(model, 'module') else model
        for _mod in _m.modules():
            if isinstance(_mod, (BatchNorm2d, SyncBatchNorm)):
                _mod.eval()
                if _mod.weight is not None:
                    _mod.weight.requires_grad_(False)
                if _mod.bias is not None:
                    _mod.bias.requires_grad_(False)
        print('[freeze_bn] frozen all BatchNorm layers (using running stats during training)')

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer: SummaryWriter = kwargs.get('writer', None)

    ema: ModelEMA = kwargs.get('ema', None)
    scaler: GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)

    accum_steps = max(1, int(kwargs.get('accum_steps', 1)))
    cur_iters = epoch * len(data_loader)

    # Distillation hyperparameters
    kd_temp = float(kwargs.get('kd_temp', 2.0))
    kd_cls_w = float(kwargs.get('kd_cls_w', 1.2))
    kd_box_w = float(kwargs.get('kd_box_w', 0.6))
    kd_conf_thresh = float(kwargs.get('kd_conf_thresh', 0.25))

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        B = samples.shape[0]
        micro_bs = max(1, B // accum_steps)
        n_micro = (B + micro_bs - 1) // micro_bs

        loss_accum = 0.0
        n_valid = 0
        last_loss_dict = None
        if scaler is None:
            optimizer.zero_grad()

        for a in range(n_micro):
            s_micro = samples[a * micro_bs:(a + 1) * micro_bs]
            t_micro = targets[a * micro_bs:(a + 1) * micro_bs]

            # Forward Teacher (Frozen, pure RGB input: s_micro[:, :3])
            teacher_outputs = None
            if teacher_model is not None:
                with torch.no_grad():
                    with torch.autocast(device_type=str(device), dtype=torch.float16, cache_enabled=True):
                        teacher_outputs = teacher_model(s_micro[:, :3])

            if scaler is not None:
                with torch.autocast(device_type=str(device), dtype=torch.float16, cache_enabled=True):
                    outputs = model(s_micro, targets=t_micro)
                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, t_micro, **metas)

                    # Distillation: Mismatched Query-level KD is permanently disabled due to DETR set prediction query permutation
                    # True DETR distillation requires Hungarian/BBox matching or dense Encoder FPN feature distillation.
                    if False and teacher_outputs is not None:
                        pass

                loss = sum(loss_dict.values())
                scaler.scale(loss / n_micro).backward()
            else:
                outputs = model(s_micro, targets=t_micro)
                loss_dict = criterion(outputs, t_micro, **metas)
                if False and teacher_outputs is not None:
                    pass

                loss = sum(loss_dict.values())
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    continue
                (loss / n_micro).backward()

            loss_accum += float(loss)
            n_valid += 1
            last_loss_dict = loss_dict

        if n_valid == 0:
            optimizer.zero_grad()
            continue

        if max_norm > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(last_loss_dict) if last_loss_dict is not None else {'loss': torch.tensor(loss_accum / n_valid)}
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
