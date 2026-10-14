"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
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
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    accum_steps = max(1, int(kwargs.get('accum_steps', 1)))
    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        # ---- gradient accumulation: split the batch into micro-batches ----
        # The desktop Wayland compositor (niri) hogs ~12GB of the 16GB GPU, leaving
        # too little headroom for a full bs4 forward+backward autograd graph. We
        # accumulate gradients over `accum_steps` micro-batches of size ~1 so the
        # per-step peak memory stays small while the *effective* batch size is kept.
        B = samples.shape[0]
        micro_bs = max(1, B // accum_steps)
        n_micro = (B + micro_bs - 1) // micro_bs

        loss_accum = 0.0
        n_valid = 0
        last_loss_dict = None
        if scaler is None:
            optimizer.zero_grad()

        for a in range(n_micro):
            s_micro = samples[a*micro_bs:(a+1)*micro_bs]
            t_micro = targets[a*micro_bs:(a+1)*micro_bs]

            if scaler is not None:
                amp_dtype = torch.float16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8) else torch.bfloat16
                with torch.autocast(device_type=str(device), dtype=amp_dtype, cache_enabled=True):
                    outputs = model(s_micro, targets=t_micro)
                has_invalid = False
                for k in ('pred_boxes', 'pred_logits'):
                    if k in outputs and (torch.isnan(outputs[k]).any() or torch.isinf(outputs[k]).any()):
                        has_invalid = True
                        break
                if has_invalid:
                    print(f'[NaN-guard] iter {global_step} micro {a}: pred NaN/Inf -> skip micro backward')
                    continue
                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, t_micro, **metas)
                loss = sum(loss_dict.values())
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f'[NaN-guard] iter {global_step} micro {a}: loss NaN/Inf -> skip micro backward')
                    continue
                scaler.scale(loss / n_micro).backward()
            else:
                outputs = model(s_micro, targets=t_micro)
                has_invalid = False
                for k in ('pred_boxes', 'pred_logits'):
                    if k in outputs and (torch.isnan(outputs[k]).any() or torch.isinf(outputs[k]).any()):
                        has_invalid = True
                        break
                if has_invalid:
                    continue
                loss_dict = criterion(outputs, t_micro, **metas)
                loss = sum(loss_dict.values())
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    continue
                (loss / n_micro).backward()

            loss_accum += float(loss)
            n_valid += 1
            last_loss_dict = loss_dict

        if n_valid == 0:
            # every micro-batch was NaN -> nothing to optimize this iter
            optimizer.zero_grad()
            continue

        if max_norm > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes, samples)

        for r in results:
            if 'scores' in r and isinstance(r['scores'], torch.Tensor):
                r['scores'] = torch.nan_to_num(r['scores'], nan=0.0, posinf=1.0, neginf=0.0)
            if 'boxes' in r and isinstance(r['boxes'], torch.Tensor):
                r['boxes'] = torch.nan_to_num(r['boxes'], nan=0.0, posinf=1.0, neginf=0.0)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
