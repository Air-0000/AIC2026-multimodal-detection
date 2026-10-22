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
from ..misc.numerical_contract import require_finite


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
    strict = bool(kwargs.get('stop_on_nonfinite', False))
    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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
        optimizer.zero_grad()
        if strict and self_lr_scheduler:
            # Apply the intended LR to this update, rather than the next one.
            optimizer = lr_scheduler.step(cur_iters + i + 1, optimizer)

        for a in range(n_micro):
            s_micro = samples[a*micro_bs:(a+1)*micro_bs].to(device, non_blocking=True)
            t_micro = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets[a*micro_bs:(a+1)*micro_bs]]
            micro_weight = s_micro.shape[0] / B if strict else 1.0 / n_micro
            if strict:
                require_finite([('samples', s_micro)], f'epoch={epoch} iter={i} micro={a} input')

            if scaler is not None:
                amp_dtype = torch.float16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8) else torch.bfloat16
                with torch.autocast(device_type=str(device), dtype=amp_dtype, cache_enabled=True):
                    outputs = model(s_micro, targets=t_micro)
                if strict:
                    require_finite(outputs.items(), f'epoch={epoch} iter={i} micro={a} forward')
                has_invalid = False
                for k in ('pred_boxes', 'pred_logits'):
                    if k in outputs and (torch.isnan(outputs[k]).any() or torch.isinf(outputs[k]).any()):
                        has_invalid = True
                        break
                if has_invalid:
                    print(f'[NaN-guard] iter {global_step} micro {a}: pred NaN/Inf -> clean buffers and skip micro backward')
                    for b in model.buffers():
                        if torch.isnan(b).any() or torch.isinf(b).any():
                            b.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
                    del outputs, s_micro, t_micro
                    continue
                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, t_micro, **metas)
                loss = sum(loss_dict.values())
                if strict:
                    require_finite([*loss_dict.items(), ('total_loss', loss)], f'epoch={epoch} iter={i} micro={a} loss')
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f'[NaN-guard] iter {global_step} micro {a}: loss NaN/Inf -> clean buffers and skip micro backward')
                    for b in model.buffers():
                        if torch.isnan(b).any() or torch.isinf(b).any():
                            b.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
                    del outputs, loss_dict, loss, s_micro, t_micro
                    continue
                scaler.scale(loss * micro_weight).backward()
            else:
                outputs = model(s_micro, targets=t_micro)
                if strict:
                    require_finite(outputs.items(), f'epoch={epoch} iter={i} micro={a} forward')
                has_invalid = False
                for k in ('pred_boxes', 'pred_logits'):
                    if k in outputs and (torch.isnan(outputs[k]).any() or torch.isinf(outputs[k]).any()):
                        has_invalid = True
                        nan_c = torch.isnan(outputs[k]).sum().item()
                        inf_c = torch.isinf(outputs[k]).sum().item()
                        print(f'[NaN-guard FP32 DETECTED] iter {global_step} micro {a} key={k}: nan={nan_c}, inf={inf_c}')
                if has_invalid:
                    print(f'[NaN-guard FP32] iter {global_step} micro {a}: pred NaN/Inf -> clean buffers and skip micro backward')
                    for b in model.buffers():
                        if torch.isnan(b).any() or torch.isinf(b).any():
                            b.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
                    del outputs, s_micro, t_micro
                    continue
                loss_dict = criterion(outputs, t_micro, **metas)
                loss = sum(loss_dict.values())
                if strict:
                    require_finite([*loss_dict.items(), ('total_loss', loss)], f'epoch={epoch} iter={i} micro={a} loss')
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f'[NaN-guard FP32] iter {global_step} micro {a}: loss NaN/Inf -> clean buffers and skip micro backward')
                    for b in model.buffers():
                        if torch.isnan(b).any() or torch.isinf(b).any():
                            b.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
                    del outputs, loss_dict, loss, s_micro, t_micro
                    continue
                (loss * micro_weight).backward()

            loss_accum += float(loss.item())
            n_valid += 1
            last_loss_dict = {k: v.detach() for k, v in loss_dict.items()}
            del outputs, loss_dict, loss, s_micro, t_micro
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if n_valid == 0:
            print(f'[NaN-guard] iter {global_step}: ALL micro-batches invalid, skipping step')
            optimizer.zero_grad()
            continue

        # Diagnose grad before step and guard against corrupted weight updates
        if strict:
            if scaler is not None:
                scaler.unscale_(optimizer)
            require_finite(((name, p.grad) for name, p in model.named_parameters() if p.grad is not None),
                           f'epoch={epoch} iter={i} gradient')
            require_finite(model.named_buffers(), f'epoch={epoch} iter={i} buffers', allow_anchor_infinity=True)
        has_invalid_grad = False
        for name, p in (() if strict else model.named_parameters()):
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                print(f'[PRE-STEP GRAD ALERT] {name}: grad has NaN/Inf! Skipping step to preserve weights.')
                has_invalid_grad = True
                break

        if has_invalid_grad:
            optimizer.zero_grad()
            continue

        if max_norm > 0:
            if scaler is not None and not strict:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=strict)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        if strict:
            require_finite(model.named_parameters(), f'epoch={epoch} iter={i} post-step parameter')
            require_finite(model.named_buffers(), f'epoch={epoch} iter={i} post-step buffer', allow_anchor_infinity=True)

        # Diagnose params after step
        for name, p in (() if strict else model.named_parameters()):
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f'[POST-STEP PARAM ALERT] {name}: parameter became NaN/Inf!')
        for name, b in (() if strict else model.named_buffers()):
            if 'decoder.anchors' not in name and (torch.isnan(b).any() or torch.isinf(b).any()):
                print(f'[POST-STEP BUFFER ALERT] {name}: buffer became NaN/Inf -> sanitized to 0.0')
                b.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            if not strict:
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
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, stop_on_nonfinite=False):
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
        if stop_on_nonfinite:
            require_finite(outputs.items(), 'validation forward')

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes, samples)
        if stop_on_nonfinite:
            for result in results:
                require_finite(result.items(), 'validation postprocessor')

        for r in (() if stop_on_nonfinite else results):
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
