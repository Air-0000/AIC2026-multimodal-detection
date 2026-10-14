"""
DNDT (Debiased Teacher for Day-to-Night Domain Adaptive Object Detection,
Cui et al. ICCV 2025) self-training solver, ported onto the DEIMv2 detector.

The paper attacks the three biases of a day->night self-training loop with:
  * DNDT  : a parameter-free physical-prior day->night transform (we already have
            this as the online augmentation RandomDNDTNightify; the SOURCE loader
            applies it so labelled daytime frames become night-like).
  * CDRC  : Cross-Domain Representation Compensating -- bidirectionally mix objects
            from real-night and night-like images (with ICFB class-frequency
            balancing) so the student sees night-domain objects in both contexts.
  * ConCal: Pseudo-label Confirmation Calibrating -- teacher (EMA) produces pseudo
            labels on REAL night; we keep a pseudo box only when the student agrees
            (teacher-student joint prediction) AND its score clears a class threshold,
            which corrects the confirmation bias from noisy pseudo labels.

Framework reuse:
  * model / criterion / postprocessor / ema / optimizer / lr-scheduler are all built
    by the standard YAMLConfig + BaseSolver machinery (we only override fit()).
  * The "teacher" is simply the EMA copy of the student (ModelEMA), updated every step.
  * The target (real night) loader is an extra dataloader (`target_dataloader`) whose
    boxes are IGNORED for training; only its images + orig_size are used.
"""
import sys
import math
import time
import json
import datetime
import random
import itertools

import torch
import torch.nn as nn
import torch.amp
import torchvision

from ..misc import dist_utils, stats
from ..optim import ModelEMA
from ..data import CocoEvaluator
from ..core import register
from .det_solver import DetSolver
from .det_engine import evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _empty_target(orig_size, device):
    return {
        'boxes': torch.zeros((0, 4), device=device),
        'labels': torch.zeros((0,), dtype=torch.int64, device=device),
        'image_id': torch.zeros((1,), dtype=torch.int64, device=device),
        'orig_size': orig_size.to(device) if isinstance(orig_size, torch.Tensor) else torch.tensor(orig_size, device=device),
        'area': torch.zeros((0,), device=device),
        'iscrowd': torch.zeros((0,), dtype=torch.int64, device=device),
    }


def _index_outputs(out, idx):
    """Index every batch-dim-0 tensor in a model outputs dict by `idx`."""
    B = out['pred_logits'].shape[0]
    return {k: (v[idx] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B else v)
            for k, v in out.items()}


class _Zip:
    """zip() with __len__ (taken from the source loader) so MetricLogger.log_every works."""
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __len__(self):
        return len(self.a)

    def __iter__(self):
        return zip(self.a, self.b)


def cdrc_mix(src_imgs, src_targets, tgt_imgs, num_classes, img_size=640,
             max_obj=4, icfb=True, p_mix=1.0):
    """CDRC: paste a few labelled SOURCE (night-like) objects into REAL-NIGHT scenes.

    Returns (mixed_imgs [B,3,H,W], mixed_targets) where mixed_targets carries the
    pasted objects' boxes (cxcywh normalised) + labels -- a fresh SUPERVISED sample
    that puts night-domain objects (with clean labels) into real-night contexts,
    compensating the model's night-object representation. ICFB upsamples rare
    classes within the batch.
    """
    B, _, sH, sW = src_imgs.shape          # crop lives in the SOURCE space
    _, _, tH, tW = tgt_imgs.shape           # paste lives in the MIXED (target) space
    mixed = tgt_imgs.clone()
    mixed_targets = []
    for b in range(B):
        tgts = src_targets[b]
        boxes = tgts['boxes']
        labels = tgts['labels']
        boxes_cpu = boxes.cpu()
        labels_cpu = labels.cpu()
        n = int(boxes_cpu.shape[0])
        if n == 0 or random.random() > p_mix:
            mixed_targets.append(_empty_target(torch.tensor([tH, tW], device=boxes.device), boxes.device))
            continue
        # ICFB: inverse within-batch class frequency (CPU, avoids per-op GPU sync)
        if icfb:
            cnt = torch.bincount(labels_cpu, minlength=num_classes).float().clamp(min=1)
            w = (1.0 / cnt[labels_cpu]); w = w / w.sum()
        else:
            w = torch.ones(n, device=boxes_cpu.device) / n
        k = min(max_obj, n)
        sel = torch.multinomial(w, k, replacement=False)
        new_boxes, new_labels = [], []
        for ji in sel.tolist():
            box = boxes_cpu[ji]
            xyxy = torchvision.ops.box_convert(box.unsqueeze(0), 'cxcywh', 'xyxy').squeeze(0) * sH
            x1, y1, x2, y2 = [int(v) for v in xyxy.clamp(0, sH).round().tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = src_imgs[b][:, y1:y2, x1:x2]
            if crop.shape[-2] == 0 or crop.shape[-1] == 0:
                continue
            ch, cw = crop.shape[-2:]
            scale = random.uniform(0.8, 1.25)
            nh, nw = max(1, int(ch * scale)), max(1, int(cw * scale))
            crop2 = torch.nn.functional.interpolate(
                crop.unsqueeze(0), size=(nh, nw), mode='bilinear', align_corners=False).squeeze(0)
            # clamp object to the target (mixed) canvas
            if crop2.shape[-2] > tH or crop2.shape[-1] > tW:
                crop2 = torch.nn.functional.interpolate(
                    crop2.unsqueeze(0), size=(min(crop2.shape[-2], tH), min(crop2.shape[-1], tW)),
                    mode='bilinear', align_corners=False).squeeze(0)
            rh, rw = crop2.shape[-2], crop2.shape[-1]
            px = random.randint(0, max(0, tW - rw))
            py = random.randint(0, max(0, tH - rh))
            mixed[b, :, py:py + rh, px:px + rw] = crop2
            nx1 = (px + 1) / tW
            ny1 = (py + 1) / tH
            nx2 = (px + rw - 1) / tW
            ny2 = (py + rh - 1) / tH
            nb = torch.tensor([(nx1 + nx2) / 2, (ny1 + ny2) / 2, (nx2 - nx1), (ny2 - ny1)],
                              device=boxes.device)
            new_boxes.append(nb)
            new_labels.append(labels[ji])
        if len(new_boxes) == 0:
            mixed_targets.append(_empty_target(torch.tensor([tH, tW], device=boxes.device), boxes.device))
        else:
            nb_ = torch.stack(new_boxes)
            nl_ = torch.stack(new_labels)
            area = (nb_[:, 2] * tW) * (nb_[:, 3] * tH)
            mixed_targets.append({
                'boxes': nb_,
                'labels': nl_,
                'image_id': tgts.get('image_id', torch.zeros((1,), dtype=torch.int64, device=boxes.device)),
                'orig_size': torch.tensor([tH, tW], device=boxes.device),
                'area': area,
                'iscrowd': torch.zeros((nl_.shape[0],), dtype=torch.int64, device=boxes.device),
            })
    return mixed, mixed_targets


def concal_filter(t_pseudo, s_pseudo, cls_thr, iou_thr=0.5):
    """ConCal: keep a teacher pseudo box only if the student predicts the SAME class
    at (IoU>=iou_thr) with a score above the (class) threshold, and the teacher score
    itself clears the threshold. Returns a list (one entry per image) of
    {labels, boxes(xyxy pix), scores} or None."""
    out = []
    for tp, sp in zip(t_pseudo, s_pseudo):
        device = tp['labels'].device
        tl = tp['labels'].cpu()
        tb = tp['boxes'].cpu()
        ts = tp['scores'].cpu()
        sl = sp['labels'].cpu()
        sb = sp['boxes'].cpu()
        ss = sp['scores'].cpu()
        keep = []
        for i in range(tl.shape[0]):
            c = int(tl[i])
            if ts[i].item() < cls_thr:
                continue
            m = (sl == c)
            if not m.any():
                continue
            ious = torchvision.ops.box_iou(tb[i:i + 1], sb[m])[0]
            j = int(ious.argmax())
            if ious[j].item() >= iou_thr and ss[m][j].item() >= cls_thr:
                keep.append(i)
        if keep:
            keep = torch.tensor(keep)
            out.append({'labels': tl[keep].to(device), 'boxes': tb[keep].to(device),
                        'scores': ts[keep].to(device)})
        else:
            out.append(None)
    return out


def make_pseudo_targets(filtered, orig_sizes, device):
    """Convert ConCal-filtered pseudo boxes (xyxy pixel) to criterion targets
    (cxcywh normalised). Images with no kept pseudo box get an empty target."""
    targets = []
    for b, f in enumerate(filtered):
        osz = orig_sizes[b]
        if f is None or f['labels'].shape[0] == 0:
            targets.append(_empty_target(osz, device))
        else:
            xyxy = f['boxes'].reshape(-1, 4)
            norm = xyxy / osz.to(device).repeat(2).clamp(min=1)
            cxcywh = torchvision.ops.box_convert(norm, 'xyxy', 'cxcywh')
            labels = f['labels'].long()
            area = (cxcywh[:, 2] * osz[0].item()) * (cxcywh[:, 3] * osz[1].item())
            targets.append({
                'boxes': cxcywh,
                'labels': labels,
                'image_id': torch.zeros((1,), dtype=torch.int64, device=device),
                'orig_size': osz.to(device),
                'area': area,
                'iscrowd': torch.zeros((labels.shape[0],), dtype=torch.int64, device=device),
            })
    return targets


# ---------------------------------------------------------------------------
# DNDT training step
# ---------------------------------------------------------------------------
def train_one_epoch_dndt(self_lr_scheduler, lr_scheduler, model, criterion, postprocessor,
                         src_loader, tgt_loader, optimizer, device, epoch, max_norm=0,
                         ema=None, scaler=None, lr_warmup_scheduler=None, writer=None,
                         d=None):
    model.train()
    criterion.train()
    if ema is not None:
        ema.module.eval()
    d = d or {}
    w_src = d.get('w_src', 1.0)
    w_cdrc = d.get('w_cdrc', 1.0)
    w_tgt = d.get('w_tgt', 1.0)
    cls_thr = d.get('cls_thr', 0.5)
    iou_thr = d.get('iou_thr', 0.5)
    max_obj = d.get('max_obj', 4)
    icfb = d.get('icfb', True)
    p_mix = d.get('p_mix', 1.0)
    num_classes = d.get('num_classes', 80)
    teacher = ema.module if ema is not None else None

    from ..misc import MetricLogger, SmoothedValue
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = d.get('print_freq', 10)
    cur_iters = epoch * len(src_loader)

    for i, (src, tgt) in enumerate(
            metric_logger.log_every(_Zip(src_loader, itertools.cycle(tgt_loader)), print_freq, header)):
        src_samples, src_targets = src
        tgt_samples, tgt_targets = tgt
        src_samples = src_samples.to(device)
        src_targets = [{k: v.to(device) for k, v in t.items()} for t in src_targets]
        tgt_samples = tgt_samples.to(device)
        tgt_targets = [{k: v.to(device) for k, v in t.items()} for t in tgt_targets]
        orig_sizes = torch.stack([t['orig_size'] for t in tgt_targets], dim=0).to(device)
        global_step = epoch * len(src_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(src_loader))

        do_amp = scaler is not None
        optimizer.zero_grad()

        # ---- 1) SOURCE supervised (day -> DNDT night-like) ----
        if do_amp:
            with torch.autocast(device_type=str(device), dtype=torch.bfloat16, cache_enabled=True):
                out_src = model(src_samples, targets=src_targets)
            with torch.autocast(device_type=str(device), enabled=False):
                loss_src = sum(criterion(out_src, src_targets, **metas).values())
        else:
            out_src = model(src_samples, targets=src_targets)
            loss_src = sum(criterion(out_src, src_targets, **metas).values())
        if do_amp:
            scaler.scale(loss_src).backward()
        else:
            loss_src.backward()

        # ---- 2) CDRC: mix source objects into real-night scenes -> supervised ----
        loss_cdrc = torch.zeros((), device=device)
        if w_cdrc > 0:
            mixed_imgs, mixed_tgts = cdrc_mix(
                src_samples, src_targets, tgt_samples, num_classes,
                img_size=src_samples.shape[-1], max_obj=max_obj, icfb=icfb, p_mix=p_mix)
            if sum(t['labels'].shape[0] for t in mixed_tgts) > 0:
                if do_amp:
                    with torch.autocast(device_type=str(device), dtype=torch.bfloat16, cache_enabled=True):
                        out_mix = model(mixed_imgs, targets=mixed_tgts)
                    with torch.autocast(device_type=str(device), enabled=False):
                        loss_cdrc = sum(criterion(out_mix, mixed_tgts, **metas).values())
                else:
                    loss_cdrc = sum(criterion(model(mixed_imgs, targets=mixed_tgts), mixed_tgts, **metas).values())
                if do_amp:
                    scaler.scale(loss_cdrc).backward()
                else:
                    loss_cdrc.backward()

        # ---- 3) TARGET self-training (ConCal) on REAL night ----
        loss_tgt = torch.zeros((), device=device)
        if w_tgt > 0 and teacher is not None:
            with torch.no_grad():
                t_out = teacher(tgt_samples)
                t_pseudo = postprocessor(t_out, orig_sizes, tgt_samples)
            # student forward needs (empty) targets so the DN group builds without crashing
            dn_targets = [_empty_target(tgt_targets[b].get('orig_size',
                             torch.tensor([src_samples.shape[-1], src_samples.shape[-1]])), device)
                          for b in range(src_samples.shape[0])]
            if do_amp:
                with torch.autocast(device_type=str(device), dtype=torch.bfloat16, cache_enabled=True):
                    s_out = model(tgt_samples, targets=dn_targets)
                with torch.no_grad():
                    s_pseudo = postprocessor(s_out, orig_sizes, tgt_samples)
                filtered = concal_filter(t_pseudo, s_pseudo, cls_thr, iou_thr)
                p_tgts = make_pseudo_targets(filtered, orig_sizes, device)
                keep = [b for b, f in enumerate(filtered) if f is not None and f['labels'].shape[0] > 0]
                if keep:
                    sub = _index_outputs(s_out, torch.tensor(keep, device=device))
                    with torch.autocast(device_type=str(device), enabled=False):
                        loss_tgt = sum(criterion(sub, [p_tgts[b] for b in keep], **metas).values())
                    scaler.scale(loss_tgt).backward()
            else:
                s_out = model(tgt_samples, targets=dn_targets)
                with torch.no_grad():
                    s_pseudo = postprocessor(s_out, orig_sizes, tgt_samples)
                filtered = concal_filter(t_pseudo, s_pseudo, cls_thr, iou_thr)
                p_tgts = make_pseudo_targets(filtered, orig_sizes, device)
                keep = [b for b, f in enumerate(filtered) if f is not None and f['labels'].shape[0] > 0]
                if keep:
                    sub = _index_outputs(s_out, torch.tensor(keep, device=device))
                    loss_tgt = sum(criterion(sub, [p_tgts[b] for b in keep], **metas).values())
                    loss_tgt.backward()

        loss = w_src * loss_src + w_cdrc * loss_cdrc + w_tgt * loss_tgt

        # optimizer step (gradients accumulated from the 3 sub-backwards above)
        if do_amp:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        optimizer.zero_grad()

        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        elif lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict({
            'loss_src': loss_src, 'loss_cdrc': loss_cdrc, 'loss_tgt': loss_tgt})
        loss_value = sum(loss_dict_reduced.values())
        if not math.isfinite(loss_value.item()):
            print("Loss is {}, stopping training".format(loss_value.item()))
            print(loss_dict_reduced)
            sys.exit(1)
        metric_logger.update(loss=loss, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for kk, vv in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{kk}', vv.item() if hasattr(vv, 'item') else vv, global_step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@register()
class DNDTSolver(DetSolver):
    def fit(self):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start DNDT training" + "-" * 43)
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch,
                                                      total_epochs=args.epoches, warmup_iter=args.warmup_iter,
                                                      flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True

        # extra target (real night, unlabeled) dataloader
        self.target_dataloader = dist_utils.warp_loader(
            self.cfg.build_dataloader('target_dataloader'), shuffle=True)
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        d = getattr(args, 'dndt', None)
        if d is None and hasattr(args, 'yaml_cfg'):
            d = args.yaml_cfg.get('dndt')
        d = d or {}
        d['num_classes'] = int(getattr(self.criterion, 'num_classes', 12))
        d['print_freq'] = getattr(args, 'print_freq', 10)
        d.setdefault('w_src', 1.0); d.setdefault('w_cdrc', 1.0); d.setdefault('w_tgt', 1.0)
        d.setdefault('cls_thr', 0.5); d.setdefault('iou_thr', 0.5)
        d.setdefault('max_obj', 4); d.setdefault('icfb', True); d.setdefault('p_mix', 1.0)
        print(f'DNDT hyperparams: {d}')

        top1 = 0
        best_stat = {'epoch': -1, }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                                                  self.val_dataloader, self.evaluator, self.device)
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            self.target_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
                self.target_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch_dndt(
                self.self_lr_scheduler, self.lr_scheduler, self.model, self.criterion, self.postprocessor,
                self.train_dataloader, self.target_dataloader, self.optimizer, self.device, epoch,
                max_norm=args.clip_max_norm, ema=self.ema, scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler, writer=self.writer, d=d)

            if not self.self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()
            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                                                  self.val_dataloader, self.evaluator, self.device)
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for ii, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{ii}'.format(k), v, epoch)
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')
                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')
                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch, 'n_parameters': n_parameters}
            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        (self.output_dir / 'eval' / 'latest.pth').touch()

        total_time = time.time() - start_time
        print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))
