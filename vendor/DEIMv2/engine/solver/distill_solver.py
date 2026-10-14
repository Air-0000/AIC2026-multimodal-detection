"""
DEIM: Online Cross-Modal Knowledge Distillation Solver
Teacher: Frozen Pure RGB 56.71% Model (puremal_refine_giou35)
Student: Sovereign Multimodal Model (RGB + Aligned IR + LBG IR + 16-bit Depth)
"""

import os
import time
import json
import datetime
from pathlib import Path

import torch

from ..misc import dist_utils, stats
from .det_solver import DetSolver
from .distill_engine import train_one_epoch_distill, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ..core import YAMLConfig


class DistillDetSolver(DetSolver):

    def _setup(self):
        super()._setup()

        # Set student decoder eval_spatial_size to None to avoid static anchor NaNs/Infs
        if hasattr(self.model, 'decoder'):
            self.model.decoder.eval_spatial_size = None
            print('[Distill] Student decoder eval_spatial_size set to None for dynamic grid anchor generation.')

        # Load and freeze Teacher Model
        teacher_cfg_path = getattr(self.cfg, 'teacher_cfg', None) or self.cfg.yaml_cfg.get('teacher_cfg', 'configs/deimv2/deim_hgnetv2_x_aic_puremal_refine_giou35.yml')
        teacher_weight_path = getattr(self.cfg, 'teacher_weight', None) or self.cfg.yaml_cfg.get('teacher_weight', 'outputs/deim_hgnetv2_x_aic_puremal_refine_giou35/best_stg1.pth')

        self.teacher_model = None
        if teacher_cfg_path and os.path.exists(teacher_cfg_path):
            print(f'[Distill] Building Teacher Model from config: {teacher_cfg_path}')
            t_cfg = YAMLConfig(teacher_cfg_path)
            self.teacher_model = t_cfg.model
            if teacher_weight_path and os.path.exists(teacher_weight_path):
                print(f'[Distill] Loading Teacher Weights from: {teacher_weight_path}')
                ckpt = torch.load(teacher_weight_path, map_location='cpu')
                if isinstance(ckpt, dict) and 'ema' in ckpt and 'module' in ckpt['ema']:
                    state_dict = ckpt['ema']['module']
                elif isinstance(ckpt, dict) and 'model' in ckpt:
                    state_dict = ckpt['model']
                else:
                    state_dict = ckpt
                msg = self.teacher_model.load_state_dict(state_dict, strict=False)
                print(f'[Distill] Teacher loaded successfully with message: {msg}')
            else:
                print(f'[Distill Warning] Teacher weight {teacher_weight_path} not found!')

            self.teacher_model = self.teacher_model.to(self.device)
            self.teacher_model.eval()
            if hasattr(self.teacher_model, 'decoder'):
                self.teacher_model.decoder.eval_spatial_size = None
                print('[Distill] Teacher decoder eval_spatial_size set to None for dynamic multi-scale distillation.')
            if hasattr(self.teacher_model, 'use_internal_sahi'):
                self.teacher_model.use_internal_sahi = False
                print('[Distill] Teacher internal SAHI disabled during training.')
            for p in self.teacher_model.parameters():
                p.requires_grad = False
            print(f'[Distill] Teacher model frozen in eval mode on {self.device}.')
        else:
            print(f'[Distill Warning] Teacher config {teacher_cfg_path} not found!')

        self.distill_kwargs = {
            'kd_temp': getattr(self.cfg, 'kd_temp', 2.0),
            'kd_cls_w': getattr(self.cfg, 'kd_cls_w', 1.2),
            'kd_box_w': getattr(self.cfg, 'kd_box_w', 0.6),
            'kd_conf_thresh': getattr(self.cfg, 'kd_conf_thresh', 0.25),
        }

    def fit(self):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 38 + " Start Distillation Training " + "-" * 39)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                args.lr_gamma,
                iter_per_epoch,
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter,
                flat_epochs=args.flat_epoch,
                no_aug_epochs=args.no_aug_epoch
            )
            self.self_lr_scheduler = True

        n_trainable = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'[Distill Student] Number of trainable parameters: {n_trainable / 1e6:.2f}M')

        top1 = 0
        best_stat = {'epoch': -1}
        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1

        for epoch in range(start_epoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch and epoch > 0 and (self.output_dir / 'best_stg1.pth').exists():
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch_distill(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                freeze_bn=getattr(args, 'freeze_bn', False),
                accum_steps=getattr(args, 'accum_steps', 1),
                teacher_model=self.teacher_model,
                **self.distill_kwargs
            )

            if not self.self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

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
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Distillation Training time {}'.format(total_time_str))
