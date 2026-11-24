"""AIC2026 Co-DINO 用的 mmengine 钩子（10-26，卡 A）。

- AicParamCheckHook：before_train（此时 load_from 已载入）检查全部参数有限，打印分类头统计；有非有限值直接报错退出。
- AicIterRecorderHook：训练迭代记 JSONL（全部损失、单步耗时、显存峰值 allocated/reserved），用于 15 步诊断与训练监控。
  10-28 改：原来遇到一次非有限主损失就报错退出——卡 E 的 s20260920 在第 11 轮第 1850 步抽到一张无真值的图，
  辅助 ATSS 头的 loss_cls1 为 NaN（其余项正常），fp16 AMP 本会跳过这一步，却被本钩子杀掉了整条训练。
  现在：单次非有限只记警告（AMP 的 GradScaler 会跳过这一步更新），最近 bad_window 步里非有限达到 max_bad 次才判为发散并报错；
  另外只全量记录前 full_log_iters 步，之后每 log_every 步记一行（非有限的步一定记），学习率只记 min/max
  （原来每步写上百个参数组的 lr，一条全量训练的日志到 0.4 GB）。参数缺省，旧配置不用改。
- AicEarlyGateHook：每次验证后把指标追加到 JSONL；在 gate_epoch（已完成的训练轮数，2 = epoch index 1）时
  若 key < thr 则写 GATE_FAIL 标记并停止训练。
"""

from __future__ import annotations

import collections
import json
import math
import os
import time

import torch
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


def _f(v):
    try:
        if isinstance(v, torch.Tensor):
            return float(v.detach().float().mean().item())
        return float(v)
    except Exception:
        return None


@HOOKS.register_module()
class AicParamCheckHook(Hook):
    priority = 'VERY_HIGH'

    def before_train(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        bad = []
        n_total = 0
        for name, p in model.named_parameters():
            n_total += p.numel()
            if not torch.isfinite(p).all():
                bad.append(name)
        for name, b in model.named_buffers():
            if b.is_floating_point() and not torch.isfinite(b).all():
                bad.append('buffer:' + name)
        runner.logger.info(f'[AicParamCheck] params={n_total} nonfinite={len(bad)} {bad[:10]}')
        for name, p in model.named_parameters():
            if ('cls_branches' in name or 'fc_cls' in name or 'atss_cls' in name or 'label_embedding' in name) and (
                    name.endswith('weight') or name.endswith('bias')):
                runner.logger.info(
                    f'[AicParamCheck] {name} shape={tuple(p.shape)} mean={p.float().mean().item():.4g} '
                    f'std={p.float().std().item() if p.numel() > 1 else 0:.4g}')
        if bad:
            raise RuntimeError(f'non-finite parameters after load: {bad[:20]}')


@HOOKS.register_module()
class AicIterRecorderHook(Hook):
    priority = 'LOW'

    def __init__(self, out_file: str, main_key: str = 'loss', full_log_iters: int = 300, log_every: int = 50,
                 bad_window: int = 200, max_bad: int = 50) -> None:
        self.out_file = out_file
        self.main_key = main_key
        self.full_log_iters = full_log_iters
        self.log_every = log_every
        self.max_bad = max_bad
        self._bad = collections.deque(maxlen=bad_window)
        self._t0 = None

    def before_train_iter(self, runner, batch_idx, data_batch=None) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0 if self._t0 else None
        rec = {'iter': runner.iter, 'epoch': runner.epoch, 'step_time': dt}
        if torch.cuda.is_available():
            rec['mem_alloc_gb'] = torch.cuda.max_memory_allocated() / 2**30
            rec['mem_reserved_gb'] = torch.cuda.max_memory_reserved() / 2**30
        try:
            shp = data_batch['inputs'][0].shape
            rec['input_hw'] = [int(shp[-2]), int(shp[-1])]
        except Exception:
            pass
        if outputs:
            for k, v in outputs.items():
                rec[k] = _f(v)
        try:
            lr = runner.optim_wrapper.get_lr().get('lr', [])
            rec['lr_min_max'] = [min(lr), max(lr)] if lr else None
        except Exception:
            pass
        main = rec.get(self.main_key)
        bad = main is not None and not math.isfinite(main)
        self._bad.append(bad)
        if runner.iter < self.full_log_iters or runner.iter % self.log_every == 0 or bad:
            os.makedirs(os.path.dirname(self.out_file) or '.', exist_ok=True)
            with open(self.out_file, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
        if bad:
            n_bad = sum(self._bad)
            nonfinite = sorted(k for k, v in rec.items() if isinstance(v, float) and not math.isfinite(v))
            runner.logger.warning(f'[AicIterRecorder] non-finite {self.main_key} at iter {runner.iter} '
                                  f'(terms {nonfinite[:8]}); {n_bad} in last {len(self._bad)} iters, AMP skips this step')
            if n_bad >= self.max_bad:
                raise RuntimeError(f'diverged: {n_bad} non-finite {self.main_key} in last {len(self._bad)} iters '
                                   f'(latest iter {runner.iter}, terms {nonfinite[:8]})')


@HOOKS.register_module()
class AicEarlyGateHook(Hook):
    priority = 'LOWEST'

    def __init__(self, metrics_file: str, gate_epoch: int = 2, key: str = 'coco/bbox_mAP', thr: float = 0.20,
                 marker: str = '') -> None:
        self.metrics_file = metrics_file
        self.gate_epoch = gate_epoch
        self.key = key
        self.thr = thr
        self.marker = marker

    def after_val_epoch(self, runner, metrics=None) -> None:
        metrics = metrics or {}
        rec = {'finished_epochs': runner.epoch, 'epoch_index': runner.epoch - 1, 'iter': runner.iter,
               'time': time.strftime('%Y-%m-%d %H:%M:%S'),
               'metrics': {k: _f(v) for k, v in metrics.items()}}
        with open(self.metrics_file, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        v = rec['metrics'].get(self.key)
        if runner.epoch == self.gate_epoch and v is not None and v < self.thr:
            msg = f'[AicEarlyGate] epoch_index {runner.epoch - 1}: {self.key}={v:.4f} < {self.thr} -> stop training'
            runner.logger.warning(msg)
            if self.marker:
                with open(self.marker, 'w') as fh:
                    fh.write(msg + '\n')
            runner.train_loop.stop_training = True
