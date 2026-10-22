#!/usr/bin/env python3
"""One full optimizer step at a chosen micro-batch: timing, VRAM, BN audit, init proof.

Loads the prepared warm-start through the framework's own tuning loader, then
runs accum forward/backward + one optimizer step with CUDA-synchronised timing.
Probe only: nothing is trained further, saved or evaluated.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


def patch_resize(node, size):
    if isinstance(node, dict):
        if str(node.get('type', '')) == 'Resize':
            node['size'] = list(size)
        for v in node.values():
            patch_resize(v, size)
    elif isinstance(node, list):
        for v in node:
            patch_resize(v, size)


FAMS = ('ir_proj', 'ir_align', 'occlusion_gate', 'coord_lift', 'ir_fuse', 'ir_res', 'bilateral_pool')


def bn_snapshot(model):
    snap = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_mean is not None:
            snap[name] = (m.running_mean.clone(), m.num_batches_tracked.clone()
                          if m.num_batches_tracked is not None else None)
    return snap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--warmstart', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--source-stages', default='2,3')
    p.add_argument('--template')
    p.add_argument('--stages', required=True)
    p.add_argument('--height', type=int, default=1440)
    p.add_argument('--width', type=int, default=2560)
    p.add_argument('--micro', type=int, default=1)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    from engine.core import YAMLConfig
    from engine.solver import TASKS

    cfg = YAMLConfig(args.config)
    y = cfg.yaml_cfg
    target_stages = [int(v) for v in args.stages.split(',')]
    y['HGNetv2MultiModal']['inject_stages'] = target_stages
    for split in ('train_dataloader', 'val_dataloader'):
        dl = y.get(split)
        if not dl:
            continue
        dl['total_batch_size'] = args.micro
        dl['num_workers'] = 0
        dl['persistent_workers'] = False
        patch_resize(dl.get('dataset'), (args.height, args.width))
    y.update(use_amp=False, accum_steps=args.accum, epoches=1,
             output_dir='/tmp/probe_v36_step', tuning=args.warmstart)
    cfg.tuning = args.warmstart   # __init__ snapshots this attribute; yaml_cfg alone is not enough

    t0 = time.time()
    solver = TASKS[y['task']](cfg)
    solver._setup()
    torch.cuda.synchronize()
    build_s = time.time() - t0
    model, criterion = solver.model, solver.criterion
    opt, ema = cfg.optimizer, getattr(solver, 'ema', None)
    resident = torch.cuda.memory_allocated() / 2 ** 30

    # ---- initialisation proof -------------------------------------------------
    cur = model.state_dict()
    src_stages = [int(v) for v in args.source_stages.split(',')]
    src = torch.load(args.source, map_location='cpu', mmap=True, weights_only=False)
    src = src.get('model', src)
    tpl = None
    if args.template:
        tpl = torch.load(args.template, map_location='cpu', mmap=True, weights_only=False)
        tpl = tpl.get('model', tpl)
    carried, fresh, fresh_mismatch = {}, {}, []
    stage_keys = set()
    for fam in FAMS:
        for i, stage in enumerate(target_stages):
            prefix = f'backbone.{fam}.{i}.'
            keys = [k for k in cur if k.startswith(prefix)]
            if not keys:
                continue
            stage_keys.update(keys)
            suffix = lambda k: k[len(prefix):]
            if stage in src_stages:
                sp = f'backbone.{fam}.{src_stages.index(stage)}.'
                d = [float((cur[k].float().cpu() - src[sp + suffix(k)].float().cpu()).abs().max())
                     for k in keys if sp + suffix(k) in src]
                carried[f'{fam}[{i}]<-stage{stage}'] = {'keys': len(d), 'max_abs': max(d) if d else None}
            else:
                n_ok = 0
                for k in keys:
                    tk = f'backbone.{fam}.0.{suffix(k)}'
                    if tpl is not None and tk in tpl:
                        if torch.equal(cur[k].float().cpu(), tpl[tk].float().cpu()):
                            n_ok += 1
                        else:
                            fresh_mismatch.append(k)
                fresh[f'{fam}[{i}]<-stage{stage}'] = {'keys': len(keys),
                                                      'matches_template': n_ok if tpl else None}
    shared_other = [k for k in cur if k not in stage_keys and k in src]
    other_diff = [k for k in shared_other
                  if not torch.equal(cur[k].float().cpu(), src[k].float().cpu())]
    init = {'carried_stage_modules': carried,
            'carried_nonzero': [k for k, v in carried.items() if v['max_abs'] not in (None, 0.0)],
            'fresh_stage_modules': fresh, 'fresh_mismatch_keys': fresh_mismatch[:8],
            'other_shared_keys': len(shared_other), 'other_shared_differing': len(other_diff),
            'other_shared_differing_sample': other_diff[:6],
            'template': args.template}

    # ---- BN audit ------------------------------------------------------------
    n_frozen = sum(1 for _, m in model.named_modules()
                   if type(m).__name__ == 'FrozenBatchNorm2d')
    n_bn = sum(1 for _, m in model.named_modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm))
    snap = bn_snapshot(model)

    dl = cfg.train_dataloader
    it = iter(dl)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    t_tot = time.time()
    micro = []
    for _ in range(args.accum):
        t = time.time()
        samples, targets = next(it)
        samples = samples.cuda(non_blocking=True)
        targets = [{k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in d.items()}
                   for d in targets]
        outputs = model(samples, targets)
        loss = sum(criterion(outputs, targets).values()) / args.accum
        loss.backward()
        torch.cuda.synchronize()
        micro.append(round(time.time() - t, 2))
    t_clip = time.time()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if ema is not None and hasattr(ema, 'update'):
        ema.update(model)
    torch.cuda.synchronize()
    step_extra = time.time() - t_clip
    total = time.time() - t_tot
    peak = torch.cuda.max_memory_allocated() / 2 ** 30

    after = bn_snapshot(model)
    changed = [n for n in snap if n in after and not torch.equal(snap[n][0], after[n][0])]
    bad = [n for n, prm in model.named_parameters()
           if prm.grad is not None and not torch.isfinite(prm.grad).all()]

    out = {'config': args.config, 'warmstart': args.warmstart, 'stages': args.stages,
           'micro_batch': args.micro, 'accum': args.accum,
           'effective_batch': args.micro * args.accum,
           'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'amp': False,
           'params_M': round(sum(q.numel() for q in model.parameters()) / 1e6, 1),
           'build_seconds': round(build_s, 1), 'resident_after_build_GiB': round(resident, 2),
           'micro_seconds': micro, 'micro_mean_seconds': round(sum(micro) / len(micro), 2),
           'step_extra_seconds': round(step_extra, 2),
           'optimizer_step_seconds': round(total, 2), 'peak_vram_GiB': round(peak, 2),
           'nonfinite_grad_params': bad[:8], 'nonfinite_grad_count': len(bad),
           'bn_audit': {'frozen_batch_norm_modules': n_frozen, 'batchnorm_modules': n_bn,
                        'running_stats_changed': changed[:8],
                        'running_stats_changed_count': len(changed)},
           'init_check': init}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('stages', 'micro_batch', 'accum', 'effective_batch',
                                          'params_M', 'micro_mean_seconds', 'optimizer_step_seconds',
                                          'peak_vram_GiB', 'nonfinite_grad_count', 'bn_audit')}, indent=2))
    print('carried_nonzero:', init['carried_nonzero'])
    print('fresh:', json.dumps(init['fresh_stage_modules']))
    print('fresh_mismatch:', init['fresh_mismatch_keys'])
    print('other_shared:', init['other_shared_keys'], 'differing:', init['other_shared_differing'],
          init['other_shared_differing_sample'])


if __name__ == '__main__':
    main()
