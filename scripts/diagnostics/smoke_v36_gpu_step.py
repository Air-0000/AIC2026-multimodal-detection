#!/usr/bin/env python3
"""One real training step at a given resolution and injection topology.

Measures peak VRAM, step wall time and gradient finiteness on actual data.
Decision probe only: it does not train, save or evaluate anything.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch


def patch_resize(node, size):
    if isinstance(node, dict):
        if str(node.get('type', '')) == 'Resize':
            node['size'] = list(size)
        for v in node.values():
            patch_resize(v, size)
    elif isinstance(node, list):
        for v in node:
            patch_resize(v, size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--stages', required=True, help='e.g. 1,2')
    p.add_argument('--height', type=int, default=1440)
    p.add_argument('--width', type=int, default=2560)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    from engine.core import YAMLConfig

    stages = [int(v) for v in args.stages.split(',')]
    cfg = YAMLConfig(args.config)
    y = cfg.yaml_cfg
    y['HGNetv2MultiModal']['inject_stages'] = stages
    for split in ('train_dataloader', 'val_dataloader'):
        dl = y.get(split)
        if not dl:
            continue
        dl['total_batch_size'] = 1
        dl['num_workers'] = 0
        dl['persistent_workers'] = False
        patch_resize(dl.get('dataset'), (args.height, args.width))
    y['use_amp'] = bool(args.amp)
    y['output_dir'] = '/tmp/smoke_v36'

    t0 = time.time()
    model = cfg.model.cuda()
    criterion = cfg.criterion.cuda()
    build_s = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    dl = cfg.train_dataloader
    t1 = time.time()
    samples, targets = next(iter(dl))
    load_s = time.time() - t1

    samples = samples.cuda(non_blocking=True)
    targets = [{k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()}
               for t in targets]
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    model.train()
    outputs = model(samples, targets)
    loss_dict = criterion(outputs, targets)
    loss = sum(loss_dict.values())
    loss.backward()
    step_s = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 2 ** 30

    bad = []
    for name, prm in model.named_parameters():
        if prm.grad is None:
            continue
        if not torch.isfinite(prm.grad).all():
            bad.append(name)
    no_grad = [n for n, prm in model.named_parameters() if prm.requires_grad and prm.grad is None]

    out = {'config': args.config, 'stages': stages,
           'input_shape': list(samples.shape),
           'device': torch.cuda.get_device_name(0),
           'torch': torch.__version__, 'amp': bool(args.amp),
           'model_params_M': round(n_params, 1),
           'build_seconds': round(build_s, 1), 'batch_fetch_seconds': round(load_s, 1),
           'step_seconds': round(step_s, 1),
           'peak_vram_GiB': round(peak, 2),
           'loss_keys': sorted(loss_dict.keys()),
           'loss_values': {k: round(float(v), 4) for k, v in loss_dict.items()},
           'nonfinite_grad_params': bad[:10], 'nonfinite_grad_count': len(bad),
           'params_without_grad': no_grad[:10], 'params_without_grad_count': len(no_grad)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
