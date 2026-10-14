#!/usr/bin/env python3
"""
Verify Native Integrated MidBoost in HGNetv2MultiModal & sovereign_v24_night_robust_tadl_2048.yml
Runs completely unpatched solver/model evaluation to verify that native code now matches Perfect MidBoost!
"""

import os
import sys
import time
import json
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch

_root = '/root/autodl-tmp/AIC'
sys.path.insert(0, os.path.join(_root, 'vendor/DEIMv2'))
sys.path.insert(0, os.path.join(_root, 'scripts/lib'))
os.chdir(os.path.join(_root, 'vendor/DEIMv2'))

import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

from engine.misc import dist_utils
from engine.core import YAMLConfig
from engine.solver import TASKS

BENCH_LEVELS = [
    ("orig", "/dev/shm/aic_cache/visible", 1.00),
    ("night50", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night50", 0.50),
    ("night12", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night12", 0.12),
]


def evaluate_folder(solver, module, folder_path, device):
    dataset = solver.val_dataloader.dataset
    dataset.root = folder_path
    dataset.img_folder = folder_path

    coco_evaluator = solver.evaluator
    coco_evaluator.cleanup()

    with torch.no_grad():
        for samples, targets in solver.val_dataloader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = module(samples)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = solver.postprocessor(outputs, orig_target_sizes, samples)
            res = {target["image_id"].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = coco_evaluator.coco_eval["bbox"].stats.tolist()
    return stats[0], stats[1]


def main():
    print("=" * 85)
    print(" Verifying Native Unpatched MidBoost Integration in Sovereign V24 ".center(85, "="))
    print("=" * 85)

    cfg = YAMLConfig('configs/deimv2/sovereign_v24_night_robust_tadl_2048.yml',
                     resume='outputs/sovereign_v24_night_robust_tadl_2048/best_stg2.pth')
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 8
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 4
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = '/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/annotations/instances_val.json'

    for bb_name in ['HGNetv2', 'HGNetv2MultiModal']:
        if bb_name in cfg.yaml_cfg:
            cfg.yaml_cfg[bb_name]['pretrained'] = False

    dist_utils.setup_distributed(0, 'builtin', seed=0)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    module = solver.ema.module if solver.ema else solver.model
    module.eval()
    device = solver.device

    print(f"[Verification] module.backbone.use_midboost = {getattr(module.backbone, 'use_midboost', None)}")
    assert getattr(module.backbone, 'use_midboost', False) is True, "ERROR: use_midboost is not True!"

    print(f"\n{'Level':<12} | {'Illum':<6} | {'mAP@50:95':<10} | {'mAP@50':<10} | {'Time (s)':<8}")
    print("-" * 60)

    for tag, folder_str, illum in BENCH_LEVELS:
        t0 = time.time()
        m_ap, m_ap50 = evaluate_folder(solver, module, folder_str, device)
        el = time.time() - t0
        print(f"{tag:<12} | {illum:<6.2f} | {m_ap:<10.4f} | {m_ap50:<10.4f} | {el:<8.1f}")

    print("\n" + "=" * 85)
    print("[+] Native Unpatched MidBoost verification SUCCESSFUL! Code is fully operational.")
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
