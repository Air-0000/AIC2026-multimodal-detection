#!/usr/bin/env python3
"""
Sovereign V24 TADL: Calibrated MidBoost Emergency Gating & Thermal Takeover
=============================================================================
Evaluates Sovereign V24 TADL across the full 11-level NightXX spectrum with
the calibrated MidBoost illumination-reliability gating mechanism:

Gating Formulation:
  b = mean(clamp(rgb_norm * std + mean, 0, 1))   # True physical scene illumination
  act = clamp((0.022 - b) / (0.022 - 0.015), 0.0, 1.0) * 0.60
  - When b >= 0.022 (Day -> Night05): act = 0.0 (Native Sovereign V24 100% intact)
  - When b <= 0.015 (Night01 -> Black): act = 0.60 (Emergency thermal takeover)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn.functional as F

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

NIGHT_LEVELS = [
    ("orig", "/dev/shm/aic_cache/visible", 1.0),
    ("night50", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night50", 0.50),
    ("night25", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night25", 0.25),
    ("night12", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night12", 0.12),
    ("night05", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night05", 0.05),
    ("night01", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night01", 0.01),
    ("night005", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night005", 0.005),
    ("night002", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night002", 0.002),
    ("night001", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night001", 0.001),
    ("night0005", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night0005", 0.0005),
    ("black", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_black", 0.000),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deimv2/sovereign_v24_night_robust_tadl_2048.yml')
    parser.add_argument('--weights', type=str, default='outputs/sovereign_v24_night_robust_tadl_2048/best_stg2.pth')
    parser.add_argument('--val-ann', type=str, default='/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/annotations/instances_val.json')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--out-json', type=str, default='/root/autodl-tmp/AIC/outputs/analysis/nightxx_eval/v24_calibrated_midboost_11levels.json')
    args = parser.parse_args()

    print("=" * 80)
    print(" Sovereign V24 TADL + Calibrated MidBoost Gating Suite ".center(80, "="))
    print("=" * 80)
    print(f"Config: {args.cfg}")
    print(f"Weights: {args.weights}")

    cfg = YAMLConfig(args.cfg, resume=args.weights)
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = args.batch_size
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 4
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = args.val_ann

    for bb_name in ['HGNetv2', 'HGNetv2MultiModal']:
        if bb_name in cfg.yaml_cfg:
            cfg.yaml_cfg[bb_name]['pretrained'] = False

    dist_utils.setup_distributed(0, 'builtin', seed=0)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    module = solver.ema.module if solver.ema else solver.model
    module.eval()
    device = solver.device

    _im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _is = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print("\nStarting 11-level NightXX evaluation:")
    print("-" * 80)
    print(f"{'Level':<12} | {'Illum':<6} | {'mAP@50:95':<10} | {'mAP50':<8} | {'Act_Mean':<8} | {'Time (s)'}")
    print("-" * 80)

    results = {}

    for tag, folder_str, illum in NIGHT_LEVELS:
        folder = Path(folder_str)
        if not folder.exists():
            print(f"Skipping {tag}: folder not found ({folder_str})")
            continue

        solver.val_dataloader.dataset.root = folder_str
        solver.val_dataloader.dataset.img_folder = folder_str
        solver.evaluator.cleanup()

        acts = []
        t0 = time.time()
        with torch.no_grad():
            for samples, targets in solver.val_dataloader:
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # De-normalize RGB and IR
                rgb_raw = torch.clamp(samples[:, :3] * _is + _im, 0.0, 1.0)
                ir_raw = torch.clamp(samples[:, 3:4] * 0.5 + 0.5, 0.0, 1.0)

                b = rgb_raw.mean(dim=(1, 2, 3), keepdim=True)
                # Calibrated MidBoost Gating: transition band [0.015, 0.022]
                # If b >= 0.022: act = 0 (100% native V24)
                # If b <= 0.015: act = 0.60 (Emergency thermal takeover)
                act = torch.clamp((0.022 - b) / (0.022 - 0.015), min=0.0, max=1.0) * 0.60
                acts.append(act.mean().item())

                if act.max() > 0.0:
                    rgb_fused = (1.0 - act) * rgb_raw + act * ir_raw
                    rgb_norm = (rgb_fused - _im) / _is
                    samples = torch.cat([rgb_norm, samples[:, 3:]], dim=1)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = module(samples)

                orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
                res_dict = solver.postprocessor(outputs, orig_target_sizes, samples)
                res = {target['image_id'].item(): output for target, output in zip(targets, res_dict)}
                solver.evaluator.update(res)

        solver.evaluator.synchronize_between_processes()
        solver.evaluator.accumulate()
        solver.evaluator.summarize()
        stats = solver.evaluator.coco_eval['bbox'].stats.tolist()
        m_ap, m_ap50 = stats[0], stats[1]
        el = time.time() - t0
        act_avg = sum(acts) / len(acts)
        results[tag] = {"illum": illum, "mAP": m_ap, "mAP50": m_ap50, "act": act_avg}
        print(f"{tag:<12} | {illum:<6.4f} | {m_ap:<10.4f} | {m_ap50:<8.4f} | {act_avg:<8.4f} | {el:<6.1f}")

    out_file = Path(args.out_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[+] All 11 levels successfully evaluated! Summary saved to {out_file}")
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
