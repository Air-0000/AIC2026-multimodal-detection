#!/usr/bin/env python3
"""
Systematic Fine-Grained Sweep for P2 Cascade Zoom Gate (Pure Affirmative Boost: beta=0.0)
Validates optimal alpha_boost, saliency_thresh, and max_area on V24 best_stg2.
"""

import os
import sys
import time
import json
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import roi_align

_root = '/root/autodl-tmp/AIC'
sys.path.insert(0, os.path.join(_root, 'vendor/DEIMv2'))
sys.path.insert(0, os.path.join(_root, 'scripts/eval'))
os.chdir(os.path.join(_root, 'vendor/DEIMv2'))

from test_p2_cascade_zoom_gate import P2CascadeZoomGate, setup_gated_forward, run_evaluation
from engine.core import YAMLConfig
from engine.solver import TASKS
from engine.misc import dist_utils

def main():
    print("=" * 85)
    print(" Pure Affirmative P2 Cascade Zoom Gate Parameter Sweep ".center(85, "="))
    print("=" * 85)

    cfg = YAMLConfig('configs/deimv2/sovereign_v24_night_robust_tadl_2048.yml',
                     resume='outputs/sovereign_v24_night_robust_tadl_2048/best_stg2.pth')
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 2
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 2
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = '/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/annotations/instances_val.json'
    cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'] = '/dev/shm/aic_cache/visible'

    for bb_name in ['HGNetv2', 'HGNetv2MultiModal']:
        if bb_name in cfg.yaml_cfg:
            cfg.yaml_cfg[bb_name]['pretrained'] = False

    dist_utils.setup_distributed(0, 'builtin', seed=0)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    module = solver.ema.module if solver.ema else solver.model
    module.eval()
    module.use_internal_tta = False
    setup_gated_forward(module)

    # Pure Baseline
    module.p2_gate = None
    t0 = time.time()
    base_res = run_evaluation(solver, module, solver.device)
    base_map = base_res['mAP50_95']
    base_map50 = base_res['mAP50']
    base_aps = base_res['AP_small']
    print(f"[-] Baseline: mAP@50:95: {base_map:.4f} | mAP@50: {base_map50:.4f} | AP_s: {base_aps:.4f} ({time.time()-t0:.1f}s)")

    sweep_configs = [
        {"alpha": 0.20, "thresh": 0.25, "area": 0.03, "s_min": 0.15, "s_max": 0.45},
        {"alpha": 0.30, "thresh": 0.30, "area": 0.03, "s_min": 0.15, "s_max": 0.45},
        {"alpha": 0.40, "thresh": 0.35, "area": 0.03, "s_min": 0.15, "s_max": 0.45},
        {"alpha": 0.30, "thresh": 0.25, "area": 0.02, "s_min": 0.12, "s_max": 0.48},
        {"alpha": 0.40, "thresh": 0.30, "area": 0.02, "s_min": 0.12, "s_max": 0.48},
        {"alpha": 0.50, "thresh": 0.40, "area": 0.02, "s_min": 0.15, "s_max": 0.45},
        {"alpha": 0.25, "thresh": 0.20, "area": 0.04, "s_min": 0.15, "s_max": 0.45},
    ]

    results = []
    best_gain = -999.0
    best_cfg = None
    best_res = None

    for i, sc in enumerate(sweep_configs):
        desc = f"alpha={sc['alpha']:.2f}_thresh={sc['thresh']:.2f}_area={sc['area']:.3f}_score=[{sc['s_min']:.2f},{sc['s_max']:.2f}]"
        gate = P2CascadeZoomGate(
            score_min=sc['s_min'],
            score_max=sc['s_max'],
            max_area=sc['area'],
            saliency_thresh=sc['thresh'],
            alpha_boost=sc['alpha'],
            beta_suppress=0.0,
            use_mlp=False
        ).cuda()

        module.p2_gate = gate
        t1 = time.time()
        res = run_evaluation(solver, module, solver.device)
        delta = res['mAP50_95'] - base_map
        delta_50 = res['mAP50'] - base_map50
        delta_s = res['AP_small'] - base_aps
        print(f"[{i+1}/{len(sweep_configs)}] {desc:<55} | mAP: {res['mAP50_95']:.4f} ({delta:+.4f}) | mAP50: {res['mAP50']:.4f} ({delta_50:+.4f}) | AP_s: {res['AP_small']:.4f} ({delta_s:+.4f}) | {time.time()-t1:.1f}s")

        results.append({
            "config": sc,
            "desc": desc,
            "mAP50_95": res['mAP50_95'],
            "mAP50": res['mAP50'],
            "AP_small": res['AP_small'],
            "delta_map": delta,
            "delta_map50": delta_50,
            "delta_aps": delta_s,
            "per_cat": res['per_cat']
        })

        if delta > best_gain:
            best_gain = delta
            best_cfg = sc
            best_res = res

    module.p2_gate = None

    print("\n" + "=" * 90)
    print(" SWEEP BEST CONFIGURATION ".center(90, "="))
    print("=" * 90)
    print(f"Best Config: {best_cfg}")
    print(f"Baseline mAP: {base_map:.4f} -> Best Gated mAP: {best_res['mAP50_95']:.4f} (Δ {best_gain:+.4f})")
    print(f"Baseline mAP50: {base_map50:.4f} -> Best Gated mAP50: {best_res['mAP50']:.4f} (Δ {best_res['mAP50'] - base_map50:+.4f})")
    print(f"Baseline AP_s: {base_aps:.4f} -> Best Gated AP_s: {best_res['AP_small']:.4f} (Δ {best_res['AP_small'] - base_aps:+.4f})")

    out_file = "/root/autodl-tmp/AIC/outputs/analysis/nightxx_eval/sweep_pure_boost_p2.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"baseline": base_res, "sweep": results, "best": best_cfg}, f, indent=2)
    print(f"[+] Saved results to {out_file}")

    dist_utils.cleanup()

if __name__ == '__main__':
    main()
