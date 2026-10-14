#!/usr/bin/env python3
"""
Sovereign V24 TADL NightXX Synthetic Degradation Suite Evaluator (Server-Side)
==============================================================================
Evaluates Sovereign V24 TADL (Night Robustness + Modality Failure Disaster Recovery)
across the full 11-level day-to-night degradation spectrum:
  - orig (100% daylight, 1.0)
  - night50 (50% illumination, 0.50)
  - night25 (25% illumination, 0.25)
  - night12 (12.5% illumination, 0.12)
  - night05 (5% illumination, 0.05)
  - night01 (1% illumination, 0.01)
  - night005 (0.5% illumination, 0.005)
  - night002 (0.2% illumination, 0.002)
  - night001 (0.1% illumination, 0.001)
  - night0005 (0.05% illumination, 0.0005)
  - black (0% illumination, 0.0)

Multimodal Channels:
  - Visible: Attenuated / Degraded RGB
  - IR Aligned: Intact (/dev/shm/aic_cache/train_aligned_ir)
  - IR LBG: Intact (/dev/shm/aic_cache/train_lbg_ir)
  - Depth: Intact (/dev/shm/aic_cache/depth_p0)

Calculates the official AIC2026 competition metric (mAP@50:95, mAP50, per-class AP)
against the 200 validation images (instances_val.json).
"""

import os
import sys
import time
import json
import argparse
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
from engine.misc.logger import MetricLogger

CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage_can", "uav", "tricycle"
]

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


def evaluate_folder(solver, module, folder_path, device, batch_size=8):
    # Switch dataset visible image root
    dataset = solver.val_dataloader.dataset
    dataset.root = folder_path
    dataset.img_folder = folder_path

    coco_evaluator = solver.evaluator
    coco_evaluator.cleanup()

    with torch.no_grad():
        for samples, targets in solver.val_dataloader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = module(samples)

            orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
            results = solver.postprocessor(outputs, orig_target_sizes, samples)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = coco_evaluator.coco_eval['bbox'].stats.tolist()

    # Extract per-class AP
    per_class_ap = {}
    try:
        eval_obj = coco_evaluator.coco_eval['bbox'].eval
        prec = eval_obj['precision']  # [10, 101, K, 4, 3]
        for k_idx, cat_id in enumerate(coco_evaluator.coco_eval['bbox'].params.catIds):
            c_prec = prec[:, :, k_idx, 0, 2]
            c_prec_valid = c_prec[c_prec > -1]
            c_ap = float(c_prec_valid.mean()) if len(c_prec_valid) > 0 else 0.0
            cat_name = CLASS_NAMES[k_idx] if k_idx < len(CLASS_NAMES) else f"class_{cat_id}"
            per_class_ap[cat_name] = round(c_ap, 4)
    except Exception as e:
        print(f"Warning extracting per-class AP: {e}")

    return {
        "mAP": stats[0],
        "mAP50": stats[1],
        "mAP75": stats[2],
        "AP_s": stats[3],
        "AP_m": stats[4],
        "AP_l": stats[5],
        "per_class": per_class_ap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deimv2/sovereign_v24_night_robust_tadl_2048.yml')
    parser.add_argument('--weights', type=str, default='outputs/sovereign_v24_night_robust_tadl_2048/best_stg2.pth')
    parser.add_argument('--val-ann', type=str, default='/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/annotations/instances_val.json')
    parser.add_argument('--out-dir', type=str, default='/root/autodl-tmp/AIC/outputs/analysis/nightxx_eval')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--levels', type=str, default='all')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" Sovereign V24 TADL NightXX Synthetic Suite Benchmark ".center(80, "="))
    print("=" * 80)
    print(f"Config: {args.cfg}")
    print(f"Weights: {args.weights}")
    print(f"Validation GT: {args.val_ann}")
    print(f"Output Dir: {out_dir}")

    cfg = YAMLConfig(args.cfg, resume=args.weights)
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = args.batch_size
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 4
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = args.val_ann

    for bb_name in ['HGNetv2', 'HGNetv2MultiModal', 'HGNetv2ViTDualStream']:
        if bb_name in cfg.yaml_cfg:
            cfg.yaml_cfg[bb_name]['pretrained'] = False

    dist_utils.setup_distributed(0, 'builtin', seed=0)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()

    module = solver.ema.module if solver.ema else solver.model
    module.eval()
    device = solver.device

    target_levels = NIGHT_LEVELS
    if args.levels != "all":
        lvl_names = set(args.levels.split(","))
        target_levels = [x for x in NIGHT_LEVELS if x[0] in lvl_names]

    results = []

    print("\nStarting evaluation across degradation levels:")
    print("-" * 80)
    print(f"{'Level':<12} | {'Illum':<6} | {'mAP@50:95':<10} | {'mAP@50':<8} | {'AP_small':<8} | {'Status'}")
    print("-" * 80)

    for tag, folder_str, illum in target_levels:
        folder = Path(folder_str)
        if not folder.exists():
            print(f"{tag:<12} | {illum:<6.4f} | {'N/A':<10} | {'N/A':<8} | {'-':<8} | Folder {folder} not found")
            continue

        t0 = time.time()
        metrics = evaluate_folder(solver, module, folder_str, device, batch_size=args.batch_size)
        elapsed = time.time() - t0

        metrics["level"] = tag
        metrics["illum"] = illum
        metrics["elapsed_sec"] = elapsed
        results.append(metrics)

        print(f"{tag:<12} | {illum:<6.4f} | {metrics['mAP']:<10.4f} | {metrics['mAP50']:<8.4f} | {metrics['AP_s']:<8.4f} | Done in {elapsed:.1f}s")

    # Save complete JSON
    out_json = out_dir / "nightxx_v24_tadl_curve.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save Markdown Report
    md_report = out_dir / "nightxx_v24_tadl_report.md"
    orig_map = results[0]["mAP"] if results else 1.0

    with open(md_report, "w", encoding="utf-8") as f:
        f.write("# NightXX 真实夜间退化谱系评测报告: Sovereign V24 TADL (Night Robust)\n\n")
        f.write(f"> **评估模型**: Sovereign V24 TADL (`best_stg2.pth`)  \n")
        f.write(f"> **增广机制**: p=0.5 对数暗光退化 + 三模态失效容灾 (RGB 15%, IR 10%, Depth 15%)  \n")
        f.write(f"> **评测集**: 官方 200 张洁净验证集 (`instances_val.json`)  \n\n")
        f.write("| 退化档位 (Level) | 照度因子 (Illumination) | 官方 mAP@50:95 | 官方 mAP@50 | 小目标 AP_s | 相对日间保持率 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            retention = (r["mAP"] / orig_map * 100) if orig_map > 0 else 0
            f.write(f"| **{r['level']}** | {r['illum']:.4f} | **{r['mAP']:.4f}** | {r['mAP50']:.4f} | {r['AP_s']:.4f} | **{retention:.1f}%** |\n")

    print("\n" + "=" * 80)
    print(" Sovereign V24 NightXX Benchmark Finished ".center(80, "="))
    print("=" * 80)
    print(f"Results JSON: {out_json}")
    print(f"Report MD:    {md_report}")

    dist_utils.cleanup()


if __name__ == '__main__':
    main()
