#!/usr/bin/env python3
"""
Evaluate Sovereign V27 on Dual Validation Sets:
  Track A: Standard Old Validation Set (instances_val.json, 1523 annotations)
  Track B: Comprehensive Clean Quality Validation Set (instances_val_clean_quality.json, 1757 annotations)
"""

import os
import sys
import copy
import torch
from pathlib import Path

_root = '/root/autodl-tmp/AIC'
if not os.path.exists(_root):
    _root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(_root, 'vendor/DEIMv2'))
sys.path.insert(0, os.path.join(_root, 'scripts/lib'))
os.chdir(os.path.join(_root, 'vendor/DEIMv2'))

from engine.core import YAMLConfig
from engine.solver.evaluator import CocoEvaluator
from torch.utils.data import DataLoader

def eval_on_ann(cfg, ckpt_path, ann_file, title="Validation Evaluation"):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print(f"Annotation file: {ann_file}")
    print("=" * 80)

    val_cfg = copy.deepcopy(cfg.val_dataloader)
    val_cfg['dataset']['ann_file'] = ann_file
    
    val_dataset = cfg.val_dataloader.dataset
    val_dataset.ann_file = ann_file
    # reload coco api
    from pycocotools.coco import COCO
    val_dataset.coco = COCO(ann_file)
    val_dataset.ids = list(sorted(val_dataset.coco.imgs.keys()))

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    from engine.solver.det_engine import evaluate
    model = cfg.model.cuda()
    postprocessor = cfg.postprocessor.cuda()
    
    base_ds = val_dataset.coco
    eval_results = evaluate(
        model,
        criterion=None,
        postprocessor=postprocessor,
        data_loader=val_loader,
        base_ds=base_ds,
        device="cuda:0",
    )
    return eval_results

def main():
    cfg_path = "configs/deimv2/sovereign_v27_pure_unbiased_2048.yml"
    ckpt_path = "outputs/sovereign_v27_pure_unbiased_2048/best_stg2.pth"
    
    cfg = YAMLConfig(cfg_path, resume=ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'ema' in ckpt and ckpt['ema']:
        state = ckpt['ema']['module']
    elif 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt
    cfg.model.load_state_dict(state, strict=False)

    ann_old = "/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/annotations/instances_val.json"
    ann_clean = "/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/annotations/instances_val_clean_quality.json"

    res_old = eval_on_ann(cfg, ckpt_path, ann_old, title="Track A: Old Raw Validation Set (1,523 annotations)")
    res_clean = eval_on_ann(cfg, ckpt_path, ann_clean, title="Track B: Comprehensive Clean Quality Validation Set (1,757 annotations)")

    print("\n" + "=" * 80)
    print(" Dual Validation Set Final Comparison Summary ".center(80, "="))
    print("=" * 80)
    print(f"{'Metric':<25} | {'Old Val (1,523 anns)':<22} | {'Comprehensive Val (1,757 anns)':<25}")
    print("-" * 80)
    # print overall comparison
    m_old = res_old[0] if isinstance(res_old, tuple) else res_old
    m_clean = res_clean[0] if isinstance(res_clean, tuple) else res_clean
    print(f"{'mAP (0.50:0.95)':<25} | {m_old.stats[0]:.4f}                 | {m_clean.stats[0]:.4f}")
    print(f"{'AP50':<25} | {m_old.stats[1]:.4f}                 | {m_clean.stats[1]:.4f}")
    print(f"{'AP75':<25} | {m_old.stats[2]:.4f}                 | {m_clean.stats[2]:.4f}")
    print(f"{'AP (small)':<25} | {m_old.stats[3]:.4f}                 | {m_clean.stats[3]:.4f}")
    print(f"{'AP (medium)':<25} | {m_old.stats[4]:.4f}                 | {m_clean.stats[4]:.4f}")
    print(f"{'AP (large)':<25} | {m_old.stats[5]:.4f}                 | {m_clean.stats[5]:.4f}")
    print(f"{'AR@50':<25} | {m_old.stats[12]:.4f}                 | {m_clean.stats[12]:.4f}")
    print(f"{'AR@75':<25} | {m_old.stats[13]:.4f}                 | {m_clean.stats[13]:.4f}")
    print("=" * 80)

if __name__ == '__main__':
    main()
