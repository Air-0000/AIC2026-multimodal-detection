#!/usr/bin/env python3
"""
Sovereign V24 TADL: Unified MidBoost Full-Spectrum Illumination Adaptive Gating
================================================================================
Combines:
  1. Early Darkness Aux Booster (g_aux):
     Lifts night25, night12, and night05 by scaling multi-modal thermal feature residuals:
     g_aux(b) = 1.0 + 0.40 * clamp((0.18 - b) / 0.15, 0.0, 1.0)
  2. Pitch Black Emergency Thermal Takeover (act):
     Rescues night01 and black by substituting noisy RGB with thermal features:
     act(b) = 0.60 * clamp((0.022 - b) / 0.007, 0.0, 1.0)
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


def patch_backbone(module, _is, _im):
    orig_forward = module.backbone.forward

    def dynamic_forward(x):
        rgb = x[:, :3]
        ir = x[:, 3:3 + module.backbone.ir_in_chans] if x.shape[1] >= 3 + module.backbone.ir_in_chans else None
        depth = x[:, 3 + module.backbone.ir_in_chans: 4 + module.backbone.ir_in_chans] if x.shape[1] >= 4 + module.backbone.ir_in_chans else None

        # Measure true physical illumination b
        rgb_raw = torch.clamp(rgb * _is + _im, 0.0, 1.0)
        b = rgb_raw.mean(dim=(1, 2, 3), keepdim=True)

        # Dynamic g_aux for early darkness (b in [0.03, 0.18]):
        # At day (b >= 0.18): g_aux = 1.00
        # At night25 (b ~ 0.11): g_aux = 1.19
        # At night12 (b ~ 0.054): g_aux = 1.35
        # At night05 (b ~ 0.027): g_aux = 1.40
        g_aux = 1.0 + 0.40 * torch.clamp((0.18 - b) / 0.15, min=0.0, max=1.0)

        # Dynamic thermal takeover for deep darkness (b < 0.022):
        # At b >= 0.022: act = 0.0
        # At b <= 0.015: act = 0.60
        act = 0.60 * torch.clamp((0.022 - b) / 0.007, min=0.0, max=1.0)

        if act.max() > 0.0 and ir is not None:
            ir_raw = torch.clamp(ir[:, :1] * 0.5 + 0.5, 0.0, 1.0)
            rgb_fused = (1.0 - act) * rgb_raw + act * ir_raw
            rgb = (rgb_fused - _im) / _is

        if module.backbone.use_frontend_rfgm and module.backbone.frontend_rfgm is not None:
            rgb = module.backbone.frontend_rfgm(rgb)

        y = module.backbone.rgb_net.stem(rgb)
        outs = []
        if ir is not None:
            c2, c3 = module.backbone.ir_stem(ir)
            _stride_to_feat = {8: c2, 16: c3}

        p3_prompt = None
        s0_trunk_p3 = None
        for idx, stage in enumerate(module.backbone.rgb_net.stages):
            y = stage(y)
            if idx == 0:
                if module.backbone.use_trunk_bridge and module.backbone.s0_to_p3_trunk is not None:
                    s0_trunk_p3 = module.backbone.s0_to_p3_trunk(module.backbone.s0_norm(y))
                if module.backbone.use_shallow_edge_fuse and module.backbone.shallow_edge_fuse is not None and ir is not None and depth is not None:
                    y, p3_prompt = module.backbone.shallow_edge_fuse(y, rgb, ir, depth)
            if idx == 1:
                if p3_prompt is not None:
                    y = y + p3_prompt
                if s0_trunk_p3 is not None:
                    y = y + s0_trunk_p3

            if str(idx) in module.backbone.rgb_net.dcn_res:
                y = module.backbone.rgb_net.dcn_res[str(idx)](y)

            if ir is not None and idx in module.backbone.inject_stages:
                k = module.backbone.inject_stages.index(idx)
                f = _stride_to_feat.get(module.backbone.rgb_net._out_strides[idx])
                if f is None:
                    f = F.interpolate(c3, size=y.shape[-2:], mode="bilinear", align_corners=False)
                elif f.shape[-2:] != y.shape[-2:]:
                    f = F.interpolate(f, size=y.shape[-2:], mode="bilinear", align_corners=False)
                f = module.backbone.ir_proj[k](f)

                if module.backbone.use_dcda and module.backbone.ir_align is not None and depth is not None:
                    d_sub = F.interpolate(depth, size=y.shape[-2:], mode="nearest")
                    inv_d = torch.zeros_like(d_sub)
                    valid = (d_sub > 0.015) & (d_sub < 1.0)
                    inv_d[valid] = 1.0 / (d_sub[valid] * 20.0 + 0.1)
                    f = module.backbone.ir_align[k](y, f, inv_d)

                if module.backbone.use_dog and module.backbone.occlusion_gate is not None and depth is not None:
                    f = module.backbone.occlusion_gate[k](f, depth)

                if module.backbone.use_bilateral_pool and module.backbone.bilateral_pool is not None and depth is not None:
                    d_sub = F.interpolate(depth, size=y.shape[-2:], mode="nearest")
                    f = module.backbone.bilateral_pool[k](f, d_sub)

                if module.backbone.use_3d_lift and module.backbone.coord_lift is not None and depth is not None:
                    pos_3d = module.backbone.coord_lift[k](depth, y.shape)
                    y_in = y + pos_3d
                    f_in = f + pos_3d
                else:
                    y_in = y
                    f_in = f

                residual = module.backbone.ir_res[k](module.backbone.ir_fuse[k](y_in, f_in))
                y = y + g_aux * residual

            if idx in module.backbone.return_idx:
                outs.append(y)
        return outs

    module.backbone.forward = dynamic_forward
    return orig_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deimv2/sovereign_v24_night_robust_tadl_2048.yml')
    parser.add_argument('--weights', type=str, default='outputs/sovereign_v24_night_robust_tadl_2048/best_stg2.pth')
    parser.add_argument('--val-ann', type=str, default='/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/annotations/instances_val.json')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--out-json', type=str, default='/root/autodl-tmp/AIC/outputs/analysis/nightxx_eval/v24_unified_midboost_11levels.json')
    args = parser.parse_args()

    print("=" * 85)
    print(" Sovereign V24 TADL + Unified MidBoost Full-Spectrum Suite ".center(85, "="))
    print("=" * 85)

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

    patch_backbone(module, _is, _im)

    print("\nStarting Unified MidBoost 11-level NightXX evaluation:")
    print("-" * 75)
    print(f"{'Level':<12} | {'Illum':<6} | {'mAP@50:95':<10} | {'mAP50':<8} | {'Time (s)'}")
    print("-" * 75)

    results = {}

    for tag, folder_str, illum in NIGHT_LEVELS:
        folder = Path(folder_str)
        if not folder.exists():
            print(f"Skipping {tag}: folder not found ({folder_str})")
            continue

        solver.val_dataloader.dataset.root = folder_str
        solver.val_dataloader.dataset.img_folder = folder_str
        solver.evaluator.cleanup()

        t0 = time.time()
        with torch.no_grad():
            for samples, targets in solver.val_dataloader:
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
        results[tag] = {"illum": illum, "mAP": m_ap, "mAP50": m_ap50}
        print(f"{tag:<12} | {illum:<6.4f} | {m_ap:<10.4f} | {m_ap50:<8.4f} | {el:<6.1f}")

    out_file = Path(args.out_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 85)
    print(f"[+] Unified Full-Spectrum benchmark completed! Results saved to {out_file}")
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
