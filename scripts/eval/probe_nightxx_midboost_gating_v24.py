#!/usr/bin/env python3
"""
Sovereign V24: MidBoost-Style Luminance-Adaptive Gating & Residual Injection Probe
==================================================================================
Tests 3 adaptive fusion mechanisms inspired by MidBoost (RDTTrack / LEVER 4) on Sovereign V24:
  Mode 0: Baseline (No gating, raw V24)
  Mode 1: Feature-level MidBoost Gating (RGB suppression w_rgb + Aux booster g_aux)
  Mode 2: Input-level Adaptive Residual Injection (Illumination-conditioned IR/Depth injection)
  Mode 3: Dual-Tier Full Collaboration (Mode 1 + Mode 2 combined)

Evaluates on 5 key benchmark levels:
  - orig (1.0, Day validation)
  - night12 (0.12, Urban streetlight)
  - night05 (0.05, Deep dark night)
  - night01 (0.01, Near pitch black)
  - black (0.00, Full sensor blindness)
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

PROBE_LEVELS = [
    ("orig", "/dev/shm/aic_cache/visible", 1.0),
    ("night12", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night12", 0.12),
    ("night05", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night05", 0.05),
    ("night01", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night01", 0.01),
    ("black", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_black", 0.000),
]


def apply_input_adaptive_residual(samples):
    """
    Mode 2: Dynamically injects IR and Depth residuals into RGB based on scene illumination.
    samples: [B, 6, H, W]
    """
    rgb = samples[:, :3]        # [B, 3, H, W]
    ir = samples[:, 3:4]         # [B, 1, H, W] heat residual
    depth = samples[:, 5:6]      # [B, 1, H, W] depth

    b = rgb.mean(dim=(1, 2, 3), keepdim=True) # [B, 1, 1, 1] in [0, 1]

    # Illumination-dependent injection scale:
    # If bright (b >= 0.25): alpha -> 0.0 (RGB pristine)
    # If dark (b < 0.15): alpha smoothly climbs to 0.25 ~ 0.35
    alpha = torch.clamp((0.25 - b) / 0.25, min=0.0, max=1.0) * 0.30
    beta = torch.clamp((0.25 - b) / 0.25, min=0.0, max=1.0) * 0.15

    L = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    R_new = rgb[:, 0:1] + alpha * (ir - L)
    M_dep = (depth > 0.01).float()
    G_new = rgb[:, 1:2] + beta * M_dep * (depth - L)
    B_new = rgb[:, 2:3]

    fused_rgb = torch.clamp(torch.cat([R_new, G_new, B_new], dim=1), 0.0, 1.0)
    samples_fused = torch.cat([fused_rgb, samples[:, 3:]], dim=1)
    return samples_fused


def patch_backbone_with_midboost_gating(backbone, gate_mode="feature"):
    """
    Patches HGNetv2MultiModal forward with MidBoost-style RGB suppression and Aux boosting.
    """
    orig_forward = backbone.forward

    def gated_forward(x):
        rgb = x[:, :3]
        ir = x[:, 3:3 + backbone.ir_in_chans] if x.shape[1] > 3 else None
        depth = x[:, 3 + backbone.ir_in_chans:3 + backbone.ir_in_chans + 1] if x.shape[1] > 3 + backbone.ir_in_chans else None

        # 1. Measure raw scene luminance b in [0, 1]
        b = rgb.mean(dim=(1, 2, 3), keepdim=True)

        # 2. MidBoost LEVER 4: RGB Reliability Gate
        # When b < 0.08, w_rgb smoothly drops to 0, muting noisy RGB features
        w_rgb = torch.sigmoid(20.0 * (b - 0.07))

        # 3. MidBoost Aux Booster:
        # When b < 0.12, g_aux climbs from 1.0 up to 1.5x, letting IR/Depth take over
        g_aux = 1.0 + 0.5 * torch.sigmoid(20.0 * (0.10 - b))

        y = backbone.rgb_net.stem(rgb)
        outs = []
        if ir is not None:
            c2, c3 = backbone.ir_stem(ir)
            _stride_to_feat = {8: c2, 16: c3}

        p3_prompt = None
        s0_trunk_p3 = None
        for idx, stage in enumerate(backbone.rgb_net.stages):
            y = stage(y)

            # Suppress noisy RGB features in deep night
            if gate_mode in ["feature", "dual"]:
                y = w_rgb * y

            if idx == 0:
                if backbone.use_trunk_bridge and backbone.s0_to_p3_trunk is not None:
                    s0_trunk_p3 = backbone.s0_to_p3_trunk(backbone.s0_norm(y))
                if backbone.use_shallow_edge_fuse and backbone.shallow_edge_fuse is not None and ir is not None and depth is not None:
                    y, p3_prompt = backbone.shallow_edge_fuse(y, rgb, ir, depth)

            if idx == 1:
                if p3_prompt is not None:
                    y = y + p3_prompt
                if s0_trunk_p3 is not None:
                    y = y + s0_trunk_p3

            if str(idx) in backbone.rgb_net.dcn_res:
                y = backbone.rgb_net.dcn_res[str(idx)](y)

            if ir is not None and idx in backbone.inject_stages:
                k = backbone.inject_stages.index(idx)
                f = _stride_to_feat.get(backbone.rgb_net._out_strides[idx])
                if f is None:
                    f = F.interpolate(c3, size=y.shape[-2:], mode='bilinear', align_corners=False)
                elif f.shape[-2:] != y.shape[-2:]:
                    f = F.interpolate(f, size=y.shape[-2:], mode='bilinear', align_corners=False)
                f = backbone.ir_proj[k](f)

                # Boost Aux features in the dark
                if gate_mode in ["feature", "dual"]:
                    f = g_aux * f

                if backbone.use_dcda and backbone.ir_align is not None and depth is not None:
                    d_sub = F.interpolate(depth, size=y.shape[-2:], mode='nearest')
                    inv_d = torch.zeros_like(d_sub)
                    valid = (d_sub > 0.015) & (d_sub < 1.0)
                    inv_d[valid] = 1.0 / (d_sub[valid] * 20.0 + 0.1)
                    f = backbone.ir_align[k](y, f, inv_d)

                if backbone.use_dog and backbone.occlusion_gate is not None and depth is not None:
                    f = backbone.occlusion_gate[k](f, depth)

                if backbone.use_bilateral_pool and backbone.bilateral_pool is not None and depth is not None:
                    d_sub = F.interpolate(depth, size=y.shape[-2:], mode='nearest')
                    f = backbone.bilateral_pool[k](f, d_sub)

                if backbone.use_3d_lift and backbone.coord_lift is not None and depth is not None:
                    pos_3d = backbone.coord_lift[k](depth, y.shape)
                    y_in = y + pos_3d
                    f_in = f + pos_3d
                else:
                    y_in = y
                    f_in = f

                y = y + backbone.ir_res[k](backbone.ir_fuse[k](y_in, f_in))

            if idx in backbone.return_idx:
                outs.append(y)
        return outs

    backbone.forward = gated_forward
    return orig_forward


def evaluate_level(solver, module, folder_path, device, mode=0, batch_size=8):
    dataset = solver.val_dataloader.dataset
    dataset.root = folder_path
    dataset.img_folder = folder_path

    coco_evaluator = solver.evaluator
    coco_evaluator.cleanup()

    with torch.no_grad():
        for samples, targets in solver.val_dataloader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Apply Mode 2 or Mode 3 (Input Adaptive Residual)
            if mode in [2, 3]:
                samples = apply_input_adaptive_residual(samples)

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
    return stats[0], stats[1] # mAP, mAP50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deimv2/sovereign_v24_night_robust_tadl_2048.yml')
    parser.add_argument('--weights', type=str, default='outputs/sovereign_v24_night_robust_tadl_2048/best_stg2.pth')
    parser.add_argument('--val-ann', type=str, default='/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/annotations/instances_val.json')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    print("=" * 85)
    print(" Sovereign V24: MidBoost-Style Adaptive Gating & Residual Probe ".center(85, "="))
    print("=" * 85)
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

    modes = [
        (0, "Baseline (Raw V24)"),
        (1, "Mode 1: Feature MidBoost Gating (RGB-Mute + Aux-Boost)"),
        (2, "Mode 2: Input Adaptive Residual (Physical Injection)"),
        (3, "Mode 3: Dual-Tier Full Collaboration (Mode 1 + 2)"),
    ]

    all_results = {}

    for mode_id, mode_name in modes:
        print("\n" + "=" * 85)
        print(f" Testing {mode_name} ".center(85, "="))
        print("=" * 85)
        print(f"{'Level':<12} | {'Illum':<6} | {'mAP@50:95':<10} | {'mAP@50':<8} | {'Time (s)':<8}")
        print("-" * 65)

        # Apply patching if mode is 1 or 3
        if mode_id in [1, 3]:
            patch_backbone_with_midboost_gating(module.backbone, gate_mode="feature" if mode_id == 1 else "dual")
        else:
            # Restore baseline forward if needed
            pass

        mode_res = {}
        for tag, folder_str, illum in PROBE_LEVELS:
            t0 = time.time()
            m_ap, m_ap50 = evaluate_level(solver, module, folder_str, device, mode=mode_id, batch_size=args.batch_size)
            el = time.time() - t0
            mode_res[tag] = {"mAP": m_ap, "mAP50": m_ap50}
            print(f"{tag:<12} | {illum:<6.4f} | {m_ap:<10.4f} | {m_ap50:<8.4f} | {el:<8.1f}")

        all_results[mode_name] = mode_res

    # Save summary json
    out_json = Path("/root/autodl-tmp/AIC/outputs/analysis/nightxx_eval/v24_midboost_gating_probe.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 85)
    print(f"[+] Probe benchmark completed! Saved to {out_json}")
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
