#!/usr/bin/env python3
"""
Denoised Foreground-Selective MidBoost (DF-MidBoost) for Sovereign V24
Tests combining:
  1. DOG Depth Gradient (Boundary Geometry)
  2. Local Shadow Detection (Underexposed regions)
  3. Thermal Peak Salience (Target heat vs diffuse road heat)
To eliminate daytime thermal clutter while boosting target recall in both bright & dark conditions!
"""

import os
import sys
import time
import json
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

BENCH_LEVELS = [
    ("orig", "/dev/shm/aic_cache/visible", 1.00),
    ("night50", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night50", 0.50),
    ("night12", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night12", 0.12),
    ("night01", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night01", 0.01),
    ("black", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_black", 0.00),
]


def patch_denoised_foreground_midboost(module):
    _im = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    _is = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device='cuda').view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device='cuda').view(1, 1, 3, 3) / 8.0

    def forward_df_midboost(x):
        rgb = x[:, :3]
        ir = x[:, 3:3 + module.backbone.ir_in_chans] if x.shape[1] >= 3 + module.backbone.ir_in_chans else None
        depth = x[:, 3 + module.backbone.ir_in_chans: 4 + module.backbone.ir_in_chans] if x.shape[1] >= 4 + module.backbone.ir_in_chans else None

        rgb_raw = torch.clamp(rgb * _is + _im, 0.0, 1.0)
        b_global = rgb_raw.mean(dim=(1, 2, 3), keepdim=True)

        # 1. Local Shadow map [B, 1, H, W]
        L_map = 0.299 * rgb_raw[:, 0:1] + 0.587 * rgb_raw[:, 1:2] + 0.114 * rgb_raw[:, 2:3]

        # 2. Deep night takeover (b < 0.022)
        act = 0.60 * torch.clamp((0.022 - b_global) / 0.007, min=0.0, max=1.0)
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
                if p3_prompt is not None: y = y + p3_prompt
                if s0_trunk_p3 is not None: y = y + s0_trunk_p3

            if str(idx) in module.backbone.rgb_net.dcn_res:
                y = module.backbone.rgb_net.dcn_res[str(idx)](y)

            if ir is not None and idx in module.backbone.inject_stages:
                k = module.backbone.inject_stages.index(idx)
                f = _stride_to_feat.get(module.backbone.rgb_net._out_strides[idx])
                if f is None:
                    f = F.interpolate(c3, size=y.shape[-2:], mode='bilinear', align_corners=False)
                elif f.shape[-2:] != y.shape[-2:]:
                    f = F.interpolate(f, size=y.shape[-2:], mode='bilinear', align_corners=False)
                f = module.backbone.ir_proj[k](f)

                if module.backbone.use_dcda and module.backbone.ir_align is not None and depth is not None:
                    d_sub = F.interpolate(depth, size=y.shape[-2:], mode='nearest')
                    inv_d = torch.zeros_like(d_sub)
                    valid = (d_sub > 0.015) & (d_sub < 1.0)
                    inv_d[valid] = 1.0 / (d_sub[valid] * 20.0 + 0.1)
                    f = module.backbone.ir_align[k](y, f, inv_d)

                if module.backbone.use_dog and module.backbone.occlusion_gate is not None and depth is not None:
                    f = module.backbone.occlusion_gate[k](f, depth)

                if module.backbone.use_bilateral_pool and module.backbone.bilateral_pool is not None and depth is not None:
                    d_sub = F.interpolate(depth, size=y.shape[-2:], mode='nearest')
                    f = module.backbone.bilateral_pool[k](f, d_sub)

                if module.backbone.use_3d_lift and module.backbone.coord_lift is not None and depth is not None:
                    pos_3d = module.backbone.coord_lift[k](depth, y.shape)
                    y_in = y + pos_3d; f_in = f + pos_3d
                else:
                    y_in = y; f_in = f

                residual = module.backbone.ir_res[k](module.backbone.ir_fuse[k](y_in, f_in))

                # --- Denoised Foreground-Selective Gating (DF-MidBoost) ---
                H_f, W_f = y.shape[-2:]
                
                # A. Spatial Shadow Term: activates in dark shadow regions
                L_feat = F.interpolate(L_map, size=(H_f, W_f), mode='area')
                shadow_mask = torch.clamp((0.28 - L_feat) / 0.22, min=0.0, max=1.0)

                # B. Depth Discontinuity Term: detects true 3D object edges (filtering flat road & sky)
                if depth is not None:
                    d_feat = F.interpolate(depth, size=(H_f, W_f), mode='bilinear', align_corners=False)
                    d_pad = F.pad(d_feat * 20.0, (1, 1, 1, 1), mode='replicate')
                    gx = F.conv2d(d_pad, sobel_x)
                    gy = F.conv2d(d_pad, sobel_y)
                    depth_edge = torch.clamp(torch.sqrt(gx**2 + gy**2 + 1e-6) / 3.0, min=0.0, max=1.0)
                else:
                    depth_edge = torch.zeros_like(shadow_mask)

                # C. Thermal Contrast Saliency: high for target hotspots, low for uniform heated asphalt
                ir_feat_map = F.interpolate(ir[:, :1], size=(H_f, W_f), mode='area')
                ir_bg = F.avg_pool2d(ir_feat_map, kernel_size=5, stride=1, padding=2)
                ir_saliency = torch.clamp((ir_feat_map - ir_bg) * 4.0, min=0.0, max=1.0)

                # D. Combined Foreground-Selective Mask (0 on flat road/sky, 1 on shadowed targets/edges)
                # In bright daylight, only shadow + edges + heat peaks pass through!
                # In darkness (b_global < 0.25), dark_ambient automatically scales up all objects!
                dark_ambient = torch.clamp((0.35 - b_global) / 0.32, min=0.0, max=1.0)
                fg_mask = torch.clamp(shadow_mask + 0.8 * depth_edge + 0.6 * ir_saliency + dark_ambient, min=0.0, max=1.0)

                # Base gain + selective boost (max 1.45x, but on flat day asphalt it stays strictly 1.00x!)
                g_spatial = 1.0 + 0.45 * fg_mask

                y = y + g_spatial * residual

            if idx in module.backbone.return_idx:
                outs.append(y)
        return outs

    module.backbone.forward = forward_df_midboost


def evaluate_level(solver, module, folder_path, device):
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
    print(" Sovereign V24: Denoised Foreground-Selective MidBoost (DF-MidBoost) ".center(85, "="))
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

    patch_denoised_foreground_midboost(module)

    print(f"\n{'Level':<12} | {'Illum':<6} | {'mAP@50:95':<10} | {'mAP@50':<10} | {'Time (s)':<8}")
    print("-" * 60)

    res = {}
    for tag, folder_str, illum in BENCH_LEVELS:
        t0 = time.time()
        m_ap, m_ap50 = evaluate_level(solver, module, folder_str, device)
        el = time.time() - t0
        res[tag] = {"mAP": m_ap, "mAP50": m_ap50}
        print(f"{tag:<12} | {illum:<6.2f} | {m_ap:<10.4f} | {m_ap50:<10.4f} | {el:<8.1f}")

    out_file = Path("/root/autodl-tmp/AIC/outputs/analysis/nightxx_eval/df_midboost_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 85)
    print(f"[+] DF-MidBoost Benchmark completed! Saved to {out_file}")
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
