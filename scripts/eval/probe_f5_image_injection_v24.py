#!/usr/bin/env python3
"""
Probe: Testing F5 Input-Level Residual Injection ('F5加进图像') on Sovereign V24
Evaluates:
  1. Sovereign V24 + Unified MidBoost (Current Best Baseline)
  2. Sovereign V24 + F5 Direct Image Injection (alpha=0.15 IR, beta=0.10 Depth)
  3. Sovereign V24 + Illumination-Adaptive F5 Image Injection (Dual-Tier Combined)
Across 4 key levels:
  - orig (1.00)
  - night12 (0.12, Urban streetlight - user's photo darkness)
  - night01 (0.01, Pitch black)
  - black (0.00, Zero photons)
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

BENCH_LEVELS = [
    ("orig", "/dev/shm/aic_cache/visible", 1.0),
    ("night12", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night12", 0.12),
    ("night01", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night01", 0.01),
    ("black", "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_black", 0.00),
]


def patch_model_forward(module, mode="unified"):
    """
    mode:
      - 'unified': Current Sovereign V24 + Unified MidBoost
      - 'f5_fixed': V24 + Fixed F5 Residual (R += 0.15*(IR-L), G += 0.10*(Depth-L))
      - 'f5_adaptive': V24 + Unified MidBoost + Adaptive F5 Residual
    """
    orig_forward = module.backbone.forward
    _im = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    _is = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    def forward_hook(x):
        rgb = x[:, :3]
        ir = x[:, 3:3 + module.backbone.ir_in_chans] if x.shape[1] >= 3 + module.backbone.ir_in_chans else None
        depth = x[:, 3 + module.backbone.ir_in_chans: 4 + module.backbone.ir_in_chans] if x.shape[1] >= 4 + module.backbone.ir_in_chans else None

        rgb_raw = torch.clamp(rgb * _is + _im, 0.0, 1.0)
        b = rgb_raw.mean(dim=(1, 2, 3), keepdim=True)
        L = 0.299 * rgb_raw[:, 0:1] + 0.587 * rgb_raw[:, 1:2] + 0.114 * rgb_raw[:, 2:3]

        if mode == "f5_fixed" and ir is not None:
            # Direct F5 image injection
            ir_raw = torch.clamp(ir[:, :1] * 0.5 + 0.5, 0.0, 1.0)
            R_fused = torch.clamp(rgb_raw[:, 0:1] + 0.15 * (ir_raw - L), 0.0, 1.0)
            if depth is not None:
                d_raw = torch.clamp(depth * 0.5 + 0.5, 0.0, 1.0)
                G_fused = torch.clamp(rgb_raw[:, 1:2] + 0.10 * (d_raw - L), 0.0, 1.0)
            else:
                G_fused = rgb_raw[:, 1:2]
            B_fused = rgb_raw[:, 2:3]
            rgb_fused = torch.cat([R_fused, G_fused, B_fused], dim=1)
            rgb = (rgb_fused - _im) / _is
            g_aux = 1.0

        elif mode == "f5_adaptive" and ir is not None:
            # F5 image injection scaled by darkness + MidBoost Aux Booster
            g_aux = 1.0 + 0.40 * torch.clamp((0.18 - b) / 0.15, min=0.0, max=1.0)
            alpha = 0.25 * torch.clamp((0.20 - b) / 0.18, min=0.0, max=1.0)
            ir_raw = torch.clamp(ir[:, :1] * 0.5 + 0.5, 0.0, 1.0)
            R_fused = torch.clamp(rgb_raw[:, 0:1] + alpha * (ir_raw - L), 0.0, 1.0)
            if depth is not None:
                d_raw = torch.clamp(depth * 0.5 + 0.5, 0.0, 1.0)
                G_fused = torch.clamp(rgb_raw[:, 1:2] + (alpha * 0.6) * (d_raw - L), 0.0, 1.0)
            else:
                G_fused = rgb_raw[:, 1:2]
            B_fused = rgb_raw[:, 2:3]
            rgb_fused = torch.cat([R_fused, G_fused, B_fused], dim=1)

            # Deep night takeover if b < 0.022
            act = 0.60 * torch.clamp((0.022 - b) / 0.007, min=0.0, max=1.0)
            if act.max() > 0.0:
                rgb_fused = (1.0 - act) * rgb_fused + act * ir_raw
            rgb = (rgb_fused - _im) / _is

        else: # unified
            g_aux = 1.0 + 0.40 * torch.clamp((0.18 - b) / 0.15, min=0.0, max=1.0)
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

                if mode in ["unified", "f5_adaptive"]:
                    f = g_aux * f

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

                y = y + module.backbone.ir_res[k](module.backbone.ir_fuse[k](y_in, f_in))

            if idx in module.backbone.return_idx:
                outs.append(y)

        return outs

    module.backbone.forward = forward_hook
    return orig_forward


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
    return stats[0], stats[1] # mAP@50:95, mAP@50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deimv2/sovereign_v24_night_robust_tadl_2048.yml')
    parser.add_argument('--weights', type=str, default='outputs/sovereign_v24_night_robust_tadl_2048/best_stg2.pth')
    parser.add_argument('--val-ann', type=str, default='/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/annotations/instances_val.json')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    print("=" * 80)
    print(" Sovereign V24: F5 Image-Level Residual Injection Benchmark ".center(80, "="))
    print("=" * 80)

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
        ("unified", "1. Sovereign V24 + Unified MidBoost (当前基准)"),
        ("f5_fixed", "2. Sovereign V24 + F5 直接残差注入图像 (Fixed 0.15 IR, 0.10 Dep)"),
        ("f5_adaptive", "3. Sovereign V24 + 自适应 F5 图像注入 + MidBoost (全域统合)"),
    ]

    results = {}
    for mode_key, mode_title in modes:
        print("\n" + "#" * 80)
        print(f" {mode_title} ".center(80, "#"))
        print("#" * 80)
        print(f"{'Level':<12} | {'mAP@50:95':<10} | {'mAP@50':<10} | {'Time (s)':<8}")
        print("-" * 50)

        patch_model_forward(module, mode=mode_key)
        mode_res = {}
        for tag, folder_str, _ in BENCH_LEVELS:
            t0 = time.time()
            m_ap, m_ap50 = evaluate_level(solver, module, folder_str, device)
            el = time.time() - t0
            mode_res[tag] = {"mAP": m_ap, "mAP50": m_ap50}
            print(f"{tag:<12} | {m_ap:<10.4f} | {m_ap50:<10.4f} | {el:<8.1f}")
        results[mode_key] = mode_res

    out_file = Path("/root/autodl-tmp/AIC/outputs/analysis/nightxx_eval/f5_image_injection_probe.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[+] All probes finished! Saved to {out_file}")
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
