#!/usr/bin/env python3
"""
Test Native-Filtering Global MidBoost for Sovereign V24:
Directly testing the user's hypothesis:
"全局不是有过滤去噪的吗? 你试试"

Tests:
1. orig (broad daylight) sensitivity to global IR residual gain:
   g_aux in [1.00, 1.08, 1.15, 1.22]
2. night50 (early darkness) sensitivity to global IR residual gain:
   g_aux in [1.00, 1.15, 1.25, 1.35]
3. Full-spectrum continuous MidBoost curve with daylight active boost:
   - Day (b >= 0.35): g_aux = 1.12 (active daytime boost relying on native DOG/PBD filtering)
   - Twilight / Early dark (0.08 <= b < 0.35): smooth ramp from 1.12 to 1.40
   - Deep night (b < 0.022): thermal takeover up to 0.60
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


def patch_backbone_with_fixed_gain(module, fixed_gain):
    _im = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    _is = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    def forward(x):
        rgb = x[:, :3]
        ir = x[:, 3:3 + module.backbone.ir_in_chans] if x.shape[1] >= 3 + module.backbone.ir_in_chans else None
        depth = x[:, 3 + module.backbone.ir_in_chans: 4 + module.backbone.ir_in_chans] if x.shape[1] >= 4 + module.backbone.ir_in_chans else None

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
                y = y + fixed_gain * residual

            if idx in module.backbone.return_idx:
                outs.append(y)
        return outs

    module.backbone.forward = forward


def patch_backbone_full_continuous(module, day_base=1.12, max_gain=1.42):
    _im = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    _is = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    def forward(x):
        rgb = x[:, :3]
        ir = x[:, 3:3 + module.backbone.ir_in_chans] if x.shape[1] >= 3 + module.backbone.ir_in_chans else None
        depth = x[:, 3 + module.backbone.ir_in_chans: 4 + module.backbone.ir_in_chans] if x.shape[1] >= 4 + module.backbone.ir_in_chans else None

        rgb_raw = torch.clamp(rgb * _is + _im, 0.0, 1.0)
        b = rgb_raw.mean(dim=(1, 2, 3), keepdim=True)

        # Continuous smooth gain starting from broad daylight (b=0.48 -> day_base):
        # As illumination drops below 0.35, gain climbs smoothly to max_gain!
        darkness_ratio = torch.clamp((0.35 - b) / 0.30, min=0.0, max=1.0)
        g_aux = day_base + (max_gain - day_base) * darkness_ratio

        # Deep night thermal takeover (b < 0.022)
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

    module.backbone.forward = forward


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
    print(" Sovereign V24: Native-Filtering Global MidBoost Probe ".center(85, "="))
    print(" Testing user hypothesis: '全局不是有过滤去噪的吗? 你试试' ".center(85, " "))
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

    results = {}

    # Test 1: orig sensitivity
    print("\n--- [PART 1] Daylight 'orig' Sensitivity to Global IR Gain ---")
    orig_folder = "/dev/shm/aic_cache/visible"
    results["orig_sensitivity"] = {}
    for g in [1.00, 1.08, 1.15, 1.22]:
        patch_backbone_with_fixed_gain(module, g)
        m_ap, m_ap50 = evaluate_folder(solver, module, orig_folder, device)
        results["orig_sensitivity"][str(g)] = {"mAP": m_ap, "mAP50": m_ap50}
        print(f"  orig | g={g:<4.2f} | mAP={m_ap:.4f} | mAP50={m_ap50:.4f}")

    # Test 2: night50 sensitivity
    print("\n--- [PART 2] Twilight 'night50' Sensitivity to Global IR Gain ---")
    n50_folder = "/root/autodl-tmp/AIC/data_aux/aic_valid_rgb_night50"
    results["night50_sensitivity"] = {}
    for g in [1.00, 1.15, 1.25, 1.35]:
        patch_backbone_with_fixed_gain(module, g)
        m_ap, m_ap50 = evaluate_folder(solver, module, n50_folder, device)
        results["night50_sensitivity"][str(g)] = {"mAP": m_ap, "mAP50": m_ap50}
        print(f"  night50 | g={g:<4.2f} | mAP={m_ap:.4f} | mAP50={m_ap50:.4f}")

    # Test 3: Continuous Smooth MidBoost (Active Day 1.12 -> Smooth Peak 1.42) across 5 key levels
    print("\n--- [PART 3] Full Continuous Active-Day MidBoost Spectrum ---")
    patch_backbone_full_continuous(module, day_base=1.12, max_gain=1.42)
    results["continuous_spectrum"] = {}
    for tag, folder_str, illum in BENCH_LEVELS:
        t0 = time.time()
        m_ap, m_ap50 = evaluate_folder(solver, module, folder_str, device)
        el = time.time() - t0
        results["continuous_spectrum"][tag] = {"illum": illum, "mAP": m_ap, "mAP50": m_ap50}
        print(f"  {tag:<10} | illum={illum:<4.2f} | mAP={m_ap:.4f} | mAP50={m_ap50:.4f} | {el:.1f}s")

    out_file = Path("/root/autodl-tmp/AIC/outputs/analysis/nightxx_eval/native_filtering_global_boost_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 85)
    print(f"[+] All tests completed! Saved to {out_file}")
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
