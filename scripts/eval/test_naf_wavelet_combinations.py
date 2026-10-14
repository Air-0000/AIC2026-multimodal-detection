#!/usr/bin/env python3
"""
Benchmark All Combinations of:
1. Frontend Lightweight NAFBlock (CVPR 2022 Image Restoration)
2. P2 Feature-Level Wavelet High-Frequency StarOperation Injection (1.25x Super-Resolution)
3. Combined Full System with P2 Cascade RoI Zoom Gate + TTA
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

_root = '/root/autodl-tmp/AIC'
sys.path.insert(0, os.path.join(_root, 'vendor/DEIMv2'))
sys.path.insert(0, os.path.join(_root, 'scripts/eval'))
os.chdir(os.path.join(_root, 'vendor/DEIMv2'))

from test_p2_cascade_zoom_gate import P2CascadeZoomGate, run_evaluation
from engine.core import YAMLConfig
from engine.solver import TASKS
from engine.misc import dist_utils
from engine.deim.sovereign_v13_modules import StarOperationConv2d


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class NAFBlock(nn.Module):
    """Nonlinear Activation Free Block for Multi-Modal Quality Restoration (CVPR 2022)."""
    def __init__(self, in_channels=6, mid_channels=32):
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
        self.sg_channels = mid_channels // 2
        self.sca_pool = nn.AdaptiveAvgPool2d(1)
        self.sca_conv = nn.Conv2d(self.sg_channels, self.sg_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.sg_channels, in_channels, kernel_size=1)
        
        # Zero initialization for clean identity at start
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.dwconv(h)
        h1, h2 = torch.split(h, self.sg_channels, dim=1)
        h = h1 * h2
        attn = self.sca_conv(self.sca_pool(h))
        h = h * torch.sigmoid(attn)
        h = self.conv2(h)
        return x + h


class FrontendNAFRestoration(nn.Module):
    """Stacked 2x NAFBlock Multi-Modal Denoising and Deblurring Head (0.002M params)."""
    def __init__(self, in_channels=6, mid_channels=32, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([NAFBlock(in_channels, mid_channels) for _ in range(num_blocks)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class WaveletHighFreqStarInjection(nn.Module):
    """
    P2 Feature-Level Wavelet High-Frequency StarOperation Injection (1.25x Super-Resolution).
    Separates P2 into LL and [LH, HL, HH], modulates and magnifies details by 1.25x via StarOperation,
    and applies high-frequency residual sharpening to Stage 0 features.
    """
    def __init__(self, channels=128, scale=1.25, blend_weight=0.20):
        super().__init__()
        self.channels = channels
        self.scale = scale
        self.blend_weight = blend_weight
        self.high_channels = channels * 3 # LH, HL, HH
        self.star_conv = StarOperationConv2d(dim=self.high_channels, mlp_ratio=1.5)

    def dwt_haar(self, x):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        
        ll = x1 + x2 + x3 + x4
        lh = -x1 - x3 + x2 + x4
        hl = -x1 + x3 - x2 + x4
        hh = x1 - x3 - x2 + x4
        return ll, lh, hl, hh

    def idwt_haar(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        x1 = (ll - lh - hl + hh) / 2.0
        x2 = (ll + lh - hl - hh) / 2.0
        x3 = (ll - lh + hl - hh) / 2.0
        x4 = (ll + lh + hl + hh) / 2.0

        out = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x1
        out[:, :, 1::2, 0::2] = x2
        out[:, :, 0::2, 1::2] = x3
        out[:, :, 1::2, 1::2] = x4
        return out

    def forward(self, p2):
        ll, lh, hl, hh = self.dwt_haar(p2)
        high = torch.cat([lh, hl, hh], dim=1)
        high_mod = self.star_conv(high) * self.scale
        lh_m, hl_m, hh_m = torch.split(high_mod, self.channels, dim=1)
        p2_enhanced = self.idwt_haar(ll, lh_m, hl_m, hh_m)
        return p2 + self.blend_weight * (p2_enhanced - p2)


def main():
    print("=" * 90)
    print(" Permutation Benchmark: NAFBlock + Wavelet Star Super-Resolution ".center(90, "="))
    print("=" * 90)

    cfg = YAMLConfig('configs/deimv2/sovereign_v24_night_robust_tadl_2048.yml',
                     resume='outputs/sovereign_v24_night_robust_tadl_2048/best_stg2_p2cascade.pth')
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 1
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 1
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
    device = solver.device

    # Ensure native TTA + P2 Cascade Zoom Gate are active as base
    module.use_internal_tta = True
    module.use_p2_cascade_zoom = True

    # 1. Base Pinnacle (TTA + P2 Cascade Gate)
    print("\n[Combo 0] Evaluating Base Pinnacle (TTA + P2 Cascade Gate)...")
    t0 = time.time()
    res_base = run_evaluation(solver, module, device)
    base_map = res_base['mAP50_95']
    base_map50 = res_base['mAP50']
    base_aps = res_base['AP_small']
    print(f"[-] Base Pinnacle: mAP={base_map:.4f} | mAP50={base_map50:.4f} | AP_s={base_aps:.4f} ({time.time()-t0:.1f}s)")

    # 2. Combo 1: Base + Frontend NAFBlock
    print("\n[Combo 1] Evaluating Base + Frontend NAFBlock (Image Quality Restoration)...")
    naf_head = FrontendNAFRestoration(in_channels=6, mid_channels=32, num_blocks=2).to(device)

    # Wrap forward to pass x through NAFBlock first
    orig_forward_1 = module.forward
    def forward_with_naf(x, targets=None):
        x_clean = naf_head(x)
        return orig_forward_1(x_clean, targets)
    module.forward = forward_with_naf

    t0 = time.time()
    res_naf = run_evaluation(solver, module, device)
    d_map_1 = res_naf['mAP50_95'] - base_map
    d_50_1 = res_naf['mAP50'] - base_map50
    d_aps_1 = res_naf['AP_small'] - base_aps
    print(f"[-] Combo 1 (NAFBlock): mAP={res_naf['mAP50_95']:.4f} ({d_map_1:+.4f}) | mAP50={res_naf['mAP50']:.4f} ({d_50_1:+.4f}) | AP_s={res_naf['AP_small']:.4f} ({d_aps_1:+.4f}) ({time.time()-t0:.1f}s)")
    module.forward = orig_forward_1

    # 3. Combo 2: Base + Wavelet High-Frequency StarConv Injection on P2
    print("\n[Combo 2] Evaluating Base + P2 Wavelet StarConv High-Frequency Injection (1.25x)...")
    wavelet_sr = WaveletHighFreqStarInjection(channels=128, scale=1.25, blend_weight=0.15).to(device)

    orig_backbone_forward = module.backbone.forward
    def backbone_with_wavelet_sr(x):
        feats = orig_backbone_forward(x)
        if len(feats) >= 1:
            feats[0] = wavelet_sr(feats[0])
        return feats
    module.backbone.forward = backbone_with_wavelet_sr

    t0 = time.time()
    res_wavelet = run_evaluation(solver, module, device)
    d_map_2 = res_wavelet['mAP50_95'] - base_map
    d_50_2 = res_wavelet['mAP50'] - base_map50
    d_aps_2 = res_wavelet['AP_small'] - base_aps
    print(f"[-] Combo 2 (Wavelet SR): mAP={res_wavelet['mAP50_95']:.4f} ({d_map_2:+.4f}) | mAP50={res_wavelet['mAP50']:.4f} ({d_50_2:+.4f}) | AP_s={res_wavelet['AP_small']:.4f} ({d_aps_2:+.4f}) ({time.time()-t0:.1f}s)")
    module.backbone.forward = orig_backbone_forward

    # 4. Combo 3: Base + NAFBlock + Wavelet StarConv (Full Fusion)
    print("\n[Combo 3] Evaluating Base + NAFBlock + P2 Wavelet StarConv (Full Fusion)...")
    module.forward = forward_with_naf
    module.backbone.forward = backbone_with_wavelet_sr

    t0 = time.time()
    res_full = run_evaluation(solver, module, device)
    d_map_3 = res_full['mAP50_95'] - base_map
    d_50_3 = res_full['mAP50'] - base_map50
    d_aps_3 = res_full['AP_small'] - base_aps
    print(f"[-] Combo 3 (Full Fusion): mAP={res_full['mAP50_95']:.4f} ({d_map_3:+.4f}) | mAP50={res_full['mAP50']:.4f} ({d_50_3:+.4f}) | AP_s={res_full['AP_small']:.4f} ({d_aps_3:+.4f}) ({time.time()-t0:.1f}s)")

    # Restore
    module.forward = orig_forward_1
    module.backbone.forward = orig_backbone_forward

    print("\n" + "=" * 95)
    print(" SUMMARY PERMUTATION MATRIX: NAFBLOCK + WAVELET STAR SUPER-RESOLUTION ".center(95, "="))
    print("=" * 95)
    print(f"{'Combination':<48} | {'mAP@50:95':<10} | {'mAP@50':<10} | {'AP_small':<10} | {'Delta':<8}")
    print("-" * 95)
    print(f"{'Combo 0: Base Pinnacle (TTA + P2 Gate)':<48} | {base_map:<10.4f} | {base_map50:<10.4f} | {base_aps:<10.4f} | {'+0.0000':<8}")
    print(f"{'Combo 1: Base + Frontend NAFBlock':<48} | {res_naf['mAP50_95']:<10.4f} | {res_naf['mAP50']:<10.4f} | {res_naf['AP_small']:<10.4f} | {d_map_1:<+8.4f}")
    print(f"{'Combo 2: Base + P2 Wavelet StarConv (1.25x)':<48} | {res_wavelet['mAP50_95']:<10.4f} | {res_wavelet['mAP50']:<10.4f} | {res_wavelet['AP_small']:<10.4f} | {d_map_2:<+8.4f}")
    print(f"{'Combo 3: Base + NAFBlock + Wavelet Star (Full)':<48} | {res_full['mAP50_95']:<10.4f} | {res_full['mAP50']:<10.4f} | {res_full['AP_small']:<10.4f} | {d_map_3:<+8.4f}")
    print("=" * 95)

    out_file = "/root/autodl-tmp/AIC/outputs/analysis/nightxx_eval/test_naf_wavelet_combinations.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "combo0": res_base,
            "combo1": res_naf,
            "combo2": res_wavelet,
            "combo3": res_full
        }, f, indent=2)
    print(f"[+] Saved results to {out_file}")

    dist_utils.cleanup()


if __name__ == '__main__':
    main()
