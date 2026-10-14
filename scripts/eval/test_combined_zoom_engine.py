#!/usr/bin/env python3
"""
Benchmark Combined Zoom Architecture:
- Option 1: In-Model Feed-Forward Zoom Engine (Dual-Window Patch Zoom)
- Option 2: Feature-Level Cascade Internalization (P2 RoI Zoom Gate)
- Combined: Option 1 + Option 2 Together
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
from torchvision.ops import batched_nms, box_convert

_root = '/root/autodl-tmp/AIC'
sys.path.insert(0, os.path.join(_root, 'vendor/DEIMv2'))
sys.path.insert(0, os.path.join(_root, 'scripts/eval'))
os.chdir(os.path.join(_root, 'vendor/DEIMv2'))

from test_p2_cascade_zoom_gate import P2CascadeZoomGate, setup_gated_forward, run_evaluation
from engine.core import YAMLConfig
from engine.solver import TASKS
from engine.misc import dist_utils


class InModelFeedForwardZoomEngine(nn.Module):
    """
    Option 1: In-Model Feed-Forward Zoom Engine (前向管线级内化变焦引擎).
    Natively decomposes high-res 16:9 canvas (1152x2048) into:
    1. Global Anchor View (P0, full canvas context)
    2. High-Res Left Optical Zoom View (P1, left 62.5% magnified 1.6x)
    3. High-Res Right Optical Zoom View (P2, right 62.5% magnified 1.6x)
    Integrates Option 2 (P2 Cascade Zoom Gate) on all views.
    Unprojects bounding boxes back to native canvas and applies NMS fusion.
    """
    def __init__(self, model, nms_iou=0.60, num_queries=300, num_classes=12):
        super().__init__()
        self.model = model
        self.nms_iou = nms_iou
        self.num_queries = num_queries
        self.num_classes = num_classes

    def forward(self, x):
        B, C, H, W = x.shape
        # Process image by image to strictly preserve GPU memory with PID 611694 running
        all_final_boxes = []
        all_final_logits = []

        # Zoom slice geometry: left 62.5% and right 62.5% with 512px overlap
        w_crop = int(W * 0.625) # 1280 pixels
        scale_x = w_crop / W    # 0.625
        offset_left = 0.0
        offset_right = (W - w_crop) / W # 0.375

        for b in range(B):
            img_b = x[b:b+1] # [1, C, H, W]

            # 1. Global View P0
            p0 = img_b

            # 2. Left Zoom View P1
            crop_left = img_b[:, :, :, 0:w_crop]
            p1 = F.interpolate(crop_left, size=(H, W), mode='bilinear', align_corners=False)

            # 3. Right Zoom View P2
            crop_right = img_b[:, :, :, (W - w_crop):W]
            p2 = F.interpolate(crop_right, size=(H, W), mode='bilinear', align_corners=False)

            # Sequential forward of 3 views (consumes < 1.5GB VRAM per view)
            views = [p0, p1, p2]
            outs = []
            for v in views:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    depth_v = v[:, 5:6] if v.shape[1] >= 6 else None
                    feats_v = self.model.backbone(v)
                    p2_feat_v = feats_v[0] if (len(feats_v) == 4 and getattr(self.model, 'use_p2_cascade_zoom', False)) else None
                    enc_v = self.model.encoder(feats_v, depth=depth_v)
                    out_v = self.model.decoder(enc_v)

                    if self.model.use_scene_gate and self.model.scene_gate is not None and depth_v is not None:
                        gate_v = self.model.scene_gate(depth_v)
                        out_v['pred_logits'] = out_v['pred_logits'] + gate_v.unsqueeze(1)

                    # Option 2 applied to each view
                    p2_gate = getattr(self.model, 'p2_cascade_gate', None)
                    if p2_gate is not None and p2_feat_v is not None:
                        out_v['pred_logits'] = p2_gate(p2_feat_v, out_v['pred_boxes'], out_v['pred_logits'])

                    outs.append(out_v)

            out_p0, out_p1, out_p2 = outs

            # Unproject coordinates
            b0 = out_p0['pred_boxes'][0] # [300, 4]
            l0 = out_p0['pred_logits'][0]

            # Left zoom unprojection
            b1 = out_p1['pred_boxes'][0].clone()
            b1[:, 0] = offset_left + b1[:, 0] * scale_x
            b1[:, 2] = b1[:, 2] * scale_x
            l1 = out_p1['pred_logits'][0]

            # Right zoom unprojection
            b2 = out_p2['pred_boxes'][0].clone()
            b2[:, 0] = offset_right + b2[:, 0] * scale_x
            b2[:, 2] = b2[:, 2] * scale_x
            l2 = out_p2['pred_logits'][0]

            # Boundary artifact filtering
            # Discard detections clipped directly by the crop boundary (right edge of P1, left edge of P2)
            xyxy1 = box_convert(b1, in_fmt="cxcywh", out_fmt="xyxy")
            xyxy2 = box_convert(b2, in_fmt="cxcywh", out_fmt="xyxy")

            keep1 = xyxy1[:, 2] < (w_crop / W - 0.015)
            keep2 = xyxy2[:, 0] > (offset_right + 0.015)

            # Combine candidates from all 3 views
            cand_boxes = torch.cat([b0, b1[keep1], b2[keep2]], dim=0)
            cand_logits = torch.cat([l0, l1[keep1], l2[keep2]], dim=0)

            # NMS fusion
            cand_xyxy = box_convert(cand_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            cand_scores, cand_labels = torch.sigmoid(cand_logits).max(dim=-1)

            keep_nms = batched_nms(cand_xyxy, cand_scores, cand_labels, iou_threshold=self.nms_iou)
            keep_nms = keep_nms[:self.num_queries]

            if len(keep_nms) < self.num_queries:
                pad_num = self.num_queries - len(keep_nms)
                b_pad = torch.zeros(pad_num, 4, device=x.device)
                l_pad = torch.full((pad_num, self.num_classes), -10.0, device=x.device)
                b_res = torch.cat([cand_boxes[keep_nms], b_pad], dim=0)
                l_res = torch.cat([cand_logits[keep_nms], l_pad], dim=0)
            else:
                b_res = cand_boxes[keep_nms]
                l_res = cand_logits[keep_nms]

            all_final_boxes.append(b_res)
            all_final_logits.append(l_res)

        return {
            'pred_boxes': torch.stack(all_final_boxes, dim=0),
            'pred_logits': torch.stack(all_final_logits, dim=0)
        }


def main():
    print("=" * 85)
    print(" Dual-Zoom Architecture Benchmark: Option 1 + Option 2 Together ".center(85, "="))
    print("=" * 85)

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
    module.use_internal_tta = False

    orig_module_forward = module.forward

    # 1. Baseline Evaluation (No Zoom Engine, No P2 Gate)
    print("\n[1/4] Running Baseline Evaluation (Standard Forward)...")
    module.use_p2_cascade_zoom = False
    module.p2_cascade_gate = None
    t0 = time.time()
    res_base = run_evaluation(solver, module, solver.device)
    print(f"[-] Baseline: mAP@50:95: {res_base['mAP50_95']:.4f} | mAP@50: {res_base['mAP50']:.4f} | AP_s: {res_base['AP_small']:.4f} ({time.time()-t0:.1f}s)")

    # 2. Option 2 Only (Feature-Level Cascade RoI Zoom Gate)
    print("\n[2/4] Running Option 2 Only (P2 Cascade RoI Zoom Gate)...")
    gate = P2CascadeZoomGate(score_min=0.15, score_max=0.45, max_area=0.02, saliency_thresh=0.40, alpha_boost=0.50, beta_suppress=0.0).cuda()
    module.use_p2_cascade_zoom = True
    module.p2_cascade_gate = gate
    t0 = time.time()
    res_opt2 = run_evaluation(solver, module, solver.device)
    print(f"[-] Option 2 Only: mAP@50:95: {res_opt2['mAP50_95']:.4f} (Δ {res_opt2['mAP50_95']-res_base['mAP50_95']:+.4f}) | mAP@50: {res_opt2['mAP50']:.4f} | AP_s: {res_opt2['AP_small']:.4f} ({time.time()-t0:.1f}s)")

    # 3. Option 1 Only (In-Model Feed-Forward Zoom Engine, No Gate)
    print("\n[3/4] Running Option 1 Only (In-Model Feed-Forward Zoom Engine, Gate=OFF)...")
    module.use_p2_cascade_zoom = False
    module.p2_cascade_gate = None
    zoom_engine = InModelFeedForwardZoomEngine(module, nms_iou=0.60)
    module.forward = zoom_engine.forward
    t0 = time.time()
    res_opt1 = run_evaluation(solver, module, solver.device)
    print(f"[-] Option 1 Only: mAP@50:95: {res_opt1['mAP50_95']:.4f} (Δ {res_opt1['mAP50_95']-res_base['mAP50_95']:+.4f}) | mAP@50: {res_opt1['mAP50']:.4f} | AP_s: {res_opt1['AP_small']:.4f} ({time.time()-t0:.1f}s)")

    # 4. Combined: Option 1 + Option 2 Together
    print("\n[4/4] Running Combined: Option 1 (Feed-Forward Zoom) + Option 2 (P2 RoI Gate)...")
    module.use_p2_cascade_zoom = True
    module.p2_cascade_gate = gate
    t0 = time.time()
    res_combined = run_evaluation(solver, module, solver.device)
    print(f"[-] Combined (Opt1 + Opt2): mAP@50:95: {res_combined['mAP50_95']:.4f} (Δ {res_combined['mAP50_95']-res_base['mAP50_95']:+.4f}) | mAP@50: {res_combined['mAP50']:.4f} | AP_s: {res_combined['AP_small']:.4f} ({time.time()-t0:.1f}s)")

    module.forward = orig_module_forward

    print("\n" + "=" * 90)
    print(" COMPLETE ABLATION COMPARISON: DUAL-ZOOM ARCHITECTURE ".center(90, "="))
    print("=" * 90)
    print(f"{'Method / Configuration':<52} | {'mAP@50:95':<10} | {'mAP@50':<10} | {'AP_small':<10} | {'Delta':<8}")
    print("-" * 90)
    print(f"{'Baseline (Standard Forward)':<52} | {res_base['mAP50_95']:<10.4f} | {res_base['mAP50']:<10.4f} | {res_base['AP_small']:<10.4f} | {'+0.0000':<8}")
    print(f"{'Option 2 (Feature-Level P2 RoI Gate)':<52} | {res_opt2['mAP50_95']:<10.4f} | {res_opt2['mAP50']:<10.4f} | {res_opt2['AP_small']:<10.4f} | {res_opt2['mAP50_95']-res_base['mAP50_95']:<+8.4f}")
    print(f"{'Option 1 (In-Model Feed-Forward Zoom Engine)':<52} | {res_opt1['mAP50_95']:<10.4f} | {res_opt1['mAP50']:<10.4f} | {res_opt1['AP_small']:<10.4f} | {res_opt1['mAP50_95']-res_base['mAP50_95']:<+8.4f}")
    print(f"{'Combined (Option 1 + Option 2 Together)':<52} | {res_combined['mAP50_95']:<10.4f} | {res_combined['mAP50']:<10.4f} | {res_combined['AP_small']:<10.4f} | {res_combined['mAP50_95']-res_base['mAP50_95']:<+8.4f}")
    print("=" * 90)

    out_file = "/root/autodl-tmp/AIC/outputs/analysis/nightxx_eval/test_combined_zoom_engine.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": res_base,
            "option2": res_opt2,
            "option1": res_opt1,
            "combined": res_combined
        }, f, indent=2)
    print(f"[+] Saved results to {out_file}")

    dist_utils.cleanup()


if __name__ == '__main__':
    main()
