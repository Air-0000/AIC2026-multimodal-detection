#!/usr/bin/env python3
"""
Test Feature-Level Cascade Internalization (Option 2: 特征层级级联内化)
Zero-overhead high-resolution zoom verification on Stage 0 (P2, 288x512) feature maps:
1. Extract RoIAlign features from P2 for decoder candidate queries.
2. Evaluate Physical Saliency & High-Frequency Spatial SNR Gate.
3. Benchmark Baseline vs P2 Cascade Zoom Gate across mAP50:95, AP_small, and per-class AP.
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
from torchvision.ops import roi_align, box_convert

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


class P2CascadeZoomGate(nn.Module):
    """
    Feature-Level Cascade RoI Zoom Gate.
    Operates directly on Backbone Stage 0 (P2, stride 4, C=128).
    Extracts 7x7 RoIAlign patches for candidate queries.
    Computes spatial center-to-surround contrast and high-frequency saliency.
    Calibrates ambiguous queries with positive boost for real targets and suppression for phantoms.
    """
    def __init__(self,
                 roi_size=7,
                 score_min=0.15,
                 score_max=0.50,
                 max_area=0.05,
                 saliency_thresh=0.30,
                 alpha_boost=0.4,
                 beta_suppress=0.6,
                 use_mlp=False,
                 num_classes=12,
                 p2_channels=128):
        super().__init__()
        self.roi_size = roi_size
        self.score_min = score_min
        self.score_max = score_max
        self.max_area = max_area
        self.saliency_thresh = saliency_thresh
        self.alpha_boost = alpha_boost
        self.beta_suppress = beta_suppress
        self.use_mlp = use_mlp
        self.num_classes = num_classes

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(p2_channels, 64),
                nn.GELU(),
                nn.Linear(64, num_classes)
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            self.mlp = None

    def compute_saliency(self, roi_feats):
        """
        roi_feats: [N, C, 7, 7]
        Center: 3x3 region (y: 2..5, x: 2..5) -> 9 cells
        Surround: outer 40 cells
        Returns scalar saliency score per RoI [N]
        """
        N, C, H, W = roi_feats.shape
        energy = torch.norm(roi_feats, dim=1) # [N, 7, 7]

        center = energy[:, 2:5, 2:5].mean(dim=(-2, -1)) # [N]
        total_sum = energy.sum(dim=(-2, -1)) # [N]
        center_sum = energy[:, 2:5, 2:5].sum(dim=(-2, -1)) # [N]
        surround = (total_sum - center_sum) / 40.0 # [N]

        std = energy.std(dim=(-2, -1)) + 1e-5 # [N]
        contrast = (center - surround) / std # [N]
        return contrast

    def forward(self, p2_feat, pred_boxes, pred_logits):
        """
        p2_feat: [B, 128, H_p2, W_p2]
        pred_boxes: [B, 300, 4] in (cx, cy, w, h) normalized [0, 1]
        pred_logits: [B, 300, num_classes]
        Returns calibrated pred_logits [B, 300, num_classes]
        """
        B, Q, num_cls = pred_logits.shape
        H_p2, W_p2 = p2_feat.shape[-2:]

        # Query scores
        scores = torch.sigmoid(pred_logits)
        max_scores, best_cls = scores.max(dim=-1) # [B, Q]

        # Calculate box area: w * h
        box_area = pred_boxes[..., 2] * pred_boxes[..., 3] # [B, Q]

        # Target mask: queries in ambiguous range and small/medium scale
        target_mask = (max_scores >= self.score_min) & (max_scores <= self.score_max) & (box_area <= self.max_area)

        if not target_mask.any():
            return pred_logits

        # Convert boxes to xyxy in P2 coordinate space
        cx = pred_boxes[..., 0] * W_p2
        cy = pred_boxes[..., 1] * H_p2
        w = pred_boxes[..., 2] * W_p2
        h = pred_boxes[..., 3] * H_p2

        x1 = torch.clamp(cx - w / 2, min=0.0, max=W_p2 - 1.0)
        y1 = torch.clamp(cy - h / 2, min=0.0, max=H_p2 - 1.0)
        x2 = torch.clamp(cx + w / 2, min=1.0, max=W_p2)
        y2 = torch.clamp(cy + h / 2, min=1.0, max=H_p2)

        # Batch indices
        batch_idx = torch.arange(B, device=p2_feat.device).unsqueeze(1).repeat(1, Q) # [B, Q]

        # Filter only target queries for RoIAlign
        b_idx_sel = batch_idx[target_mask]
        x1_sel = x1[target_mask]
        y1_sel = y1[target_mask]
        x2_sel = x2[target_mask]
        y2_sel = y2[target_mask]

        rois = torch.stack([b_idx_sel.float(), x1_sel, y1_sel, x2_sel, y2_sel], dim=1) # [K, 5]

        roi_feats = roi_align(
            p2_feat,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True
        ) # [K, 128, 7, 7]

        saliency = self.compute_saliency(roi_feats) # [K]

        delta_logits = torch.zeros_like(pred_logits) # [B, Q, num_cls]
        best_cls_sel = best_cls[target_mask] # [K]

        diff = saliency - self.saliency_thresh
        boost = torch.where(diff > 0,
                            self.alpha_boost * torch.clamp(diff, max=1.2),
                            -self.beta_suppress * torch.clamp(-diff, max=1.5))

        if self.use_mlp and self.mlp is not None:
            pooled = F.adaptive_avg_pool2d(roi_feats, 1).flatten(1) # [K, 128]
            mlp_delta = self.mlp(pooled) # [K, num_cls]
            delta_logits[target_mask] = delta_logits[target_mask] + mlp_delta
        else:
            target_indices = torch.nonzero(target_mask, as_tuple=True)
            delta_logits[target_indices[0], target_indices[1], best_cls_sel] += boost

        return pred_logits + delta_logits


def setup_gated_forward(model):
    """
    Wrap model forward once to look for model.p2_gate
    """
    if getattr(model, '_has_p2_wrap', False):
        return

    orig_forward = model.forward

    def gated_forward(x, targets=None):
        depth = x[:, 5:6] if (x.shape[1] >= 6) else None
        feats = model.backbone(x)
        p2_feat = feats[0] if len(feats) == 4 else None

        enc_feats = model.encoder(feats, depth=depth)
        out = model.decoder(enc_feats, targets)

        if model.use_scene_gate and model.scene_gate is not None and depth is not None:
            gate = model.scene_gate(depth)
            out['pred_logits'] = out['pred_logits'] + gate.unsqueeze(1)

        p2_gate = getattr(model, 'p2_gate', None)
        if p2_gate is not None and p2_feat is not None:
            out['pred_logits'] = p2_gate(p2_feat, out['pred_boxes'], out['pred_logits'])

        return out

    model.forward = gated_forward
    model._has_p2_wrap = True


def run_evaluation(solver, module, device):
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

    # Per-category stats
    per_cat_ap = {}
    cat_ids = coco_evaluator.coco_eval["bbox"].params.catIds
    for idx, cat_id in enumerate(cat_ids):
        cat_name = solver.val_dataloader.dataset.coco.loadCats(cat_id)[0]['name']
        prec = coco_evaluator.coco_eval["bbox"].eval['precision'][:, :, idx, 0, 2]
        prec = prec[prec > -1]
        ap = float(prec.mean()) if len(prec) > 0 else 0.0
        per_cat_ap[cat_name] = ap

    return {
        "mAP50_95": stats[0],
        "mAP50": stats[1],
        "mAP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "per_cat": per_cat_ap
    }


def main():
    print("=" * 85)
    print(" Option 2: Feature-Level Cascade RoI Zoom Gate Benchmark ".center(85, "="))
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
    device = solver.device

    # Ensure use_internal_tta is false during gate evaluation for pure measurement
    module.use_internal_tta = False
    setup_gated_forward(module)

    print("\n[+] 1. Running Pure Baseline Evaluation (No Gate)...")
    module.p2_gate = None
    t0 = time.time()
    base_res = run_evaluation(solver, module, device)
    print(f"[-] Baseline: mAP@50:95: {base_res['mAP50_95']:.4f} | mAP@50: {base_res['mAP50']:.4f} | AP_s: {base_res['AP_small']:.4f} ({time.time()-t0:.1f}s)")

    # Grid of candidate configurations calibrated to the empirical saliency stats (mean ~0.33, std ~0.67)
    candidate_configs = [
        {"desc": "Gentle Precision Filter (alpha=0.0, beta=0.6, thresh=0.0)",
         "score_min": 0.15, "score_max": 0.45, "saliency_thresh": 0.0, "alpha_boost": 0.0, "beta_suppress": 0.6, "max_area": 0.04},
        {"desc": "Balanced Zoom Peak Boost (alpha=0.3, beta=0.5, thresh=0.3)",
         "score_min": 0.15, "score_max": 0.45, "saliency_thresh": 0.3, "alpha_boost": 0.3, "beta_suppress": 0.5, "max_area": 0.04},
        {"desc": "High-Sharpness Boost (alpha=0.4, beta=0.8, thresh=0.4)",
         "score_min": 0.15, "score_max": 0.50, "saliency_thresh": 0.4, "alpha_boost": 0.4, "beta_suppress": 0.8, "max_area": 0.03},
        {"desc": "Micro-Target Focus (alpha=0.5, beta=1.0, thresh=0.2, max_area=0.015)",
         "score_min": 0.12, "score_max": 0.45, "saliency_thresh": 0.2, "alpha_boost": 0.5, "beta_suppress": 1.0, "max_area": 0.015},
        {"desc": "Ultra-Subtle Calibration (alpha=0.15, beta=0.25, thresh=0.25)",
         "score_min": 0.20, "score_max": 0.40, "saliency_thresh": 0.25, "alpha_boost": 0.15, "beta_suppress": 0.25, "max_area": 0.03},
    ]

    all_results = [{"Config": "Baseline (No Gate)", "mAP50_95": base_res['mAP50_95'], "mAP50": base_res['mAP50'], "AP_s": base_res['AP_small'], "Delta": 0.0, "res": base_res}]

    best_gate_cfg = None
    best_map = base_res['mAP50_95']

    for c in candidate_configs:
        print(f"\n[+] Testing Gate: {c['desc']}...")
        gate = P2CascadeZoomGate(
            score_min=c['score_min'],
            score_max=c['score_max'],
            max_area=c['max_area'],
            saliency_thresh=c['saliency_thresh'],
            alpha_boost=c['alpha_boost'],
            beta_suppress=c['beta_suppress'],
            use_mlp=False
        ).to(device)

        module.p2_gate = gate
        t_start = time.time()
        g_res = run_evaluation(solver, module, device)
        delta = g_res['mAP50_95'] - base_res['mAP50_95']
        print(f"[-] Result: mAP@50:95: {g_res['mAP50_95']:.4f} (Δ {delta:+.4f}) | mAP@50: {g_res['mAP50']:.4f} | AP_s: {g_res['AP_small']:.4f} ({time.time()-t_start:.1f}s)")

        all_results.append({
            "Config": c['desc'],
            "mAP50_95": g_res['mAP50_95'],
            "mAP50": g_res['mAP50'],
            "AP_s": g_res['AP_small'],
            "Delta": delta,
            "res": g_res
        })

        if g_res['mAP50_95'] > best_map:
            best_map = g_res['mAP50_95']
            best_gate_cfg = c

    module.p2_gate = None

    print("\n" + "=" * 90)
    print(" SUMMARY COMPARISON: Option 2 (Feature-Level Cascade Internalization) ".center(90, "="))
    print("=" * 90)
    print(f"{'Configuration':<48} | {'mAP@50:95':<10} | {'mAP@50':<10} | {'AP_small':<10} | {'Delta':<8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['Config']:<48} | {r['mAP50_95']:<10.4f} | {r['mAP50']:<10.4f} | {r['AP_s']:<10.4f} | {r['Delta']:<+8.4f}")
    print("=" * 90)

    # Save summary json
    out_json = "/root/autodl-tmp/AIC/outputs/analysis/nightxx_eval/test_p2_cascade_zoom_gate.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[+] Results saved to {out_json}")

    dist_utils.cleanup()


if __name__ == '__main__':
    main()
