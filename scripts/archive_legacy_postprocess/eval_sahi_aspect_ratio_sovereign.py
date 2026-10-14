#!/usr/bin/env python3
"""
Sovereign Multimodal SAHI (Slicing Aided Hyper Inference) + USM + IR CLAHE + Depth-Guided Thermal Boost
========================================================================================================
Key Capabilities:
1. Native 16:9 Aspect Ratio: Eliminates the 1.78x non-uniform vertical stretching distortion.
   Scales native 16:9 canvas to 1138x640 with uniform isotropic scaling factor.
2. SAHI Dual-Patch Slicing with Seam Filtering and Large-Object Gating:
   - Patch 1 (Left): [0:640]
   - Patch 2 (Right): [498:1138]
   - Patch 0 (Global): [640x640] for holistic scene context
   Small objects are magnified by 3.16x in pixel area on the 640x640 receptive field!
3. Front-end Unsharp Masking (USM): Accentuates high-frequency edges for low-contrast infrared targets.
4. Infrared CLAHE (Contrast Limited Adaptive Histogram Equalization):
   Stretches compressed thermal dynamic range in dark/shadow regions without blowing out hot spots.
5. Depth-Guided Thermal Boost (Cross-Modal Thermal Co-Enhancement):
   Extracts 3D physical boundaries from metric depth gradient to boost thermal edges along true silhouettes,
   completely suppressing thermal noise in flat surfaces (ground, sky).
6. Uses official CocoEvaluator from DEIMv2 for exact evaluation matching.
"""

import os
import sys
import copy
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# Setup project paths
AIC_ROOT = Path('/root/autodl-tmp/AIC')
if not AIC_ROOT.exists():
    AIC_ROOT = Path('/home/constant/CodeBuddy/Dev/AIC')

DEIMV2_ROOT = AIC_ROOT / 'vendor/DEIMv2'
sys.path.insert(0, str(AIC_ROOT / 'scripts/lib'))
sys.path.insert(0, str(DEIMV2_ROOT))
from engine.core import YAMLConfig
from engine.data.dataset.coco_eval import CocoEvaluator


def apply_ir_clahe(tensor_6ch, clip_limit=1.5, tile_size=8, blend_alpha=0.5):
    """
    Applies CLAHE on Infrared channels (channel 3: aligned IR, channel 4: raw LBG IR).
    Blends with original to prevent noise amplification:
    I_out = (1 - alpha) * I_orig + alpha * I_clahe.
    """
    device = tensor_6ch.device
    ir1 = tensor_6ch[3].cpu().numpy()
    ir2 = tensor_6ch[4].cpu().numpy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    c1 = clahe.apply((ir1 * 255.0).clip(0, 255).astype(np.uint8)).astype(np.float32) / 255.0
    c2 = clahe.apply((ir2 * 255.0).clip(0, 255).astype(np.uint8)).astype(np.float32) / 255.0

    out = tensor_6ch.clone()
    out[3] = torch.from_numpy((1.0 - blend_alpha) * ir1 + blend_alpha * c1).to(device)
    out[4] = torch.from_numpy((1.0 - blend_alpha) * ir2 + blend_alpha * c2).to(device)
    return out


def apply_depth_guided_thermal_boost(tensor_6ch, boost_strength=0.35, grad_thresh=0.015):
    """
    Depth-Guided Thermal Boost:
    Computes spatial gradient magnitude on metric depth (channel 5).
    Where a 3D physical object boundary exists (grad > grad_thresh),
    selectively boosts thermal edge contrast in the IR channels (channels 3 and 4).
    In flat depth regions, zero boost is applied to prevent thermal noise hallucination.
    """
    device = tensor_6ch.device
    depth = tensor_6ch[5:6].unsqueeze(0)  # [1, 1, H, W]

    # Sobel kernels for depth gradient
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3) / 8.0

    gx = F.conv2d(depth, kx, padding=1)
    gy = F.conv2d(depth, ky, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-8)  # [1, 1, H, W]

    # Normalize depth edge mask: 0 in flat surfaces, up to 1.0 along depth steps
    edge_mask = torch.clamp((grad - grad_thresh) / 0.03, 0.0, 1.0)[0]  # [1, H, W]

    # 5x5 Gaussian blur kernel for thermal unsharp mask
    k1d = torch.tensor([0.06136, 0.24477, 0.38774, 0.24477, 0.06136], dtype=torch.float32, device=device)
    k2d = torch.outer(k1d, k1d).view(1, 1, 5, 5)

    out = tensor_6ch.clone()
    for ch in [3, 4]:  # Both IR channels
        ir = tensor_6ch[ch:ch+1].unsqueeze(0)  # [1, 1, H, W]
        blurred = F.conv2d(ir, k2d, padding=2)
        diff = ir - blurred  # High-frequency thermal edge
        # Boost ONLY along genuine 3D depth contours
        boosted = ir + boost_strength * edge_mask.unsqueeze(0) * diff
        out[ch] = boosted.clamp(0.0, 1.0)[0, 0]

    return out


def apply_usm(tensor_6ch, strength=0.35):
    """
    Applies Unsharp Masking (USM) on 6-channel tensor [6, H, W] in [0, 1].
    Uses a 5x5 Gaussian blur kernel (sigma=1.0) via PyTorch depthwise 2D conv.
    """
    channels = tensor_6ch.shape[0]
    kernel_1d = torch.tensor([0.06136, 0.24477, 0.38774, 0.24477, 0.06136], dtype=torch.float32, device=tensor_6ch.device)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.repeat(channels, 1, 1, 1)  # [6, 1, 5, 5]

    inp = tensor_6ch.unsqueeze(0)  # [1, 6, H, W]
    blurred = F.conv2d(inp, kernel, padding=2, groups=channels)
    sharp = inp + strength * (inp - blurred)
    return sharp.clamp(0.0, 1.0)[0]


def evaluate_sahi_aspect_ratio(
    cfg_path,
    checkpoint_path,
    use_sahi=True,
    use_usm=True,
    usm_strength=0.35,
    use_clahe=False,
    clahe_clip=1.5,
    clahe_blend=0.5,
    use_depth_boost=False,
    depth_boost_strength=0.35,
    iou_thresh=0.65,
    conf_thresh=0.05,
    seam_margin=25.0,
    max_patch_dim=320.0,
    device='cuda'
):
    cfg_path = str(Path(cfg_path).resolve())
    checkpoint_path = str(Path(checkpoint_path).resolve())

    print("=" * 80)
    print("Sovereign Multimodal SAHI + USM + IR CLAHE + Depth-Boost Evaluator")
    print(f"  Config:     {cfg_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  SAHI (16:9 Native): {use_sahi} (seam_margin={seam_margin}, max_dim={max_patch_dim})")
    print(f"  USM Sharpening:     {use_usm} (strength={usm_strength})")
    print(f"  IR CLAHE:           {use_clahe} (clip={clahe_clip}, blend={clahe_blend})")
    print(f"  Depth-Guided Boost: {use_depth_boost} (strength={depth_boost_strength})")
    print(f"  NMS IoU Threshold:  {iou_thresh}")
    print(f"  Confidence Thresh:  {conf_thresh}")
    print("=" * 80)

    # 1. Load Model and Postprocessor
    cfg = YAMLConfig(str(cfg_path))
    if 'HGNetv2MultiModal' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2MultiModal']['pretrained'] = False

    model = cfg.model.to(device)
    postprocessor = cfg.postprocessor.to(device)

    print(f"[*] Loading checkpoint weights from {checkpoint_path}...")
    state = torch.load(str(checkpoint_path), map_location='cpu')
    if 'ema' in state and 'module' in state['ema']:
        weights = state['ema']['module']
        print("    -> Using EMA weights")
    elif 'model' in state:
        weights = state['model']
        print("    -> Using model weights")
    else:
        weights = state

    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"    -> Loaded. missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    # 2. Get Validation Dataset from YAML config
    val_loader = cfg.val_dataloader
    dataset = val_loader.dataset
    base_coco = dataset.coco
    n_images = len(dataset)
    print(f"[*] Validation dataset ready: {n_images} images.")

    # 3. Normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5], device=device).view(6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5], device=device).view(6, 1, 1)

    # Scaled 16:9 canvas dimensions
    W_canvas = 1138
    H_canvas = 640
    offset_x2 = 1138 - 640  # 498
    offset_p2 = torch.tensor([float(offset_x2), 0.0, float(offset_x2), 0.0], device=device)

    collected_preds = {}
    t0 = time.time()

    with torch.no_grad():
        for idx in range(n_images):
            # load_item gives unresized 6-channel tensor [6, H_orig, W_orig] in [0, 1]
            raw_tensor, target = dataset.load_item(idx)
            raw_tensor = raw_tensor.to(device)
            img_id = target['image_id'].item() if isinstance(target['image_id'], torch.Tensor) else int(target['image_id'])

            # Dynamic original target dimensions
            W_orig = float(target['orig_size'][0].item())
            H_orig = float(target['orig_size'][1].item())
            orig_target_sizes = torch.tensor([[W_orig, H_orig]], dtype=torch.float32, device=device)

            # Processing pipeline on 6-channel input
            img_proc = raw_tensor
            if use_clahe:
                img_proc = apply_ir_clahe(img_proc, clip_limit=clahe_clip, blend_alpha=clahe_blend)
            if use_depth_boost:
                img_proc = apply_depth_guided_thermal_boost(img_proc, boost_strength=depth_boost_strength)
            if use_usm:
                img_proc = apply_usm(img_proc, strength=usm_strength)

            if not use_sahi:
                # -------------------------------------------------------------
                # Standard Baseline: Direct resize to [640, 640]
                # -------------------------------------------------------------
                img_std = F.interpolate(img_proc.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False)[0]
                img_norm = (img_std - mean) / std

                outputs = model(img_norm.unsqueeze(0))
                results = postprocessor(outputs, orig_target_sizes, img_norm.unsqueeze(0))
                res = results[0]

                boxes = res['boxes'].cpu()
                scores = res['scores'].cpu()
                labels = res['labels'].cpu()

            else:
                # -------------------------------------------------------------
                # SAHI + 16:9 Native Aspect Ratio (Dual Patch + Global Context)
                # -------------------------------------------------------------
                # 1. Scale to uniform 16:9 canvas (1138x640)
                img_canvas = F.interpolate(img_proc.unsqueeze(0), size=(H_canvas, W_canvas), mode='bilinear', align_corners=False)[0]

                # 2. Slice patches:
                # Patch 1 (Left 640x640): [0:640]
                p1 = img_canvas[:, :, 0:640]
                # Patch 2 (Right 640x640): [498:1138]
                p2 = img_canvas[:, :, offset_x2:W_canvas]
                # Patch 0 (Global Context 640x640): Direct resize of raw image
                p0 = F.interpolate(img_proc.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False)[0]

                # Batch of 3 patches: [3, 6, 640, 640]
                batch_tensors = torch.stack([p1, p2, p0], dim=0)
                batch_norm = (batch_tensors - mean) / std

                patch_sizes_sahi = torch.tensor([
                    [640.0, 640.0],
                    [640.0, 640.0],
                    [W_orig, H_orig]
                ], dtype=torch.float32, device=device)

                outputs = model(batch_norm)
                results = postprocessor(outputs, patch_sizes_sahi, batch_norm)

                # Scaling ratios from canvas to original image
                scale_x = W_canvas / W_orig
                scale_y = H_canvas / H_orig
                scales = torch.tensor([scale_x, scale_y, scale_x, scale_y], device=device)

                # Decode Patch 1 (Left) with seam margin & max-dim filtering
                b1_raw = results[0]['boxes']
                s1_raw = results[0]['scores']
                l1_raw = results[0]['labels']
                w1_raw = b1_raw[:, 2] - b1_raw[:, 0]
                h1_raw = b1_raw[:, 3] - b1_raw[:, 1]
                valid_p1 = (b1_raw[:, 2] < (640.0 - seam_margin)) & (w1_raw < max_patch_dim) & (h1_raw < max_patch_dim)
                b1 = (b1_raw[valid_p1]) / scales
                s1 = s1_raw[valid_p1]
                l1 = l1_raw[valid_p1]

                # Decode Patch 2 (Right) with seam margin & max-dim filtering
                b2_raw = results[1]['boxes']
                s2_raw = results[1]['scores']
                l2_raw = results[1]['labels']
                w2_raw = b2_raw[:, 2] - b2_raw[:, 0]
                h2_raw = b2_raw[:, 3] - b2_raw[:, 1]
                valid_p2 = (b2_raw[:, 0] > seam_margin) & (w2_raw < max_patch_dim) & (h2_raw < max_patch_dim)
                b2 = (b2_raw[valid_p2] + offset_p2) / scales
                s2 = s2_raw[valid_p2]
                l2 = l2_raw[valid_p2]

                # Decode Patch 0 (Global) -> already mapped to [W_orig, H_orig]
                b0 = results[2]['boxes']
                s0 = results[2]['scores']
                l0 = results[2]['labels']

                # Combine all predictions
                all_boxes = torch.cat([b1, b2, b0], dim=0)
                all_scores = torch.cat([s1, s2, s0], dim=0)
                all_labels = torch.cat([l1, l2, l0], dim=0)

                # Clamp boxes to original image boundary
                all_boxes[:, 0::2].clamp_(0.0, W_orig)
                all_boxes[:, 1::2].clamp_(0.0, H_orig)

                # Filter low-confidence noise
                conf_mask = all_scores >= conf_thresh
                all_boxes = all_boxes[conf_mask]
                all_scores = all_scores[conf_mask]
                all_labels = all_labels[conf_mask]

                # Batched NMS across classes
                if len(all_boxes) > 0:
                    keep = torchvision.ops.batched_nms(all_boxes, all_scores, all_labels, iou_threshold=iou_thresh)
                    keep = keep[:300]
                    boxes = all_boxes[keep].cpu()
                    scores = all_scores[keep].cpu()
                    labels = all_labels[keep].cpu()
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    scores = torch.zeros((0,), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)

            collected_preds[img_id] = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            }

            if (idx + 1) % 50 == 0 or (idx + 1) == n_images:
                elapsed = time.time() - t0
                print(f"  [{idx+1:3d}/{n_images}] processed in {elapsed:.1f}s ({elapsed/(idx+1):.3f}s/img)")

    # 4. Standard COCO Evaluation
    print("\n" + "=" * 50 + " COCO EVALUATION " + "=" * 50)
    evaluator = CocoEvaluator(base_coco, ['bbox'])
    evaluator.cleanup()
    evaluator.update(collected_preds)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.coco_eval['bbox'].stats
    print(f"\nFinal Summary: mAP={stats[0]:.4f} | AP50={stats[1]:.4f} | AP75={stats[2]:.4f} | APs={stats[3]:.4f} | APm={stats[4]:.4f} | APl={stats[5]:.4f}")
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sovereign SAHI + USM + CLAHE + Depth-Boost Evaluation")
    parser.add_argument('-c', '--config', required=True, help="Path to model YAML config")
    parser.add_argument('-t', '--checkpoint', required=True, help="Path to checkpoint .pth")
    parser.add_argument('--no-sahi', action='store_true', help="Disable SAHI (standard 640x640)")
    parser.add_argument('--no-usm', action='store_true', help="Disable USM Sharpening")
    parser.add_argument('--usm-strength', type=float, default=0.35, help="USM Sharpening strength (default: 0.35)")
    parser.add_argument('--use-clahe', action='store_true', help="Enable Infrared CLAHE")
    parser.add_argument('--clahe-clip', type=float, default=1.5, help="CLAHE clip limit (default: 1.5)")
    parser.add_argument('--clahe-blend', type=float, default=0.5, help="CLAHE blend alpha with raw IR (default: 0.5)")
    parser.add_argument('--use-depth-boost', action='store_true', help="Enable Depth-Guided Thermal Boost")
    parser.add_argument('--depth-boost-strength', type=float, default=0.35, help="Depth-guided thermal boost strength (default: 0.35)")
    parser.add_argument('--iou-thresh', type=float, default=0.65, help="NMS IoU threshold (default: 0.65)")
    parser.add_argument('--conf-thresh', type=float, default=0.05, help="Confidence threshold (default: 0.05)")
    parser.add_argument('--seam-margin', type=float, default=25.0, help="Margin to filter patch seam cutoffs (default: 25.0)")
    parser.add_argument('--max-patch-dim', type=float, default=320.0, help="Max dimension for patch detections (default: 320.0)")
    args = parser.parse_args()

    evaluate_sahi_aspect_ratio(
        cfg_path=args.config,
        checkpoint_path=args.checkpoint,
        use_sahi=not args.no_sahi,
        use_usm=not args.no_usm,
        usm_strength=args.usm_strength,
        use_clahe=args.use_clahe,
        clahe_clip=args.clahe_clip,
        clahe_blend=args.clahe_blend,
        use_depth_boost=args.use_depth_boost,
        depth_boost_strength=args.depth_boost_strength,
        iou_thresh=args.iou_thresh,
        conf_thresh=args.conf_thresh,
        seam_margin=args.seam_margin,
        max_patch_dim=args.max_patch_dim
    )
