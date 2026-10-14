#!/usr/bin/env python3
"""
Category-Aware Adaptive SAHI (CAA-SAHI) with Front-End USM and Depth-Guided Thermal Boost.
Designed for Sovereign Multimodal V9 / V10 on the AIC Dataset.

Key innovations:
1. Native 16:9 Aspect Ratio Preservation:
   Canvas (1138x640) eliminates the native 1.78x vertical distortion.
2. Front-End USM (Unsharp Masking):
   Visible high-frequency boost (alpha=0.15) sharpens object silhouettes and fine textures.
3. Depth-Guided Thermal Boost (DGTB):
   Sobel gradient of metric depth gates thermal sharpening strictly along 3D physical boundaries,
   suppressing flat water reflections and thermal background sensor noise.
4. Category-Aware Adaptive Gating (CAA-Gating):
   - Small & dense targets (sign, garbage can, uav, car, person, light):
     benefit from dense 16:9 SAHI patches.
   - Large / aspect-ratio sensitive targets (boat, ball):
     use the global unified context (P0) to avoid hull seam cuts and aspect ratio filter mismatch.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def apply_preproc(raw):
    # raw: [6, H, W]
    # 1. Visible USM
    rgb = raw[:3]
    blur = F.avg_pool2d(rgb.unsqueeze(0), 3, stride=1, padding=1)[0]
    high = rgb - blur
    rgb_usm = (rgb + 0.15 * high).clamp(0, 1)
    
    # 2. Depth-guided thermal boost
    d = raw[5:6]
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=raw.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=raw.device).view(1, 1, 3, 3)
    gx = F.conv2d(d.unsqueeze(0), sobel_x, padding=1)
    gy = F.conv2d(d.unsqueeze(0), sobel_y, padding=1)
    g = torch.sqrt(gx*gx + gy*gy)[0]
    edge_mask = torch.clamp((g - 0.015) / 0.03, 0.0, 1.0)
    
    ir = raw[3:5]
    ir_blur = F.avg_pool2d(ir.unsqueeze(0), 3, stride=1, padding=1)[0]
    ir_high = ir - ir_blur
    ir_boost = (ir + 0.25 * edge_mask * ir_high).clamp(0, 1)
    
    return torch.cat([rgb_usm, ir_boost, raw[5:6]], dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v9_snr_fdr_rtx6000.yml")
    parser.add_argument("--checkpoint", type=str, default="outputs/deim_hgnetv2_x_aic_sovereign_v9_snr_fdr_rtx6000/best_stg1.pth")
    args = parser.parse_args()

    # Load DEIM Config
    from engine.core import YAMLConfig
    cfg = YAMLConfig(args.config)
    ds = cfg.val_dataloader.dataset

    print(f"Loading checkpoint: {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location="cpu")
    weights = ck["ema"]["module"] if "ema" in ck and "module" in ck["ema"] else ck.get("model", ck)
    cfg.model.load_state_dict(weights, strict=False)
    model = cfg.model.cuda().eval()
    post = cfg.postprocessor

    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)

    all_preds = []
    seam_margin = 25.0
    max_patch_dim = 320.0

    print("Running Category-Aware Adaptive SAHI inference on validation set...")
    with torch.no_grad():
        for idx in range(len(ds)):
            raw, t = ds.load_item(idx)
            img_id = ds.ids[idx]
            raw = raw.cuda()
            raw = apply_preproc(raw)

            W_orig = float(t["orig_size"][0].item())
            H_orig = float(t["orig_size"][1].item())

            # 3D Depth Planar Water Prior for boat
            d = raw[5]
            H_raw = d.shape[0]
            bottom_d = d[int(H_raw * 0.5):, :]
            valid_d = bottom_d[bottom_d > 0.015]
            depth_std = valid_d.std().item() if len(valid_d) > 0 else 0
            is_water_scene = depth_std > 0.09

            # Global context (P0)
            p0 = F.interpolate(raw.unsqueeze(0), size=(640, 640), mode="bilinear", align_corners=False)[0]
            # Native 16:9 canvas (P1, P2)
            canvas = F.interpolate(raw.unsqueeze(0), size=(640, 1138), mode="bilinear", align_corners=False)[0]
            p1 = canvas[:, :, 0:640]
            p2 = canvas[:, :, 498:1138]

            batch = torch.stack([p1, p2, p0], dim=0)
            norm = (batch - mean) / std
            sizes = torch.tensor([[640., 640.], [640., 640.], [W_orig, H_orig]], device="cuda")

            out = model(norm)
            res = post(out, sizes, norm)

            scale_x = 1138.0 / W_orig
            scale_y = 640.0 / H_orig
            scales = torch.tensor([scale_x, scale_y, scale_x, scale_y], device="cuda")

            b1 = res[0]["boxes"] / scales
            s1 = res[0]["scores"]
            l1 = res[0]["labels"]

            offset = torch.tensor([498.0, 0.0, 498.0, 0.0], device="cuda")
            b2 = (res[1]["boxes"] + offset) / scales
            s2 = res[1]["scores"]
            l2 = res[1]["labels"]

            p1_x2_canvas = res[0]["boxes"][:, 2]
            p1_w = res[0]["boxes"][:, 2] - res[0]["boxes"][:, 0]
            p1_h = res[0]["boxes"][:, 3] - res[0]["boxes"][:, 1]
            keep1 = (p1_x2_canvas < (640.0 - seam_margin)) & (p1_w < max_patch_dim) & (p1_h < max_patch_dim)

            p2_x1_patch = res[1]["boxes"][:, 0]
            p2_w = res[1]["boxes"][:, 2] - res[1]["boxes"][:, 0]
            p2_h = res[1]["boxes"][:, 3] - res[1]["boxes"][:, 1]
            keep2 = (p2_x1_patch > seam_margin) & (p2_w < max_patch_dim) & (p2_h < max_patch_dim)

            b0 = res[2]["boxes"]
            s0 = res[2]["scores"]
            l0 = res[2]["labels"]

            # CAA Gating: Exclude boat (1) and ball (7) from patches
            keep1_cat = keep1 & (l1 != 1) & (l1 != 7)
            keep2_cat = keep2 & (l2 != 1) & (l2 != 7)

            boxes = torch.cat([b0, b1[keep1_cat], b2[keep2_cat]], dim=0)
            scores = torch.cat([s0, s1[keep1_cat], s2[keep2_cat]], dim=0)
            labels = torch.cat([l0, l1[keep1_cat], l2[keep2_cat]], dim=0)

            for c in range(12):
                # Boat: strictly enforce water scene prior to eliminate dry-land park bench false alarms
                if c == 1 and not is_water_scene:
                    continue

                mask = labels == c
                if not mask.any():
                    continue
                c_boxes = boxes[mask]
                c_scores = scores[mask]

                # Sign confidence calibration
                if c == 4:
                    c_scores = c_scores * 1.05

                # For boat and ball: use Global P0 alone with clean NMS
                if c in [1, 7]:
                    m_p0 = l0 == c
                    if not m_p0.any():
                        continue
                    c_boxes = b0[m_p0]
                    c_scores = s0[m_p0]
                    keep_idx = nms(c_boxes, c_scores, iou_threshold=0.50)
                else:
                    keep_idx = nms(c_boxes, c_scores, iou_threshold=0.65)

                k_boxes = c_boxes[keep_idx].cpu().tolist()
                k_scores = c_scores[keep_idx].cpu().tolist()

                for b, s in zip(k_boxes, k_scores):
                    all_preds.append({
                        "image_id": img_id,
                        "category_id": c,
                        "bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]],
                        "score": s
                    })

    coco_gt = ds.coco
    coco_dt = coco_gt.loadRes(all_preds)
    ce = COCOeval(coco_gt, coco_dt, "bbox")
    ce.evaluate()
    ce.accumulate()
    ce.summarize()

    print("\n=== Per-category Final AP Summary ===")
    for cat_id, cat_info in coco_gt.cats.items():
        ce_c = COCOeval(coco_gt, coco_dt, "bbox")
        ce_c.params.catIds = [cat_id]
        ce_c.evaluate()
        ce_c.accumulate()
        ce_c.summarize()
        print(f"  {cat_info['name']:<15}: AP={ce_c.stats[0]*100:.2f}%, AP50={ce_c.stats[1]*100:.2f}%")

if __name__ == "__main__":
    main()
