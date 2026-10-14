#!/usr/bin/env python3
"""
Systematic Factor-by-Factor Ablation of All Post-Processing & Preprocessing Components
======================================================================================
Evaluates Sovereign V10 on the official Validation set (1000 images).

Factors evaluated:
1. Pure Baseline: Clean Single-Scale 640x640 (NO USM, NO DGTB, NO Water Prior, NO Sign Boost, NO Patch Slicing)
2. Front-End Physical Enhancement: + USM & DGTB
3. Classification Logit Calibration: + Sign Confidence Boost (*1.05)
4. 3D Scene Geometry Prior: + Water Depth Variance Gate for Boat (std > 0.09)
5. Multi-Scale Spatial Fusion: + CAA-SAHI 3-View Slicing (P1, P2, P0 on 16:9 canvas)
6. All Combined: Full CAA-SAHI Engine
"""

import os
import sys
import copy
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def apply_preproc(raw):
    """Front-end USM (alpha=0.15) + Depth-Guided Thermal Boost (DGTB)"""
    rgb = raw[:3]
    blur = F.avg_pool2d(rgb.unsqueeze(0), 3, stride=1, padding=1)[0]
    high = rgb - blur
    rgb_usm = (rgb + 0.15 * high).clamp(0, 1)

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

def evaluate_coco(coco_gt, all_preds, silent=False):
    if len(all_preds) == 0:
        return 0.0, 0.0, {}
    coco_dt = coco_gt.loadRes(all_preds)
    ce = COCOeval(coco_gt, coco_dt, "bbox")
    ce.evaluate()
    ce.accumulate()
    ce.summarize()
    mAP = ce.stats[0] * 100
    mAP50 = ce.stats[1] * 100

    per_cat = {}
    for cat_id, cat_info in coco_gt.cats.items():
        ce_c = COCOeval(coco_gt, coco_dt, "bbox")
        ce_c.params.catIds = [cat_id]
        ce_c.evaluate()
        ce_c.accumulate()
        ce_c.summarize()
        per_cat[cat_info["name"]] = (ce_c.stats[0]*100, ce_c.stats[1]*100)
    return mAP, mAP50, per_cat

def main():
    config_path = "configs/deimv2/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000.yml"
    checkpoint_path = "outputs/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000/best_stg1.pth"

    from engine.core import YAMLConfig
    cfg = YAMLConfig(config_path)
    ds = cfg.val_dataloader.dataset

    print(f"Loading checkpoint: {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location="cpu")
    weights = ck["ema"]["module"] if "ema" in ck and "module" in ck["ema"] else ck.get("model", ck)
    cfg.model.load_state_dict(weights, strict=False)
    model = cfg.model.cuda().eval()
    post = cfg.postprocessor

    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)

    print("Running forward passes across validation dataset (1000 images)...")
    cache_clean_p0 = []
    cache_preproc_all = []

    with torch.no_grad():
        for idx in range(len(ds)):
            raw, target = ds.load_item(idx)
            img_id = ds.ids[idx]
            W_orig = float(target["orig_size"][0].item())
            H_orig = float(target["orig_size"][1].item())
            raw = raw.cuda()

            # 1. Clean P0 Forward (No USM, No DGTB)
            p0_clean = F.interpolate(raw.unsqueeze(0), size=(640, 640), mode="bilinear", align_corners=False)
            norm_clean = (p0_clean - mean) / std
            sizes_clean = torch.tensor([[W_orig, H_orig]], device="cuda")
            out_clean = model(norm_clean)
            res_clean = post(out_clean, sizes_clean, norm_clean)[0]

            cache_clean_p0.append({
                "img_id": img_id,
                "W_orig": W_orig,
                "H_orig": H_orig,
                "boxes": res_clean["boxes"].clone(),
                "scores": res_clean["scores"].clone(),
                "labels": res_clean["labels"].clone(),
            })

            # 2. Preprocessed Forward: P0, P1, P2 with USM + DGTB
            raw_p = apply_preproc(raw)

            # Water scene prior flag
            d_chan = raw_p[5]
            bottom_d = d_chan[int(d_chan.shape[0] * 0.5):, :]
            valid_d = bottom_d[bottom_d > 0.015]
            is_water_scene = valid_d.std().item() > 0.09 if len(valid_d) > 0 else False

            p0 = F.interpolate(raw_p.unsqueeze(0), size=(640, 640), mode="bilinear", align_corners=False)[0]
            canvas = F.interpolate(raw_p.unsqueeze(0), size=(640, 1138), mode="bilinear", align_corners=False)[0]
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

            b1 = res[0]["boxes"] / scales;  s1 = res[0]["scores"];  l1 = res[0]["labels"]
            b2 = (res[1]["boxes"] + torch.tensor([498., 0., 498., 0.], device="cuda")) / scales
            s2 = res[1]["scores"];          l2 = res[1]["labels"]
            b0 = res[2]["boxes"];           s0 = res[2]["scores"];  l0 = res[2]["labels"]

            cache_preproc_all.append({
                "img_id": img_id,
                "W_orig": W_orig,
                "H_orig": H_orig,
                "is_water_scene": is_water_scene,
                "b0": b0, "s0": s0, "l0": l0,
                "b1": b1, "s1": s1, "l1": l1,
                "b2": b2, "s2": s2, "l2": l2,
                "res0_boxes": res[0]["boxes"],
                "res1_boxes": res[1]["boxes"]
            })

            if (idx + 1) % 200 == 0:
                print(f"  [{idx+1}/1000] forward cached")

    print("\nForward passes completed. Sweeping all 6 component configurations...\n")
    coco_gt = ds.coco

    configs_to_test = [
        ("1. Pure Baseline (Single-Scale 640, No USM, No DGTB, No Rules)", {"use_preproc": False, "use_sahi": False, "water_prior": False, "sign_boost": False}),
        ("2. + Front-End USM & DGTB Only",                                  {"use_preproc": True,  "use_sahi": False, "water_prior": False, "sign_boost": False}),
        ("3. + Sign Confidence Boost (*1.05) Only",                         {"use_preproc": True,  "use_sahi": False, "water_prior": False, "sign_boost": True}),
        ("4. + 3D Water Scene Depth Prior (Boat)",                          {"use_preproc": True,  "use_sahi": False, "water_prior": True,  "sign_boost": True}),
        ("5. + CAA-SAHI 3-View Patch Slicing (No Water Prior)",             {"use_preproc": True,  "use_sahi": True,  "water_prior": False, "sign_boost": True}),
        ("6. Full CAA-SAHI Engine (All Factors Combined)",                  {"use_preproc": True,  "use_sahi": True,  "water_prior": True,  "sign_boost": True}),
    ]

    results_table = []
    seam_margin = 25.0
    max_patch_dim = 320.0
    P0_ONLY_CATS = {1, 7, 11}

    for name, opt in configs_to_test:
        all_preds = []
        for idx in range(len(ds)):
            if not opt["use_preproc"]:
                # Use clean cache
                item = cache_clean_p0[idx]
                img_id = item["img_id"]
                boxes = item["boxes"]
                scores = item["scores"]
                labels = item["labels"]
                is_water_scene = True  # disabled

                for c in range(12):
                    mask = labels == c
                    if not mask.any(): continue
                    c_boxes = boxes[mask]
                    c_scores = scores[mask]
                    keep_idx = nms(c_boxes, c_scores, iou_threshold=0.65)
                    for b, s in zip(c_boxes[keep_idx].cpu().tolist(), c_scores[keep_idx].cpu().tolist()):
                        if s >= 0.05:
                            all_preds.append({"image_id": img_id, "category_id": c, "bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]], "score": s})

            else:
                item = cache_preproc_all[idx]
                img_id = item["img_id"]
                is_water_scene = item["is_water_scene"] if opt["water_prior"] else True

                b0, s0, l0 = item["b0"], item["s0"], item["l0"]
                b1, s1, l1 = item["b1"], item["s1"], item["l1"]
                b2, s2, l2 = item["b2"], item["s2"], item["l2"]

                if opt["use_sahi"]:
                    # Patch filtering
                    p1_x2 = item["res0_boxes"][:, 2]
                    p1_w = item["res0_boxes"][:, 2] - item["res0_boxes"][:, 0]
                    p1_h = item["res0_boxes"][:, 3] - item["res0_boxes"][:, 1]
                    keep1 = (p1_x2 < (640. - seam_margin)) & (p1_w < max_patch_dim) & (p1_h < max_patch_dim)

                    p2_x1 = item["res1_boxes"][:, 0]
                    p2_w = item["res1_boxes"][:, 2] - item["res1_boxes"][:, 0]
                    p2_h = item["res1_boxes"][:, 3] - item["res1_boxes"][:, 1]
                    keep2 = (p2_x1 > seam_margin) & (p2_w < max_patch_dim) & (p2_h < max_patch_dim)

                    p0m1 = torch.zeros_like(keep1, dtype=torch.bool)
                    p0m2 = torch.zeros_like(keep2, dtype=torch.bool)
                    for c in P0_ONLY_CATS: p0m1 |= (l1 == c); p0m2 |= (l2 == c)
                    k1 = keep1 & ~p0m1
                    k2 = keep2 & ~p0m2

                    boxes = torch.cat([b0, b1[k1], b2[k2]], dim=0)
                    scores = torch.cat([s0, s1[k1], s2[k2]], dim=0)
                    labels = torch.cat([l0, l1[k1], l2[k2]], dim=0)
                else:
                    boxes, scores, labels = b0, s0, l0

                for c in range(12):
                    if c == 1 and not is_water_scene:
                        continue

                    if opt["use_sahi"] and c in P0_ONLY_CATS:
                        m_p0 = l0 == c
                        if not m_p0.any(): continue
                        c_boxes = b0[m_p0]
                        c_scores = s0[m_p0]
                        iou_thr = 0.50
                    else:
                        mask = labels == c
                        if not mask.any(): continue
                        c_boxes = boxes[mask]
                        c_scores = scores[mask]
                        iou_thr = 0.65

                    if opt["sign_boost"] and c == 4:
                        c_scores = c_scores * 1.05

                    keep_idx = nms(c_boxes, c_scores.clamp(max=1.0), iou_threshold=iou_thr)
                    for b, s in zip(c_boxes[keep_idx].cpu().tolist(), c_scores[keep_idx].cpu().tolist()):
                        if s >= 0.05:
                            all_preds.append({"image_id": img_id, "category_id": c, "bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]], "score": s})

        mAP, mAP50, per_cat = evaluate_coco(coco_gt, all_preds)
        results_table.append((name, mAP, mAP50, per_cat))
        print(f"[{name}] -> mAP: {mAP:.2f}%, AP50: {mAP50:.2f}% | boat: {per_cat.get('boat',(0,))[0]:.2f}%, sign: {per_cat.get('sign',(0,))[0]:.2f}%")

    print("\n" + "="*80)
    print("              FULL FACTOR-BY-FACTOR ABLATION MATRIX SUMMARY")
    print("="*80)
    print(f"{'Configuration':<48} | {'mAP':<7} | {'AP50':<7} | {'boat':<7} | {'sign':<7}")
    print("-" * 80)
    for name, map_val, map50_val, cats in results_table:
        b_ap = cats.get('boat', (0,))[0]
        s_ap = cats.get('sign', (0,))[0]
        print(f"{name:<48} | {map_val:5.2f}% | {map50_val:5.2f}% | {b_ap:5.2f}% | {s_ap:5.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()
