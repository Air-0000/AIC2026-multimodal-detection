#!/usr/bin/env python3
"""
eval_pure_rgb_quality_labels.py
=============================================================================
Evaluate Pure RGB models against both Old GT and New Clean Quality GT:
- instances_val.json (1,523 ground truth targets)
- instances_val_clean_quality.json (1,757 ground truth targets)

Models to test:
1. PureMAL Base: outputs/deim_hgnetv2_x_aic_obj2coco_ft_puremal/best_stg2.pth
2. PureMAL Refine: outputs/deim_hgnetv2_x_aic_puremal_refine_giou35/best_stg1.pth
3. PureMAL Full (GIoU 3.5x): outputs/deim_hgnetv2_x_aic_puremal_giou35_full/best_stg2.pth

Evaluates at:
- Raw (score >= 0.005, no filtering)
- 0.20 Confidence Filtering (score >= 0.20)
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF

AIC_ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIM_ROOT = os.path.join(AIC_ROOT, "vendor/DEIMv2")
sys.path.insert(0, os.path.join(AIC_ROOT, "scripts/lib"))
sys.path.insert(0, DEIM_ROOT)
os.chdir(DEIM_ROOT)

from engine.core import YAMLConfig
from scripts.eval.official_competition_evaluator import OfficialCompetitionEvaluator, NUM_CLASSES, ID2NAME

MODELS = [
    {
        "name": "Pure RGB (obj2coco_ft_puremal best_stg2)",
        "config": os.path.join(DEIM_ROOT, "configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_puremal.yml"),
        "ckpt": os.path.join(DEIM_ROOT, "outputs/deim_hgnetv2_x_aic_obj2coco_ft_puremal/best_stg2.pth"),
    },
    {
        "name": "Pure RGB Refine (puremal_refine_giou35 best_stg1)",
        "config": os.path.join(DEIM_ROOT, "configs/deimv2/deim_hgnetv2_x_aic_puremal_refine_giou35.yml"),
        "ckpt": os.path.join(DEIM_ROOT, "outputs/deim_hgnetv2_x_aic_puremal_refine_giou35/best_stg1.pth"),
    },
    {
        "name": "Pure RGB Full (puremal_giou35_full best_stg2)",
        "config": os.path.join(DEIM_ROOT, "configs/deimv2/deim_hgnetv2_x_aic_puremal_giou35_full.yml"),
        "ckpt": os.path.join(DEIM_ROOT, "outputs/deim_hgnetv2_x_aic_puremal_giou35_full/best_stg2.pth"),
    },
]

OLD_GT = os.path.join(AIC_ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
CLEAN_GT = os.path.join(AIC_ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val_clean_quality.json")
IMG_DIR = os.path.join(AIC_ROOT, "work_dir/deimv2_dataset/valid/images")


def run_model_inference(model_info, device):
    print(f"\n---> Loading {model_info['name']} ...")
    cfg = YAMLConfig(model_info["config"])
    ck = torch.load(model_info["ckpt"], map_location="cpu")
    sd = ck["ema"]["module"] if "ema" in ck else ck["model"]
    cfg.model.load_state_dict(sd, strict=False)
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.deploy().to(device).eval()

    with open(OLD_GT) as f:
        val_data = json.load(f)
    images = val_data["images"]

    pred_dict = {}
    t0 = time.time()
    with torch.no_grad():
        for img in tqdm(images, desc=f"Inference {model_info['name'][:25]}"):
            stem = os.path.splitext(img["file_name"])[0]
            # Try jpg or png in IMG_DIR
            img_p = os.path.join(IMG_DIR, img["file_name"])
            if not os.path.exists(img_p):
                alt_ext = ".jpg" if img["file_name"].endswith(".png") else ".png"
                img_p = os.path.join(IMG_DIR, stem + alt_ext)
            
            pil_img = Image.open(img_p).convert("RGB")
            w, h = pil_img.size
            img_tensor = TF.to_tensor(pil_img.resize((640, 640), Image.BILINEAR)).unsqueeze(0).to(device)
            orig_sizes = torch.tensor([[w, h]], device=device)

            out = model(img_tensor)
            labels, boxes_pix, scores = postprocessor(out, orig_sizes)

            scores_np = scores[0].cpu().numpy()
            labels_np = labels[0].cpu().numpy()
            boxes_pix_np = boxes_pix[0].cpu().numpy()

            # Normalize to [0, 1] xyxy
            boxes_norm = np.zeros_like(boxes_pix_np)
            boxes_norm[:, [0, 2]] = np.clip(boxes_pix_np[:, [0, 2]] / float(w), 0.0, 1.0)
            boxes_norm[:, [1, 3]] = np.clip(boxes_pix_np[:, [1, 3]] / float(h), 0.0, 1.0)

            pred_dict[stem] = {
                "boxes": boxes_norm,
                "scores": scores_np,
                "labels": labels_np,
            }

    print(f"Inference completed in {time.time() - t0:.2f}s ({len(images)} images).")
    return pred_dict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" PURE RGB BENCHMARK: OLD GT vs NEW CLEAN QUALITY GT ".center(80, "="))
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    eval_old = OfficialCompetitionEvaluator(OLD_GT)
    eval_clean = OfficialCompetitionEvaluator(CLEAN_GT)

    all_results = []

    for minfo in MODELS:
        if not os.path.exists(minfo["ckpt"]):
            print(f"Skipping {minfo['name']}: checkpoint not found at {minfo['ckpt']}")
            continue
        
        preds = run_model_inference(minfo, device)

        # 1. Old GT Raw (score >= 0.005)
        res_old_raw = eval_old.evaluate(preds, score_thresh=0.005)
        # 2. Old GT Filtered (score >= 0.20)
        res_old_f20 = eval_old.evaluate(preds, score_thresh=0.20)

        # 3. Clean GT Raw (score >= 0.005)
        res_clean_raw = eval_clean.evaluate(preds, score_thresh=0.005)
        # 4. Clean GT Filtered (score >= 0.20)
        res_clean_f20 = eval_clean.evaluate(preds, score_thresh=0.20)

        all_results.append({
            "model": minfo["name"],
            "old_raw": res_old_raw,
            "old_f20": res_old_f20,
            "clean_raw": res_clean_raw,
            "clean_f20": res_clean_f20,
        })

    # Print summary table
    print("\n" + "=" * 110)
    print(f"{'Model Name':<42} | {'Old GT (Raw)':<13} | {'Old GT (0.20)':<14} | {'Clean GT (Raw)':<15} | {'Clean GT (0.20)':<15}")
    print("-" * 110)
    for r in all_results:
        mname = r["model"]
        o_raw = r["old_raw"]["mAP_50_95"] * 100
        o_f20 = r["old_f20"]["mAP_50_95"] * 100
        c_raw = r["clean_raw"]["mAP_50_95"] * 100
        c_f20 = r["clean_f20"]["mAP_50_95"] * 100
        print(f"{mname:<42} | {o_raw:6.2f}%       | {o_f20:6.2f}%        | {c_raw:6.2f}%         | {c_f20:6.2f}%")
    print("=" * 110)

    # Detailed per-class breakdown for each model
    for r in all_results:
        print("\n" + "=" * 100)
        print(f" Detailed Class Breakdown for: {r['model']} ".center(100, "="))
        print("=" * 100)
        print(f"{'Class ID & Name':<20} | {'Old GT Raw':<12} | {'Old GT 0.20':<12} | {'Clean GT Raw':<14} | {'Clean GT 0.20':<14} | {'Clean AP50':<10}")
        print("-" * 100)
        for cid in range(NUM_CLASSES):
            cname = ID2NAME[cid]
            c_or = r["old_raw"]["per_class"][cname]["ap50_95"] * 100
            c_of = r["old_f20"]["per_class"][cname]["ap50_95"] * 100
            c_cr = r["clean_raw"]["per_class"][cname]["ap50_95"] * 100
            c_cf = r["clean_f20"]["per_class"][cname]["ap50_95"] * 100
            c_cp50 = r["clean_raw"]["per_class"][cname]["ap50"] * 100
            print(f"[{cid:>2d}] {cname:<15s} | {c_or:6.2f}%      | {c_of:6.2f}%      | {c_cr:6.2f}%         | {c_cf:6.2f}%         | {c_cp50:6.2f}%")
        print("-" * 100)
        print(f"{'Overall mAP@50:95':<20} | {r['old_raw']['mAP_50_95']*100:6.2f}%      | {r['old_f20']['mAP_50_95']*100:6.2f}%      | {r['clean_raw']['mAP_50_95']*100:6.2f}%         | {r['clean_f20']['mAP_50_95']*100:6.2f}%")
        print(f"{'Overall mAP@50':<20} | {r['old_raw']['mAP_50']*100:6.2f}%      | {r['old_f20']['mAP_50']*100:6.2f}%      | {r['clean_raw']['mAP_50']*100:6.2f}%         | {r['clean_f20']['mAP_50']*100:6.2f}%")
        print(f"{'Overall mAP@75':<20} | {r['old_raw']['mAP_75']*100:6.2f}%      | {r['old_f20']['mAP_75']*100:6.2f}%      | {r['clean_raw']['mAP_75']*100:6.2f}%         | {r['clean_f20']['mAP_75']*100:6.2f}%")
        print("=" * 100)


if __name__ == "__main__":
    main()
