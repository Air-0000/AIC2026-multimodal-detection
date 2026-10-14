#!/usr/bin/env python3
"""
Official NightXX Synthetic Degradation Suite Evaluator
======================================================
Evaluates detection models (F5, Sovereign V23, etc.) across the full
9-level day-to-night degradation spectrum:
  - orig (100% daylight, 1.0)
  - night50 (50% brightness, 0.50)
  - night25 (25% brightness, 0.25)
  - night12 (12.5% brightness, 0.12)
  - night05 (5% brightness, 0.05)
  - night01 (1% brightness, 0.01)
  - night005 (0.5% brightness, 0.005)
  - night002 (0.2% brightness, 0.002)
  - night001 (0.1% brightness, 0.001)
  - night0005 (0.05% brightness, 0.0005)
  - black (0% brightness, 0.0)

Calculates the exact official AIC2026 competition metric (mAP@50:95, mAP50, per-class AP)
against the 200 validation images (instances_val.json).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from tqdm import tqdm

ROOT = Path("/home/constant/CodeBuddy/Dev/AIC")
sys.path.insert(0, str(ROOT / "vendor" / "DEIMv2"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from engine.core import YAMLConfig
from official_competition_evaluator import OfficialCompetitionEvaluator, AIC_CATEGORIES

NIGHT_LEVELS = [
    ("orig", ROOT / "work_dir/deimv2_dataset/valid/images", 1.0),
    ("night50", ROOT / "data_aux/aic_valid_rgb_night50", 0.50),
    ("night25", ROOT / "data_aux/aic_valid_rgb_night25", 0.25),
    ("night12", ROOT / "data_aux/aic_valid_rgb_night12", 0.12),
    ("night05", ROOT / "data_aux/aic_valid_rgb_night05", 0.05),
    ("night01", ROOT / "data_aux/aic_valid_rgb_night01", 0.01),
    ("night005", ROOT / "data_aux/aic_valid_rgb_night005", 0.005),
    ("night002", ROOT / "data_aux/aic_valid_rgb_night002", 0.002),
    ("night001", ROOT / "data_aux/aic_valid_rgb_night001", 0.001),
    ("night0005", ROOT / "data_aux/aic_valid_rgb_night0005", 0.0005),
    ("black", ROOT / "data_aux/aic_valid_rgb_black", 0.000),
]


def load_f5_model(weights_path: str, device: str = "cuda:0"):
    cfg_path = ROOT / "vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft.yml"
    cfg = YAMLConfig(str(cfg_path))
    cfg.yaml_cfg["eval_spatial_size"] = [1216, 2176]
    model = cfg.model.to(device).eval()
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    postprocessor = cfg.postprocessor.to(device).eval()
    return model, postprocessor


def run_f5_inference_on_folder(model, postprocessor, img_folder: Path, ann_data: dict, device: str = "cuda:0"):
    transform = T.Compose([
        T.Resize([1216, 2176]),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pred_dict = {}
    for img_info in tqdm(ann_data["images"], desc=f"Infer {img_folder.name}", leave=False):
        fname = img_info["file_name"]
        stem = Path(fname).stem
        p = img_folder / fname
        if not p.exists():
            # Try alternate extension (.png / .jpg)
            for alt_ext in [".jpg", ".png", ".jpeg"]:
                alt_p = img_folder / f"{stem}{alt_ext}"
                if alt_p.exists():
                    p = alt_p
                    break

        if not p.exists():
            continue

        pil_im = Image.open(p).convert("RGB")
        W, H = pil_im.size
        t = transform(pil_im).unsqueeze(0).to(device)
        orig_size = torch.tensor([[W, H]], device=device)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            preds = model(t)
            res = postprocessor(preds, orig_size)[0]

        pred_dict[stem] = {
            "boxes": res["boxes"].cpu().numpy(),
            "scores": res["scores"].cpu().numpy(),
            "labels": res["labels"].cpu().numpy(),
        }

    return pred_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate on NightXX Synthetic Suite")
    parser.add_argument("--model", type=str, default="f5", choices=["f5", "v23"])
    parser.add_argument("--weights", type=str, default=str(ROOT / "weights/F5.pth"))
    parser.add_argument("--ann", type=str, default=str(ROOT / "work_dir/deimv2_dataset/valid/annotations/instances_val.json"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs/analysis/nightxx_eval"))
    parser.add_argument("--levels", type=str, default="all", help="Comma-separated level names or 'all'")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" NightXX Synthetic Suite Evaluation ".center(70, "="))
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Weights: {args.weights}")
    print(f"Ground Truth: {args.ann}")
    print(f"Output Dir: {out_dir}")

    evaluator = OfficialCompetitionEvaluator(args.ann)
    with open(args.ann, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model.upper()} onto {device}...")
    model, postprocessor = load_f5_model(args.weights, device=device)
    print("[+] Model loaded successfully!")

    target_levels = NIGHT_LEVELS
    if args.levels != "all":
        lvl_names = set(args.levels.split(","))
        target_levels = [x for x in NIGHT_LEVELS if x[0] in lvl_names]

    results = []

    print("\nStarting evaluation across degradation levels:")
    print("-" * 70)
    print(f"{'Level':<12} | {'Illum':<6} | {'mAP@50:95':<10} | {'mAP@50':<8} | {'Total GT':<8} | {'Status'}")
    print("-" * 70)

    for tag, folder, illum in target_levels:
        if not folder.exists():
            print(f"{tag:<12} | {illum:<6.4f} | {'N/A':<10} | {'N/A':<8} | {'-':<8} | Directory not found")
            continue

        t0 = time.time()
        pred_dict = run_f5_inference_on_folder(model, postprocessor, folder, ann_data, device=device)
        eval_metrics = evaluator.evaluate(pred_dict, max_boxes_per_image=100, score_thresh=0.01)
        elapsed = time.time() - t0

        map_val = eval_metrics["mAP_50_95"]
        map50_val = eval_metrics["mAP_50"]

        results.append({
            "level": tag,
            "illum": illum,
            "mAP": map_val,
            "mAP50": map50_val,
            "mAP75": eval_metrics["mAP_75"],
            "per_class": eval_metrics["per_class"],
            "elapsed_sec": elapsed,
        })

        print(f"{tag:<12} | {illum:<6.4f} | {map_val:<10.4f} | {map50_val:<8.4f} | {evaluator.total_gt:<8} | Done in {elapsed:.1f}s")

    # Save complete JSON
    out_json = out_dir / f"nightxx_{args.model}_curve.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(" NightXX Evaluation Completed ".center(70, "="))
    print("=" * 70)
    print(f"Full results saved to: {out_json}")

    # Generate Markdown Summary Table
    md_report = out_dir / f"nightxx_{args.model}_report.md"
    with open(md_report, "w", encoding="utf-8") as f:
        f.write(f"# NightXX 真实夜间退化谱系评测报告: {args.model.upper()}\n\n")
        f.write(f"> **评估权重**: `{args.weights}`  \n")
        f.write(f"> **评测集**: 官方 200 张洁净验证集 (`instances_val.json`, 1,489 个真实目标)  \n\n")
        f.write("| 退化档位 (Level) | 照度因子 (Illumination) | 官方 mAP@50:95 | 官方 mAP@50 | 相对日间保持率 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        orig_map = results[0]["mAP"] if results else 1.0
        for r in results:
            retention = (r["mAP"] / orig_map * 100) if orig_map > 0 else 0
            f.write(f"| **{r['level']}** | {r['illum']:.4f} | **{r['mAP']:.4f}** | {r['mAP50']:.4f} | {retention:.1f}% |\n")

    print(f"Markdown report generated at: {md_report}")


if __name__ == "__main__":
    main()
