#!/usr/bin/env python3
"""
Evaluate F5 with Compact Post-Processing on Validation Set
==========================================================
Tests the exact impact of F5's compact post-processing on the 200 validation images:
  1. Raw Top-100 (tau >= 0.01): Baseline 0.3720
  2. Official Peak Compact (Rare 0.45, Common 0.50, Cap 60)
  3. Tier-2 Compact (Rare 0.50, Common 0.55, Cap 60)
  4. Tier-3 Compact (Rare 0.35, Common 0.40, Cap 60)
  5. Threshold sweeps (0.10 -> 0.60)
"""

import os
import sys
import json
import math
import time
from pathlib import Path
from collections import defaultdict, Counter

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

RARE_CLASSES = {1, 7, 9, 10, 11} # boat, ball, garbage can, uav, tricycle


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


def get_raw_predictions(model, postprocessor, img_folder: Path, ann_data: dict, device: str = "cuda:0"):
    transform = T.Compose([
        T.Resize([1216, 2176]),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cache_path = ROOT / "outputs" / "f5_val_raw_preds_cache.pt"
    if cache_path.exists():
        print(f"[+] Loading cached raw predictions from {cache_path}...")
        return torch.load(cache_path)

    raw_dict = {}
    for img_info in tqdm(ann_data["images"], desc="Inferring F5 on Val"):
        fname = img_info["file_name"]
        stem = Path(fname).stem
        p = img_folder / fname
        if not p.exists():
            for alt_ext in [".jpg", ".png", ".jpeg"]:
                alt_p = img_folder / f"{stem}{alt_ext}"
                if alt_p.exists():
                    p = alt_p
                    break

        pil_im = Image.open(p).convert("RGB")
        W, H = pil_im.size
        t = transform(pil_im).unsqueeze(0).to(device)
        orig_size = torch.tensor([[W, H]], device=device)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            preds = model(t)
            res = postprocessor(preds, orig_size)[0]

        raw_dict[stem] = {
            "boxes": res["boxes"].cpu().numpy(),
            "scores": res["scores"].cpu().numpy(),
            "labels": res["labels"].cpu().numpy(),
            "W": W,
            "H": H,
        }

    os.makedirs(cache_path.parent, exist_ok=True)
    torch.save(raw_dict, cache_path)
    print(f"[+] Cached raw predictions saved to {cache_path}")
    return raw_dict


def apply_compact(raw_dict, thr_dict, max_per_image=60):
    pred_dict = {}
    total_boxes = 0

    for stem, data in raw_dict.items():
        boxes = data["boxes"]
        scores = data["scores"]
        labels = data["labels"]

        cands = []
        for i, (b, s, l) in enumerate(zip(boxes, scores, labels)):
            cid = int(l)
            req_thr = thr_dict[cid] if isinstance(thr_dict, dict) else thr_dict
            if s >= req_thr:
                cands.append((cid, b, s, i))

        rare = [c for c in cands if c[0] in RARE_CLASSES]
        common = [c for c in cands if c[0] not in RARE_CLASSES]

        # Sort by score descending
        rare.sort(key=lambda x: -x[2])
        common.sort(key=lambda x: -x[2])

        selected = (rare + common)[:max_per_image]
        selected.sort(key=lambda x: -x[2])

        if selected:
            sel_boxes = np.stack([x[1] for x in selected])
            sel_scores = np.array([x[2] for x in selected], dtype=np.float32)
            sel_labels = np.array([x[0] for x in selected], dtype=np.int64)
        else:
            sel_boxes = np.zeros((0, 4), dtype=np.float32)
            sel_scores = np.zeros((0,), dtype=np.float32)
            sel_labels = np.zeros((0,), dtype=np.int64)

        pred_dict[stem] = {
            "boxes": sel_boxes,
            "scores": sel_scores,
            "labels": sel_labels,
        }
        total_boxes += len(selected)

    return pred_dict, total_boxes


def main():
    ann_path = ROOT / "work_dir/deimv2_dataset/valid/annotations/instances_val.json"
    img_folder = ROOT / "work_dir/deimv2_dataset/valid/images"
    weights_path = ROOT / "weights/F5.pth"

    with open(ann_path, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading F5 model onto {device}...")
    model, postprocessor = load_f5_model(str(weights_path), device=device)
    raw_dict = get_raw_predictions(model, postprocessor, img_folder, ann_data, device=device)

    evaluator = OfficialCompetitionEvaluator(str(ann_path))

    experiments = [
        ("Raw Top-100 (tau=0.01, no cap)", 0.01, 100),
        ("Uniform tau=0.10 (cap=60)", 0.10, 60),
        ("Uniform tau=0.20 (cap=60)", 0.20, 60),
        ("Uniform tau=0.30 (cap=60)", 0.30, 60),
        ("Uniform tau=0.35 (cap=60)", 0.35, 60),
        ("Uniform tau=0.40 (cap=60)", 0.40, 60),
        ("Uniform tau=0.45 (cap=60)", 0.45, 60),
        ("Uniform tau=0.50 (cap=60)", 0.50, 60),
        ("Uniform tau=0.55 (cap=60)", 0.55, 60),
        ("Tier-3 Compact (.35 rare / .40 common, cap 60)", {c: (0.35 if c in RARE_CLASSES else 0.40) for c in range(12)}, 60),
        ("Official Peak Compact (.45 rare / .50 common, cap 60)", {c: (0.45 if c in RARE_CLASSES else 0.50) for c in range(12)}, 60),
        ("Tier-2 Compact (.50 rare / .55 common, cap 60)", {c: (0.50 if c in RARE_CLASSES else 0.55) for c in range(12)}, 60),
    ]

    print("\n" + "=" * 80)
    print(" F5 Compact Post-Processing Ablation on 200 Val Images ".center(80, "="))
    print("=" * 80)
    print(f"{'Experiment':<46} | {'Boxes':<7} | {'Avg/Img':<7} | {'mAP@50:95':<9} | {'mAP50':<7}")
    print("-" * 80)

    for name, thrs, cap in experiments:
        preds, num_b = apply_compact(raw_dict, thrs, max_per_image=cap)
        metrics = evaluator.evaluate(preds, max_boxes_per_image=cap, score_thresh=0.0)
        map_val = metrics["mAP_50_95"]
        map50_val = metrics["mAP_50"]
        print(f"{name:<46} | {num_b:<7} | {num_b/200:<7.1f} | {map_val:<9.4f} | {map50_val:<7.4f}")


if __name__ == "__main__":
    main()
