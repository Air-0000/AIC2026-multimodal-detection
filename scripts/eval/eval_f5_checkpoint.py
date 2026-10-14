#!/usr/bin/env python3
"""
Comprehensive Local Evaluation for F5_all2000_1216x2176_e54_epoch0053_fp16.pth
=============================================================================
Evaluates the local checkpoint directly on:
  1. Official Clean Validation Set (instances_val.json, 200 images, 1,489 GTs)
  2. Clean Quality Validation Set (instances_val_clean_quality.json, 1,757 GTs, if exists)
Computes:
  - Raw Baseline (tau >= 0.01, Top-100)
  - Official Peak Compact (tau >= 0.45 rare / 0.50 common, cap 60)
  - Full 12-class AP breakdown (mAP@50:95, AP50, AP75, AP_s, AP_m, AP_l)
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

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


def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_f5_model(weights_path: str, device: str = "cuda:0"):
    cfg_path = ROOT / "vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft.yml"
    cfg = YAMLConfig(str(cfg_path))
    cfg.yaml_cfg["eval_spatial_size"] = [1216, 2176]
    model = cfg.model.to(device).eval()
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    postprocessor = cfg.postprocessor.to(device).eval()
    return model, postprocessor, ckpt.get("export_meta", {})


def run_inference(model, postprocessor, img_folder: Path, ann_data: dict, device: str = "cuda:0"):
    transform = T.Compose([
        T.Resize([1216, 2176]),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    raw_dict = {}
    for img_info in tqdm(ann_data["images"], desc="Inferring Local F5"):
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

    return raw_dict


def filter_compact(raw_dict, thr_dict, max_per_image=60):
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


def evaluate_dataset(ann_file, raw_dict, label_tag):
    if not os.path.exists(ann_file):
        print(f"[-] Annotation file not found: {ann_file}")
        return

    evaluator = OfficialCompetitionEvaluator(str(ann_file))
    print("\n" + "=" * 80)
    print(f" Evaluation Benchmark on: {label_tag} ".center(80, "="))
    print("=" * 80)

    # 1. Raw Top-100 (tau=0.01)
    raw_eval_dict = {}
    for stem, d in raw_dict.items():
        raw_eval_dict[stem] = {
            "boxes": d["boxes"],
            "scores": d["scores"],
            "labels": d["labels"],
        }
    m_raw = evaluator.evaluate(raw_eval_dict, max_boxes_per_image=100, score_thresh=0.01)

    # 2. Official Peak Compact (0.45 rare / 0.50 common, cap 60)
    thr_45_50 = {cid: (0.45 if cid in RARE_CLASSES else 0.50) for cid in range(12)}
    compact_dict, num_compact_boxes = filter_compact(raw_dict, thr_45_50, max_per_image=60)
    m_compact = evaluator.evaluate(compact_dict, max_boxes_per_image=60, score_thresh=0.0)

    # 3. Soft Compact (tau=0.30)
    compact_30, num_30 = filter_compact(raw_dict, 0.30, max_per_image=60)
    m_30 = evaluator.evaluate(compact_30, max_boxes_per_image=60, score_thresh=0.0)

    print(f"{'Post-Processing Mode':<35} | {'Boxes':<7} | {'mAP@50:95':<10} | {'mAP50':<8} | {'mAP75':<8}")
    print("-" * 80)
    print(f"{'Raw Top-100 (tau=0.01, no budget)':<35} | {sum(len(d['scores']) for d in raw_eval_dict.values()):<7} | {m_raw['mAP_50_95']:<10.4f} | {m_raw['mAP_50']:<8.4f} | {m_raw['mAP_75']:<8.4f}")
    print(f"{'Soft Compact (tau=0.30, cap 60)':<35} | {num_30:<7} | {m_30['mAP_50_95']:<10.4f} | {m_30['mAP_50']:<8.4f} | {m_30['mAP_75']:<8.4f}")
    print(f"{'Official Peak Compact (.45/.50 cap 60)':<35} | {num_compact_boxes:<7} | {m_compact['mAP_50_95']:<10.4f} | {m_compact['mAP_50']:<8.4f} | {m_compact['mAP_75']:<8.4f}")

    print("\n" + "-" * 80)
    print(" 12-Class Detailed Breakdown (Raw Top-100 vs Official Compact) ".center(80, "-"))
    print("-" * 80)
    print(f"{'Category':<15} | {'GT':<5} | {'Raw AP':<10} | {'Raw AP50':<10} | {'Compact AP':<10} | {'Compact AP50'}")
    print("-" * 80)
    for cat_name, r_stats in m_raw["per_class"].items():
        c_stats = m_compact["per_class"].get(cat_name, {"ap50_95": 0.0, "ap50": 0.0})
        gt_cnt = r_stats["gt_count"]
        print(f"{cat_name:<15} | {gt_cnt:<5} | {r_stats['ap50_95']:<10.4f} | {r_stats['ap50']:<10.4f} | {c_stats['ap50_95']:<10.4f} | {c_stats['ap50']:<10.4f}")


def main():
    target_weight = ROOT / "weights/F5.pth"  # 根目录副本已于 2026-10-07 去重（md5 与 weights/F5.pth 一致）
    val_ann_official = ROOT / "work_dir/deimv2_dataset/valid/annotations/instances_val.json"
    val_ann_quality = ROOT / "work_dir/deimv2_dataset/valid/annotations/instances_val_clean_quality.json"
    img_folder = ROOT / "work_dir/deimv2_dataset/valid/images"

    sha = get_sha256(target_weight)
    print("=" * 80)
    print(" Local F5 Epoch 53 Checkpoint Audit ".center(80, "="))
    print("=" * 80)
    print(f"Target Weight: {target_weight.name}")
    print(f"SHA-256:       {sha}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device:        {device}")

    model, postprocessor, meta = load_f5_model(str(target_weight), device=device)
    print("\nCheckpoint Metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    with open(val_ann_official, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    t0 = time.time()
    raw_dict = run_inference(model, postprocessor, img_folder, ann_data, device=device)
    dur = time.time() - t0
    print(f"\n[+] Inference finished on {len(ann_data['images'])} images in {dur:.2f}s ({len(ann_data['images'])/dur:.1f} FPS)")

    # Benchmark 1: Official Clean Validation Set
    evaluate_dataset(val_ann_official, raw_dict, "Official 200 Val Set (instances_val.json)")

    # Benchmark 2: Quality Clean Validation Set (if available)
    if val_ann_quality.exists():
        evaluate_dataset(val_ann_quality, raw_dict, "Quality Clean 200 Val Set (instances_val_clean_quality.json)")


if __name__ == "__main__":
    main()
