#!/usr/bin/env python3
"""
Official NightXX Synthetic Suite Evaluator for Corrected F5 (Tri-Modal Residual Fusion)
=======================================================================================
Evaluates F5 across the full 11-level day-to-night degradation spectrum using its
VERIFIED Tri-Modal Residual Fusion input pipeline:
  L = 0.299 * R + 0.587 * G + 0.114 * B
  R' = R + 0.15 * (IR - L)
  G' = G + 0.10 * M_depth * (Depth - L)
  B' = B
  fused = clip([R', G', B'], 0.0, 1.0)
  input = fused without ImageNet normalization (scaled in [0, 1])

Degradation Spectrum (11 levels):
  1. orig (100% daylight, 1.0)
  2. night50 (50% brightness, 0.50)
  3. night25 (25% brightness, 0.25)
  4. night12 (12% brightness, 0.12)
  5. night05 (5% brightness, 0.05)
  6. night01 (1% brightness, 0.01)
  7. night005 (0.5% brightness, 0.005)
  8. night002 (0.2% brightness, 0.002)
  9. night001 (0.1% brightness, 0.001)
  10. night0005 (0.05% brightness, 0.0005)
  11. black (0% brightness, 0.0)

Computes both:
  - RAW Baseline (Top-100, tau >= 0.01)
  - COMPACT Peak (Rare 0.45, Common 0.50, Cap 60)
"""

import os
import sys
import time
import json
import argparse
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
THR = {c: (0.45 if c in RARE_CLASSES else 0.50) for c in range(12)}

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


def find_file(folder: Path, stem: str, orig_name: str) -> Path:
    p = folder / orig_name
    if p.exists():
        return p
    for ext in [".png", ".jpg", ".jpeg"]:
        p_cand = folder / f"{stem}{ext}"
        if p_cand.exists():
            return p_cand
    return None


def run_f5_corrected_inference(
    model, postprocessor, rgb_folder: Path, ann_data: dict, ir_dir: Path, dep_dir: Path, device: str = "cuda:0"
):
    t_clean = T.Compose([
        T.Resize([1216, 2176]),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    raw_dict = {}
    compact_dict = {}

    for img_info in tqdm(ann_data["images"], desc=f"Infer F5 ({rgb_folder.name})", leave=False):
        fname = img_info["file_name"]
        stem = Path(fname).stem
        W, H = img_info["width"], img_info["height"]
        orig_size = torch.tensor([[W, H]], device=device)

        rgb_path = find_file(rgb_folder, stem, fname)
        ir_path = find_file(ir_dir, stem, fname)
        dep_path = find_file(dep_dir, stem, fname)

        if rgb_path is None or ir_path is None or dep_path is None:
            raise FileNotFoundError(f"Missing file for {stem}: rgb={rgb_path}, ir={ir_path}, dep={dep_path}")

        im_vis = Image.open(rgb_path).convert("RGB")
        im_ir = Image.open(ir_path).convert("L")
        im_dep = Image.open(dep_path)

        rgb_arr = np.array(im_vis, dtype=np.float32) / 255.0
        ir_arr = np.array(im_ir, dtype=np.float32) / 255.0
        dep_raw = np.array(im_dep, dtype=np.float32)
        dep_arr = dep_raw[..., 0] if dep_raw.ndim == 3 else dep_raw
        dep_arr = dep_arr / 65535.0 if dep_arr.max() > 255.0 else dep_arr / 255.0

        L = 0.299 * rgb_arr[..., 0] + 0.587 * rgb_arr[..., 1] + 0.114 * rgb_arr[..., 2]
        R_fused = rgb_arr[..., 0] + 0.15 * (ir_arr - L)
        M_depth = (dep_arr > 0).astype(np.float32)
        G_fused = rgb_arr[..., 1] + 0.10 * M_depth * (dep_arr - L)
        B_fused = rgb_arr[..., 2]

        fused_arr = np.clip(np.stack([R_fused, G_fused, B_fused], axis=-1), 0.0, 1.0)
        fused_im = Image.fromarray((fused_arr * 255.0).astype(np.uint8))
        inp = t_clean(fused_im).unsqueeze(0).to(device)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            preds = model(inp)
            res = postprocessor(preds, orig_size)[0]

        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res["labels"].cpu().numpy()

        raw_dict[stem] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "W": W,
            "H": H,
        }

        # Compact Peak Filtering
        cands = []
        for b, s, l in zip(boxes, scores, labels):
            cid = int(l)
            if s >= THR[cid]:
                cands.append((cid, b, s))
        rare = [c for c in cands if c[0] in RARE_CLASSES]
        common = [c for c in cands if c[0] not in RARE_CLASSES]
        rare.sort(key=lambda x: -x[2])
        common.sort(key=lambda x: -x[2])
        kept = (rare + common)[:60]
        kept.sort(key=lambda x: -x[2])

        compact_dict[stem] = {
            "boxes": np.array([x[1] for x in kept]) if kept else np.zeros((0, 4)),
            "scores": np.array([x[2] for x in kept]) if kept else np.zeros((0,)),
            "labels": np.array([x[0] for x in kept]) if kept else np.zeros((0,), dtype=int),
            "W": W,
            "H": H,
        }

    return raw_dict, compact_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate Corrected F5 on NightXX Suite")
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/constant/Downloads/Baidu/AIC-2026-54199578-我很帅对不队-复赛-贺建中/AIC-2026-54199578-模型文件/F5_all2000_1216x2176_e54_epoch0053_fp16.pth",
    )
    parser.add_argument(
        "--ann",
        type=str,
        default=str(ROOT / "work_dir/deimv2_dataset/valid/annotations/instances_val.json"),
    )
    parser.add_argument(
        "--ir-dir",
        type=str,
        default=str(ROOT / "dataset/train/infrared"),
    )
    parser.add_argument(
        "--dep-dir",
        type=str,
        default=str(ROOT / "dataset/train/depth"),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "outputs/analysis/nightxx_eval/f5_corrected_nightxx_results.json"),
    )
    args = parser.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" NightXX Evaluation: Corrected F5 (Tri-Modal Residual Fusion) ".center(80, "="))
    print("=" * 80)
    print(f"Weights: {args.weights}")
    print(f"Annotations: {args.ann}")
    print(f"Output JSON: {out_json}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[+] Loading F5 model onto {device}...")
    model, postprocessor = load_f5_model(args.weights, device=device)
    print("[+] Model loaded successfully!")

    with open(args.ann, "r", encoding="utf-8") as f:
        ann_data = json.load(f)
    evaluator = OfficialCompetitionEvaluator(args.ann)

    results = []

    print("\n" + "=" * 90)
    print(f"{'Level':<12} | {'Illum':<6} | {'RAW mAP':<9} | {'RAW AP50':<9} | {'COMPACT mAP':<12} | {'COMPACT AP50':<12} | {'Time (s)':<8}")
    print("=" * 90)

    for tag, folder, illum in NIGHT_LEVELS:
        if not folder.exists():
            print(f"{tag:<12} | {illum:<6.4f} | Folder not found: {folder}")
            continue

        t0 = time.time()
        raw_dict, compact_dict = run_f5_corrected_inference(
            model=model,
            postprocessor=postprocessor,
            rgb_folder=folder,
            ann_data=ann_data,
            ir_dir=Path(args.ir_dir),
            dep_dir=Path(args.dep_dir),
            device=device,
        )
        infer_time = time.time() - t0

        m_raw = evaluator.evaluate(raw_dict, max_boxes_per_image=100, score_thresh=0.01)
        m_compact = evaluator.evaluate(compact_dict, max_boxes_per_image=60, score_thresh=0.0)

        raw_map = m_raw["mAP_50_95"]
        raw_ap50 = m_raw["mAP_50"]
        compact_map = m_compact["mAP_50_95"]
        compact_ap50 = m_compact["mAP_50"]
        per_class_raw = m_raw["per_class"]
        per_class_compact = m_compact["per_class"]

        print(
            f"{tag:<12} | {illum:<6.4f} | {raw_map:<9.4f} | {raw_ap50:<9.4f} | {compact_map:<12.4f} | {compact_ap50:<12.4f} | {infer_time:<8.1f}"
        )

        entry = {
            "level": tag,
            "illum": illum,
            "raw": {
                "mAP_50_95": float(raw_map),
                "mAP_50": float(raw_ap50),
                "per_class": {k: float(v["ap50_95"]) for k, v in per_class_raw.items()},
                "per_class_ap50": {k: float(v["ap50"]) for k, v in per_class_raw.items()},
            },
            "compact": {
                "mAP_50_95": float(compact_map),
                "mAP_50": float(compact_ap50),
                "per_class": {k: float(v["ap50_95"]) for k, v in per_class_compact.items()},
                "per_class_ap50": {k: float(v["ap50"]) for k, v in per_class_compact.items()},
            },
            "infer_time_sec": infer_time,
        }
        results.append(entry)

        # Incrementally update output json so progress is saved
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"benchmark": "NightXX_Synthetic_Suite", "model": "F5_Corrected_TriModal", "results": results}, f, indent=2)

    print("=" * 90)
    print(f"\n[+] All 11 degradation levels evaluated successfully! Results saved to:\n    {out_json}")


if __name__ == "__main__":
    main()
