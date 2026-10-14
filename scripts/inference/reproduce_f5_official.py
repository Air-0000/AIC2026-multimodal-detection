#!/usr/bin/env python3
"""
Reproduce F5 Official Benchmark Submissions
===========================================
Runs pure FP32 inference with F5.pth on Phase 2 test set,
caches raw predictions to raw_predictions.pt,
and packages the exact benchmark submissions:
  1. submission_F5_cap60thr45_50.zip (58.105 Official Peak, 5,179 boxes)
  2. submission_F5_cap60thr50_55.zip (58.055 Benchmark)
  3. submission_F5_cap60thr35_40.zip (57.707 Benchmark)
  4. submission_F5_raw_top100.zip    (53.100 Raw Benchmark, ~60-70k boxes)
"""

import os
import sys
import math
import time
import json
import zipfile
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path("/home/ailab/ssd/bigdata/hejianzhong")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from src.core import YAMLConfig
from predict_official import aligned_inventory, make_image_index, write_json, CLASS_NAMES
from export_inference_checkpoint import select_model_state

RARE_CLASSES = {1, 7, 9, 10, 11} # boat, ball, garbage can, uav, tricycle
MAX_PER_IMAGE = 60

def package_and_verify(txt_dir, out_zip_path):
    txt_files = sorted(txt_dir.glob("*.txt"))
    assert len(txt_files) == 1000, f"Expected 1000 files, found {len(txt_files)}"
    
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_f in txt_files:
            zf.write(txt_f, arcname=txt_f.name)
            
    # Verify zip
    total_boxes = 0
    with zipfile.ZipFile(out_zip_path, "r") as zf:
        members = zf.namelist()
        assert len(members) == 1000, f"ZIP member count mismatch: {len(members)}"
        for name in members:
            data = zf.read(name).decode("utf-8")
            if not data:
                continue
            lines = data.strip().split("\r\n")
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 6, f"Invalid column count in {name}: {line}"
                cid = int(parts[0])
                cx, cy, bw, bh, sc = map(float, parts[1:])
                assert 0 <= cid <= 11
                assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                assert 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0
                assert 0.0 <= sc <= 1.0
                total_boxes += 1
                
    size_kb = os.path.getsize(out_zip_path) / 1024
    print(f"  -> Verified {out_zip_path.name}: {total_boxes:,} boxes ({total_boxes/1000:.2f}/img), {size_kb:.1f} KB")
    return total_boxes

def build_compact_submission(raw_predictions, thresholds, max_per_image, out_zip_path):
    txt_dir = out_zip_path.parent / (out_zip_path.stem + "_txt")
    os.makedirs(txt_dir, exist_ok=True)
    class_counts = Counter()

    for stem in sorted(raw_predictions.keys()):
        cands = raw_predictions[stem]
        eligible = [c for c in cands if c[5] >= thresholds[c[0]]]
        rare = sorted([c for c in eligible if c[0] in RARE_CLASSES], key=lambda c: (-c[5], c[6]))
        common = sorted([c for c in eligible if c[0] not in RARE_CLASSES], key=lambda c: (-c[5], c[6]))
        preferred = (rare + common)[:max_per_image]
        preferred.sort(key=lambda x: -x[5])
        
        lines = []
        for cid, cx, cy, bw, bh, sc, _ in preferred:
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {sc:.4f}")
            class_counts[cid] += 1
            
        with open(txt_dir / f"{stem}.txt", "wb") as f:
            if lines:
                f.write(("\r\n".join(lines) + "\r\n").encode("utf-8"))
            else:
                f.write(b"")

    total_boxes = package_and_verify(txt_dir, out_zip_path)
    return total_boxes, class_counts

def build_raw_top100_submission(raw_predictions, out_zip_path):
    txt_dir = out_zip_path.parent / (out_zip_path.stem + "_txt")
    os.makedirs(txt_dir, exist_ok=True)

    for stem in sorted(raw_predictions.keys()):
        cands = sorted(raw_predictions[stem], key=lambda x: -x[5])[:100]
        lines = []
        for cid, cx, cy, bw, bh, sc, _ in cands:
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {sc:.4f}")
            
        with open(txt_dir / f"{stem}.txt", "wb") as f:
            if lines:
                f.write(("\r\n".join(lines) + "\r\n").encode("utf-8"))
            else:
                f.write(b"")

    total_boxes = package_and_verify(txt_dir, out_zip_path)
    return total_boxes

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(" Official F5 Reproduction & Triple-Tier Verification ".center(70, "="))
    print("=" * 70)
    print(f"Device: {device}")

    cfg_path = ROOT / "src/configs/dfine/custom/aic2026/dfine_x_hires1216x2176_all2000_e54_ailab40_20260930_K2.yml"
    ckpt_path = ROOT / "_xfer/F5.pth"
    data_root = ROOT / "data/local_official_test_phase2_20260921"
    out_dir = ROOT / "outputs/f5_reproduce"
    os.makedirs(out_dir, exist_ok=True)
    raw_cache_path = out_dir / "raw_predictions.pt"

    if raw_cache_path.exists():
        print(f"\n[+] Loading cached raw predictions from {raw_cache_path}...")
        raw_predictions = torch.load(raw_cache_path, map_location="cpu", weights_only=False)
        print(f"[+] Loaded {len(raw_predictions)} image prediction entries.")
    else:
        cfg = YAMLConfig(str(cfg_path))
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state, selected = select_model_state(ck)
        print(f"[+] Loaded F5 state from {selected} ({len(state)} tensors)")

        model = cfg.model.to(device).float().eval()
        model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
        postprocessor = cfg.postprocessor.to(device).eval()

        names = aligned_inventory(data_root)
        images = make_image_index(data_root, names)
        print(f"[+] Found {len(images)} test images in {data_root}")

        index_file = out_dir / "inference_index.json"
        write_json(index_file, {"images": images, "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]})

        cfg.yaml_cfg["num_top_queries"] = 100
        loader_cfg = cfg.yaml_cfg["val_dataloader"]
        loader_cfg.update(total_batch_size=4, num_workers=2, shuffle=False, drop_last=False)
        dataset = loader_cfg["dataset"]
        dataset.update(img_folder=str(data_root / "visible"), infrared_folder=str(data_root / "infrared"),
                       depth_folder=str(data_root / "depth"), ann_file=str(index_file),
                       modality_dropout=0.0, enforce_alignment=True)
        cfg.yaml_cfg["val_dataloader"]["collate_fn"] = {"type": "BatchImageCollateFunction"}
        loader = cfg.val_dataloader

        by_id = {img["id"]: img for img in images}
        raw_predictions = {}
        source_idx = 0

        print("\n[+] Running FP32 model inference across 1,000 test images...")
        t0 = time.time()
        with torch.inference_mode():
            for batch_idx, (samples, targets) in enumerate(tqdm(loader, desc="F5 Inference")):
                samples = samples.to(device).float()
                sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
                preds = model(samples)
                results = postprocessor(preds, sizes)

                for target, res in zip(targets, results):
                    image_id = int(target["image_id"].item())
                    img_info = by_id[image_id]
                    stem = Path(img_info["file_name"]).stem
                    W, H = img_info["width"], img_info["height"]

                    lbls = res["labels"].cpu().tolist()
                    boxes = res["boxes"].float().cpu().tolist()
                    scores = res["scores"].float().cpu().tolist()

                    cands = []
                    for l, b, s in zip(lbls, boxes, scores):
                        if not (math.isfinite(s) and all(math.isfinite(x) for x in b)):
                            continue
                        x1, y1, x2, y2 = b
                        x1 = max(0.0, min(float(x1), float(W)))
                        y1 = max(0.0, min(float(y1), float(H)))
                        x2 = max(0.0, min(float(x2), float(W)))
                        y2 = max(0.0, min(float(y2), float(H)))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        cx = (x1 + x2) / (2.0 * W)
                        cy = (y1 + y2) / (2.0 * H)
                        bw = (x2 - x1) / float(W)
                        bh = (y2 - y1) / float(H)
                        cands.append((int(l), float(cx), float(cy), float(bw), float(bh), float(s), source_idx))
                        source_idx += 1
                    raw_predictions[stem] = cands

        elapsed = time.time() - t0
        print(f"\n[+] Inference finished in {elapsed:.2f}s ({len(images)/elapsed:.2f} FPS)")
        print(f"[+] Saving raw predictions cache to {raw_cache_path}...")
        torch.save(raw_predictions, raw_cache_path)

    # 1. Official 58.105 Peak: .45 rare / .50 common, cap 60
    print("\n" + "=" * 60)
    print(" [1] Packaging F5 Peak Submission (58.105): .45 / .50, cap 60 ")
    print("=" * 60)
    thr_45_50 = {cid: (0.45 if cid in RARE_CLASSES else 0.50) for cid in range(12)}
    zip_58_105 = out_dir / "submission_F5_cap60thr45_50.zip"
    cnt_45_50, dist_45_50 = build_compact_submission(raw_predictions, thr_45_50, MAX_PER_IMAGE, zip_58_105)

    # 2. Benchmark 58.055: .50 rare / .55 common, cap 60
    print("\n" + "=" * 60)
    print(" [2] Packaging F5 Tier-2 Submission (58.055): .50 / .55, cap 60 ")
    print("=" * 60)
    thr_50_55 = {cid: (0.50 if cid in RARE_CLASSES else 0.55) for cid in range(12)}
    zip_58_055 = out_dir / "submission_F5_cap60thr50_55.zip"
    build_compact_submission(raw_predictions, thr_50_55, MAX_PER_IMAGE, zip_58_055)

    # 3. Benchmark 57.707: .35 rare / .40 common, cap 60
    print("\n" + "=" * 60)
    print(" [3] Packaging F5 Tier-3 Submission (57.707): .35 / .40, cap 60 ")
    print("=" * 60)
    thr_35_40 = {cid: (0.35 if cid in RARE_CLASSES else 0.40) for cid in range(12)}
    zip_57_707 = out_dir / "submission_F5_cap60thr35_40.zip"
    build_compact_submission(raw_predictions, thr_35_40, MAX_PER_IMAGE, zip_57_707)

    # 4. Raw Top-100: 53.100
    print("\n" + "=" * 60)
    print(" [4] Packaging F5 Raw Top-100 Submission (53.100) ")
    print("=" * 60)
    zip_raw = out_dir / "submission_F5_raw_top100.zip"
    build_raw_top100_submission(raw_predictions, zip_raw)

    print("\n[+] Detailed Class Distribution for 58.105 Peak (Total: {cnt_45_50}):")
    for cid in range(12):
        print(f"  {cid:2d} ({CLASS_NAMES[cid]:<12}): {dist_45_50[cid]:5d} ({dist_45_50[cid]/cnt_45_50*100:5.2f}%)")

    print("\n" + "=" * 70)
    print(" ALL F5 PACKAGES SUCCESSFULLY PRODUCED & 4-PASS VERIFIED! ".center(70, "="))
    print("=" * 70)

if __name__ == "__main__":
    main()
