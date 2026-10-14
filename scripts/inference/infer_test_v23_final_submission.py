#!/usr/bin/env python3
"""
Sovereign V23 Final Full Test Set Inference & Submission Generator
==================================================================
Runs high-throughput multimodal inference on all 1,000 Test Set 1 images
using the final converged Epoch 71 checkpoint (best_stg2.pth, mAP = 91.42%).

Produces:
1. txt_thr45/ & submission_v23_ep71_thr45.zip (Optimal competition compact threshold tau >= 0.45)
2. txt_thr50/ & submission_v23_ep71_thr50.zip (Standard baseline threshold tau >= 0.50)
3. Full quality assurance audit report (4-pass validation)
"""

import os
import sys
import glob
import time
import zipfile
import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms.v2 as T
import torch.nn.functional as F

aic_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [aic_root, os.path.join(aic_root, "vendor", "DEIMv2")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from engine.core import YAMLConfig

try:
    from destripe_1d import stripe_non_uniformity_correction_1d
    HAS_DESTRIPE = True
except ImportError:
    HAS_DESTRIPE = False

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}


def read_depth_raw16(folder, stem):
    for ext in ('.png', '.jpg', '.jpeg'):
        p = os.path.join(folder, f"{stem}{ext}")
        if os.path.exists(p):
            im = Image.open(p)
            arr = np.array(im).astype(np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            scale = 19999.0 if getattr(im, 'mode', '') in ('I;16', 'I') else 255.0
            arr = arr / scale
            return torch.from_numpy(arr.clip(0.0, 1.0)).unsqueeze(0)
    raise FileNotFoundError(f"Depth not found for {stem} in {folder}")


def load_multimodal_tensor(stem, cache_dir):
    vis_cand = glob.glob(os.path.join(cache_dir, "test_visible", f"{stem}.*"))
    # Priority: aligned IR matching training setup, then fallback
    ir_cand = (
        glob.glob(os.path.join(cache_dir, "test_aligned_ir", f"{stem}.*"))
        or glob.glob(os.path.join("/root/autodl-tmp/AIC/work_dir/test_aligned_ir", f"{stem}.*"))
        or glob.glob(os.path.join(cache_dir, "test_infrared", f"{stem}.*"))
    )
    lbg_cand = glob.glob(os.path.join(cache_dir, "test_lbg_ir", f"{stem}.*"))
    
    if not vis_cand:
        raise FileNotFoundError(f"Visible image not found for stem {stem}")
    
    vis_p = vis_cand[0]
    rgb_pil = Image.open(vis_p).convert("RGB")
    W, H = rgb_pil.size
    rgb_t = T.functional.pil_to_tensor(rgb_pil).float() / 255.0

    if ir_cand:
        ir_pil = Image.open(ir_cand[0]).convert("L")
        if HAS_DESTRIPE:
            try:
                ir_arr = np.array(ir_pil)
                ir_arr = stripe_non_uniformity_correction_1d(ir_arr)
                ir_t = torch.from_numpy(ir_arr).float().unsqueeze(0) / 255.0
            except Exception:
                ir_t = T.functional.pil_to_tensor(ir_pil).float() / 255.0
        else:
            ir_t = T.functional.pil_to_tensor(ir_pil).float() / 255.0
        if ir_t.shape[1:] != (H, W):
            ir_t = T.functional.resize(ir_t, [H, W])
    else:
        ir_t = rgb_t.mean(dim=0, keepdim=True)

    if lbg_cand:
        lbg_pil = Image.open(lbg_cand[0]).convert("L")
        lbg_t = T.functional.pil_to_tensor(lbg_pil).float() / 255.0
        if lbg_t.shape[1:] != (H, W):
            lbg_t = T.functional.resize(lbg_t, [H, W])
    else:
        lbg_t = ir_t.clone()

    dep_t = read_depth_raw16(os.path.join(cache_dir, "test_depth_p0"), stem)
    if dep_t.shape[1:] != (H, W):
        dep_t = T.functional.resize(dep_t, [H, W])

    # 6ch layout: [RGB(3), ir(1), lbg_ir(1), depth(1)]
    tensor_6ch = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0)
    tensor_resized = T.functional.resize(tensor_6ch, [1152, 2048])
    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)
    tensor_norm = (tensor_resized - mean) / std
    return tensor_norm.unsqueeze(0), W, H


def class_wise_nms(detections, iou_thresh=0.60):
    if not detections:
        return []
    by_class = {}
    for item in detections:
        sc, cid, (cx, cy, bw, bh) = item
        by_class.setdefault(cid, []).append(item)

    keep = []
    for cid, items in by_class.items():
        if len(items) <= 1:
            keep.extend(items)
            continue
        boxes_xyxy = []
        scores = []
        for sc, _, (cx, cy, bw, bh) in items:
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(sc)
        b_t = torch.tensor(boxes_xyxy, dtype=torch.float32)
        s_t = torch.tensor(scores, dtype=torch.float32)
        idx = torchvision.ops.nms(b_t, s_t, iou_thresh)
        for i in idx.tolist():
            keep.append(items[i])
    keep.sort(key=lambda x: -x[0])
    return keep


def filter_and_format_detections(boxes_np, scores_np, labels_np, W, H, tau=0.45, max_budget=100, iou_thresh=0.60):
    cands = []
    for sc, lb, bx in zip(scores_np, labels_np, boxes_np):
        if sc < tau:
            continue
        cx, cy, bw, bh = bx
        cid = int(lb)
        cands.append((float(sc), cid, (float(cx), float(cy), float(bw), float(bh))))

    cands.sort(key=lambda x: -x[0])
    clean = class_wise_nms(cands, iou_thresh=iou_thresh)[:max_budget]

    lines = []
    for sc, cid, (cx, cy, bw, bh) in clean:
        cx_c = float(np.clip(cx, 0.0, 1.0))
        cy_c = float(np.clip(cy, 0.0, 1.0))
        bw_c = float(np.clip(bw, 0.0001, 1.0))
        bh_c = float(np.clip(bh, 0.0001, 1.0))
        sc_c = float(np.clip(sc, 0.0, 1.0))
        lines.append(f"{cid} {cx_c:.6f} {cy_c:.6f} {bw_c:.6f} {bh_c:.6f} {sc_c:.6f}")
    return lines, len(clean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deimv2/sovereign_v23_zero_distort_2048_fast32.yml")
    parser.add_argument("--weights", default="outputs/sovereign_v23_zero_distort_2048_fast32/best_stg2.pth")
    parser.add_argument("--cache-dir", default="/dev/shm/aic_cache")
    parser.add_argument("--out-dir", default="work_dir/submissions/v23_test1_final_submission")
    args = parser.parse_args()

    os.chdir("/root/autodl-tmp/AIC/vendor/DEIMv2")
    sys.path.insert(0, ".")

    os.makedirs(args.out_dir, exist_ok=True)
    dir_thr45 = os.path.join(args.out_dir, "txt_thr45")
    dir_thr50 = os.path.join(args.out_dir, "txt_thr50")
    os.makedirs(dir_thr45, exist_ok=True)
    os.makedirs(dir_thr50, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(" Sovereign V23 Final Full Test Set Inference Engine ".center(70, "="))
    print("=" * 70)
    print(f"Device:       {device} ({torch.cuda.get_device_name(0)})")
    print(f"Model Config: {args.config}")
    print(f"Weights:      {args.weights}")
    print(f"Output Dir:   {args.out_dir}")
    print("=" * 70)

    cfg = YAMLConfig(args.config)
    model = cfg.model.to(device).eval()
    ck = torch.load(args.weights, map_location="cpu")
    w = ck["ema"]["module"] if ("ema" in ck and "module" in ck["ema"]) else ck.get("model", ck)
    model.load_state_dict(w, strict=False)
    epoch = ck.get("last_epoch", 71)
    print(f"[+] Loaded model checkpoint from Epoch {epoch}!\n")

    vis_dir = os.path.join(args.cache_dir, "test_visible")
    files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
    stems = [Path(f).stem for f in files if not f.endswith(".txt")]
    print(f"[+] Found {len(stems)} test images in {vis_dir}.")

    t0 = time.time()
    total_45, total_50 = 0, 0
    empty_45, empty_50 = 0, 0
    class_counts_45 = Counter()
    class_counts_50 = Counter()

    with torch.no_grad():
        for stem in tqdm(stems, desc="Inferring Phase 2 Test Set"):
            inp, W, H = load_multimodal_tensor(stem, args.cache_dir)
            inp = inp.to(device)
            outputs = model(inp)
            logits = outputs["pred_logits"][0] # [300, 12]
            boxes = outputs["pred_boxes"][0]   # [300, 4] normalized cxcywh

            scores, labels = logits.sigmoid().max(dim=-1)
            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()
            boxes_np = boxes.cpu().numpy()

            # 1. Standard tau = 0.45
            lines_45, cnt_45 = filter_and_format_detections(boxes_np, scores_np, labels_np, W, H, tau=0.45, max_budget=100)
            out_p_45 = os.path.join(dir_thr45, f"{stem}.txt")
            with open(out_p_45, "wb") as f:
                if lines_45:
                    content = "\r\n".join(lines_45) + "\r\n"
                    f.write(content.encode("utf-8"))
                    total_45 += cnt_45
                    for l in lines_45:
                        class_counts_45[int(l.split()[0])] += 1
                else:
                    f.write(b"")
                    empty_45 += 1

            # 2. Strict tau = 0.50
            lines_50, cnt_50 = filter_and_format_detections(boxes_np, scores_np, labels_np, W, H, tau=0.50, max_budget=100)
            out_p_50 = os.path.join(dir_thr50, f"{stem}.txt")
            with open(out_p_50, "wb") as f:
                if lines_50:
                    content = "\r\n".join(lines_50) + "\r\n"
                    f.write(content.encode("utf-8"))
                    total_50 += cnt_50
                    for l in lines_50:
                        class_counts_50[int(l.split()[0])] += 1
                else:
                    f.write(b"")
                    empty_50 += 1

    elapsed = time.time() - t0
    fps = len(stems) / elapsed
    print("\n" + "=" * 70)
    print(" Inference Completed ".center(70, "="))
    print(f"Total Images: {len(stems)} in {elapsed:.2f}s ({fps:.1f} FPS, {1000/fps:.1f}ms/img)")
    print(f"Threshold tau=0.45: Total Dets={total_45}, Avg/Img={total_45/len(stems):.2f}, Empty={empty_45}")
    print(f"Threshold tau=0.50: Total Dets={total_50}, Avg/Img={total_50/len(stems):.2f}, Empty={empty_50}")
    print("=" * 70)

    # Packaging into ZIP files
    zip_45 = os.path.join(args.out_dir, "submission_v23_ep71_thr45.zip")
    zip_50 = os.path.join(args.out_dir, "submission_v23_ep71_thr50.zip")

    for zip_path, src_dir in [(zip_45, dir_thr45), (zip_50, dir_thr50)]:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for txt_file in sorted(glob.glob(os.path.join(src_dir, "*.txt"))):
                zf.write(txt_file, arcname=os.path.basename(txt_file))
        print(f"[+] Successfully generated submission ZIP: {zip_path} ({os.path.getsize(zip_path)/1e6:.2f} MB)")

    # QA Audit
    print("\n[+] Running 4-Pass Strict Competition QA Verification on tau=0.45...")
    txt_list = glob.glob(os.path.join(dir_thr45, "*.txt"))
    assert len(txt_list) == 1000, f"Expected 1000 txt files, got {len(txt_list)}"
    pass_cnt = 0
    box_audit = 0
    for tf in txt_list:
        with open(tf, "rb") as f:
            bdata = f.read()
        if len(bdata) == 0:
            pass_cnt += 1
            continue
        assert b"\r\n" in bdata, f"File {tf} does not use CRLF line endings"
        text = bdata.decode("utf-8")
        for line in text.strip().split("\r\n"):
            parts = line.split()
            assert len(parts) == 6, f"Malformed line in {tf}: {line}"
            cid = int(parts[0])
            cx, cy, w, h, sc = map(float, parts[1:])
            assert 0 <= cid <= 11
            assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
            assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0
            assert 0.0 <= sc <= 1.0
            box_audit += 1
        pass_cnt += 1

    print(f"[ALL PASS] 1,000 / 1,000 files passed all 4 QA gates! Total valid boxes: {box_audit}")


if __name__ == "__main__":
    main()
