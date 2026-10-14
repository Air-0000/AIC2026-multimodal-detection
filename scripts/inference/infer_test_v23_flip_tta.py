#!/usr/bin/env python3
"""
Sovereign V23 Zero-Distortion Horizontal Flip TTA Submission Generator
======================================================================
Performs high-throughput 2-view Horizontal Flip Test-Time Augmentation (TTA)
with Weighted Box Fusion (WBF) on all 1,000 Phase 2 test images.

Pipeline:
1. Batch input: [original, horizontal_flip] (batch_size=2) in a single GPU pass.
2. Mirror-restores flipped predictions: cx -> 1.0 - cx.
3. Class-wise Weighted Box Fusion (WBF) combining both views:
   - Eliminates single-side spatial jitter, boosting AP75-AP95 localization.
   - Recalls asymmetrically shadowed / occluded targets.
   - Suppresses single-view spurious noise.
4. Outputs:
   - submission_v23_flip_tta_thr45.zip (tau = 0.45, high-confidence benchmark)
   - submission_v23_flip_tta_thr40.zip (tau = 0.40, expanded recall benchmark)
"""

import os
import sys
import glob
import time
import zipfile
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms.v2 as T
import torch.nn.functional as F

aic_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [aic_root, os.path.join(aic_root, "vendor", "DEIMv2"), os.path.join(aic_root, "scripts", "lib")]:
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

    # 6ch layout: [RGB(3), aligned_ir(1), lbg_ir(1), depth(1)]
    tensor_6ch = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0)
    tensor_resized = T.functional.resize(tensor_6ch, [1152, 2048])
    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)
    tensor_norm = (tensor_resized - mean) / std
    return tensor_norm.unsqueeze(0), W, H


def iou_xyxy(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / ua if ua > 0 else 0.0


def wbf_2view_fusion(boxes_v0, scores_v0, labels_v0, boxes_v1, scores_v1, labels_v1, min_pre_tau=0.25, iou_thr=0.55):
    """
    Fuses View 0 (Original) and View 1 (Horizontally Flipped) predictions.
    boxes_v0, boxes_v1 are normalized cxcywh [0, 1].
    """
    all_cands = []
    # View 0 (Original)
    for bx, sc, lb in zip(boxes_v0, scores_v0, labels_v0):
        if sc >= min_pre_tau:
            cx, cy, w, h = bx
            all_cands.append((int(lb), [cx - w/2, cy - h/2, cx + w/2, cy + h/2], float(sc), 0))
    # View 1 (Flipped: cx restored to 1.0 - cx)
    for bx, sc, lb in zip(boxes_v1, scores_v1, labels_v1):
        if sc >= min_pre_tau:
            cx, cy, w, h = bx
            cx_restored = 1.0 - cx
            all_cands.append((int(lb), [cx_restored - w/2, cy - h/2, cx_restored + w/2, cy + h/2], float(sc), 1))

    fused_by_class = []
    for cls in sorted(set(b[0] for b in all_cands)):
        members = sorted([b for b in all_cands if b[0] == cls], key=lambda b: b[2], reverse=True)
        clusters = []
        for box in members:
            for cl in clusters:
                if iou_xyxy(box[1], cl[0][1]) >= iou_thr:
                    cl.append(box)
                    break
            else:
                clusters.append([box])

        for cl in clusters:
            confs = np.array([b[2] for b in cl])
            ws = confs / confs.sum()
            x1 = sum(b[1][0] * w for b, w in zip(cl, ws))
            y1 = sum(b[1][1] * w for b, w in zip(cl, ws))
            x2 = sum(b[1][2] * w for b, w in zip(cl, ws))
            y2 = sum(b[1][3] * w for b, w in zip(cl, ws))

            view_set = set(b[3] for b in cl)
            # If both views agreed: strong consensus, full average score
            # If only 1 view saw it: slightly dampened to suppress spurious noise
            if len(view_set) >= 2:
                c = float(confs.mean())
            else:
                c = float(confs.mean() * 0.96)

            cx = min(max((x1 + x2) / 2.0, 0.0), 1.0)
            cy = min(max((y1 + y2) / 2.0, 0.0), 1.0)
            bw = min(max(x2 - x1, 0.0001), 1.0)
            bh = min(max(y2 - y1, 0.0001), 1.0)
            fused_by_class.append((float(c), cls, (cx, cy, bw, bh), len(view_set)))

    fused_by_class.sort(key=lambda x: -x[0])
    return fused_by_class


def format_detections(fused_cands, tau=0.45, max_budget=100):
    lines = []
    for sc, cid, (cx, cy, bw, bh), n_views in fused_cands:
        if sc < tau:
            continue
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {sc:.6f}")
        if len(lines) >= max_budget:
            break
    return lines


def main():
    parser = argparse.ArgumentParser(description="Sovereign V23 Horizontal Flip TTA Engine")
    parser.add_argument("--config", default="configs/deimv2/sovereign_v23_zero_distort_2048_fast32.yml")
    parser.add_argument("--weights", default="outputs/sovereign_v23_zero_distort_2048_fast32/best_stg2.pth")
    parser.add_argument("--cache-dir", default="/dev/shm/aic_cache")
    parser.add_argument("--out-dir", default="work_dir/v23_flip_tta_submission")
    args = parser.parse_args()

    os.chdir("/root/autodl-tmp/AIC/vendor/DEIMv2")
    sys.path.insert(0, ".")

    os.makedirs(args.out_dir, exist_ok=True)
    dir_thr45 = os.path.join(args.out_dir, "txt_thr45")
    dir_thr40 = os.path.join(args.out_dir, "txt_thr40")
    os.makedirs(dir_thr45, exist_ok=True)
    os.makedirs(dir_thr40, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(" Sovereign V23 Horizontal Flip TTA (2-View Batch + WBF) Engine ".center(75, "="))
    print("=" * 75)
    print(f"Device:       {device} ({torch.cuda.get_device_name(0)})")
    print(f"Model Config: {args.config}")
    print(f"Weights:      {args.weights}")
    print(f"Output Dir:   {args.out_dir}")
    print(f"Modality:     RGB + Aligned IR (Destriped) + LBG IR + Depth P0 (16-bit)")
    print(f"TTA Views:    View 0 (Original) + View 1 (Horizontal Flip)")
    print("=" * 75)

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
    print(f"[+] Found {len(stems)} test images in {vis_dir}.\n")

    t0 = time.time()
    total_45, total_40 = 0, 0
    empty_45, empty_40 = 0, 0
    both_view_agreed_cnt = 0
    total_cand_cnt = 0

    with torch.no_grad():
        for stem in tqdm(stems, desc="2-View Flip TTA Inference"):
            tensor_v0, W, H = load_multimodal_tensor(stem, args.cache_dir)
            tensor_v1 = torch.flip(tensor_v0, dims=[-1])
            batch = torch.cat([tensor_v0, tensor_v1], dim=0).to(device) # [2, 6, 1152, 2048]

            outputs = model(batch)
            logits = outputs["pred_logits"] # [2, 300, 12]
            boxes = outputs["pred_boxes"]   # [2, 300, 4]

            scores, labels = logits.sigmoid().max(dim=-1) # [2, 300]
            sc_np = scores.cpu().numpy()
            lb_np = labels.cpu().numpy()
            bx_np = boxes.cpu().numpy()

            fused_cands = wbf_2view_fusion(
                bx_np[0], sc_np[0], lb_np[0],
                bx_np[1], sc_np[1], lb_np[1],
                min_pre_tau=0.20,
                iou_thr=0.55
            )

            for cand in fused_cands:
                total_cand_cnt += 1
                if cand[3] >= 2:
                    both_view_agreed_cnt += 1

            # 1. tau = 0.45
            lines_45 = format_detections(fused_cands, tau=0.45, max_budget=100)
            out_p_45 = os.path.join(dir_thr45, f"{stem}.txt")
            with open(out_p_45, "wb") as f:
                if lines_45:
                    f.write(("\r\n".join(lines_45) + "\r\n").encode("utf-8"))
                    total_45 += len(lines_45)
                else:
                    f.write(b"")
                    empty_45 += 1

            # 2. tau = 0.40
            lines_40 = format_detections(fused_cands, tau=0.40, max_budget=100)
            out_p_40 = os.path.join(dir_thr40, f"{stem}.txt")
            with open(out_p_40, "wb") as f:
                if lines_40:
                    f.write(("\r\n".join(lines_40) + "\r\n").encode("utf-8"))
                    total_40 += len(lines_40)
                else:
                    f.write(b"")
                    empty_40 += 1

    elapsed = time.time() - t0
    fps = len(stems) / elapsed
    agreement_rate = (both_view_agreed_cnt / total_cand_cnt * 100) if total_cand_cnt > 0 else 0.0

    print("\n" + "=" * 75)
    print(" Flip TTA Inference Completed ".center(75, "="))
    print(f"Total Images:     {len(stems)} in {elapsed:.2f}s ({fps:.2f} FPS batch-parallel)")
    print(f"2-View Consensus: {agreement_rate:.1f}% ({both_view_agreed_cnt}/{total_cand_cnt} candidates)")
    print(f"Threshold tau=0.45: Total Dets={total_45}, Avg/Img={total_45/len(stems):.2f}, Empty={empty_45}")
    print(f"Threshold tau=0.40: Total Dets={total_40}, Avg/Img={total_40/len(stems):.2f}, Empty={empty_40}")
    print("=" * 75)

    # Packaging into ZIP files
    zip_45 = os.path.join(args.out_dir, "submission_v23_flip_tta_thr45.zip")
    zip_40 = os.path.join(args.out_dir, "submission_v23_flip_tta_thr40.zip")

    for zip_path, src_dir in [(zip_45, dir_thr45), (zip_40, dir_thr40)]:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for txt_file in sorted(glob.glob(os.path.join(src_dir, "*.txt"))):
                zf.write(txt_file, arcname=os.path.basename(txt_file))
        print(f"[+] Successfully generated submission ZIP: {zip_path} ({os.path.getsize(zip_path)/1e6:.2f} MB)")

    # Strict QA Gate on tau=0.45
    print("\n[+] Running 4-Pass Strict QA Gate on tau=0.45 ZIP...")
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
        assert b"\r\n" in bdata, f"File {tf} missing CRLF"
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
