#!/usr/bin/env python3
"""
Sovereign V23 Unfiltered Top-100 Submission Generator & Comparator
==================================================================
Runs inference on all 1,000 test images using Sovereign V23 (best_stg2.pth):
1. Native Single-View (Unfiltered Top-100, score >= 0.01, max 100 dets/img)
2. Flip TTA 2-View (WBF Fusion Top-100, score >= 0.01, max 100 dets/img)
Packages both into submission zips and performs full QA validation.
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


def wbf_2view_fusion(boxes_v0, scores_v0, labels_v0, boxes_v1, scores_v1, labels_v1, min_pre_tau=0.01, iou_thr=0.55):
    all_cands = []
    for bx, sc, lb in zip(boxes_v0, scores_v0, labels_v0):
        if sc >= min_pre_tau:
            cx, cy, w, h = bx
            all_cands.append((int(lb), [cx - w/2, cy - h/2, cx + w/2, cy + h/2], float(sc), 0))
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
            if len(view_set) >= 2:
                c = float(confs.mean())
            else:
                c = float(confs.mean() * 0.98)

            cx = min(max((x1 + x2) / 2.0, 0.0), 1.0)
            cy = min(max((y1 + y2) / 2.0, 0.0), 1.0)
            bw = min(max(x2 - x1, 0.0001), 1.0)
            bh = min(max(y2 - y1, 0.0001), 1.0)
            fused_by_class.append((float(c), cls, (cx, cy, bw, bh)))

    fused_by_class.sort(key=lambda x: -x[0])
    return fused_by_class


def main():
    parser = argparse.ArgumentParser(description="Sovereign V23 Unfiltered Top-100 Inference")
    parser.add_argument("--config", default="configs/deimv2/sovereign_v23_zero_distort_2048_fast32.yml")
    parser.add_argument("--weights", default="outputs/sovereign_v23_zero_distort_2048_fast32/best_stg2.pth")
    parser.add_argument("--cache-dir", default="/dev/shm/aic_cache")
    parser.add_argument("--out-dir", default="work_dir/submissions/v23_unfiltered_top100")
    parser.add_argument("--score-thresh", type=float, default=0.01)
    parser.add_argument("--max-dets", type=int, default=100)
    args = parser.parse_args()

    os.chdir("/root/autodl-tmp/AIC/vendor/DEIMv2")
    sys.path.insert(0, ".")

    os.makedirs(args.out_dir, exist_ok=True)
    dir_native = os.path.join(args.out_dir, "txt_native_top100")
    dir_flip = os.path.join(args.out_dir, "txt_flip_tta_top100")
    os.makedirs(dir_native, exist_ok=True)
    os.makedirs(dir_flip, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(" Sovereign V23 Unfiltered Top-100 Submission Generator ".center(75, "="))
    print("=" * 75)
    print(f"Device:       {device} ({torch.cuda.get_device_name(0)})")
    print(f"Model Config: {args.config}")
    print(f"Weights:      {args.weights}")
    print(f"Score Thresh: {args.score_thresh}")
    print(f"Max Dets:     {args.max_dets} per image")
    print(f"Output Dir:   {args.out_dir}")
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
    total_native = 0
    total_flip = 0
    per_class_native = Counter()
    per_class_flip = Counter()
    scores_native = []
    scores_flip = []

    with torch.no_grad():
        for stem in tqdm(stems, desc="V23 Inference"):
            tensor_v0, W, H = load_multimodal_tensor(stem, args.cache_dir)
            tensor_v1 = torch.flip(tensor_v0, dims=[-1])
            batch = torch.cat([tensor_v0, tensor_v1], dim=0).to(device)

            outputs = model(batch)
            logits = outputs["pred_logits"] # [2, 300, 12]
            boxes = outputs["pred_boxes"]   # [2, 300, 4]

            scores, labels = logits.sigmoid().max(dim=-1) # [2, 300]
            sc_np = scores.cpu().numpy()
            lb_np = labels.cpu().numpy()
            bx_np = boxes.cpu().numpy()

            # ----------------- 1. Native View 0 (Unfiltered Top-100) -----------------
            v0_sc = sc_np[0]
            v0_lb = lb_np[0]
            v0_bx = bx_np[0]

            keep_0 = v0_sc >= args.score_thresh
            k0_sc = v0_sc[keep_0]
            k0_lb = v0_lb[keep_0]
            k0_bx = v0_bx[keep_0]

            order_0 = k0_sc.argsort()[::-1][:args.max_dets]
            lines_native = []
            for idx in order_0:
                cid = int(k0_lb[idx])
                s = float(k0_sc[idx])
                cx, cy, w, h = k0_bx[idx]
                cx = max(0.0, min(1.0, float(cx)))
                cy = max(0.0, min(1.0, float(cy)))
                w = max(0.0001, min(1.0, float(w)))
                h = max(0.0001, min(1.0, float(h)))
                lines_native.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {s:.4f}")
                per_class_native[cid] += 1
                scores_native.append(s)

            out_native_p = os.path.join(dir_native, f"{stem}.txt")
            with open(out_native_p, "wb") as f:
                if lines_native:
                    f.write(("\r\n".join(lines_native) + "\r\n").encode("utf-8"))
                    total_native += len(lines_native)
                else:
                    f.write(b"")

            # ----------------- 2. Flip TTA 2-View (WBF Fusion Top-100) -----------------
            fused_cands = wbf_2view_fusion(
                v0_bx, v0_sc, v0_lb,
                bx_np[1], sc_np[1], lb_np[1],
                min_pre_tau=args.score_thresh,
                iou_thr=0.55
            )

            lines_flip = []
            for sc, cid, (cx, cy, w, h) in fused_cands:
                if sc < args.score_thresh:
                    continue
                cx = max(0.0, min(1.0, float(cx)))
                cy = max(0.0, min(1.0, float(cy)))
                w = max(0.0001, min(1.0, float(w)))
                h = max(0.0001, min(1.0, float(h)))
                lines_flip.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {sc:.4f}")
                per_class_flip[cid] += 1
                scores_flip.append(sc)
                if len(lines_flip) >= args.max_dets:
                    break

            out_flip_p = os.path.join(dir_flip, f"{stem}.txt")
            with open(out_flip_p, "wb") as f:
                if lines_flip:
                    f.write(("\r\n".join(lines_flip) + "\r\n").encode("utf-8"))
                    total_flip += len(lines_flip)
                else:
                    f.write(b"")

    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(" Inference Completed ".center(75, "="))
    print(f"Elapsed: {elapsed:.2f}s ({len(stems)/elapsed:.2f} FPS)")
    print(f"Native Top-100: Total Boxes = {total_native}, Avg/Img = {total_native/len(stems):.2f}")
    print(f"Flip TTA Top-100: Total Boxes = {total_flip}, Avg/Img = {total_flip/len(stems):.2f}")
    print("=" * 75)

    # Package into ZIP files
    zip_native = os.path.join(args.out_dir, "submission_v23_native_unfiltered_top100.zip")
    zip_flip = os.path.join(args.out_dir, "submission_v23_flip_tta_unfiltered_top100.zip")

    for zip_p, src_dir in [(zip_native, dir_native), (zip_flip, dir_flip)]:
        with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
            for txt_f in sorted(glob.glob(os.path.join(src_dir, "*.txt"))):
                zf.write(txt_f, arcname=os.path.basename(txt_f))
        print(f"[+] Packaged {zip_p} ({os.path.getsize(zip_p)/1e6:.2f} MB)")

    # Run QA verification on both
    for name, src_dir in [("Native Top-100", dir_native), ("Flip TTA Top-100", dir_flip)]:
        txt_list = glob.glob(os.path.join(src_dir, "*.txt"))
        assert len(txt_list) == 1000, f"Expected 1000 txt files, got {len(txt_list)}"
        box_cnt = 0
        for tf in txt_list:
            with open(tf, "rb") as f:
                bdata = f.read()
            if not bdata:
                continue
            assert b"\r\n" in bdata, f"Missing CRLF in {tf}"
            text = bdata.decode("utf-8")
            for line in text.strip().split("\r\n"):
                parts = line.split()
                assert len(parts) == 6, f"Invalid format in {tf}: {line}"
                cid = int(parts[0])
                cx, cy, w, h, sc = map(float, parts[1:])
                assert 0 <= cid <= 11
                assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0
                assert 0.0 <= sc <= 1.0
                box_cnt += 1
        print(f"[QA PASS] {name}: 1,000 files verified, total {box_cnt} valid boxes.")


if __name__ == "__main__":
    main()
