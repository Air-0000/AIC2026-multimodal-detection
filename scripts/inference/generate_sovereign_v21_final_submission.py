#!/usr/bin/env python3
"""
Sovereign V21 Apex Soup Final Official Submission Generator
===========================================================
Production-grade submission generator combining:
1. Sovereign V21 Apex Soup (0.60 V21 Distill + 0.40 Teacher V20, 141MB FP16)
2. Query Argmax Single-Class Resolution (Eliminates multi-label phantom boxes)
3. Class-wise NMS (IoU=0.55) (Eliminates duplicate boxes on the same object)
4. Confidence Threshold Gate (tau >= 0.500) matching 6,000 ~ 7,500 target box prior
5. Hard Per-Image Budget Truncation (Max 28 boxes per image)
6. Strict Official Specification Compliance:
   - Exactly 1000 TXT files matching Phase 2 stems
   - Empty files for zero-detection images
   - Normalized [0, 1] cx cy w h conf, CRLF endings
   - Verified with scripts/eval/check_submission.py -> RESULT: PASS
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
import torch.nn.functional as F
import torchvision.transforms.functional as TF

aic_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in [aic_root, os.path.join(aic_root, 'vendor', 'DEIMv2'), os.path.join(aic_root, 'scripts', 'lib')]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from engine.core import YAMLConfig
from scripts.analysis.eval_orthogonal_6ch_soup import (
    compute_phi_vt_and_3d_edges,
    ir_method1_multiscale_mca,
    ir_method3_anisotropic_thermal_deblooming,
)

MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}


def load_multimodal_tensor(stem, test_dir, device):
    """
    Standard native multi-modal tensor reader matching Sovereign V21:
      ch 0,1,2 = RGB visible
      ch 3     = Aligned IR (Anisotropic Deblooming)
      ch 4     = LBG IR (MCA * phi_vt + 0.3 * Edge3D)
      ch 5     = Metric Depth P0 (normalized [0, 1])
    """
    # 1. RGB Visible
    vis_cand = glob.glob(os.path.join(test_dir, 'visible', f'{stem}.*'))
    if not vis_cand:
        vis_cand = glob.glob(os.path.join(test_dir, f'{stem}.*'))
    if not vis_cand:
        raise FileNotFoundError(f"Visible image not found for stem {stem} in {test_dir}")

    vis_p = vis_cand[0]
    vis_im = Image.open(vis_p).convert('RGB')
    orig_w, orig_h = vis_im.size
    rgb_t = TF.pil_to_tensor(vis_im).float() / 255.0
    vis_bgr = cv2.imread(vis_p)

    # 2. IR with Deblooming & MCA
    ir_cand = glob.glob(os.path.join(test_dir, 'infrared', f'{stem}.*'))
    if not ir_cand:
        ir_cand = glob.glob(os.path.join(test_dir, 'aligned_ir', f'{stem}.*'))
    if not ir_cand:
        ir_cand = glob.glob(os.path.join('work_dir', 'test_aligned_ir', f'{stem}.*'))

    if ir_cand:
        ir_im = Image.open(ir_cand[0]).convert('L')
        ir_deb = ir_method3_anisotropic_thermal_deblooming(np.array(ir_im), vis_bgr)
        mca_np = ir_method1_multiscale_mca(ir_deb)
        ir_t = torch.from_numpy(ir_deb.astype(np.float32) / 255.0).unsqueeze(0)
        if ir_t.shape[1:] != (orig_h, orig_w):
            ir_t = TF.resize(ir_t, [orig_h, orig_w])
    else:
        ir_deb = np.array(vis_im.convert('L'))
        mca_np = ir_deb
        ir_t = rgb_t.mean(dim=0, keepdim=True)

    # 3. Metric Depth & Surface Normal Gate
    dep_cand = glob.glob(os.path.join(test_dir, 'depth_p0', f'{stem}.*'))
    if not dep_cand:
        dep_cand = glob.glob(os.path.join(test_dir, 'depth', f'{stem}.*'))

    if dep_cand:
        dep_im = Image.open(dep_cand[0])
        dep_arr = np.array(dep_im, dtype=np.float32)
        if dep_arr.ndim == 3:
            dep_arr = dep_arr[..., 0]
        scale = 19999.0 if getattr(dep_im, 'mode', '') in ('I;16', 'I') else (65535.0 if dep_arr.max() > 255.5 else 255.0)
        dep_norm = np.clip(dep_arr / scale, 0.0, 1.0)
        dep_t = torch.from_numpy(dep_norm).unsqueeze(0)
        if dep_t.shape[1:] != (orig_h, orig_w):
            dep_t = TF.resize(dep_t, [orig_h, orig_w])
        phi_vt, edge_3d = compute_phi_vt_and_3d_edges(dep_arr)
    else:
        dep_t = torch.zeros((1, orig_h, orig_w), dtype=torch.float32)
        phi_vt = np.ones((orig_h, orig_w), dtype=np.float32)
        edge_3d = np.zeros((orig_h, orig_w), dtype=np.float32)

    # 4. LBG Channel Assembly
    ch4_np = np.clip(mca_np.astype(np.float32) * phi_vt + 0.30 * 255.0 * edge_3d, 0, 255).astype(np.uint8)
    lbg_t = torch.from_numpy(ch4_np.astype(np.float32) / 255.0).unsqueeze(0)
    if lbg_t.shape[1:] != (orig_h, orig_w):
        lbg_t = TF.resize(lbg_t, [orig_h, orig_w])

    # 5. Concat 6ch, resize 640x640, normalize
    t6 = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0)
    inp = ((TF.resize(t6, [640, 640]).to(device) - MEAN_6CH.to(device)) / STD_6CH.to(device)).unsqueeze(0)
    return inp, orig_w, orig_h


def box_iou_xyxy(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-9)


def class_wise_nms(candidates, iou_thresh=0.55):
    """
    Suppresses duplicate overlapping queries of the same class.
    candidates: list of (score, cid, [cx, cy, w, h]) sorted descending by score.
    """
    kept = []
    for sc, cid, (cx, cy, bw, bh) in candidates:
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        b_xyxy = (x1, y1, x2, y2)

        is_dup = False
        for _, kcid, (kcx, kcy, kbw, kbh) in kept:
            if cid == kcid:
                kx1 = kcx - kbw / 2.0
                ky1 = kcy - kbh / 2.0
                kx2 = kcx + kbw / 2.0
                ky2 = kcy + kbh / 2.0
                kb_xyxy = (kx1, ky1, kx2, ky2)

                if box_iou_xyxy(b_xyxy, kb_xyxy) >= iou_thresh:
                    is_dup = True
                    break

        if not is_dup:
            kept.append((sc, cid, (cx, cy, bw, bh)))
    return kept


def main():
    parser = argparse.ArgumentParser(description="Sovereign V21 Apex Soup Final Official Submission Generator")
    parser.add_argument("--config", default="vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_v14_rtx6000.yml")
    parser.add_argument("--weights", default="weights/sovereign_v21_apex_soup_581.pth")
    parser.add_argument("--test-dir", default="dataset/test/AIC2026_PHASE_2_1000")
    parser.add_argument("--out-dir", default="work_dir/submissions/submission_sovereign_v21_apex_soup_thr50")
    parser.add_argument("--zip-name", default="submission_sovereign_v21_apex_soup_thr50.zip")
    parser.add_argument("--thresh", type=float, default=0.50, help="Confidence threshold gate (default: 0.50)")
    parser.add_argument("--nms-iou", type=float, default=0.55, help="Class-wise NMS IoU threshold (default: 0.55)")
    parser.add_argument("--max-dets", type=int, default=28, help="Max detections per image (default: 28)")
    args = parser.parse_args()

    pred_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===================================================================")
    print(" Sovereign V21 Apex Soup Official Submission Generator (Release) ")
    print("===================================================================")
    print(f"Model Config:     {args.config}")
    print(f"Weights:          {args.weights}")
    print(f"Test Set Dir:     {args.test_dir}")
    print(f"Output Dir:       {args.out_dir}")
    print(f"Confidence Gate:  tau >= {args.thresh:.3f}")
    print(f"Class-wise NMS:   IoU >= {args.nms_iou:.2f}")
    print(f"Per-Image Limit:  Top-{args.max_dets} boxes")
    print(f"Inference Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("===================================================================")

    cfg = YAMLConfig(args.config)
    model = cfg.model.to(device).eval()
    ck = torch.load(args.weights, map_location="cpu")
    w = ck["ema"]["module"] if ("ema" in ck and "module" in ck["ema"]) else ck.get("model", ck)
    model.load_state_dict(w, strict=False)

    vis_dir = os.path.join(args.test_dir, "visible")
    img_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
    stems = [Path(f).stem for f in img_files if not f.endswith(".txt")]
    print(f"Discovered {len(stems)} official Phase 2 test images.\n")

    total_dets = 0
    empty_files = 0
    class_counter = Counter()

    t0 = time.time()
    with torch.no_grad():
        for idx, stem in enumerate(tqdm(stems, desc="Generating V21 Apex Soup Predictions")):
            try:
                inp, orig_w, orig_h = load_multimodal_tensor(stem, args.test_dir, device)
            except Exception as e:
                print(f"\n[!] Error loading {stem}: {e}, outputting empty file.")
                with open(os.path.join(pred_dir, f"{stem}.txt"), "wb") as f:
                    f.write(b"")
                empty_files += 1
                continue

            # Model Forward Pass
            outputs = model(inp)
            logits = outputs["pred_logits"][0]  # [num_queries, num_classes]
            boxes = outputs["pred_boxes"][0]    # [num_queries, 4] normalized cxcywh

            # Stage 1: Query Argmax Single-Class Resolution (Zero multi-label cross-leakage)
            scores, labels = logits.sigmoid().max(dim=-1)

            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()
            boxes_np = boxes.cpu().numpy()

            # Stage 2: Confidence Threshold Filtering
            candidates = []
            for sc, lb, bx in zip(scores_np, labels_np, boxes_np):
                if sc >= args.thresh:
                    candidates.append((float(sc), int(lb), (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]))))

            # Sort descending by score
            candidates.sort(key=lambda x: -x[0])

            # Stage 3: Class-wise NMS (Eliminates duplicate boxes on the same object)
            clean_candidates = class_wise_nms(candidates, iou_thresh=args.nms_iou)

            # Stage 4: Top-28 Per Image Budget Truncation
            final_boxes = clean_candidates[:args.max_dets]

            out_lines = []
            for sc, cid, (cx, cy, bw, bh) in final_boxes:
                cx_c = float(np.clip(cx, 0.0, 1.0))
                cy_c = float(np.clip(cy, 0.0, 1.0))
                bw_c = float(np.clip(bw, 0.0, 1.0))
                bh_c = float(np.clip(bh, 0.0, 1.0))
                sc_c = float(np.clip(sc, 0.0, 1.0))

                out_lines.append(f"{cid} {cx_c:.6f} {cy_c:.6f} {bw_c:.6f} {bh_c:.6f} {sc_c:.6f}")
                class_counter[cid] += 1
                total_dets += 1

            # Stage 5: Write Official Formatted Text File (CRLF endings)
            out_path = os.path.join(pred_dir, f"{stem}.txt")
            with open(out_path, "wb") as f:
                if out_lines:
                    content = "\r\n".join(out_lines) + "\r\n"
                    f.write(content.encode("utf-8"))
                else:
                    f.write(b"")
                    empty_files += 1

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(" INFERENCE COMPLETE ".center(70, "="))
    print("=" * 70)
    print(f"Total Test Images: {len(stems)}")
    print(f"Total Detections:  {total_dets} (avg {total_dets/len(stems):.2f} boxes/img)")
    print(f"Empty Txt Files:   {empty_files}")
    print(f"Time Taken:        {elapsed:.1f}s ({elapsed/len(stems)*1000:.1f}ms/img)")
    print("-------------------------------------------------------------------")
    print("Predictions Class Distribution:")
    for cid in range(12):
        cname = CLASS_NAMES[cid]
        cnt = class_counter[cid]
        print(f"  [{cid:>2d}] {cname:<12s}: {cnt:>5d} ({cnt/len(stems):.2f}/img)")
    print("=" * 70)

    # Pack to ZIP
    zip_path = os.path.join(args.out_dir, args.zip_name)
    print(f"\nPacking {len(stems)} files to ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for txt_f in sorted(glob.glob(os.path.join(pred_dir, "*.txt"))):
            z.write(txt_f, os.path.basename(txt_f))

    zip_size_kb = os.path.getsize(zip_path) / 1024
    print(f"Successfully generated ZIP: {zip_path} ({zip_size_kb:.1f} KB)")

    # Run Official Format Audit Script
    check_script = os.path.join(aic_root, "scripts/eval/check_submission.py")
    if os.path.exists(check_script):
        print("\nRunning Official Submission Format Verification...")
        ret = os.system(f"python3 {check_script} {pred_dir} {vis_dir}")
        if ret == 0:
            print("\n[SUCCESS] SUBMISSION PACKAGE PASSED ALL OFFICIAL COMPETITION SPEC CHECKS!")
        else:
            print("\n[WARNING] check_submission reported warnings. Please inspect.")


if __name__ == "__main__":
    main()
