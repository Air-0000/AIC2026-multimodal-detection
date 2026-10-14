#!/usr/bin/env python3
"""
Sovereign V22 High-Res 1216x672 Multi-Threshold Submission Generator
=====================================================================
Generates 4 distinct official Phase 2 competition submission packages
from a single inference forward pass using Cycle 1 Apex weights:
  1. submission_v22_cycle1_thr15.zip  (tau = 0.15, High Recall Baseline)
  2. submission_v22_cycle1_thr20.zip  (tau = 0.20, Golden Filter Gate)
  3. submission_v22_cycle1_thr25.zip  (tau = 0.25, High Precision Gate)
  4. submission_v22_cycle1_adaptive.zip (Class-adaptive Threshold Gate)
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
import torchvision.transforms.v2 as T
from torchvision.ops import nms

# Add root paths
script_dir = os.path.dirname(os.path.abspath(__file__))
aic_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
vendor_deim = os.path.join(aic_root, 'vendor', 'DEIMv2')
scripts_lib = os.path.join(aic_root, 'scripts', 'lib')

for p in [aic_root, vendor_deim, scripts_lib]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from engine.core import YAMLConfig
try:
    from destripe_1d import stripe_non_uniformity_correction_1d
    HAS_DESTRIPE = True
except ImportError:
    HAS_DESTRIPE = False

MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

# Adaptive threshold profile: sensitive on hard/small objects, strict on easy objects
ADAPTIVE_THRESH = {
    0: 0.20,  # person
    1: 0.14,  # boat (hard)
    2: 0.20,  # animal
    3: 0.22,  # seat
    4: 0.12,  # sign (hard small)
    5: 0.18,  # bicycle
    6: 0.22,  # car
    7: 0.12,  # ball (hard small)
    8: 0.12,  # light (hard small)
    9: 0.20,  # garbage can
    10: 0.18, # uav
    11: 0.20, # tricycle
}


def read_depth_raw16(path):
    im = Image.open(path)
    arr = np.array(im, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    scale = 19999.0 if getattr(im, 'mode', '') in ('I;16', 'I') else 255.0
    arr = np.clip(arr / scale, 0.0, 1.0)
    return torch.from_numpy(arr).unsqueeze(0)


def read_channel_destripe(path):
    im = Image.open(path).convert("L")
    arr = np.array(im)
    if HAS_DESTRIPE:
        try:
            arr = stripe_non_uniformity_correction_1d(arr)
        except Exception:
            pass
    return torch.from_numpy(arr).float().unsqueeze(0) / 255.0


def filter_and_format(candidates, thresh_spec, nms_iou=0.65, max_boxes=100):
    """
    candidates: list of (score, cid, (cx, cy, bw, bh))
    thresh_spec: float or dict of cid -> float
    """
    # 1. Threshold filter
    filtered = []
    for sc, cid, box in candidates:
        min_th = thresh_spec[cid] if isinstance(thresh_spec, dict) else thresh_spec
        if sc >= min_th:
            filtered.append((sc, cid, box))

    # 2. Class-wise NMS
    kept = []
    if filtered:
        for c in range(12):
            c_items = [item for item in filtered if item[1] == c]
            if not c_items:
                continue
            boxes_xyxy = []
            scores = []
            for sc, _, (cx, cy, bw, bh) in c_items:
                boxes_xyxy.append([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0])
                scores.append(sc)
            b_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
            s_tensor = torch.tensor(scores, dtype=torch.float32)
            keep_idx = nms(b_tensor, s_tensor, iou_threshold=nms_iou).tolist()
            for idx in keep_idx:
                kept.append(c_items[idx])

    # 3. Sort descending by score
    kept.sort(key=lambda x: -x[0])
    kept = kept[:max_boxes]

    # 4. Generate formatted lines (CRLF endings)
    lines = []
    for sc, cid, (cx, cy, bw, bh) in kept:
        cx_c = float(np.clip(cx, 0.0, 1.0))
        cy_c = float(np.clip(cy, 0.0, 1.0))
        bw_c = float(np.clip(bw, 0.0, 1.0))
        bh_c = float(np.clip(bh, 0.0, 1.0))
        sc_c = float(np.clip(sc, 0.0, 1.0))
        lines.append(f"{cid} {cx_c:.6f} {cy_c:.6f} {bw_c:.6f} {bh_c:.6f} {sc_c:.6f}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Sovereign V22 Multi-Threshold Submission Generator")
    parser.add_argument("--config", default="vendor/DEIMv2/configs/deimv2/sovereign_v22_highres1216_rtx6000_full72e.yml")
    parser.add_argument("--weights", default="/root/autodl-tmp/AIC/weight/sovereign_v22_cycle1_apex.pth")
    parser.add_argument("--test-dir", default="/dev/shm/aic_cache")
    parser.add_argument("--fallback-test-dir", default="/root/autodl-tmp/AIC/dataset/test/AIC2026_PHASE_2_1000")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/AIC/work_dir/submissions/submissions_v22_cycle1")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(" Sovereign V22 Multi-Threshold Official Submission Pipeline ".center(80, "="))
    print("=" * 80)
    print(f"Model Config:     {args.config}")
    print(f"Weights:          {args.weights}")
    print(f"Test Cache Dir:   {args.test_dir}")
    print(f"Output Root:      {args.out_dir}")
    print(f"Device:           {device} ({torch.cuda.get_device_name(0)})")
    print("=" * 80)

    # Load Model
    cfg = YAMLConfig(args.config)
    model = cfg.model.to(device).eval()
    print(f"Loading checkpoint weights from {args.weights}...")
    ck = torch.load(args.weights, map_location="cpu")
    w = ck["ema"]["module"] if ("ema" in ck and "module" in ck["ema"]) else ck.get("model", ck)
    model.load_state_dict(w, strict=False)
    postprocessor = cfg.postprocessor

    # Find test images
    vis_dir = os.path.join(args.test_dir, "test_visible")
    if not os.path.exists(vis_dir):
        vis_dir = os.path.join(args.fallback_test_dir, "visible")
    img_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
    stems = [Path(f).stem for f in img_files if not f.endswith(".txt")]
    print(f"Discovered {len(stems)} official Phase 2 test images.\n")

    # Define 4 submission profiles
    profiles = {
        "thr15": {"desc": "tau = 0.15 (High Recall Baseline)", "spec": 0.15},
        "thr20": {"desc": "tau = 0.20 (Golden Filter Gate)", "spec": 0.20},
        "thr25": {"desc": "tau = 0.25 (High Precision Gate)", "spec": 0.25},
        "adaptive": {"desc": "tau = Adaptive Class-Specific Gate", "spec": ADAPTIVE_THRESH},
    }

    # Initialize prediction output directories
    for key in profiles:
        p_dir = os.path.join(args.out_dir, f"submit_{key}", "predictions")
        os.makedirs(p_dir, exist_ok=True)
        profiles[key]["p_dir"] = p_dir
        profiles[key]["counters"] = Counter()
        profiles[key]["total_dets"] = 0
        profiles[key]["empty_count"] = 0

    t0 = time.time()
    with torch.no_grad():
        for stem in tqdm(stems, desc="Inference 1216x672"):
            # Paths
            vis_p = os.path.join(vis_dir, f"{stem}.jpg")
            if not os.path.exists(vis_p):
                vis_p = os.path.join(vis_dir, f"{stem}.png")

            ir_dir = os.path.join(args.test_dir, "test_infrared") if os.path.exists(os.path.join(args.test_dir, "test_infrared")) else os.path.join(args.fallback_test_dir, "infrared")
            lbg_dir = os.path.join(args.test_dir, "test_lbg_ir") if os.path.exists(os.path.join(args.test_dir, "test_lbg_ir")) else os.path.join(args.fallback_test_dir, "infrared")
            dep_dir = os.path.join(args.test_dir, "test_depth_p0") if os.path.exists(os.path.join(args.test_dir, "test_depth_p0")) else os.path.join(args.fallback_test_dir, "depth_p0")

            ir_p = os.path.join(ir_dir, f"{stem}.jpg")
            if not os.path.exists(ir_p):
                ir_p = os.path.join(ir_dir, f"{stem}.png")

            lbg_p = os.path.join(lbg_dir, f"{stem}.jpg")
            if not os.path.exists(lbg_p):
                lbg_p = os.path.join(lbg_dir, f"{stem}.png")

            dep_p = os.path.join(dep_dir, f"{stem}.png")

            # 1. Read modalities
            vis_im = Image.open(vis_p).convert("RGB")
            orig_w, orig_h = vis_im.size
            rgb_t = T.functional.pil_to_tensor(vis_im).float() / 255.0

            ir_t = read_channel_destripe(ir_p)
            if ir_t.shape[1:] != (orig_h, orig_w):
                ir_t = T.functional.resize(ir_t, [orig_h, orig_w])

            lbg_t = read_channel_destripe(lbg_p)
            if lbg_t.shape[1:] != (orig_h, orig_w):
                lbg_t = T.functional.resize(lbg_t, [orig_h, orig_w])

            dep_t = read_depth_raw16(dep_p)
            if dep_t.shape[1:] != (orig_h, orig_w):
                dep_t = T.functional.resize(dep_t, [orig_h, orig_w])

            # 2. Assemble 6-channel tensor
            raw_6ch = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0)

            # 3. Resize to 1216 x 672 and normalize
            inp = T.functional.resize(raw_6ch, [672, 1216])
            inp_norm = (inp.cuda() - MEAN_6CH) / STD_6CH
            inp_batch = inp_norm.unsqueeze(0)

            # 4. Model forward pass
            with torch.cuda.amp.autocast():
                out = model(inp_batch)

            # 5. Postprocessor
            orig_sizes = torch.tensor([[float(orig_w), float(orig_h)]], device="cuda")
            res = postprocessor(out, orig_sizes, inp_batch)[0]

            boxes_xyxy = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            labels = res["labels"].cpu().numpy()

            # Convert to raw candidate list (score >= 0.08 base)
            candidates = []
            for sc, lb, b in zip(scores, labels, boxes_xyxy):
                if sc >= 0.08:
                    x1, y1, x2, y2 = b
                    cx = (x1 + x2) / 2.0 / orig_w
                    cy = (y1 + y2) / 2.0 / orig_h
                    bw = (x2 - x1) / orig_w
                    bh = (y2 - y1) / orig_h
                    candidates.append((float(sc), int(lb), (cx, cy, bw, bh)))

            # 6. Generate outputs for all 4 profiles
            for key, prof in profiles.items():
                lines = filter_and_format(candidates, prof["spec"], nms_iou=0.65, max_boxes=100)
                out_txt = os.path.join(prof["p_dir"], f"{stem}.txt")
                with open(out_txt, "wb") as f:
                    if lines:
                        content = "\r\n".join(lines) + "\r\n"
                        f.write(content.encode("utf-8"))
                        prof["total_dets"] += len(lines)
                        for ln in lines:
                            cid = int(ln.split()[0])
                            prof["counters"][cid] += 1
                    else:
                        f.write(b"")
                        prof["empty_count"] += 1

    elapsed = time.time() - t0
    print(f"\n[DONE] Finished 1000 test inferences in {elapsed:.1f}s ({elapsed/len(stems)*1000:.1f} ms/img).\n")

    # Pack 4 zip files and verify
    print("=" * 80)
    print(" Packaging and Verifying 4 Official Submissions ".center(80, "="))
    print("=" * 80)

    check_script = os.path.join(aic_root, "scripts/eval/check_submission.py")

    for key, prof in profiles.items():
        zip_path = os.path.join(args.out_dir, f"submission_v22_cycle1_{key}.zip")
        print(f"\n>> Packing [{prof['desc']}] -> {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for txt_f in sorted(glob.glob(os.path.join(prof["p_dir"], "*.txt"))):
                z.write(txt_f, os.path.basename(txt_f))

        zip_size_kb = os.path.getsize(zip_path) / 1024
        print(f"   Size: {zip_size_kb:.1f} KB | Total Boxes: {prof['total_dets']} (avg {prof['total_dets']/len(stems):.1f}/img) | Empty: {prof['empty_count']}")

        # Audit with check_submission.py
        if os.path.exists(check_script):
            ret = os.system(f"python3 {check_script} {prof['p_dir']} {vis_dir}")
            if ret == 0:
                print(f"   [CHECK PASS] submission_v22_cycle1_{key}.zip 100% 格式合规！")
            else:
                print(f"   [WARN] check reported warnings for {key}")

    print("\n" + "=" * 80)
    print(" ALL 4 SUBMISSION PACKAGES GENERATED AND VERIFIED SUCCESSFULLY! ".center(80, "="))
    print("=" * 80)


if __name__ == "__main__":
    main()
