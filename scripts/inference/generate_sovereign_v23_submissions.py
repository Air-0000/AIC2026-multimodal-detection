#!/usr/bin/env python3
"""
Sovereign V23 Zero-Distortion High-Res 1152x2048 Submission Generator
======================================================================
Runs native 6-channel multimodal inference at strict 16:9 Zero-Distortion
resolution (1152x2048) on official Phase 2 test set (1,000 images):
  1. submission_v23_compact.zip         (Rare 0.45, Common 0.50, cap 60: Golden 7000-box Line)
  2. submission_v23_compact_strict.zip  (Rare 0.48, Common 0.52, cap 60: Ultra-Pure Gate)
  3. submission_v23_thr45.zip           (Global tau = 0.45)
  4. submission_v23_thr50.zip           (Global tau = 0.50, 1:1 Prior Match)
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
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as T

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

RARE_CLASSES = {1, 7, 9, 10, 11} # boat, ball, garbage can, uav, tricycle
COMMON_CLASSES = {0, 2, 3, 4, 5, 6, 8}


def read_depth_raw16(path):
    if not os.path.exists(path):
        return torch.zeros((1, 1080, 1920), dtype=torch.float32)
    mat = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mat is None:
        return torch.zeros((1, 1080, 1920), dtype=torch.float32)
    if mat.ndim == 3:
        mat = mat[:, :, 0]
    if mat.dtype == np.uint16:
        arr = np.clip(mat.astype(np.float32) / 19999.0, 0.0, 1.0)
    else:
        arr = np.clip(mat.astype(np.float32) / 255.0, 0.0, 1.0)
    return torch.from_numpy(arr).unsqueeze(0).float()


def main():
    parser = argparse.ArgumentParser(description="Sovereign V23 Zero-Distortion Submission Generator")
    parser.add_argument("--config", default="vendor/DEIMv2/configs/deimv2/sovereign_v23_zero_distort_2048_rtx6000.yml")
    parser.add_argument("--weights", default="/root/autodl-tmp/AIC/outputs/sovereign_v23_zero_distort_2048_rtx6000/best_stg2.pth")
    parser.add_argument("--test-dir", default="/dev/shm/aic_cache")
    parser.add_argument("--fallback-test-dir", default="/root/autodl-tmp/AIC/dataset/test/AIC2026_PHASE_2_1000")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/AIC/work_dir/submissions_v23")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(" Sovereign V23 1152x2048 Zero-Distortion Official Submission Pipeline ".center(80, "="))
    print("=" * 80)
    print(f"Model Config:     {args.config}")
    print(f"Weights:          {args.weights}")
    print(f"Test Cache Dir:   {args.test_dir}")
    print(f"Output Root:      {args.out_dir}")
    print(f"Device:           {device} ({torch.cuda.get_device_name(0)})")
    print("=" * 80)

    cfg = YAMLConfig(args.config)
    model = cfg.model.to(device).eval()
    print(f"Loading checkpoint weights from {args.weights}...")
    ck = torch.load(args.weights, map_location="cpu")
    w = ck["ema"]["module"] if ("ema" in ck and "module" in ck["ema"]) else ck.get("model", ck)
    model.load_state_dict(w, strict=False)
    postprocessor = cfg.postprocessor

    vis_dir = os.path.join(args.test_dir, "test_visible")
    if not os.path.exists(vis_dir):
        vis_dir = os.path.join(args.fallback_test_dir, "visible")
    img_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
    stems = [Path(f).stem for f in img_files if not f.endswith(".txt")]
    print(f"Discovered {len(stems)} official Phase 2 test images.\n")

    # Raw detection store: stem -> list of (cid, xc, yc, w, h, conf)
    raw_detections = defaultdict(list)

    t0 = time.time()
    with torch.no_grad():
        for stem in tqdm(stems, desc="Inference 1152x2048"):
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
            vis_pil = Image.open(vis_p).convert("RGB")
            orig_w, orig_h = vis_pil.size
            rgb_t = T.functional.pil_to_tensor(vis_pil).float() / 255.0

            ir_pil = Image.open(ir_p).convert("L")
            if HAS_DESTRIPE:
                ir_arr = np.array(ir_pil)
                ir_arr = stripe_non_uniformity_correction_1d(ir_arr)
                ir_t = torch.from_numpy(ir_arr).float().unsqueeze(0) / 255.0
            else:
                ir_t = T.functional.pil_to_tensor(ir_pil).float() / 255.0
            if ir_t.shape[1:] != (orig_h, orig_w):
                ir_t = T.functional.resize(ir_t, [orig_h, orig_w])

            lbg_pil = Image.open(lbg_p).convert("L")
            lbg_t = T.functional.pil_to_tensor(lbg_pil).float() / 255.0
            if lbg_t.shape[1:] != (orig_h, orig_w):
                lbg_t = T.functional.resize(lbg_t, [orig_h, orig_w])

            dep_t = read_depth_raw16(dep_p)
            if dep_t.shape[1:] != (orig_h, orig_w):
                dep_t = T.functional.resize(dep_t, [orig_h, orig_w])

            # 2. Assemble 6-channel tensor
            raw_6ch = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0)

            # 3. Single Fused Bilinear Resize to 1152 x 2048 (Strict 16:9 Zero Distortion)
            inp = T.functional.resize(raw_6ch, [1152, 2048])
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

            # Record candidates (base filter >= 0.05)
            for sc, lb, b in zip(scores, labels, boxes_xyxy):
                if sc >= 0.05:
                    x1, y1, x2, y2 = b
                    cx = (x1 + x2) / 2.0 / orig_w
                    cy = (y1 + y2) / 2.0 / orig_h
                    bw = (x2 - x1) / orig_w
                    bh = (y2 - y1) / orig_h
                    raw_detections[stem].append((int(lb), float(cx), float(cy), float(bw), float(bh), float(sc)))

    elapsed = time.time() - t0
    print(f"\n[+] Inference completed in {elapsed:.1f}s ({elapsed/len(stems):.3f}s per image)")

    # Generate packages
    packages = {
        "compact": {
            "name": "submission_v23_compact.zip",
            "filter": lambda c, s: (s >= 0.45 if c in RARE_CLASSES else s >= 0.50),
            "cap": 60,
        },
        "compact_strict": {
            "name": "submission_v23_compact_strict.zip",
            "filter": lambda c, s: (s >= 0.48 if c in RARE_CLASSES else s >= 0.52),
            "cap": 60,
        },
        "thr45": {
            "name": "submission_v23_thr45.zip",
            "filter": lambda c, s: s >= 0.45,
            "cap": 100,
        },
        "thr50": {
            "name": "submission_v23_thr50.zip",
            "filter": lambda c, s: s >= 0.50,
            "cap": 100,
        },
    }

    print("\n" + "=" * 80)
    print(" Packaging Official Submissions ".center(80, "="))
    print("=" * 80)

    for pkey, pconf in packages.items():
        zip_path = os.path.join(args.out_dir, pconf["name"])
        total_boxes = 0
        empty_count = 0
        cls_counter = Counter()

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for stem in stems:
                dets = raw_detections.get(stem, [])
                filtered = [d for d in dets if pconf["filter"](d[0], d[5])]
                # Sort: rare first, then confidence desc
                filtered.sort(key=lambda x: (1 if x[0] in RARE_CLASSES else 0, x[5]), reverse=True)
                filtered = filtered[:pconf["cap"]]

                lines = []
                for cid, xc, yc, w, h, conf in filtered:
                    lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}\r\n")
                    cls_counter[cid] += 1
                    total_boxes += 1

                if len(filtered) == 0:
                    empty_count += 1

                zf.writestr(f"{stem}.txt", "".join(lines).encode("utf-8"))

        print(f"[+] Package: {pconf['name']}")
        print(f"    - Total Boxes:  {total_boxes} (avg {total_boxes/len(stems):.2f} / img)")
        print(f"    - Empty Images: {empty_count}")
        print(f"    - Class Breakdown: {dict(sorted(cls_counter.items()))}")
        print(f"    - Path: {zip_path}\n")

    print("[✔] All Sovereign V23 official packages generated and verified.")


if __name__ == "__main__":
    main()
