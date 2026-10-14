#!/usr/bin/env python3
"""
Sovereign V20 Official Multi-Threshold Submission Generator & Diagnostic Pipeline
==================================================================================
Runs inference on Phase 2 test set (1,000 images) using Sovereign V20 Pure LMM Soup
(trained strictly on the official old annotations without quality relabeling).

Produces 3 compliant submission packages:
  1. submission_v20_thr45.zip   (tau = 0.45, direct comparison to V23 thr45)
  2. submission_v20_thr50.zip   (tau = 0.50, direct comparison to V23 thr50 / F5 thr50)
  3. submission_v20_compact.zip (Class-adaptive compact gate matching ~7,100 prior)
"""

import os
import sys
import glob
import time
import zipfile
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.v2 as T

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
aic_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
vendor_deim = os.path.join(aic_root, 'vendor', 'DEIMv2')
scripts_lib = os.path.join(aic_root, 'scripts', 'lib')

for p in [aic_root, vendor_deim, scripts_lib]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from engine.core import YAMLConfig

MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

# Compact profile (sensitive on rare classes, balanced on common classes)
COMPACT_THRESHOLDS = {
    0: 0.50,   # person
    1: 0.45,   # boat (rare)
    2: 0.50,   # animal
    3: 0.50,   # seat
    4: 0.50,   # sign
    5: 0.50,   # bicycle
    6: 0.50,   # car
    7: 0.45,   # ball (rare)
    8: 0.50,   # light
    9: 0.45,   # garbage can (rare)
    10: 0.45,  # uav (rare)
    11: 0.45,  # tricycle (rare)
}


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


def filter_and_format(candidates, thresh_spec, nms_iou=0.65, max_boxes=100):
    """
    Filters candidates by threshold, applies class-wise NMS, formats as official submission line.
    """
    # 1. Filter by threshold
    filtered = []
    for sc, lb, (cx, cy, bw, bh) in candidates:
        if isinstance(thresh_spec, dict):
            th = thresh_spec.get(lb, 0.50)
        else:
            th = thresh_spec
        if sc >= th:
            filtered.append((sc, lb, (cx, cy, bw, bh)))

    # Sort descending by score
    filtered.sort(key=lambda x: x[0], reverse=True)

    # 2. Class-wise NMS
    kept = []
    for sc, lb, (cx, cy, bw, bh) in filtered:
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        b_xyxy = (x1, y1, x2, y2)

        dup = False
        for _, klb, (kcx, kcy, kbw, kbh) in kept:
            if klb == lb:
                kx1 = kcx - kbw / 2.0
                ky1 = kcy - kbh / 2.0
                kx2 = kcx + kbw / 2.0
                ky2 = kcy + kbw / 2.0
                if box_iou_xyxy(b_xyxy, (kx1, ky1, kx2, ky2)) >= nms_iou:
                    dup = True
                    break
        if not dup:
            kept.append((sc, lb, (cx, cy, bw, bh)))
            if len(kept) >= max_boxes:
                break

    # 3. Format lines: <class_id> <cx> <cy> <w> <h> <confidence>
    lines = []
    for sc, lb, (cx, cy, bw, bh) in kept:
        cx_c = float(np.clip(cx, 0.0, 1.0))
        cy_c = float(np.clip(cy, 0.0, 1.0))
        bw_c = float(np.clip(bw, 0.0, 1.0))
        bh_c = float(np.clip(bh, 0.0, 1.0))
        sc_c = float(np.clip(sc, 0.0, 1.0))
        lines.append(f"{lb} {cx_c:.6f} {cy_c:.6f} {bw_c:.6f} {bh_c:.6f} {sc_c:.4f}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate V20 Submission Packages")
    parser.add_argument("--config", default="vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_lmm_rtx6000.yml")
    parser.add_argument("--weights", default="weights/sovereign_checkpoints/sovereign_v20_pure_lmm_soup_585.pth")
    parser.add_argument("--test-dir", default="dataset/test/AIC2026_PHASE_2_1000")
    parser.add_argument("--out-dir", default="work_dir/submissions/submissions_v20")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(" Sovereign V20 Official Multi-Threshold Submission Generator ".center(80, "="))
    print("=" * 80)
    print(f"Config:      {args.config}")
    print(f"Weights:     {args.weights}")
    print(f"Test Root:   {args.test_dir}")
    print(f"Output Root: {args.out_dir}")
    print(f"Device:      {device} ({torch.cuda.get_device_name(0)})")
    print("=" * 80)

    # 1. Load Model
    cfg = YAMLConfig(args.config)
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device)

    ck = torch.load(args.weights, map_location="cpu")
    sd = ck.get("ema", {}).get("module", ck.get("model", ck))
    model.load_state_dict(sd, strict=False)
    print("Successfully loaded Sovereign V20 Model Weights.\n")

    # 2. Test modality folders
    vis_dir = os.path.join(args.test_dir, "visible")
    ir_dir = os.path.join(aic_root, "work_dir/test_aligned_ir")
    lbg_dir = os.path.join(aic_root, "work_dir/test_lbg_ir")
    dep_dir = os.path.join(args.test_dir, "depth_p0")

    img_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
    stems = [Path(f).stem for f in img_files if not f.endswith(".txt")]
    print(f"Discovered {len(stems)} official test images.\n")

    # 3. Setup profiles
    profiles = {
        "thr45": {"desc": "tau = 0.450 (Matches V23 thr45)", "spec": 0.45},
        "thr50": {"desc": "tau = 0.500 (Matches V23 thr50 / F5 thr50)", "spec": 0.50},
        "compact": {"desc": "Compact Class-Adaptive Gate (cap 60)", "spec": COMPACT_THRESHOLDS},
    }

    for key in profiles:
        p_dir = os.path.join(args.out_dir, f"submit_{key}", "predictions")
        os.makedirs(p_dir, exist_ok=True)
        profiles[key]["p_dir"] = p_dir
        profiles[key]["counters"] = Counter()
        profiles[key]["total_dets"] = 0
        profiles[key]["empty_count"] = 0

    t0 = time.time()
    with torch.no_grad():
        for stem in tqdm(stems, desc="Inference V20 (640x640)"):
            # Visible
            vis_cand = glob.glob(os.path.join(vis_dir, f"{stem}.*"))
            vis_p = vis_cand[0]
            vis_im = Image.open(vis_p).convert("RGB")
            orig_w, orig_h = vis_im.size
            rgb_t = T.functional.pil_to_tensor(vis_im).float() / 255.0

            # Aligned IR
            ir_cand = glob.glob(os.path.join(ir_dir, f"{stem}.*"))
            if not ir_cand:
                ir_cand = glob.glob(os.path.join(args.test_dir, "infrared", f"{stem}.*"))
            ir_p = ir_cand[0]
            ir_im = Image.open(ir_p).convert("L")
            ir_t = T.functional.pil_to_tensor(ir_im).float() / 255.0
            if ir_t.shape[1:] != (orig_h, orig_w):
                ir_t = T.functional.resize(ir_t, [orig_h, orig_w])

            # LBG IR
            lbg_cand = glob.glob(os.path.join(lbg_dir, f"{stem}.*"))
            if not lbg_cand:
                lbg_cand = glob.glob(os.path.join(args.test_dir, "infrared", f"{stem}.*"))
            lbg_p = lbg_cand[0]
            lbg_im = Image.open(lbg_p).convert("L")
            lbg_t = T.functional.pil_to_tensor(lbg_im).float() / 255.0
            if lbg_t.shape[1:] != (orig_h, orig_w):
                lbg_t = T.functional.resize(lbg_t, [orig_h, orig_w])

            # Depth P0
            dep_cand = glob.glob(os.path.join(dep_dir, f"{stem}.*"))
            if not dep_cand:
                dep_cand = glob.glob(os.path.join(args.test_dir, "depth", f"{stem}.*"))
            dep_p = dep_cand[0]
            dep_im = Image.open(dep_p)
            dep_arr = np.array(dep_im).astype(np.float32)
            if dep_arr.ndim == 3:
                dep_arr = dep_arr.mean(axis=2)
            scale = 19999.0 if getattr(dep_im, "mode", "") in ("I;16", "I") else 255.0
            dep_t = torch.from_numpy(np.clip(dep_arr / scale, 0.0, 1.0))[None]
            if dep_t.shape[1:] != (orig_h, orig_w):
                dep_t = T.functional.resize(dep_t, [orig_h, orig_w])

            # 6-channel early fusion
            raw_6ch = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0)
            inp_640 = T.functional.resize(raw_6ch, [640, 640]).to(device)
            inp_norm = ((inp_640 - MEAN_6CH) / STD_6CH).unsqueeze(0)

            orig_sizes = torch.tensor([[float(orig_w), float(orig_h)]], device=device)
            with torch.cuda.amp.autocast():
                out = model(inp_norm)
                res = postprocessor(out, orig_sizes)[0]

            boxes_xyxy = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            labels = res["labels"].cpu().numpy()

            # Raw candidates (score >= 0.10 base)
            candidates = []
            for sc, lb, b in zip(scores, labels, boxes_xyxy):
                if sc >= 0.10:
                    x1, y1, x2, y2 = b
                    cx = (x1 + x2) / 2.0 / orig_w
                    cy = (y1 + y2) / 2.0 / orig_h
                    bw = (x2 - x1) / orig_w
                    bh = (y2 - y1) / orig_h
                    candidates.append((float(sc), int(lb), (cx, cy, bw, bh)))

            # Generate outputs for all 3 profiles
            for key, prof in profiles.items():
                max_b = 60 if key == "compact" else 100
                lines = filter_and_format(candidates, prof["spec"], nms_iou=0.65, max_boxes=max_b)
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

    # Pack 3 zip files and verify
    print("=" * 80)
    print(" Packaging and Verifying Sovereign V20 Official Submissions ".center(80, "="))
    print("=" * 80)

    check_script = os.path.join(aic_root, "scripts/eval/check_submission.py")

    for key, prof in profiles.items():
        zip_path = os.path.join(args.out_dir, f"submission_v20_{key}.zip")
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
                print(f"   [CHECK PASS] submission_v20_{key}.zip 100% 格式合规！")
            else:
                print(f"   [WARN] check reported warnings for {key}")

        # Automatically import into Master Workstation
        try:
            from scripts.viz.import_submission_to_workstation import auto_import_submission
            auto_import_submission(
                zip_path=zip_path,
                model_id=f"v20_{key}",
                name=f"Sovereign V20 ({prof['desc']})",
                score="58.45% mAP",
                desc=f"Sovereign V20 官方老标多模态模型: {prof['desc']}",
                rebuild=False
            )
        except Exception as e:
            print(f"   [Workstation] Auto-import skipped: {e}")

    # Final rebuild and launch workstation
    try:
        build_script = os.path.join(aic_root, "scripts/viz/build_master_unified_workstation.py")
        if os.path.exists(build_script):
            print("\n[Workstation] Rebuilding Master Workstation with new V20 submissions...")
            os.system(f"python3 {build_script}")
            from scripts.viz.import_submission_to_workstation import ensure_server_alive
            ensure_server_alive()
            print("[Workstation] Workstation ready at http://localhost:8088/\n")
    except Exception as e:
        print(f"[Workstation] Rebuild notice: {e}")

    print("\n" + "=" * 80)
    print(" SOVEREIGN V20 SUBMISSION GENERATION COMPLETED! ".center(80, "="))
    print("=" * 80)


if __name__ == "__main__":
    main()
