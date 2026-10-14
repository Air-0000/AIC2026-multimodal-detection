#!/usr/bin/env python3
"""
Sovereign V25 Test Set 2 Inference Engine & 3-Way Comparative Audit
===================================================================
Models Evaluated:
  1. Sovereign V25 (Native Model-Repair: SDQS + NDSL + Native MidBoost)
  2. Sovereign V24 (Night-Robust TADL / best_stg2.pth)
  3. F5 Official Benchmark Package (58.105 AP peak / 7,097 boxes)

Executes:
  - Multimodal DataLoader for 1,000 Phase 2 Test Set images.
  - V25 Inference with native MidBoost & SDQS decoder query selection.
  - Packaging official submission zips (Compact cap=60, thr45, thr50).
  - 3-Way Consensus & Discrepancy Audit (V25 vs F5, V25 vs V24).
  - Detailed per-class detection count and consensus recall reporting.
"""

import os
import sys
import glob
import time
import json
import zipfile
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

_root = '/root/autodl-tmp/AIC'
if not os.path.exists(_root):
    _root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(_root, 'vendor/DEIMv2'))
sys.path.insert(0, os.path.join(_root, 'scripts/lib'))
os.chdir(os.path.join(_root, 'vendor/DEIMv2'))

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

RARE_CLASSES = {1, 7, 9, 10, 11} # boat, ball, garbage can, uav, tricycle
COMMON_CLASSES = {0, 2, 3, 4, 5, 6, 8}

MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)


class TestSet2Dataset(Dataset):
    """
    Multimodal dataset for Phase 2 Test Set (1,000 images).
    Loads RGB, Aligned IR (destriped), LBG IR, and Depth P0 (16-bit).
    """
    def __init__(self, cache_dir="/dev/shm/aic_cache"):
        self.cache_dir = cache_dir
        vis_files = sorted(glob.glob(os.path.join(cache_dir, "test_visible", "*.*")))
        self.stems = [os.path.splitext(os.path.basename(f))[0] for f in vis_files if not f.endswith(".txt")]
        assert len(self.stems) == 1000, f"Expected 1,000 test images, got {len(self.stems)}"

    def __len__(self):
        return len(self.stems)

    def _read_depth(self, stem):
        for ext in ('.png', '.jpg'):
            p = os.path.join(self.cache_dir, "test_depth_p0", f"{stem}{ext}")
            if os.path.exists(p):
                im = Image.open(p)
                arr = np.array(im).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr.mean(axis=2)
                scale = 19999.0 if getattr(im, 'mode', '') in ('I;16', 'I') else 255.0
                arr = (arr / scale).clip(0.0, 1.0)
                return torch.from_numpy(arr).unsqueeze(0)
        raise FileNotFoundError(f"Depth not found for {stem}")

    def __getitem__(self, idx):
        stem = self.stems[idx]
        # 1. Visible RGB
        vis_cands = glob.glob(os.path.join(self.cache_dir, "test_visible", f"{stem}.*"))
        vis_pil = Image.open(vis_cands[0]).convert("RGB")
        orig_w, orig_h = vis_pil.size
        rgb_t = T.functional.pil_to_tensor(vis_pil).float() / 255.0

        # 2. Aligned IR
        ir_cands = glob.glob(os.path.join(self.cache_dir, "test_aligned_ir", f"{stem}.*"))
        ir_pil = Image.open(ir_cands[0]).convert("L")
        if HAS_DESTRIPE:
            try:
                ir_arr = np.array(ir_pil)
                ir_clean = stripe_non_uniformity_correction_1d(ir_arr, max_stripe_width=5, blend=0.85)
                ir_pil = Image.fromarray(ir_clean)
            except Exception:
                pass
        ir_t = T.functional.pil_to_tensor(ir_pil).float() / 255.0

        # 3. LBG IR
        lbg_cands = glob.glob(os.path.join(self.cache_dir, "test_ir_lbg", f"{stem}.*"))
        if lbg_cands:
            lbg_pil = Image.open(lbg_cands[0]).convert("L")
            lbg_t = T.functional.pil_to_tensor(lbg_pil).float() / 255.0
        else:
            lbg_t = ir_t.clone()

        # 4. Depth P0
        depth_t = self._read_depth(stem)

        # Fuse to 6 channels: [RGB (3), IR (1), LBG (1), Depth (1)]
        sample = torch.cat([rgb_t, ir_t, lbg_t, depth_t], dim=0)

        # Resize to native inference resolution (1152, 2048)
        sample = F.interpolate(sample.unsqueeze(0), size=(1152, 2048), mode="bilinear", align_corners=False).squeeze(0)
        sample = (sample - MEAN_6CH) / STD_6CH

        return sample, orig_w, orig_h, stem


def run_inference(model, postprocessor, dataloader, desc="V25 Inference"):
    raw_detections = defaultdict(list)
    illum_records = {}

    _im = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    _is = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    t0 = time.time()
    with torch.no_grad():
        for batch_samples, batch_w, batch_h, batch_stems in tqdm(dataloader, desc=desc):
            batch_samples = batch_samples.cuda()

            # Record illumination b
            rgb_raw = torch.clamp(batch_samples[:, :3] * _is + _im, 0.0, 1.0)
            b_vals = rgb_raw.mean(dim=(1, 2, 3)).cpu().numpy()
            for stem, b_val in zip(batch_stems, b_vals):
                illum_records[stem] = float(b_val)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(batch_samples)

            orig_sizes = torch.tensor([[float(w), float(h)] for w, h in zip(batch_w, batch_h)], device="cuda")
            results = postprocessor(outputs, orig_sizes, batch_samples)

            for res, orig_w, orig_h, stem in zip(results, batch_w, batch_h, batch_stems):
                w_f = float(orig_w)
                h_f = float(orig_h)
                boxes = res["boxes"].float().cpu().numpy()
                scores = res["scores"].float().cpu().numpy()
                labels = res["labels"].cpu().numpy()

                for sc, lb, b in zip(scores, labels, boxes):
                    if sc >= 0.05:
                        x1, y1, x2, y2 = b
                        cx = (x1 + x2) / 2.0 / w_f
                        cy = (y1 + y2) / 2.0 / h_f
                        bw = (x2 - x1) / w_f
                        bh = (y2 - y1) / h_f
                        raw_detections[stem].append((int(lb), float(cx), float(cy), float(bw), float(bh), float(sc)))

    elapsed = time.time() - t0
    return raw_detections, illum_records, elapsed


def package_submission(raw_detections, stems, out_zip_path, filter_fn, cap=60):
    total_boxes = 0
    empty_images = 0
    cls_counts = Counter()

    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for stem in stems:
            dets = raw_detections.get(stem, [])
            filtered = [d for d in dets if filter_fn(d[0], d[5])]
            filtered.sort(key=lambda x: (1 if x[0] in RARE_CLASSES else 0, x[5]), reverse=True)
            filtered = filtered[:cap]

            lines = []
            for cid, cx, cy, bw, bh, sc in filtered:
                cx_c = float(np.clip(cx, 0.0, 1.0))
                cy_c = float(np.clip(cy, 0.0, 1.0))
                bw_c = float(np.clip(bw, 0.0001, 1.0))
                bh_c = float(np.clip(bh, 0.0001, 1.0))
                sc_c = float(np.clip(sc, 0.0, 1.0))
                lines.append(f"{cid} {cx_c:.6f} {cy_c:.6f} {bw_c:.6f} {bh_c:.6f} {sc_c:.4f}\r\n")
                cls_counts[cid] += 1
                total_boxes += 1

            if not filtered:
                empty_images += 1

            zf.writestr(f"{stem}.txt", "".join(lines).encode("utf-8"))

    return {
        "zip_path": str(out_zip_path),
        "total_boxes": total_boxes,
        "empty_images": empty_images,
        "cls_counts": dict(sorted(cls_counts.items())),
        "avg_boxes": total_boxes / len(stems),
        "size_kb": os.path.getsize(out_zip_path) / 1024,
    }


def parse_submission_zip(zip_path):
    data = {}
    total_boxes = 0
    scores = []
    cls_counts = Counter()
    img_counts = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = sorted(zf.namelist())
        for name in members:
            if not name.endswith('.txt'):
                continue
            stem = Path(name).stem
            content = zf.read(name).decode('utf-8').strip()
            boxes = []
            if content:
                for line in content.replace('\r\n', '\n').split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        cid = int(parts[0])
                        cx, cy, bw, bh, sc = map(float, parts[1:6])
                        boxes.append((cid, cx, cy, bw, bh, sc))
                        cls_counts[cid] += 1
                        scores.append(sc)
            data[stem] = boxes
            img_counts.append(len(boxes))
            total_boxes += len(boxes)

    return {
        "data": data,
        "total_boxes": total_boxes,
        "scores": np.array(scores, dtype=np.float32) if scores else np.array([], dtype=np.float32),
        "cls_counts": cls_counts,
        "img_counts": np.array(img_counts, dtype=np.int32),
        "empty_images": int(sum(1 for c in img_counts if c == 0)),
    }


def calc_box_iou(box1, box2):
    b1_x1 = box1[1] - box1[3] / 2.0
    b1_y1 = box1[2] - box1[4] / 2.0
    b1_x2 = box1[1] + box1[3] / 2.0
    b1_y2 = box1[2] + box1[4] / 2.0

    b2_x1 = box2[1] - box2[3] / 2.0
    b2_y1 = box2[2] - box2[4] / 2.0
    b2_x2 = box2[1] + box2[3] / 2.0
    b2_y2 = box2[2] + box2[4] / 2.0

    ix1 = max(b1_x1, b2_x1)
    iy1 = max(b1_y1, b2_y1)
    ix2 = min(b1_x2, b2_x2)
    iy2 = min(b1_y2, b2_y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area1 = max(0.0, b1_x2 - b1_x1) * max(0.0, b1_y2 - b1_y1)
    area2 = max(0.0, b2_x2 - b2_x1) * max(0.0, b2_y2 - b2_y1)
    union = area1 + area2 - inter
    return inter / union if union > 1e-9 else 0.0


def compare_two_packages(pkg_a, pkg_b, name_a="F5", name_b="V25"):
    all_stems = sorted(set(pkg_a["data"].keys()) | set(pkg_b["data"].keys()))
    
    sc_a = pkg_a["scores"]
    sc_b = pkg_b["scores"]

    macro_stats = {
        f"{name_a}_total": pkg_a["total_boxes"],
        f"{name_b}_total": pkg_b["total_boxes"],
        "diff": pkg_b["total_boxes"] - pkg_a["total_boxes"],
        "ratio_pct": pkg_b["total_boxes"] / pkg_a["total_boxes"] * 100 if pkg_a["total_boxes"] > 0 else 0,
        f"{name_a}_empty": pkg_a["empty_images"],
        f"{name_b}_empty": pkg_b["empty_images"],
        f"{name_a}_mean_score": float(np.mean(sc_a)) if len(sc_a) > 0 else 0.0,
        f"{name_b}_mean_score": float(np.mean(sc_b)) if len(sc_b) > 0 else 0.0,
        f"{name_a}_median_score": float(np.median(sc_a)) if len(sc_a) > 0 else 0.0,
        f"{name_b}_median_score": float(np.median(sc_b)) if len(sc_b) > 0 else 0.0,
    }

    class_comparison = {}
    for cid in range(12):
        cname = CLASS_NAMES[cid]
        cnt_a = pkg_a["cls_counts"][cid]
        cnt_b = pkg_b["cls_counts"][cid]
        class_comparison[cname] = {
            "cid": cid,
            f"{name_a}_count": cnt_a,
            f"{name_b}_count": cnt_b,
            "diff": cnt_b - cnt_a,
            "ratio_pct": cnt_b / cnt_a * 100 if cnt_a > 0 else 0.0,
        }

    total_matched = 0
    matched_ious = []
    class_matched = Counter()

    for stem in all_stems:
        boxes_a = pkg_a["data"].get(stem, [])
        boxes_b = pkg_b["data"].get(stem, [])

        matched_b_indices = set()
        for b_a in boxes_a:
            cid_a = b_a[0]
            best_iou = 0.0
            best_idx = -1
            for idx_b, b_b in enumerate(boxes_b):
                if idx_b in matched_b_indices or b_b[0] != cid_a:
                    continue
                iou = calc_box_iou(b_a, b_b)
                if iou >= 0.50 and iou > best_iou:
                    best_iou = iou
                    best_idx = idx_b

            if best_idx >= 0:
                matched_b_indices.add(best_idx)
                total_matched += 1
                matched_ious.append(best_iou)
                class_matched[cid_a] += 1

    consensus_stats = {
        "matched_consensus_boxes": total_matched,
        f"{name_a}_recall_pct": total_matched / pkg_a["total_boxes"] * 100 if pkg_a["total_boxes"] > 0 else 0.0,
        f"{name_b}_agreement_pct": total_matched / pkg_b["total_boxes"] * 100 if pkg_b["total_boxes"] > 0 else 0.0,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "median_iou": float(np.median(matched_ious)) if matched_ious else 0.0,
    }

    for cid in range(12):
        cname = CLASS_NAMES[cid]
        cnt_a = pkg_a["cls_counts"][cid]
        m = class_matched[cid]
        class_comparison[cname]["matched"] = m
        class_comparison[cname][f"{name_a}_recall_pct"] = m / cnt_a * 100 if cnt_a > 0 else 0.0

    return {
        "macro": macro_stats,
        "consensus": consensus_stats,
        "per_class": class_comparison,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deimv2/sovereign_v25_model_repair_midboost.yml")
    parser.add_argument("--weights", default="outputs/sovereign_v25_model_repair_midboost_2048/best_stg2.pth")
    parser.add_argument("--cache-dir", default="/dev/shm/aic_cache")
    parser.add_argument("--f5-zip", default="/root/autodl-tmp/AIC/work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip")
    parser.add_argument("--v24-zip", default="/root/autodl-tmp/AIC/outputs/submissions/submissions_v24/submission_v24_midboost_compact.zip")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/AIC/outputs/submissions/submissions_v25")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" Sovereign V25 Inference & 3-Way Comparative Audit Engine ".center(85, "="))
    print("=" * 85)
    print(f"Config:           {args.config}")
    print(f"Weights:          {args.weights}")
    print(f"Cache Dir:        {args.cache_dir}")
    print(f"F5 Benchmark:     {args.f5_zip}")
    print(f"V24 Benchmark:    {args.v24_zip}")
    print(f"Output Dir:       {out_dir}")
    print("=" * 85)

    # 1. Fallback to checkpoint0007.pth or last.pth if best_stg2.pth not yet created
    weights_path = Path(args.weights)
    if not weights_path.exists():
        candidates = [
            weights_path.parent / "best_stg2.pth",
            weights_path.parent / "checkpoint0007.pth",
            weights_path.parent / "last.pth",
            weights_path.parent / "best_stg1.pth",
        ]
        for cand in candidates:
            if cand.exists():
                weights_path = cand
                print(f"[!] Using candidate checkpoint: {weights_path}")
                break

    # 2. Dataset & DataLoader
    dataset = TestSet2Dataset(args.cache_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stems = dataset.stems

    # 3. Model Build & Load
    cfg = YAMLConfig(args.config)
    model = cfg.model.cuda().eval()
    print(f"\n[+] Loading Sovereign V25 weights from {weights_path}...")
    ck = torch.load(str(weights_path), map_location='cpu')
    w = ck['ema']['module'] if ('ema' in ck and 'module' in ck['ema']) else ck.get('model', ck)
    msg = model.load_state_dict(w, strict=False)
    print(f"[+] Loaded weights successfully ({len(w)} tensors).")
    print(f"    Missing keys: {msg.missing_keys}")
    print(f"    Unexpected keys: {msg.unexpected_keys}")
    postprocessor = cfg.postprocessor.cuda().eval()

    # 4. Run Inference
    print("\n" + "#" * 85)
    print(" Sovereign V25 Full-Resolution Native Inference (1152x2048) ".center(85, "#"))
    print("#" * 85)
    raw_v25, illum_records, elapsed_v25 = run_inference(model, postprocessor, dataloader, desc="V25 Inference")
    print(f"[✔] V25 inference complete in {elapsed_v25:.1f}s ({elapsed_v25/1000:.3f}s/img)")

    # 5. Packaging Submissions
    print("\n" + "=" * 85)
    print(" Packaging Sovereign V25 Submission Files ".center(85, "="))
    print("=" * 85)
    compact_filter = lambda c, s: (s >= 0.45 if c in RARE_CLASSES else s >= 0.50)
    thr45_filter = lambda c, s: s >= 0.45
    thr50_filter = lambda c, s: s >= 0.50

    pkg_v25_compact = package_submission(raw_v25, stems, out_dir / "submission_v25_compact.zip", compact_filter, cap=60)
    pkg_v25_thr45   = package_submission(raw_v25, stems, out_dir / "submission_v25_thr45.zip", thr45_filter, cap=100)
    pkg_v25_thr50   = package_submission(raw_v25, stems, out_dir / "submission_v25_thr50.zip", thr50_filter, cap=100)

    print(f"[+] Packaged V25 Compact: {pkg_v25_compact['total_boxes']} boxes, {pkg_v25_compact['empty_images']} empty images, {pkg_v25_compact['size_kb']:.1f} KB")
    print(f"[+] Packaged V25 Thr45:   {pkg_v25_thr45['total_boxes']} boxes, {pkg_v25_thr45['empty_images']} empty images, {pkg_v25_thr45['size_kb']:.1f} KB")
    print(f"[+] Packaged V25 Thr50:   {pkg_v25_thr50['total_boxes']} boxes, {pkg_v25_thr50['empty_images']} empty images, {pkg_v25_thr50['size_kb']:.1f} KB")

    # 6. Parse Reference Packages
    print("\n" + "=" * 85)
    print(" Parsing Reference Benchmarks (F5 Official & V24 MidBoost) ".center(85, "="))
    print("=" * 85)
    f5_parsed = parse_submission_zip(args.f5_zip) if os.path.exists(args.f5_zip) else None
    v24_parsed = parse_submission_zip(args.v24_zip) if os.path.exists(args.v24_zip) else None
    v25_parsed = parse_submission_zip(out_dir / "submission_v25_compact.zip")

    # 7. Audit Comparisons
    cmp_v25_vs_f5 = compare_two_packages(f5_parsed, v25_parsed, name_a="F5", name_b="V25") if f5_parsed else {}
    cmp_v25_vs_v24 = compare_two_packages(v24_parsed, v25_parsed, name_a="V24", name_b="V25") if v24_parsed else {}
    cmp_v24_vs_f5 = compare_two_packages(f5_parsed, v24_parsed, name_a="F5", name_b="V24") if (f5_parsed and v24_parsed) else {}

    # 8. Print Executive Table
    print("\n" + "=" * 85)
    print(" 3-Way Comparative Box Count & Discrepancy Matrix ".center(85, "="))
    print("=" * 85)
    print(f"{'Category':<15} | {'F5 Peak':<10} | {'V24 Prior':<10} | {'V25 Repaired':<12} | {'V25 vs F5':<12} | {'V25 vs V24':<12}")
    print("-" * 85)
    for cid in range(12):
        cname = CLASS_NAMES[cid]
        cnt_f5 = f5_parsed["cls_counts"][cid] if f5_parsed else 0
        cnt_v24 = v24_parsed["cls_counts"][cid] if v24_parsed else 0
        cnt_v25 = v25_parsed["cls_counts"][cid]
        diff_f5 = cnt_v25 - cnt_f5
        diff_v24 = cnt_v25 - cnt_v24
        print(f"{cname:<15} | {cnt_f5:<10} | {cnt_v24:<10} | {cnt_v25:<12} | {diff_f5:+<12} | {diff_v24:+<12}")
    print("-" * 85)
    total_f5 = f5_parsed["total_boxes"] if f5_parsed else 0
    total_v24 = v24_parsed["total_boxes"] if v24_parsed else 0
    total_v25 = v25_parsed["total_boxes"]
    print(f"{'TOTAL BOXES':<15} | {total_f5:<10} | {total_v24:<10} | {total_v25:<12} | {total_v25 - total_f5:+<12} | {total_v25 - total_v24:+<12}")
    empty_f5 = f5_parsed["empty_images"] if f5_parsed else 0
    empty_v24 = v24_parsed["empty_images"] if v24_parsed else 0
    empty_v25 = v25_parsed["empty_images"]
    print(f"{'EMPTY IMGS':<15} | {empty_f5:<10} | {empty_v24:<10} | {empty_v25:<12} | {empty_v25 - empty_f5:+<12} | {empty_v25 - empty_v24:+<12}")
    print("=" * 85)

    # 9. Save Audit Report
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": str(weights_path),
        "total_test_images": len(stems),
        "packages": {
            "v25_compact": pkg_v25_compact,
            "v25_thr45": pkg_v25_thr45,
            "v25_thr50": pkg_v25_thr50,
            "v24_compact": {
                "total_boxes": total_v24,
                "empty_images": empty_v24,
            },
            "f5_official": {
                "total_boxes": total_f5,
                "empty_images": empty_f5,
            }
        },
        "compare_v25_vs_f5": cmp_v25_vs_f5,
        "compare_v25_vs_v24": cmp_v25_vs_v24,
        "compare_v24_vs_f5": cmp_v24_vs_f5,
    }

    report_path = out_dir / "v25_vs_v24_vs_f5_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Full audit report saved to {report_path}")


if __name__ == "__main__":
    main()
