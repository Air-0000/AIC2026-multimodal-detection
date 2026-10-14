#!/usr/bin/env python3
"""
Sovereign V24 Dual-Version (Native vs Unified MidBoost) Test Set 2 Inference Engine
& Comprehensive Comparison Audit Against Official F5 Benchmark Package
=================================================================================
Executes:
  1. Fast multi-modal pipelined DataLoader for all 1,000 Phase 2 test images.
  2. Pass 1: Sovereign V24 Native (No Gating / 基础版).
  3. Pass 2: Sovereign V24 + Unified MidBoost (Adaptive Thermal Gating / 统合版).
  4. Packaging of official submissions (Compact 60-cap, thr45, thr50).
  5. Exhaustive cross-model consensus audit vs F5 submission package (58.105 peak).
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
    High-throughput multimodal dataset for Phase 2 Test Set (1,000 images).
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
                ir_arr = stripe_non_uniformity_correction_1d(ir_arr)
                ir_t = torch.from_numpy(ir_arr).float().unsqueeze(0) / 255.0
            except Exception:
                ir_t = T.functional.pil_to_tensor(ir_pil).float() / 255.0
        else:
            ir_t = T.functional.pil_to_tensor(ir_pil).float() / 255.0
        if ir_t.shape[1:] != (orig_h, orig_w):
            ir_t = T.functional.resize(ir_t, [orig_h, orig_w])

        # 3. LBG IR
        lbg_cands = glob.glob(os.path.join(self.cache_dir, "test_lbg_ir", f"{stem}.*"))
        lbg_pil = Image.open(lbg_cands[0]).convert("L")
        lbg_t = T.functional.pil_to_tensor(lbg_pil).float() / 255.0
        if lbg_t.shape[1:] != (orig_h, orig_w):
            lbg_t = T.functional.resize(lbg_t, [orig_h, orig_w])

        # 4. Depth P0
        dep_t = self._read_depth(stem)
        if dep_t.shape[1:] != (orig_h, orig_w):
            dep_t = T.functional.resize(dep_t, [orig_h, orig_w])

        # 5. Assemble 6-channel tensor and normalize
        tensor_6ch = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0)
        tensor_resized = T.functional.resize(tensor_6ch, [1152, 2048])
        tensor_norm = (tensor_resized - MEAN_6CH) / STD_6CH

        return tensor_norm, orig_w, orig_h, stem


def patch_backbone_unified_midboost(module):
    """
    Applies Sovereign V24 Unified MidBoost illumination-adaptive gating.
    Lifts twilight / night performance without daylight regression.
    """
    orig_forward = module.backbone.forward
    _im = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    _is = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    def dynamic_forward(x):
        rgb = x[:, :3]
        ir = x[:, 3:3 + module.backbone.ir_in_chans] if x.shape[1] >= 3 + module.backbone.ir_in_chans else None
        depth = x[:, 3 + module.backbone.ir_in_chans: 4 + module.backbone.ir_in_chans] if x.shape[1] >= 4 + module.backbone.ir_in_chans else None

        # Measure true physical illumination b
        rgb_raw = torch.clamp(rgb * _is + _im, 0.0, 1.0)
        b = rgb_raw.mean(dim=(1, 2, 3), keepdim=True)

        # Dynamic g_aux for early darkness (b in [0.03, 0.18]):
        g_aux = 1.0 + 0.40 * torch.clamp((0.18 - b) / 0.15, min=0.0, max=1.0)

        # Dynamic thermal takeover for deep darkness (b < 0.022):
        act = 0.60 * torch.clamp((0.022 - b) / 0.007, min=0.0, max=1.0)

        if act.max() > 0.0 and ir is not None:
            ir_raw = torch.clamp(ir[:, :1] * 0.5 + 0.5, 0.0, 1.0)
            rgb_fused = (1.0 - act) * rgb_raw + act * ir_raw
            rgb = (rgb_fused - _im) / _is

        if module.backbone.use_frontend_rfgm and module.backbone.frontend_rfgm is not None:
            rgb = module.backbone.frontend_rfgm(rgb)

        y = module.backbone.rgb_net.stem(rgb)
        outs = []
        if ir is not None:
            c2, c3 = module.backbone.ir_stem(ir)
            _stride_to_feat = {8: c2, 16: c3}

        p3_prompt = None
        s0_trunk_p3 = None
        for idx, stage in enumerate(module.backbone.rgb_net.stages):
            y = stage(y)
            if idx == 0:
                if module.backbone.use_trunk_bridge and module.backbone.s0_to_p3_trunk is not None:
                    s0_trunk_p3 = module.backbone.s0_to_p3_trunk(module.backbone.s0_norm(y))
                if module.backbone.use_shallow_edge_fuse and module.backbone.shallow_edge_fuse is not None and ir is not None and depth is not None:
                    y, p3_prompt = module.backbone.shallow_edge_fuse(y, rgb, ir, depth)
            if idx == 1:
                if p3_prompt is not None:
                    y = y + p3_prompt
                if s0_trunk_p3 is not None:
                    y = y + s0_trunk_p3

            if str(idx) in module.backbone.rgb_net.dcn_res:
                y = module.backbone.rgb_net.dcn_res[str(idx)](y)

            if ir is not None and idx in module.backbone.inject_stages:
                k = module.backbone.inject_stages.index(idx)
                f = _stride_to_feat.get(module.backbone.rgb_net._out_strides[idx])
                if f is None:
                    f = F.interpolate(c3, size=y.shape[-2:], mode="bilinear", align_corners=False)
                elif f.shape[-2:] != y.shape[-2:]:
                    f = F.interpolate(f, size=y.shape[-2:], mode="bilinear", align_corners=False)
                f = module.backbone.ir_proj[k](f)

                if module.backbone.use_dcda and module.backbone.ir_align is not None and depth is not None:
                    d_sub = F.interpolate(depth, size=y.shape[-2:], mode="nearest")
                    inv_d = torch.zeros_like(d_sub)
                    valid = (d_sub > 0.015) & (d_sub < 1.0)
                    inv_d[valid] = 1.0 / (d_sub[valid] * 20.0 + 0.1)
                    f = module.backbone.ir_align[k](y, f, inv_d)

                if module.backbone.use_dog and module.backbone.occlusion_gate is not None and depth is not None:
                    f = module.backbone.occlusion_gate[k](f, depth)

                if module.backbone.use_bilateral_pool and module.backbone.bilateral_pool is not None and depth is not None:
                    d_sub = F.interpolate(depth, size=y.shape[-2:], mode="nearest")
                    f = module.backbone.bilateral_pool[k](f, d_sub)

                if module.backbone.use_3d_lift and module.backbone.coord_lift is not None and depth is not None:
                    pos_3d = module.backbone.coord_lift[k](depth, y.shape)
                    y_in = y + pos_3d
                    f_in = f + pos_3d
                else:
                    y_in = y
                    f_in = f

                residual = module.backbone.ir_res[k](module.backbone.ir_fuse[k](y_in, f_in))
                y = y + g_aux * residual

            if idx in module.backbone.return_idx:
                outs.append(y)
        return outs

    module.backbone.forward = dynamic_forward
    return orig_forward


def run_inference_pass(model, postprocessor, dataloader, desc="Inference"):
    raw_detections = defaultdict(list)
    illum_records = {}

    _im = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    _is = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    t0 = time.time()
    with torch.no_grad():
        for batch_samples, batch_w, batch_h, batch_stems in tqdm(dataloader, desc=desc):
            batch_samples = batch_samples.cuda()

            # Record illumination
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
            # Sort: rare classes first, then confidence desc
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
    # cx, cy, bw, bh -> xyxy
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


def compare_two_packages(pkg_a, pkg_b, name_a="F5", name_b="V24"):
    all_stems = sorted(set(pkg_a["data"].keys()) | set(pkg_b["data"].keys()))
    
    # 1. Macro Stats
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
        f"{name_a}_p75_score": float(np.percentile(sc_a, 75)) if len(sc_a) > 0 else 0.0,
        f"{name_b}_p75_score": float(np.percentile(sc_b, 75)) if len(sc_b) > 0 else 0.0,
    }

    # 2. Per-Class Box Counts
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

    # 3. Spatial IoU Consensus Matching (IoU >= 0.50, same class)
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

    # Attach per-class recall
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
    parser.add_argument("--config", default="configs/deimv2/sovereign_v24_night_robust_tadl_2048.yml")
    parser.add_argument("--weights", default="outputs/sovereign_v24_night_robust_tadl_2048/best_stg2.pth")
    parser.add_argument("--cache-dir", default="/dev/shm/aic_cache")
    parser.add_argument("--f5-zip", default="/root/autodl-tmp/AIC/work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/AIC/outputs/submissions/submissions_v24")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" Sovereign V24 Dual-Version (Native vs Unified MidBoost) Inference Engine ".center(85, "="))
    print("=" * 85)
    print(f"Config:           {args.config}")
    print(f"Weights:          {args.weights}")
    print(f"Cache Dir:        {args.cache_dir}")
    print(f"F5 Benchmark:     {args.f5_zip}")
    print(f"Output Dir:       {out_dir}")
    print(f"Batch Size:       {args.batch_size}")
    print("=" * 85)

    # 1. Dataset & DataLoader
    dataset = TestSet2Dataset(args.cache_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stems = dataset.stems

    # 2. Build Model
    cfg = YAMLConfig(args.config)
    model = cfg.model.cuda().eval()
    print(f"\n[+] Loading Sovereign V24 weights from {args.weights}...")
    ck = torch.load(args.weights, map_location='cpu')
    w = ck['ema']['module'] if ('ema' in ck and 'module' in ck['ema']) else ck.get('model', ck)
    model.load_state_dict(w, strict=True)
    postprocessor = cfg.postprocessor.cuda().eval()

    # -------------------------------------------------------------
    # PASS 1: Sovereign V24 Native (No Gating / 基础版)
    # -------------------------------------------------------------
    print("\n" + "#" * 85)
    print(" Pass 1: Sovereign V24 Native (No Gating / 基础版) ".center(85, "#"))
    print("#" * 85)
    raw_native, illum_records, time_native = run_inference_pass(
        model, postprocessor, dataloader, desc="V24 Native Inference"
    )
    print(f"[✔] V24 Native inference complete in {time_native:.1f}s ({time_native/1000:.3f}s/img)")

    # -------------------------------------------------------------
    # PASS 2: Sovereign V24 + Unified MidBoost (Adaptive Thermal Gating)
    # -------------------------------------------------------------
    print("\n" + "#" * 85)
    print(" Pass 2: Sovereign V24 + Unified MidBoost (Adaptive Thermal Gating / 统合版) ".center(85, "#"))
    print("#" * 85)
    patch_backbone_unified_midboost(model)
    raw_midboost, _, time_midboost = run_inference_pass(
        model, postprocessor, dataloader, desc="V24 + Unified MidBoost Inference"
    )
    print(f"[✔] V24 + Unified MidBoost inference complete in {time_midboost:.1f}s ({time_midboost/1000:.3f}s/img)")

    # -------------------------------------------------------------
    # Packaging Submissions
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" Packaging Submission Files ".center(85, "="))
    print("=" * 85)

    compact_filter = lambda c, s: (s >= 0.45 if c in RARE_CLASSES else s >= 0.50)
    thr45_filter = lambda c, s: s >= 0.45
    thr50_filter = lambda c, s: s >= 0.50

    # Native packages
    pkg_v24_native_compact = package_submission(raw_native, stems, out_dir / "submission_v24_native_compact.zip", compact_filter, cap=60)
    pkg_v24_native_thr45   = package_submission(raw_native, stems, out_dir / "submission_v24_native_thr45.zip", thr45_filter, cap=100)
    pkg_v24_native_thr50   = package_submission(raw_native, stems, out_dir / "submission_v24_native_thr50.zip", thr50_filter, cap=100)

    # MidBoost packages
    pkg_v24_midboost_compact = package_submission(raw_midboost, stems, out_dir / "submission_v24_midboost_compact.zip", compact_filter, cap=60)
    pkg_v24_midboost_thr45   = package_submission(raw_midboost, stems, out_dir / "submission_v24_midboost_thr45.zip", thr45_filter, cap=100)
    pkg_v24_midboost_thr50   = package_submission(raw_midboost, stems, out_dir / "submission_v24_midboost_thr50.zip", thr50_filter, cap=100)

    print(f"[+] Packaged Native Compact:   {pkg_v24_native_compact['total_boxes']} boxes, {pkg_v24_native_compact['size_kb']:.1f} KB")
    print(f"[+] Packaged MidBoost Compact: {pkg_v24_midboost_compact['total_boxes']} boxes, {pkg_v24_midboost_compact['size_kb']:.1f} KB")

    # -------------------------------------------------------------
    # Parse F5 Submission & Run Comprehensive Comparison Audit
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" Cross-Model Comparative Audit vs Official F5 Submission Package ".center(85, "="))
    print("=" * 85)

    f5_parsed = parse_submission_zip(args.f5_zip)
    v24_native_parsed = parse_submission_zip(out_dir / "submission_v24_native_compact.zip")
    v24_midboost_parsed = parse_submission_zip(out_dir / "submission_v24_midboost_compact.zip")

    cmp_native_vs_f5 = compare_two_packages(f5_parsed, v24_native_parsed, name_a="F5", name_b="V24_Native")
    cmp_midboost_vs_f5 = compare_two_packages(f5_parsed, v24_midboost_parsed, name_a="F5", name_b="V24_MidBoost")
    cmp_midboost_vs_native = compare_two_packages(v24_native_parsed, v24_midboost_parsed, name_a="V24_Native", name_b="V24_MidBoost")

    # Illumination Census
    b_values = np.array(list(illum_records.values()))
    day_count = int(np.sum(b_values >= 0.18))
    twilight_count = int(np.sum((b_values >= 0.03) & (b_values < 0.18)))
    night_count = int(np.sum(b_values < 0.03))

    # Intra-V24 changed images
    diff_images = []
    for stem in stems:
        d_nat = v24_native_parsed["data"].get(stem, [])
        d_mid = v24_midboost_parsed["data"].get(stem, [])
        if len(d_nat) != len(d_mid) or d_nat != d_mid:
            diff_images.append({
                "stem": stem,
                "illum": illum_records[stem],
                "native_count": len(d_nat),
                "midboost_count": len(d_mid),
                "diff": len(d_mid) - len(d_nat),
            })

    # Assemble Audit Summary
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_set_size": len(stems),
        "illumination_census": {
            "mean_b": float(np.mean(b_values)),
            "min_b": float(np.min(b_values)),
            "max_b": float(np.max(b_values)),
            "daylight_images (b >= 0.18)": day_count,
            "twilight_images (0.03 <= b < 0.18)": twilight_count,
            "night_pitch_black (b < 0.03)": night_count,
        },
        "packages": {
            "v24_native_compact": pkg_v24_native_compact,
            "v24_midboost_compact": pkg_v24_midboost_compact,
            "f5_official": {
                "total_boxes": f5_parsed["total_boxes"],
                "empty_images": f5_parsed["empty_images"],
                "mean_score": float(np.mean(f5_parsed["scores"])),
            }
        },
        "compare_v24_native_vs_f5": cmp_native_vs_f5,
        "compare_v24_midboost_vs_f5": cmp_midboost_vs_f5,
        "compare_midboost_vs_native": cmp_midboost_vs_native,
        "midboost_impact_on_test2": {
            "total_images_affected": len(diff_images),
            "affected_images": diff_images[:30], # sample
        }
    }

    # Save JSON report
    report_json_path = out_dir / "v24_dual_vs_f5_test2_audit_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Saved full audit report JSON to {report_json_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
