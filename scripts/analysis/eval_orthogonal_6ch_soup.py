#!/usr/bin/env python3
"""
Empirical Benchmark of All-Inclusive Orthogonal 6-Channel Split
===============================================================
Evaluates the 6-channel orthogonal split on the Reigning Champion
(Sovereign Pure-LMM 50-50 Model Soup, baseline 0.5842 mAP, 0.6530 AP75):

Channel Formulation:
  Ch 0~2 (RGB): Pure visible light (preserves macro structure: car, boat, bus)
  Ch 3 (IR): Thermal de-bloomed / PSF deblurred infrared core
  Ch 4 (Aux / 3D Fault + Phi_VT):
    - Baseline: Multi-scale MCA Top-Hat (IR-1)
    - Variant 1: MCA * Phi_VT (3D curvature water/ground suppression gate)
    - Variant 2: MCA * Phi_VT + 0.15 * 3D Geometric Fracture
    - Variant 3: MCA * Phi_VT + 0.30 * 3D Geometric Fracture
    - Variant 4: MCA * Phi_VT + 0.50 * 3D Geometric Fracture
    - Variant 5: Pure 3D Geometric Fracture * Phi_VT
    - Variant 6: Debloomed IR * Phi_VT (Ch 3) + MCA * Phi_VT (Ch 4)
  Ch 5 (Depth): Absolute P0 Metric Depth
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

aic_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in [aic_root, os.path.join(aic_root, 'vendor', 'DEIMv2'), os.path.join(aic_root, 'scripts', 'lib'),
          '/root/autodl-tmp/AIC', '/root/autodl-tmp/AIC/vendor/DEIMv2', '/root/autodl-tmp/AIC/scripts/lib']:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import faster_coco_eval
faster_coco_eval.init_as_pycocotools()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from engine.core import YAMLConfig

MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1).cuda()
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1).cuda()


def ir_method1_multiscale_mca(ir_u8, scales=(5, 11, 21, 41, 81)):
    out = np.zeros_like(ir_u8, dtype=np.float32)
    total_w = 0.0
    for k in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        wth = cv2.morphologyEx(ir_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        bth = cv2.morphologyEx(ir_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        w = 1.0 / np.sqrt(k)
        contrast_k = wth / (1.0 + 0.1 * bth)
        out += w * contrast_k
        total_w += w
    out = (out / total_w) * 1.6
    return np.clip(out, 0, 255).astype(np.uint8)


def ir_method3_anisotropic_thermal_deblooming(ir_u8, vis_bgr, n_iters=3, dt=0.20):
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx_v = cv2.Sobel(vg, cv2.CV_32F, 1, 0, ksize=3)
    gy_v = cv2.Sobel(vg, cv2.CV_32F, 0, 1, ksize=3)
    mag_v = np.sqrt(gx_v**2 + gy_v**2)
    barrier = np.exp(- (mag_v / 0.12)**2)

    u = ir_u8.astype(np.float32)
    for _ in range(n_iters):
        gx_u = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
        gy_u = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
        mag_u = np.sqrt(gx_u**2 + gy_u**2)
        lap_u = cv2.Laplacian(u, cv2.CV_32F, ksize=3)
        shock = - np.sign(lap_u) * mag_u * barrier
        u = u + dt * shock
    return np.clip(u, 0, 255).astype(np.uint8)


def compute_phi_vt_and_3d_edges(depth_arr):
    """
    Computes 3D volumetric curvature gate (Phi_VT) and Disparity 3D fracture edges.
    depth_arr is float32 depth in mm (or meters).
    """
    # Convert mm to meters: depth_m in [0.5, 50.0]
    depth_m = np.clip(depth_arr / 1000.0 if depth_arr.max() > 255.0 else depth_arr / 20.0, 0.5, 50.0)

    # 1. 3D surface normal & mean curvature divergence (Phi_VT)
    dz_dx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    div_n = np.abs(cv2.Sobel(-dz_dx / norm, cv2.CV_32F, 1, 0) + cv2.Sobel(-dz_dy / norm, cv2.CV_32F, 0, 1))

    # flat surfaces (water, road, flat ground) have div_n close to 0 -> flat_score -> 1.0
    flat_score = np.exp(- (div_n / 0.15)**2)
    # phi_vt gate: 0.25 on flat water reflections, 1.0 on 3D structures / objects
    phi_vt_gate = (1.0 - 0.75 * flat_score).astype(np.float32)

    # 2. Disparity-Domain 3D Geometric Fracture (Steep step edges)
    disp = 1.0 / np.clip(depth_m, 0.5, 50.0)
    gx_d = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    edge_3d = np.clip(np.sqrt(gx_d**2 + gy_d**2) * 4.0, 0.0, 1.0).astype(np.float32)

    return phi_vt_gate, edge_3d


def run_benchmark_series(model, post, val_data, coco_gt):
    stems = sorted(list(val_data.keys()))

    configs = [
        ("0. 基线: V17模型汤 + 组合A (Debloom-IR + MCA + P0深度)", "baseline", {}),
        ("1. 正交分流: Ch4 = MCA × Phi_VT门控 (水面伪热斑抑制)", "ch4_phi_vt", {"alpha_3d": 0.0, "gate_ch3": False}),
        ("2. 正交分流: Ch4 = MCA × Phi_VT + 0.15 × 3D几何断层", "ch4_phi_vt_3d", {"alpha_3d": 0.15, "gate_ch3": False}),
        ("3. 正交分流: Ch4 = MCA × Phi_VT + 0.30 × 3D几何断层 (全要素均衡)", "ch4_phi_vt_3d", {"alpha_3d": 0.30, "gate_ch3": False}),
        ("4. 正交分流: Ch4 = MCA × Phi_VT + 0.50 × 3D几何断层 (强化几何边界)", "ch4_phi_vt_3d", {"alpha_3d": 0.50, "gate_ch3": False}),
        ("5. 正交分流: Ch4 = 纯3D几何断层 × Phi_VT (彻底解耦热与几何)", "ch4_pure_3d", {"alpha_3d": 1.0, "gate_ch3": False}),
        ("6. 正交分流: Ch3与Ch4双门控 (IR×Phi_VT + MCA×Phi_VT+0.3*3D)", "ch3_ch4_dual_gate", {"alpha_3d": 0.30, "gate_ch3": True}),
    ]

    all_results = []

    for label, mode, params in configs:
        print(f"\n=======================================================", flush=True)
        print(f"Running: {label} ...", flush=True)
        print(f"=======================================================", flush=True)
        t0 = time.time()
        detections = []

        with torch.no_grad():
            for stem in stems:
                d = val_data[stem]
                H, W = d['H'], d['W']
                rgb_t = d['rgb_t']
                ir_deb = d['ir_deb']
                mca_np = d['mca_np']
                phi_vt_gate = d['phi_vt_gate']
                edge_3d = d['edge_3d']
                dep_t = d['dep_t']
                img_id = d['id']

                # Compute Ch 3
                if params.get("gate_ch3", False):
                    ir_ch3_np = np.clip(ir_deb.astype(np.float32) * phi_vt_gate, 0, 255).astype(np.uint8)
                else:
                    ir_ch3_np = ir_deb
                ir_t = torch.from_numpy(ir_ch3_np.astype(np.float32) / 255.0).unsqueeze(0)
                if ir_t.shape[1:] != (H, W): ir_t = TF.resize(ir_t, [H, W])

                # Compute Ch 4
                if mode == "baseline":
                    ch4_np = mca_np
                elif mode == "ch4_phi_vt":
                    ch4_np = np.clip(mca_np.astype(np.float32) * phi_vt_gate, 0, 255).astype(np.uint8)
                elif mode == "ch4_phi_vt_3d" or mode == "ch3_ch4_dual_gate":
                    alpha = params["alpha_3d"]
                    fused_ch4 = mca_np.astype(np.float32) * phi_vt_gate + alpha * 255.0 * edge_3d
                    ch4_np = np.clip(fused_ch4, 0, 255).astype(np.uint8)
                elif mode == "ch4_pure_3d":
                    ch4_np = np.clip(255.0 * edge_3d * phi_vt_gate, 0, 255).astype(np.uint8)
                else:
                    ch4_np = mca_np

                lbg_t = torch.from_numpy(ch4_np.astype(np.float32) / 255.0).unsqueeze(0)
                if lbg_t.shape[1:] != (H, W): lbg_t = TF.resize(lbg_t, [H, W])

                # Assemble 6-channel tensor
                tensor_6ch = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0)
                tensor_640 = TF.resize(tensor_6ch, [640, 640]).cuda()
                inputs = ((tensor_640 - MEAN_6CH) / STD_6CH).unsqueeze(0)

                orig_target_sizes = torch.tensor([[float(W), float(H)]], device="cuda")

                outputs = model(inputs)
                res = post(outputs, orig_target_sizes)[0]

                labels = res['labels'].cpu().numpy()
                boxes = res['boxes'].cpu().numpy()
                scores = res['scores'].cpu().numpy()

                for lbl, box, sc in zip(labels, boxes, scores):
                    x1, y1, x2, y2 = box
                    detections.append({
                        "image_id": img_id,
                        "category_id": int(lbl),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(sc)
                    })

        dt = time.time() - t0
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

        s = ev.stats
        res = {
            "name": label,
            "mAP": s[0],
            "AP50": s[1],
            "AP75": s[2],
            "APs": s[3],
            "APm": s[4],
            "APl": s[5],
            "AR100": s[8],
            "ARs": s[9],
            "time": dt
        }
        all_results.append(res)
        print(f"[{label}] -> mAP: {s[0]:.4f} | AP50: {s[1]:.4f} | AP75: {s[2]:.4f} | APs: {s[3]:.4f} | APl: {s[5]:.4f} ({dt:.1f}s)", flush=True)

    print("\n" + "="*104)
    print("ALL-INCLUSIVE ORTHOGONAL 6-CHANNEL SPLIT EMPIRICAL BENCHMARK SUMMARY")
    print("="*104)
    print(f"{'Configuration':<54} | {'mAP':<7} | {'AP50':<7} | {'AP75':<7} | {'APs':<7} | {'APl':<7} | {'ARs':<7}")
    print("-" * 104)
    for r in all_results:
        print(f"{r['name']:<54} | {r['mAP']:.4f}  | {r['AP50']:.4f}  | {r['AP75']:.4f}  | {r['APs']:.4f}  | {r['APl']:.4f}  | {r['ARs']:.4f}")
    print("="*104)
    return all_results


def main():
    root = "/root/autodl-tmp/AIC"
    ann_p = os.path.join(root, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
    coco_gt = COCO(ann_p)
    stem2id = {os.path.splitext(img['file_name'])[0]: img['id'] for img in coco_gt.dataset['images']}
    stems = sorted(list(stem2id.keys()))

    vis_dir = os.path.join(root, "work_dir/deimv2_dataset/valid/images")
    ir_dir = "/dev/shm/aic_cache/train_aligned_ir"
    dep_dir = "/dev/shm/aic_cache/depth_p0"

    print(f"Pre-caching {len(stems)} validation images into RAM ...", flush=True)
    t_start = time.time()
    val_data = {}

    for s in stems:
        vis_p = os.path.join(vis_dir, f"{s}.jpg") if os.path.exists(os.path.join(vis_dir, f"{s}.jpg")) else os.path.join(vis_dir, f"{s}.png")
        ir_p = os.path.join(ir_dir, f"{s}.jpg") if os.path.exists(os.path.join(ir_dir, f"{s}.jpg")) else os.path.join(ir_dir, f"{s}.png")
        dep_p = os.path.join(dep_dir, f"{s}.png")

        vis_im = Image.open(vis_p).convert("RGB")
        W, H = vis_im.size
        rgb_t = TF.pil_to_tensor(vis_im).float() / 255.0
        vis_bgr = cv2.imread(vis_p)

        ir_im = Image.open(ir_p).convert("L")
        ir_np = np.array(ir_im)
        ir_deb = ir_method3_anisotropic_thermal_deblooming(ir_np, vis_bgr)
        mca_np = ir_method1_multiscale_mca(ir_deb)

        dep_im = Image.open(dep_p)
        dep_arr = np.array(dep_im, dtype=np.float32)
        scale = 19999.0 if getattr(dep_im, 'mode', '') in ('I;16', 'I') else (65535.0 if dep_arr.max() > 255.5 else 255.0)
        d_norm = np.clip(dep_arr / scale, 0.0, 1.0)
        dep_t = torch.from_numpy(d_norm).unsqueeze(0)
        if dep_t.shape[1:] != (H, W): dep_t = TF.resize(dep_t, [H, W])

        phi_vt_gate, edge_3d = compute_phi_vt_and_3d_edges(dep_arr)

        val_data[s] = {
            "id": stem2id[s],
            "W": W,
            "H": H,
            "rgb_t": rgb_t,
            "ir_deb": ir_deb,
            "mca_np": mca_np,
            "phi_vt_gate": phi_vt_gate,
            "edge_3d": edge_3d,
            "dep_t": dep_t
        }

    print(f"Pre-caching complete in {time.time() - t_start:.2f}s.", flush=True)

    # Load Model Soup
    soup_p = os.path.join(root, "vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_pure_lmm_soup/best_stg1.pth")
    lmm_cfg = os.path.join(root, "vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_lmm_rtx6000.yml")

    print(f"Loading Model Soup from {soup_p} ...", flush=True)
    cfg = YAMLConfig(lmm_cfg)
    model = cfg.model.cuda().eval()
    ck = torch.load(soup_p, map_location="cpu")
    w = ck["ema"]["module"] if "ema" in ck and "module" in ck["ema"] else ck.get("model", ck)
    model.load_state_dict(w, strict=False)
    post = cfg.postprocessor.cuda()

    run_benchmark_series(model, post, val_data, coco_gt)


if __name__ == '__main__':
    main()
