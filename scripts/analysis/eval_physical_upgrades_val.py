#!/usr/bin/env python3
"""
Physical Layer Upgrades Empirical Validation on 200-Image Validation Split
========================================================================
Compares:
  1. Sovereign LMM Baseline (Current inputs: Aligned IR + 31x31 LBG + P0 Depth)
  2. + Method IR-1 (Multi-Scale Morphological Top-Hat / MCA, replacing 31x31 LBG)
  3. + Method IR-3 (Anisotropic Thermal De-blooming on Aligned IR)
  4. + Method Depth-1 (Occlusion-Aware Bilateral Guided Filter on P0 Depth)
  5. Combined All-Star Physical Stack (IR-1 + IR-3 + Depth-1)
"""

import os
import sys
import glob
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


# =============================================================================
# Physical Layer Functions
# =============================================================================

def ir_method1_multiscale_mca(ir_u8, scales=(5, 11, 21, 41, 81)):
    """Method IR-1: 多尺度形态学分量分析 (Multi-Scale Morphological Top-Hat / MCA)"""
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
    """Method IR-3: 结构张量各向异性热晕逆收拢 (Anisotropic Thermal De-blooming)"""
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


def depth_method1_occlusion_aware_bilateral_guided(depth_m, vis_bgr, r=4, eps=1e-3, tau_jump=0.15):
    """Method Depth-1: 遮挡感知视差域双边引导滤波 (Occlusion-Aware Bilateral Guided)"""
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    disp = 1.0 / np.clip(depth_m, 0.5, 60.0)
    
    gx_d = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    jump_mag = np.sqrt(gx_d**2 + gy_d**2)
    
    adaptive_eps = eps * (1.0 + 80.0 * (jump_mag > tau_jump).astype(np.float32))
    
    mean_I = cv2.boxFilter(vg, -1, (2*r+1, 2*r+1))
    mean_p = cv2.boxFilter(disp, -1, (2*r+1, 2*r+1))
    mean_Ip = cv2.boxFilter(vg * disp, -1, (2*r+1, 2*r+1))
    var_I = cv2.boxFilter(vg * vg, -1, (2*r+1, 2*r+1)) - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + adaptive_eps)
    b = mean_p - a * mean_I
    
    disp_filtered = cv2.boxFilter(a, -1, (2*r+1, 2*r+1)) * vg + cv2.boxFilter(b, -1, (2*r+1, 2*r+1))
    depth_refined = 1.0 / np.clip(disp_filtered, 0.016, 2.0)
    return np.where(jump_mag > tau_jump, depth_m, depth_refined)


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_variant(model, post, stems, paths, coco_gt, name, use_mca=False, use_debloom=False, use_depth1=False):
    print(f"\n==================================================")
    print(f"Evaluating: {name}")
    print(f"Options: MCA={use_mca}, Debloom={use_debloom}, Depth1={use_depth1}")
    print(f"==================================================")

    detections = []
    t0 = time.time()

    with torch.no_grad():
        for idx, stem in enumerate(stems):
            vis_p = paths['vis'][stem]
            ir_p = paths['ir'][stem]
            lbg_p = paths['lbg'][stem]
            dep_p = paths['dep'][stem]

            vis_im = Image.open(vis_p).convert("RGB")
            W, H = vis_im.size
            rgb_t = TF.pil_to_tensor(vis_im).float() / 255.0

            ir_im = Image.open(ir_p).convert("L")
            ir_np = np.array(ir_im)

            # Optional IR-3 Deblooming
            if use_debloom:
                vis_bgr = cv2.imread(vis_p)
                ir_np = ir_method3_anisotropic_thermal_deblooming(ir_np, vis_bgr)

            ir_t = torch.from_numpy(ir_np.astype(np.float32) / 255.0).unsqueeze(0)
            if ir_t.shape[1:] != (H, W): ir_t = TF.resize(ir_t, [H, W])

            # Optional IR-1 MCA
            if use_mca:
                mca_np = ir_method1_multiscale_mca(ir_np)
                lbg_t = torch.from_numpy(mca_np.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                lbg_im = Image.open(lbg_p).convert("L")
                lbg_t = TF.pil_to_tensor(lbg_im).float() / 255.0
            if lbg_t.shape[1:] != (H, W): lbg_t = TF.resize(lbg_t, [H, W])

            # Optional Depth-1
            dep_im = Image.open(dep_p)
            dep_arr = np.array(dep_im, dtype=np.float32)
            scale = 19999.0 if getattr(dep_im, 'mode', '') in ('I;16', 'I') else (65535.0 if dep_arr.max() > 255.5 else 255.0)
            depth_m = np.clip(dep_arr / 1000.0, 0.0, 65.535) # in meters
            
            if use_depth1:
                vis_bgr = cv2.imread(vis_p) if 'vis_bgr' not in locals() else vis_bgr
                depth_m = depth_method1_occlusion_aware_bilateral_guided(depth_m, vis_bgr)
            
            d_norm = np.clip((depth_m * 1000.0) / scale, 0.0, 1.0)
            dep_t = torch.from_numpy(d_norm).unsqueeze(0)
            if dep_t.shape[1:] != (H, W): dep_t = TF.resize(dep_t, [H, W])

            # Assemble & Normalize
            tensor_6ch = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0)
            tensor_640 = TF.resize(tensor_6ch, [640, 640]).cuda()
            inputs = ((tensor_640 - MEAN_6CH) / STD_6CH).unsqueeze(0)

            orig_target_sizes = torch.tensor([[float(W), float(H)]], device="cuda")

            outputs = model(inputs)
            res = post(outputs, orig_target_sizes)[0]

            labels = res['labels'].cpu().numpy()
            boxes = res['boxes'].cpu().numpy() # [x1, y1, x2, y2]
            scores = res['scores'].cpu().numpy()

            img_id = paths['stem2id'][stem]
            for lbl, box, sc in zip(labels, boxes, scores):
                x1, y1, x2, y2 = box
                detections.append({
                    "image_id": img_id,
                    "category_id": int(lbl),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(sc)
                })

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx+1}/{len(stems)} images...", flush=True)

    dt = time.time() - t0
    print(f"Inference finished in {dt:.1f}s ({len(stems)/dt:.1f} FPS)")

    coco_dt = coco_gt.loadRes(detections)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats = ev.stats
    mAP = stats[0]
    AP50 = stats[1]
    AP75 = stats[2]
    APs = stats[3]
    APm = stats[4]
    APl = stats[5]

    print(f"\n[{name}] Final Metrics:")
    print(f"  mAP (0.50:0.95): {mAP:.4f}")
    print(f"  AP50           : {AP50:.4f}")
    print(f"  AP75           : {AP75:.4f}")
    print(f"  AP_small       : {APs:.4f}")
    print(f"  AP_medium      : {APm:.4f}")
    print(f"  AP_large       : {APl:.4f}")

    return {
        "name": name,
        "mAP": mAP,
        "AP50": AP50,
        "AP75": AP75,
        "APs": APs,
        "APm": APm,
        "APl": APl
    }


def main():
    root = "/root/autodl-tmp/AIC"
    cfg_p = os.path.join(root, "vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_lmm_rtx6000.yml")
    ckpt_p = os.path.join(root, "vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_lmm_rtx6000/best_stg1.pth")
    ann_p = os.path.join(root, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")

    print("Loading Ground Truth COCO Annotations...")
    coco_gt = COCO(ann_p)
    stem2id = {os.path.splitext(img['file_name'])[0]: img['id'] for img in coco_gt.dataset['images']}
    stems = sorted(list(stem2id.keys()))
    print(f"Found {len(stems)} validation images.")

    paths = {'vis': {}, 'ir': {}, 'lbg': {}, 'dep': {}, 'stem2id': stem2id}
    vis_dir = os.path.join(root, "work_dir/deimv2_dataset/valid/images")
    ir_dir = "/dev/shm/aic_cache/train_aligned_ir"
    lbg_dir = "/dev/shm/aic_cache/train_lbg_ir"
    dep_dir = "/dev/shm/aic_cache/depth_p0"

    for s in stems:
        paths['vis'][s] = os.path.join(vis_dir, f"{s}.jpg") if os.path.exists(os.path.join(vis_dir, f"{s}.jpg")) else os.path.join(vis_dir, f"{s}.png")
        paths['ir'][s] = os.path.join(ir_dir, f"{s}.jpg") if os.path.exists(os.path.join(ir_dir, f"{s}.jpg")) else os.path.join(ir_dir, f"{s}.png")
        paths['lbg'][s] = os.path.join(lbg_dir, f"{s}.jpg") if os.path.exists(os.path.join(lbg_dir, f"{s}.jpg")) else os.path.join(lbg_dir, f"{s}.png")
        paths['dep'][s] = os.path.join(dep_dir, f"{s}.png")

    print(f"Loading Model Config: {cfg_p}")
    cfg = YAMLConfig(cfg_p)
    model = cfg.model.cuda().eval()
    ckpt = torch.load(ckpt_p, map_location="cpu")
    weights = ckpt['ema']['module'] if ('ema' in ckpt and 'module' in ckpt['ema']) else ckpt.get('model', ckpt)
    model.load_state_dict(weights, strict=False)
    post = cfg.postprocessor.cuda()

    results = []

    # 1. Baseline
    r0 = evaluate_variant(model, post, stems, paths, coco_gt, "1. Sovereign LMM Baseline (Current 31x31 LBG)", use_mca=False, use_debloom=False, use_depth1=False)
    results.append(r0)

    # 2. Method IR-1 (MCA)
    r1 = evaluate_variant(model, post, stems, paths, coco_gt, "2. + Method IR-1 (Multi-Scale MCA Top-Hat)", use_mca=True, use_debloom=False, use_depth1=False)
    results.append(r1)

    # 3. Method IR-1 (MCA) + Method IR-3 (Thermal De-blooming)
    r2 = evaluate_variant(model, post, stems, paths, coco_gt, "3. + Method IR-3 (Anisotropic De-blooming) + IR-1", use_mca=True, use_debloom=True, use_depth1=False)
    results.append(r2)

    # 4. Method Depth-1 + IR-1 + IR-3
    r3 = evaluate_variant(model, post, stems, paths, coco_gt, "4. + Method Depth-1 (Bilateral Guided) + All Physical Upgrades", use_mca=True, use_debloom=True, use_depth1=True)
    results.append(r3)

    print("\n" + "="*80)
    print("PHYSICAL LAYER UPGRADES EMPIRICAL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Variant':<55} | {'mAP':<7} | {'AP50':<7} | {'AP75':<7} | {'AP_small':<7}")
    print("-" * 88)
    for r in results:
        print(f"{r['name']:<55} | {r['mAP']:.4f}  | {r['AP50']:.4f}  | {r['AP75']:.4f}  | {r['APs']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
