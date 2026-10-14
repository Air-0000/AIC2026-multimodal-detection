#!/usr/bin/env python3
"""
Exact Empirical Benchmark of Combination C (Model Soup + Combination A)
======================================================================
Evaluates:
  1. Sovereign LMM (Single Model 58.19%) + Combination A
  2. Sovereign Pure (Single Model 58.21%) + Combination A
  3. Sovereign Soup (Pure 50% + LMM 50%) + Combination A
  4. Sovereign V14 Soup (58.30% Checkpoint) + Combination A
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


def evaluate_model(model, post, stems, paths, coco_gt, name):
    print(f"\nEvaluating: {name} ...", flush=True)
    detections = []
    t0 = time.time()

    with torch.no_grad():
        for idx, stem in enumerate(stems):
            vis_p = paths['vis'][stem]
            ir_p = paths['ir'][stem]
            dep_p = paths['dep'][stem]

            vis_im = Image.open(vis_p).convert("RGB")
            W, H = vis_im.size
            rgb_t = TF.pil_to_tensor(vis_im).float() / 255.0

            ir_im = Image.open(ir_p).convert("L")
            ir_np = np.array(ir_im)

            # Combination A: IR-3 Debloom on Aligned IR
            vis_bgr = cv2.imread(vis_p)
            ir_deb = ir_method3_anisotropic_thermal_deblooming(ir_np, vis_bgr)
            ir_t = torch.from_numpy(ir_deb.astype(np.float32) / 255.0).unsqueeze(0)
            if ir_t.shape[1:] != (H, W): ir_t = TF.resize(ir_t, [H, W])

            # Combination A: IR-1 MCA for LBG IR
            mca_np = ir_method1_multiscale_mca(ir_deb)
            lbg_t = torch.from_numpy(mca_np.astype(np.float32) / 255.0).unsqueeze(0)
            if lbg_t.shape[1:] != (H, W): lbg_t = TF.resize(lbg_t, [H, W])

            # Combination A: P0 Depth
            dep_im = Image.open(dep_p)
            dep_arr = np.array(dep_im, dtype=np.float32)
            scale = 19999.0 if getattr(dep_im, 'mode', '') in ('I;16', 'I') else (65535.0 if dep_arr.max() > 255.5 else 255.0)
            d_norm = np.clip(dep_arr / scale, 0.0, 1.0)
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
            boxes = res['boxes'].cpu().numpy()
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

    dt = time.time() - t0
    coco_dt = coco_gt.loadRes(detections)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats = ev.stats
    res_dict = {
        "name": name,
        "mAP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "APs": stats[3],
        "APm": stats[4],
        "APl": stats[5]
    }
    print(f"[{name}] -> mAP: {stats[0]:.4f}, AP50: {stats[1]:.4f}, AP75: {stats[2]:.4f}, APs: {stats[3]:.4f} ({dt:.1f}s)")
    return res_dict


def main():
    root = "/root/autodl-tmp/AIC"
    ann_p = os.path.join(root, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
    coco_gt = COCO(ann_p)
    stem2id = {os.path.splitext(img['file_name'])[0]: img['id'] for img in coco_gt.dataset['images']}
    stems = sorted(list(stem2id.keys()))

    paths = {'vis': {}, 'ir': {}, 'dep': {}, 'stem2id': stem2id}
    vis_dir = os.path.join(root, "work_dir/deimv2_dataset/valid/images")
    ir_dir = "/dev/shm/aic_cache/train_aligned_ir"
    dep_dir = "/dev/shm/aic_cache/depth_p0"

    for s in stems:
        paths['vis'][s] = os.path.join(vis_dir, f"{s}.jpg") if os.path.exists(os.path.join(vis_dir, f"{s}.jpg")) else os.path.join(vis_dir, f"{s}.png")
        paths['ir'][s] = os.path.join(ir_dir, f"{s}.jpg") if os.path.exists(os.path.join(ir_dir, f"{s}.jpg")) else os.path.join(ir_dir, f"{s}.png")
        paths['dep'][s] = os.path.join(dep_dir, f"{s}.png")

    lmm_cfg = os.path.join(root, "vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_lmm_rtx6000.yml")
    v14_cfg = os.path.join(root, "vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_v14_rtx6000.yml")

    pure_p = os.path.join(root, "vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_pure_rtx6000/best_stg1.pth")
    lmm_p = os.path.join(root, "vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_lmm_rtx6000/best_stg1.pth")
    v14_soup_p = os.path.join(root, "vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_v14_soup/best_stg1.pth")

    results = []

    # 1. Sovereign LMM (Single Model) + Combination A
    cfg = YAMLConfig(lmm_cfg)
    model = cfg.model.cuda().eval()
    ck_l = torch.load(lmm_p, map_location="cpu")
    w_l = ck_l["ema"]["module"] if "ema" in ck_l and "module" in ck_l["ema"] else ck_l.get("model", ck_l)
    model.load_state_dict(w_l, strict=False)
    post = cfg.postprocessor.cuda()
    r1 = evaluate_model(model, post, stems, paths, coco_gt, "1. Sovereign LMM (Single Model) + 组合A")
    results.append(r1)

    # 2. Sovereign Pure (Single Model) + Combination A
    ck_p = torch.load(pure_p, map_location="cpu")
    w_p = ck_p["ema"]["module"] if "ema" in ck_p and "module" in ck_p["ema"] else ck_p.get("model", ck_p)
    model.load_state_dict(w_p, strict=False)
    r2 = evaluate_model(model, post, stems, paths, coco_gt, "2. Sovereign Pure (Single Model) + 组合A")
    results.append(r2)

    # 3. Sovereign Pure (50%) + LMM (50%) Model Soup + Combination A
    print("\nCreating Soup: 50% Pure + 50% LMM ...")
    w_soup_50 = {}
    for k in w_l.keys():
        if k in w_p:
            w_soup_50[k] = 0.5 * w_p[k] + 0.5 * w_l[k]
        else:
            w_soup_50[k] = w_l[k]
    model.load_state_dict(w_soup_50, strict=False)
    r3 = evaluate_model(model, post, stems, paths, coco_gt, "3. 组合C: Pure(50%) + LMM(50%) 模型汤 + 组合A")
    results.append(r3)

    # 4. Sovereign V14 Soup Checkpoint + Combination A
    if os.path.exists(v14_soup_p):
        print(f"\nLoading V14 Soup Checkpoint ({v14_soup_p}) ...")
        cfg_v14 = YAMLConfig(v14_cfg)
        model_v14 = cfg_v14.model.cuda().eval()
        ck_v14 = torch.load(v14_soup_p, map_location="cpu")
        w_v14 = ck_v14["ema"]["module"] if "ema" in ck_v14 and "module" in ck_v14["ema"] else ck_v14.get("model", ck_v14)
        model_v14.load_state_dict(w_v14, strict=False)
        post_v14 = cfg_v14.postprocessor.cuda()
        r4 = evaluate_model(model_v14, post_v14, stems, paths, coco_gt, "4. 组合C (历史最高检查点 V14-Soup) + 组合A")
        results.append(r4)

    print("\n" + "="*86)
    print("EXACT EMPIRICAL BENCHMARK OF COMBINATION C (MODEL SOUP + COMBINATION A)")
    print("="*86)
    print(f"{'Configuration':<52} | {'mAP':<7} | {'AP50':<7} | {'AP75':<7} | {'AP_small':<7}")
    print("-" * 86)
    for r in results:
        print(f"{r['name']:<52} | {r['mAP']:.4f}  | {r['AP50']:.4f}  | {r['AP75']:.4f}  | {r['APs']:.4f}")
    print("="*86)


if __name__ == '__main__':
    main()
