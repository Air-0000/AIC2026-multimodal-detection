"""
Experiment 4: Model-Level Internal CAA-SAHI IoU & Query Calibration Benchmark
Evaluates the effect of Internal CAA-SAHI NMS IoU threshold:
0. Baseline: iou_threshold = 0.65 (Current default)
1. Tight NMS: iou_threshold = 0.50 (Aggressively eliminate slice seam duplicates)
2. Moderately Tight NMS: iou_threshold = 0.55
3. Moderate NMS: iou_threshold = 0.60
4. Moderately Relaxed NMS: iou_threshold = 0.70
5. Relaxed NMS: iou_threshold = 0.75
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
    depth_m = np.clip(depth_arr / 1000.0 if depth_arr.max() > 255.0 else depth_arr / 20.0, 0.5, 50.0)
    dz_dx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    div_n = np.abs(cv2.Sobel(-dz_dx / norm, cv2.CV_32F, 1, 0) + cv2.Sobel(-dz_dy / norm, cv2.CV_32F, 0, 1))
    flat_score = np.exp(- (div_n / 0.15)**2)
    phi_vt_gate = (1.0 - 0.75 * flat_score).astype(np.float32)

    disp = 1.0 / np.clip(depth_m, 0.5, 50.0)
    gx_d = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    edge_3d = np.clip(np.sqrt(gx_d**2 + gy_d**2) * 4.0, 0.0, 1.0).astype(np.float32)
    return phi_vt_gate, edge_3d

def precompute_tensors(coco_gt):
    root = "/root/autodl-tmp/AIC"
    stem2id = {os.path.splitext(img['file_name'])[0]: img['id'] for img in coco_gt.dataset['images']}
    stems = sorted(list(stem2id.keys()))

    vis_dir = os.path.join(root, "work_dir/deimv2_dataset/valid/images")
    ir_dir = "/dev/shm/aic_cache/train_aligned_ir"
    dep_dir = "/dev/shm/aic_cache/depth_p0"

    print(f"Pre-caching and pre-computing inputs for {len(stems)} validation images ...", flush=True)
    t_start = time.time()
    cached_inputs = []

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

        ir_t = torch.from_numpy(ir_deb.astype(np.float32) / 255.0).unsqueeze(0)
        if ir_t.shape[1:] != (H, W): ir_t = TF.resize(ir_t, [H, W])

        ch4 = mca_np.astype(np.float32) * phi_vt_gate + 0.30 * 255.0 * edge_3d
        ch4_np = np.clip(ch4, 0, 255).astype(np.uint8)
        lbg_t = torch.from_numpy(ch4_np.astype(np.float32) / 255.0).unsqueeze(0)
        if lbg_t.shape[1:] != (H, W): lbg_t = TF.resize(lbg_t, [H, W])

        tensor_6ch = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0)
        tensor_640 = TF.resize(tensor_6ch, [640, 640]).cuda()
        inputs = ((tensor_640 - MEAN_6CH) / STD_6CH).unsqueeze(0)

        cached_inputs.append((stem2id[s], W, H, inputs))

    print(f"Pre-caching complete in {time.time() - t_start:.2f}s.", flush=True)
    return cached_inputs

def main():
    root = "/root/autodl-tmp/AIC"
    ann_p = os.path.join(root, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
    coco_gt = COCO(ann_p)

    cached_inputs = precompute_tensors(coco_gt)

    soup_p = os.path.join(root, "vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_pure_lmm_soup/best_stg1.pth")
    lmm_cfg = os.path.join(root, "vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_lmm_rtx6000.yml")

    print(f"Loading Model Soup from {soup_p} ...", flush=True)
    cfg = YAMLConfig(lmm_cfg)
    model = cfg.model.cuda().eval()
    ck = torch.load(soup_p, map_location="cpu")
    w = ck["ema"]["module"] if "ema" in ck and "module" in ck["ema"] else ck.get("model", ck)
    model.load_state_dict(w, strict=False)
    post = cfg.postprocessor.cuda()

    experiments = [
        ("0. 基线: SAHI NMS IoU = 0.65 (当前基准)", 0.65),
        ("1. 强力去重: SAHI NMS IoU = 0.50 (深度压制缝隙伪框)", 0.50),
        ("2. 严密去重: SAHI NMS IoU = 0.55", 0.55),
        ("3. 适度去重: SAHI NMS IoU = 0.60", 0.60),
        ("4. 宽松合并: SAHI NMS IoU = 0.70", 0.70),
        ("5. 极度宽松: SAHI NMS IoU = 0.75", 0.75),
    ]

    all_results = []

    for label, iou_thresh in experiments:
        print(f"\n=======================================================", flush=True)
        print(f"Running Experiment 4: {label} ...", flush=True)
        print(f"=======================================================", flush=True)

        model.sahi_iou_threshold = iou_thresh

        t0 = time.time()
        detections = []

        with torch.no_grad():
            for img_id, W, H, inputs in cached_inputs:
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

        s_stat = ev.stats
        res = {
            "name": label,
            "mAP": s_stat[0],
            "AP50": s_stat[1],
            "AP75": s_stat[2],
            "APs": s_stat[3],
            "APm": s_stat[4],
            "APl": s_stat[5],
            "ARs": s_stat[9],
            "time": dt
        }
        all_results.append(res)
        print(f"[{label}] -> mAP: {s_stat[0]:.4f} | AP50: {s_stat[1]:.4f} | AP75: {s_stat[2]:.4f} | APs: {s_stat[3]:.4f} | APl: {s_stat[5]:.4f} ({dt:.1f}s)", flush=True)

    print("\n" + "="*104)
    print("EXPERIMENT 4: MODEL-LEVEL INTERNAL CAA-SAHI IoU CALIBRATION BENCHMARK SUMMARY")
    print("="*104)
    print(f"{'Configuration':<56} | {'mAP':<7} | {'AP50':<7} | {'AP75':<7} | {'APs':<7} | {'APl':<7} | {'ARs':<7}")
    print("-" * 104)
    for r in all_results:
        print(f"{r['name']:<56} | {r['mAP']:.4f}  | {r['AP50']:.4f}  | {r['AP75']:.4f}  | {r['APs']:.4f}  | {r['APl']:.4f}  | {r['ARs']:.4f}")
    print("="*104)

if __name__ == '__main__':
    main()
