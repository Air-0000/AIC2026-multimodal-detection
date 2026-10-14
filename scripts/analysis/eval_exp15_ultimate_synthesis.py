"""
Experiment 15: Sovereign V20 Ultimate Physical & Architectural Synthesis (终极合流决选)
===================================================================================
Synthesizes all individual empirical champions from Exp 5 to Exp 14:
1. Baseline Champion: Sovereign V20 Apex 50/50 + FCSPM(0.08, 0.50) + 2D Edge 0.30
2. Physical Geometry Apex: Sovereign V20 Apex 52/48 + FCSPM(0.08, 0.50) + 3D-PPR(d=15, 0.06m) + RGB-guided depth
3. Thermal Enhanced Apex: Multi-band IR (a=0.75, b=1.20) + FCSPM(0.08, 0.50) + 3D-PPR(0.06m)
4. Ultimate Sovereign Apex: Multi-band FCSPM + 3D-PPR(0.06m) + RGB-guided depth + 52/48 Soup + lambda=1.05
5. High-Recall Sovereign Apex: Multi-band FCSPM + 3D-PPR(0.06m) + 50/50 Soup + lambda=1.10

Strictly on 200 Val split. Evaluates official COCO and score >= 0.50 filtered metrics.
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

def compute_3d_ppr(depth_arr, d_filter=15, thresh=0.06):
    depth_m = np.clip(depth_arr / 1000.0 if depth_arr.max() > 255.0 else depth_arr / 20.0, 0.5, 50.0).astype(np.float32)
    valid = (depth_m > 0.5) & (depth_m < 48.0)
    d_smooth = cv2.bilateralFilter(depth_m, d=d_filter, sigmaColor=1.5, sigmaSpace=25)
    dz_dx = cv2.Sobel(d_smooth, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(d_smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag_grad = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    residual_3d = np.abs(depth_m - d_smooth) / mag_grad
    residual_3d[~valid] = 0.0
    ppr_score = np.clip((residual_3d - thresh) / (1.0 - thresh + 1e-4), 0.0, 1.0).astype(np.float32)
    return ppr_score

def compute_fcspm_ir(ir_u8, vis_bgr, cutoff_ratio=0.08, max_weight=0.50):
    vis_gray = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY)
    if vis_gray.shape != ir_u8.shape:
        vis_gray = cv2.resize(vis_gray, (ir_u8.shape[1], ir_u8.shape[0]))

    H, W = ir_u8.shape
    f_ir = np.fft.fftshift(np.fft.fft2(ir_u8.astype(np.float32)))
    f_vis = np.fft.fftshift(np.fft.fft2(vis_gray.astype(np.float32)))

    mag_ir = np.abs(f_ir)
    phase_ir = np.angle(f_ir)
    phase_vis = np.angle(f_vis)

    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    radius = np.sqrt((x - cx)**2 + (y - cy)**2)
    cutoff = min(H, W) * cutoff_ratio

    w_p = np.clip((radius - cutoff) / (min(H, W) * 0.25 + 1e-4), 0.0, 1.0) * max_weight
    sin_p = (1.0 - w_p) * np.sin(phase_ir) + w_p * np.sin(phase_vis)
    cos_p = (1.0 - w_p) * np.cos(phase_ir) + w_p * np.cos(phase_vis)
    synth_phase = np.arctan2(sin_p, cos_p)

    f_synth = mag_ir * np.exp(1j * synth_phase)
    ir_synth = np.real(np.fft.ifft2(np.fft.ifftshift(f_synth)))
    return np.clip(ir_synth, 0, 255).astype(np.uint8)

def decompose_multiband_ir(ir_u8, alpha=0.75, beta=1.20, gamma=1.10):
    f = ir_u8.astype(np.float32)
    low = cv2.GaussianBlur(f, (0, 0), sigmaX=15, sigmaY=15)
    blur_mid = cv2.GaussianBlur(f, (0, 0), sigmaX=3, sigmaY=3)
    mid = blur_mid - low
    high = f - blur_mid
    recon = alpha * low + beta * mid + gamma * high
    return np.clip(recon, 0, 255).astype(np.uint8)

def inpaint_rgb_guided(depth_arr, vis_bgr):
    mask_zero = (depth_arr <= 10.0).astype(np.uint8)
    if mask_zero.sum() == 0:
        return depth_arr
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_depth = cv2.dilate(depth_arr, kernel)
    return np.where(mask_zero > 0, dilated_depth, depth_arr)

def load_weight_dict(ck_path):
    ck = torch.load(ck_path, map_location="cpu")
    w = ck["ema"]["module"] if "ema" in ck and "module" in ck["ema"] else ck.get("model", ck)
    return w

def make_soup(weights_and_alphas):
    soup = {}
    base_w, _ = weights_and_alphas[0]
    for k in base_w.keys():
        if all(k in w for w, _ in weights_and_alphas):
            if base_w[k].dtype in (torch.float32, torch.float16, torch.float64):
                v_sum = sum(w[k].float() * alpha for w, alpha in weights_and_alphas)
                soup[k] = v_sum.to(base_w[k].dtype)
            else:
                soup[k] = base_w[k].clone()
        else:
            soup[k] = base_w[k].clone()
    return soup

def main():
    root = "/root/autodl-tmp/AIC"
    ann_p = os.path.join(root, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
    coco_gt = COCO(ann_p)
    stem2id = {os.path.splitext(img['file_name'])[0]: img['id'] for img in coco_gt.dataset['images']}
    stems = sorted(list(stem2id.keys()))

    vis_dir = os.path.join(root, "work_dir/deimv2_dataset/valid/images")
    ir_dir = "/dev/shm/aic_cache/train_aligned_ir"
    dep_dir = "/dev/shm/aic_cache/depth_p0"

    print(f"Pre-caching and pre-computing inputs for {len(stems)} validation images ...", flush=True)
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
        mca_deb = ir_method1_multiscale_mca(ir_deb)

        # Baseline FCSPM IR (cutoff 0.08, w 0.50)
        ir_fcspm = compute_fcspm_ir(ir_deb, vis_bgr, cutoff_ratio=0.08, max_weight=0.50)
        ir_fcspm_t = torch.from_numpy(ir_fcspm.astype(np.float32) / 255.0).unsqueeze(0)
        if ir_fcspm_t.shape[1:] != (H, W): ir_fcspm_t = TF.resize(ir_fcspm_t, [H, W])

        # Thermal-suppressed FCSPM IR (alpha=0.75, beta=1.20, gamma=1.10)
        ir_multiband = decompose_multiband_ir(ir_fcspm, alpha=0.75, beta=1.20, gamma=1.10)
        ir_mb_t = torch.from_numpy(ir_multiband.astype(np.float32) / 255.0).unsqueeze(0)
        if ir_mb_t.shape[1:] != (H, W): ir_mb_t = TF.resize(ir_mb_t, [H, W])

        dep_im = Image.open(dep_p)
        dep_arr = np.array(dep_im, dtype=np.float32)
        scale = 19999.0 if getattr(dep_im, 'mode', '') in ('I;16', 'I') else (65535.0 if dep_arr.max() > 255.5 else 255.0)

        # Raw normalized depth
        d_norm_raw = np.clip(dep_arr / scale, 0.0, 1.0)
        dep_raw_t = torch.from_numpy(d_norm_raw).unsqueeze(0)
        if dep_raw_t.shape[1:] != (H, W): dep_raw_t = TF.resize(dep_raw_t, [H, W])

        # RGB-guided inpainting depth
        dep_inpaint = inpaint_rgb_guided(dep_arr, vis_bgr)
        d_norm_inp = np.clip(dep_inpaint / scale, 0.0, 1.0)
        dep_inp_t = torch.from_numpy(d_norm_inp).unsqueeze(0)
        if dep_inp_t.shape[1:] != (H, W): dep_inp_t = TF.resize(dep_inp_t, [H, W])

        # Ch4 variant A: 2D Disparity Edge 0.30
        phi_vt_gate, edge_3d = compute_phi_vt_and_3d_edges(dep_arr)
        mca_f = mca_deb.astype(np.float32)
        ch4_a = mca_f * phi_vt_gate + 0.30 * 255.0 * edge_3d
        lbg_2d_t = torch.from_numpy(np.clip(ch4_a, 0, 255).astype(np.uint8).astype(np.float32) / 255.0).unsqueeze(0)
        if lbg_2d_t.shape[1:] != (H, W): lbg_2d_t = TF.resize(lbg_2d_t, [H, W])

        # Ch4 variant B: 3D-PPR Residual (d=15, thresh=0.06m)
        ppr_score = compute_3d_ppr(dep_arr, d_filter=15, thresh=0.06)
        ch4_b = mca_f * phi_vt_gate + 0.30 * 255.0 * ppr_score
        lbg_3d_t = torch.from_numpy(np.clip(ch4_b, 0, 255).astype(np.uint8).astype(np.float32) / 255.0).unsqueeze(0)
        if lbg_3d_t.shape[1:] != (H, W): lbg_3d_t = TF.resize(lbg_3d_t, [H, W])

        # Pack combinations
        # 1. Baseline: FCSPM + 2D Edge + Raw Depth
        inp_1 = TF.resize(torch.cat([rgb_t, ir_fcspm_t, lbg_2d_t, dep_raw_t], dim=0), [640, 640])
        # 2. Geometry: FCSPM + 3D-PPR + RGB Inpaint Depth
        inp_2 = TF.resize(torch.cat([rgb_t, ir_fcspm_t, lbg_3d_t, dep_inp_t], dim=0), [640, 640])
        # 3. Thermal + 3D-PPR
        inp_3 = TF.resize(torch.cat([rgb_t, ir_mb_t, lbg_3d_t, dep_raw_t], dim=0), [640, 640])
        # 4. Ultimate Synthesis: Thermal FCSPM + 3D-PPR + RGB Inpaint Depth
        inp_4 = TF.resize(torch.cat([rgb_t, ir_mb_t, lbg_3d_t, dep_inp_t], dim=0), [640, 640])

        val_data[s] = {
            "id": stem2id[s],
            "W": W, "H": H,
            "inp_1": inp_1,
            "inp_2": inp_2,
            "inp_3": inp_3,
            "inp_4": inp_4,
        }

    print(f"Pre-caching complete in {time.time() - t_start:.2f}s.", flush=True)

    pure_p = os.path.join(root, "vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_pure_rtx6000/best_stg1.pth")
    lmm_p = os.path.join(root, "vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_lmm_rtx6000/best_stg1.pth")
    lmm_cfg = os.path.join(root, "vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_lmm_rtx6000.yml")

    w_pure = load_weight_dict(pure_p)
    w_lmm = load_weight_dict(lmm_p)

    cfg = YAMLConfig(lmm_cfg)
    model = cfg.model.cuda().eval()
    post = cfg.postprocessor.cuda()

    synthesis_benchmarks = [
        ("0. Sovereign V20 基准冠 (50/50 + FCSPM + 2D Edge, Exp 8/9)", "inp_1", 0.50, 0.50, 1.00),
        ("1. 几何流形极值版 (52/48 + FCSPM + 3D-PPR + Inpaint深度)", "inp_2", 0.52, 0.48, 1.00),
        ("2. 热像微弱爆发版 (50/50 + 多频段IR + 3D-PPR, Exp 12增强)", "inp_3", 0.50, 0.50, 1.00),
        ("3. 终极合流巅峰版 (52/48 + 多频段FCSPM + 3D-PPR + Inpaint + scale=1.05)", "inp_4", 0.52, 0.48, 1.05),
        ("4. 极限高置信召回版 (50/50 + 多频段FCSPM + 3D-PPR + scale=1.10)", "inp_3", 0.50, 0.50, 1.10),
    ]

    all_results = []

    for label, input_key, alpha_pure, alpha_lmm, ir_scale in synthesis_benchmarks:
        print(f"\n=======================================================", flush=True)
        print(f"Evaluating {label} ...", flush=True)
        print(f"=======================================================", flush=True)

        soup = make_soup([(w_pure, alpha_pure), (w_lmm, alpha_lmm)])
        if ir_scale != 1.00:
            for k in soup.keys():
                if "backbone.ir_res." in k:
                    soup[k] = soup[k] * ir_scale
        model.load_state_dict(soup, strict=False)

        detections_all = []
        detections_conf05 = []

        with torch.no_grad():
            for s in stems:
                d = val_data[s]
                H, W = d['H'], d['W']
                img_id = d['id']
                tensor_640 = d[input_key].cuda()
                inputs = ((tensor_640 - MEAN_6CH) / STD_6CH).unsqueeze(0)

                orig_target_sizes = torch.tensor([[float(W), float(H)]], device="cuda")
                outputs = model(inputs)
                res = post(outputs, orig_target_sizes)[0]

                labels = res['labels'].cpu().numpy()
                boxes = res['boxes'].cpu().numpy()
                scores = res['scores'].cpu().numpy()

                for lbl, box, sc in zip(labels, boxes, scores):
                    x1, y1, x2, y2 = box
                    det_dict = {
                        "image_id": img_id,
                        "category_id": int(lbl),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(sc)
                    }
                    detections_all.append(det_dict)
                    if float(sc) >= 0.50:
                        detections_conf05.append(det_dict)

        # Official evaluation
        coco_dt_all = coco_gt.loadRes(detections_all)
        ev_all = COCOeval(coco_gt, coco_dt_all, "bbox")
        ev_all.evaluate()
        ev_all.accumulate()
        ev_all.summarize()
        stats_all = ev_all.stats

        # Conf >= 0.50 evaluation
        coco_dt_05 = coco_gt.loadRes(detections_conf05)
        ev_05 = COCOeval(coco_gt, coco_dt_05, "bbox")
        ev_05.evaluate()
        ev_05.accumulate()
        ev_05.summarize()
        stats_05 = ev_05.stats

        num_dets = len(detections_conf05)
        avg_dets_per_img = num_dets / len(stems)

        record = {
            "name": label,
            "raw_mAP": stats_all[0],
            "raw_AP50": stats_all[1],
            "raw_AP75": stats_all[2],
            "raw_APs": stats_all[3],
            "raw_APl": stats_all[5],
            "raw_ARs": stats_all[9],
            "conf05_dets_per_img": avg_dets_per_img,
            "conf05_mAP": stats_05[0],
            "conf05_AP50": stats_05[1],
            "conf05_AP75": stats_05[2],
            "conf05_APs": stats_05[3],
            "conf05_APl": stats_05[5],
            "conf05_ARs": stats_05[9],
        }
        all_results.append(record)

    print("\n" + "="*128)
    print("EXPERIMENT 15: SOVEREIGN V20 ULTIMATE SYNTHESIS BENCHMARK (OFFICIAL VS SCORE >= 0.50)")
    print("="*128)
    print(f"{'Configuration':<56} | {'mAP':<6} | {'AP50':<6} | {'AP75':<6} | {'APs':<6} | {'ARs':<6} || {'0.5_mAP':<7} | {'0.5_AP50':<8} | {'0.5_AP75':<8} | {'Dets/Img'}")
    print("-" * 128)
    for r in all_results:
        print(f"{r['name']:<56} | {r['raw_mAP']:.4f} | {r['raw_AP50']:.4f} | {r['raw_AP75']:.4f} | {r['raw_APs']:.4f} | {r['raw_ARs']:.4f} || {r['conf05_mAP']:.4f}  | {r['conf05_AP50']:.4f}   | {r['conf05_AP75']:.4f}   | {r['conf05_dets_per_img']:.2f}")
    print("="*128)

if __name__ == '__main__':
    main()
