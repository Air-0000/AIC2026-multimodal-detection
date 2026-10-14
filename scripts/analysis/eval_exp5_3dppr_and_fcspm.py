"""
Experiment 5: 3D-PPR (3D Planar Protrusion Residual) and FCSPM (Fourier Phase Locking)
======================================================================================
Empirical Validation on Sovereign 6-Channel Model Soup Pipeline (200 Val Split)
Testing:
0. Baseline: Current Champion Orthogonal Split (mAP = 0.5838, AP75 = 0.6528)
1. 3D-PPR in Ch 4: Ch4 = MCA * Phi_VT + 0.30 * PPR (3D Local Planar Protrusion Residual)
2. 3D-PPR High-Pass: Ch4 = MCA * Phi_VT + 0.45 * PPR
3. Pure 3D-PPR: Ch4 = PPR * 255.0 (Pure 3D Protrusion Relief, 0% ground clutter)
4. FCSPM Thermal Core in Ch 3: Phase-Locked IR in Ch3 + Baseline Ch4
5. FCSPM + 3D-PPR Dual Synergy: Phase-Locked IR in Ch3 + 3D-PPR in Ch4
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

def compute_3d_ppr(depth_arr):
    """
    3D Planar Protrusion Residual (3D-PPR)
    Computes perpendicular point-to-plane residual from the local fitted 3D background manifold.
    Completely zeroes out flat road, ground plane, sloping grass, and water surfaces.
    Pops out standing physical targets (pedestrians, vehicles, poles, boats).
    """
    depth_m = np.clip(depth_arr / 1000.0 if depth_arr.max() > 255.0 else depth_arr / 20.0, 0.5, 50.0).astype(np.float32)
    valid = (depth_m > 0.5) & (depth_m < 48.0)

    # 1. Bilateral approximation of local 3D background surface manifold
    d_smooth = cv2.bilateralFilter(depth_m, d=15, sigmaColor=1.5, sigmaSpace=25)

    # 2. Local surface gradient of the background plane
    dz_dx = cv2.Sobel(d_smooth, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(d_smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag_grad = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)

    # 3. True perpendicular Euclidean distance to the local manifold
    residual_3d = np.abs(depth_m - d_smooth) / mag_grad
    residual_3d[~valid] = 0.0

    # 4. Filter micro ground roughness (> 0.12m) and normalize to [0, 1]
    ppr_score = np.clip((residual_3d - 0.12) / 0.88, 0.0, 1.0).astype(np.float32)
    return ppr_score

def compute_fcspm_ir(ir_u8, vis_bgr):
    """
    Fourier Cross-Spectral Phase Modulation (FCSPM)
    Locks the high-frequency Fourier phase of IR to the crisp optical boundary of Visible light,
    retaining raw thermal radiance magnitude without spatial convolution halo or boundary blurring.
    """
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
    cutoff = min(H, W) * 0.05

    # High frequency phase blend weight
    w_p = np.clip((radius - cutoff) / (min(H, W) * 0.25 + 1e-4), 0.0, 1.0) * 0.35

    sin_p = (1.0 - w_p) * np.sin(phase_ir) + w_p * np.sin(phase_vis)
    cos_p = (1.0 - w_p) * np.cos(phase_ir) + w_p * np.cos(phase_vis)
    synth_phase = np.arctan2(sin_p, cos_p)

    f_synth = mag_ir * np.exp(1j * synth_phase)
    ir_synth = np.real(np.fft.ifft2(np.fft.ifftshift(f_synth)))
    ir_synth = np.clip(ir_synth, 0, 255).astype(np.uint8)
    return ir_synth

def main():
    root = "/root/autodl-tmp/AIC"
    ann_p = os.path.join(root, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
    coco_gt = COCO(ann_p)
    stem2id = {os.path.splitext(img['file_name'])[0]: img['id'] for img in coco_gt.dataset['images']}
    stems = sorted(list(stem2id.keys()))

    vis_dir = os.path.join(root, "work_dir/deimv2_dataset/valid/images")
    ir_dir = "/dev/shm/aic_cache/train_aligned_ir"
    dep_dir = "/dev/shm/aic_cache/depth_p0"

    print(f"Pre-caching and pre-computing inputs with 3D-PPR and FCSPM for {len(stems)} validation images ...", flush=True)
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

        # FCSPM
        ir_fcspm = compute_fcspm_ir(ir_deb, vis_bgr)
        mca_fcspm = ir_method1_multiscale_mca(ir_fcspm)

        dep_im = Image.open(dep_p)
        dep_arr = np.array(dep_im, dtype=np.float32)
        scale = 19999.0 if getattr(dep_im, 'mode', '') in ('I;16', 'I') else (65535.0 if dep_arr.max() > 255.5 else 255.0)
        d_norm = np.clip(dep_arr / scale, 0.0, 1.0)
        dep_t = torch.from_numpy(d_norm).unsqueeze(0)
        if dep_t.shape[1:] != (H, W): dep_t = TF.resize(dep_t, [H, W])

        phi_vt_gate, edge_3d = compute_phi_vt_and_3d_edges(dep_arr)
        ppr_score = compute_3d_ppr(dep_arr)

        val_data[s] = {
            "id": stem2id[s],
            "W": W, "H": H,
            "rgb_t": rgb_t,
            "ir_deb": ir_deb,
            "mca_np": mca_np,
            "ir_fcspm": ir_fcspm,
            "mca_fcspm": mca_fcspm,
            "phi_vt_gate": phi_vt_gate,
            "edge_3d": edge_3d,
            "ppr_score": ppr_score,
            "dep_t": dep_t
        }

    print(f"Pre-caching complete in {time.time() - t_start:.2f}s.", flush=True)

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
        ("0. 基线: 冠军正交分流 (Debloom-IR + MCA*Phi_VT + 0.30*2D断层)", "baseline"),
        ("1. 3D-PPR注入: Ch4 = MCA*Phi_VT + 0.30*3D局部平面凸起残差", "ppr_030"),
        ("2. 3D-PPR强化: Ch4 = MCA*Phi_VT + 0.45*3D局部平面凸起残差", "ppr_045"),
        ("3. 纯3D-PPR浮雕: Ch4 = 纯3D局部平面凸起残差 (彻底归零路面水面)", "ppr_pure"),
        ("4. FCSPM频域相位锁定: Ch3 = 傅里叶相位锁定IR + 基线Ch4", "fcspm_ir"),
        ("5. FCSPM + 3D-PPR双项协同: 频域相位锁定IR + 3D平面突出度残差", "synergy_fcspm_ppr"),
    ]

    all_results = []

    for label, mode in experiments:
        print(f"\n=======================================================", flush=True)
        print(f"Running Experiment 5: {label} ...", flush=True)
        print(f"=======================================================", flush=True)

        t0 = time.time()
        detections = []

        with torch.no_grad():
            for s in stems:
                d = val_data[s]
                H, W = d['H'], d['W']
                rgb_t = d['rgb_t']
                dep_t = d['dep_t']
                img_id = d['id']

                # Select Ch3 IR
                if mode in ("fcspm_ir", "synergy_fcspm_ppr"):
                    ir_np = d['ir_fcspm']
                    mca_np = d['mca_fcspm']
                else:
                    ir_np = d['ir_deb']
                    mca_np = d['mca_np']

                ir_t = torch.from_numpy(ir_np.astype(np.float32) / 255.0).unsqueeze(0)
                if ir_t.shape[1:] != (H, W): ir_t = TF.resize(ir_t, [H, W])

                # Select Ch4
                phi_vt_gate = d['phi_vt_gate']
                edge_3d = d['edge_3d']
                ppr_score = d['ppr_score']
                mca_f = mca_np.astype(np.float32)

                if mode == "baseline" or mode == "fcspm_ir":
                    ch4 = mca_f * phi_vt_gate + 0.30 * 255.0 * edge_3d
                elif mode == "ppr_030":
                    ch4 = mca_f * phi_vt_gate + 0.30 * 255.0 * ppr_score
                elif mode == "ppr_045":
                    ch4 = mca_f * phi_vt_gate + 0.45 * 255.0 * ppr_score
                elif mode == "ppr_pure":
                    ch4 = 255.0 * ppr_score
                elif mode == "synergy_fcspm_ppr":
                    ch4 = mca_f * phi_vt_gate + 0.30 * 255.0 * ppr_score

                ch4_np = np.clip(ch4, 0, 255).astype(np.uint8)
                lbg_t = torch.from_numpy(ch4_np.astype(np.float32) / 255.0).unsqueeze(0)
                if lbg_t.shape[1:] != (H, W): lbg_t = TF.resize(lbg_t, [H, W])

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
    print("EXPERIMENT 5: 3D-PPR AND FCSPM PHYSICAL SYNERGY BENCHMARK SUMMARY")
    print("="*104)
    print(f"{'Configuration':<64} | {'mAP':<7} | {'AP50':<7} | {'AP75':<7} | {'APs':<7} | {'APl':<7} | {'ARs':<7}")
    print("-" * 104)
    for r in all_results:
        print(f"{r['name']:<64} | {r['mAP']:.4f}  | {r['AP50']:.4f}  | {r['AP75']:.4f}  | {r['APs']:.4f}  | {r['APl']:.4f}  | {r['ARs']:.4f}")
    print("="*104)

if __name__ == '__main__':
    main()
