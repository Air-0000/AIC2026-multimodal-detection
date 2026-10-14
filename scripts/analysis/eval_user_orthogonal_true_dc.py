#!/usr/bin/env python3
"""
Rigorous Benchmark of User's 5-Point Architecture with Proper DC/AC Decoupling
=============================================================================
Channel Mapping:
  Ch 0~2 (RGB): 100% 原始高分辨率 (零人工模糊衰减，死守大目标 AP_large 与 AP75 底线)
  Ch 3 (IR 专用流): PSF 光学逆卷积 + 各向异性热晕刚体收拢 (保持完整热核 DC 辐射底数，锐度翻 2.4 倍)
  Ch 4 (几何与高频流): 
    - 方案 4A: 纯 3D 视差几何断层场 Edge_3D
    - 方案 4B: 3D 视差几何断层场 Edge_3D × Phi_VT
    - 方案 4C: 多尺度 MCA × Phi_VT + 0.30 × Edge_3D (黄金均衡)
  Ch 5 (绝对深度): Method Depth-1 遮挡感知双边引导深度 vs P0 绝对深度
  Backbone 融合节点: Method Joint-3 3D 体积-热密度相干核 Phi_VT 特征层动态截断 (消灭地面积水伪热斑)
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
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


def op_thermal_psf_deconv(ir_u8, sigma=1.2, iters=4):
    """PSF 光学逆卷积 (去镜头衍射模糊)"""
    img = np.maximum(ir_u8.astype(np.float32) / 255.0, 1e-4)
    ksize = int(6 * sigma + 1) | 1
    psf = cv2.getGaussianKernel(ksize, sigma)
    psf = psf @ psf.T
    psf = psf / psf.sum()
    u = img.copy()
    for _ in range(iters):
        conv_u = cv2.filter2D(u, -1, psf, borderType=cv2.BORDER_REFLECT)
        conv_u = np.maximum(conv_u, 1e-4)
        u = u * cv2.filter2D(img / conv_u, -1, psf, borderType=cv2.BORDER_REFLECT)
    return np.clip(u * 255.0, 0, 255).astype(np.uint8)


def ir_method3_anisotropic_thermal_deblooming(ir_u8, vis_bgr, n_iters=3, dt=0.20):
    """各向异性逆热扩散 (热晕向可见光刚体边缘收拢)"""
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


def op_multiscale_mca(ir_u8, scales=(5, 11, 21, 41, 81)):
    """多尺度 MCA Top-Hat"""
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


def compute_3d_disparity_edge(depth_m):
    """3D 视差几何断层场 Edge_3D"""
    disp = 1.0 / np.clip(depth_m, 0.5, 50.0)
    gx_d = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    edge_3d = np.clip(np.sqrt(gx_d**2 + gy_d**2) * 4.0, 0.0, 1.0).astype(np.float32)
    return (edge_3d * 255.0).astype(np.uint8)


def depth_method1_occlusion_aware_bilateral_guided(depth_m, vis_bgr, r=4, eps=1e-3, tau_jump=0.08):
    """Method Depth-1: 遮挡感知视差域双边引导滤波"""
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
    return np.where(jump_mag > tau_jump, depth_m, depth_refined).astype(np.float32)


def compute_phi_vt_curvature_gate(depth_m):
    """Method Joint-3: 3D 体积-热密度相干核 Phi_VT"""
    dz_dx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    div_n = np.abs(cv2.Sobel(-dz_dx / norm, cv2.CV_32F, 1, 0) + cv2.Sobel(-dz_dy / norm, cv2.CV_32F, 0, 1))

    flat_score = np.exp(- (div_n / 0.15)**2)
    phi_vt_gate = (1.0 - 0.75 * flat_score).astype(np.float32)
    return phi_vt_gate


def evaluate_configuration(model, post, val_data, coco_gt, name, cfg_opts):
    stems = sorted(list(val_data.keys()))
    detections = []
    t0 = time.time()

    hook_handle = None
    if cfg_opts.get("backbone_feature_gate", False):
        current_phi_vt_tensor = [None]
        def ir_stem_hook(module, input, output):
            if current_phi_vt_tensor[0] is not None:
                phi_vt_t = current_phi_vt_tensor[0]
                c2, c3 = output
                gate_c2 = F.interpolate(phi_vt_t, size=c2.shape[2:], mode='bilinear', align_corners=False)
                gate_c3 = F.interpolate(phi_vt_t, size=c3.shape[2:], mode='bilinear', align_corners=False)
                return (c2 * gate_c2, c3 * gate_c3)
            return output
        hook_handle = model.backbone.ir_stem.register_forward_hook(ir_stem_hook)

    with torch.no_grad():
        for stem in stems:
            d = val_data[stem]
            H, W = d['H'], d['W']
            img_id = d['id']
            rgb_t = d['rgb_t']

            # Channel 3: Full DC thermal radiance with physical deconvolution & deblooming
            ir_np = d['ir_psf_deb']
            ir_t = torch.from_numpy(ir_np.astype(np.float32) / 255.0).unsqueeze(0)
            if ir_t.shape[1:] != (H, W): ir_t = TF.resize(ir_t, [H, W])

            # Channel 4: Geometric Prior Mode
            mode4 = cfg_opts.get("ch4_mode")
            if mode4 == "pure_edge3d":
                ch4_np = d['edge_3d']
            elif mode4 == "edge3d_phi_vt":
                ch4_np = np.clip(d['edge_3d'].astype(np.float32) * d['phi_vt_gate'], 0, 255).astype(np.uint8)
            elif mode4 == "mca_phi_vt_030_edge3d":
                ch4_np = np.clip(d['mca_np'].astype(np.float32) * d['phi_vt_gate'] + 0.30 * d['edge_3d'].astype(np.float32), 0, 255).astype(np.uint8)
            else:
                ch4_np = d['mca_np']

            lbg_t = torch.from_numpy(ch4_np.astype(np.float32) / 255.0).unsqueeze(0)
            if lbg_t.shape[1:] != (H, W): lbg_t = TF.resize(lbg_t, [H, W])

            # Channel 5: Depth Mode
            if cfg_opts.get("ch5_mode") == "biguide":
                dep_t = d['dep_biguide_t']
            else:
                dep_t = d['dep_p0_t']

            t6 = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0)
            inputs = ((TF.resize(t6, [640, 640]).cuda() - MEAN_6CH) / STD_6CH).unsqueeze(0)

            if cfg_opts.get("backbone_feature_gate", False):
                gate_640 = torch.from_numpy(d['phi_vt_gate']).unsqueeze(0).unsqueeze(0).cuda()
                current_phi_vt_tensor[0] = gate_640

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

    if hook_handle is not None:
        hook_handle.remove()

    dt = time.time() - t0
    coco_dt = coco_gt.loadRes(detections)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    s = ev.stats
    res_dict = {
        "name": name,
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
    print(f"[{name}] -> mAP: {s[0]:.4f} | AP50: {s[1]:.4f} | AP75: {s[2]:.4f} | APs: {s[3]:.4f} | APl: {s[5]:.4f} | ARs: {s[9]:.4f} ({dt:.1f}s)", flush=True)
    return res_dict


def main():
    root = "/root/autodl-tmp/AIC"
    ann_p = os.path.join(root, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
    coco_gt = COCO(ann_p)
    stem2id = {os.path.splitext(img['file_name'])[0]: img['id'] for img in coco_gt.dataset['images']}
    stems = sorted(list(stem2id.keys()))

    vis_dir = os.path.join(root, "work_dir/deimv2_dataset/valid/images")
    ir_dir = "/dev/shm/aic_cache/train_aligned_ir"
    dep_dir = "/dev/shm/aic_cache/depth_p0"

    print(f"Pre-caching and computing physical representations for {len(stems)} images...", flush=True)
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
        ir_raw = np.array(ir_im)

        # Ch 3: PSF deblur on raw IR + IR-3 debloom (keeping full DC thermal radiance)
        ir_psf = op_thermal_psf_deconv(ir_raw, sigma=1.2, iters=4)
        ir_psf_deb = ir_method3_anisotropic_thermal_deblooming(ir_psf, vis_bgr)

        # Ch 4: MCA from debloomed IR
        mca_np = op_multiscale_mca(ir_psf_deb)

        # Depth P0 & Depth-1
        dep_im = Image.open(dep_p)
        dep_arr = np.array(dep_im, dtype=np.float32)
        scale = 19999.0 if getattr(dep_im, 'mode', '') in ('I;16', 'I') else (65535.0 if dep_arr.max() > 255.5 else 255.0)
        d_norm = np.clip(dep_arr / scale, 0.0, 1.0)
        dep_p0_t = torch.from_numpy(d_norm).unsqueeze(0)
        if dep_p0_t.shape[1:] != (H, W): dep_p0_t = TF.resize(dep_p0_t, [H, W])

        depth_m = np.clip(dep_arr / 1000.0 if dep_arr.max() > 255.0 else dep_arr / 20.0, 0.5, 50.0)
        edge_3d = compute_3d_disparity_edge(depth_m)

        dep_biguide_m = depth_method1_occlusion_aware_bilateral_guided(depth_m, vis_bgr)
        dep_biguide_norm = np.clip(dep_biguide_m * 1000.0 / scale, 0.0, 1.0)
        dep_biguide_t = torch.from_numpy(dep_biguide_norm).unsqueeze(0)
        if dep_biguide_t.shape[1:] != (H, W): dep_biguide_t = TF.resize(dep_biguide_t, [H, W])

        phi_vt_gate = compute_phi_vt_curvature_gate(depth_m)

        val_data[s] = {
            "id": stem2id[s], "W": W, "H": H,
            "rgb_t": rgb_t,
            "ir_psf_deb": ir_psf_deb,
            "mca_np": mca_np,
            "edge_3d": edge_3d,
            "dep_p0_t": dep_p0_t,
            "dep_biguide_t": dep_biguide_t,
            "phi_vt_gate": phi_vt_gate
        }

    print(f"Pre-caching complete in {time.time() - t_start:.2f}s.", flush=True)

    soup_p = os.path.join(root, "vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_pure_lmm_soup/best_stg1.pth")
    lmm_cfg = os.path.join(root, "vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_lmm_rtx6000.yml")

    cfg = YAMLConfig(lmm_cfg)
    model = cfg.model.cuda().eval()
    ck = torch.load(soup_p, map_location="cpu")
    w = ck["ema"]["module"] if "ema" in ck and "module" in ck["ema"] else ck.get("model", ck)
    model.load_state_dict(w, strict=False)
    post = cfg.postprocessor.cuda()

    benchmarks = [
        ("A. 用户五要素全集 (Ch4=纯3D几何断层, Ch5=Depth1双边引导, 特征层Phi_VT门控)", {
            "ch4_mode": "pure_edge3d",
            "ch5_mode": "biguide",
            "backbone_feature_gate": True
        }),
        ("B. 用户五要素全集 (Ch4=纯3D几何断层, Ch5=Depth1双边引导, 无特征门控)", {
            "ch4_mode": "pure_edge3d",
            "ch5_mode": "biguide",
            "backbone_feature_gate": False
        }),
        ("C. 用户五要素全集 + 能量流协同 (Ch4=MCA×Phi_VT+0.3*Edge_3D, Ch5=Depth1双边引导)", {
            "ch4_mode": "mca_phi_vt_030_edge3d",
            "ch5_mode": "biguide",
            "backbone_feature_gate": False
        }),
        ("D. 终极巅峰对比 (Ch4=MCA×Phi_VT+0.3*Edge_3D, Ch5=P0绝对深度) [现役0.5851记录]", {
            "ch4_mode": "mca_phi_vt_030_edge3d",
            "ch5_mode": "p0",
            "backbone_feature_gate": False
        }),
        ("E. 终极巅峰 + 特征层Phi_VT门控 (Ch4=MCA×Phi_VT+0.3*Edge_3D, Ch5=P0, 特征层门控)", {
            "ch4_mode": "mca_phi_vt_030_edge3d",
            "ch5_mode": "p0",
            "backbone_feature_gate": True
        })
    ]

    results = []
    for label, opts in benchmarks:
        print(f"\n=======================================================", flush=True)
        print(f"Running: {label} ...", flush=True)
        print(f"=======================================================", flush=True)
        r = evaluate_configuration(model, post, val_data, coco_gt, label, opts)
        results.append(r)

    print("\n" + "="*112)
    print("USER 5-POINT PROPER DC/AC ARCHITECTURE BENCHMARK SUMMARY")
    print("="*112)
    print(f"{'Configuration':<62} | {'mAP':<7} | {'AP50':<7} | {'AP75':<7} | {'APs':<7} | {'APl':<7} | {'ARs':<7}")
    print("-" * 112)
    for r in results:
        print(f"{r['name']:<62} | {r['mAP']:.4f}  | {r['AP50']:.4f}  | {r['AP75']:.4f}  | {r['APs']:.4f}  | {r['APl']:.4f}  | {r['ARs']:.4f}")
    print("="*112)


if __name__ == '__main__':
    main()
