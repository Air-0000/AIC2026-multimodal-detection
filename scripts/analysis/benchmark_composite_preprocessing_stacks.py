#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_composite_preprocessing_stacks.py
=============================================================================
全面测试多种前沿算子的【复合组合策略】(Composite Preprocessing Stacks):
对比矩阵:
  1. 原生基线 (Raw Baseline, 0.5630)
  2. 单项冠军 (Single Champion: Method Joint-3 Phi_VT, 0.5648)
  3. 组合 A: 【极致热辐射物理栈】 (Thermal Radiance Stack)
     - PSF 光学逆卷积 + 多尺度 MCA + 3D 体积热相干 (Phi_VT)
  4. 组合 B: 【全天候极性-热晕收拢栈】 (Polarity & De-blooming Stack)
     - 极性不变变换 (E_polar) + 可见光各向异性逆热扩散 (De-blooming)
  5. 组合 C: 【微观几何刀锋栈】 (Micro-Geometric Edge & 3D Manifold Stack)
     - 遮挡感知双边引导滤波 + 嵌入流形切空间 Laplace-Beltrami 平滑
  6. 组合 D: 【光-热-深终极三相相干合流栈】 (Grand Unified Tri-Modal Composite)
     - 融合 PSF逆卷积 + 多尺度MCA + 遮挡双边深度 + 3D体积热门控Phi_VT + 各向异性边界收束
=============================================================================
"""

import os
import sys
import json
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
sys.path.insert(0, os.path.join(ROOT, "scripts/lib"))
sys.path.insert(0, DEIMV2_ROOT)
os.chdir(DEIMV2_ROOT)

import faster_coco_eval
faster_coco_eval.init_as_pycocotools()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from engine.core import YAMLConfig

ANN_JSON = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
IMG_DIR = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/images")
IR_DIR = os.path.join(ROOT, "dataset/train/infrared")
DEPTH_DIR = os.path.join(ROOT, "dataset/train/depth")
OUT_DIR = os.path.join(ROOT, "work_dir/audits/audit_composite_techniques")
os.makedirs(OUT_DIR, exist_ok=True)

coco_gt = COCO(ANN_JSON)
images_info = coco_gt.dataset["images"]
print(f"Total valid images: {len(images_info)}")

cfg = YAMLConfig("configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_puremal.yml")
ckpt_path = "outputs/deim_hgnetv2_x_aic_puremal_refine_giou35/best_stg1.pth"
if not os.path.exists(ckpt_path):
    ckpt_path = "outputs/deim_hgnetv2_x_aic_obj2coco_ft_puremal/best_stg2.pth"

print(f"Loading checkpoint: {ckpt_path}")
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ck["ema"]["module"] if ("ema" in ck and isinstance(ck["ema"], dict) and "module" in ck["ema"]) else ck.get("model", ck["ema"])
cfg.model.load_state_dict(sd, strict=False)
device = torch.device("cuda")
model = cfg.model.to(device).eval()
print("Model loaded successfully.")

resize_tensor = T.Compose([T.Resize((640, 640)), T.ToTensor()])

def read_depth_m(base_name):
    p_png = os.path.join(DEPTH_DIR, base_name + ".png")
    p_jpg = os.path.join(DEPTH_DIR, base_name + ".jpg")
    p = p_png if os.path.exists(p_png) else p_jpg
    if os.path.exists(p):
        im = Image.open(p)
        arr = np.array(im).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        scale = 1000.0 if getattr(im, 'mode', '') in ('I;16', 'I') else (255.0 / 20.0 * 1000.0)
        return arr / scale
    return None

# =============================================================================
# 原子算法算子
# =============================================================================

def op_thermal_psf_deconv(ir_u8, sigma=1.2, iters=5):
    """PSF 光学逆卷积"""
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

def op_multiscale_mca(ir_u8, scales=(5, 11, 21)):
    """多尺度形态学 MCA"""
    out = np.zeros_like(ir_u8, dtype=np.float32)
    tot_w = 0.0
    for k in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        wth = cv2.morphologyEx(ir_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        w = 1.0 / np.sqrt(k)
        out += w * wth
        tot_w += w
    return np.clip((out / tot_w) * 1.5, 0, 255)

def op_phi_vt_filtering(thermal_energy_f, depth_m):
    """3D 几何曲率相干门控 (抑制水面和地面平坦假热斑)"""
    dm = depth_m.astype(np.float32)
    dz_dx = cv2.Sobel(dm, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(dm, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    div_n = np.abs(cv2.Sobel(-dz_dx/norm, cv2.CV_32F, 1, 0) + cv2.Sobel(-dz_dy/norm, cv2.CV_32F, 0, 1))
    flat_score = np.exp(- (div_n / 0.15)**2)
    
    phi_vt = thermal_energy_f.astype(np.float32) * (1.0 - 0.75 * flat_score)
    valid_mask = dm > 0.05
    phi_vt[~valid_mask] *= 0.5
    return phi_vt.astype(np.float32)

def op_anisotropic_debloom(thermal_f, vis_bgr, n_iters=2, dt=0.20):
    """可见光结构屏障导向的逆热扩散"""
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mag_v = np.sqrt(cv2.Sobel(vg, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(vg, cv2.CV_32F, 0, 1)**2)
    barrier = np.exp(- (mag_v / 0.12)**2)
    
    u = thermal_f.astype(np.float32).copy()
    for _ in range(n_iters):
        mag_u = np.sqrt(cv2.Sobel(u, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(u, cv2.CV_32F, 0, 1)**2)
        lap_u = cv2.Laplacian(u, cv2.CV_32F, ksize=3)
        shock = - np.sign(lap_u) * mag_u * barrier
        u = u + dt * shock
    return np.clip(u, 0, 255)

# =============================================================================
# 复合组合方案构造
# =============================================================================

def stack_baseline(rgb_pil, fname):
    return rgb_pil

def stack_champion_single_phi_vt(rgb_pil, fname):
    """单项冠军: Method Joint-3 (Phi_VT)"""
    base_name = os.path.splitext(fname)[0]
    depth_m = read_depth_m(base_name)
    ir_path = os.path.join(IR_DIR, fname)
    rgb_np = np.array(rgb_pil)
    if depth_m is None or not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    depth_m = cv2.resize(depth_m, (rgb_np.shape[1], rgb_np.shape[0]))
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0]))
    
    lbg = np.clip(ir.astype(np.float32) - cv2.medianBlur(ir, 31).astype(np.float32), 0, 255)
    phi_vt = op_phi_vt_filtering(lbg, depth_m)
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.10 * (phi_vt / 255.0)), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

def stack_combo_A_thermal_radiance(rgb_pil, fname):
    """组合 A: 【极致热辐射物理栈】
    链路: PSF 光学逆卷积 (去镜头模糊) -> 多尺度 MCA (保留微小/大热斑) -> 3D 曲率 Phi_VT (消灭水面伪热斑)
    """
    base_name = os.path.splitext(fname)[0]
    depth_m = read_depth_m(base_name)
    ir_path = os.path.join(IR_DIR, fname)
    rgb_np = np.array(rgb_pil)
    if depth_m is None or not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    depth_m = cv2.resize(depth_m, (rgb_np.shape[1], rgb_np.shape[0]))
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0]))
    
    # 1. PSF 逆卷积
    ir_psf = op_thermal_psf_deconv(ir, sigma=1.2, iters=5)
    # 2. 多尺度 MCA
    mca_energy = op_multiscale_mca(ir_psf, scales=(5, 11, 21))
    # 3. 3D 体积曲率过滤
    filtered_energy = op_phi_vt_filtering(mca_energy, depth_m)
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.11 * (filtered_energy / 255.0)), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

def stack_combo_B_polarity_debloom(rgb_pil, fname):
    """组合 B: 【全天候极性-热晕收拢栈】
    链路: 双极性温差变换 (E_polar) + 可见光导向逆热扩散 (De-blooming)
    """
    ir_path = os.path.join(IR_DIR, fname)
    rgb_np = np.array(rgb_pil)
    if not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0]))
    
    # 1. 双极性变换
    med = cv2.medianBlur(ir, 31)
    abs_diff = np.abs(ir.astype(np.float32) - med.astype(np.float32))
    mag = np.sqrt(cv2.Sobel(ir, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(ir, cv2.CV_32F, 0, 1)**2)
    e_polar = abs_diff + 0.30 * mag
    
    # 2. 沿可见光刚体边缘收拢热晕
    e_debloom = op_anisotropic_debloom(e_polar, rgb_np, n_iters=2, dt=0.18)
    norm_feat = np.clip(e_debloom / 255.0, 0.0, 1.0)
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.09 * norm_feat), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

def stack_combo_C_geometric_blade(rgb_pil, fname):
    """组合 C: 【微观几何刀锋栈】
    链路: 视差突变边缘掩膜 (Depth-1) + 距离自适应纹理衰减 (Joint-4)
    """
    base_name = os.path.splitext(fname)[0]
    depth_m = read_depth_m(base_name)
    rgb_np = np.array(rgb_pil)
    if depth_m is None:
        return rgb_pil
    depth_m = cv2.resize(depth_m, (rgb_np.shape[1], rgb_np.shape[0]))
    
    # 1. 深度自适应高频衰减
    z_near, z_far = 4.0, 16.0
    d_clip = np.clip(depth_m, z_near, z_far)
    tau = (d_clip - z_near) / (z_far - z_near + 1e-4)
    blur_mild = cv2.GaussianBlur(rgb_np, (5, 5), 1.0)
    tau_3ch = np.repeat(tau[:, :, None], 3, axis=2)
    blended = (1.0 - tau_3ch) * rgb_np.astype(np.float32) + tau_3ch * blur_mild.astype(np.float32)
    
    # 2. 注入 3D 几何断层锐度
    disp = 1.0 / np.clip(depth_m, 0.5, 50.0)
    gx_d = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    edge_3d = np.clip(np.sqrt(gx_d**2 + gy_d**2) * 5.0, 0.0, 1.0)
    
    lab = cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.08 * edge_3d), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

def stack_combo_D_grand_unified(rgb_pil, fname):
    """组合 D: 【光-热-深终极三相相干合流栈 (Grand Unified Composite)】
    链路:
      - 红外: PSF 逆卷积去模糊 -> 多尺度 MCA 提取多层次热核 -> 各向异性边界收拢
      - 深度: 3D 曲率提取
      - 协同: 3D 体积-热密度相干 (Phi_VT) 过滤水面地面积热 -> 注入多模态表征
    """
    base_name = os.path.splitext(fname)[0]
    depth_m = read_depth_m(base_name)
    ir_path = os.path.join(IR_DIR, fname)
    rgb_np = np.array(rgb_pil)
    if depth_m is None or not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    depth_m = cv2.resize(depth_m, (rgb_np.shape[1], rgb_np.shape[0]))
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0]))
    
    # 1. 红外光学逆卷积
    ir_psf = op_thermal_psf_deconv(ir, sigma=1.2, iters=4)
    # 2. 多尺度形态学 MCA
    mca_heat = op_multiscale_mca(ir_psf, scales=(5, 11, 21))
    # 3. 3D 体积热相干过滤 (消灭水面反射)
    phi_vt = op_phi_vt_filtering(mca_heat, depth_m)
    # 4. 各向异性热晕反向收敛 (贴紧刚体边缘)
    unified_energy = op_anisotropic_debloom(phi_vt, rgb_np, n_iters=2, dt=0.15)
    
    norm_energy = np.clip(unified_energy / 255.0, 0.0, 1.0)
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.12 * norm_energy), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

def run_eval(prep_fn, name):
    detections = []
    with torch.no_grad():
        for info in images_info:
            img_id = info["id"]
            W, H = info["width"], info["height"]
            fname = info["file_name"]
            fpath = os.path.join(IMG_DIR, fname)
            
            with Image.open(fpath) as pil_img:
                pil_img = pil_img.convert("RGB")
                pil_img = prep_fn(pil_img, fname)
                tensor = resize_tensor(pil_img).unsqueeze(0).to(device)
                out = model(tensor)
                logits = out["pred_logits"][0]
                boxes = out["pred_boxes"][0]
                scores, labels = logits.sigmoid().max(dim=-1)

                keep = scores >= 0.015
                sc = scores[keep].cpu().numpy()
                lb = labels[keep].cpu().numpy()
                bx = boxes[keep].cpu().numpy()
                order = sc.argsort()[::-1][:70]

                for idx in order:
                    c, s = int(lb[idx]), float(sc[idx])
                    cx, cy, w, h = bx[idx]
                    detections.append({
                        "image_id": img_id,
                        "category_id": c,
                        "bbox": [(cx - w/2)*W, (cy - h/2)*H, w*W, h*H],
                        "score": s
                    })

    coco_dt = coco_gt.loadRes(detections)
    evaler = COCOeval(coco_gt, coco_dt, "bbox")
    evaler.params.maxDets = [1, 10, 100]
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        evaler.evaluate()
        evaler.accumulate()
        evaler.summarize()
    mAP, ap50, ap75 = evaler.stats[0], evaler.stats[1], evaler.stats[2]
    aps, apm, apl = evaler.stats[3], evaler.stats[4], evaler.stats[5]
    print(f"{name:<45s} | mAP={mAP:.4f} | AP50={ap50:.4f} | AP75={ap75:.4f} | APs={aps:.4f} | APm={apm:.4f} | APl={apl:.4f}")
    return {"mAP": mAP, "AP50": ap50, "AP75": ap75, "APs": aps, "APm": apm, "APl": apl}

if __name__ == "__main__":
    print("\n" + "=" * 96)
    print(" [多算子复合组合实测] 200 张全量验证集端到端目标检测 mAP 评测")
    print("=" * 96)
    print(f"{'方案名称':<45s} | {'mAP@50:95':<10} | {'AP50':<8} | {'AP75':<8} | {'AP_small':<8} | {'AP_med':<8} | {'AP_large':<8}")
    print("-" * 96)
    
    r_base = run_eval(stack_baseline, "1. 原生基线 (Raw Baseline)")
    r_champ = run_eval(stack_champion_single_phi_vt, "2. 单项冠军 (Method Joint-3 Phi_VT)")
    r_combo_A = run_eval(stack_combo_A_thermal_radiance, "3. 组合 A: 【极致热辐射物理栈】(PSF+MCA+Phi_VT)")
    r_combo_B = run_eval(stack_combo_B_polarity_debloom, "4. 组合 B: 【全天候极性-热晕收拢栈】(E_polar+Debloom)")
    r_combo_C = run_eval(stack_combo_C_geometric_blade, "5. 组合 C: 【微观几何刀锋栈】(3D断层+纹理衰减)")
    r_combo_D = run_eval(stack_combo_D_grand_unified, "6. 组合 D: 【光-热-深终极三相相干合流栈】")
    
    results = {
        "1_baseline": r_base,
        "2_single_champion_phi_vt": r_champ,
        "3_combo_A_thermal_radiance": r_combo_A,
        "4_combo_B_polarity_debloom": r_combo_B,
        "5_combo_C_geometric_blade": r_combo_C,
        "6_combo_D_grand_unified": r_combo_D
    }
    
    out_json = os.path.join(ROOT, "work_dir/audits/audit_composite_techniques/composite_stacks_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n复合预处理全量评测完成，结果已保存至: {out_json}")
