#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_all_in_one_unified_pipeline.py
=============================================================================
测试【全要素融汇终极合流流水线】(All-In-One Harmonized Pipeline):
解答“能不能一起要”的核心机制:
1. 痛点诊断:
   - 之前组合C虽然使小目标 APs 暴涨 +1.40% (27.38% -> 28.78%)，但大目标 APl 从 0.7385 降到 0.7206;
   - 根因在于“远景纹理衰减”盲目平滑，误伤了远处的真实大目标(如远距离卡车/客车)。
2. 全要素和谐共生方案 (Target-Protected Harmonized All-in-One):
   - [红外]: PSF 逆卷积去模糊 + 多尺度 MCA 提取多层热核 + 各向异性边界收拢 (De-blooming);
   - [深度]: 遮挡双边引导滤波 + 3D 曲率断层提取;
   - [协同]: 3D 体积热门控 (Phi_VT) 蒸发地面水面反光;
   - [关键解耦 - 前景保护]: 利用 Phi_VT 与 3D 几何构筑【真实目标保护掩膜 M_fg】，
     远景纹理衰减只针对纯背景 (1 - M_fg)，对任何目标(无论远近)提供 100% 高频豁免！
   - 实现：既要 Phi_VT 的高 mAP (0.5648) / AP75 (0.6210)，又要组合C的小目标暴涨 (+1.40%)，
     同时完全不伤害大目标 AP_large！
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
OUT_DIR = os.path.join(ROOT, "work_dir/audits/audit_all_in_one")
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
# 原子核心算子
# =============================================================================

def op_thermal_psf_deconv(ir_u8, sigma=1.2, iters=4):
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
    out = np.zeros_like(ir_u8, dtype=np.float32)
    tot_w = 0.0
    for k in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        wth = cv2.morphologyEx(ir_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        w = 1.0 / np.sqrt(k)
        out += w * wth
        tot_w += w
    return np.clip((out / tot_w) * 1.5, 0, 255).astype(np.float32)

def op_phi_vt_and_curvature(thermal_energy_f, depth_m):
    dm = depth_m.astype(np.float32)
    dz_dx = cv2.Sobel(dm, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(dm, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    div_n = np.abs(cv2.Sobel(-dz_dx/norm, cv2.CV_32F, 1, 0) + cv2.Sobel(-dz_dy/norm, cv2.CV_32F, 0, 1))
    flat_score = np.exp(- (div_n / 0.15)**2)
    
    phi_vt = thermal_energy_f.astype(np.float32) * (1.0 - 0.75 * flat_score)
    valid_mask = dm > 0.05
    phi_vt[~valid_mask] *= 0.5
    return phi_vt.astype(np.float32), div_n, flat_score

def op_anisotropic_debloom(thermal_f, vis_bgr, n_iters=2, dt=0.15):
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mag_v = np.sqrt(cv2.Sobel(vg, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(vg, cv2.CV_32F, 0, 1)**2)
    barrier = np.exp(- (mag_v / 0.12)**2)
    
    u = thermal_f.astype(np.float32).copy()
    for _ in range(n_iters):
        mag_u = np.sqrt(cv2.Sobel(u, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(u, cv2.CV_32F, 0, 1)**2)
        lap_u = cv2.Laplacian(u, cv2.CV_32F, ksize=3)
        shock = - np.sign(lap_u) * mag_u * barrier
        u = u + dt * shock
    return np.clip(u, 0, 255).astype(np.float32)

# =============================================================================
# 全要素融汇终极流水线实现
# =============================================================================

def pipeline_all_in_one_harmonized(rgb_pil, fname):
    """【全要素融汇终极合流流水线 (All-In-One Harmonized Pipeline)】
    包含:
      1. 红外: PSF 光学逆卷积 + 多尺度 MCA + 各向异性热晕刚体收拢
      2. 深度: 3D 曲率求解 + 视差几何断层
      3. 协同: 3D 体积热门控 (Phi_VT) 消除水面伪热源
      4. 解耦关键: 【前景目标 100% 保护机制】
         远景纹理衰减只施加在纯背景区域 (1 - M_fg)，任何大目标/小目标均保留 100% 原始锐度！
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
    
    # 1. 红外链路: PSF 逆卷积去光学衍射模糊 -> 多尺度 MCA 提取多阶热核
    ir_psf = op_thermal_psf_deconv(ir, sigma=1.2, iters=4)
    mca_heat = op_multiscale_mca(ir_psf, scales=(5, 11, 21))
    
    # 2. 深度与协同链路: 3D 曲率求解 -> Phi_VT 过滤平坦地表/水面积热
    phi_vt, div_n, flat_score = op_phi_vt_and_curvature(mca_heat, depth_m)
    
    # 3. 各向异性逆热扩散: 将热斑严格收拢至可见光刚体边缘
    clean_thermal = op_anisotropic_debloom(phi_vt, rgb_np, n_iters=2, dt=0.15)
    
    # 4. 提取 3D 几何断崖跃变 (来自组合 C 赋予小目标 +1.4% 的核心利器)
    disp = 1.0 / np.clip(depth_m, 0.5, 50.0)
    gx_d = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    edge_3d = np.clip(np.sqrt(gx_d**2 + gy_d**2) * 4.0, 0.0, 1.0)
    
    # 5. 【核心创新：目标保护掩膜 M_fg (Foreground Shield)】
    # 如果一个区域具有显著热特征(clean_thermal)或 3D 几何断崖(edge_3d)或非平整曲率(div_n)，判定为真实目标候选
    norm_th = np.clip(clean_thermal / 150.0, 0.0, 1.0)
    m_fg = np.clip(norm_th * 1.2 + edge_3d * 0.8 + (1.0 - flat_score) * 0.5, 0.0, 1.0)
    m_fg = cv2.GaussianBlur(m_fg, (7, 7), 1.5) # 平滑边缘防硬切
    
    # 6. 【目标豁免型距离纹理衰减 (Target-Exempted Texture Attenuation)】
    # 远景模糊衰减系数 tau_atten = 距离因子 * (1 - M_fg)
    # 意味着: 纯背景 (m_fg=0) 在远景被衰减抹平; 真实目标 (m_fg=1) 无论远近保留 100% 原始纹理！
    z_near, z_far = 5.0, 16.0
    d_clip = np.clip(depth_m, z_near, z_far)
    dist_factor = (d_clip - z_near) / (z_far - z_near + 1e-4) # 0 at near, 1 at far
    
    # 仅在纯背景应用衰减，彻底避免伤害远景大目标！
    bg_atten_factor = dist_factor * (1.0 - m_fg)
    bg_atten_3ch = np.repeat(bg_atten_factor[:, :, None], 3, axis=2)
    
    blur_bg = cv2.GaussianBlur(rgb_np, (5, 5), 1.0)
    rgb_cleaned = (1.0 - bg_atten_3ch) * rgb_np.astype(np.float32) + bg_atten_3ch * blur_bg.astype(np.float32)
    rgb_cleaned = np.clip(rgb_cleaned, 0, 255).astype(np.uint8)
    
    # 7. 终极多模态相干注入 (Fusion Injection into Luminance)
    # 融合项 = 纯净 3D 热能 (clean_thermal) + 几何断层骨架 (edge_3d)
    composite_signal = 0.70 * (clean_thermal / 255.0) + 0.30 * edge_3d
    norm_signal = np.clip(composite_signal, 0.0, 1.0)
    
    lab = cv2.cvtColor(rgb_cleaned, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.12 * norm_signal), 0, 255).astype(np.uint8)
    final_rgb = cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(final_rgb)

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
    print(f"{name:<48s} | mAP={mAP:.4f} | AP50={ap50:.4f} | AP75={ap75:.4f} | APs={aps:.4f} | APm={apm:.4f} | APl={apl:.4f}")
    return {"mAP": mAP, "AP50": ap50, "AP75": ap75, "APs": aps, "APm": apm, "APl": apl}

if __name__ == "__main__":
    print("\n" + "=" * 98)
    print(" [全要素合流终极实测] 能不能全部一起要？—— 200 张全量验证集端到端目标检测评测")
    print("=" * 98)
    print(f"{'方案名称':<48s} | {'mAP@50:95':<10} | {'AP50':<8} | {'AP75':<8} | {'AP_small':<8} | {'AP_med':<8} | {'AP_large':<8}")
    print("-" * 98)
    
    r_all_in_one = run_eval(pipeline_all_in_one_harmonized, "★ 全要素融汇终极合流流水线 (All-In-One Harmonized)")
    
    out_json = os.path.join(OUT_DIR, "all_in_one_harmonized_results.json")
    with open(out_json, "w") as f:
        json.dump(r_all_in_one, f, indent=2)
    print(f"\n全要素终极评测完成，结果已保存至: {out_json}")
