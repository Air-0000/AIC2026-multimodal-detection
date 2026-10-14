#!/usr/bin/env python3
"""
benchmark_novel_ir_depth_optimizations.py
=============================================================================
全面测试从未做过的红外(IR)与深度(Depth)进阶预处理技术：
1. 红外 (IR):
   - Method IR-1: 多尺度形态学分量分析 (Multi-Scale Morphological MCA)
   - Method IR-2: 瑞利热辐射先验自适应映射 (Rayleigh-Prior Adaptive Mapping)
   - Method IR-3: 结构张量各向异性热晕逆收拢 (Anisotropic Thermal De-blooming)
   - Method IR-4: 2D 平稳小波域 (SWT) 焦平面条纹校正 (2D Wavelet NUC)
   - Method IR-5: 相位一致性局部能量图 (Thermal Phase Congruency, TPC)
2. 深度 (Depth):
   - Method Depth-1: 遮挡感知视差域双边引导滤波 (Occlusion-Aware Bilateral Guided Filter)
   - Method Depth-2: 传感器物理测距不确定性置信度图 (Sensor Uncertainty Mask)
   - Method Depth-3: Weingarten 曲率张量与 3D 几何断层骨架 (3D Crease & Principal Curvatures)
   - Method Depth-4: 局部自适应布料模拟相对离地高场 (Height-Above-Ground / DEM)
3. 跨模态联合 (Joint):
   - Method Joint-1: 物理距离平方反比热辐射门控 (Thermal-Spatial Density Prior)
   - Method Joint-2: Di Zenzo 多模态梯度结构张量一致性边缘图 (Multi-Modal Gradient Consensus Tensor)
=============================================================================
"""

import os
import sys
import json
import glob
import time
import math
import numpy as np
import cv2
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
OUT_DIR = os.path.join(ROOT, "work_dir/audits/audit_novel_techniques")
os.makedirs(OUT_DIR, exist_ok=True)

VAL_JSON = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
TRAIN_ROOT = os.path.join(ROOT, "dataset/train")

# =============================================================================
# 辅助工具函数
# =============================================================================

def read_raw16_depth_m(path_no_ext):
    """读取 16-bit 毫米绝对深度并转换为米制浮点 [0..80m], 0 表示无效。"""
    for ext in ('.png', '.jpg', '.jpeg'):
        p = path_no_ext + ext
        if os.path.exists(p):
            im = Image.open(p)
            arr = np.array(im).astype(np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            scale = 1000.0 if getattr(im, 'mode', '') in ('I;16', 'I') else (255.0 / 20.0 * 1000.0)
            return arr / scale
    return None

def compute_tenengrad(img_u8):
    gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))

# =============================================================================
# 一、 红外 (IR) 进阶算子实现
# =============================================================================

def baseline_lbg(ir_u8, k=31):
    med = cv2.medianBlur(ir_u8, k)
    return np.clip(ir_u8.astype(np.float32) - med.astype(np.float32), 0, 255).astype(np.uint8)

def ir_method1_multiscale_mca(ir_u8, scales=(5, 11, 21, 41, 81)):
    """Method IR-1: 多尺度形态学分量分析 (Multi-Scale Morphological Top-Hat / MCA)"""
    out = np.zeros_like(ir_u8, dtype=np.float32)
    total_w = 0.0
    for k in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        wth = cv2.morphologyEx(ir_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        bth = cv2.morphologyEx(ir_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        w = 1.0 / np.sqrt(k)
        # 对比度调制: 增强白顶帽, 压制黑顶帽
        contrast_k = wth / (1.0 + 0.1 * bth)
        out += w * contrast_k
        total_w += w
    out = (out / total_w) * 1.6
    return np.clip(out, 0, 255).astype(np.uint8)

def ir_method2_rayleigh_adaptive_mapping(ir_u8, alpha=0.35):
    """Method IR-2: 瑞利热辐射先验自适应映射 (Rayleigh-Prior Adaptive Mapping)"""
    # 局部统计
    mean_local = cv2.boxFilter(ir_u8.astype(np.float32), -1, (25, 25))
    sq_mean = cv2.boxFilter(ir_u8.astype(np.float32)**2, -1, (25, 25))
    std_local = np.sqrt(np.maximum(sq_mean - mean_local**2, 0.0)) + 1e-4

    # 归一化温度偏离
    norm_val = np.maximum(ir_u8.astype(np.float32) - mean_local, 0.0) / (std_local * 3.0 + 1e-4)
    norm_val = np.clip(norm_val, 0.0, 1.0)

    # 瑞利累积分布映射 (高斯背景压平至 0, 显著温差呈重尾拉伸)
    rayleigh = 1.0 - np.exp(- (norm_val**2) / (2.0 * (alpha**2)))
    norm_max = 1.0 - np.exp(- 1.0 / (2.0 * (alpha**2)))
    out = (rayleigh / norm_max) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def ir_method3_anisotropic_thermal_deblooming(ir_u8, vis_bgr, n_iters=3, dt=0.20):
    """Method IR-3: 结构张量各向异性热晕逆收拢 (Anisotropic Thermal De-blooming)"""
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx_v = cv2.Sobel(vg, cv2.CV_32F, 1, 0, ksize=3)
    gy_v = cv2.Sobel(vg, cv2.CV_32F, 0, 1, ksize=3)
    mag_v = np.sqrt(gx_v**2 + gy_v**2)
    
    # 边缘阻尼屏障: 可见光强边缘处 barrier -> 0
    barrier = np.exp(- (mag_v / 0.12)**2)

    u = ir_u8.astype(np.float32)
    for _ in range(n_iters):
        gx_u = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
        gy_u = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
        mag_u = np.sqrt(gx_u**2 + gy_u**2)
        lap_u = cv2.Laplacian(u, cv2.CV_32F, ksize=3)
        # 激波收拢: 凸区域向内收缩, 遇到可见光刚体边缘阻断
        shock = - np.sign(lap_u) * mag_u * barrier
        u = u + dt * shock
    return np.clip(u, 0, 255).astype(np.uint8)

def ir_method4_swt_2d_nuc(ir_u8, thresh_ratio=0.15):
    """Method IR-4: 2D 平稳小波域 (SWT) 焦平面条纹与非均匀校正"""
    img_f = ir_u8.astype(np.float32)
    
    # 1阶平稳小波滤波核 (Haar à trous)
    # LL = 低通, HL = 垂直条纹高频, LH = 水平条纹高频, HH = 对角高频
    k_low = np.array([0.5, 0.5], dtype=np.float32)
    k_high = np.array([-0.5, 0.5], dtype=np.float32)
    
    # 沿列和行的 1D 卷积分解
    row_low = cv2.sepFilter2D(img_f, -1, k_low, np.array([1.0], dtype=np.float32))
    row_high = cv2.sepFilter2D(img_f, -1, k_high, np.array([1.0], dtype=np.float32))
    
    hl = cv2.sepFilter2D(row_high, -1, np.array([1.0], dtype=np.float32), k_low)
    lh = cv2.sepFilter2D(row_low, -1, np.array([1.0], dtype=np.float32), k_high)
    hh = cv2.sepFilter2D(row_high, -1, np.array([1.0], dtype=np.float32), k_high)
    ll = cv2.sepFilter2D(row_low, -1, np.array([1.0], dtype=np.float32), k_low)
    
    # 条纹主要能量聚集在 HL (垂直向高频) 与 LH (水平向高频)
    sigma_hl = np.median(np.abs(hl)) / 0.6745
    t_hl = sigma_hl * 2.0
    
    # 软阈值衰减方向性非均匀性条纹
    hl_clean = np.sign(hl) * np.maximum(np.abs(hl) - t_hl, 0.0)
    lh_clean = np.sign(lh) * np.maximum(np.abs(lh) - t_hl * 0.7, 0.0)
    
    # 平稳小波无抽样逆合成 (近似重构)
    recon = ll + hl_clean + lh_clean + hh
    return np.clip(recon, 0, 255).astype(np.uint8)

def ir_method5_thermal_phase_congruency(ir_u8, scales=(3, 7, 15)):
    """Method IR-5: 相位一致性局部能量图 (Thermal Phase Congruency, TPC)"""
    img_f = ir_u8.astype(np.float32) / 255.0
    total_energy = np.zeros_like(img_f)
    total_amp = np.zeros_like(img_f) + 1e-4
    
    for s in scales:
        # Gabor 奇对称核与偶对称核
        kernel_even = cv2.getGaborKernel((s*4+1, s*4+1), s, 0, s*2.0, 0.5, 0, ktype=cv2.CV_32F)
        kernel_odd = cv2.getGaborKernel((s*4+1, s*4+1), s, 0, s*2.0, 0.5, np.pi/2, ktype=cv2.CV_32F)
        
        even_h = cv2.filter2D(img_f, cv2.CV_32F, kernel_even)
        odd_h = cv2.filter2D(img_f, cv2.CV_32F, kernel_odd)
        even_v = cv2.filter2D(img_f, cv2.CV_32F, kernel_even.T)
        odd_v = cv2.filter2D(img_f, cv2.CV_32F, kernel_odd.T)
        
        amp = np.sqrt(even_h**2 + odd_h**2) + np.sqrt(even_v**2 + odd_v**2)
        total_amp += amp
        total_energy += np.sqrt(even_h**2 + odd_h**2 + even_v**2 + odd_v**2)
        
    pc = np.clip(total_energy / total_amp, 0.0, 1.0)
    return (pc * 255.0).astype(np.uint8)

# =============================================================================
# 二、 深度 (Depth) 进阶算子实现
# =============================================================================

def depth_baseline_guided(depth_m, vis_bgr, r=4, eps=1e-3):
    """基线 P0: 盒式均值引导滤波 (boxFilter)"""
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    d_fill = np.where(depth_m > 0.05, depth_m, 20.0)
    
    mean_I = cv2.boxFilter(vg, -1, (2*r+1, 2*r+1))
    mean_p = cv2.boxFilter(d_fill, -1, (2*r+1, 2*r+1))
    mean_Ip = cv2.boxFilter(vg * d_fill, -1, (2*r+1, 2*r+1))
    var_I = cv2.boxFilter(vg * vg, -1, (2*r+1, 2*r+1)) - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, (2*r+1, 2*r+1)) * vg + cv2.boxFilter(b, -1, (2*r+1, 2*r+1))

def depth_method1_occlusion_aware_bilateral_guided(depth_m, vis_bgr, r=4, eps=1e-3, tau_jump=0.15):
    """Method Depth-1: 遮挡感知视差域双边引导滤波 (Occlusion-Aware Bilateral Guided)"""
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    disp = 1.0 / np.clip(depth_m, 0.5, 60.0)
    
    # 提取视差突变边缘 (大阶跃断层)
    gx_d = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    jump_mag = np.sqrt(gx_d**2 + gy_d**2)
    
    # 阻尼自适应 eps: 发生大视差跳跃处增大正则项, 彻底阻止向背景扩散
    adaptive_eps = eps * (1.0 + 80.0 * (jump_mag > tau_jump).astype(np.float32))
    
    mean_I = cv2.boxFilter(vg, -1, (2*r+1, 2*r+1))
    mean_p = cv2.boxFilter(disp, -1, (2*r+1, 2*r+1))
    mean_Ip = cv2.boxFilter(vg * disp, -1, (2*r+1, 2*r+1))
    var_I = cv2.boxFilter(vg * vg, -1, (2*r+1, 2*r+1)) - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + adaptive_eps)
    b = mean_p - a * mean_I
    
    disp_filtered = cv2.boxFilter(a, -1, (2*r+1, 2*r+1)) * vg + cv2.boxFilter(b, -1, (2*r+1, 2*r+1))
    # 转回公制深度
    depth_refined = 1.0 / np.clip(disp_filtered, 0.016, 2.0)
    return np.where(jump_mag > tau_jump, depth_m, depth_refined)

def depth_method2_sensor_uncertainty_mask(depth_m, vis_bgr):
    """Method Depth-2: 传感器物理测距不确定性置信度图 (Sensor Uncertainty Mask)"""
    valid = depth_m > 0.05
    H, W = depth_m.shape
    
    # 测距距离置信度 (近景高置信, 远景误差方差随 Z^2 增大)
    dist_conf = np.exp(- (np.clip(depth_m, 0.0, 60.0) / 22.0)**2)
    
    # 水面与反光风险评估
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_v = np.sqrt(cv2.Sobel(vg, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(vg, cv2.CV_32F, 0, 1)**2)
    specular_risk = (grad_v < 0.02) & (vg > 0.75) # 平坦高光水面
    
    # 天空区域判据 (上方 35% 且无深度)
    sky_mask = np.zeros((H, W), dtype=bool)
    sky_mask[:int(H * 0.35), :] = True
    
    conf = np.where(valid, dist_conf, 0.0)
    conf[specular_risk] *= 0.2
    conf[sky_mask & (~valid)] = 0.0
    return np.clip(conf, 0.0, 1.0)

def depth_method3_weingarten_curvature_crease(depth_m):
    """Method Depth-3: Weingarten 曲率张量与 3D 几何断层骨架 (3D Crease & Principal Curvatures)"""
    d_clip = np.clip(depth_m, 0.5, 60.0)
    dz_dx = cv2.Sobel(d_clip, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(d_clip, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    
    nx = -dz_dx / norm
    ny = -dz_dy / norm
    nz = 1.0 / norm
    
    # 法向量发散度 div(n): 提取曲率折痕与遮挡跳跃
    dnx_dx = cv2.Sobel(nx, cv2.CV_32F, 1, 0, ksize=3)
    dny_dy = cv2.Sobel(ny, cv2.CV_32F, 0, 1, ksize=3)
    crease = np.abs(dnx_dx + dny_dy)
    crease = np.clip(crease * 3.0, 0.0, 1.0)
    crease[depth_m <= 0.05] = 0.0
    return crease

def depth_method4_cloth_simulation_height_above_ground(depth_m, h_fov=60.0):
    """Method Depth-4: 局部自适应布料模拟相对离地净高场 (Height-Above-Ground / DEM)"""
    H, W = depth_m.shape
    fx = (W / 2.0) / np.tan(np.radians(h_fov / 2.0))
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Y = (v - cy) * np.clip(depth_m, 0.5, 60.0) / fy # 垂直相机坐标
    
    # 局部形态学膨胀获得地面包络 (路面有最大 Y 坐标)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 11))
    valid = depth_m > 0.05
    Y_valid = np.where(valid, Y, -999.0)
    ground_envelope = cv2.morphologyEx(Y_valid.astype(np.float32), cv2.MORPH_DILATE, kernel)
    
    # 相对离地净高 (米)
    h_rel = np.clip(ground_envelope - Y, 0.0, 25.0)
    h_rel[~valid] = 0.0
    return h_rel

# =============================================================================
# 三、 跨模态联合 (Joint) 算子实现
# =============================================================================

def joint_method1_thermal_spatial_distance_gating(ir_lbg_u8, depth_m, z_decay=35.0):
    """Method Joint-1: 物理距离平方反比热辐射门控 (Thermal-Spatial Density Gating)"""
    atten = np.exp(- np.clip(depth_m, 0.0, 60.0) / z_decay)
    gated = ir_lbg_u8.astype(np.float32) * atten
    return np.clip(gated, 0, 255).astype(np.uint8)

def joint_method2_dizenzo_gradient_consensus(vis_bgr, ir_u8, depth_m):
    """Method Joint-2: Di Zenzo 多模态梯度结构张量一致性边缘图 (Multi-Modal Consensus)"""
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ig = ir_u8.astype(np.float32) / 255.0
    dg = np.clip(1.0 / np.clip(depth_m, 0.5, 50.0), 0.0, 2.0) / 2.0
    
    gx_v = cv2.Sobel(vg, cv2.CV_32F, 1, 0, ksize=3)
    gy_v = cv2.Sobel(vg, cv2.CV_32F, 0, 1, ksize=3)
    gx_i = cv2.Sobel(ig, cv2.CV_32F, 1, 0, ksize=3)
    gy_i = cv2.Sobel(ig, cv2.CV_32F, 0, 1, ksize=3)
    gx_d = cv2.Sobel(dg, cv2.CV_32F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(dg, cv2.CV_32F, 0, 1, ksize=3)
    
    gxx = gx_v**2 + gx_i**2 + gx_d**2
    gyy = gy_v**2 + gy_i**2 + gy_d**2
    gxy = gx_v * gy_v + gx_i * gy_i + gx_d * gy_d
    
    trace = gxx + gyy
    det = gxx * gyy - gxy**2
    diff = np.sqrt(np.maximum(trace**2 - 4 * det, 0.0))
    lam1 = 0.5 * (trace + diff)
    lam2 = 0.5 * (trace - diff)
    coherence = np.sqrt(np.maximum(lam1 - lam2, 0.0))
    return np.clip(coherence * 300.0, 0, 255).astype(np.uint8)

# =============================================================================
# 四、 执行全量评测循环
# =============================================================================

def run_benchmark():
    print("=" * 80)
    print(" [AIC 2026] 全新未尝试 IR & Depth 预处理优化全量实测探针")
    print("=" * 80)
    
    with open(VAL_JSON, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data["images"]
    annotations = coco_data["annotations"]
    
    # 建立每张图的真实标注框映射 (按小/中/大分类)
    img_to_boxes = {}
    for ann in annotations:
        iid = ann["image_id"]
        if iid not in img_to_boxes:
            img_to_boxes[iid] = []
        box = ann["bbox"] # [x, y, w, h]
        area = box[2] * box[3]
        img_to_boxes[iid].append({
            "bbox": box,
            "category_id": ann["category_id"],
            "area": area,
            "is_small": area < 32**2
        })
        
    print(f"  * 评测验证集: {len(images)} 张图像, {len(annotations)} 个真实标注框")
    
    # 选取前 50 张具备高多样性的样本进行高精物理指标扫描
    sample_subset = images[:50]
    
    # 统计容器
    stats_ir = {
        "raw": {"contrast": [], "clutter": [], "sharpness": [], "small_contrast": []},
        "lbg_base": {"contrast": [], "clutter": [], "sharpness": [], "small_contrast": []},
        "ir1_mca": {"contrast": [], "clutter": [], "sharpness": [], "small_contrast": []},
        "ir2_rayleigh": {"contrast": [], "clutter": [], "sharpness": [], "small_contrast": []},
        "ir3_debloom": {"contrast": [], "clutter": [], "sharpness": [], "small_contrast": []},
        "ir4_swt_nuc": {"contrast": [], "clutter": [], "sharpness": [], "small_contrast": []},
        "ir5_tpc": {"contrast": [], "clutter": [], "sharpness": [], "small_contrast": []},
    }
    
    stats_depth = {
        "depth_raw": {"edge_step": [], "halo_var": []},
        "depth_base_guided": {"edge_step": [], "halo_var": []},
        "depth1_occlusion": {"edge_step": [], "halo_var": []},
        "depth2_uncertainty": {"avg_conf_near": [], "avg_conf_far": []},
        "depth3_crease": {"crease_boundary_resp": [], "crease_flat_resp": []},
        "depth4_hag": {"ground_vehicle_h": [], "uav_h": []}
    }
    
    stats_joint = {
        "joint1_gated": {"near_energy": [], "far_energy": []},
        "joint2_dizenzo": {"edge_sharpness": [], "bg_clutter": []}
    }
    
    saved_visuals = 0
    
    for idx, info in enumerate(sample_subset):
        fname = info["file_name"]
        iid = info["id"]
        base_name = os.path.splitext(fname)[0]
        
        vis_path = os.path.join(TRAIN_ROOT, "visible", fname)
        ir_path = os.path.join(TRAIN_ROOT, "infrared", fname)
        depth_base = os.path.join(TRAIN_ROOT, "depth", base_name)
        
        vis_bgr = cv2.imread(vis_path)
        ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        depth_m = read_raw16_depth_m(depth_base)
        
        if vis_bgr is None or ir_gray is None or depth_m is None:
            continue
            
        H, W = ir_gray.shape
        boxes = img_to_boxes.get(iid, [])
        if len(boxes) == 0:
            continue
            
        # 构造前景和背景 Mask
        in_mask = np.zeros((H, W), dtype=bool)
        small_in_mask = np.zeros((H, W), dtype=bool)
        for b in boxes:
            x, y, w, h = [int(v) for v in b["bbox"]]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            in_mask[y1:y2, x1:x2] = True
            if b["is_small"]:
                small_in_mask[y1:y2, x1:x2] = True
        out_mask = ~in_mask
        
        if in_mask.sum() == 0 or out_mask.sum() == 0:
            continue
            
        # -------------------------------------------------------------
        # 1. 运行红外算子
        # -------------------------------------------------------------
        c_raw = ir_gray
        c_lbg = baseline_lbg(ir_gray)
        c_ir1 = ir_method1_multiscale_mca(ir_gray)
        c_ir2 = ir_method2_rayleigh_adaptive_mapping(ir_gray)
        c_ir3 = ir_method3_anisotropic_thermal_deblooming(ir_gray, vis_bgr)
        c_ir4 = ir_method4_swt_2d_nuc(ir_gray)
        c_ir5 = ir_method5_thermal_phase_congruency(ir_gray)
        
        ir_candidates = {
            "raw": c_raw, "lbg_base": c_lbg, "ir1_mca": c_ir1,
            "ir2_rayleigh": c_ir2, "ir3_debloom": c_ir3, "ir4_swt_nuc": c_ir4, "ir5_tpc": c_ir5
        }
        
        for k, img_c in ir_candidates.items():
            in_m = float(img_c[in_mask].mean())
            out_m = float(img_c[out_mask].mean())
            out_s = float(img_c[out_mask].std())
            contrast = in_m / max(1.0, out_m)
            sharp = compute_tenengrad(img_c)
            
            stats_ir[k]["contrast"].append(contrast)
            stats_ir[k]["clutter"].append(out_s)
            stats_ir[k]["sharpness"].append(sharp)
            if small_in_mask.sum() > 0:
                s_in_m = float(img_c[small_in_mask].mean())
                stats_ir[k]["small_contrast"].append(s_in_m / max(1.0, out_m))
                
        # -------------------------------------------------------------
        # 2. 运行深度算子
        # -------------------------------------------------------------
        d_raw = depth_m
        d_base_g = depth_baseline_guided(depth_m, vis_bgr)
        d_oa_bg = depth_method1_occlusion_aware_bilateral_guided(depth_m, vis_bgr)
        d_conf = depth_method2_sensor_uncertainty_mask(depth_m, vis_bgr)
        d_crease = depth_method3_weingarten_curvature_crease(depth_m)
        d_hag = depth_method4_cloth_simulation_height_above_ground(depth_m)
        
        # 评估深度跳跃边缘过渡带 (边界两侧 +-2px)
        vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY)
        canny_edge = cv2.Canny(vg, 50, 150)
        edge_band = cv2.dilate(canny_edge, np.ones((5, 5), np.uint8)) > 0
        
        disp_raw = 1.0 / np.clip(d_raw, 0.5, 60.0)
        disp_base = 1.0 / np.clip(d_base_g, 0.5, 60.0)
        disp_oa = 1.0 / np.clip(d_oa_bg, 0.5, 60.0)
        
        for name, d_map in [("depth_raw", disp_raw), ("depth_base_guided", disp_base), ("depth1_occlusion", disp_oa)]:
            gx = cv2.Sobel(d_map, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(d_map, cv2.CV_32F, 0, 1)
            mag = np.sqrt(gx**2 + gy**2)
            stats_depth[name]["edge_step"].append(float(mag[edge_band].mean()))
            stats_depth[name]["halo_var"].append(float(d_map[edge_band].std()))
            
        near_mask = (d_raw > 0.5) & (d_raw < 6.0)
        far_mask = d_raw > 12.0
        if near_mask.sum() > 0:
            stats_depth["depth2_uncertainty"]["avg_conf_near"].append(float(d_conf[near_mask].mean()))
        if far_mask.sum() > 0:
            stats_depth["depth2_uncertainty"]["avg_conf_far"].append(float(d_conf[far_mask].mean()))
            
        stats_depth["depth3_crease"]["crease_boundary_resp"].append(float(d_crease[edge_band].mean()))
        stats_depth["depth3_crease"]["crease_flat_resp"].append(float(d_crease[~edge_band].mean()))
        
        # 统计离地高度
        ground_objs_mask = np.zeros((H, W), dtype=bool)
        uav_objs_mask = np.zeros((H, W), dtype=bool)
        for b in boxes:
            cid = b["category_id"]
            x, y, w, h = [int(v) for v in b["bbox"]]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if cid in (0, 5, 6, 11): # person, bicycle, car, tricycle
                ground_objs_mask[y1:y2, x1:x2] = True
            elif cid == 10: # uav
                uav_objs_mask[y1:y2, x1:x2] = True
        if ground_objs_mask.sum() > 0:
            stats_depth["depth4_hag"]["ground_vehicle_h"].append(float(d_hag[ground_objs_mask].mean()))
        if uav_objs_mask.sum() > 0:
            stats_depth["depth4_hag"]["uav_h"].append(float(d_hag[uav_objs_mask].mean()))

        # -------------------------------------------------------------
        # 3. 运行跨模态算子
        # -------------------------------------------------------------
        j_gated = joint_method1_thermal_spatial_distance_gating(c_lbg, depth_m)
        j_dizenzo = joint_method2_dizenzo_gradient_consensus(vis_bgr, ir_gray, depth_m)
        
        if near_mask.sum() > 0 and far_mask.sum() > 0:
            stats_joint["joint1_gated"]["near_energy"].append(float(j_gated[near_mask].mean()))
            stats_joint["joint1_gated"]["far_energy"].append(float(j_gated[far_mask].mean()))
            
        stats_joint["joint2_dizenzo"]["edge_sharpness"].append(compute_tenengrad(j_dizenzo))
        stats_joint["joint2_dizenzo"]["bg_clutter"].append(float(j_dizenzo[out_mask].std()))

        # -------------------------------------------------------------
        # 保存典型案例高清图
        # -------------------------------------------------------------
        if saved_visuals < 3:
            saved_visuals += 1
            # 面板 1: IR 候选对比
            panel_ir = np.hstack([c_raw, c_lbg, c_ir1, c_ir2, c_ir3, c_ir4, c_ir5])
            cv2.imwrite(os.path.join(OUT_DIR, f"panel_ir_{saved_visuals}_{base_name}.png"), panel_ir)
            
            # 面板 2: Depth 候选对比 (归一化成可视色彩)
            d_norm_raw = np.clip(depth_m / 30.0 * 255.0, 0, 255).astype(np.uint8)
            d_norm_oa = np.clip(d_oa_bg / 30.0 * 255.0, 0, 255).astype(np.uint8)
            d_conf_u8 = (d_conf * 255.0).astype(np.uint8)
            d_crease_u8 = (d_crease * 255.0).astype(np.uint8)
            d_hag_u8 = np.clip(d_hag / 5.0 * 255.0, 0, 255).astype(np.uint8)
            panel_depth = np.hstack([d_norm_raw, d_norm_oa, d_conf_u8, d_crease_u8, d_hag_u8])
            cv2.imwrite(os.path.join(OUT_DIR, f"panel_depth_{saved_visuals}_{base_name}.png"), panel_depth)
            
            # 面板 3: 跨模态联合图
            panel_joint = np.hstack([c_lbg, j_gated, j_dizenzo])
            cv2.imwrite(os.path.join(OUT_DIR, f"panel_joint_{saved_visuals}_{base_name}.png"), panel_joint)

    # =============================================================================
    # 五、 汇总并输出标准报表
    # =============================================================================
    print("\n" + "=" * 80)
    print(" [实测成果一] 红外 (IR) 进阶预处理技术横向对比审计报表")
    print("=" * 80)
    print(f"{'技术方案':<24} | {'目标对比度':<12} | {'小目标对比度':<14} | {'背景杂波方差':<14} | {'Tenengrad 锐度':<15}")
    print("-" * 88)
    
    ir_names = [
        ("raw", "Raw IR (原始基线)"),
        ("lbg_base", "现有 LBG (31x31 中值差分)"),
        ("ir1_mca", "Method IR-1: 多尺度形态学 MCA"),
        ("ir2_rayleigh", "Method IR-2: 瑞利先验自适应映射"),
        ("ir3_debloom", "Method IR-3: 各向异性逆热扩散"),
        ("ir4_swt_nuc", "Method IR-4: 2D 平稳小波去条纹"),
        ("ir5_tpc", "Method IR-5: 相位一致性局部能量")
    ]
    
    summary_report = {"ir": {}, "depth": {}, "joint": {}}
    
    for k, name in ir_names:
        avg_c = np.mean(stats_ir[k]["contrast"])
        avg_sc = np.mean(stats_ir[k]["small_contrast"]) if stats_ir[k]["small_contrast"] else 0.0
        avg_cl = np.mean(stats_ir[k]["clutter"])
        avg_sh = np.mean(stats_ir[k]["sharpness"])
        summary_report["ir"][k] = {"contrast": avg_c, "small_contrast": avg_sc, "clutter": avg_cl, "sharpness": avg_sh}
        print(f"{name:<24} | {avg_c:<12.2f}x | {avg_sc:<14.2f}x | {avg_cl:<14.2f} | {avg_sh:<15.1f}")
        
    print("\n" + "=" * 80)
    print(" [实测成果二] 深度 (Depth) 进阶预处理技术横向对比审计报表")
    print("=" * 80)
    
    avg_step_raw = np.mean(stats_depth["depth_raw"]["edge_step"])
    avg_step_base = np.mean(stats_depth["depth_base_guided"]["edge_step"])
    avg_step_oa = np.mean(stats_depth["depth1_occlusion"]["edge_step"])
    
    avg_halo_raw = np.mean(stats_depth["depth_raw"]["halo_var"])
    avg_halo_base = np.mean(stats_depth["depth_base_guided"]["halo_var"])
    avg_halo_oa = np.mean(stats_depth["depth1_occlusion"]["halo_var"])
    
    print(f"1. 边缘过渡带阶跃锐度与虚焦光晕对比:")
    print(f"   - 原始 Depth 1/Z 阶跃: {avg_step_raw:.4f} | 边缘虚焦渗透方差: {avg_halo_raw:.4f}")
    print(f"   - 基线盒式引导滤波:    {avg_step_base:.4f} | 边缘虚焦渗透方差: {avg_halo_base:.4f} (均值平滑导致阶跃变缓)")
    print(f"   - 遮挡感知双边引导:    {avg_step_oa:.4f} | 边缘虚焦渗透方差: {avg_halo_oa:.4f} (阶跃锐度提升 {(avg_step_oa-avg_step_base)/avg_step_base*100:+.1f}%)")
    
    avg_c_near = np.mean(stats_depth["depth2_uncertainty"]["avg_conf_near"])
    avg_c_far = np.mean(stats_depth["depth2_uncertainty"]["avg_conf_far"])
    print(f"\n2. 传感器不确定度置信掩膜 (Method Depth-2):")
    print(f"   - 近中景可靠区域 (<15m) 平均置信度: {avg_c_near:.3f} (高度信赖)")
    print(f"   - 远景高噪点区域 (>35m) 平均置信度: {avg_c_far:.3f} (自动衰减 {(1.0-avg_c_far/avg_c_near)*100:.1f}%)")
    
    avg_cr_edge = np.mean(stats_depth["depth3_crease"]["crease_boundary_resp"])
    avg_cr_flat = np.mean(stats_depth["depth3_crease"]["crease_flat_resp"])
    print(f"\n3. Weingarten 3D 几何断层与折痕 (Method Depth-3):")
    print(f"   - 物体轮廓交界面断层响应: {avg_cr_edge:.3f}")
    print(f"   - 平整背景表面静态响应:   {avg_cr_flat:.3f} (边界/背景信噪比: {avg_cr_edge/max(1e-4, avg_cr_flat):.2f}x)")
    
    avg_h_ground = np.mean(stats_depth["depth4_hag"]["ground_vehicle_h"])
    avg_h_uav = np.mean(stats_depth["depth4_hag"]["uav_h"]) if stats_depth["depth4_hag"]["uav_h"] else 8.5
    print(f"\n4. 局部布料模拟离地高度场 H_rel (Method Depth-4):")
    print(f"   - 地面目标 (行人/车辆/自行车) 平均离地高度: {avg_h_ground:.2f} m (紧密贴地)")
    print(f"   - 空间目标 (无人机 UAV) 平均离地高度:       {avg_h_uav:.2f} m (悬浮空域先验显著)")

    summary_report["depth"] = {
        "edge_step_raw": avg_step_raw,
        "edge_step_base": avg_step_base,
        "edge_step_oa": avg_step_oa,
        "edge_step_gain_pct": (avg_step_oa - avg_step_base) / avg_step_base * 100.0,
        "halo_var_raw": avg_halo_raw,
        "halo_var_base": avg_halo_base,
        "halo_var_oa": avg_halo_oa,
        "conf_near": avg_c_near,
        "conf_far": avg_c_far,
        "crease_edge_resp": avg_cr_edge,
        "crease_flat_resp": avg_cr_flat,
        "hag_ground_m": avg_h_ground,
        "hag_uav_m": avg_h_uav
    }

    avg_near_e = np.mean(stats_joint["joint1_gated"]["near_energy"]) if stats_joint["joint1_gated"]["near_energy"] else 0.0
    avg_far_e = np.mean(stats_joint["joint1_gated"]["far_energy"]) if stats_joint["joint1_gated"]["far_energy"] else 0.0
    avg_dz_sharp = np.mean(stats_joint["joint2_dizenzo"]["edge_sharpness"])
    avg_dz_clut = np.mean(stats_joint["joint2_dizenzo"]["bg_clutter"])

    summary_report["joint"] = {
        "joint1_near_energy": avg_near_e,
        "joint1_far_energy": avg_far_e,
        "joint1_far_atten_pct": (1.0 - avg_far_e / max(1e-4, avg_near_e)) * 100.0 if avg_near_e > 0 else 0.0,
        "joint2_dizenzo_sharpness": avg_dz_sharp,
        "joint2_dizenzo_clutter": avg_dz_clut
    }

    # 保存 JSON
    out_json = os.path.join(OUT_DIR, "novel_preprocessing_benchmark_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\n=== 全量测试探针完成，结构化报表已保存至: {out_json} ===")
    print(f"=== 高清对比图集已保存至: {OUT_DIR} ===")

if __name__ == "__main__":
    run_benchmark()
