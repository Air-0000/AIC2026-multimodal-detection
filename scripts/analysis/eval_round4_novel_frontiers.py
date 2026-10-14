#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_round4_novel_frontiers.py
=============================================================================
第四轮前沿预处理方案探索 (200 张全量验证集端到端评测):
1. Method 1 (热力学): 稳态热传导内部热源反演 (Thermodynamic Heat Generation Inversion, THGI)
2. Method 2 (3D地形): 3D 地形高程归一化场 (Height-Above-Ground DEM, HAG-DEM)
3. Method 3 (图形学): 3D 几何屏幕空间环境遮挡接触阴影 (Depth SSAO / Contact Crevice)
4. Method 4 (微分几何): 局部空间-热辐射结构张量角点相干能 (Structure Tensor Corner Energy, STCE)
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
OUT_DIR = os.path.join(ROOT, "work_dir/audits/audit_round4_techniques")
os.makedirs(OUT_DIR, exist_ok=True)

coco_gt = COCO(ANN_JSON)
images_info = coco_gt.dataset["images"]
print(f"Total valid images: {len(images_info)}")

cfg = YAMLConfig("configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_puremal.yml")
ckpt_path = "outputs/deim_hgnetv2_x_aic_puremal_refine_giou35/best_stg1.pth"
if not os.path.exists(ckpt_path):
    ckpt_path = "outputs/deim_hgnetv2_x_aic_obj2coco_ft_puremal/best_stg2.pth"

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
# 0. 原生基线
# =============================================================================
def prep_baseline(rgb_pil, fname):
    return rgb_pil

# =============================================================================
# 1. Method 1: 热力学内部热源反演 (THGI, Thermodynamic Heat Generation Inversion)
# 物理原理: 稳态热传导微分方程为 ∇·(k∇T) + q_gen = 0。
# 自然被动热背景(路面、土壤、墙壁)由于无内部热源，满足拉普拉斯热平衡 ∇^2 T ≈ 0;
# 人体(代谢37℃)、车辆(内燃机/电池/排气管>80℃)、无人机(电机电调)具有强主动生热率 q_gen > 0,
# 其温度场在局部表现为负拉普拉斯极值 (局部极大凹陷): q_gen ∝ -∇^2 T。
# =============================================================================
def prep_round4_thgi(rgb_pil, fname):
    ir_path = os.path.join(IR_DIR, fname)
    rgb_np = np.array(rgb_pil)
    if not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0])).astype(np.float32)
    
    # 尺度热传导平滑，滤除微小高斯探测器散粒噪声
    ir_smooth = cv2.GaussianBlur(ir, (5, 5), 1.2)
    
    # 计算热通量散度 (负拉普拉斯算子 -∇^2 T)
    lap = cv2.Laplacian(ir_smooth, cv2.CV_32F, ksize=3)
    # 主动内部热源区域: -∇^2 T > 0
    active_heat_gen = np.clip(-lap, 0, 255)
    
    # 热通量自适应归一化 (热边界梯度约束)
    gx = cv2.Sobel(ir_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ir_smooth, cv2.CV_32F, 0, 1, ksize=3)
    heat_flux_norm = np.sqrt(gx**2 + gy**2)
    
    # 内部生热源指标: 既具有中心凸出(-lap > 0), 周围又有强向外热扩散梯度
    thgi_source = np.clip((active_heat_gen * (1.0 + heat_flux_norm / 15.0)) / 40.0, 0.0, 1.0)
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.12 * thgi_source), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

# =============================================================================
# 2. Method 2: 3D 地形高程归一化场 (HAG-DEM, Height-Above-Ground)
# 自动驾驶与野外感知物理原理: 距离 Z 仅仅是传感器径向距离，无法判断物体是否是障碍物。
# 真实障碍物的物理本质是“垂直地表有净高度”。
# 拟合地面流形方程 Y_ground(Z) = a*Z + b，计算每个点相对真实地表的高程净值 HAG。
# 地面点 HAG ≈ 0，车辆/行人 HAG ∈ [0.2m, 2.5m]，低空无人机 HAG > 3m。
# =============================================================================
def prep_round4_hag_dem(rgb_pil, fname):
    base_name = os.path.splitext(fname)[0]
    depth_m = read_depth_m(base_name)
    rgb_np = np.array(rgb_pil)
    if depth_m is None:
        return rgb_pil
    depth_m = cv2.resize(depth_m, (rgb_np.shape[1], rgb_np.shape[0])).astype(np.float32)
    H, W = depth_m.shape
    
    valid = (depth_m > 0.5) & (depth_m < 18.0)
    if valid.sum() < 100:
        return rgb_pil
        
    # 估计相机视场垂直物理坐标 Y (假设垂直视场角约 50度, fy ≈ H / (2 * tan(25 deg)) ≈ 1.07 * H)
    fy = 1.07 * H
    cy = H / 2.0
    v_indices = np.arange(H)[:, None].repeat(W, axis=1).astype(np.float32)
    # 物理 Y 坐标 (向下为正, 即地面在下方)
    Y_m = (v_indices - cy) * depth_m / fy
    
    # 提取底部 30% 区域且有效深度的点拟合地面基底
    ground_band = valid.copy()
    ground_band[:int(H * 0.65), :] = False
    
    if ground_band.sum() > 50:
        z_sample = depth_m[ground_band]
        y_sample = Y_m[ground_band]
        # 鲁棒中位数回归地面 Y = slope * Z + intercept
        # 路面向远方延伸时, Y 随 Z 线性变化
        slope, intercept = np.polyfit(z_sample, y_sample, deg=1)
        # 地面在各点的参考高度
        y_ground_ref = slope * depth_m + intercept
        # 计算相对地面垂直净高度 (单位: 米, 向上为正)
        hag = -(Y_m - y_ground_ref)
        hag[~valid] = 0.0
    else:
        hag = np.zeros_like(depth_m)
        
    # 目标物理高度窗口: 行人与车辆都在 0.2m 到 2.2m 之间
    # 突出具有物理障碍高程的区域
    obstacle_mask = np.clip((hag - 0.15) / 1.5, 0.0, 1.0)
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.10 * obstacle_mask), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

# =============================================================================
# 3. Method 3: 3D 屏幕空间环境遮挡接触阴影 (Depth SSAO / Contact Crevice)
# 图形学原理: 在 3D 几何中，地面接触面(车轮底盘、行人双脚、建筑缝隙)周围具有极大的半球遮蔽率(Ambient Occlusion)。
# 开放平地 AO ≈ 0 (全开阔空间);
# 提取局部深度几何的半球凹陷度 (Crevice Cavity)，精准锁定目标与地面的物理接触根基(Ground Contact Anchor)。
# =============================================================================
def prep_round4_depth_ssao(rgb_pil, fname):
    base_name = os.path.splitext(fname)[0]
    depth_m = read_depth_m(base_name)
    rgb_np = np.array(rgb_pil)
    if depth_m is None:
        return rgb_pil
    depth_m = cv2.resize(depth_m, (rgb_np.shape[1], rgb_np.shape[0])).astype(np.float32)
    
    valid = (depth_m > 0.2) & (depth_m < 19.0)
    
    # 采样 4 个正交方向的深度视差突变凹陷 (屏幕空间法向半球遮挡简化逼近)
    # kernel: [-1, 2, -1] 局部二阶凹度
    k_h = np.array([[-1, 2, -1]], dtype=np.float32)
    k_v = np.array([[-1], [2], [-1]], dtype=np.float32)
    
    concave_h = cv2.filter2D(depth_m, cv2.CV_32F, k_h)
    concave_v = cv2.filter2D(depth_m, cv2.CV_32F, k_v)
    
    # 局部凹陷遮蔽: 深度相比两侧明显更深或更浅产生的接触褶皱
    ao_cavity = np.clip(np.abs(concave_h) + np.abs(concave_v), 0, 2.0)
    ao_cavity[~valid] = 0.0
    
    # 归一化接触锚点得分
    ao_score = np.clip(ao_cavity / 0.8, 0.0, 1.0)
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # 在目标接触根部与轮廓凹槽处提供对比度加固 (提升边界回归质量)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.08 * ao_score), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

# =============================================================================
# 4. Method 4: 局部结构张量角点相干能 (STCE, Structure Tensor Corner Energy)
# 微分几何与信息论原理:
# 结构张量 J = [Ix^2, Ix*Iy; Ix*Iy, Iy^2] 的特征值分解反映局部几何流形:
# - (λ1 >> λ2 ≈ 0): 1D 直线伪影(电线、车道标线、路缘石、光柱)
# - (λ1 ≈ λ2 > 0): 2D 有界紧致结构(行人头部轮廓、车辆4角与底盘角点、无人机机身)
# 抑制 1D 线状杂波干扰，全聚焦 2D 封闭物理目标的角点能量。
# =============================================================================
def prep_round4_stce(rgb_pil, fname):
    ir_path = os.path.join(IR_DIR, fname)
    rgb_np = np.array(rgb_pil)
    if not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0])).astype(np.float32)
    
    # 计算红外梯度
    gx = cv2.Sobel(ir, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ir, cv2.CV_32F, 0, 1, ksize=3)
    
    # 局部结构张量分量平滑
    j_xx = cv2.GaussianBlur(gx**2, (7, 7), 1.5)
    j_yy = cv2.GaussianBlur(gy**2, (7, 7), 1.5)
    j_xy = cv2.GaussianBlur(gx * gy, (7, 7), 1.5)
    
    # 结构张量特征值指标: 行列式与迹
    # det(J) = λ1 * λ2, trace(J) = λ1 + λ2
    det_j = j_xx * j_yy - j_xy**2
    trace_j = j_xx + j_yy
    
    # 经典 Harris-Förstner 2D 紧致角点/斑点能量: det(J) - 0.04 * trace(J)^2
    corner_energy = det_j - 0.04 * (trace_j**2)
    corner_energy = np.clip(corner_energy, 0, None)
    
    # 归一化角点能量
    stce_norm = np.clip(corner_energy / (trace_j + 1e-4) / 25.0, 0.0, 1.0)
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.12 * stce_norm), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

# =============================================================================
# 评测主循环
# =============================================================================
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
    print(f"{name:<42s} | mAP={mAP:.4f} | AP50={ap50:.4f} | AP75={ap75:.4f} | APs={aps:.4f} | APm={apm:.4f} | APl={apl:.4f}")
    return {"mAP": mAP, "AP50": ap50, "AP75": ap75, "APs": aps, "APm": apm, "APl": apl}

if __name__ == "__main__":
    print("\n" + "=" * 98)
    print(" [第四轮前沿探索] 200 张验证集端到端目标检测 mAP 评测 (热力学生热反演/3D地形HAG/SSAO/结构张量)")
    print("=" * 98)
    print(f"{'方案名称':<42s} | {'mAP@50:95':<10} | {'AP50':<8} | {'AP75':<8} | {'AP_small':<8} | {'AP_med':<8} | {'AP_large':<8}")
    print("-" * 98)
    
    r_base = run_eval(prep_baseline, "0. 原生基线 (Raw Baseline)")
    r_thgi = run_eval(prep_round4_thgi, "1. Method 1: 热力学内部热源反演 (THGI)")
    r_hag = run_eval(prep_round4_hag_dem, "2. Method 2: 3D 地形高程归一化场 (HAG-DEM)")
    r_ssao = run_eval(prep_round4_depth_ssao, "3. Method 3: 3D 屏幕空间环境遮挡 (Depth-SSAO)")
    r_stce = run_eval(prep_round4_stce, "4. Method 4: 结构张量角点相干能 (STCE)")
    
    results = {
        "baseline": r_base,
        "method1_thgi": r_thgi,
        "method2_hag_dem": r_hag,
        "method3_depth_ssao": r_ssao,
        "method4_stce": r_stce
    }
    
    out_json = os.path.join(OUT_DIR, "round4_frontier_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n第四轮评测完成，结果已保存至: {out_json}")
