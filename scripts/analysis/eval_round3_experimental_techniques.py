#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_round3_experimental_techniques.py
=============================================================================
测试第三轮全新未探索范式预处理方案在 200 张全量验证集上的端到端下游检测 mAP:
方向包含:
  1. Direction 1 (频域交叉谱): 傅里叶交叉谱高频相位锁定 (Fourier Cross-Spectral Phase Modulation, FCSPM)
  2. Direction 2 (仿生神经): 视网膜节细胞 DoG 侧向抑制与自适应增益控制 (Retinal Lateral Inhibition & Gain Control)
  3. Direction 3 (3D几何): 3D 局部平面突出度残差场 (3D Planar Protrusion Residual, PPR)
  4. Direction 4 (跨模态张量): 局部多模态梯度方向余弦相干场 (Local Gradient Orientation Consistency, LGOC)
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
OUT_DIR = os.path.join(ROOT, "work_dir/audits/audit_round3_techniques")
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
# 0. 原生基线
# =============================================================================
def prep_baseline(rgb_pil, fname):
    return rgb_pil

# =============================================================================
# 1. Direction 1: 傅里叶交叉谱相位合成 (Fourier Cross-Spectral Phase Modulation, FCSPM)
# 物理原理: Oppenheim-Lim 傅里叶定理表明相位谱(Phase)决定物体的几何空间轮廓与精确边界,
# 幅值谱(Magnitude)决定总能量与热辐射强度。
# 算法: 将可见光的高频边缘相位注入红外幅值谱, 逆变换后即可获得具有亚像素级光学轮廓的热能图。
# =============================================================================
def prep_dir1_fcspm(rgb_pil, fname):
    ir_path = os.path.join(IR_DIR, fname)
    rgb_np = np.array(rgb_pil)
    if not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0]))
    
    # 提取可见光亮度
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    
    f_ir = np.fft.fft2(ir.astype(np.float32))
    f_l = np.fft.fft2(l_chan.astype(np.float32))
    
    f_ir_shift = np.fft.fftshift(f_ir)
    f_l_shift = np.fft.fftshift(f_l)
    
    mag_ir = np.abs(f_ir_shift)
    phase_ir = np.angle(f_ir_shift)
    phase_l = np.angle(f_l_shift)
    
    # 构建高频权重掩膜 (中心为DC低频, 边缘为高频)
    H, W = ir.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    radius = np.sqrt((x - cx)**2 + (y - cy)**2)
    cutoff = min(H, W) * 0.05
    # 高频相位逐步由可见光引导
    phase_blend_weight = np.clip((radius - cutoff) / (min(H, W) * 0.25 + 1e-4), 0.0, 1.0) * 0.40
    
    # 复合相位
    sin_p = (1.0 - phase_blend_weight) * np.sin(phase_ir) + phase_blend_weight * np.sin(phase_l)
    cos_p = (1.0 - phase_blend_weight) * np.cos(phase_ir) + phase_blend_weight * np.cos(phase_l)
    synth_phase = np.arctan2(sin_p, cos_p)
    
    # 逆变换回空间域
    f_synth_shift = mag_ir * np.exp(1j * synth_phase)
    f_synth = np.fft.ifftshift(f_synth_shift)
    ir_synth = np.real(np.fft.ifft2(f_synth))
    ir_synth = np.clip(ir_synth, 0, 255).astype(np.uint8)
    
    # 注入到亮度通道 (低对比度柔和注入)
    diff = ir_synth.astype(np.float32) - cv2.medianBlur(ir_synth, 31).astype(np.float32)
    diff_pos = np.clip(diff, 0, 255)
    
    l_enh = np.clip(l_chan.astype(np.float32) * (1.0 + 0.12 * (diff_pos / 255.0)), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a_chan, b_chan]), cv2.COLOR_LAB2RGB))

# =============================================================================
# 2. Direction 2: 仿生视网膜神经侧向抑制与局部增益控制 (Retinal DoG & Contrast Gain Control)
# 生物学原理: 模拟脊椎动物视网膜节细胞与水平细胞的前馈侧向抑制网络 (On-Center Off-Surround),
# 完全消除夏日地面/暖墙等大面积弥散热穹顶, 极度敏锐激发紧凑型动物/人/车辆目标。
# =============================================================================
def prep_dir2_retinal_dog(rgb_pil, fname):
    ir_path = os.path.join(IR_DIR, fname)
    rgb_np = np.array(rgb_pil)
    if not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0])).astype(np.float32)
    
    # Center-surround DoG (中心兴奋, 外周强抑制)
    g_center = cv2.GaussianBlur(ir, (5, 5), 1.2)
    g_surround = cv2.GaussianBlur(ir, (25, 25), 6.0)
    dog = g_center - 0.95 * g_surround
    
    # 双极响应 (On-center 兴奋)
    dog_on = np.clip(dog, 0, 255)
    
    # 局部对比度自适应分母归一化 (Divisive Contrast Normalization)
    local_var = cv2.GaussianBlur(dog_on**2, (31, 31), 8.0)
    dog_norm = dog_on / np.sqrt(local_var + 16.0) # 无量纲紧致激发响应
    dog_norm = np.clip(dog_norm / 3.0, 0.0, 1.0) # 压缩到 [0, 1]
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.12 * dog_norm), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

# =============================================================================
# 3. Direction 3: 3D 局部平面突出度残差场 (3D Planar Protrusion Residual, PPR)
# 几何原理: 任何背景物理表面(路面、墙面、水面、天花板)在局部空间都严格满足 3D 平面方程;
# 真实障碍物(人、车、低空悬停无人机)必然具有显著的垂直平面法向位移残差。
# =============================================================================
def prep_dir3_3d_planar_protrusion(rgb_pil, fname):
    base_name = os.path.splitext(fname)[0]
    depth_m = read_depth_m(base_name)
    rgb_np = np.array(rgb_pil)
    if depth_m is None:
        return rgb_pil
    depth_m = cv2.resize(depth_m, (rgb_np.shape[1], rgb_np.shape[0])).astype(np.float32)
    
    # 有效深度掩膜
    valid = (depth_m > 0.1) & (depth_m < 19.5)
    d_clean = depth_m.copy()
    d_clean[~valid] = np.nan
    
    # 局部 3D 平面低通趋势拟合 (大尺度导向/双边逼近背景流形)
    d_smooth = cv2.bilateralFilter(depth_m, d=15, sigmaColor=1.5, sigmaSpace=25)
    
    # 计算点到局部背景流形的垂直法向残差
    dz_dx = cv2.Sobel(d_smooth, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(d_smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag_grad = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    
    # 3D 残差
    residual_3d = np.abs(depth_m - d_smooth) / mag_grad
    residual_3d[~valid] = 0.0
    
    # 过滤微小噪声，提取 > 0.15m (15cm) 的真实立体凸起
    protrusion_score = np.clip((residual_3d - 0.15) / 0.85, 0.0, 1.0)
    
    # 注入到 RGB 边界与暗处立体结构
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.10 * protrusion_score), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

# =============================================================================
# 4. Direction 4: 局部多模态梯度方向余弦相干场 (Local Gradient Orientation Consistency, LGOC)
# 信息论原理: 真正的物理实体边缘在 RGB 和红外波段下具有严格共线的空间切向量 (|cos θ| ≈ 1);
# 单一模态的光影幻觉(如地面车灯倒影、树影、路面反光条)在红外下法向不共线 (|cos θ| ≈ 0)。
# =============================================================================
def prep_dir4_lgoc_consistency(rgb_pil, fname):
    ir_path = os.path.join(IR_DIR, fname)
    rgb_np = np.array(rgb_pil)
    if not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0])).astype(np.float32)
    
    # 灰度与梯度
    gray_rgb = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    gx_rgb = cv2.Sobel(gray_rgb, cv2.CV_32F, 1, 0, ksize=3)
    gy_rgb = cv2.Sobel(gray_rgb, cv2.CV_32F, 0, 1, ksize=3)
    mag_rgb = np.sqrt(gx_rgb**2 + gy_rgb**2)
    
    gx_ir = cv2.Sobel(ir, cv2.CV_32F, 1, 0, ksize=3)
    gy_ir = cv2.Sobel(ir, cv2.CV_32F, 0, 1, ksize=3)
    mag_ir = np.sqrt(gx_ir**2 + gy_ir**2)
    
    # 计算点积与余弦相干度 |cos θ|
    dot = gx_rgb * gx_ir + gy_rgb * gy_ir
    cos_sim = np.abs(dot) / (mag_rgb * mag_ir + 1e-4) # |cos θ| ∈ [0, 1]
    
    # 结合双模态联合显著度 (只有双模态都有一定梯度的边界才赋权)
    joint_salience = np.sqrt(mag_rgb * mag_ir)
    joint_salience = np.clip(joint_salience / 40.0, 0.0, 1.0)
    
    # 相干物理边界图
    lgoc_edge = cos_sim * joint_salience
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.12 * lgoc_edge), 0, 255).astype(np.uint8)
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
    print(" [第三轮前沿探索] 200 张验证集端到端目标检测 mAP 评测 (4大全新未探索物理/神经范式)")
    print("=" * 98)
    print(f"{'方案名称':<42s} | {'mAP@50:95':<10} | {'AP50':<8} | {'AP75':<8} | {'AP_small':<8} | {'AP_med':<8} | {'AP_large':<8}")
    print("-" * 98)
    
    r_base = run_eval(prep_baseline, "0. 原生基线 (Raw Baseline)")
    r_fcspm = run_eval(prep_dir1_fcspm, "1. Dir-1: 傅里叶交叉谱相位合成 (FCSPM)")
    r_retinal = run_eval(prep_dir2_retinal_dog, "2. Dir-2: 仿生视网膜DoG侧向抑制 (Retinal)")
    r_ppr = run_eval(prep_dir3_3d_planar_protrusion, "3. Dir-3: 3D局部平面凸起残差 (3D-PPR)")
    r_lgoc = run_eval(prep_dir4_lgoc_consistency, "4. Dir-4: 梯度方向余弦相干场 (LGOC)")
    
    results = {
        "baseline": r_base,
        "direction1_fcspm": r_fcspm,
        "direction2_retinal_dog": r_retinal,
        "direction3_3d_ppr": r_ppr,
        "direction4_lgoc": r_lgoc
    }
    
    out_json = os.path.join(OUT_DIR, "round3_experimental_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n第三轮评测完成，结果已保存至: {out_json}")
