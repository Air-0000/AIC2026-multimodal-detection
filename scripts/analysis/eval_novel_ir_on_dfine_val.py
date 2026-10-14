#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_novel_ir_on_dfine_val.py
测试全新预处理算子在 200 张真实验证集上的下游检测 mAP 变化:
对比:
  1. 原生基线 (Baseline)
  2. 逆热扩散收拢增强 (Anisotropic Thermal De-blooming)
  3. 多尺度形态学 MCA 显著性注入 (Multi-Scale MCA Saliency Injection)
  4. 瑞利分布热辐射映射 (Rayleigh Prior Contrast Enhancement)
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

coco_gt = COCO(ANN_JSON)
images_info = coco_gt.dataset["images"]
print(f"Total valid images: {len(images_info)}")

# 加载当前最佳 checkpoint
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

# 预处理函数定义
def preprocess_baseline(rgb_pil, ir_path):
    return rgb_pil

def preprocess_debloom_sharpen(rgb_pil, ir_path):
    """Method IR-3: 结构张量各向异性逆热扩散导向锐化"""
    rgb_np = np.array(rgb_pil)
    if not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0]))
    
    # 提取逆热扩散边缘掩膜
    vg = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    barrier = np.exp(- (np.sqrt(cv2.Sobel(vg, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(vg, cv2.CV_32F, 0, 1)**2) / 0.12)**2)
    lap_u = cv2.Laplacian(ir.astype(np.float32), cv2.CV_32F, ksize=3)
    shock = - np.sign(lap_u) * barrier
    
    # 将收拢边界微残差轻量注入亮度通道
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) + shock * 6.0, 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

def preprocess_multiscale_mca(rgb_pil, ir_path):
    """Method IR-1: 多尺度形态学 MCA 热显著性加权注入"""
    rgb_np = np.array(rgb_pil)
    if not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0]))
    
    # 计算多尺度顶帽
    out_mca = np.zeros_like(ir, dtype=np.float32)
    tot_w = 0.0
    for k in (5, 11, 21):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        wth = cv2.morphologyEx(ir, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        w = 1.0 / np.sqrt(k)
        out_mca += w * wth
        tot_w += w
    mca_saliency = np.clip((out_mca / tot_w) / 255.0, 0.0, 1.0)
    
    # 目标区域自适应微调局部高频
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.12 * mca_saliency), 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

def run_eval(preprocess_fn, name):
    detections = []
    with torch.no_grad():
        for info in images_info:
            img_id = info["id"]
            W, H = info["width"], info["height"]
            fname = info["file_name"]
            fpath = os.path.join(IMG_DIR, fname)
            ir_path = os.path.join(IR_DIR, fname)
            
            with Image.open(fpath) as pil_img:
                pil_img = pil_img.convert("RGB")
                pil_img = preprocess_fn(pil_img, ir_path)
                tensor = resize_tensor(pil_img).unsqueeze(0).to(device)
                out = model(tensor)
                logits = out["pred_logits"][0]
                boxes = out["pred_boxes"][0]
                scores, labels = logits.sigmoid().max(dim=-1)

                keep = scores >= 0.015 # 采用安全过滤门限
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
    aps = evaler.stats[3]  # AP_small
    apm = evaler.stats[4]  # AP_medium
    apl = evaler.stats[5]  # AP_large
    print(f"{name:<35s} | mAP={mAP:.4f} | AP50={ap50:.4f} | AP75={ap75:.4f} | APs={aps:.4f} | APm={apm:.4f} | APl={apl:.4f}")
    return {"mAP": mAP, "AP50": ap50, "AP75": ap75, "APs": aps, "APm": apm, "APl": apl}

if __name__ == "__main__":
    print("\n" + "=" * 90)
    print(" [下游检测实测] 200 张全量验证集端到端目标检测 mAP 评测")
    print("=" * 90)
    print(f"{'方案名称':<35s} | {'mAP@50:95':<10} | {'AP50':<8} | {'AP75':<8} | {'AP_small':<8} | {'AP_med':<8} | {'AP_large':<8}")
    print("-" * 90)
    
    res_base = run_eval(preprocess_baseline, "1. 原生基线 (Raw Baseline)")
    res_mca = run_eval(preprocess_multiscale_mca, "2. Method IR-1 (多尺度 MCA 显著注入)")
    res_debloom = run_eval(preprocess_debloom_sharpen, "3. Method IR-3 (各向异性逆热扩散收拢)")
    
    results = {
        "baseline": res_base,
        "multiscale_mca": res_mca,
        "debloom": res_debloom
    }
    
    out_json = os.path.join(ROOT, "work_dir/audits/audit_novel_techniques/downstream_detection_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n下游检测测评完成，结果已保存至: {out_json}")
