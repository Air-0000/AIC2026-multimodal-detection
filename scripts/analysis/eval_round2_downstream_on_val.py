#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_round2_downstream_on_val.py
测试第二批前沿预处理在 200 张全量验证集上的端到端下游检测 mAP 变化:
对比:
  1. 原生基线 (Raw Baseline)
  2. Method Joint-4: 距离驱动自适应高频纹理衰减 (Depth-Driven Adaptive Texture Attenuation)
  3. Method Joint-3: 3D 体积-热密度相干比 (Volume-Thermal Density Ratio, Phi_VT)
  4. Method IR-9: 昼夜温差双极性不变变换 (Thermal Polarity Invariant Transform, E_polar)
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

def prep_baseline(rgb_pil, fname):
    return rgb_pil

def prep_method_joint4_texture_atten(rgb_pil, fname):
    """Method Joint-4: 距离驱动自适应高频纹理衰减 (近景保真, 远景压制杂波)"""
    base_name = os.path.splitext(fname)[0]
    depth_m = read_depth_m(base_name)
    rgb_np = np.array(rgb_pil)
    if depth_m is None:
        return rgb_pil
    depth_m = cv2.resize(depth_m, (rgb_np.shape[1], rgb_np.shape[0]))
    
    z_near, z_far = 4.0, 16.0
    d_clip = np.clip(depth_m, z_near, z_far)
    tau = (d_clip - z_near) / (z_far - z_near + 1e-4) # 0 at near, 1 at far
    
    # 柔和保边模糊 (双边/高斯)
    blur_mild = cv2.GaussianBlur(rgb_np, (5, 5), 1.0)
    tau_3ch = np.repeat(tau[:, :, None], 3, axis=2)
    blended = (1.0 - tau_3ch) * rgb_np.astype(np.float32) + tau_3ch * blur_mild.astype(np.float32)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def prep_method_joint3_phi_vt(rgb_pil, fname):
    """Method Joint-3: 3D 体积-热密度相干比 (抑制水面/地面平坦反光)"""
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
    
    # LBG
    lbg = np.clip(ir.astype(np.float32) - cv2.medianBlur(ir, 31).astype(np.float32), 0, 255)
    
    # 3D 曲率与平坦度
    dz_dx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    div_n = np.abs(cv2.Sobel(-dz_dx/norm, cv2.CV_32F, 1, 0) + cv2.Sobel(-dz_dy/norm, cv2.CV_32F, 0, 1))
    flat_score = np.exp(- (div_n / 0.15)**2)
    
    # 过滤平坦表面热辐射
    phi_vt = lbg * (1.0 - 0.75 * flat_score)
    valid_mask = depth_m > 0.05
    phi_vt[~valid_mask] *= 0.5
    
    # 注入亮度高频
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.10 * (phi_vt / 255.0)), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

def prep_method_ir9_polar(rgb_pil, fname):
    """Method IR-9: 昼夜温差双极性不变特征变换 (E_polar 注入)"""
    ir_path = os.path.join(IR_DIR, fname)
    rgb_np = np.array(rgb_pil)
    if not os.path.exists(ir_path):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
    ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0]))
    
    med = cv2.medianBlur(ir, 31)
    abs_diff = np.abs(ir.astype(np.float32) - med.astype(np.float32))
    mag = np.sqrt(cv2.Sobel(ir, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(ir, cv2.CV_32F, 0, 1)**2)
    e_polar = abs_diff + 0.35 * mag
    norm_polar = np.clip(e_polar / 255.0, 0.0, 1.0)
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enh = np.clip(l.astype(np.float32) * (1.0 + 0.08 * norm_polar), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB))

def run_downstream_eval(prep_fn, name):
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
    print("\n" + "=" * 94)
    print(" [第二轮下游全量评测] 200 张验证集端到端目标检测 mAP 评测")
    print("=" * 94)
    print(f"{'方案名称':<42s} | {'mAP@50:95':<10} | {'AP50':<8} | {'AP75':<8} | {'AP_small':<8} | {'AP_med':<8} | {'AP_large':<8}")
    print("-" * 94)
    
    r_base = run_downstream_eval(prep_baseline, "1. 原生基线 (Raw Baseline)")
    r_joint4 = run_downstream_eval(prep_method_joint4_texture_atten, "2. Method Joint-4 (距离自适应纹理衰减)")
    r_joint3 = run_downstream_eval(prep_method_joint3_phi_vt, "3. Method Joint-3 (3D体积-热密度相干比)")
    r_ir9 = run_downstream_eval(prep_method_ir9_polar, "4. Method IR-9 (温差双极性不变变换)")
    
    results = {
        "baseline": r_base,
        "method_joint4": r_joint4,
        "method_joint3": r_joint3,
        "method_ir9": r_ir9
    }
    
    out_json = os.path.join(ROOT, "work_dir/audits/audit_round2_techniques/round2_downstream_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n第二轮下游检测全量评测完成，结果已保存至: {out_json}")
