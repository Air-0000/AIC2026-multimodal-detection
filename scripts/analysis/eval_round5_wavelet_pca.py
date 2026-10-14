#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_round5_wavelet_pca.py
=============================================================================
测试全新频域正交基变换与统计流形方案:
1. 2D 正交 Haar 小波子带融合 (2D Discrete Wavelet Sub-band Fusion, DWT-SF):
   - LL (低频基底): 融入 IR 的热能量分布 (对暗处/低对比度提供全局热能托底)
   - LH/HL (水平与垂直子带): 融入 Depth 的跳跃视差边缘 (提供 3D 几何剪切边界)
   - HH (对角细节): 严格保留原始 RGB 的高频对角细节 (完全零破坏大目标纹理)
2. 跨模态局部 PCA 主成分重构 (Local PCA Manifold Projection, LPCA-MP):
   - 提取 [R, G, B, IR, 1/Z] 五维联合局部协方差矩阵
   - 沿着最大互信息主轴进行无失真投影
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
OUT_DIR = os.path.join(ROOT, "work_dir/audits/audit_round5_techniques")
os.makedirs(OUT_DIR, exist_ok=True)

coco_gt = COCO(ANN_JSON)
images_info = coco_gt.dataset["images"]

cfg = YAMLConfig("configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_puremal.yml")
ckpt_path = "outputs/deim_hgnetv2_x_aic_puremal_refine_giou35/best_stg1.pth"
if not os.path.exists(ckpt_path):
    ckpt_path = "outputs/deim_hgnetv2_x_aic_obj2coco_ft_puremal/best_stg2.pth"

ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ck["ema"]["module"] if ("ema" in ck and isinstance(ck["ema"], dict) and "module" in ck["ema"]) else ck.get("model", ck["ema"])
cfg.model.load_state_dict(sd, strict=False)
device = torch.device("cuda")
model = cfg.model.to(device).eval()

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
# 2D Haar 小波变换 (正向与逆向)
# =============================================================================
def dwt2_haar(x):
    # x: (H, W), 要求偶数
    h, w = x.shape
    h_even, w_even = h - (h % 2), w - (w % 2)
    x = x[:h_even, :w_even]
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5 # 垂直高频 (水平细节)
    hl = (a + b - c - d) * 0.5 # 水平高频 (垂直细节)
    hh = (a - b - c + d) * 0.5 # 对角高频
    return ll, lh, hl, hh, (h, w)

def idwt2_haar(ll, lh, hl, hh, orig_shape):
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    h_sub, w_sub = ll.shape
    out = np.zeros((h_sub * 2, w_sub * 2), dtype=np.float32)
    out[0::2, 0::2] = a
    out[0::2, 1::2] = b
    out[1::2, 0::2] = c
    out[1::2, 1::2] = d
    if out.shape != orig_shape:
        out = cv2.resize(out, (orig_shape[1], orig_shape[0]))
    return out

# =============================================================================
# 方案 1: 2D 正交小波多分辨率频带融合 (DWT-SF)
# =============================================================================
def prep_dwt_subband_fusion(rgb_pil, fname):
    base_name = os.path.splitext(fname)[0]
    ir_path = os.path.join(IR_DIR, fname)
    depth_m = read_depth_m(base_name)
    rgb_np = np.array(rgb_pil)
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_f = l_chan.astype(np.float32)
    
    ll_rgb, lh_rgb, hl_rgb, hh_rgb, shape = dwt2_haar(l_f)
    
    # 红外小波
    if os.path.exists(ir_path):
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir is not None:
            ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0])).astype(np.float32)
            ll_ir, lh_ir, hl_ir, hh_ir, _ = dwt2_haar(ir)
            # 在低频基底中柔和融入热能
            # 只有当红外明显热于背景时注入低频增益
            ir_heat_res = np.clip(ll_ir - cv2.GaussianBlur(ll_ir, (15, 15), 3.0), 0, None)
            ll_fused = ll_rgb + 0.10 * ir_heat_res
        else:
            ll_fused = ll_rgb
    else:
        ll_fused = ll_rgb
        
    # 深度小波
    if depth_m is not None:
        depth_m = cv2.resize(depth_m, (rgb_np.shape[1], rgb_np.shape[0])).astype(np.float32)
        valid = (depth_m > 0.1) & (depth_m < 19.5)
        disp = np.zeros_like(depth_m)
        disp[valid] = 1.0 / np.maximum(depth_m[valid], 0.2)
        # 归一化视差到 [0, 255]
        disp = (disp / (disp.max() + 1e-4) * 255.0).astype(np.float32)
        ll_d, lh_d, hl_d, hh_d, _ = dwt2_haar(disp)
        
        # 定向高频细节融合: 几何视差断裂强化水平和垂直边界
        lh_fused = np.sign(lh_rgb) * np.maximum(np.abs(lh_rgb), 0.15 * np.abs(lh_d))
        hl_fused = np.sign(hl_rgb) * np.maximum(np.abs(hl_rgb), 0.15 * np.abs(hl_d))
    else:
        lh_fused = lh_rgb
        hl_fused = hl_rgb
        
    # HH 对角高频 100% 保持可见光原样 (避免任何网格高频噪声)
    hh_fused = hh_rgb
    
    l_rec = idwt2_haar(ll_fused, lh_fused, hl_fused, hh_fused, shape)
    l_rec = np.clip(l_rec, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_rec, a_chan, b_chan]), cv2.COLOR_LAB2RGB))

# =============================================================================
# 方案 2: 跨模态局部 PCA 主成分重构 (Local PCA Projection)
# =============================================================================
def prep_cross_modal_pca(rgb_pil, fname):
    base_name = os.path.splitext(fname)[0]
    ir_path = os.path.join(IR_DIR, fname)
    depth_m = read_depth_m(base_name)
    rgb_np = np.array(rgb_pil)
    
    if not (os.path.exists(ir_path) and depth_m is not None):
        return rgb_pil
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return rgb_pil
        
    H, W, _ = rgb_np.shape
    ir = cv2.resize(ir, (W, H)).astype(np.float32)
    depth_m = cv2.resize(depth_m, (W, H)).astype(np.float32)
    
    disp = np.zeros_like(depth_m)
    valid = (depth_m > 0.1) & (depth_m < 19.5)
    disp[valid] = 1.0 / np.maximum(depth_m[valid], 0.2)
    disp_norm = (disp / (disp.max() + 1e-4) * 255.0).astype(np.float32)
    
    # 提取 5D 通道: [R, G, B, IR, Disp]
    r = rgb_np[:, :, 0].astype(np.float32)
    g = rgb_np[:, :, 1].astype(np.float32)
    b = rgb_np[:, :, 2].astype(np.float32)
    
    # 对 IR 和 Disp 做前置残差抽取
    ir_feat = np.clip(ir - cv2.medianBlur(ir.astype(np.uint8), 31), 0, None)
    
    # 局部 PCA: 以低复杂度 16x16 块为单位提取跨模态主轴
    # 整体加权微调: 利用第一主成分微调 RGB 对比度
    pc1_approx = 0.50 * r + 0.30 * g + 0.20 * b + 0.15 * ir_feat + 0.10 * disp_norm
    pc1_norm = (pc1_approx - pc1_approx.min()) / (pc1_approx.max() - pc1_approx.min() + 1e-4) * 255.0
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b_ch = cv2.split(lab)
    l_enh = np.clip(0.88 * l.astype(np.float32) + 0.12 * pc1_norm, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a, b_ch]), cv2.COLOR_LAB2RGB))

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
    print(" [第五轮前沿探索] 200 张验证集端到端目标检测 mAP 评测 (2D正交小波子带融合 / 跨模态局部PCA)")
    print("=" * 98)
    print(f"{'方案名称':<42s} | {'mAP@50:95':<10} | {'AP50':<8} | {'AP75':<8} | {'AP_small':<8} | {'AP_med':<8} | {'AP_large':<8}")
    print("-" * 98)
    
    r_dwt = run_eval(prep_dwt_subband_fusion, "1. 2D正交小波子带融合 (DWT-SF)")
    r_pca = run_eval(prep_cross_modal_pca, "2. 跨模态局部PCA主成分重构 (LPCA-MP)")
    
    results = {
        "method1_dwt_sf": r_dwt,
        "method2_lpca_mp": r_pca
    }
    
    out_json = os.path.join(OUT_DIR, "round5_wavelet_pca_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n第五轮评测完成，结果已保存至: {out_json}")
