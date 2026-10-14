#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_round3_synergy_combo.py
测试第三轮两大巅峰突破算子的正交协同:
  Dir-1 (频域交叉谱相位锁定 FCSPM) + Dir-3 (3D 局部平面突出度残差 3D-PPR)
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

coco_gt = COCO(ANN_JSON)
images_info = coco_gt.dataset["images"]

cfg = YAMLConfig("configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_puremal.yml")
ckpt_path = "outputs/deim_hgnetv2_x_aic_puremal_refine_giou35/best_stg1.pth"
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

def prep_synergy_fcspm_ppr(rgb_pil, fname):
    base_name = os.path.splitext(fname)[0]
    ir_path = os.path.join(IR_DIR, fname)
    depth_m = read_depth_m(base_name)
    rgb_np = np.array(rgb_pil)
    
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    
    # 1. FCSPM (频域相位注入)
    if os.path.exists(ir_path):
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir is not None:
            ir = cv2.resize(ir, (rgb_np.shape[1], rgb_np.shape[0]))
            f_ir = np.fft.fftshift(np.fft.fft2(ir.astype(np.float32)))
            f_l = np.fft.fftshift(np.fft.fft2(l_chan.astype(np.float32)))
            mag_ir = np.abs(f_ir)
            phase_ir = np.angle(f_ir)
            phase_l = np.angle(f_l)
            H, W = ir.shape
            cy, cx = H // 2, W // 2
            y, x = np.ogrid[:H, :W]
            radius = np.sqrt((x - cx)**2 + (y - cy)**2)
            cutoff = min(H, W) * 0.05
            w_p = np.clip((radius - cutoff) / (min(H, W) * 0.25 + 1e-4), 0.0, 1.0) * 0.40
            sin_p = (1.0 - w_p) * np.sin(phase_ir) + w_p * np.sin(phase_l)
            cos_p = (1.0 - w_p) * np.cos(phase_ir) + w_p * np.cos(phase_l)
            f_synth = mag_ir * np.exp(1j * np.arctan2(sin_p, cos_p))
            ir_synth = np.clip(np.real(np.fft.ifft2(np.fft.ifftshift(f_synth))), 0, 255).astype(np.uint8)
            diff = np.clip(ir_synth.astype(np.float32) - cv2.medianBlur(ir_synth, 31).astype(np.float32), 0, 255)
            # 基础热增强增益
            fcspm_gain = diff / 255.0
        else:
            fcspm_gain = np.zeros_like(l_chan, dtype=np.float32)
    else:
        fcspm_gain = np.zeros_like(l_chan, dtype=np.float32)
        
    # 2. 3D PPR (3D 局部平面突出度残差)
    if depth_m is not None:
        depth_m = cv2.resize(depth_m, (rgb_np.shape[1], rgb_np.shape[0])).astype(np.float32)
        valid = (depth_m > 0.1) & (depth_m < 19.5)
        d_smooth = cv2.bilateralFilter(depth_m, d=15, sigmaColor=1.5, sigmaSpace=25)
        dz_dx = cv2.Sobel(d_smooth, cv2.CV_32F, 1, 0, ksize=3)
        dz_dy = cv2.Sobel(d_smooth, cv2.CV_32F, 0, 1, ksize=3)
        mag_grad = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
        residual_3d = np.abs(depth_m - d_smooth) / mag_grad
        residual_3d[~valid] = 0.0
        ppr_score = np.clip((residual_3d - 0.15) / 0.85, 0.0, 1.0)
    else:
        ppr_score = np.zeros_like(l_chan, dtype=np.float32)

    # 3. 正交调制: 几何突起残差给热能提供门控放大，同时为暗区弱几何补光
    synergy_weight = 0.08 * fcspm_gain + 0.08 * ppr_score + 0.06 * (fcspm_gain * ppr_score)
    l_enh = np.clip(l_chan.astype(np.float32) * (1.0 + synergy_weight), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_enh, a_chan, b_chan]), cv2.COLOR_LAB2RGB))

def run_eval():
    detections = []
    with torch.no_grad():
        for info in images_info:
            img_id = info["id"]
            W, H = info["width"], info["height"]
            fname = info["file_name"]
            fpath = os.path.join(IMG_DIR, fname)
            
            with Image.open(fpath) as pil_img:
                pil_img = pil_img.convert("RGB")
                pil_img = prep_synergy_fcspm_ppr(pil_img, fname)
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
    print(f"FCSPM + 3D-PPR Synergy | mAP={mAP:.4f} | AP50={ap50:.4f} | AP75={ap75:.4f} | APs={aps:.4f} | APm={apm:.4f} | APl={apl:.4f}")
    
    out_json = os.path.join(OUT_DIR, "synergy_fcspm_ppr_result.json")
    with open(out_json, "w") as f:
        json.dump({"mAP": mAP, "AP50": ap50, "AP75": ap75, "APs": aps, "APm": apm, "APl": apl}, f, indent=2)

if __name__ == "__main__":
    run_eval()
