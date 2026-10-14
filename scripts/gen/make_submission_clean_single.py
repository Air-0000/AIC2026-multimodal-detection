#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_submission_clean_single.py - 纯净原生单模型推理打包器 (无任何 TTA / 人工加权 / 互斥 / 门槛)
完全还原此前 54.2 分数的原生口径:
1. 单尺度 640x640 原生输入
2. 单 Query 严格取 argmax 类别: sc, lb = out['pred_logits'].sigmoid().max(-1)
3. 归一化 cxcywh 原生输出 (模型解码器直接吐出, 无任何坐标转换/聚类扰动)
4. 纯净全局阈值 score >= 0.01 (无类别加权, 无门槛歧视, 无人为互斥)
5. 自动对齐 Phase 2 官方 1000 张文件名及规范
"""

import os
import sys
import glob
import json
import zipfile
import hashlib
import argparse
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T

REPO = "/home/constant/CodeBuddy/Dev/AIC"
V2 = os.path.join(REPO, "vendor", "DEIMv2")
sys.path.insert(0, V2)
sys.path.insert(0, os.path.join(REPO, "scripts", "lib"))
os.chdir(V2)

from engine.core import YAMLConfig

CONFIG_PATH = os.path.join(V2, "configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_mal2_val_orig.yml")
CKPT_PATH = os.path.join(V2, "outputs/deim_hgnetv2_x_aic_obj2coco_ft_mal2/best_stg2.pth")
PHASE2_DIR = os.path.join(REPO, "dataset/test/AIC2026_PHASE_2_1000/visible")
OUT_DIR = os.path.join(REPO, "work_dir/submissions/submission_clean_single")
ZIP_NAME = "submission_clean_single.zip"

def log(*args):
    print("[clean_submit]", *args, flush=True)

def load_native_model(config_path, ckpt_path):
    log("Loading model from config:", config_path)
    cfg = YAMLConfig(config_path)
    model = cfg.model
    
    log("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = None
    if "ema" in ckpt and isinstance(ckpt["ema"], dict):
        m = ckpt["ema"].get("module", ckpt["ema"].get("model"))
        if isinstance(m, dict):
            state = m
            log("Loaded state_dict from ckpt['ema']")
    if state is None and "model" in ckpt:
        state = ckpt["model"]
        log("Loaded state_dict from ckpt['model']")
        
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"Load result: missing={len(missing)}, unexpected={len(unexpected)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--ckpt", default=CKPT_PATH)
    parser.add_argument("--test-dir", default=PHASE2_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--zip-name", default=ZIP_NAME)
    parser.add_argument("--score-thresh", type=float, default=0.01)
    parser.add_argument("--max-dets", type=int, default=70, help="Max detections per image")
    args = parser.parse_args()

    pred_dir = os.path.join(args.out_dir, "predictions")
    zip_path = os.path.join(args.out_dir, args.zip_name)
    
    import shutil
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.makedirs(pred_dir, exist_ok=True)
    
    model = load_native_model(args.config, args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".png", ".jpeg"))]
    log(f"Found {len(image_paths)} images in {args.test_dir}")
    
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor() # scale to [0, 1] float32
    ])

    total_dets = 0
    per_class_counts = {c: 0 for c in range(12)}
    
    with torch.no_grad():
        for img_p in tqdm(image_paths, desc="Clean Single Infer"):
            stem = os.path.splitext(os.path.basename(img_p))[0]
            with Image.open(img_p) as pil_img:
                pil_img = pil_img.convert("RGB")
                tensor = transform(pil_img).unsqueeze(0).to(device)
                
                out = model(tensor)
                logits = out["pred_logits"][0] # [num_queries, num_classes]
                boxes = out["pred_boxes"][0]   # [num_queries, 4] normalized cxcywh
                
                # 严格原生口径: 每个 query 只取最大置信度的单个类别 (无多标签污染)
                scores, labels = logits.sigmoid().max(dim=-1)
                
                keep = scores >= args.score_thresh
                k_scores = scores[keep].cpu().numpy()
                k_labels = labels[keep].cpu().numpy()
                k_boxes = boxes[keep].cpu().numpy()
                
                # 按置信度降序排序并截断至 max-dets
                order = k_scores.argsort()[::-1]
                if args.max_dets > 0:
                    order = order[:args.max_dets]
                
                out_txt = os.path.join(pred_dir, f"{stem}.txt")
                with open(out_txt, "w", newline="") as f:
                    for idx in order:
                        c = int(k_labels[idx])
                        s = float(k_scores[idx])
                        cx, cy, w, h = k_boxes[idx]
                        
                        # 约束在 [0, 1] 范围
                        cx = max(0.0, min(1.0, float(cx)))
                        cy = max(0.0, min(1.0, float(cy)))
                        w = max(0.0, min(1.0, float(w)))
                        h = max(0.0, min(1.0, float(h)))
                        
                        f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {s:.4f}\r\n")
                        total_dets += 1
                        per_class_counts[c] += 1

    log(f"Inference done! Total boxes={total_dets}, avg={total_dets/len(image_paths):.1f}/img")
    log("Class distribution:", per_class_counts)

    # 打包为 zip
    log("Packing to zip:", zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_f in sorted(glob.glob(os.path.join(pred_dir, "*.txt"))):
            zf.write(txt_f, os.path.basename(txt_f))

    md5 = hashlib.md5(open(zip_path, "rb").read()).hexdigest()
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    log(f"Zip ready: {zip_path} ({zip_size_mb:.2f} MB, MD5={md5})")
    
    # 官方校验脚本
    check_script = os.path.join(REPO, "scripts/eval/check_submission.py")
    ret = os.system(f"python3 {check_script} {pred_dir} {args.test_dir}")
    if ret == 0:
        log("SUCCESS: Clean submission package passed all official checks!")
    else:
        log("WARNING: check_submission returned non-zero code!")

if __name__ == "__main__":
    main()
