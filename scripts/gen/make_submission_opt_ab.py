#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_submission_opt_ab.py - 生成基于方案 A+B (微尺度主标锁定) 及针对性调优的官方测试集提交包

特性与核心优化:
1. 方案 B: 微尺度 TTA [608, 640, 672], 严格无翻转 (No-Flip, 避免对称误报与坐标浮点偏移)
2. 方案 A: 主标坐标锁定 (Master 640 Lock), 彻底杜绝 WBF 坐标加权导致的边界模糊, 保护 AP@75-95
3. 置信度策略: conf_mode="max", 避免平均置信度把真阳性拖死在背景负样本之后
4. 类别特异性补偿 (Class Boost): boat x 1.10, sign x 1.15, light x 1.05 (挽救 mal2 误杀的弱目标)
5. 背景噪声准入清洗 (Admission Threshold): person/animal >= 0.02, 其余 >= 0.01
6. 跨类别互斥压制 (Cross-Class Mutex): light 压制 ball, animal 压制 boat, sign 压制 ball
7. 规范合规: Top-100 降序截断, 归一化 cxcywh, CRLF (\r\n) 换行符, 自动执行 check_submission 校验
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
import numpy as np

# ----------------- 路径设置 -----------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
V2 = os.path.join(REPO, "vendor", "DEIMv2")
sys.path.insert(0, V2)
sys.path.insert(0, os.path.join(REPO, "scripts", "lib"))
os.chdir(V2)

from engine.core import YAMLConfig

CONFIG_PATH = os.path.join(V2, "configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_mal2_val_orig.yml")
CKPT_PATH = os.path.join(V2, "outputs/deim_hgnetv2_x_aic_obj2coco_ft_mal2/best_stg2.pth")
PHASE2_DIR = os.path.join(REPO, "dataset/test/AIC2026_PHASE_2_1000/visible")
TEST_IMG_DIR = PHASE2_DIR if os.path.isdir(PHASE2_DIR) else os.path.join(REPO, "dataset/test/visible")
OUT_DIR = os.path.join(REPO, "work_dir/submissions/submission_opt_ab")
PRED_TXT_DIR = os.path.join(OUT_DIR, "predictions")
ZIP_PATH = os.path.join(OUT_DIR, "submission_opt_ab.zip")

SCALES = [608, 640, 672]
IOU_CLUSTER = 0.70
MUTEX_IOU = 0.50
MAX_DETS = 100

CLASS_WEIGHTS = {
    1: 1.10,  # boat: 补偿 mal2 抑制
    4: 1.15,  # sign: 补偿低分 TP 排名
    8: 1.05,  # light: 适度提权
}

PER_CLASS_THRESH = {
    0: 0.02,  # person: 拦截 5000+ 个极低分背景噪点
    2: 0.02,  # animal: 拦截 4000+ 个极低分背景噪点
}
DEFAULT_THRESH = 0.01

def log(*args):
    print("[submit_ab]", *args, flush=True)

def box_iou_xyxy(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / ua if ua > 0 else 0.0

class InferModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "eval_spatial_size"):
            self.model.decoder.eval_spatial_size = None
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig):
        return self.postprocessor(self.model(images), orig)

def load_infer_model(config_path, ckpt_path):
    log("Loading model config from", config_path)
    cfg = YAMLConfig(config_path)
    
    log("Loading checkpoint from", ckpt_path)
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
    
    missing, unexpected = cfg.model.load_state_dict(state, strict=False)
    log(f"Load result: missing={len(missing)}, unexpected={len(unexpected)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_net = InferModel(cfg).to(device).eval()
    return infer_net

def fuse_multiscale_detections(view_candidates):
    """
    view_candidates: list of (cid, [x1, y1, x2, y2], score, scale)
    """
    fused = []
    unique_classes = set(c[0] for c in view_candidates)

    for cls in unique_classes:
        cls_boxes = sorted([b for b in view_candidates if b[0] == cls], key=lambda x: x[2], reverse=True)
        clusters = []
        for it in cls_boxes:
            for cl in clusters:
                if box_iou_xyxy(it[1], cl[0][1]) >= IOU_CLUSTER:
                    cl.append(it)
                    break
            else:
                clusters.append([it])

        for cl in clusters:
            confs = [b[2] for b in cl]
            c640 = [b for b in cl if b[3] == 640]
            # 方案 A: 只要包含 640 主标，强制锁定 640 坐标！
            if c640:
                final_box = list(max(c640, key=lambda x: x[2])[1])
            else:
                final_box = list(max(cl, key=lambda x: x[2])[1])
            
            # 置信度取聚类内最大值
            fused.append((cls, final_box, float(max(confs))))

    # 跨类别互斥过滤 (Cross-Class Mutex)
    fused.sort(key=lambda x: x[2], reverse=True)
    cleaned = []
    suppressed = set()
    for i in range(len(fused)):
        if i in suppressed:
            continue
        c1, b1, s1 = fused[i]
        for j in range(i + 1, len(fused)):
            if j in suppressed:
                continue
            c2, b2, s2 = fused[j]
            should_suppress = False
            # 1. Light (8) 压制 Ball (7), Animal (2) 压制 Boat (1)
            if (c1 == 8 and c2 == 7) or (c1 == 2 and c2 == 1):
                if box_iou_xyxy(b1, b2) >= MUTEX_IOU:
                    should_suppress = True
            # 2. Sign (4) 压制 Ball (7)
            elif (c1 == 4 and c2 == 7):
                if box_iou_xyxy(b1, b2) >= MUTEX_IOU:
                    should_suppress = True

            if should_suppress:
                suppressed.add(j)

        cleaned.append(fused[i])

    # 降序排序并截取 Top-100
    cleaned.sort(key=lambda x: x[2], reverse=True)
    return cleaned[:MAX_DETS]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH, help="Model val yaml config path")
    parser.add_argument("--ckpt", default=CKPT_PATH, help="Model checkpoint path")
    parser.add_argument("--test-dir", default=TEST_IMG_DIR, help="Test image directory")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output submission directory")
    parser.add_argument("--zip-name", default="submission_opt_ab.zip", help="Name of zip package")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for dry run")
    args = parser.parse_args()

    pred_txt_dir = os.path.join(args.out_dir, "predictions")
    zip_path = os.path.join(args.out_dir, args.zip_name)

    import shutil
    if os.path.exists(pred_txt_dir):
        shutil.rmtree(pred_txt_dir)
    os.makedirs(pred_txt_dir, exist_ok=True)
    model = load_infer_model(args.config, args.ckpt)

    image_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".png", ".jpeg"))]
    if args.limit:
        image_paths = image_paths[:args.limit]
    log(f"Found {len(image_paths)} test images in {args.test_dir}")
    if len(image_paths) == 0:
        raise RuntimeError("No images found to process!")

    transforms_cache = {s: T.Compose([T.Resize((s, s)), T.ToTensor()]) for s in SCALES}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_dets = 0
    pbar = tqdm(image_paths, desc="TTA A+B Inferring")
    
    with torch.no_grad():
        for img_p in pbar:
            stem = os.path.splitext(os.path.basename(img_p))[0]
            with Image.open(img_p) as pil_img:
                pil_img = pil_img.convert("RGB")
                orig_w, orig_h = pil_img.size
                orig_tensor = torch.tensor([[orig_w, orig_h]], device=device)

                view_candidates = []
                for s in SCALES:
                    tr = transforms_cache[s]
                    tensor = tr(pil_img).unsqueeze(0).to(device)
                    labels, boxes, scores = model(tensor, orig_tensor)
                    
                    labels = labels[0].cpu()
                    boxes = boxes[0].cpu().numpy()  # [x1, y1, x2, y2]
                    scores = scores[0].cpu().numpy()

                    for k in range(len(scores)):
                        cid = int(labels[k].item())
                        score_val = float(scores[k])
                        
                        # 规则 1: 类别准入门槛过滤
                        min_thresh = PER_CLASS_THRESH.get(cid, DEFAULT_THRESH)
                        if score_val < min_thresh:
                            continue
                        
                        # 规则 2: 类别特异性置信度缩放
                        scaled_score = min(1.0, score_val * CLASS_WEIGHTS.get(cid, 1.0))
                        box_coords = boxes[k].tolist() # [x1, y1, x2, y2] in pixels
                        view_candidates.append((cid, box_coords, scaled_score, s))

            # 方案 A+B 融合
            final_dets = fuse_multiscale_detections(view_candidates)
            total_dets += len(final_dets)

            # 写入 txt 文件 (CRLF 换行, 归一化 cxcywh)
            out_txt = os.path.join(pred_txt_dir, f"{stem}.txt")
            with open(out_txt, "w", newline="") as f:
                for c, (x1, y1, x2, y2), s in final_dets:
                    # 像素坐标转换为归一化 cx, cy, w, h
                    w_box = max(0.0, x2 - x1)
                    h_box = max(0.0, y2 - y1)
                    cx = (x1 + x2) / 2.0 / orig_w
                    cy = (y1 + y2) / 2.0 / orig_h
                    w_norm = w_box / orig_w
                    h_norm = h_box / orig_h

                    # 约束在 [0, 1] 范围内
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))
                    f.write(f"{c} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f} {s:.4f}\r\n")

    log(f"All {len(image_paths)} images processed! Total boxes={total_dets}, avg={total_dets/len(image_paths):.1f}/img")

    # 如果是全量 1000 张，打包为 zip 并执行官方校验
    if not args.limit:
        log("Creating submission zip at", zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for txt_file in sorted(glob.glob(os.path.join(pred_txt_dir, "*.txt"))):
                zf.write(txt_file, os.path.basename(txt_file))

        md5 = hashlib.md5(open(zip_path, "rb").read()).hexdigest()
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        log(f"Zip created successfully: {zip_path} ({zip_size_mb:.2f} MB, MD5={md5})")

        # 自动执行 check_submission 校验
        log("Running official submission validation check...")
        check_script = os.path.join(REPO, "scripts/eval/check_submission.py")
        ret = os.system(f"python3 {check_script} {pred_txt_dir} {args.test_dir}")
        if ret == 0:
            log("SUCCESS: Submission passed all validation checks!")
        else:
            log("WARNING: check_submission returned non-zero code!")

if __name__ == "__main__":
    main()
