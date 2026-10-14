#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_submission_sovereign_multimodal.py - AIC 2026 Phase 2 终极大融合模型官方提交包生成器
支持:
1. 6通道全多模态 (RGB + Aligned IR + LBG IR + P0 16-bit Metric Depth)
2. Sovereign V5 (Clean GIoU 3.5x + MAL2, 56.77% 历史最高分模型)
3. 官方物理后处理流水线:
   - 水际线纵横比截断约束 (Waterline Rectification, H > 1.25W)
   - 孪生框高 IoU 软融合 (Soft-WBF, IoU >= 0.82 仅对 car/seat/boat)
4. 输出逐字节严密对齐官方范例: <category_id> <cx> <cy> <w> <h> <score>
"""

import os
import sys
import glob
import json
import zipfile
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

REPO = "/root/autodl-tmp/AIC"
V2 = os.path.join(REPO, "vendor", "DEIMv2")
sys.path.insert(0, V2)
sys.path.insert(0, os.path.join(REPO, "scripts", "lib"))
sys.path.insert(0, REPO)
os.chdir(V2)

from engine.core import YAMLConfig

DEFAULT_CONFIG = "configs/deimv2/deim_hgnetv2_x_aic_sovereign_v5_refine_giou35_rtx6000.yml"
DEFAULT_CKPT = "outputs/deim_hgnetv2_x_aic_sovereign_v5_refine_giou35_rtx6000/best_stg1.pth"
DEFAULT_TEST_DIR = os.path.join(REPO, "dataset/test/AIC2026_PHASE_2_1000")
DEFAULT_OUT_DIR = os.path.join(REPO, "work_dir/submission_sovereign_v5")
ZIP_NAME = "submission_sovereign_v5.zip"

MEAN = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)

def log(*args):
    print("[sovereign_submit]", *args, flush=True)

def rectify_boat_waterline(boxes, scores, labels, beta=0.96, max_aspect=1.25):
    """
    收紧被水面倒影拉长的船只底边
    boxes: [N, 4] in normalized cxcywh format
    """
    if len(boxes) == 0:
        return boxes
    boxes = boxes.clone()
    is_boat = (labels == 1)
    if not is_boat.any():
        return boxes
    
    # cx, cy, w, h -> x1, y1, x2, y2
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    
    for i in range(len(boxes)):
        if is_boat[i]:
            curr_w = w[i].item()
            curr_h = h[i].item()
            if curr_h > max_aspect * curr_w:
                # Excessively tall boat box (captured underwater reflection)
                old_y2 = y2[i].item()
                curr_y1 = y1[i].item()
                new_y2 = curr_y1 + beta * (old_y2 - curr_y1)
                y2[i] = new_y2
                # recompute cy, h
                boxes[i, 1] = (curr_y1 + new_y2) / 2.0
                boxes[i, 3] = new_y2 - curr_y1
    return boxes

def intra_class_soft_wbf(boxes, scores, labels, iou_thresh=0.82):
    """
    孪生框去重: 仅对 car(6), seat(3), boat(1) 进行高阈值软融合
    """
    if len(boxes) <= 1:
        return boxes, scores, labels
    
    target_classes = {1, 3, 6} # boat, seat, car
    keep_boxes, keep_scores, keep_labels = [], [], []
    
    unique_labels = torch.unique(labels)
    for c in unique_labels:
        mask = (labels == c)
        c_boxes = boxes[mask]
        c_scores = scores[mask]
        
        if c.item() not in target_classes or len(c_boxes) <= 1:
            keep_boxes.append(c_boxes)
            keep_scores.append(c_scores)
            keep_labels.append(labels[mask])
            continue
            
        # Convert cxcywh to xyxy for IoU computation
        cx, cy, w, h = c_boxes.unbind(-1)
        xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)
        
        # Simple clustering
        n = len(c_boxes)
        used = [False] * n
        c_out_b, c_out_s = [], []
        
        for i in range(n):
            if used[i]:
                continue
            b1 = c_boxes[i]
            s1 = c_scores[i]
            box1 = xyxy[i]
            cluster_b = [b1]
            cluster_s = [s1]
            used[i] = True
            
            for j in range(i + 1, n):
                if used[j]:
                    continue
                box2 = xyxy[j]
                # compute IoU
                inter_x1 = torch.max(box1[0], box2[0])
                inter_y1 = torch.max(box1[1], box2[1])
                inter_x2 = torch.min(box1[2], box2[2])
                inter_y2 = torch.min(box1[3], box2[3])
                inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - inter_area
                iou = inter_area / (union + 1e-7)
                
                if iou >= iou_thresh:
                    cluster_b.append(c_boxes[j])
                    cluster_s.append(c_scores[j])
                    used[j] = True
                    
            if len(cluster_b) > 1:
                stacked_b = torch.stack(cluster_b)
                stacked_s = torch.stack(cluster_s)
                # Weighted average box
                merged_b = (stacked_b * stacked_s.unsqueeze(-1)).sum(dim=0) / stacked_s.sum()
                merged_s = stacked_s.max()
                c_out_b.append(merged_b)
                c_out_s.append(merged_s)
            else:
                c_out_b.append(b1)
                c_out_s.append(s1)
                
        keep_boxes.append(torch.stack(c_out_b))
        keep_scores.append(torch.stack(c_out_s))
        keep_labels.append(torch.full((len(c_out_b),), c, dtype=labels.dtype, device=labels.device))
        
    return torch.cat(keep_boxes, dim=0), torch.cat(keep_scores, dim=0), torch.cat(keep_labels, dim=0)

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

def read_sample_6ch(img_path, test_root):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    
    # 1. Visible (RGB, 3ch)
    vis_im = Image.open(img_path).convert("RGB")
    orig_w, orig_h = vis_im.size
    vis_im = vis_im.resize((640, 640), Image.BILINEAR)
    vis_t = TF.pil_to_tensor(vis_im).float() / 255.0 # [3, 640, 640]
    
    # 2. Aligned IR (1ch)
    ir_align_path = os.path.join(REPO, "work_dir/test_aligned_ir", f"{stem}.jpg")
    if not os.path.exists(ir_align_path):
        ir_align_path = os.path.join(REPO, "work_dir/test_aligned_ir", f"{stem}.png")
    ir_align_im = Image.open(ir_align_path).convert("L").resize((640, 640), Image.BILINEAR)
    ir_align_t = TF.pil_to_tensor(ir_align_im).float() / 255.0 # [1, 640, 640]
    
    # 3. LBG IR (1ch)
    ir_lbg_path = os.path.join(REPO, "work_dir/test_lbg_ir", f"{stem}.jpg")
    if not os.path.exists(ir_lbg_path):
        ir_lbg_path = os.path.join(REPO, "work_dir/test_lbg_ir", f"{stem}.png")
    ir_lbg_im = Image.open(ir_lbg_path).convert("L").resize((640, 640), Image.BILINEAR)
    ir_lbg_t = TF.pil_to_tensor(ir_lbg_im).float() / 255.0 # [1, 640, 640]
    
    # 4. Depth P0 (16-bit, 1ch)
    depth_path = os.path.join(test_root, "depth_p0", f"{stem}.png")
    if os.path.exists(depth_path):
        d_im = Image.open(depth_path)
        d_arr = np.array(d_im).astype(np.float32)
        scale = 19999.0 if getattr(d_im, 'mode', '') in ('I;16', 'I') else 255.0
        d_arr = np.clip(d_arr / scale, 0.0, 1.0)
        d_t = torch.from_numpy(d_arr).unsqueeze(0).unsqueeze(0)
        d_t = torch.nn.functional.interpolate(d_t, size=(640, 640), mode="bilinear", align_corners=False).squeeze(0)
    else:
        # Fallback to zeros if not yet written
        d_t = torch.zeros((1, 640, 640), dtype=torch.float32)
        
    x_6ch = torch.cat([vis_t, ir_align_t, ir_lbg_t, d_t], dim=0) # [6, 640, 640]
    x_6ch = (x_6ch - MEAN) / STD
    return x_6ch, orig_w, orig_h, stem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--zip-name", default=ZIP_NAME)
    parser.add_argument("--score-thresh", type=float, default=0.01)
    parser.add_argument("--max-dets", type=int, default=70)
    args = parser.parse_args()

    pred_dir = os.path.join(args.out_dir, "predictions")
    zip_path = os.path.join(args.out_dir, args.zip_name)
    
    import shutil
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.makedirs(pred_dir, exist_ok=True)
    
    model = load_native_model(args.config, args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vis_dir = os.path.join(args.test_dir, "visible")
    image_paths = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".png", ".jpeg"))]
    log(f"Found {len(image_paths)} test images in {vis_dir}")
    assert len(image_paths) == 1000, f"Expected 1000 test images, got {len(image_paths)}"

    total_dets = 0
    per_class_counts = {c: 0 for c in range(12)}
    
    # Process in batches
    batch_size = 16
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inferring Sovereign V5"):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        meta = []
        for p in batch_paths:
            x_6ch, w, h, stem = read_sample_6ch(p, args.test_dir)
            tensors.append(x_6ch)
            meta.append((stem, w, h))
            
        batch_x = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            out = model(batch_x)
            
        pred_logits = out["pred_logits"].sigmoid() # [B, 300, 12]
        pred_boxes = out["pred_boxes"]            # [B, 300, 4] normalized cxcywh
        
        for b_idx in range(len(batch_paths)):
            stem, w, h = meta[b_idx]
            b_logits = pred_logits[b_idx]
            b_boxes = pred_boxes[b_idx]
            
            # Argmax label and score
            scores, labels = b_logits.max(dim=-1)
            
            # Filter score >= threshold
            valid_mask = scores >= args.score_thresh
            c_scores = scores[valid_mask]
            c_labels = labels[valid_mask]
            c_boxes = b_boxes[valid_mask]
            
            # 1. Apply Waterline Rectification on boat
            c_boxes = rectify_boat_waterline(c_boxes, c_scores, c_labels, beta=0.96, max_aspect=1.25)
            
            # 2. Apply Soft-WBF on twin queries
            c_boxes, c_scores, c_labels = intra_class_soft_wbf(c_boxes, c_scores, c_labels, iou_thresh=0.82)
            
            # 3. Top-k sorting
            if len(c_scores) > args.max_dets:
                topk_idx = torch.topk(c_scores, k=args.max_dets).indices
                c_scores = c_scores[topk_idx]
                c_labels = c_labels[topk_idx]
                c_boxes = c_boxes[topk_idx]
                
            # Write txt file
            txt_path = os.path.join(pred_dir, f"{stem}.txt")
            with open(txt_path, "w", encoding="utf-8", newline="\r\n") as f:
                for det_i in range(len(c_scores)):
                    cid = int(c_labels[det_i].item())
                    sc = float(c_scores[det_i].item())
                    cx = float(c_boxes[det_i, 0].item())
                    cy = float(c_boxes[det_i, 1].item())
                    bw = float(c_boxes[det_i, 2].item())
                    bh = float(c_boxes[det_i, 3].item())
                    
                    # Clamp coordinates to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))
                    
                    line = f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {sc:.4f}\r\n"
                    f.write(line)
                    total_dets += 1
                    per_class_counts[cid] += 1

    log(f"Inference finished. Total detections: {total_dets}")
    log("Per-class detection distribution:")
    for cid, cnt in sorted(per_class_counts.items()):
        log(f"  Class {cid:2d}: {cnt:5d} detections ({cnt/total_dets*100:.2f}%)")
        
    # Create zip file with flat structure
    log(f"Packaging into {zip_path}...")
    txt_files = sorted(glob.glob(os.path.join(pred_dir, "*.txt")))
    assert len(txt_files) == 1000, f"Expected 1000 txt files, got {len(txt_files)}"
    
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for tf in txt_files:
            zf.write(tf, arcname=os.path.basename(tf))
            
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    log(f"Submission zip generated successfully: {zip_path} ({zip_size_mb:.2f} MB)")
    
    # Copy to root directory for easy access
    root_zip = os.path.join(REPO, ZIP_NAME)
    shutil.copyfile(zip_path, root_zip)
    log(f"Copied to project root: {root_zip}")

if __name__ == "__main__":
    main()
