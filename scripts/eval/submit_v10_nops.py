#!/usr/bin/env python3
"""
Sovereign V10 — Fixed Channels, Single-Scale Submission
========================================================
Single global 640x640 inference only. No SAHI patches / water prior / sign calibration.
Output: per-image .txt  <cls> <cx> <cy> <w> <h> <conf>  (normalized, CRLF, <=100 boxes)

Channel layout (matches training exactly):
  ch 0,1,2 = RGB visible
  ch 3     = heat_IR  = clip(IR_gray - RGB_gray, 0, 1)       [training: ir_folder]
  ch 4     = lbg_IR   = clip(IR - medianBlur(IR,31), 0, 255) [training: raw_ir_folder]
  ch 5     = depth_p0 (16-bit correctly normalized)
"""

import os
import glob
import argparse
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.ops import nms
from PIL import Image
import numpy as np


def apply_preproc(raw):
    """USM + DGTB front-end (identical to eval_caa_sahi_sovereign_v9.py)"""
    rgb  = raw[:3]
    blur = F.avg_pool2d(rgb.unsqueeze(0), 3, stride=1, padding=1)[0]
    rgb_usm = (rgb + 0.15 * (rgb - blur)).clamp(0, 1)
    d = raw[5:6]
    sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=raw.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], device=raw.device).view(1,1,3,3)
    g = torch.sqrt(F.conv2d(d.unsqueeze(0), sobel_x, padding=1)**2 +
                   F.conv2d(d.unsqueeze(0), sobel_y, padding=1)**2)[0]
    edge_mask = (g - 0.015).div(0.03).clamp(0, 1)
    ir = raw[3:5]
    ir_boost = (ir + 0.25 * edge_mask * (ir - F.avg_pool2d(ir.unsqueeze(0), 3, stride=1, padding=1)[0])).clamp(0, 1)
    return torch.cat([rgb_usm, ir_boost, raw[5:6]], dim=0)


def build_lbg_ir(ir_gray_np_u8):
    """Median top-hat: clip(IR - medianBlur(IR,31), 0, 255) / 255.
    Identical to prep_ir_localbg.py local_bg_suppress()."""
    bg  = cv2.medianBlur(ir_gray_np_u8, 31)
    lbg = np.clip(ir_gray_np_u8.astype(np.float32) - bg.astype(np.float32), 0, 255)
    return lbg / 255.0


def load_depth_tensor(path, device, H, W):
    """16-bit correct depth read (matches coco_dataset_mm._read_depth_raw16)."""
    img  = Image.open(path)
    arr  = np.array(img, dtype=np.float32)
    mode = getattr(img, 'mode', '')
    scale = 19999.0 if mode in ('I;16', 'I') else (65535.0 if arr.max() > 255.5 else 255.0)
    arr  = (arr / scale).clip(0.0, 1.0)
    t    = torch.from_numpy(arr).unsqueeze(0).to(device)
    if t.shape[1] != H or t.shape[2] != W:
        t = F.interpolate(t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000.yml")
    parser.add_argument("--checkpoint",  default="outputs/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000/best_stg1.pth")
    parser.add_argument("--test_dir",    default="/root/autodl-tmp/AIC/dataset/test/AIC2026_PHASE_2_1000")
    parser.add_argument("--out_dir",     default="/root/autodl-tmp/AIC/work_dir/submission_v10_fixed")
    parser.add_argument("--conf_thresh", type=float, default=0.15)
    parser.add_argument("--nms_iou",     type=float, default=0.65)
    parser.add_argument("--max_boxes",   type=int,   default=100)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from engine.core import YAMLConfig
    cfg = YAMLConfig(args.config)
    print(f"Loading checkpoint: {args.checkpoint}")
    ck      = torch.load(args.checkpoint, map_location="cpu")
    weights = ck["ema"]["module"] if "ema" in ck and "module" in ck.get("ema", {}) else ck.get("model", ck)
    cfg.model.load_state_dict(weights, strict=False)
    model = cfg.model.cuda().eval()
    post  = cfg.postprocessor

    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5], device="cuda").view(6,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5], device="cuda").view(6,1,1)

    vis_dir    = os.path.join(args.test_dir, "visible")
    ir_dir     = os.path.join(args.test_dir, "infrared")
    depth_dir  = os.path.join(args.test_dir, "depth_p0")
    lbg_ir_dir = "/root/autodl-tmp/AIC/work_dir/test_lbg_ir"   # precomputed LBG-IR for test set
    print(f"LBG-IR cache: {lbg_ir_dir}  (exists={os.path.isdir(lbg_ir_dir)})")

    vis_paths = sorted(glob.glob(os.path.join(vis_dir, "*.jpg")) +
                       glob.glob(os.path.join(vis_dir, "*.png")))
    print(f"Found {len(vis_paths)} test images")

    processed = 0
    with torch.no_grad():
        for vis_path in vis_paths:
            stem       = os.path.splitext(os.path.basename(vis_path))[0]
            ir_path    = os.path.join(ir_dir,    stem + ".jpg")
            depth_path = os.path.join(depth_dir, stem + ".png")
            lbg_path   = os.path.join(lbg_ir_dir, stem + ".jpg")
            if not os.path.exists(ir_path):
                ir_path  = os.path.join(ir_dir, stem + ".png")
            if not os.path.exists(lbg_path):
                lbg_path = os.path.join(lbg_ir_dir, stem + ".png")
            if not os.path.exists(ir_path) or not os.path.exists(depth_path):
                print(f"  [WARN] missing modality for {stem}")
                continue

            # ── Assemble CORRECT 6-channel tensor ─────────────────────────
            vis_pil  = Image.open(vis_path).convert("RGB")
            H, W     = vis_pil.size[1], vis_pil.size[0]
            rgb      = TF.pil_to_tensor(vis_pil).float().div(255.0).cuda()   # [3,H,W]

            # IR grayscale
            ir_pil   = Image.open(ir_path).convert("L")
            ir_np_u8 = np.array(ir_pil, dtype=np.uint8)
            ir_gray  = torch.from_numpy(ir_np_u8.astype(np.float32)/255.0).unsqueeze(0).cuda()  # [1,H,W]
            if ir_gray.shape[1] != H or ir_gray.shape[2] != W:
                ir_gray  = F.interpolate(ir_gray.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=False)[0]
                ir_np_u8 = (ir_gray[0].cpu().numpy()*255).clip(0,255).astype(np.uint8)

            # ch3: heat_IR = clip(IR_gray - RGB_gray, 0, 1)
            rgb_gray = rgb.mean(dim=0, keepdim=True)                          # [1,H,W]
            heat_ir  = (ir_gray - rgb_gray).clamp(0, 1)                      # [1,H,W]

            # ch4: LBG-IR (load precomputed; compute on-the-fly if missing)
            if os.path.exists(lbg_path):
                lbg_pil = Image.open(lbg_path).convert("L")
                lbg_arr = np.array(lbg_pil, dtype=np.float32) / 255.0
                lbg_ir  = torch.from_numpy(lbg_arr).unsqueeze(0).cuda()
                if lbg_ir.shape[1] != H or lbg_ir.shape[2] != W:
                    lbg_ir = F.interpolate(lbg_ir.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=False)[0]
            else:
                lbg_arr = build_lbg_ir(ir_np_u8)
                lbg_ir  = torch.from_numpy(lbg_arr).float().unsqueeze(0).cuda()

            # ch5: depth_p0 (16-bit correct)
            depth = load_depth_tensor(depth_path, "cuda", H, W)

            raw = torch.cat([rgb, heat_ir, lbg_ir, depth], dim=0)            # [6,H,W]
            raw = apply_preproc(raw)

            # ── Single global 640x640 forward ─────────────────────────────
            p0   = F.interpolate(raw.unsqueeze(0), size=(640, 640), mode="bilinear", align_corners=False)
            norm = (p0 - mean) / std
            sizes = torch.tensor([[float(W), float(H)]], device="cuda")

            out = model(norm)
            res = post(out, sizes, norm)

            boxes  = res[0]["boxes"]
            scores = res[0]["scores"]
            labels = res[0]["labels"]

            # Confidence threshold
            keep_conf = scores >= args.conf_thresh
            boxes  = boxes[keep_conf]
            scores = scores[keep_conf]
            labels = labels[keep_conf]

            # Per-category NMS → submission lines
            lines = []
            for c in range(12):
                mask = labels == c
                if not mask.any():
                    continue
                c_boxes  = boxes[mask]
                c_scores = scores[mask]
                keep_idx = nms(c_boxes, c_scores, iou_threshold=args.nms_iou)
                for b, s in zip(c_boxes[keep_idx].cpu().tolist(), c_scores[keep_idx].cpu().tolist()):
                    x1, y1, x2, y2 = b
                    cx = max(0., min(1., (x1+x2)/2./W))
                    cy = max(0., min(1., (y1+y2)/2./H))
                    bw = max(0., min(1., (x2-x1)/W))
                    bh = max(0., min(1., (y2-y1)/H))
                    lines.append((s, f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {min(s,1.):.6f}"))

            lines.sort(key=lambda x: x[0], reverse=True)
            lines = lines[:args.max_boxes]

            out_path = os.path.join(args.out_dir, stem + ".txt")
            with open(out_path, "w", newline="\r\n") as f:
                for _, line in lines:
                    f.write(line + "\n")

            processed += 1
            if processed % 100 == 0 or processed == 1:
                print(f"  [{processed}/{len(vis_paths)}] {stem}.txt → {len(lines)} boxes")

    print(f"\nDone. {processed} files → {args.out_dir}")
    print(f"Submission TXT count: {len(glob.glob(os.path.join(args.out_dir, '*.txt')))} / {len(vis_paths)}")


if __name__ == "__main__":
    main()
