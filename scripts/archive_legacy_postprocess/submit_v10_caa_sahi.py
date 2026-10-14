#!/usr/bin/env python3
"""
Sovereign V10 CAA-SAHI Submission Package Generator (Fixed Channels v2)
========================================================================
Generates per-image .txt prediction files for the AIC 2026 Phase 2 test set (1000 images).

Output format per file:
  <cls> <cx> <cy> <w> <h> <conf>   (all coords normalized 0-1, CRLF, <=100 boxes)

Channel layout (matches training exactly):
  ch 0,1,2 = RGB visible
  ch 3     = heat_IR  = clip(IR_gray - RGB_gray, 0, 1)       [training: ir_folder]
  ch 4     = lbg_IR   = clip(IR - medianBlur(IR,31), 0) / 255 [training: raw_ir_folder]
  ch 5     = depth_p0 (16-bit correctly normalized)

CAA-SAHI Engine:
  - Native 16:9 canvas (1138x640) for left/right patches
  - Front-end USM (alpha=0.15) + Depth-Guided Thermal Boost
  - 3D Water Planar Prior: boat only emitted when std(D_bottom)>0.09
  - boat(1), ball(7), tricycle(11): P0 global only, NMS iou=0.50
  - sign(4): confidence x1.05
  - all others: patch fusion, NMS iou=0.65
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


# ── Preprocessing ─────────────────────────────────────────────────────────────

def apply_preproc(raw):
    """raw: [6, H, W] float32 on CUDA, range [0,1]"""
    rgb = raw[:3]
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
    """Median top-hat: matches prep_ir_localbg.py local_bg_suppress() exactly."""
    bg  = cv2.medianBlur(ir_gray_np_u8, 31)
    lbg = np.clip(ir_gray_np_u8.astype(np.float32) - bg.astype(np.float32), 0, 255)
    return lbg / 255.0


def load_depth_tensor(path, device, H, W):
    """16-bit correct depth read matching coco_dataset_mm._read_depth_raw16."""
    img  = Image.open(path)
    arr  = np.array(img, dtype=np.float32)
    mode = getattr(img, 'mode', '')
    scale = 19999.0 if mode in ('I;16', 'I') else (65535.0 if arr.max() > 255.5 else 255.0)
    arr  = (arr / scale).clip(0.0, 1.0)
    t    = torch.from_numpy(arr).unsqueeze(0).to(device)
    if t.shape[1] != H or t.shape[2] != W:
        t = F.interpolate(t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    return t


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000.yml")
    parser.add_argument("--checkpoint",  default="outputs/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000/best_stg1.pth")
    parser.add_argument("--test_dir",    default="/root/autodl-tmp/AIC/dataset/test/AIC2026_PHASE_2_1000")
    parser.add_argument("--out_dir",     default="/root/autodl-tmp/AIC/work_dir/submission_v10_caa_sahi_fixed")
    parser.add_argument("--conf_thresh", type=float, default=0.15)
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

    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)

    vis_dir    = os.path.join(args.test_dir, "visible")
    ir_dir     = os.path.join(args.test_dir, "infrared")
    depth_dir  = os.path.join(args.test_dir, "depth_p0")
    lbg_ir_dir = "/root/autodl-tmp/AIC/work_dir/test_lbg_ir"
    print(f"LBG-IR cache: {lbg_ir_dir}  (exists={os.path.isdir(lbg_ir_dir)})")

    vis_paths = sorted(glob.glob(os.path.join(vis_dir, "*.jpg")) +
                       glob.glob(os.path.join(vis_dir, "*.png")))
    print(f"Found {len(vis_paths)} test images in {vis_dir}")

    seam_margin   = 25.0
    max_patch_dim = 320.0
    P0_ONLY_CATS  = {1, 7, 11}   # boat, ball, tricycle

    processed = 0
    with torch.no_grad():
        for vis_path in vis_paths:
            stem       = os.path.splitext(os.path.basename(vis_path))[0]
            ir_path    = os.path.join(ir_dir,     stem + ".jpg")
            depth_path = os.path.join(depth_dir,  stem + ".png")
            lbg_path   = os.path.join(lbg_ir_dir, stem + ".jpg")
            if not os.path.exists(ir_path):
                ir_path  = os.path.join(ir_dir, stem + ".png")
            if not os.path.exists(lbg_path):
                lbg_path = os.path.join(lbg_ir_dir, stem + ".png")
            if not os.path.exists(ir_path):
                print(f"  [WARN] IR not found for {stem}, skipping"); continue
            if not os.path.exists(depth_path):
                print(f"  [WARN] depth not found for {stem}, skipping"); continue

            # ── Assemble CORRECT 6-channel tensor ─────────────────────────
            vis_pil  = Image.open(vis_path).convert("RGB")
            H, W     = vis_pil.size[1], vis_pil.size[0]
            rgb      = TF.pil_to_tensor(vis_pil).float().div(255.0).cuda()   # [3,H,W]

            ir_pil   = Image.open(ir_path).convert("L")
            ir_np_u8 = np.array(ir_pil, dtype=np.uint8)
            ir_gray  = torch.from_numpy(ir_np_u8.astype(np.float32)/255.0).unsqueeze(0).cuda()
            if ir_gray.shape[1] != H or ir_gray.shape[2] != W:
                ir_gray  = F.interpolate(ir_gray.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=False)[0]
                ir_np_u8 = (ir_gray[0].cpu().numpy()*255).clip(0,255).astype(np.uint8)

            # ch3: heat_IR = clip(IR_gray - RGB_gray, 0, 1)
            heat_ir = (ir_gray - rgb.mean(dim=0, keepdim=True)).clamp(0, 1)

            # ch4: LBG-IR
            if os.path.exists(lbg_path):
                lbg_pil = Image.open(lbg_path).convert("L")
                lbg_arr = np.array(lbg_pil, dtype=np.float32) / 255.0
                lbg_ir  = torch.from_numpy(lbg_arr).unsqueeze(0).cuda()
                if lbg_ir.shape[1] != H or lbg_ir.shape[2] != W:
                    lbg_ir = F.interpolate(lbg_ir.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=False)[0]
            else:
                lbg_ir = torch.from_numpy(build_lbg_ir(ir_np_u8)).float().unsqueeze(0).cuda()

            # ch5: depth
            depth = load_depth_tensor(depth_path, "cuda", H, W)

            raw = torch.cat([rgb, heat_ir, lbg_ir, depth], dim=0)  # [6,H,W]
            raw = apply_preproc(raw)

            W_orig, H_orig = float(W), float(H)

            # ── 3D Water Planar Prior ─────────────────────────────────────
            d_chan   = raw[5]
            bottom_d = d_chan[int(d_chan.shape[0] * 0.5):, :]
            valid_d  = bottom_d[bottom_d > 0.015]
            is_water_scene = valid_d.std().item() > 0.09 if len(valid_d) > 0 else False

            # ── Three views ───────────────────────────────────────────────
            p0     = F.interpolate(raw.unsqueeze(0), size=(640, 640),  mode="bilinear", align_corners=False)[0]
            canvas = F.interpolate(raw.unsqueeze(0), size=(640, 1138), mode="bilinear", align_corners=False)[0]
            p1 = canvas[:, :, 0:640]
            p2 = canvas[:, :, 498:1138]

            batch = torch.stack([p1, p2, p0], dim=0)
            norm  = (batch - mean) / std
            sizes = torch.tensor([[640., 640.], [640., 640.], [W_orig, H_orig]], device="cuda")

            out = model(norm)
            res = post(out, sizes, norm)

            scale_x = 1138.0 / W_orig
            scale_y = 640.0  / H_orig
            scales  = torch.tensor([scale_x, scale_y, scale_x, scale_y], device="cuda")

            b1 = res[0]["boxes"] / scales;  s1 = res[0]["scores"];  l1 = res[0]["labels"]
            b2 = (res[1]["boxes"] + torch.tensor([498.,0.,498.,0.], device="cuda")) / scales
            s2 = res[1]["scores"];  l2 = res[1]["labels"]
            b0 = res[2]["boxes"];   s0 = res[2]["scores"];  l0 = res[2]["labels"]

            # Seam / size filters
            p1_x2 = res[0]["boxes"][:, 2]
            p1_w  = res[0]["boxes"][:, 2] - res[0]["boxes"][:, 0]
            p1_h  = res[0]["boxes"][:, 3] - res[0]["boxes"][:, 1]
            keep1 = (p1_x2 < (640. - seam_margin)) & (p1_w < max_patch_dim) & (p1_h < max_patch_dim)

            p2_x1 = res[1]["boxes"][:, 0]
            p2_w  = res[1]["boxes"][:, 2] - res[1]["boxes"][:, 0]
            p2_h  = res[1]["boxes"][:, 3] - res[1]["boxes"][:, 1]
            keep2 = (p2_x1 > seam_margin) & (p2_w < max_patch_dim) & (p2_h < max_patch_dim)

            # Exclude P0-only cats from patches
            p0m1 = torch.zeros_like(keep1, dtype=torch.bool)
            p0m2 = torch.zeros_like(keep2, dtype=torch.bool)
            for c in P0_ONLY_CATS:
                p0m1 |= (l1 == c);  p0m2 |= (l2 == c)
            keep1_cat = keep1 & ~p0m1
            keep2_cat = keep2 & ~p0m2

            boxes  = torch.cat([b0, b1[keep1_cat], b2[keep2_cat]], dim=0)
            scores = torch.cat([s0, s1[keep1_cat], s2[keep2_cat]], dim=0)
            labels = torch.cat([l0, l1[keep1_cat], l2[keep2_cat]], dim=0)

            # ── Per-category NMS → submission lines ──────────────────────
            lines = []
            for c in range(12):
                if c == 1 and not is_water_scene:
                    continue
                if c in P0_ONLY_CATS:
                    m_p0 = l0 == c
                    if not m_p0.any(): continue
                    c_boxes = b0[m_p0];  c_scores = s0[m_p0];  iou_thr = 0.50
                else:
                    mask = labels == c
                    if not mask.any(): continue
                    c_boxes = boxes[mask];  c_scores = scores[mask];  iou_thr = 0.65

                conf_mask = c_scores >= args.conf_thresh
                c_boxes = c_boxes[conf_mask];  c_scores = c_scores[conf_mask]
                if c_boxes.shape[0] == 0: continue

                if c == 4:
                    c_scores = c_scores * 1.05

                keep_idx = nms(c_boxes, c_scores.clamp(max=1.0), iou_threshold=iou_thr)
                for b, s in zip(c_boxes[keep_idx].cpu().tolist(), c_scores[keep_idx].cpu().tolist()):
                    x1, y1, x2, y2 = b
                    cx = max(0., min(1., (x1+x2)/2./W_orig))
                    cy = max(0., min(1., (y1+y2)/2./H_orig))
                    bw = max(0., min(1., (x2-x1)/W_orig))
                    bh = max(0., min(1., (y2-y1)/H_orig))
                    lines.append((float(s), f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {min(s,1.):.6f}"))

            lines.sort(key=lambda x: x[0], reverse=True)
            lines = lines[:args.max_boxes]

            out_path = os.path.join(args.out_dir, stem + ".txt")
            with open(out_path, "w", newline="\r\n") as f:
                for _, line in lines:
                    f.write(line + "\n")

            processed += 1
            if processed % 50 == 0 or processed == 1:
                print(f"  [{processed}/{len(vis_paths)}] {stem}.txt -> {len(lines)} boxes")

    print(f"\nDone. {processed} files written to: {args.out_dir}")
    print(f"Submission TXT count: {len(glob.glob(os.path.join(args.out_dir, '*.txt')))} / {len(vis_paths)}")


if __name__ == "__main__":
    main()
