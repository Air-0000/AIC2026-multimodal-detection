#!/usr/bin/env python3
"""
Ablation: run CAA-SAHI eval with WRONG channel loading (ir3[:2] bug)
to quantify the multimodal channel corruption penalty.
Compare result with correct loading (57.40%) to get the delta.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import numpy as np

def apply_preproc(raw):
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

def load_image_tensor(path, device):
    """WRONG: load IR as RGB and take first 2 channels (the bug)."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000.yml")
    parser.add_argument("--checkpoint", default="outputs/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000/best_stg1.pth")
    args = parser.parse_args()

    from engine.core import YAMLConfig
    cfg = YAMLConfig(args.config)
    ds  = cfg.val_dataloader.dataset   # loads via correct pipeline normally

    print(f"Loading checkpoint: {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location="cpu")
    weights = ck["ema"]["module"] if "ema" in ck and "module" in ck.get("ema", {}) else ck.get("model", ck)
    cfg.model.load_state_dict(weights, strict=False)
    model = cfg.model.cuda().eval()
    post  = cfg.postprocessor

    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)

    # Locate val image folders (same as dataset but we'll reload images with WRONG channels)
    val_img_dir = ds.root           # visible folder
    val_ir_dir  = ds.ir_folder      # heat IR folder (train_aligned_ir cache)
    # We will IGNORE the dataset's correct load_item and instead do the WRONG loading

    # To replicate the submission bug: load IR from ir_folder as RGB, take [:2]
    # IR folder contains grayscale PNGs stored as 1-channel (convert("L") gives the right thing,
    # but the bug does convert("RGB") and takes first 2 channels → ch3≈ch4)

    all_preds = []
    seam_margin   = 25.0
    max_patch_dim = 320.0
    P0_ONLY_CATS  = {1, 7, 11}

    print("Running WRONG-channel ablation on val set...")
    with torch.no_grad():
        for idx in range(len(ds)):
            img_id   = ds.ids[idx]
            img_info = ds.coco.loadImgs(img_id)[0]
            img_name = img_info["file_name"]

            vis_path   = os.path.join(ds.root, img_name)
            stem       = os.path.splitext(os.path.basename(img_name))[0]
            ir_path    = os.path.join(ds.ir_folder,    stem + ".jpg") if hasattr(ds, 'ir_folder') and ds.ir_folder else None
            depth_path = os.path.join(ds.depth_folder, stem + ".png") if hasattr(ds, 'depth_folder') and ds.depth_folder else None

            if ir_path and not os.path.exists(ir_path):
                ir_path = os.path.join(ds.ir_folder, stem + ".png")
            if depth_path and not os.path.exists(depth_path):
                depth_path = os.path.join(ds.depth_folder, stem + ".jpg")

            # Load RGB correctly
            vis_pil = Image.open(vis_path).convert("RGB")
            H, W    = vis_pil.size[1], vis_pil.size[0]
            arr = np.array(vis_pil, dtype=np.float32) / 255.0
            rgb = torch.from_numpy(arr).permute(2, 0, 1).cuda()  # [3,H,W]

            # ── BUG: load IR as RGB, take first 2 channels ──
            if ir_path and os.path.exists(ir_path):
                ir_img = Image.open(ir_path).convert("RGB")  # BUG: should be L
                ir_arr = np.array(ir_img, dtype=np.float32) / 255.0
                ir3    = torch.from_numpy(ir_arr).permute(2, 0, 1).cuda()  # [3,H,W]
                if ir3.shape[1] != H or ir3.shape[2] != W:
                    ir3 = F.interpolate(ir3.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=False)[0]
                ch34 = ir3[:2]   # BUG: ch3≈ch4 (R and G of grayscale IR image)
            else:
                ch34 = torch.zeros(2, H, W, device="cuda")

            # Load depth correctly
            if depth_path and os.path.exists(depth_path):
                d_img = Image.open(depth_path)
                d_arr = np.array(d_img, dtype=np.float32)
                mode  = getattr(d_img, 'mode', '')
                scale = 19999.0 if mode in ('I;16','I') else (65535.0 if d_arr.max() > 255.5 else 255.0)
                d_arr = (d_arr / scale).clip(0, 1)
                depth = torch.from_numpy(d_arr).unsqueeze(0).cuda()
                if depth.shape[1] != H or depth.shape[2] != W:
                    depth = F.interpolate(depth.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=False)[0]
            else:
                depth = torch.zeros(1, H, W, device="cuda")

            raw = torch.cat([rgb, ch34, depth], dim=0)   # [6,H,W] — BUG channel layout
            raw = apply_preproc(raw)

            W_orig, H_orig = float(W), float(H)

            d_chan   = raw[5]
            bottom_d = d_chan[int(d_chan.shape[0]*0.5):, :]
            valid_d  = bottom_d[bottom_d > 0.015]
            is_water_scene = valid_d.std().item() > 0.09 if len(valid_d) > 0 else False

            p0     = F.interpolate(raw.unsqueeze(0), size=(640,640),  mode="bilinear", align_corners=False)[0]
            canvas = F.interpolate(raw.unsqueeze(0), size=(640,1138), mode="bilinear", align_corners=False)[0]
            p1 = canvas[:, :, 0:640]
            p2 = canvas[:, :, 498:1138]

            batch = torch.stack([p1, p2, p0], dim=0)
            norm  = (batch - mean) / std
            sizes = torch.tensor([[640.,640.],[640.,640.],[W_orig,H_orig]], device="cuda")

            out = model(norm)
            res = post(out, sizes, norm)

            scale_x = 1138.0/W_orig; scale_y = 640.0/H_orig
            scales  = torch.tensor([scale_x, scale_y, scale_x, scale_y], device="cuda")

            b1 = res[0]["boxes"]/scales; s1=res[0]["scores"]; l1=res[0]["labels"]
            b2 = (res[1]["boxes"]+torch.tensor([498.,0.,498.,0.],device="cuda"))/scales
            s2=res[1]["scores"]; l2=res[1]["labels"]
            b0=res[2]["boxes"]; s0=res[2]["scores"]; l0=res[2]["labels"]

            p1_x2=res[0]["boxes"][:,2]; p1_w=res[0]["boxes"][:,2]-res[0]["boxes"][:,0]; p1_h=res[0]["boxes"][:,3]-res[0]["boxes"][:,1]
            keep1=(p1_x2<(640.-seam_margin))&(p1_w<max_patch_dim)&(p1_h<max_patch_dim)
            p2_x1=res[1]["boxes"][:,0]; p2_w=res[1]["boxes"][:,2]-res[1]["boxes"][:,0]; p2_h=res[1]["boxes"][:,3]-res[1]["boxes"][:,1]
            keep2=(p2_x1>seam_margin)&(p2_w<max_patch_dim)&(p2_h<max_patch_dim)

            p0m1=torch.zeros_like(keep1,dtype=torch.bool); p0m2=torch.zeros_like(keep2,dtype=torch.bool)
            for c in P0_ONLY_CATS: p0m1|=(l1==c); p0m2|=(l2==c)
            keep1_cat=keep1&~p0m1; keep2_cat=keep2&~p0m2

            boxes =torch.cat([b0,b1[keep1_cat],b2[keep2_cat]],dim=0)
            scores=torch.cat([s0,s1[keep1_cat],s2[keep2_cat]],dim=0)
            labels=torch.cat([l0,l1[keep1_cat],l2[keep2_cat]],dim=0)

            for c in range(12):
                if c==1 and not is_water_scene: continue
                if c in P0_ONLY_CATS:
                    m_p0=l0==c
                    if not m_p0.any(): continue
                    c_boxes=b0[m_p0]; c_scores=s0[m_p0]; iou_thr=0.50
                else:
                    mask=labels==c
                    if not mask.any(): continue
                    c_boxes=boxes[mask]; c_scores=scores[mask]; iou_thr=0.65
                if c==4: c_scores=c_scores*1.05
                keep_idx=nms(c_boxes, c_scores.clamp(max=1.0), iou_threshold=iou_thr)
                for b,s in zip(c_boxes[keep_idx].cpu().tolist(), c_scores[keep_idx].cpu().tolist()):
                    all_preds.append({"image_id":img_id,"category_id":c,
                                      "bbox":[b[0],b[1],b[2]-b[0],b[3]-b[1]],"score":s})

            if (idx+1) % 100 == 0:
                print(f"  [{idx+1}/{len(ds)}] done")

    coco_gt = ds.coco
    coco_dt = coco_gt.loadRes(all_preds)
    ce = COCOeval(coco_gt, coco_dt, "bbox")
    ce.evaluate(); ce.accumulate(); ce.summarize()
    mAP_wrong = ce.stats[0] * 100

    print(f"\n{'='*50}")
    print(f"  WRONG-channel CAA-SAHI mAP: {mAP_wrong:.2f}%")
    print(f"  Correct-channel CAA-SAHI:   57.40%")
    print(f"  Channel bug penalty:        {57.40 - mAP_wrong:+.2f}%")
    print(f"{'='*50}")

    print("\n=== Per-category ===")
    for cat_id, cat_info in coco_gt.cats.items():
        ce_c = COCOeval(coco_gt, coco_dt, "bbox")
        ce_c.params.catIds = [cat_id]
        ce_c.evaluate(); ce_c.accumulate(); ce_c.summarize()
        print(f"  {cat_info['name']:<15}: AP={ce_c.stats[0]*100:.2f}%")

if __name__ == "__main__":
    main()
