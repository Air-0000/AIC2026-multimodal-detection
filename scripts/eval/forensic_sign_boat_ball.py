#!/usr/bin/env python3
"""
Forensic Analysis of Sign, Boat, and Ball Failure Modes in Sovereign V9.
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.ops import box_iou, nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from engine.core import YAMLConfig

def apply_preproc(raw):
    rgb = raw[:3]
    blur = F.avg_pool2d(rgb.unsqueeze(0), 3, stride=1, padding=1)[0]
    high = rgb - blur
    rgb_usm = (rgb + 0.15 * high).clamp(0, 1)
    
    d = raw[5:6]
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=raw.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=raw.device).view(1, 1, 3, 3)
    gx = F.conv2d(d.unsqueeze(0), sobel_x, padding=1)
    gy = F.conv2d(d.unsqueeze(0), sobel_y, padding=1)
    g = torch.sqrt(gx*gx + gy*gy)[0]
    edge_mask = torch.clamp((g - 0.015) / 0.03, 0.0, 1.0)
    
    ir = raw[3:5]
    ir_blur = F.avg_pool2d(ir.unsqueeze(0), 3, stride=1, padding=1)[0]
    ir_high = ir - ir_blur
    ir_boost = (ir + 0.25 * edge_mask * ir_high).clamp(0, 1)
    
    return torch.cat([rgb_usm, ir_boost, raw[5:6]], dim=0)

def main():
    cfg = YAMLConfig("configs/deimv2/deim_hgnetv2_x_aic_sovereign_v9_snr_fdr_rtx6000.yml")
    ds = cfg.val_dataloader.dataset

    checkpoint = "outputs/deim_hgnetv2_x_aic_sovereign_v9_snr_fdr_rtx6000/best_stg1.pth"
    ck = torch.load(checkpoint, map_location="cpu")
    weights = ck["ema"]["module"] if "ema" in ck and "module" in ck["ema"] else ck.get("model", ck)
    cfg.model.load_state_dict(weights, strict=False)
    model = cfg.model.cuda().eval()
    post = cfg.postprocessor

    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5], device="cuda").view(6, 1, 1)

    seam_margin = 25.0
    max_patch_dim = 320.0

    preds_by_cat = {1: [], 4: [], 7: []} # boat, sign, ball
    gts_by_cat = {1: [], 4: [], 7: []}

    print("Running inference and extracting multimodal physical features...")
    with torch.no_grad():
        for idx in range(len(ds)):
            raw, t = ds.load_item(idx)
            img_id = ds.ids[idx]
            raw_cuda = raw.cuda()
            raw_proc = apply_preproc(raw_cuda)

            W_orig = float(t["orig_size"][0].item())
            H_orig = float(t["orig_size"][1].item())

            # Gather GTs for this image
            img_anns = ds.coco.imgToAnns.get(img_id, [])
            for ann in img_anns:
                cid = ann["category_id"]
                if cid in gts_by_cat:
                    bx, by, bw, bh = ann["bbox"]
                    gts_by_cat[cid].append({
                        "img_id": img_id,
                        "box": [bx, by, bx+bw, by+bh],
                        "area": bw * bh,
                        "aspect": bw / (bh + 1e-4)
                    })

            # Check water scene
            d = raw_cuda[5]
            H_raw = d.shape[0]
            bottom_d = d[int(H_raw * 0.5):, :]
            valid_d = bottom_d[bottom_d > 0.015]
            depth_std = valid_d.std().item() if len(valid_d) > 0 else 0
            is_water_scene = depth_std > 0.09

            p0 = F.interpolate(raw_proc.unsqueeze(0), size=(640, 640), mode="bilinear", align_corners=False)[0]
            canvas = F.interpolate(raw_proc.unsqueeze(0), size=(640, 1138), mode="bilinear", align_corners=False)[0]
            p1 = canvas[:, :, 0:640]
            p2 = canvas[:, :, 498:1138]

            batch = torch.stack([p1, p2, p0], dim=0)
            norm = (batch - mean) / std
            sizes = torch.tensor([[640., 640.], [640., 640.], [W_orig, H_orig]], device="cuda")

            out = model(norm)
            res = post(out, sizes, norm)

            scale_x = 1138.0 / W_orig
            scale_y = 640.0 / H_orig
            scales = torch.tensor([scale_x, scale_y, scale_x, scale_y], device="cuda")

            b1 = res[0]["boxes"] / scales
            s1 = res[0]["scores"]
            l1 = res[0]["labels"]

            offset = torch.tensor([498.0, 0.0, 498.0, 0.0], device="cuda")
            b2 = (res[1]["boxes"] + offset) / scales
            s2 = res[1]["scores"]
            l2 = res[1]["labels"]

            p1_x2_canvas = res[0]["boxes"][:, 2]
            p1_w = res[0]["boxes"][:, 2] - res[0]["boxes"][:, 0]
            p1_h = res[0]["boxes"][:, 3] - res[0]["boxes"][:, 1]
            keep1 = (p1_x2_canvas < (640.0 - seam_margin)) & (p1_w < max_patch_dim) & (p1_h < max_patch_dim)

            p2_x1_patch = res[1]["boxes"][:, 0]
            p2_w = res[1]["boxes"][:, 2] - res[1]["boxes"][:, 0]
            p2_h = res[1]["boxes"][:, 3] - res[1]["boxes"][:, 1]
            keep2 = (p2_x1_patch > seam_margin) & (p2_w < max_patch_dim) & (p2_h < max_patch_dim)

            b0 = res[2]["boxes"]
            s0 = res[2]["scores"]
            l0 = res[2]["labels"]

            keep1_cat = keep1 & (l1 != 1) & (l1 != 7)
            keep2_cat = keep2 & (l2 != 1) & (l2 != 7)

            boxes = torch.cat([b0, b1[keep1_cat], b2[keep2_cat]], dim=0)
            scores = torch.cat([s0, s1[keep1_cat], s2[keep2_cat]], dim=0)
            labels = torch.cat([l0, l1[keep1_cat], l2[keep2_cat]], dim=0)

            for c in [1, 4, 7]:
                if c == 1 and not is_water_scene:
                    continue
                mask = labels == c
                if not mask.any():
                    continue
                c_boxes = boxes[mask]
                c_scores = scores[mask]
                if c in [1, 7]:
                    m_p0 = l0 == c
                    if not m_p0.any():
                        continue
                    c_boxes = b0[m_p0]
                    c_scores = s0[m_p0]
                    keep_idx = nms(c_boxes, c_scores, iou_threshold=0.50)
                else:
                    keep_idx = nms(c_boxes, c_scores, iou_threshold=0.65)

                k_boxes = c_boxes[keep_idx]
                k_scores = c_scores[keep_idx]

                for b, s in zip(k_boxes, k_scores):
                    x1, y1, x2, y2 = b.cpu().tolist()
                    w = x2 - x1
                    h = y2 - y1
                    # Crop raw modalities to check physical properties
                    # raw has shape [6, H, W] in [0, 1]
                    ix1 = max(0, min(int(x1), int(W_orig)-1))
                    ix2 = max(ix1+1, min(int(x2), int(W_orig)))
                    iy1 = max(0, min(int(y1), int(H_orig)-1))
                    iy2 = max(iy1+1, min(int(y2), int(H_orig)))
                    
                    crop_rgb = raw_cuda[:3, iy1:iy2, ix1:ix2]
                    crop_ir = raw_cuda[3:5, iy1:iy2, ix1:ix2]
                    crop_d = raw_cuda[5:6, iy1:iy2, ix1:ix2]
                    
                    rgb_mean = crop_rgb.mean().item()
                    ir_mean = crop_ir[0].mean().item()
                    ir_lbg_mean = crop_ir[1].mean().item()
                    d_mean = crop_d.mean().item()
                    
                    preds_by_cat[c].append({
                        "img_id": img_id,
                        "box": [x1, y1, x2, y2],
                        "score": s.item(),
                        "w": w,
                        "h": h,
                        "area": w * h,
                        "aspect": w / (h + 1e-4),
                        "rgb_mean": rgb_mean,
                        "ir_mean": ir_mean,
                        "ir_lbg_mean": ir_lbg_mean,
                        "d_mean": d_mean
                    })

    # Now analyze sign predictions (c=4)
    print("\n" + "="*60)
    print("DETAILED FORENSIC AUDIT: SIGN (Category 4)")
    print("="*60)
    sign_preds = preds_by_cat[4]
    sign_gts = gts_by_cat[4]
    print(f"Total Sign GTs: {len(sign_gts)}, Total Sign Preds: {len(sign_preds)}")

    # Match predictions to GT per image
    tp_signs = []
    fp_signs = []
    for p in sign_preds:
        img_id = p["img_id"]
        img_gts = [g for g in sign_gts if g["img_id"] == img_id]
        if not img_gts:
            fp_signs.append(p)
            continue
        p_box = torch.tensor([p["box"]])
        gt_boxes = torch.tensor([g["box"] for g in img_gts])
        ious = box_iou(p_box, gt_boxes)[0]
        max_iou, max_idx = ious.max(dim=0)
        p["max_iou"] = max_iou.item()
        p["matched_gt"] = img_gts[max_idx.item()]
        if max_iou >= 0.5:
            tp_signs.append(p)
        else:
            fp_signs.append(p)

    print(f"True Positives: {len(tp_signs)}, False Positives: {len(fp_signs)}")
    print(f"\n--- True Positives (N={len(tp_signs)}) ---")
    tp_scores = [p["score"] for p in tp_signs]
    tp_rgb = [p["rgb_mean"] for p in tp_signs]
    tp_ir = [p["ir_mean"] for p in tp_signs]
    tp_aspect = [p["aspect"] for p in tp_signs]
    print(f"Score: mean={np.mean(tp_scores):.3f}, min={np.min(tp_scores):.3f}, max={np.max(tp_scores):.3f}")
    print(f"RGB mean: {np.mean(tp_rgb):.3f}, IR mean: {np.mean(tp_ir):.3f}, IR/RGB ratio: {np.mean(tp_ir)/(np.mean(tp_rgb)+1e-4):.3f}")
    print(f"Aspect ratio (w/h): mean={np.mean(tp_aspect):.3f}, min={np.min(tp_aspect):.3f}, max={np.max(tp_aspect):.3f}")

    print(f"\n--- False Positives (N={len(fp_signs)}) ---")
    fp_scores = [p["score"] for p in fp_signs]
    fp_rgb = [p["rgb_mean"] for p in fp_signs]
    fp_ir = [p["ir_mean"] for p in fp_signs]
    fp_aspect = [p["aspect"] for p in fp_signs]
    print(f"Score: mean={np.mean(fp_scores):.3f}, min={np.min(fp_scores):.3f}, max={np.max(fp_scores):.3f}")
    print(f"RGB mean: {np.mean(fp_rgb):.3f}, IR mean: {np.mean(fp_ir):.3f}, IR/RGB ratio: {np.mean(fp_ir)/(np.mean(fp_rgb)+1e-4):.3f}")
    print(f"Aspect ratio (w/h): mean={np.mean(fp_aspect):.3f}, min={np.min(fp_aspect):.3f}, max={np.max(fp_aspect):.3f}")

    # Inspect the highest-score False Positives
    fp_signs_sorted = sorted(fp_signs, key=lambda x: x["score"], reverse=True)
    print("\nTop 15 Highest-Confidence Sign False Positives:")
    for i, fp in enumerate(fp_signs_sorted[:15]):
        iou_str = f"max_iou={fp.get('max_iou', 0):.2f}"
        print(f"  [{i+1}] Img {fp['img_id']}: score={fp['score']:.3f}, box={[round(x,1) for x in fp['box']]}, "
              f"w={fp['w']:.1f}, h={fp['h']:.1f}, aspect={fp['aspect']:.2f}, "
              f"RGB={fp['rgb_mean']:.3f}, IR={fp['ir_mean']:.3f}, {iou_str}")

if __name__ == "__main__":
    main()
