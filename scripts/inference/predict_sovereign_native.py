#!/usr/bin/env python3
"""
Sovereign DEIM Native Unified Predictor & Submission Generator
==============================================================
Zero External Post-Processing. Zero Manual Heuristics. Zero WBF.

Preprocessed Input Modalities (100% Matching Training & Documentation):
  - Visible (RGB): 3 channels, loaded via PIL Image, normalized ImageNet mean/std
  - Aligned IR: 1 channel, precomputed offline in work_dir/test_aligned_ir, normalized (x - 0.5) / 0.5
  - LBG IR: 1 channel, precomputed offline in work_dir/test_lbg_ir, normalized (x - 0.5) / 0.5
  - Metric Depth P0: 1 channel, precomputed 16-bit in depth_p0, normalized (x - 0.5) / 0.5

End-to-End Pipeline:
  1. Assemble 6-channel tensor [RGB(3), Aligned_IR(1), LBG_IR(1), Depth(1)]
  2. Bilinear resize to (640, 640)
  3. Full 6-channel normalization:
     mean = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
     std  = [0.229, 0.224, 0.225, 0.5, 0.5, 0.5]
  4. Native forward pass: outputs = model(images)
     - Internalized CAA-SAHI 3-View canvas projection (Aspect-ratio distortion free)
     - GPU-Native Batched NMS (IoU=0.65)
  5. PostProcessor: results = postprocessor(outputs, orig_target_sizes)
  6. Format to Official AIC Submission Specifications:
     - 1 text file per image: <class_id> <cx> <cy> <w> <h> <confidence>
     - Normalized coordinates in [0, 1]
     - Max 100 boxes per image
     - CRLF line endings (\\r\\n)
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

aic_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in [aic_root, os.path.join(aic_root, 'vendor', 'DEIMv2'), os.path.join(aic_root, 'scripts', 'lib'),
          '/root/autodl-tmp/AIC', '/root/autodl-tmp/AIC/vendor/DEIMv2', '/root/autodl-tmp/AIC/scripts/lib']:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from engine.core import YAMLConfig


MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)


def compute_phi_vt_gate(depth_arr):
    depth_m = np.clip(depth_arr / 1000.0 if depth_arr.max() > 255.0 else depth_arr / 20.0, 0.5, 50.0)
    dz_dx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    div_n = np.abs(cv2.Sobel(-dz_dx / norm, cv2.CV_32F, 1, 0) + cv2.Sobel(-dz_dy / norm, cv2.CV_32F, 0, 1))
    flat_score = np.exp(- (div_n / 0.15)**2)
    phi_vt_gate = (1.0 - 0.75 * flat_score).astype(np.float32)
    return phi_vt_gate


def load_multimodal_tensor(img_stem, test_dir, device, aligned_ir_dir=None, lbg_ir_dir=None, depth_dir=None):
    """
    Standard native multi-modal tensor reader matching CocoDetectionMultiModal exactly:
      ch 0,1,2 = RGB visible
      ch 3     = Aligned IR (normalized [-1, 1])
      ch 4     = LBG IR (precomputed offline, normalized [-1, 1])
      ch 5     = P0 Metric Depth (raw 16-bit mm / 19999.0, normalized [-1, 1])
    """
    # 1. RGB Visible
    vis_cand = glob.glob(os.path.join(test_dir, 'visible', f'{img_stem}.*'))
    if not vis_cand:
        vis_cand = glob.glob(os.path.join(test_dir, f'{img_stem}.*'))
    if not vis_cand:
        raise FileNotFoundError(f"Visible image not found for stem {img_stem} in {test_dir}")
    
    vis_pil = Image.open(vis_cand[0]).convert("RGB")
    orig_w, orig_h = vis_pil.size
    rgb_t = TF.pil_to_tensor(vis_pil).float() / 255.0 # (3, H, W)

    # 2. Aligned IR (Precomputed offline)
    ir_cand = []
    if aligned_ir_dir and os.path.exists(aligned_ir_dir):
        ir_cand = glob.glob(os.path.join(aligned_ir_dir, f'{img_stem}.*'))
    if not ir_cand:
        ir_cand = glob.glob(os.path.join('work_dir', 'test_aligned_ir', f'{img_stem}.*')) + \
                  glob.glob(os.path.join('/root/autodl-tmp/AIC/work_dir', 'test_aligned_ir', f'{img_stem}.*'))
    if not ir_cand:
        ir_cand = glob.glob(os.path.join(test_dir, 'aligned_ir', f'{img_stem}.*')) + \
                  glob.glob(os.path.join(test_dir, 'infrared', f'{img_stem}.*'))

    if ir_cand:
        ir_pil = Image.open(ir_cand[0]).convert("L")
        ir_t = TF.pil_to_tensor(ir_pil).float() / 255.0
        if ir_t.shape[1:] != (orig_h, orig_w):
            ir_t = TF.resize(ir_t, [orig_h, orig_w])
    else:
        ir_t = rgb_t.mean(dim=0, keepdim=True)

    # 3. LBG IR (Precomputed offline, matching train_lbg_ir)
    lbg_cand = []
    if lbg_ir_dir and os.path.exists(lbg_ir_dir):
        lbg_cand = glob.glob(os.path.join(lbg_ir_dir, f'{img_stem}.*'))
    if not lbg_cand:
        lbg_cand = glob.glob(os.path.join('work_dir', 'test_lbg_ir', f'{img_stem}.*')) + \
                   glob.glob(os.path.join('/root/autodl-tmp/AIC/work_dir', 'test_lbg_ir', f'{img_stem}.*'))

    if lbg_cand:
        lbg_pil = Image.open(lbg_cand[0]).convert("L")
        lbg_t = TF.pil_to_tensor(lbg_pil).float() / 255.0
        if lbg_t.shape[1:] != (orig_h, orig_w):
            lbg_t = TF.resize(lbg_t, [orig_h, orig_w])
    else:
        # Fallback to zeros if missing
        lbg_t = torch.zeros((1, orig_h, orig_w), dtype=torch.float32)

    # 4. Metric Depth (P0 16-bit mm / 19999.0)
    d_cand = []
    if depth_dir and os.path.exists(depth_dir):
        d_cand = glob.glob(os.path.join(depth_dir, f'{img_stem}.*'))
    if not d_cand:
        d_cand = glob.glob(os.path.join(test_dir, 'depth_p0', f'{img_stem}.*')) + \
                 glob.glob(os.path.join(test_dir, 'depth', f'{img_stem}.*')) + \
                 glob.glob(os.path.join('/root/autodl-tmp/AIC/dataset/test/AIC2026_PHASE_2_1000/depth_p0', f'{img_stem}.*'))

    if d_cand:
        d_img = Image.open(d_cand[0])
        d_arr = np.array(d_img, dtype=np.float32)
        scale = 19999.0 if getattr(d_img, 'mode', '') in ('I;16', 'I') else (65535.0 if d_arr.max() > 255.5 else 255.0)
        d_norm = np.clip(d_arr / scale, 0.0, 1.0)
        d_t = torch.from_numpy(d_norm).unsqueeze(0)
        if d_t.shape[1:] != (orig_h, orig_w):
            d_t = TF.resize(d_t, [orig_h, orig_w])
        phi_vt = compute_phi_vt_gate(d_arr)
    else:
        d_t = torch.zeros((1, orig_h, orig_w), dtype=torch.float32)
        phi_vt = np.ones((orig_h, orig_w), dtype=np.float32)

    # Concatenate 6 channels: [RGB(3), Aligned_IR(1), LBG_IR(1), Depth(1)]
    tensor_6ch = torch.cat([rgb_t, ir_t, lbg_t, d_t], dim=0) # (6, H, W) in [0, 1]

    # Bilinear resize to 640x640 (matching val_dataloader transforms: Resize([640, 640]))
    tensor_640 = TF.resize(tensor_6ch, [640, 640])

    # Full 6-channel normalization matching val_dataloader:
    # mean: [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
    # std:  [0.229, 0.224, 0.225, 0.5, 0.5, 0.5]
    mean = MEAN_6CH.to(device)
    std = STD_6CH.to(device)
    t = (tensor_640.to(device) - mean) / std

    return t, orig_h, orig_w, phi_vt


def main():
    parser = argparse.ArgumentParser(description="Sovereign DEIM Native Unified Predictor")
    parser.add_argument("-c", "--config", required=True, help="Path to model YAML config")
    parser.add_argument("-r", "--checkpoint", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--test-dir", required=True, help="Directory containing test dataset")
    parser.add_argument("--aligned-ir-dir", default=None, help="Directory containing precomputed aligned IR")
    parser.add_argument("--lbg-ir-dir", default=None, help="Directory containing precomputed LBG IR")
    parser.add_argument("--depth-dir", default=None, help="Directory containing precomputed P0 metric depth")
    parser.add_argument("-o", "--output-dir", default="work_dir/submission_native", help="Output submission directory")
    parser.add_argument("--score-threshold", type=float, default=0.02, help="Confidence cutoff (default: 0.02)")
    parser.add_argument("--max-boxes", type=int, default=100, help="Maximum boxes per image (default: 100)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone-feature-gate", action="store_true", help="Enable Method Joint-3 Phi_VT Backbone feature layer gating")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading Sovereign model from config: {args.config}")
    cfg = YAMLConfig(args.config)
    model = cfg.model.to(device)

    print(f"Loading weights from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    weights = ckpt['ema']['module'] if ('ema' in ckpt and 'module' in ckpt['ema']) else ckpt.get('model', ckpt)
    model.load_state_dict(weights, strict=False)
    model.eval()

    if args.backbone_feature_gate:
        current_phi_vt_tensor = [None]
        def ir_stem_hook(module, input, output):
            if current_phi_vt_tensor[0] is not None:
                phi_vt_t = current_phi_vt_tensor[0]
                c2, c3 = output
                gate_c2 = F.interpolate(phi_vt_t, size=c2.shape[2:], mode='bilinear', align_corners=False)
                gate_c3 = F.interpolate(phi_vt_t, size=c3.shape[2:], mode='bilinear', align_corners=False)
                return (c2 * gate_c2, c3 * gate_c3)
            return output
        model.backbone.ir_stem.register_forward_hook(ir_stem_hook)
        print("[DEIM] Backbone Feature-layer Phi_VT Cross-modal Gate: ACTIVATED (water/ground suppression)")

    postprocessor = cfg.postprocessor.to(device) if hasattr(cfg, 'postprocessor') else None

    # Discover all test images
    vis_dir = os.path.join(args.test_dir, 'visible')
    if os.path.exists(vis_dir):
        image_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
    else:
        image_files = sorted(glob.glob(os.path.join(args.test_dir, "*.*")))

    stems = [Path(f).stem for f in image_files if not f.endswith('.txt')]
    print(f"Found {len(stems)} test images to predict. Output: {args.output_dir}")
    print(f"Configuration: threshold={args.score_threshold}, max_boxes={args.max_boxes}")

    total_boxes = 0
    with torch.no_grad():
        for idx, stem in enumerate(stems):
            try:
                tensor, orig_h, orig_w, phi_vt = load_multimodal_tensor(
                    stem, args.test_dir, device,
                    aligned_ir_dir=args.aligned_ir_dir,
                    lbg_ir_dir=args.lbg_ir_dir,
                    depth_dir=args.depth_dir
                )
            except Exception as e:
                print(f"[{idx+1}/{len(stems)}] Error loading {stem}: {e}")
                with open(os.path.join(args.output_dir, f"{stem}.txt"), "wb") as f:
                    f.write(b"")
                continue

            if args.backbone_feature_gate:
                current_phi_vt_tensor[0] = torch.from_numpy(phi_vt).unsqueeze(0).unsqueeze(0).to(device)

            # Native forward pass (Internalized CAA-SAHI runs automatically inside model.forward)
            inputs = tensor.unsqueeze(0) # (1, 6, 640, 640)
            orig_target_sizes = torch.tensor([[orig_w, orig_h]], device=device)

            outputs = model(inputs)

            if postprocessor is not None:
                results = postprocessor(outputs, orig_target_sizes)[0]
                labels = results['labels'].cpu().numpy()
                boxes = results['boxes'].cpu().numpy() # [x1, y1, x2, y2] absolute pixels
                scores = results['scores'].cpu().numpy()
            else:
                pred_logits = outputs['pred_logits'][0]
                pred_boxes = outputs['pred_boxes'][0]
                prob = torch.sigmoid(pred_logits)
                scores, labels = prob.max(-1)
                boxes = pred_boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()

            # Filter by score threshold & sort descending
            mask = scores >= args.score_threshold
            valid_labels = labels[mask]
            valid_boxes = boxes[mask]
            valid_scores = scores[mask]

            sort_idx = np.argsort(-valid_scores)[:args.max_boxes]
            valid_labels = valid_labels[sort_idx]
            valid_boxes = valid_boxes[sort_idx]
            valid_scores = valid_scores[sort_idx]

            out_lines = []
            for lbl, box, sc in zip(valid_labels, valid_boxes, valid_scores):
                if len(box) == 4:
                    if postprocessor is not None:
                        # box is [x1, y1, x2, y2] in absolute pixels
                        x1, y1, x2, y2 = box
                        cx = ((x1 + x2) / 2.0) / orig_w
                        cy = ((y1 + y2) / 2.0) / orig_h
                        w = (x2 - x1) / orig_w
                        h = (y2 - y1) / orig_h
                    else:
                        # box is [cx, cy, w, h] normalized
                        cx, cy, w, h = box

                    cx = np.clip(cx, 0.0, 1.0)
                    cy = np.clip(cy, 0.0, 1.0)
                    w = np.clip(w, 0.0, 1.0)
                    h = np.clip(h, 0.0, 1.0)
                    sc = np.clip(sc, 0.0, 1.0)

                    out_lines.append(f"{int(lbl)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {sc:.6f}")

            # Write CRLF formatted file
            out_path = os.path.join(args.output_dir, f"{stem}.txt")
            with open(out_path, "wb") as f:
                if out_lines:
                    content = "\r\n".join(out_lines) + "\r\n"
                    f.write(content.encode("utf-8"))
                    total_boxes += len(out_lines)
                else:
                    f.write(b"")

            if (idx + 1) % 100 == 0 or idx == len(stems) - 1:
                print(f"[{idx+1}/{len(stems)}] Processed {stem}, total valid boxes so far: {total_boxes}")

    print(f"\nSuccessfully generated native submission for {len(stems)} images at: {args.output_dir}")
    print(f"Total detections generated: {total_boxes} (avg {total_boxes/max(1, len(stems)):.1f} boxes/img)")


if __name__ == '__main__':
    main()
