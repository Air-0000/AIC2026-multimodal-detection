#!/usr/bin/env python3
"""
Sovereign V23 Test Set Visual Inspection Script
===============================================
Runs safe, low-VRAM inference using the latest Epoch 65 checkpoint (best_stg2.pth)
on 8 representative test images from Test Set 1.
Generates:
1. High-resolution annotated RGB images with predictions.
2. Dual-modality side-by-side comparison (RGB vs Aligned IR).
3. A comprehensive visual gallery artifact for user inspection.
"""

import os
import sys
import glob
import time
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.v2 as T
import torch.nn.functional as F

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

CLASS_COLORS = {
    0: (255, 220, 0),    # person: cyan-yellow (BGR: blue-green-red) -> yellow/gold
    1: (0, 165, 255),    # boat: orange
    2: (0, 255, 0),      # animal: bright green
    3: (255, 191, 0),    # seat: deep sky blue
    4: (0, 0, 255),      # sign: bright red
    5: (255, 0, 255),    # bicycle: magenta
    6: (0, 255, 255),    # car: bright yellow (BGR: 0, 255, 255)
    7: (0, 215, 255),    # ball: gold
    8: (180, 105, 255),  # light: light pink/purple
    9: (50, 205, 50),    # garbage can: lime green
    10: (255, 100, 100), # uav: blue-violet
    11: (0, 200, 200),   # tricycle: dark cyan
}

TARGET_STEMS = [
    ("003429", "多目标城市场景 (行人/车辆/指示牌/路灯/自行车)"),
    ("000021_015_00000073", "复杂弱目标与三轮车 (小球/三轮车/指示牌/行人)"),
    ("003268", "水域微光场景 (水面船只与倒影解耦)"),
    ("00000010", "高难开阔空域 (低空微小无人机/车辆/行人)"),
    ("000153", "夜间暗光高密街区 (路灯/车辆/立柱/行人密集流)"),
    ("hehe_46_00000008", "微小地砖小球与长椅 (小球/座椅/垃圾桶/暗区目标)"),
    ("00000018", "白天典型交通路口 (高密度车流与行人过街)"),
    ("001672", "夜间高光斑逆光交叉路口 (强车灯/反光与弱目标)"),
]


def read_depth_raw16(folder, stem):
    for ext in ('.png', '.jpg', '.jpeg'):
        p = os.path.join(folder, f"{stem}{ext}")
        if os.path.exists(p):
            im = Image.open(p)
            arr = np.array(im).astype(np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            scale = 19999.0 if getattr(im, 'mode', '') in ('I;16', 'I') else 255.0
            arr = arr / scale
            return torch.from_numpy(arr.clip(0.0, 1.0)).unsqueeze(0)
    raise FileNotFoundError(f"Depth not found for {stem} in {folder}")


def load_multimodal_sample(stem, cache_dir, orig_size=None):
    vis_path = os.path.join(cache_dir, "test_visible", f"{stem}.jpg")
    if not os.path.exists(vis_path):
        vis_path = os.path.join(cache_dir, "test_visible", f"{stem}.png")
    
    ir_path = os.path.join(cache_dir, "test_infrared", f"{stem}.jpg")
    if not os.path.exists(ir_path):
        ir_path = os.path.join(cache_dir, "test_infrared", f"{stem}.png")
        
    lbg_path = os.path.join(cache_dir, "test_lbg_ir", f"{stem}.jpg")
    if not os.path.exists(lbg_path):
        lbg_path = os.path.join(cache_dir, "test_lbg_ir", f"{stem}.png")

    rgb_pil = Image.open(vis_path).convert("RGB")
    W, H = rgb_pil.size
    
    # 1. RGB
    rgb_t = T.functional.pil_to_tensor(rgb_pil).float() / 255.0
    
    # 2. Aligned IR
    ir_pil = Image.open(ir_path).convert("L")
    ir_t = T.functional.pil_to_tensor(ir_pil).float() / 255.0
    if ir_t.shape[1:] != (H, W):
        ir_t = T.functional.resize(ir_t, [H, W])
        
    # 3. LBG IR
    lbg_pil = Image.open(lbg_path).convert("L")
    lbg_t = T.functional.pil_to_tensor(lbg_pil).float() / 255.0
    if lbg_t.shape[1:] != (H, W):
        lbg_t = T.functional.resize(lbg_t, [H, W])
        
    # 4. Depth P0
    dep_t = read_depth_raw16(os.path.join(cache_dir, "test_depth_p0"), stem)
    if dep_t.shape[1:] != (H, W):
        dep_t = T.functional.resize(dep_t, [H, W])
        
    # Stack 6 channels
    # Layout: [RGB(3), ir(1), lbg_ir(1), depth(1)]
    tensor_6ch = torch.cat([rgb_t, ir_t, lbg_t, dep_t], dim=0) # [6, H, W]
    
    # Resize to [1152, 2048]
    tensor_resized = T.functional.resize(tensor_6ch, [1152, 2048])
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)
    tensor_norm = (tensor_resized - mean) / std
    
    return tensor_norm.unsqueeze(0), (W, H), vis_path, ir_path


def render_detections(vis_path, ir_path, detections, score_thr=0.45):
    """
    Renders bounding boxes on visible image and produces a side-by-side RGB+IR visualization.
    """
    img_bgr = cv2.imread(vis_path)
    ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    ir_bgr = cv2.applyColorMap(ir_gray, cv2.COLORMAP_INFERNO)
    
    H, W = img_bgr.shape[:2]
    if ir_bgr.shape[:2] != (H, W):
        ir_bgr = cv2.resize(ir_bgr, (W, H))

    vis_canvas = img_bgr.copy()
    ir_canvas = ir_bgr.copy()
    
    valid_dets = [d for d in detections if d['score'] >= score_thr]
    # Sort by score ascending so higher scores are drawn on top
    valid_dets.sort(key=lambda x: x['score'])
    
    for d in valid_dets:
        cid = int(d['label'])
        score = float(d['score'])
        x1, y1, x2, y2 = map(int, d['bbox'])
        x1, x2 = max(0, min(W - 1, x1)), max(0, min(W - 1, x2))
        y1, y2 = max(0, min(H - 1, y1)), max(0, min(H - 1, y2))
        
        color = CLASS_COLORS.get(cid, (0, 255, 0))
        cname = CLASS_NAMES.get(cid, str(cid))
        label_text = f"{cname} {score:.2f}"
        
        # Draw on visible
        cv2.rectangle(vis_canvas, (x1, y1), (x2, y2), color, 2)
        
        # Text badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thick = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thick)
        bg_y1 = max(0, y1 - th - 6)
        bg_y2 = y1
        bg_x1 = x1
        bg_x2 = min(W, x1 + tw + 6)
        
        cv2.rectangle(vis_canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
        cv2.rectangle(vis_canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
        cv2.putText(vis_canvas, label_text, (x1 + 3, y1 - 4), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        
        # Also draw thin box on IR
        cv2.rectangle(ir_canvas, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(ir_canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
        cv2.rectangle(ir_canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
        cv2.putText(ir_canvas, label_text, (x1 + 3, y1 - 4), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        
    # Create Side-by-Side Dual-Modality Image (Width = 2048, Height = 576 each pane scaled)
    pane_w = 1024
    pane_h = int(pane_w * H / W)
    vis_small = cv2.resize(vis_canvas, (pane_w, pane_h), interpolation=cv2.INTER_AREA)
    ir_small = cv2.resize(ir_canvas, (pane_w, pane_h), interpolation=cv2.INTER_AREA)
    
    # Add title bar on top of each pane
    cv2.putText(vis_small, f"Visible (RGB) [Dets: {len(valid_dets)}]", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ir_small, "Aligned Infrared (Colormap)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    dual_pane = np.hstack([vis_small, ir_small])
    return vis_canvas, dual_pane, valid_dets


def main():
    print("=" * 70)
    print(" Sovereign V23 Test Set 1 Safe Inference & Visual Inspection ".center(70, "="))
    print("=" * 70)
    
    # Redline VRAM protection
    torch.cuda.set_per_process_memory_fraction(0.03) # strictly cap at ~2.8 GB
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, VRAM Hard Cap: 3% (~2.88 GB)")
    
    os.chdir("/root/autodl-tmp/AIC/vendor/DEIMv2")
    sys.path.insert(0, ".")
    
    from engine.core import YAMLConfig
    cfg_path = "configs/deimv2/sovereign_v23_zero_distort_2048_fast32.yml"
    weights_path = "outputs/sovereign_v23_zero_distort_2048_fast32/best_stg2.pth"
    cache_dir = "/dev/shm/aic_cache"
    out_dir = "/root/autodl-tmp/AIC/work_dir/v23_test_visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading config: {cfg_path}")
    cfg = YAMLConfig(cfg_path)
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor
    
    print(f"Loading checkpoint: {weights_path}")
    ck = torch.load(weights_path, map_location="cpu")
    w = ck["ema"]["module"] if ("ema" in ck and "module" in ck["ema"]) else ck.get("model", ck)
    model.load_state_dict(w, strict=False)
    epoch = ck.get("last_epoch", "unknown")
    print(f"Successfully loaded checkpoint from Epoch {epoch}!")
    
    all_results = {}
    dual_panes = []
    
    print("\nStarting Test Inference on 8 Key Archetypes...")
    t0 = time.time()
    
    for stem, desc in TARGET_STEMS:
        t_img = time.time()
        tensor_norm, (W, H), vis_path, ir_path = load_multimodal_sample(stem, cache_dir)
        tensor_norm = tensor_norm.to(device)
        orig_sizes = torch.tensor([[W, H]], device=device) # PostProcessor expects [W, H] for [x1, y1, x2, y2]
        
        with torch.no_grad():
            outputs = model(tensor_norm)
            res = postprocessor(outputs, orig_sizes, tensor_norm)[0]
            
        labels = res['labels'].cpu().numpy()
        boxes = res['boxes'].cpu().numpy()
        scores = res['scores'].cpu().numpy()
        
        dets = []
        for l, b, s in zip(labels, boxes, scores):
            dets.append({
                'label': int(l),
                'bbox': [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                'score': float(s),
            })
            
        vis_annotated, dual_pane, valid_dets = render_detections(vis_path, ir_path, dets, score_thr=0.45)
        
        # Save individual images
        cv2.imwrite(os.path.join(out_dir, f"vis_{stem}.jpg"), vis_annotated)
        cv2.imwrite(os.path.join(out_dir, f"dual_{stem}.jpg"), dual_pane)
        dual_panes.append((stem, desc, dual_pane, len(valid_dets)))
        
        # Class distribution in valid dets
        class_counts = {}
        for vd in valid_dets:
            cname = CLASS_NAMES[vd['label']]
            class_counts[cname] = class_counts.get(cname, 0) + 1
        
        dt = (time.time() - t_img) * 1000
        print(f"  [{stem}] ({desc[:18]}...): {len(valid_dets)} dets (tau>=0.45) in {dt:.1f}ms. Counts: {class_counts}")
        all_results[stem] = {
            'desc': desc,
            'total_dets_45': len(valid_dets),
            'counts': class_counts,
            'orig_size': (W, H),
        }
        
    print(f"\nAll 8 images inferred in {time.time() - t0:.2f}s!")
    
    # Create 4x2 or 8-tile Gallery Collage
    # Each dual_pane is (pane_h, 2048) -> stack vertically with titles
    gallery_tiles = []
    for stem, desc, dp, det_cnt in dual_panes:
        # Add a title bar on top of dual_pane
        h_dp, w_dp = dp.shape[:2]
        banner = np.zeros((48, w_dp, 3), dtype=np.uint8)
        cv2.rectangle(banner, (0, 0), (w_dp, 48), (30, 30, 30), -1)
        title_str = f"Image ID: {stem} | {desc} | Detections (conf >= 0.45): {det_cnt}"
        cv2.putText(banner, title_str, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        tile = np.vstack([banner, dp])
        gallery_tiles.append(tile)
        
    # Stack into 2 master gallery images (4 scenes each)
    gallery_part1 = np.vstack(gallery_tiles[:4])
    gallery_part2 = np.vstack(gallery_tiles[4:])
    
    cv2.imwrite(os.path.join(out_dir, "v23_test_gallery_part1.jpg"), gallery_part1, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(out_dir, "v23_test_gallery_part2.jpg"), gallery_part2, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved master gallery images to {out_dir}!")
    
    # Save JSON summary
    import json
    with open(os.path.join(out_dir, "inspection_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("Summary JSON saved.")


if __name__ == "__main__":
    main()
