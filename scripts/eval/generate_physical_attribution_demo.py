"""
Generate Multimodal Physical Attribution Demonstration.
Renders RGB, Thermal IR, Metric Depth P0, and 3D Metric Distance Detections.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt

ROOT = "/root/autodl-tmp/AIC"
DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
sys.path.insert(0, DEIMV2_ROOT)
os.chdir(DEIMV2_ROOT)

from engine.core import YAMLConfig
from depth_coplanar_filter import prune_coplanar_false_positives
from destripe_1d import stripe_non_uniformity_correction_1d

CLASSES = [
    'person', 'boat', 'animal', 'seat', 'sign', 'bicycle',
    'car', 'ball', 'light', 'garbage can', 'uav', 'tricycle'
]

CLASS_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255)
]

def generate_demo(img_name="00000106.jpg", out_path="/root/autodl-tmp/AIC/multimodal_physical_attribution_demo.png"):
    cfg_file = "configs/deimv2/deim_hgnetv2_x_aic_sovereign_v2_rtx6000.yml"
    weights_file = "./outputs/deim_hgnetv2_x_aic_sovereign_v2_rtx6000/best_stg1.pth"
    
    print(f"[*] Loading model from {weights_file}...")
    cfg = YAMLConfig(cfg_file)
    ck = torch.load(weights_file, map_location="cpu")
    sd = ck["ema"]["module"] if "ema" in ck else ck["model"]
    cfg.model.load_state_dict(sd, strict=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        def forward(self, images, orig):
            return self.postprocessor(self.model(images), orig)

    infer = Model().cuda().eval()

    # Paths
    vis_path = f"/dev/shm/aic_cache/visible/{img_name}"
    ir_path = f"/dev/shm/aic_cache/train_aligned_ir/{img_name}"
    lbg_path = f"/dev/shm/aic_cache/train_lbg_ir/{img_name}"
    depth_path = f"/dev/shm/aic_cache/depth_p0/{os.path.splitext(img_name)[0]}.png"

    # 1. Load modalities
    rgb_bgr = cv2.imread(vis_path)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        ir = np.zeros((H, W), dtype=np.uint8)
    ir = stripe_non_uniformity_correction_1d(ir)

    lbg = cv2.imread(lbg_path, cv2.IMREAD_GRAYSCALE)
    if lbg is None:
        lbg = np.zeros((H, W), dtype=np.uint8)

    depth_16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_16 is None:
        depth_16 = np.zeros((H, W), dtype=np.uint16)
    depth_mm = depth_16.astype(np.float32)

    # 2. Build 6-channel tensor
    rgb_f = (rgb.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    ir_f = (ir.astype(np.float32) / 255.0 - 0.5) / 0.5
    lbg_f = (lbg.astype(np.float32) / 255.0 - 0.5) / 0.5
    depth_norm = np.clip(depth_mm / 20000.0, 0.0, 1.0)
    depth_f = (depth_norm - 0.5) / 0.5

    img_6ch = np.dstack([rgb_f, ir_f[:, :, None], lbg_f[:, :, None], depth_f[:, :, None]])
    img_6ch_resized = cv2.resize(img_6ch, (640, 640))
    tensor = torch.from_numpy(img_6ch_resized).permute(2, 0, 1).unsqueeze(0).float().cuda()
    orig = torch.tensor([[W, H]], dtype=torch.float32).cuda()

    # 3. Inference
    with torch.no_grad():
        labels, boxes, scores = infer(tensor, orig)
        labels = labels[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()

    # 4. Filter confident detections
    conf_thr = 0.35
    valid_mask = scores >= conf_thr
    v_boxes = boxes[valid_mask]
    v_scores = scores[valid_mask]
    v_labels = labels[valid_mask]

    # 5. Apply Ground Coplanar Filter
    keep_mask = prune_coplanar_false_positives(v_boxes, v_scores, v_labels, depth_mm)

    # 6. Extract physical depth Z for each box
    box_depths = []
    for b in v_boxes:
        x1, y1, x2, y2 = np.clip(b.astype(int), 0, [W-1, H-1, W-1, H-1])
        patch = depth_mm[y1:y2, x1:x2]
        valid_p = patch[patch > 500] # ignore <0.5m
        if len(valid_p) > 0:
            med_z = np.median(valid_p) / 1000.0 # meters
        else:
            med_z = 0.0
        box_depths.append(med_z)

    # 7. Render 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    plt.subplots_adjust(wspace=0.04, hspace=0.08)

    # Panel 1: RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("[Visible RGB] Modality Channel 0-2 (Color & Surface)", fontsize=13, pad=6)
    axes[0, 0].axis("off")

    # Panel 2: Aligned Thermal IR
    axes[0, 1].imshow(ir, cmap="inferno")
    axes[0, 1].set_title("[Aligned Thermal IR] Modality Channel 3 (Thermal Radiation)", fontsize=13, pad=6)
    axes[0, 1].axis("off")

    # Panel 3: Metric Depth P0 (Turbo colormap)
    depth_vis = np.clip(depth_mm, 500, 30000)
    depth_vis = ((depth_vis - 500) / 29500.0 * 255.0).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(depth_colored_rgb)
    axes[1, 0].set_title("[Unified P0 Metric Depth] Modality Channel 5 (Geometric 3D Relief)", fontsize=13, pad=6)
    axes[1, 0].axis("off")

    # Panel 4: Multimodal Detections with 3D Metric Z
    det_img = rgb.copy()
    for idx in range(len(v_boxes)):
        b = v_boxes[idx].astype(int)
        lbl = int(v_labels[idx])
        sc = v_scores[idx]
        is_kept = keep_mask[idx]
        z_m = box_depths[idx]

        cls_name = CLASSES[lbl] if lbl < len(CLASSES) else f"cls_{lbl}"
        color = CLASS_COLORS[lbl % len(CLASS_COLORS)] if is_kept else (128, 128, 128)

        # Draw bbox
        cv2.rectangle(det_img, (b[0], b[1]), (b[2], b[3]), color, 2)

        # Caption: Class + Score + Metric Depth Z
        if z_m > 0:
            caption = f"{cls_name} {sc:.2f} | Z={z_m:.1f}m"
        else:
            caption = f"{cls_name} {sc:.2f}"

        if not is_kept:
            caption += " [PRUNED: Ground Coplanar]"

        # Text background
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(det_img, (b[0], max(0, b[1] - th - 6)), (b[0] + tw + 4, b[1]), color, -1)
        cv2.putText(det_img, caption, (b[0] + 2, b[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    axes[1, 1].imshow(det_img)
    axes[1, 1].set_title(f"[Sovereign V2 3D Metric Detections] Active Detections with Z Distance (m)", fontsize=13, pad=6)
    axes[1, 1].axis("off")

    plt.suptitle(f"Sovereign Multimodal V2: Physical Metric Perception Proof ({img_name})", fontsize=15, y=0.96)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved demonstration to {out_path}")

if __name__ == "__main__":
    generate_demo()
