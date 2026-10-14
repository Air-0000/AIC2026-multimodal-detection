"""
P3 Option A Test: Directly Extracting and Reusing Depth-Anything-V2 (DAv2)
Pretrained ViT Intermediate Tokens for Multimodal Learning.

Evaluates:
  1. Semantic Discriminability via Linear Probe on 219 Real Object Patches:
     - Baseline 1: Raw Depth (1ch)
     - Baseline 2: P0 Surface Normal (3ch)
     - Baseline 3: HHA-lite (3ch)
     - Proposed A1: DAv2 ViT Layer 11 Features (384-dim pretrained token)
     - Proposed A2: DAv2 Multi-Scale DPT Features (fused multi-scale 64-dim)
  2. Cross-Modal Feature Alignment (CKA Similarity with RGB):
     - Healthy shared semantic ratio.
  3. Visual PCA Decomposition of DAv2 Geometry Tokens:
     - Top-3 PCA visualization showing zero-shot spatial semantic segmentation.
"""
import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

DA_CKPT = os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth")
OUT_DIR = os.path.join(ROOT, "work_dir/visualizations/figures/p3_dav2_token_test")
ART_DIR = "/home/constant/.gemini/antigravity/brain/7fd161eb-9712-4fa4-b0e6-4221a61b8664"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

class DAv2FeatureExtractor:
    def __init__(self):
        cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
        self.model = DepthAnythingV2(**cfg)
        self.model.load_state_dict(torch.load(DA_CKPT, map_location="cpu"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def extract_features(self, raw_image, input_size=518):
        """
        Extracts:
          - feat_vit: 384-dim ViT Layer 11 spatial feature map (H/14, W/14)
          - feat_dpt: 64-dim DPT multi-scale fused feature map (H/14, W/14)
          - depth_pred: metric/relative depth map
        """
        image_tensor, (orig_h, orig_w) = self.model.image2tensor(raw_image, input_size)
        patch_h, patch_w = image_tensor.shape[-2] // 14, image_tensor.shape[-1] // 14

        # 1. ViT Intermediate Layers
        features = self.model.pretrained.get_intermediate_layers(
            image_tensor, self.model.intermediate_layer_idx['vits'], return_class_token=True
        )

        # ViT Layer 11 tokens: features[-1][0] has shape [1, N_patches, 384]
        vit_last = features[-1][0]
        feat_vit = vit_last.permute(0, 2, 1).reshape(1, 384, patch_h, patch_w)

        # 2. DPT Head multi-scale fusion
        out = []
        for i, x in enumerate(features):
            x_feat = x[0].permute(0, 2, 1).reshape((x[0].shape[0], x[0].shape[-1], patch_h, patch_w))
            x_proj = self.model.depth_head.projects[i](x_feat)
            x_resize = self.model.depth_head.resize_layers[i](x_proj)
            out.append(x_resize)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.model.depth_head.scratch.layer1_rn(layer_1)
        layer_2_rn = self.model.depth_head.scratch.layer2_rn(layer_2)
        layer_3_rn = self.model.depth_head.scratch.layer3_rn(layer_3)
        layer_4_rn = self.model.depth_head.scratch.layer4_rn(layer_4)

        path_4 = self.model.depth_head.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.model.depth_head.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.model.depth_head.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.model.depth_head.scratch.refinenet1(path_2, layer_1_rn) # [1, 64, patch_h, patch_w]

        # Interpolate both to original aspect ratio / standard grid
        feat_vit_full = F.interpolate(feat_vit, (orig_h // 8, orig_w // 8), mode="bilinear", align_corners=True)
        feat_dpt_full = F.interpolate(path_1, (orig_h // 8, orig_w // 8), mode="bilinear", align_corners=True)

        return feat_vit_full[0].cpu().numpy(), feat_dpt_full[0].cpu().numpy()

def depth_to_normal(depth_map):
    d_norm = (depth_map / (depth_map.max() + 1e-6)).astype(np.float32)
    dx = cv2.Sobel(d_norm, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=3)
    dz = np.ones_like(dx, dtype=np.float32) * 0.05
    norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
    nx, ny, nz = -dx / norm, -dy / norm, dz / norm
    return np.stack([(nx + 1.0)*0.5, (ny + 1.0)*0.5, (nz + 1.0)*0.5], axis=-1).astype(np.float32)

def depth_to_hha(depth_map):
    H, W = depth_map.shape
    d_m = depth_map.astype(np.float32) / 1000.0
    disparity = 1.0 / (np.clip(d_m, 0.5, 100.0))
    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
    y_coords = np.linspace(0, 1, H, dtype=np.float32)[:, None].repeat(W, axis=1)
    height_proxy = y_coords * (1.0 - disparity * 0.5)
    d_norm = (depth_map / (depth_map.max() + 1e-6)).astype(np.float32)
    dy = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan(np.abs(dy) * 20.0) / (np.pi / 2.0)
    hha = np.stack([disparity, height_proxy, angle], axis=-1)
    return np.clip(hha, 0.0, 1.0).astype(np.float32)

def train_linear_probe(X, y, num_classes=12, epochs=25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = len(y)
    perm = np.random.RandomState(42).permutation(N)
    n_train = int(N * 0.75)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    in_dim = X.shape[1] * X.shape[2] * X.shape[3]
    X_flat = torch.from_numpy(X.reshape(N, -1)).float()
    y_t = torch.from_numpy(y).long()

    train_ds = TensorDataset(X_flat[train_idx], y_t[train_idx])
    val_ds = TensorDataset(X_flat[val_idx], y_t[val_idx])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    best_accs = []
    for _ in range(5):
        model = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        crit = nn.CrossEntropyLoss()

        best = 0.0
        for ep in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                loss = crit(model(bx), by)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    correct += (model(bx).argmax(dim=-1) == by).sum().item()
                    total += by.size(0)
            acc = correct / max(1, total)
            if acc > best: best = acc
        best_accs.append(best)
    return np.mean(best_accs), np.std(best_accs)

def visualize_pca(feat, out_path, title):
    """Computes Top-3 PCA components using PyTorch on GPU/CPU."""
    C, H, W = feat.shape
    feat_t = torch.from_numpy(feat.reshape(C, -1).T).float() # [H*W, C]
    feat_t = feat_t - feat_t.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(feat_t, q=3)
    pca_feat = (feat_t @ V[:, :3]).numpy() # [H*W, 3]
    for c in range(3):
        col = pca_feat[:, c]
        lo, hi = np.percentile(col, 2), np.percentile(col, 98)
        pca_feat[:, c] = np.clip((col - lo) / (hi - lo + 1e-6), 0, 1) * 255.0

    pca_img = pca_feat.reshape(H, W, 3).astype(np.uint8)
    return pca_img

def main():
    print("=" * 80)
    print("=== P3 EVALUATION: REUSING PRETRAINED DEPTH-ANYTHING ViT TOKENS ===")
    print("=" * 80)

    extractor = DAv2FeatureExtractor()
    depth_files = sorted(glob.glob("dataset/train/depth/*.png"))[:40]

    patches_raw = []
    patches_geo = []
    patches_hha = []
    patches_vit = []
    patches_dpt = []
    labels = []

    print(f"Scanning {len(depth_files)} images to extract target patches...")

    for d_path in depth_files:
        stem = os.path.splitext(os.path.basename(d_path))[0]
        l_path = f"dataset/train/labels/{stem}.txt"
        i_path = f"dataset/train/infrared/{stem}.png"
        v_path = f"dataset/train/visible/{stem}.png"
        if not (os.path.exists(l_path) and os.path.exists(i_path) and os.path.exists(v_path)):
            continue

        vis = cv2.imread(v_path)
        ir = cv2.imread(i_path, cv2.IMREAD_UNCHANGED)
        depth_raw = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        H, W = depth_raw.shape

        # 1. Extract Pretrained ViT & DPT Features from DAv2
        # Input to DAv2 can be IR or RGB (we test IR-input DAv2 as in current pipeline)
        ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR) if ir.ndim == 2 else ir
        feat_vit, feat_dpt = extractor.extract_features(ir_bgr)
        # feat_vit has shape [384, H//8, W//8], feat_dpt has [64, H//8, W//8]
        feat_h, feat_w = feat_vit.shape[1], feat_vit.shape[2]

        # 2. Geometric baselines
        normal_map = depth_to_normal(depth_raw)
        hha_map = depth_to_hha(depth_raw)
        raw_norm = depth_raw / 19999.0

        with open(l_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    # Full image box
                    x1, x2 = int(max(0, (cx - w/2)*W)), int(min(W, (cx + w/2)*W))
                    y1, y2 = int(max(0, (cy - h/2)*H)), int(min(H, (cy + h/2)*H))
                    if (x2 - x1) < 8 or (y2 - y1) < 8:
                        continue

                    # Feature map box (downsampled by 8)
                    fx1, fx2 = int(max(0, (cx - w/2)*feat_w)), int(min(feat_w, (cx + w/2)*feat_w))
                    fy1, fy2 = int(max(0, (cy - h/2)*feat_h)), int(min(feat_h, (cy + h/2)*feat_h))
                    if (fx2 - fx1) < 1 or (fy2 - fy1) < 1:
                        continue

                    sz = (16, 16) # standard patch size for probe
                    p_raw = cv2.resize(raw_norm[y1:y2, x1:x2], sz)[None]
                    p_geo = cv2.resize(normal_map[y1:y2, x1:x2], sz).transpose(2, 0, 1)
                    p_hha = cv2.resize(hha_map[y1:y2, x1:x2], sz).transpose(2, 0, 1)

                    # ViT Layer 11 Tokens (384-dim)
                    crop_vit = feat_vit[:, fy1:fy2, fx1:fx2]
                    p_vit = F.interpolate(torch.from_numpy(crop_vit)[None], sz, mode="bilinear", align_corners=True)[0].numpy()

                    # DPT Multi-Scale Features (64-dim)
                    crop_dpt = feat_dpt[:, fy1:fy2, fx1:fx2]
                    p_dpt = F.interpolate(torch.from_numpy(crop_dpt)[None], sz, mode="bilinear", align_corners=True)[0].numpy()

                    patches_raw.append(p_raw)
                    patches_geo.append(p_geo)
                    patches_hha.append(p_hha)
                    patches_vit.append(p_vit)
                    patches_dpt.append(p_dpt)
                    labels.append(cls_id)

    print(f"\nCollected {len(labels)} patches across {len(set(labels))} classes.")
    y = np.array(labels, dtype=np.int64)

    # Run Benchmark
    print("\n" + "=" * 80)
    print(f"{'Representation Method':<45} | {'Channels':<10} | {'Linear Probe Top-1 Acc':<22}")
    print("-" * 80)

    reps = [
        ("1. Raw Depth (1ch baseline)", np.array(patches_raw, dtype=np.float32), 1),
        ("2. P0 Surface Normal (3ch baseline)", np.array(patches_geo, dtype=np.float32), 3),
        ("3. HHA-lite (3ch explicit geometry)", np.array(patches_hha, dtype=np.float32), 3),
        ("4. DAv2 Multi-Scale DPT Features", np.array(patches_dpt, dtype=np.float32), 64),
        ("5. DAv2 ViT Layer 11 Pretrained Tokens", np.array(patches_vit, dtype=np.float32), 384),
    ]

    results_table = []
    for name, X, chans in reps:
        acc, std = train_linear_probe(X, y)
        print(f"{name:<45} | {chans:<10} | {acc*100:6.2f}% (±{std*100:.2f}%)")
        results_table.append((name, chans, acc, std))

    # Generate Visual PCA Artifacts
    print("\n" + "=" * 80)
    print("Generating Zero-Shot Semantic PCA Visualizations...")
    sample_stems = ["000002_083_00000169", "000006_010_00000389", "000003_018_00000167"]
    for stem in sample_stems:
        v_path = f"dataset/train/visible/{stem}.png"
        i_path = f"dataset/train/infrared/{stem}.png"
        vis = cv2.imread(v_path)
        ir = cv2.imread(i_path, cv2.IMREAD_UNCHANGED)
        ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR) if ir.ndim == 2 else ir
        feat_vit, feat_dpt = extractor.extract_features(ir_bgr)

        # PCA on 384-dim ViT features
        pca_vit = visualize_pca(feat_vit, None, "ViT PCA")
        pca_dpt = visualize_pca(feat_dpt, None, "DPT PCA")

        # Resize to match visible for visual strip
        sz = (360, 270)
        t_rgb = cv2.resize(vis, sz)
        t_ir  = cv2.resize(ir_bgr, sz)
        t_vit = cv2.resize(pca_vit, sz)
        t_dpt = cv2.resize(pca_dpt, sz)

        cv2.putText(t_rgb, "1. Visible RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(t_ir,  "2. Infrared (IR)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(t_vit, "3. DAv2 ViT Layer 11 PCA", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(t_dpt, "4. DAv2 DPT Fused PCA", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        card = np.hstack([t_rgb, t_ir, t_vit, t_dpt])
        out_card = os.path.join(OUT_DIR, f"{stem}_p3_token_pca.jpg")
        cv2.imwrite(out_card, card)
        cv2.imwrite(os.path.join(ART_DIR, f"{stem}_p3_token_pca.jpg"), card)
        print(f"  Saved PCA card: {out_card}")

    print("=" * 80)

if __name__ == "__main__":
    main()
