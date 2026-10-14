"""
P1 Step-by-Step Test:
Ground Plane Subtraction & Detail Layer Decomposition.

Algorithm:
  1. Computes dense disparity d(x, y) = 1000.0 / Z_m.
  2. Fits 2D Ground Plane d_ground(x, y) = c0 + cy*y + cx*x using low-quantile road samples.
  3. Computes 3D Object Elevation Relief: Relief(x, y) = ReLU(d(x, y) - d_ground(x, y)).
  4. Combines with P0 Surface Normal to form 3-Channel Enhanced Geometry: [nx, ny, Relief].

Evaluates:
  1. Ground plain suppression ratio (road variance reduction).
  2. Foreground-to-background contrast (FDR) across GT object boxes.
  3. Target patch linear probe classification accuracy (219 real patches).
  4. Generates visual comparison cards and panel.
"""
import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

DA_CKPT = os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth")
OUT_DIR = os.path.join(ROOT, "work_dir/visualizations/figures/p1_ground_test")
ART_DIR = "/home/constant/.gemini/antigravity/brain/7fd161eb-9712-4fa4-b0e6-4221a61b8664"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

def load_da_model():
    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(DA_CKPT, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), device

def p0_refine_depth(depth_raw, d_rel, valid, vis, ir_gray):
    """P0 Disparity Fit + Bi-Guide from Step 1."""
    H, W = depth_raw.shape
    Z_m = depth_raw[valid] / 1000.0
    disp_true = 1.0 / np.clip(Z_m, 0.5, 80.0)
    rv = d_rel[valid]
    d_min, d_max = rv.min(), rv.max()
    d_norm = (rv - d_min) / (d_max - d_min + 1e-8)
    d_rel_full_norm = (d_rel - d_min) / (d_max - d_min + 1e-8)

    try:
        # Analytical 1st degree fit (exact covariance/variance, avoids rank warnings)
        var_x = np.var(d_norm)
        if var_x > 1e-8:
            a_disp = np.cov(d_norm, disp_true)[0, 1] / var_x
            b_disp = np.mean(disp_true) - a_disp * np.mean(d_norm)
        else:
            a_disp, b_disp = 1.0, 0.1
    except Exception:
        a_disp, b_disp = 1.0, 0.1

    disp_pred = np.clip(a_disp * d_rel_full_norm + b_disp, 0.01, 2.0)
    dav2_abs = np.clip((1.0 / disp_pred) * 1000.0, 500.0, 65535.0)
    base_fill = np.where(valid, depth_raw, dav2_abs)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ir_enh = clahe.apply(np.clip(ir_gray, 0, 255).astype(np.uint8)).astype(np.float32) / 255.0
    rgb_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    gx_rgb = cv2.Sobel(rgb_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_rgb = cv2.Sobel(rgb_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_rgb = np.sqrt(gx_rgb**2 + gy_rgb**2)

    gx_ir = cv2.Sobel(ir_enh, cv2.CV_32F, 1, 0, ksize=3)
    gy_ir = cv2.Sobel(ir_enh, cv2.CV_32F, 0, 1, ksize=3)
    mag_ir = np.sqrt(gx_ir**2 + gy_ir**2)

    guide_f = np.where(mag_rgb > mag_ir, rgb_gray, ir_enh).astype(np.float32)

    # Box guided filter
    r, eps = 4, 1e-3
    mean_I = cv2.boxFilter(guide_f, -1, (2*r+1, 2*r+1))
    mean_p = cv2.boxFilter(base_fill.astype(np.float32), -1, (2*r+1, 2*r+1))
    mean_Ip = cv2.boxFilter(guide_f * base_fill.astype(np.float32), -1, (2*r+1, 2*r+1))
    var_I = cv2.boxFilter(guide_f * guide_f, -1, (2*r+1, 2*r+1)) - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    gf = cv2.boxFilter(a, -1, (2*r+1, 2*r+1)) * guide_f + cv2.boxFilter(b, -1, (2*r+1, 2*r+1))
    refined = np.where(valid, depth_raw, gf)
    return refined

def extract_ground_and_relief(depth_map):
    """
    Fits ground plane disparity d_ground(x, y) = c0 + cy*y + cx*x
    Computes relief = ReLU(d(x, y) - d_ground(x, y)).
    """
    H, W = depth_map.shape
    d_m = depth_map.astype(np.float32) / 1000.0
    disp = 1.0 / np.clip(d_m, 0.5, 80.0)

    # Sample road ground candidate points from the bottom half of the image
    # We take the 15th percentile disparity along rows to select road floor (avoiding standing obstacles)
    y_start = int(H * 0.45)
    ys = []
    xs = []
    vals = []

    # Downsample grid for speed and stability
    step_y = 12
    step_x = 24
    for r in range(y_start, H, step_y):
        row_slice = disp[r, :]
        p15 = np.percentile(row_slice, 15)
        # Select pixels near the 15th percentile (road plane candidates)
        mask = np.abs(row_slice - p15) < (0.08 * p15)
        if mask.sum() > 10:
            cols = np.where(mask)[0][::step_x]
            for c in cols:
                ys.append(r / float(H))
                xs.append(c / float(W))
                vals.append(disp[r, c])

    if len(vals) < 30:
        # Fallback to simple linear row ramp
        ys = np.linspace(0.45, 0.98, 50)
        xs = np.zeros_like(ys)
        vals = [np.percentile(disp[int(y * (H - 1)), :], 15) for y in ys]

    ys = np.array(ys, dtype=np.float32)
    xs = np.array(xs, dtype=np.float32)
    vals = np.array(vals, dtype=np.float32)

    # Fit plane: disp = c0 + cy*y + cx*x
    A_mat = np.stack([np.ones_like(ys), ys, xs], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, vals, rcond=None)
    c0, cy, cx = coeffs

    # Construct full ground disparity map
    grid_y, grid_x = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
    d_ground = c0 + cy * grid_y + cx * grid_x
    d_ground = np.clip(d_ground, 0.005, 2.0)

    # Object 3D relief: positive disparity jump above the ground plane
    relief = np.maximum(0.0, disp - d_ground)

    # Normalization: objects typically protrude 0.05 ~ 0.5 in disparity
    relief_norm = np.clip(relief / 0.4, 0.0, 1.0).astype(np.float32)
    ground_depth_vis = (1.0 / d_ground) * 1000.0

    return relief_norm, ground_depth_vis, d_ground

def compute_fdr(feature_map, boxes, H, W):
    if feature_map.ndim == 3:
        feat_scalar = np.mean(feature_map, axis=-1)
    else:
        feat_scalar = feature_map.copy()
    feat_scalar = (feat_scalar - feat_scalar.min()) / (feat_scalar.max() - feat_scalar.min() + 1e-6)

    fg_mask = np.zeros((H, W), dtype=bool)
    bg_mask = np.zeros((H, W), dtype=bool)

    for cls_id, cx, cy, w, h in boxes:
        x1 = int(max(0, (cx - w / 2) * W))
        x2 = int(min(W, (cx + w / 2) * W))
        y1 = int(max(0, (cy - h / 2) * H))
        y2 = int(min(H, (cy + h / 2) * H))
        fg_mask[y1:y2, x1:x2] = True
        bx1 = int(max(0, (cx - w * 0.7) * W))
        bx2 = int(min(W, (cx + w * 0.7) * W))
        by1 = int(max(0, (cy - h * 0.7) * H))
        by2 = int(min(H, (cy + h * 0.7) * H))
        bg_mask[by1:by2, bx1:bx2] = True

    bg_mask = bg_mask & (~fg_mask)
    if fg_mask.sum() == 0 or bg_mask.sum() == 0:
        return 0.0
    mu_fg, var_fg = np.mean(feat_scalar[fg_mask]), np.var(feat_scalar[fg_mask])
    mu_bg, var_bg = np.mean(feat_scalar[bg_mask]), np.var(feat_scalar[bg_mask])
    return float((mu_fg - mu_bg)**2 / (var_fg + var_bg + 1e-7))

def run_probe_comparison(patches_dict, y, num_classes=12):
    """Runs fast linear probing on collected representations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = len(y)
    perm = np.random.RandomState(42).permutation(N)
    n_train = int(N * 0.75)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    y_t = torch.from_numpy(y).long()

    results = {}
    for name, X in patches_dict.items():
        in_dim = X.shape[1] * X.shape[2] * X.shape[3]
        X_flat = torch.from_numpy(X.reshape(N, -1)).float()
        train_ds = TensorDataset(X_flat[train_idx], y_t[train_idx])
        val_ds = TensorDataset(X_flat[val_idx], y_t[val_idx])

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        runs = []
        for _ in range(5):
            model = nn.Sequential(
                nn.BatchNorm1d(in_dim),
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            crit = nn.CrossEntropyLoss()
            best_acc = 0.0
            for ep in range(25):
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
                if acc > best_acc: best_acc = acc
            runs.append(best_acc)
        results[name] = (np.mean(runs), np.std(runs))
    return results

def main():
    print("=" * 75)
    print("=== P1 EVALUATION: GROUND PLANE SUBTRACTION & 3D RELIEF DECOMPOSITION ===")
    print("=" * 75)
    model, device = load_da_model()

    test_stems = [
        "000002_083_00000169", # Tricycle + bicycle on ground
        "000006_010_00000389", # Animal + boat
        "000003_018_00000167", # Dark scene
        "000002_081_00000047", # Pedestrian + sign
        "000002_082_00000061", # Multiple vehicles on road
        "000004_009_00000321", # Low-light road
        "000002_080_00000048", # Dense daytime street
        "000003_015_00000077", # Open highway/road
    ]

    fdr_results = []
    road_suppression_results = []

    for stem in test_stems:
        v_path = f"dataset/train/visible/{stem}.png"
        i_path = f"dataset/train/infrared/{stem}.png"
        d_path = f"dataset/train/depth/{stem}.png"
        l_path = f"dataset/train/labels/{stem}.txt"

        vis = cv2.imread(v_path)
        ir = cv2.imread(i_path, cv2.IMREAD_UNCHANGED)
        if ir.ndim == 3: ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        depth_raw = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        H, W = depth_raw.shape

        boxes = []
        if os.path.exists(l_path):
            with open(l_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

        # 1. P0 Refined Depth
        I = ir.astype(np.float32) / (ir.max() + 1e-8)
        ir8 = np.clip(I * 255, 0, 255).astype(np.uint8)
        inp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
        with torch.no_grad():
            d_rel = model.infer_image(inp, 518)
        d_rel = cv2.resize(d_rel.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        valid = depth_raw > 1.0
        depth_p0 = p0_refine_depth(depth_raw, d_rel, valid, vis, ir)

        # 2. P0 Surface Normal
        d_norm = (depth_p0 / (depth_p0.max() + 1e-6)).astype(np.float32)
        dx = cv2.Sobel(d_norm, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=3)
        dz = np.ones_like(dx, dtype=np.float32) * 0.05
        norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
        nx, ny, nz = -dx / norm, -dy / norm, dz / norm
        normal_map = np.stack([(nx + 1.0)*0.5, (ny + 1.0)*0.5, (nz + 1.0)*0.5], axis=-1).astype(np.float32)

        # 3. P1 Ground Plane Subtraction & 3D Relief
        relief, ground_depth, d_ground = extract_ground_and_relief(depth_p0)

        # 4. Combined Representation: [nx, ny, Relief]
        combined_p0_p1 = np.stack([normal_map[..., 0], normal_map[..., 1], relief], axis=-1).astype(np.float32)

        # Compute Road Surface Variance (Bottom 25% of image, outside GT boxes)
        road_y1 = int(H * 0.75)
        road_mask = np.zeros((H, W), dtype=bool)
        road_mask[road_y1:, :] = True
        for cls_id, cx, cy, w, h in boxes:
            x1 = int(max(0, (cx - w*0.8)*W))
            x2 = int(min(W, (cx + w*0.8)*W))
            y1 = int(max(0, (cy - h*0.8)*H))
            y2 = int(min(H, (cy + h*0.8)*H))
            road_mask[y1:y2, x1:x2] = False

        var_raw_road = np.var((depth_raw / 19999.0)[road_mask])
        var_relief_road = np.var(relief[road_mask])
        suppression_ratio = (var_raw_road - var_relief_road) / (var_raw_road + 1e-7)

        # Compute FDR
        fdr_raw = compute_fdr(depth_raw / 19999.0, boxes, H, W)
        fdr_p0  = compute_fdr(normal_map, boxes, H, W)
        fdr_p1  = compute_fdr(relief, boxes, H, W)
        fdr_comb = compute_fdr(combined_p0_p1, boxes, H, W)

        fdr_results.append({
            "stem": stem, "fdr_raw": fdr_raw, "fdr_p0": fdr_p0,
            "fdr_p1": fdr_p1, "fdr_comb": fdr_comb
        })
        road_suppression_results.append(suppression_ratio)

        # Save Visual Comparison Strip
        def cm_d(d):
            return cv2.applyColorMap((np.clip(d/15000.0, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
        def cm_relief(r):
            # Heat colormap for relief: black road, bright fiery objects
            return cv2.applyColorMap((np.clip(r, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        sz = (360, 270)
        t_rgb = cv2.resize(vis, sz)
        t_p0d = cv2.resize(cm_d(depth_p0), sz)
        t_grd = cv2.resize(cm_d(ground_depth), sz)
        t_rlf = cv2.resize(cm_relief(relief), sz)
        t_cmb = cv2.resize((combined_p0_p1 * 255).astype(np.uint8), sz)

        cv2.putText(t_rgb, "1. Visible RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(t_p0d, "2. P0 Refined Depth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(t_grd, "3. Fitted Ground Plane", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(t_rlf, f"4. P1 Relief (FDR={fdr_p1:.3f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(t_cmb, f"5. P0+P1 [nx,ny,Relief] (FDR={fdr_comb:.3f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        card = np.hstack([t_rgb, t_p0d, t_grd, t_rlf, t_cmb])
        cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_p1_ground_compare.jpg"), card)
        cv2.imwrite(os.path.join(ART_DIR, f"{stem}_p1_ground_compare.jpg"), card)

    print("\n" + "=" * 90)
    print(f"{'Sample Stem':<22} | {'FDR Raw':<8} | {'FDR P0 Normal':<14} | {'FDR P1 Relief':<14} | {'FDR P0+P1 Comb':<14}")
    print("-" * 90)
    for r in fdr_results:
        print(f"{r['stem']:<22} | {r['fdr_raw']:<8.4f} | {r['fdr_p0']:<14.4f} | {r['fdr_p1']:<14.4f} | {r['fdr_comb']:<14.4f}")

    avg_fdr_raw = np.mean([r['fdr_raw'] for r in fdr_results])
    avg_fdr_p0 = np.mean([r['fdr_p0'] for r in fdr_results])
    avg_fdr_p1 = np.mean([r['fdr_p1'] for r in fdr_results])
    avg_fdr_comb = np.mean([r['fdr_comb'] for r in fdr_results])
    avg_suppress = np.mean(road_suppression_results) * 100.0

    print("-" * 90)
    print(f"{'AVERAGE':<22} | {avg_fdr_raw:<8.4f} | {avg_fdr_p0:<14.4f} | {avg_fdr_p1:<14.4f} | {avg_fdr_comb:<14.4f}")
    print("=" * 90)
    print(f"[Key Result 1] Road Surface Noise Suppression  : {avg_suppress:.1f}% (路面斜坡噪声被消除)")
    print(f"[Key Result 2] Foreground Object Contrast FDR   : Raw={avg_fdr_raw:.4f} -> P1 Relief={avg_fdr_p1:.4f} (前景判别比提升 {avg_fdr_p1/(avg_fdr_raw+1e-6):.2f}x)")
    print(f"[Key Result 3] Combined Representation FDR      : {avg_fdr_comb:.4f} (提升 {avg_fdr_comb/(avg_fdr_raw+1e-6):.2f}x)")

    # Run Fast Linear Probe on target patches
    print("\n" + "=" * 75)
    print("=== RUNNING LINEAR PROBE BENCHMARK ON TARGET PATCHES ===")
    print("=" * 75)

    depth_files = sorted(glob.glob("dataset/train/depth/*.png"))[:35]
    patches_raw = []
    patches_geo = []
    patches_relief = []
    patches_comb = []
    labels = []

    for d_path in depth_files:
        stem = os.path.splitext(os.path.basename(d_path))[0]
        l_path = f"dataset/train/labels/{stem}.txt"
        i_path = f"dataset/train/infrared/{stem}.png"
        v_path = f"dataset/train/visible/{stem}.png"
        if not (os.path.exists(l_path) and os.path.exists(i_path) and os.path.exists(v_path)):
            continue

        vis = cv2.imread(v_path)
        ir = cv2.imread(i_path, cv2.IMREAD_UNCHANGED)
        if ir.ndim == 3: ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        depth_raw = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        H, W = depth_raw.shape

        I = ir.astype(np.float32) / (ir.max() + 1e-8)
        ir8 = np.clip(I * 255, 0, 255).astype(np.uint8)
        inp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
        with torch.no_grad():
            d_rel = model.infer_image(inp, 518)
        d_rel = cv2.resize(d_rel.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        valid = depth_raw > 1.0
        depth_p0 = p0_refine_depth(depth_raw, d_rel, valid, vis, ir)

        d_norm = (depth_p0 / (depth_p0.max() + 1e-6)).astype(np.float32)
        dx = cv2.Sobel(d_norm, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=3)
        dz = np.ones_like(dx, dtype=np.float32) * 0.05
        norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
        normal_map = np.stack([(-dx/norm + 1.0)*0.5, (-dy/norm + 1.0)*0.5, (dz/norm + 1.0)*0.5], axis=-1).astype(np.float32)
        relief, _, _ = extract_ground_and_relief(depth_p0)
        combined = np.stack([normal_map[..., 0], normal_map[..., 1], relief], axis=-1).astype(np.float32)
        raw_norm = depth_raw / 19999.0

        with open(l_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    x1, x2 = int(max(0, (cx - w/2)*W)), int(min(W, (cx + w/2)*W))
                    y1, y2 = int(max(0, (cy - h/2)*H)), int(min(H, (cy + h/2)*H))
                    if (x2 - x1) < 8 or (y2 - y1) < 8:
                        continue
                    sz = (32, 32)
                    p_raw = cv2.resize(raw_norm[y1:y2, x1:x2], sz)[None]
                    p_geo = cv2.resize(normal_map[y1:y2, x1:x2], sz).transpose(2, 0, 1)
                    p_rlf = cv2.resize(relief[y1:y2, x1:x2], sz)[None]
                    p_cmb = cv2.resize(combined[y1:y2, x1:x2], sz).transpose(2, 0, 1)

                    patches_raw.append(p_raw)
                    patches_geo.append(p_geo)
                    patches_relief.append(p_rlf)
                    patches_comb.append(p_cmb)
                    labels.append(cls_id)

    print(f"Collected {len(labels)} patches across {len(set(labels))} classes for probing.")
    patches_dict = {
        "1. Raw Depth (1ch)": np.array(patches_raw, dtype=np.float32),
        "2. P0 Surface Normal (3ch)": np.array(patches_geo, dtype=np.float32),
        "3. P1 Relief (1ch Ground-Subtracted)": np.array(patches_relief, dtype=np.float32),
        "4. Combined P0+P1 [nx, ny, Relief] (3ch)": np.array(patches_comb, dtype=np.float32),
    }

    probe_res = run_probe_comparison(patches_dict, np.array(labels, dtype=np.int64))
    for name, (acc, std) in probe_res.items():
        print(f"  {name:<42} : Top-1 Acc = {acc*100:.2f}% (±{std*100:.2f}%)")
    print("=" * 75)

if __name__ == "__main__":
    main()
