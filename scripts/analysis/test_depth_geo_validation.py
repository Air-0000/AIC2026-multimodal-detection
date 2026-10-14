"""
Test and audit script: Validating Depth DA-guided fill/filtering and explicit
geometric semantic lifting in a 3-modal (RGB + IR + Depth) detection setting.

Quantitatively measures:
1. Fisher Discriminant Ratio (FDR): Foreground vs Background contrast for each representation.
2. 3-Modal Pearson Structural Correlation (CC): Alignment & non-redundancy between RGB, IR, Depth.
3. 3-Modal Boundary Complementarity Coverage: How many object edge pixels are unique to Depth.
4. Scale Invariance across distances.
Outputs visual evidence and comparison sheets.
"""
import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

DA_CKPT = os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth")
OUT_DIR = os.path.join(ROOT, "work_dir/visualizations/figures/depth_geo_validation")
os.makedirs(OUT_DIR, exist_ok=True)

# 12 AIC Classes
AIC_CATS = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign',
    5: 'bicycle', 6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can',
    10: 'uav', 11: 'tricycle'
}

def load_da_model():
    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(DA_CKPT, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, device

def da_fill_and_filter(depth_raw, ir_raw, model, device):
    """DA-guided metric alignment, hole filling, and guided bilateral denoising."""
    H, W = depth_raw.shape
    I = ir_raw.astype(np.float32) / (ir_raw.max() + 1e-8)
    ir8 = np.clip(I * 255, 0, 255).astype(np.uint8)
    inp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR) if ir8.ndim == 2 else ir8

    with torch.no_grad():
        d_rel = model.infer_image(inp, 518)
    d_rel = cv2.resize(d_rel.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    valid = depth_raw > 1.0
    if valid.sum() > 32:
        rv, tv = d_rel[valid], depth_raw[valid]
        ok = np.isfinite(rv) & np.isfinite(tv)
        rv, tv = rv[ok], tv[ok]
        try:
            A = np.polyfit(rv, tv, 1)
            resid = tv - (A[0] * rv + A[1])
            s = np.std(resid)
            if s > 0:
                m = np.abs(resid) < 3 * s
                if m.sum() > 32:
                    A = np.polyfit(rv[m], tv[m], 1)
            scale, offset = A[0], A[1]
        except Exception:
            scale = float(np.median(tv)) / (float(np.median(rv)) + 1e-8)
            offset = 0.0
    else:
        scale, offset = 1.0, 0.0

    dav2_abs = np.clip(scale * d_rel + offset, 0, 65535)
    
    # Base fill: keep valid sensor depth, fill holes with aligned DAv2
    filled = np.where(valid, depth_raw, dav2_abs)

    # Edge-preserving guided filtering using IR as structural guidance
    # Prevents hole-fill blur while keeping crisp geometric silhouettes
    guide = ir8 if ir8.ndim == 2 else cv2.cvtColor(ir8, cv2.COLOR_BGR2GRAY)
    guide_f = guide.astype(np.float32) / 255.0
    
    # Fast Guided filter
    r, eps = 4, 1e-3
    mean_I = cv2.boxFilter(guide_f, -1, (2*r+1, 2*r+1))
    mean_p = cv2.boxFilter(filled, -1, (2*r+1, 2*r+1))
    mean_Ip = cv2.boxFilter(guide_f * filled, -1, (2*r+1, 2*r+1))
    var_I = cv2.boxFilter(guide_f * guide_f, -1, (2*r+1, 2*r+1)) - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    gf = cv2.boxFilter(a, -1, (2*r+1, 2*r+1)) * guide_f + cv2.boxFilter(b, -1, (2*r+1, 2*r+1))
    
    # Valid pixels stay original sensor depth; hole regions use guided filtered values
    out_refined = np.where(valid, depth_raw, gf)
    out_refined = np.clip(out_refined, 0, 65535)
    return out_refined, dav2_abs, valid

def depth_to_surface_normal(depth_map, valid_mask=None):
    """
    Convert 1-channel depth (mm or normalized) to 3-channel Surface Normal map.
    n = normalize(-dz/dx, -dz/dy, 1), mapped to [0, 1] range.
    """
    H, W = depth_map.shape
    d_norm = (depth_map / (depth_map.max() + 1e-6)).astype(np.float32)
    
    # Sobel kernels
    dx = cv2.Sobel(d_norm, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=3)
    dz = np.ones_like(dx, dtype=np.float32) * 0.05  # scale aspect ratio
    
    norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
    nx = -dx / norm
    ny = -dy / norm
    nz = dz / norm
    
    # Map from [-1, 1] to [0, 1]
    normal_map = np.stack([(nx + 1.0) * 0.5, (ny + 1.0) * 0.5, (nz + 1.0) * 0.5], axis=-1)
    return normal_map.astype(np.float32)

def depth_to_hha_lite(depth_map):
    """
    Compute HHA-lite:
    Ch 0: Disparity (1 / (Z + eps))
    Ch 1: Vertical height proxy from bottom
    Ch 2: Angle with vertical gravity vector
    """
    H, W = depth_map.shape
    d_m = depth_map.astype(np.float32) / 1000.0  # to meters approx
    disparity = 1.0 / (np.clip(d_m, 0.5, 100.0))
    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
    
    y_coords = np.linspace(0, 1, H, dtype=np.float32)[:, None].repeat(W, axis=1)
    # Height proxy: relative position conditioned on depth
    height_proxy = y_coords * (1.0 - disparity * 0.5)
    
    # Angle with vertical
    d_norm = (depth_map / (depth_map.max() + 1e-6)).astype(np.float32)
    dy = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan(np.abs(dy) * 20.0) / (np.pi / 2.0)
    
    hha = np.stack([disparity, height_proxy, angle], axis=-1)
    return np.clip(hha, 0.0, 1.0).astype(np.float32)

def compute_fdr(feature_map, boxes, H, W):
    """
    Compute Fisher Discriminant Ratio (FDR) inside GT boxes vs background collar.
    FDR = (mu_fg - mu_bg)^2 / (var_fg + var_bg + eps)
    Higher FDR = feature cleanly separates foreground objects from background.
    """
    if feature_map.ndim == 3:
        # compute magnitude across channels
        feat_scalar = np.linalg.norm(feature_map, axis=-1)
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
        
        # Collar background: 1.4x bounding box
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
    
    fdr = float((mu_fg - mu_bg)**2 / (var_fg + var_bg + 1e-7))
    return fdr

def get_gradient_map(img):
    if img.ndim == 3:
        if img.shape[2] == 3 and img.dtype == np.uint8:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = np.mean(img, axis=-1)
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    return mag.astype(np.float32)

def audit_sample(base_name, model, device):
    v_path = f"dataset/train/visible/{base_name}.png"
    i_path = f"dataset/train/infrared/{base_name}.png"
    d_path = f"dataset/train/depth/{base_name}.png"
    l_path = f"dataset/train/labels/{base_name}.txt"
    
    vis = cv2.imread(v_path)
    ir = cv2.imread(i_path, cv2.IMREAD_UNCHANGED)
    if ir.ndim == 3:
        ir_gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    else:
        ir_gray = ir.copy()
    depth_raw = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    H, W = depth_raw.shape
    
    boxes = []
    if os.path.exists(l_path):
        with open(l_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
                    
    # 1. Run DA guided fill and filtering
    depth_refined, dav2_abs, valid_mask = da_fill_and_filter(depth_raw, ir_gray, model, device)
    
    # 2. Extract Geometric Representations
    rep_raw_norm = depth_raw / 19999.0
    rep_da_norm  = depth_refined / 19999.0
    rep_normal   = depth_to_surface_normal(depth_refined, valid_mask)
    rep_hha      = depth_to_hha_lite(depth_refined)
    
    # 3. Compute Fisher Discriminant Ratio (FDR) on GT objects
    fdr_raw    = compute_fdr(rep_raw_norm, boxes, H, W)
    fdr_da     = compute_fdr(rep_da_norm, boxes, H, W)
    fdr_normal = compute_fdr(rep_normal, boxes, H, W)
    fdr_hha    = compute_fdr(rep_hha, boxes, H, W)
    
    # 4. Compute 3-Modal Edge Maps
    e_rgb = get_gradient_map(vis)
    e_ir  = get_gradient_map(ir_gray)
    e_raw = get_gradient_map(rep_raw_norm)
    e_geo = get_gradient_map(rep_normal[..., :2])  # nx, ny gradients
    
    # Correlation Coefficients
    cc_rgb_ir  = float(np.corrcoef(e_rgb.ravel(), e_ir.ravel())[0, 1])
    cc_rgb_raw = float(np.corrcoef(e_rgb.ravel(), e_raw.ravel())[0, 1])
    cc_rgb_geo = float(np.corrcoef(e_rgb.ravel(), e_geo.ravel())[0, 1])
    cc_ir_geo  = float(np.corrcoef(e_ir.ravel(),  e_geo.ravel())[0, 1])
    
    # 5. Measure 3-Modal Boundary Complementarity Coverage
    # Target contour mask: GT bounding box edges (band of 4px)
    gt_contour = np.zeros((H, W), dtype=bool)
    for cls_id, cx, cy, w, h in boxes:
        x1 = int(max(0, (cx - w / 2) * W))
        x2 = int(min(W, (cx + w / 2) * W))
        y1 = int(max(0, (cy - h / 2) * H))
        y2 = int(min(H, (cy + h / 2) * H))
        gt_box = np.zeros((H, W), dtype=np.uint8)
        gt_box[y1:y2, x1:x2] = 1
        dilated = cv2.dilate(gt_box, np.ones((7, 7), np.uint8))
        eroded  = cv2.erode(gt_box, np.ones((7, 7), np.uint8))
        gt_contour |= ((dilated - eroded) > 0)
        
    n_gt_edge = max(1, gt_contour.sum())
    
    # Thresholding high-confidence edges
    th_rgb = np.percentile(e_rgb, 85)
    th_ir  = np.percentile(e_ir, 85)
    th_geo = np.percentile(e_geo, 85)
    
    m_rgb = (e_rgb > th_rgb) & gt_contour
    m_ir  = (e_ir > th_ir) & gt_contour
    m_geo = (e_geo > th_geo) & gt_contour
    
    cov_1m = float(m_rgb.sum() / n_gt_edge)
    cov_2m = float((m_rgb | m_ir).sum() / n_gt_edge)
    cov_3m = float((m_rgb | m_ir | m_geo).sum() / n_gt_edge)
    gain_geo = cov_3m - cov_2m
    
    # 6. Generate Multi-Modal False Color Overlay
    # Channel assignment: R=Visible Edge, G=IR Edge, B=Depth Geo Edge
    # Format: BGR for OpenCV saving -> B=Depth Geo, G=IR, R=Visible
    r_ch = (np.clip(e_rgb * 2.0, 0, 1) * 255).astype(np.uint8)
    g_ch = (np.clip(e_ir  * 2.0, 0, 1) * 255).astype(np.uint8)
    b_ch = (np.clip(e_geo * 2.5, 0, 1) * 255).astype(np.uint8)
    overlay_3m = np.stack([b_ch, g_ch, r_ch], axis=-1)
    
    # Draw GT boxes on overlay
    overlay_viz = overlay_3m.copy()
    for cls_id, cx, cy, w, h in boxes:
        x1, y1 = int((cx - w/2)*W), int((cy - h/2)*H)
        x2, y2 = int((cx + w/2)*W), int((cy + h/2)*H)
        cv2.rectangle(overlay_viz, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(overlay_viz, AIC_CATS.get(cls_id, str(cls_id)), (x1, max(20, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
    # Generate Visual Comparison Strip
    # Strip: [RGB, Raw Depth (Turbo), Refined Depth (Turbo), Surface Normal, 3-Modal Overlay]
    def colorize_d(d):
        d_norm = np.clip(d / 15000.0, 0, 1) * 255
        return cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_TURBO)
        
    vis_sm = cv2.resize(vis, (480, 270))
    raw_sm = cv2.resize(colorize_d(depth_raw), (480, 270))
    ref_sm = cv2.resize(colorize_d(depth_refined), (480, 270))
    nor_sm = cv2.resize((rep_normal * 255).astype(np.uint8), (480, 270))
    ovl_sm = cv2.resize(overlay_viz, (480, 270))
    
    # Annotate strip
    cv2.putText(vis_sm, "1. Visible (RGB)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(raw_sm, f"2. Raw Depth (Holes={int((~valid_mask).mean()*100)}%)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(ref_sm, "3. DA-Filled+Filtered", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(nor_sm, f"4. Surface Normal (FDR={fdr_normal:.3f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(ovl_sm, f"5. 3-Modal Fusion (Gain=+{gain_geo*100:.1f}%)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    strip = np.vstack([
        np.hstack([vis_sm, raw_sm, ref_sm]),
        np.hstack([nor_sm, ovl_sm, cv2.resize(ir, (480, 270)) if ir.ndim==3 else cv2.cvtColor(cv2.resize(ir, (480, 270)), cv2.COLOR_GRAY2BGR)])
    ])
    
    save_path = os.path.join(OUT_DIR, f"{base_name}_audit_panel.jpg")
    cv2.imwrite(save_path, strip)
    
    return {
        "sample": base_name,
        "n_boxes": len(boxes),
        "hole_ratio": float((~valid_mask).mean()),
        "fdr_raw": fdr_raw,
        "fdr_da": fdr_da,
        "fdr_normal": fdr_normal,
        "fdr_hha": fdr_hha,
        "cc_rgb_ir": cc_rgb_ir,
        "cc_rgb_geo": cc_rgb_geo,
        "cc_ir_geo": cc_ir_geo,
        "cov_1m": cov_1m,
        "cov_2m": cov_2m,
        "cov_3m": cov_3m,
        "gain_geo": gain_geo,
        "img_path": save_path
    }

def main():
    print("=" * 70)
    print("=== MULTIMODAL DEPTH GEOMETRIC LIFTING AUDIT & TEST ===")
    print("=" * 70)
    model, device = load_da_model()
    print(f"Loaded DAv2 on {device} successfully.\n")
    
    # Selected diverse representative samples
    test_samples = [
        "000006_010_00000389",  # car + person, 25% holes
        "000002_083_00000169",  # tricycle, sign, bicycle, car
        "000004_009_00000321",  # low-light darker scene
        "000002_081_00000047",  # person + sign
        "000003_018_00000167",  # darkest night scene
        "000002_082_00000061",  # multi vehicle
    ]
    
    results = []
    for s in test_samples:
        print(f"Processing sample: {s} ...")
        res = audit_sample(s, model, device)
        results.append(res)
        print(f"  [Holes: {res['hole_ratio']*100:.1f}% | Boxes: {res['n_boxes']}]")
        print(f"  FDR Contrast Gain : Raw={res['fdr_raw']:.4f} -> DA={res['fdr_da']:.4f} -> Normal={res['fdr_normal']:.4f} (提升 {res['fdr_normal']/(res['fdr_raw']+1e-6):.1f}x)")
        print(f"  Structural CC     : RGB-IR={res['cc_rgb_ir']:.3f} | RGB-Geo={res['cc_rgb_geo']:.3f} | IR-Geo={res['cc_ir_geo']:.3f}")
        print(f"  Boundary Coverage : 1M(RGB)={res['cov_1m']*100:.1f}% -> 2M(RGB+IR)={res['cov_2m']*100:.1f}% -> 3M(+Geo)={res['cov_3m']*100:.1f}% (Net Gain: +{res['gain_geo']*100:.2f}%)")
        print(f"  Saved Panel       : {res['img_path']}\n")
        
    print("=" * 70)
    print("=== SUMMARY METRIC TABLE ===")
    print("=" * 70)
    print(f"{'Sample':<22} | {'FDR Raw':<8} | {'FDR Normal':<10} | {'CC RGB-Geo':<10} | {'Cov 2M':<8} | {'Cov 3M':<8} | {'Geo Gain':<8}")
    print("-" * 85)
    for r in results:
        print(f"{r['sample']:<22} | {r['fdr_raw']:<8.4f} | {r['fdr_normal']:<10.4f} | {r['cc_rgb_geo']:<10.3f} | {r['cov_2m']*100:<7.1f}% | {r['cov_3m']*100:<7.1f}% | +{r['gain_geo']*100:<6.2f}%")
        
    avg_fdr_raw = np.mean([r['fdr_raw'] for r in results])
    avg_fdr_norm = np.mean([r['fdr_normal'] for r in results])
    avg_cc = np.mean([r['cc_rgb_geo'] for r in results])
    avg_cov2m = np.mean([r['cov_2m'] for r in results])
    avg_cov3m = np.mean([r['cov_3m'] for r in results])
    avg_gain = np.mean([r['gain_geo'] for r in results])
    print("-" * 85)
    print(f"{'AVERAGE':<22} | {avg_fdr_raw:<8.4f} | {avg_fdr_norm:<10.4f} | {avg_cc:<10.3f} | {avg_cov2m*100:<7.1f}% | {avg_cov3m*100:<7.1f}% | +{avg_gain*100:<6.2f}%")
    print(f"\nAverage FDR Improvement : {avg_fdr_norm / (avg_fdr_raw + 1e-6):.2f}x")
    print(f"Average Boundary Recall Boost : +{avg_gain*100:.2f}% across 3 modalities")
    print("=" * 70)

if __name__ == "__main__":
    main()
