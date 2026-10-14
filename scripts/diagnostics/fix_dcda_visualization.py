import os, sys, glob, time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, "/root/autodl-tmp/AIC/vendor/DEIMv2")
from engine.core import YAMLConfig

MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(1, 6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(1, 6, 1, 1)

def find_file(directory, stem):
    for ext in [".png", ".jpg", ".jpeg"]:
        p = os.path.join(directory, stem + ext)
        if os.path.exists(p):
            return p
    return None

def load_test_sample(stem, root="/root/autodl-tmp/AIC"):
    vis_p = find_file(os.path.join(root, "dataset/test/AIC2026_PHASE_2_1000/visible"), stem)
    ir_p = find_file(os.path.join(root, "dataset/test/AIC2026_PHASE_2_1000/infrared"), stem)
    depth_p = find_file(os.path.join(root, "dataset/test/AIC2026_PHASE_2_1000/depth_p0"), stem)
    depth_raw_p = find_file(os.path.join(root, "dataset/test/AIC2026_PHASE_2_1000/depth"), stem)
    
    vis_pil = Image.open(vis_p).convert("RGB")
    ir_pil = Image.open(ir_p).convert("L")
    depth_pil = Image.open(depth_p)
    depth_raw_pil = Image.open(depth_raw_p).convert("L") if depth_raw_p else None
    
    orig_w, orig_h = vis_pil.size
    
    rgb_t = T.functional.pil_to_tensor(vis_pil).float() / 255.0
    ir_t = T.functional.pil_to_tensor(ir_pil).float() / 255.0
    lbg_t = ir_t.clone()
    
    d_arr = np.array(depth_pil).astype(np.float32)
    if d_arr.ndim == 3:
        d_arr = d_arr.mean(axis=2)
    scale = 19999.0 if getattr(depth_pil, "mode", "") in ("I;16", "I") else 255.0
    d_norm = np.clip(d_arr / scale, 0.0, 1.0)
    depth_t = torch.from_numpy(d_norm).unsqueeze(0)
    
    sample = torch.cat([rgb_t, ir_t, lbg_t, depth_t], dim=0).unsqueeze(0)
    sample = F.interpolate(sample, size=(1152, 2048), mode="bilinear", align_corners=False)
    sample_norm = (sample - MEAN_6CH) / STD_6CH
    
    return sample_norm, rgb_t, ir_t, depth_t, d_arr, depth_raw_pil, orig_w, orig_h

# Load model
cfg_path = "/root/autodl-tmp/AIC/vendor/DEIMv2/configs/deimv2/sovereign_v28_disparity_awakening_2048.yml"
ckpt_path = "/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/sovereign_v28_disparity_awakening_2048/best_stg1.pth"

cfg = YAMLConfig(cfg_path)
model = cfg.model
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()
model.cuda()

stems_to_test = ["00000018", "000351", "001539"]

captured = {}
def hook_dcda(stage_idx):
    def hook_fn(module, inputs, output):
        rgb_f, ir_f, inv_d = inputs
        with torch.no_grad():
            z_rgb = module.reduce_rgb(rgb_f)
            z_ir = module.reduce_ir(ir_f)
            z_disp = module.disp_proj(inv_d)
            cat_feat = torch.cat([z_rgb, z_ir, z_disp], dim=1)
            o = module.offset_net(cat_feat)
            off_res, mask = torch.split(o, [18, 9], dim=1)
            mask = torch.sigmoid(mask)
            phys_dy = module.phys_disp_scale[0:1] * inv_d * 10.0
            phys_dx = module.phys_disp_scale[1:2] * inv_d * 10.0
            phys_prior = torch.cat([phys_dy, phys_dx], dim=1).repeat(1, 9, 1, 1)
            total_off = off_res + phys_prior
            alpha = torch.clamp(torch.tanh(module.gate_net(cat_feat)), min=0.0, max=1.0)
            warped_z = module.dcn(z_ir, total_off, mask)
            warped_ir = module.out_norm(module.expand(warped_z))
            
            lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=rgb_f.device).view(1, 1, 3, 3)
            e_rgb = F.conv2d(z_rgb.mean(dim=1, keepdim=True), lap, padding=1).abs()
            e_ir_raw = F.conv2d(z_ir.mean(dim=1, keepdim=True), lap, padding=1).abs()
            e_ir_warped = F.conv2d(warped_z.mean(dim=1, keepdim=True), lap, padding=1).abs()
            
            captured["stage_" + str(stage_idx)] = {
                "inv_d": inv_d[:1].detach().cpu(),
                "total_off": total_off[:1].detach().cpu(),
                "alpha": alpha[:1].detach().cpu(),
                "z_rgb_edge": e_rgb[:1, 0].detach().cpu(),
                "z_ir_raw_edge": e_ir_raw[:1, 0].detach().cpu(),
                "z_ir_warped_edge": e_ir_warped[:1, 0].detach().cpu(),
            }
    return hook_fn

for i, align_mod in enumerate(model.backbone.ir_align):
    align_mod.register_forward_hook(hook_dcda(i))

os.makedirs("/root/autodl-tmp/AIC/outputs/dcda_audit_fixed", exist_ok=True)

for stem in stems_to_test:
    sample_norm, rgb_raw, ir_raw, depth_raw, d_arr_raw, raw_depth_pil, orig_w, orig_h = load_test_sample(stem)
    with torch.no_grad():
        _ = model(sample_norm.cuda())
    
    d0 = captured["stage_0"]
    inv_d0 = d0["inv_d"][0, 0].numpy()
    alpha0 = d0["alpha"][0, 0].numpy()
    total_off0 = d0["total_off"][0].numpy()
    dy0 = total_off0[0::2]
    dx0 = total_off0[1::2]
    off_mag0 = np.sqrt(dy0**2 + dx0**2).mean(axis=0)
    
    e_rgb = d0["z_rgb_edge"][0].numpy()
    e_ir_raw = d0["z_ir_raw_edge"][0].numpy()
    e_ir_warped = d0["z_ir_warped_edge"][0].numpy()
    
    # Clean boundary padding artifacts (2px border)
    H_e, W_e = e_rgb.shape
    e_rgb_clean = e_rgb.copy()
    e_ir_clean = e_ir_warped.copy()
    e_rgb_clean[:2, :] = 0; e_rgb_clean[-2:, :] = 0; e_rgb_clean[:, :2] = 0; e_rgb_clean[:, -2:] = 0
    e_ir_clean[:2, :] = 0; e_ir_clean[-2:, :] = 0; e_ir_clean[:, :2] = 0; e_ir_clean[:, -2:] = 0
    
    # 98th percentile contrast stretching for vivid edge visibility
    p98_rgb = np.percentile(e_rgb_clean[2:-2, 2:-2], 98) + 1e-6
    p98_ir = np.percentile(e_ir_clean[2:-2, 2:-2], 98) + 1e-6
    
    norm_rgb_e = np.clip(e_rgb_clean / p98_rgb, 0.0, 1.0)
    norm_ir_w = np.clip(e_ir_clean / p98_ir, 0.0, 1.0)
    
    # Multi-modal edge overlay: Red = RGB edge, Green = Aligned Warped IR edge
    # Perfect alignment produces bright yellow
    overlay = np.zeros((H_e, W_e, 3), dtype=np.float32)
    overlay[:, :, 0] = norm_rgb_e
    overlay[:, :, 1] = norm_ir_w
    
    # Also create a Before vs After Edge comparison:
    e_ir_raw_clean = e_ir_raw.copy()
    e_ir_raw_clean[:2, :] = 0; e_ir_raw_clean[-2:, :] = 0; e_ir_raw_clean[:, :2] = 0; e_ir_raw_clean[:, -2:] = 0
    p98_ir_raw = np.percentile(e_ir_raw_clean[2:-2, 2:-2], 98) + 1e-6
    norm_ir_raw = np.clip(e_ir_raw_clean / p98_ir_raw, 0.0, 1.0)
    
    overlay_before = np.zeros((H_e, W_e, 3), dtype=np.float32)
    overlay_before[:, :, 0] = norm_rgb_e
    overlay_before[:, :, 1] = norm_ir_raw
    
    # Metric Depth Visualization:
    # d_arr_raw in mm. Convert to meters.
    # Sky is clamped to 65535 (65.5m). Foreground ground is 0.2m..5m.
    # We display depth with a dedicated LogNorm or robust percentile colormap so the ground gradient is crystal clear!
    d_meters = d_arr_raw.copy() / 1000.0
    ground_mask = d_meters < 10.0
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Panel 1: RGB
    axes[0, 0].imshow(rgb_raw.permute(1, 2, 0).numpy())
    axes[0, 0].set_title(f"Visible RGB ({stem})", fontsize=13, fontweight='bold')
    axes[0, 0].axis("off")
    
    # Panel 2: IR
    axes[0, 1].imshow(ir_raw[0].numpy(), cmap="inferno")
    axes[0, 1].set_title("Raw Infrared Thermal (IR)", fontsize=13, fontweight='bold')
    axes[0, 1].axis("off")
    
    # Panel 3: High-Contrast Metric Depth
    d_vis = np.clip(d_meters, 0.1, 40.0)
    im_d = axes[0, 2].imshow(d_vis, cmap="turbo", norm=mcolors.LogNorm(vmin=0.15, vmax=40.0))
    axes[0, 2].set_title("Metric Depth Z (Log Scale 0.15m - 40m)", fontsize=13, fontweight='bold')
    cbar_d = plt.colorbar(im_d, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar_d.set_label("Distance (m)", fontsize=10)
    axes[0, 2].axis("off")
    
    # Panel 4: DCDA Offset Magnitude
    im3 = axes[1, 0].imshow(off_mag0, cmap="magma")
    axes[1, 0].set_title("DCDA Offset Field Magnitude |Δp| (Stride 16)", fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].axis("off")
    
    # Panel 5: Gate Alpha
    im4 = axes[1, 1].imshow(alpha0, cmap="plasma", vmin=0, vmax=max(0.01, float(alpha0.max())))
    axes[1, 1].set_title(f"Convex Gate Alpha Map (Max={alpha0.max():.3f})", fontsize=13, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].axis("off")
    
    # Panel 6: Edge Overlay (Red: RGB, Green: Warped IR, Yellow: Aligned!)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Edge Alignment (Red: RGB, Green: Warped IR, Yellow: Aligned)", fontsize=13, fontweight='bold')
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    out_fig = f"/root/autodl-tmp/AIC/outputs/dcda_audit_fixed/audit_fixed_{stem}.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated fixed audit: {out_fig}")

    # Generate a direct Before vs After Edge Overlay Comparison
    fig2, (ax_bef, ax_aft) = plt.subplots(1, 2, figsize=(16, 8))
    ax_bef.imshow(overlay_before)
    ax_bef.set_title("BEFORE Alignment (Raw IR & RGB Edges) - Note Misaligned Red/Green Ghosting", fontsize=12, fontweight='bold')
    ax_bef.axis("off")
    
    ax_aft.imshow(overlay)
    ax_aft.set_title("AFTER DCDA Alignment (Warped IR & RGB Edges) - Snapped into Unified Yellow", fontsize=12, fontweight='bold')
    ax_aft.axis("off")
    
    plt.tight_layout()
    out_fig2 = f"/root/autodl-tmp/AIC/outputs/dcda_audit_fixed/compare_before_after_{stem}.png"
    plt.savefig(out_fig2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated before/after edge comparison: {out_fig2}")

print("All diagnostic figures successfully rendered!")
