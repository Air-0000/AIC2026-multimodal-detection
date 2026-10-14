"""
CSSP (Cascaded Spectral-Spatial Perception) & ABC (Attention with Bilinear Correlation)
Multimodal Enhancement Modules for DEIMv2.

References:
1. MODA / OSSDet (AAAI):
   - Cascaded Spectral-Spatial joint Perception (CSSP)
   - Cross-Spectral Feature Refinement & Object-Aware Saliency Masking (CSFR)
   - Formula: Spectral Diff |F_rgb - F_ir| -> Multi-scale Spatial Saliency Mask M -> F * (1 + M)
2. ABC (ICME 2023):
   - Attention with Bilinear Correlation for Infrared Small Target Detection
   - Preserves high-resolution shallow details via skip connection bypass
   - Enhances local contrast to prevent small targets from being submerged by background clutter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.core.workspace import register


class SqueezeExcitation(nn.Module):
    """Channel attention module for spectral discrepancy reweighting."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 8), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(channels // reduction, 8), channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class CSSPModule(nn.Module):
    """
    Cascaded Spectral-Spatial Perception (CSSP) Module.
    
    1. Spectral Perception:
       - Computes cross-spectral discrepancy: |F_rgb - F_ir|
       - Fuses [F_rgb, F_ir, |F_rgb - F_ir|] with spectral channel attention.
    2. Spatial Perception:
       - Cascaded multi-scale receptive field (standard 3x3 + dilated 3x3)
         to perceive point-like small targets against surrounding background.
    3. Object-Aware Saliency Masking:
       - Predicts a normalized 2D saliency mask M in [0, 1].
       - Amplifies target features: F_enhanced = F_fused * (1 + alpha * M).
       - Zero-initialized alpha guarantees exact identity at Step 0.
    """
    def __init__(self, rgb_dim, ir_dim, out_dim=None, mid_dim=64):
        super().__init__()
        out_dim = out_dim or rgb_dim
        self.rgb_dim = rgb_dim
        self.ir_dim = ir_dim
        self.out_dim = out_dim
        
        # 1. Spectral Projection (align channel dimensions if needed)
        self.rgb_proj = nn.Conv2d(rgb_dim, mid_dim, 1, bias=False) if rgb_dim != mid_dim else nn.Identity()
        self.ir_proj = nn.Conv2d(ir_dim, mid_dim, 1, bias=False) if ir_dim != mid_dim else nn.Identity()
        
        # Spectral fusion: [F_rgb, F_ir, |F_rgb - F_ir|] -> 3 * mid_dim
        self.spectral_fusion = nn.Sequential(
            nn.Conv2d(mid_dim * 3, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            SqueezeExcitation(mid_dim, reduction=4)
        )
        
        # 2. Multi-scale Cascaded Spatial Perception
        self.spatial_conv1 = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1, groups=mid_dim, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.GELU()
        )
        self.spatial_conv2 = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 3, padding=2, dilation=2, groups=mid_dim, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.GELU()
        )
        
        # 3. Object-Aware Saliency Mask Head
        self.saliency_head = nn.Sequential(
            nn.Conv2d(mid_dim * 2, mid_dim // 2, 1, bias=False),
            nn.BatchNorm2d(mid_dim // 2),
            nn.GELU(),
            nn.Conv2d(mid_dim // 2, 1, 1, bias=True)
        )
        
        # 4. Feature projection back to out_dim
        self.out_proj = nn.Sequential(
            nn.Conv2d(mid_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        
        # Learnable gating scalar (Zero-init -> exact identity at Step 0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.mask_scale = nn.Parameter(torch.tensor(1.0))
        
        # Zero-initialize the last projection to guarantee exact identity at Step 0
        nn.init.zeros_(self.saliency_head[-1].weight)
        nn.init.zeros_(self.saliency_head[-1].bias)

    def forward(self, rgb_f, ir_f):
        """
        rgb_f: [B, C_rgb, H, W]
        ir_f:  [B, C_ir, H, W]
        returns: enhanced_feat [B, C_out, H, W], saliency_mask [B, 1, H, W]
        """
        # Interpolate IR if spatial sizes differ slightly
        if ir_f.shape[2:] != rgb_f.shape[2:]:
            ir_f = F.interpolate(ir_f, size=rgb_f.shape[2:], mode='bilinear', align_corners=False)
            
        f_rgb = self.rgb_proj(rgb_f)
        f_ir = self.ir_proj(ir_f)
        
        # Spectral discrepancy: |F_rgb - F_ir|
        f_diff = torch.abs(f_rgb - f_ir)
        
        # Spectral Joint Fusion
        cat_spectral = torch.cat([f_rgb, f_ir, f_diff], dim=1)
        f_spectral = self.spectral_fusion(cat_spectral)
        
        # Cascaded Multi-Scale Spatial Perception
        s1 = self.spatial_conv1(f_spectral)
        s2 = self.spatial_conv2(f_spectral)
        f_spatial = torch.cat([s1, s2], dim=1)
        
        # Saliency Mask in [0, 1]
        saliency_mask = torch.sigmoid(self.saliency_head(f_spatial))
        
        # Object-Aware Saliency Enhancement
        f_enhanced = f_spectral * (1.0 + self.mask_scale * saliency_mask)
        f_out = self.out_proj(f_enhanced)
        
        # Residual addition: At Step 0 (gamma=0), exactly rgb_f!
        out = rgb_f + self.gamma * f_out
        return out, saliency_mask


class ABCLcmAttention(nn.Module):
    """
    ABC (Attention with Bilinear Correlation / Local Contrast Measure) Module.
    
    Implements shallow high-resolution skip-connection and local contrast attention:
    1. Computes local center vs surround contrast using difference-of-Gaussians (DoG) kernels.
    2. Uses depthwise bilinear correlation to suppress flat background clutter and
       boost localized point targets (drones, balls, uavs).
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Depthwise local contrast filter (Center vs Surround)
        self.dw_contrast = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        # Initialize dw_contrast with a 5x5 Laplacian / Mexican-Hat like kernel
        with torch.no_grad():
            k5 = torch.tensor([
                [-0.04, -0.04, -0.04, -0.04, -0.04],
                [-0.04,  0.08,  0.16,  0.08, -0.04],
                [-0.04,  0.16,  0.32,  0.16, -0.04],
                [-0.04,  0.08,  0.16,  0.08, -0.04],
                [-0.04, -0.04, -0.04, -0.04, -0.04]
            ], dtype=torch.float32)
            for c in range(channels):
                self.dw_contrast.weight.data[c, 0] = k5

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        
        # Bilinear channel correlation
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.alpha = nn.Parameter(torch.zeros(1))  # Zero-init identity

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        contrast = self.dw_contrast(x)
        contrast = self.bn(contrast)
        contrast = self.act(contrast)
        attn = torch.sigmoid(self.pw(contrast))
        
        # Modulated enhancement with zero-init residual
        out = x + self.alpha * (x * attn)
        return out


@register()
class JointCSSPABCShallowFusion(nn.Module):
    """
    Combined Cascaded Spectral-Spatial Perception + ABC Local Contrast Module
    designed specifically for shallow stage fusion in DEIMv2.
    
    Can be placed at:
    - Stage 0 (160x160): Captures ultra-fine drone structures before downsampling.
    - Stage 1 (80x80): Enriches P3 features for small-target queries.
    """
    def __init__(self, rgb_dim=64, ir_dim=64, out_dim=64, use_abc=True):
        super().__init__()
        self.cssp = CSSPModule(rgb_dim, ir_dim, out_dim=out_dim)
        self.abc = ABCLcmAttention(out_dim) if use_abc else nn.Identity()

    def forward(self, rgb_f, ir_f):
        f_cssp, mask = self.cssp(rgb_f, ir_f)
        f_final = self.abc(f_cssp)
        return f_final, mask
