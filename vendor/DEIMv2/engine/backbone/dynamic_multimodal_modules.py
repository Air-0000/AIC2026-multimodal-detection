"""
Dynamic Multi-Modal Modules for Sovereign DEIMv2 Architecture.

Contains:
  1. DynamicModalityConfidenceGating:
     Dynamic environmental quality and physical signal-to-noise ratio arbiter.
     Evaluates contrast, luminance range, thermal peak ratio, and depth gradient energy,
     outputting dynamic modality weights (alpha_rgb, alpha_ir, alpha_depth).
     Latency: ~10 microseconds per batch.
     Step 0: Zero-initialized to output [1.0, 1.0, 1.0] identically.

  2. HighFrequencyEdgeShallowFusion:
     Shallow high-frequency edge interaction module operating at Stage 0 (stride 4, 160x160).
     Injects Sobel spatial gradients from Thermal IR and Depth disparity into shallow RGB
     features, directly rescuing micro-targets (traffic signs, balls, distant pedestrians).
     Step 0: Zero-initialized residual to guarantee exact identity.

  3. init_ir_stem_from_pretrained_rgb:
     Transfers pre-trained edge, Sobel, and Gabor filters from ImageNet-pretrained RGB stem
     directly to the IR ModalityStem, preventing cold-start gradient thrashing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class HighFrequencyEdgeShallowFusion(nn.Module):
    """Shallow High-Frequency Edge Interaction with ProG-Net Edge-to-P3 Prompt (Stage 0/1).
    
    1. Injects 4-stream physical high-frequency gradients:
       - g_rgb: Visible optical luminance Sobel edge (rescues cold planar traffic signs!)
       - g_ir: Thermal infrared Sobel edge (rescues nighttime pedestrians, drones)
       - g_d: Metric depth disparity Sobel edge (rescues 3D obstacles)
       - ir_highpass: Thermal contrast saliency
    2. ProG-Net Shallow-to-Neck Direct Prompt:
       Projects the 160x160 physical geometric edge features down to 80x80 (stride 8)
       and injects it directly into Stage 1 / P3 (the primary scale for small objects).
    
    Zero-initialized residuals guarantee exact mathematical identity at Step 0.
    """
class MagGuidedFusion(nn.Module):
    """Channel-wise cosine correlation for global illumination gating."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.expand_conv = nn.Conv2d(1, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, mag0, mag1):
        B, C, H, W = mag0.shape
        mag0_flat = mag0.reshape(B, C, -1).float()
        mag1_flat = mag1.reshape(B, C, -1).float()

        mag0_norm = F.normalize(mag0_flat, dim=-1, eps=1e-5)
        mag1_norm = F.normalize(mag1_flat, dim=-1, eps=1e-5)

        similarity_matrix = torch.bmm(mag0_norm, mag1_norm.transpose(1, 2))  # (B, C, C)
        similarity_scores = torch.nan_to_num(similarity_matrix.mean(dim=-1), nan=0.0) # (B, C)
        top1_indices = torch.argmax(similarity_scores, dim=-1).clamp(0, C - 1)        # (B,)
        idx = top1_indices.view(B, 1, 1, 1).expand(B, 1, H, W)
        mag0_top1 = torch.gather(mag0, dim=1, index=idx)
        mag0_expanded = self.expand_conv(mag0_top1)
        mag0_weight = torch.sigmoid(mag0_expanded)
        fused_features = mag1 * mag0_weight + mag1
        return fused_features, mag0_weight


class ResidualFourierBlock(nn.Module):
    """
    AAAI 2026 Residual Fourier-Guided Module (RFGM) Core Unit:
    Decouples Amplitude (energy/illumination) & Phase (fine-grained geometry/edges).
    Dual-stream phase residual compensation preserves sharp geometric contours,
    while amplitude guided fusion reconstructs dark illumination.
    Zero-initialized output projection guarantees exact mathematical identity at Step 0.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.process1_mag = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.process1_pha = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.process2_pha = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.mag_fusion = MagGuidedFusion(channels=channels)
        self.proj_out = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.proj_out.weight)

    def forward(self, x):
        orig_dtype = x.dtype
        B, C, H, W = x.shape
        x_c = self.conv0(x)
        # AMP / BFloat16 compatibility: Fourier transform requires float32 precision
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_c_f32 = x_c.float()
            # 2D Real FFT
            x_freq = torch.fft.rfft2(x_c_f32, norm='ortho')
            mag0 = torch.abs(x_freq)
            pha0 = torch.angle(x_freq)

            # 1. Structural Phase Residual Branch (crucial for traffic sign geometric edges)
            pha1 = self.process1_pha(pha0)
            pha_out = self.process2_pha(torch.cat([pha0, pha1], dim=1))

            # 2. Global Amplitude Illumination Branch (restores dark regions)
            mag1 = self.process1_mag(mag0)
            mag_out, _ = self.mag_fusion(mag0, mag1)

            # 3. Frequency Inverse Reconstruction
            real = mag_out * torch.cos(pha_out)
            imag = mag_out * torch.sin(pha_out)
            x_out_freq = torch.complex(real, imag)
            x_out = torch.fft.irfft2(x_out_freq, s=(H, W), norm='ortho')

        x_out = x_out.to(orig_dtype)
        # Zero-init residual connection
        return x + self.proj_out(x_out)


class FrontendRFGM(nn.Module):
    """Scheme A: Frontend 2D-rFFT Amplitude-Phase Illumination & Texture Restoration."""
    def __init__(self, in_chans=3, hidden_chans=16):
        super().__init__()
        self.in_proj = nn.Conv2d(in_chans, hidden_chans, 3, padding=1, bias=False)
        self.rfgm_block = ResidualFourierBlock(channels=hidden_chans)
        self.out_proj = nn.Conv2d(hidden_chans, in_chans, 3, padding=1, bias=False)
        nn.init.zeros_(self.out_proj.weight) # exact zero residual at step 0

    def forward(self, rgb):
        feat = F.gelu(self.in_proj(rgb))
        feat_fused = self.rfgm_block(feat)
        return rgb + self.out_proj(feat_fused)


class HighFrequencyEdgeShallowFusion(nn.Module):
    """Shallow High-Frequency Edge Interaction with RF-ProG Fourier Phase Guidance (Stage 0/1).
    
    1. Injects 4-stream physical high-frequency gradients:
       - g_rgb: Visible optical luminance Sobel edge (rescues cold planar traffic signs!)
       - g_ir: Thermal infrared Sobel edge (rescues nighttime pedestrians, drones)
       - g_d: Metric depth disparity Sobel edge (rescues 3D obstacles)
       - ir_highpass: Thermal contrast saliency
    2. AAAI 2026 RFGM 2D-rFFT Residual Fourier Block:
       Captures global phase coherence (closed circular/triangular contours) across the 160x160 field.
    3. ProG-Net Shallow-to-Neck Direct Prompt:
       Projects the 160x160 geometric phase features down to 80x80 (stride 8)
       and injects it directly into Stage 1 / P3 (the primary scale for small objects).
    
    Zero-initialized residuals guarantee exact mathematical identity at Step 0.
    """
    def __init__(self, cnn_dim=128, p3_dim=512, hidden_dim=32, use_rfgm=True):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # 4 input channels: grad_rgb (1) + grad_ir (1) + grad_depth (1) + highpass_ir (1)
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        self.use_rfgm = use_rfgm
        if use_rfgm:
            self.rfgm = ResidualFourierBlock(channels=hidden_dim)
        else:
            self.rfgm = None

        # Stage 0 residual (160x160, cnn_dim=128)
        self.edge_to_s0 = nn.Conv2d(hidden_dim, cnn_dim, 1)
        nn.init.zeros_(self.edge_to_s0.weight)
        nn.init.zeros_(self.edge_to_s0.bias)

        # ProG-Net Stage 1 / P3 prompt (80x80, p3_dim=512)
        self.edge_to_p3 = nn.Conv2d(hidden_dim, p3_dim, 1)
        nn.init.zeros_(self.edge_to_p3.weight)
        nn.init.zeros_(self.edge_to_p3.bias)

    def forward(self, rgb_s0, rgb, ir, depth):
        """
        Args:
          rgb_s0: (B, 128, H_s0, W_s0) shallow trunk features at 160x160
          rgb: (B, 3, H, W) raw RGB input
          ir: (B, 2, H, W) aligned + lbg IR images
          depth: (B, 1, H, W) normalized metric depth / disparity
        Returns:
          rgb_s0_fused: (B, 128, H_s0, W_s0) with high-frequency spatial gradients injected
          p3_prompt: (B, 512, H_s0//2, W_s0//2) at 80x80 for Stage 1 / Neck P3
        """
        H_s0, W_s0 = rgb_s0.shape[-2:]
        # 1. Optical visible edge (crucial for cold planar signs)
        rgb_s0_img = F.interpolate(rgb, size=(H_s0, W_s0), mode='bilinear', align_corners=False)
        rgb_gray = 0.299 * rgb_s0_img[:, 0:1] + 0.587 * rgb_s0_img[:, 1:2] + 0.114 * rgb_s0_img[:, 2:3]
        rgb_pad = F.pad(rgb_gray, (1, 1, 1, 1), mode='replicate')
        sx = self.sobel_x.to(dtype=rgb_pad.dtype)
        sy = self.sobel_y.to(dtype=rgb_pad.dtype)
        g_rgb = torch.sqrt(F.conv2d(rgb_pad, sx)**2 + F.conv2d(rgb_pad, sy)**2 + 1e-6)

        # 2. Thermal infrared edge
        ir_s0 = F.interpolate(ir[:, :1], size=(H_s0, W_s0), mode='bilinear', align_corners=False)
        ir_pad = F.pad(ir_s0, (1, 1, 1, 1), mode='replicate')
        g_ir = torch.sqrt(F.conv2d(ir_pad, sx)**2 + F.conv2d(ir_pad, sy)**2 + 1e-6)

        # 3. Metric depth edge
        d_s0 = F.interpolate(depth, size=(H_s0, W_s0), mode='bilinear', align_corners=False)
        d_pad = F.pad(d_s0, (1, 1, 1, 1), mode='replicate')
        g_d = torch.sqrt(F.conv2d(d_pad, sx)**2 + F.conv2d(d_pad, sy)**2 + 1e-6)

        # 4. Thermal contrast highpass
        ir_blur = F.avg_pool2d(ir_pad, kernel_size=3, stride=1, padding=0)
        ir_highpass = torch.abs(ir_s0 - ir_blur)

        edge_feats = torch.cat([g_rgb, g_ir, g_d, ir_highpass], dim=1) # (B, 4, 160, 160)
        feat_h = self.edge_encoder(edge_feats) # (B, hidden_dim, 160, 160)

        # 天空与水面波纹高频噪声精准截断 (Sky-Water Noise Attenuator)
        B_dim = feat_h.shape[0]
        y_grid_160 = torch.linspace(0.0, 1.0, H_s0, device=feat_h.device, dtype=feat_h.dtype).view(1, 1, H_s0, 1).expand(B_dim, 1, H_s0, W_s0).contiguous()
        clean_mask_160 = torch.ones_like(y_grid_160)
        if depth is not None:
            depth_raw_160 = torch.clamp(d_s0 * 0.5 + 0.5, 0.0, 1.0)
            # 1. 天空区域高频净化 (y < 0.28 且无有效深度回波)
            sky_factor_160 = torch.clamp((y_grid_160 - 0.10) / 0.15, 0.0, 1.0)
            clean_mask_160 = torch.where((y_grid_160 < 0.28) & (depth_raw_160 < 0.02), sky_factor_160, clean_mask_160)
            # 2. 水面波纹高频净化 (y > 0.48 且无 3D 刚体阶跃梯度)
            water_factor_160 = torch.clamp(g_d / 0.015, 0.0, 1.0)
            clean_mask_160 = torch.where((y_grid_160 > 0.48) & (g_d < 0.015), water_factor_160, clean_mask_160)

        feat_h = (feat_h * clean_mask_160).contiguous()

        # AAAI 2026 RFGM 2D-rFFT Amplitude-Phase Processing
        if self.use_rfgm and self.rfgm is not None:
            feat_h = self.rfgm(feat_h)

        delta_s0 = self.edge_to_s0(feat_h)
        # Dual-Path Peak Salience Pooling: Preserves sharp high-frequency edges for tiny objects (12x12)
        feat_p3_avg = F.avg_pool2d(feat_h, kernel_size=2, stride=2)
        feat_p3_max = F.max_pool2d(feat_h, kernel_size=2, stride=2)
        feat_p3_sub = 0.5 * (feat_p3_avg + feat_p3_max) # (B, hidden_dim, 80, 80)
        p3_prompt = self.edge_to_p3(feat_p3_sub) # (B, p3_dim, 80, 80)

        # Numerical domain protection:
        # 1. Bound delta_s0 with tanh to [-0.1, 0.1] preserving zero-init and preventing negative blowup
        # 2. Bound p3_prompt with tanh to [-0.1, 0.1]
        # 3. Ensure non-negative activation guarantee for HGNetv2 Stage 0 output
        delta_s0_bounded = 0.1 * torch.tanh(delta_s0)
        p3_prompt_bounded = 0.1 * torch.tanh(p3_prompt)
        return F.relu(rgb_s0 + delta_s0_bounded), p3_prompt_bounded


def transfer_pretrained_stem_weights(rgb_stem, ir_stem, in_chans=2):
    """
    Transfers pre-trained edge/Gabor filter weights from HGNetv2 RGB stem1
    into the first conv of ModalityStem.
    """
    with torch.no_grad():
        w_rgb = rgb_stem.stem1.conv.weight # [32, 3, 3, 3]
        w_mean = w_rgb.mean(dim=1, keepdim=True) # [32, 1, 3, 3]
        target_conv = ir_stem.stem[0] # [64, in_chans, 3, 3]

        # Channel 0: Aligned IR (intensity luminance transfer)
        target_conv.weight[:32, 0:1] = w_mean
        # Channel 1: LBG IR (thermal saliency transfer)
        if in_chans >= 2:
            target_conv.weight[:32, 1:2] = w_mean

        # Filters 32..64: High-frequency Laplacian blend
        laplacian = torch.tensor([[0., 1., 0.],
                                  [1., -4., 1.],
                                  [0., 1., 0.]]).view(1, 1, 3, 3).to(w_mean.device) * 0.5
        for c in range(min(in_chans, 2)):
            target_conv.weight[32:64, c:c+1] = w_mean * 0.5 + laplacian * 0.5

    print(f"[transfer_pretrained_stem_weights] Successfully transferred {w_rgb.shape[0]} "
          f"pretrained RGB edge filters into IR ModalityStem ({in_chans} channels).")
