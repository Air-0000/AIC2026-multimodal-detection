"""HGNetv2 multimodal backbone for D-FINE-X (RGB main stream + IR auxiliary).

Migration of the RDTTrack (ViT) multimodal recipe onto the D-FINE-X CNN trunk:

  * RGB main stream = stock HGNetv2 (D-FINE-X), unchanged forward semantics,
    same return_idx / _out_channels interface as the single-modality backbone,
    so HybridEncoder / DFINETransformer configs stay byte-identical.
  * IR auxiliary stream = ModalityStem (the proven 3-conv stem from
    multimodal_backbones) on channels [3 : 3+ir_in_chans] of the 6ch cached
    input (heat residual + raw IR; the trailing depth channel is ignored).
  * Injection = per-stage CSSA (CSD, CVPR 2023) or BiDirFusion cross-attention
    fused feature, added back through a ZERO-INIT 1x1 conv so the module is an
    exact identity at step 0 -- day behaviour is anchored to the RGB baseline
    and night opens up as the residual learns (same trick as DCNResidual /
    RGBGuidedDCN already in this repo).
  * Modality dropout (_zero_stream, ir_drop_prob) matches RDTTrack training.

The wrapper mirrors HGNetv2.forward stage-by-stage so dcn_res refinements and
return_idx bookkeeping behave exactly as the single-modality backbone.
All aux submodules are prefixed `ir_` so optimizer param-group regexes can
select the whole from-scratch stream (backbone\\.ir_ -> head lr).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.ops as ops

from .hgnetv2 import HGNetv2
from .multimodal_backbones import ModalityStem, CSSA, BiDirFusion, CmDRMFusion, _zero_stream
from .dynamic_multimodal_modules import (
    HighFrequencyEdgeShallowFusion,
    transfer_pretrained_stem_weights
)

from engine.core.workspace import register


class DepthConditionedDeformableAlignment(nn.Module):
    """Depth-Conditioned Deformable Alignment (DCDA) for Cross-Modal IR-RGB Fusion.
    
    Predicts sampling offsets from [RGB_feat, IR_feat, 1/Z disparity prior] to sample
    IR features along epipolar curves without tearing continuous physical structures.
    Uses depthwise-DCN bottleneck (same as HGNetv2 DCNResidual) to keep param debt minimal (<0.4M).
    
    Upgraded for Physical Disparity Awakening (路径 B):
      1. Physical Epipolar Highway: Explicit learnable scaling of 1/Z disparity vector [dy, dx].
      2. Adaptive Gated Convex Replacement: Replaces unaligned feature smoothly via
         aligned_ir = ir_f + alpha * (warped_ir - ir_f), eradicating double-edge ghosting.
      3. Cross-Modal High-Frequency Consistency Cache for end-to-end alignment guidance.
    """
    def __init__(self, cnn_dim, red=64):
        super().__init__()
        self.reduce_rgb = nn.Conv2d(cnn_dim, red, 1, bias=False)
        self.reduce_ir = nn.Conv2d(cnn_dim, red, 1, bias=False)
        self.disp_proj = nn.Conv2d(1, red // 4, 1, bias=False)
        
        # Physical disparity scale: [scale_y, scale_x] (init to small non-zero so gradients flow from step 0)
        self.phys_disp_scale = nn.Parameter(torch.tensor([0.01, 0.01]).view(2, 1, 1))
        
        in_pred = red * 2 + red // 4
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_pred, red, 3, padding=1, bias=False),
            nn.BatchNorm2d(red),
            nn.GELU(),
            nn.Conv2d(red, 27, 3, padding=1)  # 18 offsets + 9 modulation masks
        )
        self.dcn = ops.DeformConv2d(red, red, 3, padding=1, groups=red, bias=False)
        self.expand = nn.Conv2d(red, cnn_dim, 1, bias=False)
        self.out_norm = nn.BatchNorm2d(cnn_dim)
        
        # Convex replacement gate net (zero-init -> exact identity at Step 0)
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_pred, red // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(red // 4),
            nn.GELU(),
            nn.Conv2d(red // 4, 1, 1)
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)
        
        # Zero-init offset net and expand weight to guarantee exact identity at Step 0
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)
        nn.init.zeros_(self.expand.weight)
        
        # Cache for loss (bottleneck level for direct gradient flow)
        self.cache_z_rgb = None
        self.cache_warped_z = None
        self.cache_off = None
        self.cache_inv_d = None

    def forward(self, rgb_f, ir_f, inv_d):
        z_rgb = self.reduce_rgb(rgb_f)
        z_ir = self.reduce_ir(ir_f)
        z_disp = self.disp_proj(inv_d)
        cat_feat = torch.cat([z_rgb, z_ir, z_disp], dim=1)
        
        o = self.offset_net(cat_feat)
        off_res, mask = torch.split(o, [18, 9], dim=1)
        mask = torch.sigmoid(mask)
        
        # Physical epipolar prior: [B, 1, H, W] * scale
        phys_dy = self.phys_disp_scale[0:1] * inv_d * 10.0
        phys_dx = self.phys_disp_scale[1:2] * inv_d * 10.0
        phys_prior = torch.cat([phys_dy, phys_dx], dim=1).repeat(1, 9, 1, 1).contiguous()
        total_off = (off_res + phys_prior).contiguous()
        mask = mask.contiguous()
        z_ir = z_ir.contiguous()
        
        warped_z = self.dcn(z_ir, total_off, mask)
        warped_ir = self.out_norm(self.expand(warped_z))
        
        # Convex alignment gate: alpha in [0, 1]
        # At step 0: gate_net is 0 -> tanh(0) = 0 -> alpha = 0 -> aligned_ir = ir_f (exact identity)
        alpha = torch.clamp(torch.tanh(self.gate_net(cat_feat)), min=0.0, max=1.0)
        aligned_ir = ir_f + alpha * (warped_ir - ir_f)
        
        if self.training:
            self.cache_z_rgb = z_rgb
            self.cache_warped_z = warped_z
            self.cache_off = total_off
            self.cache_inv_d = inv_d
            
        return aligned_ir


class MetricCoordinateLift3D(nn.Module):
    """3D Metric Coordinate Lift (3D-Lift) for Multi-Modal Representation.
    
    Transforms 2D feature grid (u, v) and metric depth Z into physical 3D camera coordinates:
        X = (u - cx) / fx * Z
        Y = (v - cy) / fy * Z
        Z = Z
    Projects (X, Y, Z) via lightweight Conv to feature dimension C.
    Zero-initialized residual to guarantee exact identity at Step 0.
    """
    def __init__(self, cnn_dim, hidden_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, cnn_dim, 1, bias=False),
            nn.BatchNorm2d(cnn_dim)
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, depth, feat_shape):
        B, C, H_f, W_f = feat_shape
        d_sub = F.interpolate(depth, size=(H_f, W_f), mode='nearest')
        Z = torch.clamp(d_sub * 20.0, min=0.3, max=20.0)
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H_f, device=depth.device, dtype=depth.dtype),
            torch.linspace(-1.0, 1.0, W_f, device=depth.device, dtype=depth.dtype),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).contiguous()
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).contiguous()
        
        X = grid_x * 0.75 * Z
        Y = grid_y * 0.55 * Z
        pts_3d = torch.cat([X, Y, Z], dim=1).contiguous()
        pos_3d = self.proj(pts_3d)
        return pos_3d


class DepthOcclusionGating(nn.Module):
    """Depth Discontinuity Occlusion Gating (DOG) for Multi-Modal Fusion.
    
    Detects depth step jumps (|∇Z|) between foreground and background objects.
    In thermal IR, heat easily bleeds across boundaries into the background.
    DOG calculates depth gradient magnitude, learns a boundary occlusion mask,
    and gates the IR features to prevent thermal bloom from polluting 
    background RGB representations.
    Zero-initialized to guarantee exact identity at Step 0.
    """
    def __init__(self, cnn_dim, hidden_dim=32):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        self.gate_net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, cnn_dim, 1, bias=True)
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, ir_feat, depth):
        B, C, H_f, W_f = ir_feat.shape
        d_sub = F.interpolate(depth, size=(H_f, W_f), mode='bilinear', align_corners=False)
        Z = d_sub * 20.0
        
        d_pad = F.pad(Z, (1, 1, 1, 1), mode='replicate').contiguous()
        sx = self.sobel_x.to(dtype=d_pad.dtype)
        sy = self.sobel_y.to(dtype=d_pad.dtype)
        gx = F.conv2d(d_pad, sx)
        gy = F.conv2d(d_pad, sy)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        
        grad_norm = torch.clamp(grad_mag / 5.0, 0.0, 1.0)
        z_norm = torch.clamp(Z / 20.0, 0.0, 1.0)
        
        edge_feat = torch.cat([grad_norm, z_norm], dim=1).contiguous()
        gate = torch.tanh(self.gate_net(edge_feat))
        return ir_feat * (1.0 + gate)


@register()
class HGNetv2MultiModal(nn.Module):
    """D-FINE-X RGB trunk (HGNetv2) + IR aux stream injected at stage outputs.

    Args mirror HGNetv2 one-to-one (forwarded verbatim), plus:
      ir_in_chans:   channels sliced as x[:, 3:3+ir_in_chans] (2 = heat+raw_ir)
      inject_stages: 0-based stage indices receiving IR injection. Stage strides
                     are [4, 8, 16, 32]; ModalityStem yields 1/8 and 1/16 maps,
                     so stages with stride 8/16 get direct features and any
                     other stage gets bilinear-resized c3 (degenerate but kept
                     for flexibility). Default [2, 3] = strides 8/16, channels
                     [512, 1024] on B5.
      fusion:        'cssa' (channel switching + spatial attention) or
                     'bidir' (BiDirFusion scaled dot-product cross-attention).
      use_dcda:      True to enable Depth-Conditioned Deformable Alignment (DCDA).
      stem_base:     ModalityStem width (outputs 2*stem_base channels).
      ir_drop_prob:  training-time modality dropout of the whole IR stream.
    """

    def __init__(self,
                 name='B5',
                 ir_in_chans=1,
                 inject_stages=[2, 3],
                 fusion='cssa',
                 use_dcda=False,
                 use_3d_lift=False,
                 use_dog=False,
                 use_bilateral_pool=False,
                 use_shallow_edge_fuse=False,
                 use_rfgm_edge=False,
                 use_frontend_rfgm=False,
                 use_trunk_bridge=False,
                 pretrained_ir_stem=False,
                 stem_base=64,
                 ir_drop_prob=0.15,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 local_model_dir='weight/hgnetv2/',
                 act='relu',
                 dcn_stages=None,
                 dcn_v3=False,
                 dcn_groups=8,
                 use_mixstyle=False,
                 mixstyle_p=0.4,
                 use_midboost=False,
                 use_cssp=False,
                 **kwargs,
                 ):
        super().__init__()
        self.use_midboost = use_midboost
        if use_midboost:
            print("[HGNetv2MultiModal] Integrated MidBoost (Adaptive Thermal Gating & Day Zero-Distortion): ON")
        # ---- RGB main stream: stock HGNetv2 (own freezing + pretrained load) --
        self.rgb_net = HGNetv2(name,
                               use_lab=use_lab,
                               return_idx=return_idx,
                               freeze_stem_only=freeze_stem_only,
                               freeze_at=freeze_at,
                               freeze_norm=freeze_norm,
                               pretrained=pretrained,
                               local_model_dir=local_model_dir,
                               act=act,
                               dcn_stages=dcn_stages,
                               dcn_v3=dcn_v3,
                               dcn_groups=dcn_groups)
        self.return_idx = list(return_idx)
        self._out_channels = [self.rgb_net._out_channels[i] for i in self.return_idx]
        self._out_strides = [self.rgb_net._out_strides[i] for i in self.return_idx]

        # ---- IR auxiliary stream -------------------------------------------
        self.ir_in_chans = ir_in_chans
        self.ir_drop_prob = ir_drop_prob
        self.inject_stages = sorted(int(i) for i in inject_stages)
        _strides = self.rgb_net._out_strides
        _chans = self.rgb_net._out_channels
        for i in self.inject_stages:
            assert 0 <= i < len(_strides), f'inject stage {i} out of range'
        self.ir_stem = ModalityStem(in_chans=ir_in_chans, base=stem_base)
        self.ir_proj = nn.ModuleList([
            nn.Conv2d(stem_base * 2, _chans[i], 1, bias=False)
            for i in self.inject_stages])
        
        self.use_dcda = use_dcda
        if use_dcda:
            self.ir_align = nn.ModuleList([
                DepthConditionedDeformableAlignment(_chans[i])
                for i in self.inject_stages])
        else:
            self.ir_align = None

        self.use_dog = use_dog
        if use_dog:
            self.occlusion_gate = nn.ModuleList([
                DepthOcclusionGating(_chans[i])
                for i in self.inject_stages])
        else:
            self.occlusion_gate = None

        self.use_3d_lift = use_3d_lift
        if use_3d_lift:
            self.coord_lift = nn.ModuleList([
                MetricCoordinateLift3D(_chans[i])
                for i in self.inject_stages])
        else:
            self.coord_lift = None

        self.use_bilateral_pool = use_bilateral_pool
        if use_bilateral_pool:
            from .bilateral_pooling import DisparityBilateralPooling2d
            self.bilateral_pool = nn.ModuleList([
                DisparityBilateralPooling2d(_chans[i], stride=1, sigma_disp=0.05)
                for i in self.inject_stages])
        else:
            self.bilateral_pool = None

        if fusion == 'cssa':
            _fuse_cls = CSSA
        elif fusion == 'cmdrm':
            _fuse_cls = CmDRMFusion
        else:
            _fuse_cls = BiDirFusion
        self.ir_fuse = nn.ModuleList([
            _fuse_cls(_chans[i]) for i in self.inject_stages])
        # zero-init residual: exact identity at step 0
        self.ir_res = nn.ModuleList([
            nn.Conv2d(_chans[i], _chans[i], 1) for i in self.inject_stages])
        for r in self.ir_res:
            nn.init.zeros_(r.weight)
            nn.init.zeros_(r.bias)

        self.use_frontend_rfgm = use_frontend_rfgm
        if use_frontend_rfgm:
            from .dynamic_multimodal_modules import FrontendRFGM
            self.frontend_rfgm = FrontendRFGM(in_chans=3, hidden_chans=16)
        else:
            self.frontend_rfgm = None

        self.use_shallow_edge_fuse = use_shallow_edge_fuse
        if use_shallow_edge_fuse:
            s0_ch = self.rgb_net._out_channels[0] # 128 channels
            s1_ch = self.rgb_net._out_channels[1] # 512 channels
            self.shallow_edge_fuse = HighFrequencyEdgeShallowFusion(cnn_dim=s0_ch, p3_dim=s1_ch, hidden_dim=32, use_rfgm=use_rfgm_edge)
        else:
            self.shallow_edge_fuse = None

        self.use_trunk_bridge = use_trunk_bridge
        if use_trunk_bridge:
            s0_ch = self.rgb_net._out_channels[0] # 128 channels
            s1_ch = self.rgb_net._out_channels[1] # 512 channels
            self.s0_norm = nn.BatchNorm2d(s0_ch)
            self.s0_to_p3_trunk = nn.Conv2d(s0_ch, s1_ch, kernel_size=3, stride=2, padding=1, bias=False)
            nn.init.zeros_(self.s0_to_p3_trunk.weight)
            print("[HGNetv2MultiModal] Direct Trunk Residual Bridge (MSRB Stage 0 -> Stage 1 P3): ON (zero-init)")
        else:
            self.s0_norm = None
            self.s0_to_p3_trunk = None

        if pretrained_ir_stem:
            transfer_pretrained_stem_weights(self.rgb_net.stem, self.ir_stem, in_chans=ir_in_chans)

        self.use_cssp = use_cssp
        if use_cssp:
            from .cssp_module import JointCSSPABCShallowFusion
            s0_ch = self.rgb_net._out_channels[0]
            self.cssp_shallow = JointCSSPABCShallowFusion(rgb_dim=s0_ch, ir_dim=stem_base * 2, out_dim=s0_ch, use_abc=True)
            print("[HGNetv2MultiModal] Cascaded Spectral-Spatial Perception (CSSP) + ABC: ON (zero-init)")
        else:
            self.cssp_shallow = None

        self.use_mixstyle = use_mixstyle
        if use_mixstyle:
            from engine.deim.sovereign_v13_modules import MixStyleCrossDomain
            self.mixstyle = MixStyleCrossDomain(p=mixstyle_p, alpha=0.1)
            print(f"[HGNetv2MultiModal] V1-185 MixStyle Cross-Domain: ON (p={mixstyle_p}, alpha=0.1)")
        else:
            self.mixstyle = None

        n_align = sum(p.numel() for p in self.ir_align.parameters()) if self.ir_align else 0
        n_dog = sum(p.numel() for p in self.occlusion_gate.parameters()) if self.occlusion_gate else 0
        n_lift = sum(p.numel() for p in self.coord_lift.parameters()) if self.coord_lift else 0
        n_s0 = sum(p.numel() for p in self.shallow_edge_fuse.parameters()) if self.shallow_edge_fuse else 0
        n_aux = sum(p.numel() for p in self.ir_stem.parameters()) + \
            sum(p.numel() for p in self.ir_proj.parameters()) + \
            sum(p.numel() for p in self.ir_fuse.parameters()) + \
            sum(p.numel() for p in self.ir_res.parameters()) + n_align + n_dog + n_lift + n_s0
        n_rgb_train = sum(p.numel() for p in self.rgb_net.parameters() if p.requires_grad)
        print(f'[HGNetv2MultiModal] IR aux params: {n_aux / 1e6:.2f}M '
              f'(stem {stem_base}{" [Pretrained]" if pretrained_ir_stem else ""} -> proj -> '
              f'{f"S0_Edge(160x160)+P3_Prompt -> " if use_shallow_edge_fuse else ""}'
              f'{f"DCDA(align) -> " if use_dcda else ""}{f"DOG(gate) -> " if use_dog else ""}{f"3DLift -> " if use_3d_lift else ""}{fusion} inject @stages '
              f'{self.inject_stages}, strides {[_strides[i] for i in self.inject_stages]}, '
              f'zero-init residual; IR modality-drop p={ir_drop_prob}); '
              f'RGB trainable {n_rgb_train / 1e6:.1f}M')

    def forward(self, x):
        x = x.contiguous()
        rgb = x[:, :3].contiguous()
        ir = x[:, 3:3 + self.ir_in_chans].contiguous() if x.shape[1] >= 3 + self.ir_in_chans else None
        depth = x[:, 3 + self.ir_in_chans: 4 + self.ir_in_chans].contiguous() if x.shape[1] >= 4 + self.ir_in_chans else None
        if depth is not None:
            depth = torch.clamp(depth * 0.5 + 0.5, 0.0, 1.0).contiguous()

        g_aux = 1.0
        if self.use_midboost and not self.training:
            _im = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            _is = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            rgb_raw = torch.clamp(rgb * _is + _im, 0.0, 1.0)
            b = rgb_raw.mean(dim=(1, 2, 3), keepdim=True)
            # Sovereign V24 Perfect MidBoost (Full-Spectrum Continuous Gating):
            # Day (b >= 0.35): g_aux = 1.00 (Zero distortion, preserving 0.8924 daylight peak)
            # Twilight to Deep Night (b < 0.35): smooth continuous climb up to 1.42x
            g_aux = 1.0 + 0.42 * torch.clamp((0.35 - b) / 0.30, min=0.0, max=1.0)
            # Deep night thermal takeover (b < 0.022)
            act = 0.60 * torch.clamp((0.022 - b) / 0.007, min=0.0, max=1.0)
            if act.max() > 0.0 and ir is not None:
                ir_raw = torch.clamp(ir[:, :1] * 0.5 + 0.5, 0.0, 1.0)
                rgb_fused = (1.0 - act) * rgb_raw + act * ir_raw
                rgb = (rgb_fused - _im) / _is

        if self.use_frontend_rfgm and self.frontend_rfgm is not None:
            rgb = self.frontend_rfgm(rgb)

        # run the RGB trunk stage-by-stage (mirrors HGNetv2.forward incl. dcn_res)
        y = self.rgb_net.stem(rgb)
        outs = []
        if ir is not None:
            c2, c3 = self.ir_stem(ir)                 # ~1/8, ~1/16 at 2*stem_base
            _stride_to_feat = {8: c2, 16: c3}
            if _zero_stream(self.training, self.ir_drop_prob):
                c2 = torch.zeros_like(c2)
                c3 = torch.zeros_like(c3)
                _stride_to_feat = {8: c2, 16: c3}

        p3_prompt = None
        s0_trunk_p3 = None
        p3_out = None
        for idx, stage in enumerate(self.rgb_net.stages):
            y = stage(y)
            # Stage 0 (stride 4, 160x160): Shallow high-frequency edge interaction & Trunk Bridge
            if idx == 0:
                if self.use_trunk_bridge and self.s0_to_p3_trunk is not None:
                    s0_trunk_p3 = self.s0_to_p3_trunk(self.s0_norm(y))
                if self.use_shallow_edge_fuse and self.shallow_edge_fuse is not None and ir is not None and depth is not None:
                    y, p3_prompt = self.shallow_edge_fuse(y, rgb, ir, depth)
                    y = F.relu(y)  # ensure non-negative activation guarantee
                if self.use_cssp and self.cssp_shallow is not None and ir is not None:
                    y, cssp_mask = self.cssp_shallow(y, c2)

            # Stage 1 (stride 8, 80x80): Prepare high-resolution geometric edge prompt for Neck P3
            # NOTE: Do NOT inject p3_prompt into trunk y! Trunk y continues uncorrupted into Stage 2/3.
            if idx == 1:
                p3_out = y
                if p3_prompt is not None:
                    p3_out = F.relu(p3_out + p3_prompt)
                if s0_trunk_p3 is not None:
                    p3_out = p3_out + s0_trunk_p3
                if self.use_mixstyle and self.mixstyle is not None and self.training:
                    y = self.mixstyle(y)

            if str(idx) in self.rgb_net.dcn_res:
                y = self.rgb_net.dcn_res[str(idx)](y)
            if ir is not None and idx in self.inject_stages:
                k = self.inject_stages.index(idx)
                f = _stride_to_feat.get(self.rgb_net._out_strides[idx])
                if f is None:
                    f = F.interpolate(c3, size=y.shape[-2:], mode='bilinear',
                                      align_corners=False)
                elif f.shape[-2:] != y.shape[-2:]:
                    f = F.interpolate(f, size=y.shape[-2:], mode='bilinear',
                                      align_corners=False)
                f = self.ir_proj[k](f)

                # Depth-Conditioned Deformable Alignment (DCDA)
                if self.use_dcda and self.ir_align is not None and depth is not None:
                    d_sub = F.interpolate(depth, size=y.shape[-2:], mode='nearest')
                    valid_mask = (d_sub > 0.015) & (d_sub < 1.0)
                    inv_d = torch.where(valid_mask, 1.0 / (d_sub * 20.0 + 0.1), torch.zeros_like(d_sub))
                    f = self.ir_align[k](y, f, inv_d)

                # Depth Discontinuity Occlusion Gating (DOG)
                if self.use_dog and self.occlusion_gate is not None and depth is not None:
                    f = self.occlusion_gate[k](f, depth)

                # Disparity-Bilateral Anti-Aliased Pooling
                if self.use_bilateral_pool and self.bilateral_pool is not None and depth is not None:
                    d_sub = F.interpolate(depth, size=y.shape[-2:], mode='nearest')
                    f = self.bilateral_pool[k](f, d_sub)

                # 3D Metric Coordinate Lift (3D-Lift)
                if self.use_3d_lift and self.coord_lift is not None and depth is not None:
                    pos_3d = self.coord_lift[k](depth, y.shape).contiguous()
                    y_in = (y + pos_3d).contiguous()
                    f_in = (f + pos_3d).contiguous()
                else:
                    y_in = y
                    f_in = f

                residual = self.ir_res[k](self.ir_fuse[k](y_in, f_in)).contiguous()
                y = y + (g_aux * residual if (self.use_midboost and not self.training) else residual)
            if idx in self.return_idx:
                if idx == 1 and p3_out is not None:
                    outs.append(p3_out)
                else:
                    outs.append(y)
        return outs

    def get_align_loss(self):
        """Cross-Modal High-Frequency Alignment Loss (路径 B: 物理视差觉醒)."""
        if not self.use_dcda or self.ir_align is None:
            return torch.tensor(0.0)

        total_loss = None
        n_layers = 0
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)

        for align_mod in self.ir_align:
            if align_mod.cache_z_rgb is None or align_mod.cache_warped_z is None:
                continue
            z_rgb = align_mod.cache_z_rgb
            warped_z = align_mod.cache_warped_z
            off = align_mod.cache_off
            inv_d = align_mod.cache_inv_d

            dev = z_rgb.device
            dtype = z_rgb.dtype
            lap_k = lap.to(device=dev, dtype=dtype)

            # High-frequency edge cosine similarity directly in bottleneck space
            e_rgb = F.conv2d(z_rgb.mean(dim=1, keepdim=True), lap_k, padding=1).abs()
            e_ir = F.conv2d(warped_z.mean(dim=1, keepdim=True), lap_k, padding=1).abs()

            norm_rgb = torch.norm(e_rgb.flatten(1), p=2, dim=1, keepdim=True) + 1e-3
            norm_ir = torch.norm(e_ir.flatten(1), p=2, dim=1, keepdim=True) + 1e-3
            sim = ((e_rgb.flatten(1) / norm_rgb) * (e_ir.flatten(1) / norm_ir)).sum(dim=1).mean()
            l_edge = 1.0 - sim

            # Far background offset suppression (when Z > 20m, inv_d < 0.05, offset -> 0)
            l_far = (off.abs() * (inv_d < 0.05).float()).mean()

            # Spatial smoothness of offset field (Total Variation)
            l_tv = (off[:, :, 1:, :] - off[:, :, :-1, :]).abs().mean() + (off[:, :, :, 1:] - off[:, :, :, :-1]).abs().mean()

            loss_stage = l_edge + 0.1 * l_far + 0.05 * l_tv
            total_loss = loss_stage if total_loss is None else total_loss + loss_stage
            n_layers += 1

        if total_loss is None or n_layers == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return total_loss / n_layers

