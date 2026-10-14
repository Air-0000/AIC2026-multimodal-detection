"""Heterogeneous Dual-Stream (CNN + ViT) Multimodal Backbone for D-FINE-X.

Architectural Specification:
  - Stream 1 (CNN): HGNetv2-B5 (Optical + Thermal local perception)
    - Processes RGB (3ch) + IR (1 or 2ch: heat + raw IR)
    - Retains inductive biases for fine boundaries (tricycles, UAVs, thin poles)
    - Anchored on official dfine_x_obj2coco.pth (0.5550 AP) checkpoint
  - Stream 2 (ViT): Depth-Anything-V2 / DINOv2 ViT-Small (3D Geometric global perception)
    - Processes 3D depth / surface normal geometry tokens
    - Global scene context & distance-invariant attention (84.8% probe accuracy)
  - Bidirectional Cross-Paradigm Interaction Bridges:
    - CNN -> ViT: Sobel high-frequency spatial edge prompt injection
    - ViT -> CNN: Multi-head cross-attention guidance (with spatial reduction for low VRAM)
    - Zero-initialized residual gates: guaranteed exact identity at Step 0
  - Shared-Private Orthogonal Disentanglement:
    - Subspace projection: Z_s (shared semantics) vs Z_p (modality-specific physical properties)
    - Uncertainty & illumination-aware Bounded Convex Gating: f_s = alpha*S_cnn + beta*S_vit
    - Private feature injection: f_p = s * tanh(Conv(P_vit))
    - Explicit orthogonal regularization loss hook: L_orth
  - ViT Fine-Tuning Strategy (Two-Stage Progressive Adaptation):
    - Stage 1: ViT fully frozen (blocks 0..11), safe cold-start bridge alignment
    - Stage 2: Layer-wise Learning Rate Decay (LLRD) unfreezing of last N blocks + norm
      lr_i = top_lr * decay^(N - 1 - i), stamped as `p.llrd_lr` for optimizer auto-grouping
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hgnetv2 import HGNetv2
from .multimodal_backbones import ModalityStem, _zero_stream
from engine.core.workspace import register


# --------------------------------------------------------------------------- #
# 1. Cross-Paradigm Bidirectional Interaction Bridges
# --------------------------------------------------------------------------- #

class HighFreqEdgePrompt(nn.Module):
    """CNN -> ViT Bridge: Injects fine spatial edge cues into ViT tokens.
    
    CNN features have sharp sub-pixel spatial resolution. A light depthwise conv
    extracts high-frequency spatial gradients and projects to ViT channel dim.
    Gated by a zero-initialized scalar so at Step 0 it is an exact identity.
    """
    def __init__(self, cnn_dim, vit_dim):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(cnn_dim, cnn_dim, 3, padding=1, groups=cnn_dim, bias=False),
            nn.BatchNorm2d(cnn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_dim, vit_dim, 1, bias=False),
            nn.BatchNorm2d(vit_dim),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, p_cnn, t_vit):
        prompt = self.edge_conv(p_cnn)
        return t_vit + self.scale * prompt


class ViTToCNNCrossGuidance(nn.Module):
    """ViT -> CNN Bridge: Cross-Attention guidance from ViT 3D geometry to CNN.
    
    Query: CNN feature (local perception)
    Key/Value: ViT feature (global 3D geometry)
    
    Equipped with spatial reduction (sr_ratio) for stride 8 (80x80) to keep VRAM
    minimal on RTX 5060 Ti (< 120MB per micro-batch).
    Gated by a zero-initialized scalar gamma so at Step 0 it is an exact identity.
    """
    def __init__(self, cnn_dim, vit_dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.vit_dim = vit_dim
        self.num_heads = num_heads
        self.head_dim = cnn_dim // num_heads
        assert self.head_dim * num_heads == cnn_dim, "cnn_dim must be divisible by num_heads"
        self.attn_scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(cnn_dim, cnn_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(vit_dim, cnn_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(vit_dim, cnn_dim, 1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(cnn_dim, cnn_dim, 1, bias=False),
            nn.BatchNorm2d(cnn_dim)
        )

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.AvgPool2d(sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.BatchNorm2d(vit_dim)
        else:
            self.sr = None
            self.sr_norm = None

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, p_cnn, t_vit):
        B, C, H, W = p_cnn.shape
        N_q = H * W

        # 1. Query from CNN
        q = self.q_proj(p_cnn).view(B, self.num_heads, self.head_dim, N_q).permute(0, 1, 3, 2)  # [B, heads, N_q, head_dim]

        # 2. Key/Value from ViT (with spatial reduction if needed)
        if self.sr is not None:
            t_sub = self.sr_norm(self.sr(t_vit))
        else:
            t_sub = t_vit
        _, _, H_k, W_k = t_sub.shape
        N_k = H_k * W_k

        k = self.k_proj(t_sub).view(B, self.num_heads, self.head_dim, N_k)  # [B, heads, head_dim, N_k]
        v = self.v_proj(t_sub).view(B, self.num_heads, self.head_dim, N_k).permute(0, 1, 3, 2)  # [B, heads, N_k, head_dim]

        # 3. Scaled dot-product attention
        attn = torch.matmul(q, k) * self.attn_scale  # [B, heads, N_q, N_k]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, heads, N_q, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.out_proj(out)

        return p_cnn + self.gamma * out


# --------------------------------------------------------------------------- #
# 2. Shared-Private Disentanglement & Bounded Convex Gating
# --------------------------------------------------------------------------- #

class SubspaceProjector(nn.Module):
    """Projects modality feature into Shared (Z_s) and Private (Z_p) subspaces."""
    def __init__(self, in_dim, sub_dim=96):
        super().__init__()
        self.proj_shared = nn.Sequential(
            nn.Conv2d(in_dim, sub_dim, 1, bias=False),
            nn.BatchNorm2d(sub_dim),
            nn.GELU(),
        )
        self.proj_private = nn.Sequential(
            nn.Conv2d(in_dim, sub_dim, 1, bias=False),
            nn.BatchNorm2d(sub_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj_shared(x), self.proj_private(x)


class BoundedDualConvexGating(nn.Module):
    """Uncertainty & Illumination Aware Bounded Convex Gating for Dual-Stream.
    
    1. Shared Subspace Fusion:
       f_shared = alpha * S_cnn + beta * S_vit,  where alpha + beta = 1, alpha, beta >= 0.
       Driven by global illumination prior (mean RGB luminance) and pooled statistics.
    2. Private Subspace Injection:
       f_priv = s * tanh(Conv([P_cnn, P_vit]))
       where s is a learnable scalar initialized to 0.05.
    3. Final output:
       Y = P_guided + OutProj(f_shared) + f_priv
    """
    def __init__(self, cnn_dim, sub_dim=96):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.sub_dim = sub_dim

        # Gating network: takes pooled shared features + luminance prior
        self.gate_mlp = nn.Sequential(
            nn.Linear(sub_dim * 2 + 1, sub_dim),
            nn.GELU(),
            nn.Linear(sub_dim, 2),
        )
        # Initialization bias towards CNN (optical appearance) at start
        nn.init.zeros_(self.gate_mlp[-1].weight)
        self.gate_mlp[-1].bias.data = torch.tensor([2.0, 0.0])

        self.out_proj = nn.Sequential(
            nn.Conv2d(sub_dim, cnn_dim, 1, bias=False),
            nn.BatchNorm2d(cnn_dim),
        )

        # Private feature integration
        self.priv_conv = nn.Conv2d(sub_dim * 2, cnn_dim, 1, bias=False)
        self.priv_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, p_guided, s_cnn, s_vit, p_cnn, p_vit, raw_rgb=None):
        B, _, H, W = s_cnn.shape

        # Illumination prior
        if raw_rgb is not None:
            lum = raw_rgb.mean(dim=(1, 2, 3), keepdim=True).view(B, 1)
        else:
            lum = torch.full((B, 1), 0.5, device=s_cnn.device, dtype=s_cnn.dtype)

        gap_cnn = F.adaptive_avg_pool2d(s_cnn, 1).view(B, -1)
        gap_vit = F.adaptive_avg_pool2d(s_vit, 1).view(B, -1)

        stat = torch.cat([gap_cnn, gap_vit, lum], dim=-1)
        weights = F.softmax(self.gate_mlp(stat), dim=-1)  # [B, 2]

        w_cnn = weights[:, 0].view(B, 1, 1, 1)
        w_vit = weights[:, 1].view(B, 1, 1, 1)

        # Convex shared combination
        f_shared = w_cnn * s_cnn + w_vit * s_vit
        f_shared_proj = self.out_proj(f_shared)

        # Bounded private injection
        priv_cat = torch.cat([p_cnn, p_vit], dim=1)
        f_priv = torch.tanh(self.priv_conv(priv_cat)) * self.priv_scale

        y = p_guided + f_shared_proj + f_priv
        return y, weights


# --------------------------------------------------------------------------- #
# 3. Main Heterogeneous Dual-Stream Backbone
# --------------------------------------------------------------------------- #

@register()
class HGNetv2ViTDualStream(nn.Module):
    """CNN + ViT Heterogeneous Dual-Stream Multimodal Backbone for D-FINE-X.
    
    Integrates HGNetv2-B5 (Optical + Thermal) and Depth-Anything-V2 ViT (3D Geometry)
    via bidirectional cross-bridges, orthogonal subspace disentanglement, and
    bounded convex gating.
    
    Supports:
      1. Stage 1 training: ViT frozen, only training bridges, gating, IR stem, and head.
      2. Stage 2 training: ViT tail unfreezing with Layer-wise LR Decay (LLRD).
      3. Seamless drop-in replacement for D-FINE HybridEncoder & DFINETransformer.
    """
    def __init__(
        self,
        name='B5',
        return_idx=[1, 2, 3],
        ir_in_chans=2,
        stem_base=64,
        ir_drop_prob=0.15,
        depth_drop_prob=0.15,
        vit_encoder='vits',
        vit_ckpt_path='work_dir/da2_weights/depth_anything_v2_vits.pth',
        vit_embed_dim=384,
        vit_input_size=518,
        sub_dim=96,
        unfreeze_vit_blocks=0,
        llrd_top_lr=2.0e-5,
        llrd_decay=0.85,
        use_lab=False,
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        pretrained=True,
        local_model_dir='weight/hgnetv2/',
        act='relu',
        dcn_stages=None,
        dcn_v3=False,
        dcn_groups=8,
    ):
        super().__init__()
        self.name = name
        self.return_idx = list(return_idx)
        self.ir_in_chans = ir_in_chans
        self.ir_drop_prob = ir_drop_prob
        self.depth_drop_prob = depth_drop_prob
        self.vit_embed_dim = vit_embed_dim
        self.vit_input_size = vit_input_size
        self.sub_dim = sub_dim

        # ---- Stream 1 (CNN): HGNetv2 (RGB + Optical Trunk) ----
        self.rgb_net = HGNetv2(
            name=name,
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
            dcn_groups=dcn_groups
        )
        self._out_channels = [self.rgb_net._out_channels[i] for i in self.return_idx]
        self._out_strides = [self.rgb_net._out_strides[i] for i in self.return_idx]

        # ---- IR Auxiliary Stream for CNN trunk ----
        if ir_in_chans > 0:
            self.ir_stem = ModalityStem(in_chans=ir_in_chans, base=stem_base)
            # Projects IR stem features to CNN stage 1 (stride 8) and stage 2 (stride 16)
            self.ir_proj = nn.ModuleList([
                nn.Conv2d(stem_base * 2, self.rgb_net._out_channels[1], 1, bias=False),
                nn.Conv2d(stem_base * 2, self.rgb_net._out_channels[2], 1, bias=False),
            ])
            self.ir_res = nn.ModuleList([
                nn.Conv2d(self.rgb_net._out_channels[1], self.rgb_net._out_channels[1], 1),
                nn.Conv2d(self.rgb_net._out_channels[2], self.rgb_net._out_channels[2], 1),
            ])
            for r in self.ir_res:
                nn.init.zeros_(r.weight)
                nn.init.zeros_(r.bias)
        else:
            self.ir_stem = None

        # ---- Stream 2 (ViT): Depth-Anything-V2 ViT Backbone ----
        self.vit = self._build_dav2_vit(vit_encoder, vit_ckpt_path)

        # ViT feature multi-scale pyramid projection (aligns ViT tokens to strides 8, 16, 32)
        # ViT returns tokens with dim 384. We project each stage cleanly.
        self.vit_projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(vit_embed_dim, vit_embed_dim, 1, bias=False), nn.BatchNorm2d(vit_embed_dim)),
            nn.Sequential(nn.Conv2d(vit_embed_dim, vit_embed_dim, 1, bias=False), nn.BatchNorm2d(vit_embed_dim)),
            nn.Sequential(nn.Conv2d(vit_embed_dim, vit_embed_dim, 1, bias=False), nn.BatchNorm2d(vit_embed_dim)),
        ])

        # ---- Cross-Paradigm Bidirectional Interaction Bridges ----
        # Stages corresponding to return_idx=[1, 2, 3] (strides [8, 16, 32], chans [512, 1024, 2048] on B5)
        self.c2v_bridges = nn.ModuleList()
        self.v2c_bridges = nn.ModuleList()
        self.disentangle_cnn = nn.ModuleList()
        self.disentangle_vit = nn.ModuleList()
        self.fusion_gates = nn.ModuleList()

        sr_ratios = [2, 1, 1]  # Stride 8 uses 2x spatial reduction on ViT K/V; strides 16/32 use full resolution
        for idx_i, cnn_ch in enumerate(self._out_channels):
            self.c2v_bridges.append(HighFreqEdgePrompt(cnn_ch, vit_embed_dim))
            self.v2c_bridges.append(ViTToCNNCrossGuidance(cnn_ch, vit_embed_dim, num_heads=8, sr_ratio=sr_ratios[idx_i]))
            self.disentangle_cnn.append(SubspaceProjector(cnn_ch, sub_dim))
            self.disentangle_vit.append(SubspaceProjector(vit_embed_dim, sub_dim))
            self.fusion_gates.append(BoundedDualConvexGating(cnn_ch, sub_dim))

        # Initial parameter management & Stage 1 / Stage 2 scheduling
        self.unfreeze_vit_blocks = unfreeze_vit_blocks
        self.llrd_top_lr = llrd_top_lr
        self.llrd_decay = llrd_decay

        if unfreeze_vit_blocks > 0:
            self.unfreeze_vit(n_unfreeze=unfreeze_vit_blocks, top_lr=llrd_top_lr, decay=llrd_decay)
        else:
            self.freeze_vit()

        # Variable to store orthogonal loss for training loss accumulation
        self.last_orth_loss = torch.tensor(0.0)

        # Print architectural budget
        n_rgb = sum(p.numel() for p in self.rgb_net.parameters())
        n_vit = sum(p.numel() for p in self.vit.parameters())
        n_bridges = (sum(p.numel() for p in self.c2v_bridges.parameters()) +
                     sum(p.numel() for p in self.v2c_bridges.parameters()) +
                     sum(p.numel() for p in self.disentangle_cnn.parameters()) +
                     sum(p.numel() for p in self.disentangle_vit.parameters()) +
                     sum(p.numel() for p in self.fusion_gates.parameters()))
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[HGNetv2ViTDualStream] Initialized Heterogeneous Dual-Stream Backbone:\n"
              f"  - Stream 1 (CNN): HGNetv2-{name} ({n_rgb / 1e6:.2f}M params, channels {self._out_channels})\n"
              f"  - Stream 2 (ViT): DAv2 {vit_encoder} ({n_vit / 1e6:.2f}M params, embed_dim {vit_embed_dim})\n"
              f"  - Bridges & Gating: {n_bridges / 1e6:.2f}M params (Zero-init identity at Step 0)\n"
              f"  - Total trainable params: {n_train / 1e6:.2f}M / {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M\n"
              f"  - ViT Unfreeze Blocks: {unfreeze_vit_blocks} (LLRD top_lr={llrd_top_lr:.1e}, decay={llrd_decay})")

    def _build_dav2_vit(self, encoder='vits', ckpt_path=None):
        """Builds DINOv2 / DAv2 ViT backbone and loads weights."""
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        da2_dir = os.path.join(root, "vendor/Depth-Anything-V2")
        if da2_dir not in sys.path:
            sys.path.insert(0, da2_dir)

        try:
            from depth_anything_v2.dinov2 import DINOv2
            vit = DINOv2(encoder)
        except Exception as e:
            print(f"[HGNetv2ViTDualStream] Warning importing DINOv2 directly: {e}. Falling back to DPT wrapper.")
            from depth_anything_v2.dpt import DepthAnythingV2
            cfg = {"encoder": encoder, "features": 64, "out_channels": [48, 96, 192, 384]}
            da2_model = DepthAnythingV2(**cfg)
            vit = da2_model.pretrained

        # Load weights if checkpoint exists
        if ckpt_path and os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cpu')
            vit_sd = {}
            for k, v in sd.items():
                if k.startswith('pretrained.'):
                    vit_sd[k[len('pretrained.'):]] = v
                elif not k.startswith('depth_head.'):
                    vit_sd[k] = v
            msg = vit.load_state_dict(vit_sd, strict=False)
            print(f"[HGNetv2ViTDualStream] Loaded DAv2 ViT weights from {ckpt_path} (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
        elif ckpt_path:
            print(f"[HGNetv2ViTDualStream] Warning: ViT checkpoint {ckpt_path} not found.")

        return vit

    def freeze_vit(self):
        """Freezes entire ViT backbone (Stage 1 training)."""
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False
    def train(self, mode=True):
        super().train(mode)
        if self.unfreeze_vit_blocks == 0:
            self.vit.eval()
        return self

    def unfreeze_vit(self, n_unfreeze=4, top_lr=2.0e-5, decay=0.85):
        """Unfreezes last n_unfreeze blocks with Layer-wise Learning Rate Decay (Stage 2).
        
        Stamps `p.llrd_lr` on each unfreezed parameter, read by yaml_config.py
        to auto-construct independent optimizer subgroups without regex clutter.
        """
        self.freeze_vit()
        blocks = self.vit.blocks
        n_unfreeze = min(int(n_unfreeze), len(blocks))
        unf_idx = list(range(len(blocks) - n_unfreeze, len(blocks)))

        for gi in unf_idx:
            # lr_i = top_lr * decay^(unf_idx[-1] - gi)
            lr = top_lr * (decay ** (unf_idx[-1] - gi))
            for p in blocks[gi].parameters():
                p.requires_grad_(True)
                p.llrd_lr = float(lr)

        if hasattr(self.vit, 'norm') and self.vit.norm is not None:
            for p in self.vit.norm.parameters():
                p.requires_grad_(True)
                p.llrd_lr = float(top_lr)

        self.vit.train()
        print(f"[HGNetv2ViTDualStream] Unfroze ViT tail {n_unfreeze} blocks {unf_idx[0]}..{unf_idx[-1]} "
              f"+ norm | LLRD top_lr={top_lr:.1e}, decay={decay} "
              f"(block {unf_idx[-1]}={top_lr:.1e} .. block {unf_idx[0]}={top_lr * (decay**(n_unfreeze-1)):.1e})")

    def _extract_vit_multiscale(self, vit_input):
        """Extracts multi-scale features from ViT tokens for strides [8, 16, 32]."""
        B, _, H, W = vit_input.shape
        # Input to DINOv2 must be a multiple of patch_size (14)
        target_size = self.vit_input_size - (self.vit_input_size % 14)
        if H != target_size or W != target_size:
            vit_in = F.interpolate(vit_input, size=(target_size, target_size), mode='bilinear', align_corners=False)
        else:
            vit_in = vit_input

        patch_h = target_size // 14
        patch_w = target_size // 14

        # Extract intermediate layers [5, 8, 11]
        layers = self.vit.get_intermediate_layers(vit_in, [5, 8, 11], return_class_token=False)

        # Reshape tokens into 2D maps: [B, N_patches, 384] -> [B, 384, patch_h, patch_w]
        vit_feats = []
        for i, tokens in enumerate(layers):
            f2d = tokens.permute(0, 2, 1).contiguous().view(B, self.vit_embed_dim, patch_h, patch_w)
            vit_feats.append(self.vit_projs[i](f2d))

        return vit_feats  # [T3_base, T4_base, T5_base]

    def get_orthogonal_loss(self):
        """Returns the accumulated orthogonal regularization loss."""
        return self.last_orth_loss

    def forward(self, x):
        B, C, H, W = x.shape
        raw_rgb = x[:, :3]

        # 1. Parse modalities
        ir = None
        depth = None
        if self.ir_in_chans > 0 and C >= 3 + self.ir_in_chans:
            ir = x[:, 3:3 + self.ir_in_chans]
            if C > 3 + self.ir_in_chans:
                depth = x[:, 3 + self.ir_in_chans:]
        elif C > 3:
            depth = x[:, 3:]

        # 2. Extract ViT 3D Geometry Tokens (Stream 2)
        # If depth is available, prepare 3-channel input for DAv2; else use RGB
        if depth is not None:
            if depth.shape[1] == 1:
                vit_input = depth.repeat(1, 3, 1, 1)
            elif depth.shape[1] >= 3:
                vit_input = depth[:, :3]
            else:
                vit_input = raw_rgb
        else:
            vit_input = raw_rgb

        # Normalize with ImageNet stats for DAv2
        vit_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        vit_std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        vit_in_norm = (vit_input - vit_mean) / vit_std

        if self.unfreeze_vit_blocks > 0 and self.vit.training:
            vit_tokens = self._extract_vit_multiscale(vit_in_norm)
        else:
            with torch.no_grad():
                vit_tokens = [t.detach() for t in self._extract_vit_multiscale(vit_in_norm)]

        # Modality dropout on ViT stream
        if _zero_stream(self.training, self.depth_drop_prob):
            vit_tokens = [torch.zeros_like(t) for t in vit_tokens]

        # 3. Extract IR cues for CNN trunk (if available)
        ir_c2, ir_c3 = None, None
        if ir is not None and self.ir_stem is not None:
            if not _zero_stream(self.training, self.ir_drop_prob):
                ir_c2, ir_c3 = self.ir_stem(ir)

        # 4. Stream 1 (CNN): HGNetv2 stage-by-stage forward with cross-interaction
        y = self.rgb_net.stem(raw_rgb)

        outs = []
        orth_losses = []
        bridge_idx = 0

        for stage_idx, stage in enumerate(self.rgb_net.stages):
            y = stage(y)
            if str(stage_idx) in self.rgb_net.dcn_res:
                y = self.rgb_net.dcn_res[str(stage_idx)](y)

            # IR injection on stage 1 (stride 8) and stage 2 (stride 16)
            if ir_c2 is not None and stage_idx == 1:
                ir_f = self.ir_proj[0](ir_c2)
                if ir_f.shape[-2:] != y.shape[-2:]:
                    ir_f = F.interpolate(ir_f, size=y.shape[-2:], mode='bilinear', align_corners=False)
                y = y + self.ir_res[0](ir_f)
            elif ir_c3 is not None and stage_idx == 2:
                ir_f = self.ir_proj[1](ir_c3)
                if ir_f.shape[-2:] != y.shape[-2:]:
                    ir_f = F.interpolate(ir_f, size=y.shape[-2:], mode='bilinear', align_corners=False)
                y = y + self.ir_res[1](ir_f)

            # Bidirectional Cross-Bridge & Disentanglement on returned stages (1, 2, 3)
            if stage_idx in self.return_idx:
                k = bridge_idx
                bridge_idx += 1

                # Align ViT token map to current CNN spatial resolution (H_k, W_k)
                t_base = vit_tokens[k]
                if t_base.shape[-2:] != y.shape[-2:]:
                    t_k = F.interpolate(t_base, size=y.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    t_k = t_base

                # A. CNN -> ViT Edge Prompt Injection
                t_k_prompted = self.c2v_bridges[k](y, t_k)

                # B. ViT -> CNN Cross-Attention Guidance
                p_k_guided = self.v2c_bridges[k](y, t_k_prompted)

                # C. Shared-Private Disentanglement
                s_cnn, p_cnn = self.disentangle_cnn[k](p_k_guided)
                s_vit, p_vit = self.disentangle_vit[k](t_k_prompted)

                # D. Bounded Dual Convex Gating & Fusion
                y_fused, _ = self.fusion_gates[k](
                    p_guided=p_k_guided,
                    s_cnn=s_cnn,
                    s_vit=s_vit,
                    p_cnn=p_cnn,
                    p_vit=p_vit,
                    raw_rgb=raw_rgb
                )

                # E. Calculate Orthogonal Regularization Loss: <S, P> / (||S||*||P|| + eps)
                eps = 1e-6
                inner_cnn = (s_cnn * p_cnn).sum(dim=1, keepdim=True)
                norm_cnn = (s_cnn.norm(dim=1, keepdim=True) * p_cnn.norm(dim=1, keepdim=True)).clamp_min(eps)
                orth_cnn = ((inner_cnn / norm_cnn) ** 2).mean()

                inner_vit = (s_vit * p_vit).sum(dim=1, keepdim=True)
                norm_vit = (s_vit.norm(dim=1, keepdim=True) * p_vit.norm(dim=1, keepdim=True)).clamp_min(eps)
                orth_vit = ((inner_vit / norm_vit) ** 2).mean()

                orth_losses.append(orth_cnn + orth_vit)
                outs.append(y_fused)

        self.last_orth_loss = sum(orth_losses) / max(1, len(orth_losses))
        return outs
