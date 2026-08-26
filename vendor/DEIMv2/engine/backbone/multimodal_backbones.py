"""
Multimodal backbones for DEIMv2 (AIC) -- three staged experiments.

These implement the three reference-paper methods, staged so each can be tried
independently (per the user's "try one at a time" plan):

  Step 1 -- [1] CSD (Channel Switching + Spatial Attention, CVPR 2023)
            DINOv3STAsCSD: RGB backbone (pretrained) + light IR stem, fused by a
            learned spatial attention gate M(x,y) deciding per-location which
            modality to trust. Channel switching is applied at train time
            (randomly drop the IR branch with probability p).

  Step 2 -- [4] Bi-directional Fusion + Deformation-Sensitive Loss (Info Fusion 2025)
            DINOv3STAsBiDir: same two-stream layout but fuses via a bi-directional
            cross-attention (RGB->IR and IR->RGB) instead of a single gate, which
            is robust to the per-scene geometric misalignment (no hard registration).

  Step 3 -- [5] Collaborating Vision / Depth / Thermal (NeurIPS 2025 D&B)
            DINOv3STAsTriModal: RGB+heat share one DINOv3 stem (in_chans=4), depth
            has its own stem, and the three feature streams interact through a
            cross-modal attention block at each scale. Depth carries geometry,
            IR carries thermal cues, RGB carries appearance -- kept orthogonal at
            the feature level.

All three return (c2, c3, c4) exactly like the original DINOv3STAs so the
HybridEncoder / decoder need no changes.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from ..core import register
from .dinov3_adapter import DINOv3STAs, SpatialPriorModulev2
from .dinov3 import DinoVisionTransformer


# --------------------------------------------------------------------------- #
# Light modality stems (small CNNs producing 3 multi-scale feature maps)
# --------------------------------------------------------------------------- #
class ModalityStem(nn.Module):
    """A tiny CNN that turns a 1-channel modality (IR-heat or depth) into three
    multi-scale feature maps aligned with the DINOv3 output scales (1/8,1/16,1/32
    after the backbone's internal handling). We keep it deliberately small; the
    heavy semantics come from the pretrained RGB DINOv3 stream."""

    def __init__(self, in_chans=1, base=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, base, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.GELU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
        )
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(base * 2, base * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
        )

    def forward(self, x):
        c1 = self.stem(x)      # ~1/4
        c2 = self.conv2(c1)    # ~1/8
        c3 = self.conv3(c2)    # ~1/16
        return c2, c3


# --------------------------------------------------------------------------- #
# Fusion modules
# --------------------------------------------------------------------------- #
class SpatialGate(nn.Module):
    """CSD spatial attention: learn M in [0,1] from concatenated rgb/ir features
    and compute  out = M*rgb + (1-M)*ir."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, rgb, ir):
        m = self.conv(torch.cat([rgb, ir], dim=1))  # [B,1,H,W]
        return m * rgb + (1.0 - m) * ir


class BiDirFusion(nn.Module):
    """Bi-directional cross-attention fusion (RGB<->IR) at the feature level.
    Uses simple scaled dot-product attention over spatial tokens."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.q_rgb = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_ir = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.q_ir = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_rgb = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.proj = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def _attn(self, q_feat, kv_feat, q_proj, kv_proj, dim):
        b, _, h, w = q_feat.shape
        q = q_proj(q_feat).reshape(b, self.heads, dim // self.heads, h * w)
        k, v = kv_proj(kv_feat).chunk(2, dim=1)
        k = k.reshape(b, self.heads, dim // self.heads, h * w)
        v = v.reshape(b, self.heads, dim // self.heads, h * w)
        attn = (q.transpose(2, 3) @ k).softmax(dim=-1)  # [B,H,Wq,Wk]
        out = (attn @ v.transpose(2, 3)).transpose(2, 3)
        return out.reshape(b, dim, h, w)

    def forward(self, rgb, ir):
        f_rgb = self._attn(rgb, ir, self.q_rgb, self.kv_ir, rgb.shape[1])  # rgb queries ir
        f_ir = self._attn(ir, rgb, self.q_ir, self.kv_rgb, ir.shape[1])    # ir queries rgb
        out = self.norm(self.proj(torch.cat([f_rgb, f_ir], dim=1)))
        return out


class CrossModalAttention(nn.Module):
    """Cross-modal attention for three streams (rgb+heat, depth). Used by Step3.
    depth acts as the query source attending to the rgb+heat features."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, query_feat, keyval_feat, dim):
        b, _, h, w = query_feat.shape
        q = self.q(query_feat).reshape(b, self.heads, dim // self.heads, h * w)
        k, v = self.kv(keyval_feat).chunk(2, dim=1)
        k = k.reshape(b, self.heads, dim // self.heads, h * w)
        v = v.reshape(b, self.heads, dim // self.heads, h * w)
        attn = (q.transpose(2, 3) @ k).softmax(dim=-1)
        out = (attn @ v.transpose(2, 3)).transpose(2, 3).reshape(b, dim, h, w)
        return self.norm(self.proj(out))


# --------------------------------------------------------------------------- #
# Shared helper: extract DINOv3 multi-scale features from an RGB (3ch) input
# --------------------------------------------------------------------------- #
def _load_dinov3_sd(weights_path):
    """Load a DEIMv2 DINOv3 checkpoint and strip it down to the `dinov3` stem
    state dict (the checkpoints store everything under `model.backbone.dinov3.*`)."""
    ck = torch.load(weights_path, map_location="cpu")
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    out = {}
    for k, v in sd.items():
        if k.startswith("backbone.dinov3."):
            out[k[len("backbone.dinov3."):]] = v
        elif isinstance(k, str) and not k.startswith(("backbone.", "encoder.", "decoder.")):
            out[k] = v
    return out


def _dinov3_features(dinov3, x, interaction_indexes, patch_size):
    H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
    H_toks, W_toks = x.shape[2] // patch_size, x.shape[3] // patch_size
    bs = x.shape[0]
    all_layers = dinov3.get_intermediate_layers(x, n=interaction_indexes, return_class_token=True)
    if len(all_layers) == 1:
        all_layers = [all_layers[0], all_layers[0], all_layers[0]]
    sem_feats = []
    num_scales = len(all_layers) - 2
    for i, sem_feat in enumerate(all_layers):
        feat, _ = sem_feat
        sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
        rh, rw = int(H_c * 2 ** (num_scales - i)), int(W_c * 2 ** (num_scales - i))
        sem_feat = F.interpolate(sem_feat, size=[rh, rw], mode="bilinear", align_corners=False)
        sem_feats.append(sem_feat)
    return sem_feats  # [c2_feat, c3_feat, c4_feat] before STA concat


# --------------------------------------------------------------------------- #
# Step 1: CSD (spatial attention)
# --------------------------------------------------------------------------- #
@register()
class DINOv3STAsCSD(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        ir_drop_prob=0.5,  # channel-switching: drop IR stream at train time
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob

        # RGB stream: pretrained DINOv3 (in_chans=3, weights preserved)
        self.rgb_dinov3 = DinoVisionTransformer(name=name, in_chans=3)
        if weights_path and os.path.exists(weights_path):
            self.rgb_dinov3.load_state_dict(_load_dinov3_sd(weights_path))
            print(f"[CSD] loaded DINOv3 weights from {weights_path}")
        self.embed_dim = self.rgb_dinov3.embed_dim

        # IR heat stream
        self.ir_stem = ModalityStem(in_chans=1, base=64)
        self.ir_proj = nn.ModuleList([
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
        ])
        self.ir_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])

        # spatial gate fuses rgb & ir at the two finer scales
        self.gate = nn.ModuleList([SpatialGate(self.embed_dim), SpatialGate(self.embed_dim)])

        # detail (STA) for the rgb stream
        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

    def forward(self, x):
        # x: [B,4,H,W] = RGB(3) + heat(1)
        rgb = x[:, :3]
        ir = x[:, 3:4]
        bs = x.shape[0]

        rgb_feats = _dinov3_features(self.rgb_dinov3, rgb, self.interaction_indexes, self.patch_size)
        ir_c2, ir_c3 = self.ir_stem(ir)
        ir_feats = [self.ir_norms[0](self.ir_proj[0](ir_c2)),
                    self.ir_norms[1](self.ir_proj[1](ir_c3))]

        # channel switching (train only)
        if self.training and torch.rand(1).item() < self.ir_drop_prob:
            ir_feats = [torch.zeros_like(f) for f in ir_feats]

        # fuse at the two finer scales (c2, c3); c4 uses rgb only
        fused = []
        fused.append(self.gate[0](rgb_feats[0], ir_feats[0]))
        fused.append(self.gate[1](rgb_feats[1], ir_feats[1]))
        fused.append(rgb_feats[2])

        detail = self.sta(rgb)
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Step 2: Bi-directional fusion (+ deformation-sensitive loss handled in solver)
# --------------------------------------------------------------------------- #
@register()
class DINOv3STAsBiDir(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        ir_drop_prob=0.5,
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob

        self.rgb_dinov3 = DinoVisionTransformer(name=name, in_chans=3)
        if weights_path and os.path.exists(weights_path):
            self.rgb_dinov3.load_state_dict(_load_dinov3_sd(weights_path))
        self.embed_dim = self.rgb_dinov3.embed_dim

        self.ir_stem = ModalityStem(in_chans=1, base=64)
        self.ir_proj = nn.ModuleList([
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
        ])
        self.ir_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])

        self.fuse = nn.ModuleList([BiDirFusion(self.embed_dim) for _ in range(2)])

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

    def forward(self, x):
        rgb = x[:, :3]
        ir = x[:, 3:4]
        rgb_feats = _dinov3_features(self.rgb_dinov3, rgb, self.interaction_indexes, self.patch_size)
        ir_c2, ir_c3 = self.ir_stem(ir)
        ir_feats = [self.ir_norms[0](self.ir_proj[0](ir_c2)),
                    self.ir_norms[1](self.ir_proj[1](ir_c3))]
        if self.training and torch.rand(1).item() < self.ir_drop_prob:
            ir_feats = [torch.zeros_like(f) for f in ir_feats]

        fused = [self.fuse[0](rgb_feats[0], ir_feats[0]),
                 self.fuse[1](rgb_feats[1], ir_feats[1]),
                 rgb_feats[2]]
        detail = self.sta(rgb)
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Step 3: Tri-modal (RGB+heat stem + Depth stem + cross-modal attention)
# --------------------------------------------------------------------------- #
@register()
class DINOv3STAsTriModal(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        ir_drop_prob=0.5,
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob

        # RGB + heat share one DINOv3 stem (in_chans=4, expanded from pretrained)
        self.rgbheat_dinov3 = DinoVisionTransformer(name=name, in_chans=4)
        if weights_path and os.path.exists(weights_path):
            sd = _load_dinov3_sd(weights_path)
            w = sd.get("patch_embed.proj.weight")
            if w is not None and w.shape[1] == 3:
                mean_w = w.mean(dim=1, keepdim=True)
                extra = mean_w.expand(-1, 1, -1, -1)
                sd["patch_embed.proj.weight"] = torch.cat([w, extra], dim=1)
            self.rgbheat_dinov3.load_state_dict(sd)
        self.embed_dim = self.rgbheat_dinov3.embed_dim

        # Depth stream
        self.depth_stem = ModalityStem(in_chans=1, base=64)
        self.depth_proj = nn.ModuleList([
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
        ])
        self.depth_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])

        # cross-modal attention: depth queries rgb+heat features
        self.cma = nn.ModuleList([CrossModalAttention(self.embed_dim) for _ in range(2)])

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

    def forward(self, x):
        # x: [B,5,H,W] = RGB(3) + heat(1) + depth(1)
        rgbheat = x[:, :4]
        depth = x[:, 4:5]
        rh_feats = _dinov3_features(self.rgbheat_dinov3, rgbheat, self.interaction_indexes, self.patch_size)
        d_c2, d_c3 = self.depth_stem(depth)
        d_feats = [self.depth_norms[0](self.depth_proj[0](d_c2)),
                   self.depth_norms[1](self.depth_proj[1](d_c3))]
        if self.training and torch.rand(1).item() < self.ir_drop_prob:
            d_feats = [torch.zeros_like(f) for f in d_feats]

        fused = [self.cma[0](d_feats[0], rh_feats[0], self.embed_dim),
                 self.cma[1](d_feats[1], rh_feats[1], self.embed_dim),
                 rh_feats[2]]
        detail = self.sta(rgbheat[:, :3])
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]
