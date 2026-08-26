"""
Self-contained DINOv3 ViT-L/16 backbone for RF-DETR.

This mirrors Meta's official DINOv3 VisionTransformer (ViT-L/16, lvd1689m
pretrain) closely enough to load the released checkpoint key-for-key, while
exposing the same interface RF-DETR's ``Backbone`` expects (a list of
``(B, C, H, W)`` feature maps per requested layer index).

Key architectural differences vs the stock DINOv2 encoder that RF-DETR ships:
  * RoPE (rotary position embedding) instead of learned absolute position emb.
  * 4 "storage tokens" prepended to the sequence.
  * ``mask_k_bias`` (k-part of qkv bias is zeroed) — irrelevant for detection,
    but must be replicated for the pretrained weights to be valid.
  * LayerScale + standard LayerNorm, MLP ratio 4, depth 24, dim 1024, heads 16,
    patch_size 16.

Only global (non-windowed) attention is implemented: at resolution 560 the patch
grid is 35x35 = 1225 tokens + 5 special tokens, which is cheap enough.

References: facebookresearch/dinov3 (layers/rope_position_encoding.py,
layers/attention.py, layers/block.py, layers/patch_embed.py,
models/vision_transformer.py).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rfdetr.models.backbone.backbone import Backbone, BackboneBase
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.backbone import Joiner
from rfdetr.models.position_encoding import build_position_encoding
from rfdetr.utilities.tensors import NestedTensor


# --------------------------------------------------------------------------- #
# Channel-wise remap: transfer a conv/linear weight from a source model whose   #
# channel counts differ (e.g. DINOv2 768 -> DINOv3 1024) by interpolating along  #
# the differing axis. Used to warm-start the DINOv3 MultiScaleProjector from an  #
# rf-detr checkpoint so it is not randomly initialised.                         #
# --------------------------------------------------------------------------- #
def _remap_tensor(src: Tensor, dst_shape: tuple[int, ...]) -> Tensor:
    """Return a tensor of ``dst_shape`` derived from ``src`` by 1-D linear
    interpolation along any differing axis (for conv/linear weights). Falls back
    to truncating/padding if interpolation is not applicable."""
    if tuple(src.shape) == tuple(dst_shape):
        return src.clone()
    w = src.float()
    # treat 2-D (linear) weights as 4-D convs with 1x1 kernels
    if w.dim() == 2:
        w = w.unsqueeze(-1).unsqueeze(-1)
        squeeze_out = True
    else:
        squeeze_out = False
    dst = list(dst_shape)
    if w.dim() == 2:
        dst = list(dst_shape) + [1, 1]

    O, I = w.shape[0], w.shape[1]
    # interpolate input channels (axis 1): lay out as [1, rest, I] so F.interpolate
    # expands the LAST dim (I) to dst[1], keeping rest = O*K*K fixed.
    if I != dst[1]:
        rest = O * w.shape[2] * w.shape[3]
        flat = w.permute(1, 0, 2, 3).reshape(I, -1).permute(1, 0).unsqueeze(0)  # [1, rest, I]
        flat = F.interpolate(flat, size=dst[1], mode="linear", align_corners=True)  # [1, rest, dst[1]]
        w = flat.squeeze(0).permute(1, 0).reshape(dst[1], O, w.shape[2], w.shape[3])  # [dst[1], O, K, K]
        I = dst[1]
    # interpolate output channels (axis 0): lay out as [1, rest', O]
    if O != dst[0]:
        rest = I * w.shape[2] * w.shape[3]
        flat = w.reshape(O, -1).permute(1, 0).unsqueeze(0)  # [1, rest', O]
        flat = F.interpolate(flat, size=dst[0], mode="linear", align_corners=True)
        w = flat.squeeze(0).permute(1, 0).reshape(dst[0], I, w.shape[2], w.shape[3])

    if squeeze_out:
        w = w.squeeze(-1).squeeze(-1)
    if tuple(w.shape) != tuple(dst_shape):
        # last-resort: truncate / zero-pad to the exact dst shape
        sl = tuple(slice(0, min(s, d)) for s, d in zip(w.shape, dst_shape))
        out = src.new_zeros(dst_shape)
        out[sl] = w[sl]
        return out
    return w.to(src.dtype)


# --------------------------------------------------------------------------- #
# RoPE                                                                         #
# --------------------------------------------------------------------------- #
def rope_rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)


class RopePositionEmbedding(nn.Module):
    def __init__(self, embed_dim: int, *, num_heads: int, base: float = 100.0,
                 normalize_coords: str = "separate") -> None:
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        self.num_heads = num_heads
        self.D_head = embed_dim // num_heads
        self.base = base
        self.normalize_coords = normalize_coords
        self.periods = nn.Buffer(torch.empty(self.D_head // 4))
        periods = base ** (2 * torch.arange(self.D_head // 4) / (self.D_head // 2))
        self.periods.data = periods

    def forward(self, *, H: int, W: int):
        dtype = torch.float32 if self.periods.dtype == torch.float32 else self.periods.dtype
        dev = self.periods.device
        if self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, device=dev, dtype=dtype) / H
            coords_w = torch.arange(0.5, W, device=dev, dtype=dtype) / W
        else:
            coords_h = torch.arange(0.5, H, device=dev, dtype=dtype) / H
            coords_w = torch.arange(0.5, W, device=dev, dtype=dtype) / W
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        ).flatten(0, 1)
        coords = 2.0 * coords - 1.0  # [-1, 1]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)  # [HW, D_head]
        return torch.sin(angles), torch.cos(angles)


# --------------------------------------------------------------------------- #
# Attention                                                                     #
# --------------------------------------------------------------------------- #
class LinearKMaskedBias(nn.Linear):
    """Linear with a learned ``bias_mask`` buffer multiplied into the bias.

    DINOv3 initialises the mask to 1 everywhere except the key-portion (set to 0),
    so during (unmasked) detection inference the key projection receives no bias.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            # qkv are concatenated as [q | k | v]; zero the *key* bias so the key
            # projection receives no (learned) bias during unmasked detection.
            out = self.out_features
            assert out % 3 == 0, "LinearKMaskedBias expects qkv-concatenated output"
            per = out // 3
            self.register_buffer(
                "bias_mask",
                torch.cat([torch.ones(per), torch.zeros(per), torch.ones(per)]).to(self.bias.dtype),
            )

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, qkv_bias: bool = True,
                 proj_bias: bool = True, mask_k_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        linear_cls = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_cls(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def apply_rope(self, q: Tensor, k: Tensor, rope):
        sin, cos = rope
        q_dtype, k_dtype = q.dtype, k.dtype
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)
        return q.to(q_dtype), k.to(k_dtype)

    def forward(self, x: Tensor, rope=None) -> Tensor:
        qkv = self.qkv(x)
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None,
                 act_layer=nn.GELU, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 1024) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)            # (B, C, H, W)
        return x.permute(0, 2, 3, 1)  # (B, H, W, C)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 4.0,
                 qkv_bias: bool = True, proj_bias: bool = True, ffn_bias: bool = True,
                 init_values: float = 1.0, norm_layer=nn.LayerNorm,
                 mask_k_bias: bool = True) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, proj_bias, mask_k_bias)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * ffn_ratio), bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, rope=None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVisionTransformer(nn.Module):
    """DINOv3 ViT-L/16 (depth 24, dim 1024, heads 16, 4 storage tokens, RoPE).

    Returns a list of ``(B, C, H, W)`` feature maps, one per requested block
    index, in ascending order.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 1024, depth: int = 24, num_heads: int = 16,
                 ffn_ratio: float = 4.0, qkv_bias: bool = True, drop_path_rate: float = 0.0,
                 layerscale_init: float = 1.0, ffn_bias: bool = True, n_storage_tokens: int = 4,
                 mask_k_bias: bool = True, out_feature_indexes: tuple[int, ...] = (2, 5, 8, 11),
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        norm_layer = lambda d: nn.LayerNorm(d, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.out_feature_indexes = list(out_feature_indexes)
        self.n_storage_tokens = n_storage_tokens
        self.gradient_checkpointing = gradient_checkpointing

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.rope_embed = RopePositionEmbedding(embed_dim, num_heads=num_heads, base=100.0)

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads, ffn_ratio, qkv_bias, True, ffn_bias,
                               layerscale_init, norm_layer, mask_k_bias)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self._out_feature_channels = [embed_dim] * len(out_feature_indexes)
        self.shape: tuple[int, int] = (img_size, img_size)

    def forward(self, x: Tensor) -> list[Tensor]:
        B, _, H, W = x.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        x = self.patch_embed(x).flatten(1, 2)  # (B, L, C)
        cls = self.cls_token.expand(B, -1, -1)
        storage = self.storage_tokens.expand(B, -1, -1)
        x = torch.cat([cls, storage, x], dim=1)  # (B, 5+L, C)
        rope = self.rope_embed(H=ph, W=pw)

        feats: list[Tensor] = []
        for i, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, rope, use_reentrant=False)
            else:
                x = blk(x, rope=rope)
            if i in self.out_feature_indexes:
                patch = x[:, self.n_storage_tokens + 1:, :]  # drop cls + storage
                patch = self.norm(patch)
                f = patch.reshape(B, ph, pw, self.embed_dim).permute(0, 3, 1, 2)
                feats.append(f)
        return feats


# --------------------------------------------------------------------------- #
# RF-DETR Backbone wrapper                                                      #
# --------------------------------------------------------------------------- #
class DinoV3Backbone(Backbone):
    """Drop-in replacement for ``Backbone`` that uses a DINOv3 ViT-L/16 encoder.

    Skips ``Backbone.__init__`` (which asserts a ``dinov2`` encoder name and would
    build the wrong-size DINOv2 model) and builds the DINOv3 encoder + projector
    directly. Inherits ``get_named_param_lr_pairs`` and the nn.Module machinery.
    """

    def __init__(self, name: str, out_feature_indexes: list[int] | None,
                 shape: tuple[int, int], use_registers: bool = False,
                 use_windowed_attn: bool = False, gradient_checkpointing: bool = False,
                 load_dinov2_weights: bool = False, patch_size: int = 16,
                 num_windows: int = 1, positional_encoding_size: int = 37,
                 drop_path_rate: float = 0.0, window_block_indexes=None,
                 weights_path: str | None = None,
                 freeze_blocks: int = 0,
                 init_projector_from: str | None = None) -> None:
        BackboneBase.__init__(self)
        if isinstance(shape, int):
            shape = (shape, shape)
        out_feature_indexes = list(out_feature_indexes or [2, 5, 8, 11])

        self.shape = shape
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.positional_encoding_size = positional_encoding_size
        self.layer_norm = False
        self.rms_norm = False
        self.out_channels = 256
        self.projector_scale = ["P4"]
        self.pos_enc_scales = list(range(1, len(out_feature_indexes) + 1))
        self._export = False

        self.encoder = DinoVisionTransformer(
            img_size=shape[0], patch_size=patch_size, embed_dim=1024, depth=24,
            num_heads=16, out_feature_indexes=tuple(out_feature_indexes),
            n_storage_tokens=4, mask_k_bias=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu")
            missing, unexpected = self.encoder.load_state_dict(sd, strict=True)
            assert not missing and not unexpected, f"DINOv3 load mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
        self.encoder.shape = shape

        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]
        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=self.out_channels,
            scale_factors=scale_factors,
            layer_norm=self.layer_norm,
            rms_norm=self.rms_norm,
        )
        self.cross_attn_projector = None

        # Warm-start the projector from an rf-detr checkpoint (e.g. base/o365).
        # Channel counts differ (DINOv2 768 -> DINOv3 1024), so weights are
        # interpolated along the mismatched axis rather than copied blindly.
        if init_projector_from:
            self._init_projector_from(init_projector_from)

        # Partial unfreeze: freeze the first `freeze_blocks` transformer blocks
        # (and the stem) so the strong DINOv3 pretrain is preserved, while the
        # last blocks + projector + decoder adapt to the detection head.
        if freeze_blocks > 0:
            self._freeze_encoder(freeze_blocks)

    def _init_projector_from(self, ckpt_path: str) -> None:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ck = ck.get("model", ck) if isinstance(ck, dict) else ck
        prefix = "backbone.0.projector."
        src = {k[len(prefix):]: v for k, v in ck.items() if k.startswith(prefix)}
        if not src:
            return
        dst = self.projector.state_dict()
        loaded = {}
        for k, v in dst.items():
            if k in src:
                s = src[k]
                if tuple(s.shape) == tuple(v.shape):
                    loaded[k] = s
                else:
                    try:
                        loaded[k] = _remap_tensor(s, tuple(v.shape))
                    except Exception as e:
                        raise RuntimeError(f"remap failed for {k}: src={tuple(s.shape)} dst={tuple(v.shape)}") from e
        self.projector.load_state_dict(loaded, strict=False)
        print(f"[dinov3] projector warm-started from {ckpt_path}: "
              f"{len(loaded)}/{len(dst)} tensors remapped")

    def _freeze_encoder(self, freeze_blocks: int) -> None:
        e = self.encoder
        depth = len(e.blocks)
        freeze_blocks = min(freeze_blocks, depth)
        # freeze stem
        for m in (e.patch_embed, e.rope_embed):
            for p in m.parameters():
                p.requires_grad = False
        for t in (e.cls_token, e.storage_tokens, e.mask_token):
            t.requires_grad = False
        for p in e.norm.parameters():
            p.requires_grad = False
        # freeze first `freeze_blocks` blocks
        for i, blk in enumerate(e.blocks):
            if i < freeze_blocks:
                for p in blk.parameters():
                    p.requires_grad = False
        unfrozen = sum(1 for blk in e.blocks
                        if any(p.requires_grad for p in blk.parameters()))
        print(f"[dinov3] encoder partial-unfreeze: froze {freeze_blocks}/{depth} blocks; "
              f"trainable blocks = {unfrozen}")

    def forward(self, tensor_list: NestedTensor):
        raw_feats = self.encoder(tensor_list.tensors)
        feats = self.projector(raw_feats)
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out, None

    def export(self):
        return self


def build_dinov3_backbone(name: str, out_feature_indexes, shape, **kwargs) -> Joiner:
    """Mirror ``build_backbone``'s return type (a Joiner) for dinov3 encoders."""
    weights_path = kwargs.pop("weights_path", None)
    backbone = DinoV3Backbone(
        name=name, out_feature_indexes=out_feature_indexes, shape=shape,
        weights_path=weights_path, **kwargs,
    )
    pos = build_position_encoding(256, "sine")
    return Joiner(backbone, pos)
