"""Co-DETR's ViT-L backbone, simple feature pyramid and layer-decay optimizer, ported to mmdet 3.x / mmengine.

Source: Sense-X/Co-DETR (mmdet 2.25.3), mmdet/models/backbones/vit.py (EVA-02 style ViT: RoPE, SwiGLU with sub-LN,
window attention in all blocks but 3, 7, 11, 15, 19, 23), mmdet/models/necks/sfp.py and
mmdet/core/optimizers/layer_decay_optimizer_constructor.py. Imported through custom_imports by the ViT-L configs.

Parameter and buffer names are unchanged, so the released checkpoints load after convert_codino_vit_ckpt.py.
Changes that keep the math: attention through torch's scaled_dot_product_attention instead of the explicit softmax,
RoPE without einops (outer products by broadcasting, not einsum, so TF32 cannot touch the angles) and cached per
feature size, activation checkpointing through torch.utils.checkpoint instead of fairscale, and timm / mmcv-1
helpers replaced by mmcv-2 / mmengine ones. Options the released configs do not use fail instead of being ignored.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.optim import DefaultOptimWrapperConstructor

from mmdet.registry import MODELS, OPTIM_WRAPPER_CONSTRUCTORS


def rotate_half(x):
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_tables(dim, pt_seq_len, h, w, device=None, theta=10000):
    """cos / sin tables (h * w, 2 * dim) of upstream's 2-D rotary embedding: `dim` channels per axis, positions
    rescaled so that h (and w) tokens span pt_seq_len pretraining positions."""
    freqs = 1. / (theta**(torch.arange(0, dim, 2, device=device)[:dim // 2].float() / dim))
    fh = ((torch.arange(h, device=device) / h * pt_seq_len)[:, None] * freqs).repeat_interleave(2, -1)
    fw = ((torch.arange(w, device=device) / w * pt_seq_len)[:, None] * freqs).repeat_interleave(2, -1)
    f = torch.cat([fh[:, None, :].expand(h, w, -1), fw[None, :, :].expand(h, w, -1)], dim=-1)
    return f.cos().view(-1, f.shape[-1]), f.sin().view(-1, f.shape[-1])


class VisionRotaryEmbeddingFast(nn.Module):
    """Fixed-size RoPE of the window-attention blocks; the tables are buffers in the released checkpoints."""

    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None):
        super().__init__()
        size = ft_seq_len or pt_seq_len
        cos, sin = rope_tables(dim, pt_seq_len, size, size)
        self.register_buffer('freqs_cos', cos)
        self.register_buffer('freqs_sin', sin)

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class SwiGLU(nn.Module):

    def __init__(self, in_features, hidden_features, norm_layer):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.ffn_ln = norm_layer(hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(self.ffn_ln(self.act(self.w1(x)) * self.w2(x)))


class Attention(nn.Module):
    """Multi-head attention with q / v biases (no k bias) and RoPE on q and k. Window blocks get the fixed rope
    module; global blocks (rope=None) build tables for the actual token grid, as upstream's get_rope does."""

    def __init__(self, dim, num_heads, qkv_bias=True, rope=None, pt_seq_len=16):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.rope = rope
        self.pt_seq_len = pt_seq_len
        self.proj = nn.Linear(dim, dim)
        self._rope_cache = {}

    def _global_rope(self, t, h, w):
        key = (h, w, t.device)
        if key not in self._rope_cache:
            self._rope_cache[key] = rope_tables(t.shape[-1] // 2, self.pt_seq_len, h, w, device=t.device)
        cos, sin = self._rope_cache[key]
        return t * cos + rotate_half(t) * sin

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        q = F.linear(x, self.q_proj.weight, self.q_bias).reshape(B, H * W, self.num_heads, -1).transpose(1, 2)
        k = F.linear(x, self.k_proj.weight).reshape(B, H * W, self.num_heads, -1).transpose(1, 2)
        v = F.linear(x, self.v_proj.weight, self.v_bias).reshape(B, H * W, self.num_heads, -1).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q).type_as(v), self.rope(k).type_as(v)
        else:
            q, k = self._global_rope(q, H, W).type_as(v), self._global_rope(k, H, W).type_as(v)
        x = F.scaled_dot_product_attention(q, k, v)  # scale head_dim ** -0.5, as upstream
        return self.proj(x.transpose(1, 2).reshape(B, H * W, C)).view(B, H, W, C)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C), (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous() if (Hp > H or Wp > W) else x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop_path, norm_layer, window_size, rope, pt_seq_len):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, rope=rope, pt_seq_len=pt_seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), norm_layer=norm_layer)
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


def get_abs_pos(abs_pos, has_cls_token, hw):
    """Absolute position embedding resized (bicubic) from the pretraining grid to the token grid, (1, H, W, C)."""
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    size = int(abs_pos.shape[1]**0.5)
    assert size * size == abs_pos.shape[1]
    if size == h and size == w:
        return abs_pos.reshape(1, h, w, -1)
    pos = F.interpolate(abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2), size=(h, w), mode='bicubic',
                        align_corners=False)
    return pos.permute(0, 2, 3, 1)


@MODELS.register_module()
class CoDETRViT(BaseModule):
    """Upstream `ViT` with the DETR-augmentation settings of co_dino_5scale_vit_large_coco.py (use_lsj=False: the
    global blocks compute RoPE for the actual grid, so input sizes may vary). Returns [(B, C, H/16, W/16)] in FP32.

    autocast_dtype ('bfloat16' / 'float16', default None = FP32) runs the patch embedding and the blocks under CUDA
    autocast, for speed; not upstream. Only the backbone: mmcv 2.1's deformable attention, RoIAlign and focal-loss
    kernels in the heads accept float / half only, so whole-model BF16 autocast would fail there."""

    def __init__(self,
                 img_size=1024,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4 * 2 / 3,
                 qkv_bias=True,
                 drop_path_rate=0.0,
                 pt_hw_seq_len=16,
                 window_size=0,
                 window_block_indexes=(),
                 residual_block_indexes=(),
                 use_act_checkpoint=False,
                 use_lsj=False,
                 pretrain_img_size=224,
                 pretrain_use_cls_token=True,
                 autocast_dtype=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if use_lsj or residual_block_indexes:
            raise NotImplementedError('use_lsj and residual blocks are not ported (the released ViT-L detection '
                                      'configs use neither)')
        if autocast_dtype not in (None, 'bfloat16', 'float16'):
            raise ValueError(f'autocast_dtype must be None, bfloat16 or float16, got {autocast_dtype!r}')
        self.autocast_dtype = getattr(torch, autocast_dtype) if autocast_dtype else None
        del img_size  # only sized the LSJ global RoPE upstream
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.use_act_checkpoint = use_act_checkpoint
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        num_patches = (pretrain_img_size // patch_size)**2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + int(pretrain_use_cls_token), embed_dim))
        self.rope_win = VisionRotaryEmbeddingFast(embed_dim // num_heads // 2, pt_hw_seq_len, window_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            windowed = i in window_block_indexes
            self.blocks.append(
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias, dpr[i], norm_layer,
                      window_size if windowed else 0, self.rope_win if windowed else None, pt_hw_seq_len))
        self.out_norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        with torch.autocast('cuda', dtype=self.autocast_dtype or torch.float32,
                            enabled=self.autocast_dtype is not None and x.is_cuda):
            x = self.patch_embed(x)
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2]))
            for blk in self.blocks:
                if self.use_act_checkpoint and self.training:
                    x = cp.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
        return [self.out_norm(x.float()).permute(0, 3, 1, 2).contiguous()]


class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm of upstream's SFP (eps 1e-6)."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


@MODELS.register_module()
class CoDETRSFP(BaseModule):
    """Upstream `SFP`: strides 4 (use_p2), 8, 16, 32, 64 from the single stride-16 ViT map."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 use_p2=False,
                 use_act_checkpoint=False,
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list) and len(in_channels) == 1
        assert num_outs == 4 + int(use_p2), 'SFP always returns p3-p6 (+ p2)'
        c, o = in_channels[0], out_channels
        self.use_p2 = use_p2
        self.use_act_checkpoint = use_act_checkpoint

        def conv(i, j, k, s=1):
            return nn.Conv2d(i, j, kernel_size=k, stride=s, padding=k // 2, bias=False)

        def tail(i):
            return [conv(i, o, 1), LayerNorm2d(o), nn.GELU(), conv(o, o, 3), LayerNorm2d(o)]

        if use_p2:
            self.p2 = nn.Sequential(nn.Upsample(scale_factor=2), conv(c, c // 2, 3), LayerNorm2d(c // 2), nn.GELU(),
                                    nn.Upsample(scale_factor=2), conv(c // 2, c // 4, 3), LayerNorm2d(c // 4),
                                    nn.GELU(), *tail(c // 4))
        self.p3 = nn.Sequential(nn.Upsample(scale_factor=2), conv(c, c // 2, 3), LayerNorm2d(c // 2), nn.GELU(),
                                *tail(c // 2))
        self.p4 = nn.Sequential(*tail(c))
        self.p5 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), *tail(c))
        self.p6 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), conv(c, c, 3, 2), LayerNorm2d(c),
                                nn.GELU(), *tail(c))

    def _run(self, branch, x):
        if self.use_act_checkpoint and self.training:
            return cp.checkpoint(branch, x, use_reentrant=False)
        return branch(x)

    def forward(self, inputs):
        x = inputs[0]
        outs = [self._run(self.p3, x), self._run(self.p4, x), self._run(self.p5, x), self._run(self.p6, x)]
        if self.use_p2:
            outs = [self._run(self.p2, x)] + outs
        return tuple(outs)


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class CoDETRViTLayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Upstream's LayerDecayOptimizerConstructor for ViT (paramwise_cfg: num_layers, layer_decay_rate).

    With L = num_layers + 2: patch_embed / pos_embed get layer 0, backbone.blocks.i layer i + 1, everything else
    (neck, heads, out_norm) layer L - 1; lr = base_lr * rate ** (L - 1 - layer). 1-D tensors and biases get no
    weight decay. Upstream's ('pos_embed', 'cls_token') exemption compares full parameter names and so never
    matches; kept that way, pos_embed is decayed.
    """

    def add_params(self, params, module, **kwargs):
        num_layers = self.paramwise_cfg['num_layers'] + 2
        rate = self.paramwise_cfg['layer_decay_rate']
        groups = {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = param.ndim == 1 or name.endswith('.bias') or name in ('pos_embed', 'cls_token')
            if name in ('backbone.cls_token', 'backbone.mask_token', 'backbone.pos_embed') or \
                    name.startswith('backbone.patch_embed'):
                layer = 0
            elif name.startswith('backbone.blocks'):
                layer = int(name.split('.')[2]) + 1
            else:
                layer = num_layers - 1
            key = (layer, no_decay)
            if key not in groups:
                scale = rate**(num_layers - layer - 1)
                groups[key] = dict(params=[], lr=scale * self.base_lr, weight_decay=0. if no_decay else self.base_wd)
            groups[key]['params'].append(param)
        if get_dist_info()[0] == 0:
            summary = {f'layer {l} {"no_decay" if nd else "decay"}': (len(g['params']), f'{g["lr"]:.3g}')
                       for (l, nd), g in sorted(groups.items())}
            MMLogger.get_current_instance().info(f'layer-decay param groups (count, lr): {summary}')
        params.extend(groups[k] for k in sorted(groups))
