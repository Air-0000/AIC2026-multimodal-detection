"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from .sovereign_v6_modules import BiDirectionalSSM2d, MultiHeadLinearAttention2d

from ..core import register

__all__ = ['HybridEncoder']


class HaarWaveletDownsample(nn.Module):
    """基于 Haar 离散小波的高频无损下采样层 (来源: V1-246 MWHL)
    采用标准分组卷积 (Grouped Conv2d, groups=C) 实现，硬件级 16 字节对齐，彻底免疫 Volta (SM_70/V100) FP16 内存不对齐异常
    将 (B, C, 2H, 2W) 下采样为 (B, 4C, H, W)，显式分离 LL、LH、HL(垂直)、HH
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        w_base = torch.tensor([
            [[0.5, 0.5], [0.5, 0.5]],      # LL: x1 + x3 + x2 + x4
            [[-0.5, -0.5], [0.5, 0.5]],    # HL: -x1 + x2 - x3 + x4
            [[-0.5, 0.5], [-0.5, 0.5]],    # LH: -x1 - x2 + x3 + x4
            [[0.5, -0.5], [-0.5, 0.5]],    # HH: x1 - x2 - x3 + x4
        ], dtype=torch.float32).unsqueeze(1) # (4, 1, 2, 2)
        weight = w_base.repeat(in_channels, 1, 1, 1) # (4*in_channels, 1, 2, 2)
        self.register_buffer('weight', weight, persistent=False)

    def forward(self, x):
        B, C, H2, W2 = x.shape
        w = self.weight.to(device=x.device, dtype=x.dtype)
        out_conv = F.conv2d(x, w, stride=2, groups=C)
        H, W = out_conv.shape[-2:]
        return out_conv.view(B, C, 4, H, W).permute(0, 2, 1, 3, 4).reshape(B, 4 * C, H, W).contiguous()


class LocalMixedCrossConv(nn.Module):
    """【V1代码讲解162】局部混合模块 LMM (正交双向十字卷积 3x7 + 7x3).
    通过正交非对称卷积并联，同时覆盖极细垂直立柱(7x3)与大长宽比横向船体/悬臂(3x7)，
    彻底纠正单一垂直 11x1 核对水平船体与十字路标的各向异性感知盲区。
    """
    def __init__(self, dim):
        super().__init__()
        # 垂直分支: 7x3 (加强纵向立柱与垂直接缝)
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=(7, 3), padding=(3, 1), groups=dim, bias=False)
        # 水平分支: 3x7 (加强横向扁平船体与十字路标横梁)
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=(3, 7), padding=(1, 3), groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv_v(x) + self.conv_h(x)))


class P2CrossScaleRefinementV10_1(nn.Module):
    """Sovereign V10.1 (P2-CSRG+) 终极强化版:
    - LMM 正交双向十字卷积 (V1-162): 3x7(水平扁平船体) + 7x3(垂直极细立柱)
    - 小波垂直高频下采样 (V1-246): 解决 160x160 -> 80x80 立柱平滑消失
    - 空间显著性频率门控 (V1-248): 水面/背景自适应截断，消除虚警
    - 严格零初始化 (Zero-Init): 初始时刻 100.00% 等价于 Sovereign V10 预训练检查点
    """
    def __init__(self, p2_dim=128, p3_dim=256, hidden_dim=64, use_lfp=False):
        super().__init__()
        self.conv_in = nn.Conv2d(p2_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        
        # 1. 【V1-162】LMM 正交双向十字卷积 (3x7 水平 + 7x3 垂直) 替换单一纵向 11x1
        self.lmm_strip = LocalMixedCrossConv(hidden_dim)
        self.vertical_strip = self.lmm_strip
        
        # 2. 小波下采样 + 双路保峰池化融合通道
        self.dwt = HaarWaveletDownsample(hidden_dim)
        self.downsample = nn.Conv2d(hidden_dim * 4 + hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn_down = nn.BatchNorm2d(hidden_dim)

        # 2.1 V13 LFP (Low-Frequency Purification Module, CVPR 2026, V1-231)
        self.use_lfp = use_lfp
        if self.use_lfp:
            from .sovereign_v13_modules import LowFrequencyPurificationModule
            self.lfp = LowFrequencyPurificationModule(dim=hidden_dim, low_dim=p3_dim)
            print("[P2-CSRG+] V1-231 LFP (Low-Frequency Purification): ON (zero-init)")
        
        # 3. 空间自适应高频门控 (CSR 机制)
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 4. 零初始化输出投影 (绝对安全保障，未出现在旧权重中，杜绝旧权重污染未对齐的中间特征)
        self.proj_v10_1 = nn.Conv2d(hidden_dim, p3_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj_v10_1.weight)
        nn.init.zeros_(self.proj_v10_1.bias)

    def forward(self, p3_feat, p2_feat, depth=None):
        h = self.act(self.bn_in(self.conv_in(p2_feat)))
        h_strip = self.vertical_strip(h)
        h_feat = h + h_strip
        
        h_dwt = self.dwt(h_feat) # (B, hidden_dim * 4, 80, 80)
        h_avg = F.avg_pool2d(h_feat, kernel_size=2, stride=2)
        h_max = F.max_pool2d(h_feat, kernel_size=2, stride=2)
        h_combined = torch.cat([h_dwt, h_avg, h_max], dim=1) # (B, 6*hidden_dim, 80, 80)
        
        h_fused = self.act(self.bn_down(self.downsample(h_combined)))
        if self.use_lfp and hasattr(self, 'lfp'):
            h_fused = self.lfp(h_fused, f_low=p3_feat)
        spatial_gate = self.gate(h_fused)

        # 天空与水面波纹高频噪声精准截断 (Sky-Water Noise Attenuator)
        B, _, H_f, W_f = h_fused.shape
        with torch.no_grad():
            y_grid = torch.linspace(0.0, 1.0, H_f, device=h_fused.device, dtype=h_fused.dtype).view(1, 1, H_f, 1).expand(B, 1, H_f, W_f).contiguous()
            clean_mask = torch.ones_like(y_grid)
            if depth is not None:
                depth_raw = torch.clamp(F.interpolate(depth.to(dtype=h_fused.dtype), size=(H_f, W_f), mode='nearest') * 0.5 + 0.5, 0.0, 1.0)
                # 1. 天空区域高频净化 (y < 0.28 且无有效深度回波)
                sky_factor = torch.clamp((y_grid - 0.10) / 0.15, 0.0, 1.0)
                clean_mask = torch.where((y_grid < 0.28) & (depth_raw < 0.02), sky_factor, clean_mask)

                # 2. 水面波纹高频净化 (y > 0.48 且无 3D 刚体阶跃梯度)
                d_pad = F.pad(depth_raw, (1, 1, 1, 1), mode='replicate')
                grad_x = d_pad[:, :, 1:-1, 2:] - d_pad[:, :, 1:-1, :-2]
                grad_y = d_pad[:, :, 2:, 1:-1] - d_pad[:, :, :-2, 1:-1]
                grad_d = torch.sqrt(grad_x**2 + grad_y**2 + 1e-4)
                water_factor = torch.clamp(grad_d / 0.015, 0.0, 1.0)
                clean_mask = torch.where((y_grid > 0.48) & (grad_d < 0.015), water_factor, clean_mask).to(dtype=h_fused.dtype)

        gated_feat = (h_fused * spatial_gate * clean_mask).to(dtype=h_fused.dtype)
        delta = self.proj_v10_1(gated_feat)
        return p3_feat + delta


class P2CrossScaleRefinement(nn.Module):
    """P2 (160x160) Cross-Scale High-Frequency Residual Guidance Block.
    Performs peak-salience dual pooling (Avg + Max) and multi-scale convolution
    from 160x160 (P2, 128ch) down to 80x80 (P3, 256ch).
    Zero-initialized to guarantee 100.00% exact numerical identity at Step 0.
    """
    def __init__(self, p2_dim=128, p3_dim=256, hidden_dim=64):
        super().__init__()
        self.conv_in = nn.Conv2d(p2_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        
        # Dual-Path Anti-Aliased Peak Pooling (Avg + Max avoids losing sub-pixel peaks)
        self.downsample = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn_down = nn.BatchNorm2d(hidden_dim)
        
        # Output projection with Zero-Init
        self.proj_out = nn.Conv2d(hidden_dim, p3_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, p3_feat, p2_feat):
        # p3_feat: (B, 256, H, W) e.g. (B, 256, 80, 80)
        # p2_feat: (B, 128, 2*H, 2*W) e.g. (B, 128, 160, 160)
        h = self.act(self.bn_in(self.conv_in(p2_feat)))
        h_avg = F.avg_pool2d(h, kernel_size=2, stride=2)
        h_max = F.max_pool2d(h, kernel_size=2, stride=2)
        h_down = torch.cat([h_avg, h_max], dim=1) # (B, hidden_dim * 2, H, W)
        h_fused = self.act(self.bn_down(self.downsample(h_down))) # (B, hidden_dim, H, W)
        delta = self.proj_out(h_fused) # (B, p3_dim, H, W)
        return p3_feat + delta


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# self.cv1 = Conv(c1, c2, 1, 1)
class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None, use_anti_alias=False, anti_alias_alpha=0.0):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)
        self.use_anti_alias = use_anti_alias
        if use_anti_alias:
            from .anti_aliasing import AntiAliasedDownsample
            self.anti_alias = AntiAliasedDownsample(c2, initial_alpha=anti_alias_alpha, learnable=True)
        else:
            self.anti_alias = nn.Identity()

    def forward(self, x):
        return self.anti_alias(self.cv2(self.cv1(x)))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class PKIVGGBlock(VGGBlock):
    """VGGBlock + zero-init PKI large-kernel side branch (PKINet, CVPR2024).

    The re-param main path (3x3+1x1) is UNTOUCHED, so pretrained checkpoint
    weights load as-is. The PKI side branch (5x5 DW + 7x7 dilated DW -> 1x1
    fusion) starts with a ZERO-init fusion conv, so at init the block is
    numerically identical to VGGBlock -- safe to stack on a strong ckpt.
    """
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__(ch_in, ch_out, act=act)
        self.pki_dw5 = nn.Conv2d(ch_out, ch_out, 5, 1, 2, groups=ch_out, bias=False)
        self.pki_dw7 = nn.Conv2d(ch_out, ch_out, 7, 1, 6, dilation=2, groups=ch_out, bias=False)
        self.pki_fuse = nn.Conv2d(ch_out * 2, ch_out, 1, bias=False)
        nn.init.zeros_(self.pki_fuse.weight)   # zero-init => exact identity at start

    def forward(self, x):
        y = super().forward(x)
        p = torch.cat([self.pki_dw5(y), self.pki_dw7(y)], dim=1)
        return y + self.pki_fuse(p)


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu",
                 csp_type='csp2',
                 bottletype=VGGBlock,
                 ):
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        # fork fix: 'csp' branch was missing (UnboundLocalError). The D-FINE official
        # release checkpoints (e.g. dfine_x_obj2coco.pth) use the classic two-branch
        # CSPLayer with expansion=1.0, i.e. csp_type='csp'.
        if csp_type == 'csp2':
            InnerCSPLayer = CSPLayer2
        else:
            InnerCSPLayer = CSPLayer
        self.cv2 = nn.Sequential(InnerCSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=bottletype), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(InnerCSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=bottletype), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = [t.contiguous() for t in self.cv1(x).chunk(2, 1)]
        for m in [self.cv2, self.cv3]:
            y.append(m(y[-1]).contiguous())
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = [t.contiguous() for t in self.cv1(x).split((self.c, self.c), 1)]
        for m in [self.cv2, self.cv3]:
            y.append(m(y[-1]).contiguous())
        return self.cv4(torch.cat(y, 1))


# This layer is equivalent to RepC3 in YOLOs repo
class CSPLayer2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock,
                 ):
        super(CSPLayer2, self).__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels * 2, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        return self.conv3(y[0] + self.bottlenecks(y[1]))

class RepNCSPELAN5(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)

        self.cv2 = nn.Sequential(CSPLayer2(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock))
        self.cv3 = nn.Sequential(CSPLayer2(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        out = self.cv4(torch.cat(y, 1))
        return out

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        out = self.cv4(torch.cat(y, 1))
        return out

# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 csp_type='csp',
                 fuse_op='cat',
                 pki_bottleneck=False,
                 use_ssm_levels=None,
                 ssm_d_state=16,
                 use_mhla_levels=None,
                 mhla_num_heads=4,
                 use_p2_guidance=False,
                 p2_dim=128,
                 p2_version='v10',
                 use_lfp=False,
                 use_pbd=False,
                 use_starconv=False,
                 use_anti_alias=False,
                 anti_alias_alpha=0.0,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.fuse_op = fuse_op
        self.use_anti_alias = use_anti_alias
        self.anti_alias_alpha = anti_alias_alpha
        if use_anti_alias:
            print(f"[HybridEncoder] Anti-Aliasing BlurPool Downsampling: ON (initial_alpha={anti_alias_alpha})")
        bottletype = PKIVGGBlock if pki_bottleneck else VGGBlock
        if pki_bottleneck:
            print("[HybridEncoder] PKIBlock bottleneck: ON (zero-init PKI branch)")

        # P2 (160x160) Cross-Scale Residual Guidance (Sovereign V10 & V10.1+)
        self.use_p2_guidance = use_p2_guidance
        self.p2_version = p2_version
        self.use_lfp = use_lfp
        if use_p2_guidance:
            if p2_version == "v10_1":
                self.p2_refinement = P2CrossScaleRefinementV10_1(p2_dim=p2_dim, p3_dim=hidden_dim, hidden_dim=64, use_lfp=use_lfp)
                print(f"[HybridEncoder] P2-CSRG+ (MWHL Wavelet + V1-162 LMM Cross-Conv 3x7+7x3 + CSR Gate, LFP={use_lfp}): ON (zero-init)")
            else:
                self.p2_refinement = P2CrossScaleRefinement(p2_dim=p2_dim, p3_dim=hidden_dim, hidden_dim=64)
                print("[HybridEncoder] P2 Cross-Scale Residual Guidance (160x160 -> P3): ON (zero-init)")
        else:
            self.p2_refinement = None

        # V1创新038: PBD (Peak-Background Decoupling) for P3
        self.use_pbd = use_pbd
        if use_pbd:
            from .sovereign_v13_modules import PeakBackgroundDecoupling
            self.pbd = PeakBackgroundDecoupling(dim=hidden_dim)
            print("[HybridEncoder] V1创新038 PBD (Peak-Background Decoupling) on P3: ON (zero-init)")
        else:
            self.pbd = None

        # V1-011: StarConv (Microsoft Star Operation, CVPR 2024) on P4
        self.use_starconv = use_starconv
        if use_starconv:
            from .sovereign_v13_modules import StarOperationConv2d
            self.starconv = StarOperationConv2d(dim=hidden_dim)
            print("[HybridEncoder] V1-011 StarConv (Microsoft Star Operation, CVPR 2024) on P4: ON (zero-init)")
        else:
            self.starconv = None

        # Mamba 2D-SSM for P3/P4 linear global context (Sovereign V6)
        self.use_ssm_levels = use_ssm_levels
        if use_ssm_levels is not None and len(use_ssm_levels) > 0:
            self.use_ssm_levels = [int(i) for i in use_ssm_levels]
            self.ssm_blocks = nn.ModuleDict({
                str(lvl): BiDirectionalSSM2d(hidden_dim, d_state=ssm_d_state)
                for lvl in self.use_ssm_levels
            })
            print(f"[HybridEncoder] Mamba 2D-SSM enabled on levels {self.use_ssm_levels} (zero-init, O(N) linear complexity)")
        else:
            self.use_ssm_levels = []
            self.ssm_blocks = None

        # MHLA Multi-Head Linear Attention (V1-218 / ICLR 2026, Sovereign V12)
        self.use_mhla_levels = use_mhla_levels
        if use_mhla_levels is not None and len(use_mhla_levels) > 0:
            self.use_mhla_levels = [int(i) for i in use_mhla_levels]
            self.mhla_blocks = nn.ModuleDict({
                str(lvl): MultiHeadLinearAttention2d(hidden_dim, num_heads=mhla_num_heads)
                for lvl in self.use_mhla_levels
            })
            print(f"[HybridEncoder] MHLA Linear Attention enabled on levels {self.use_mhla_levels} (zero-init, O(N) linear dual-core)")
        else:
            self.use_mhla_levels = []
            self.mhla_blocks = None

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if in_channel != hidden_dim:
                proj = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                            ('norm', nn.BatchNorm2d(hidden_dim))
                        ]))
            else:
                proj = nn.Identity()
            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
            )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        input_dim = hidden_dim if self.fuse_op == 'sum' else hidden_dim * 2   # deim use sum instead of cat

        Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1)
        SCDown_Conv = nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, use_anti_alias=use_anti_alias, anti_alias_alpha=anti_alias_alpha))

        c1, c2, c3, c4, num_blocks = input_dim, hidden_dim, hidden_dim*2, round(expansion * hidden_dim // 2), round(3 * depth_mult)
        if version == 'dfine':
            Fuse_Block = RepNCSPELAN4(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act, csp_type=csp_type, bottletype=bottletype)
        elif version == 'deim':
            Fuse_Block = RepNCSPELAN5(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act)
        else:       # RT-DETR
            Fuse_Block = CSPLayer(in_channels=c1, out_channels=c2, num_blocks=num_blocks, act=act, \
                                    expansion=expansion, bottletype=bottletype)
            Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act)
            SCDown_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(copy.deepcopy(Lateral_Conv))
            self.fpn_blocks.append(copy.deepcopy(Fuse_Block))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(copy.deepcopy(SCDown_Conv))
            self.pan_blocks.append(copy.deepcopy(Fuse_Block))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats, depth=None):
        if len(feats) == len(self.in_channels) + 1 and self.use_p2_guidance:
            p2_feat = feats[0]
            feats = feats[1:]
        else:
            p2_feat = None
            if self.use_p2_guidance:
                # return_idx no longer yields the extra P2 level: the guidance
                # branch would otherwise do nothing without a word.
                raise ValueError(
                    f'use_p2_guidance is on but the backbone returned {len(feats)} levels '
                    f'while the encoder expects {len(self.in_channels)}+1; check return_idx.')

        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Mamba 2D-SSM linear global context enhancement (Sovereign V6)
        if self.ssm_blocks is not None:
            for lvl in self.use_ssm_levels:
                if lvl < len(proj_feats):
                    proj_feats[lvl] = self.ssm_blocks[str(lvl)](proj_feats[lvl])

        # MHLA Linear Attention global non-local interaction (Sovereign V12)
        if self.mhla_blocks is not None:
            for lvl in self.use_mhla_levels:
                if lvl < len(proj_feats):
                    proj_feats[lvl] = self.mhla_blocks[str(lvl)](proj_feats[lvl])

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                # Always build a dynamic pos_embed so inference works at ANY input
                # size (needed for multi-scale TTA). The fixed eval_spatial_size
                # buffer path is only valid for the exact training resolution.
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            fused_feat = (upsample_feat + feat_low).contiguous() \
                if self.fuse_op == 'sum' else torch.concat([upsample_feat, feat_low], dim=1).contiguous()
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](fused_feat)
            inner_outs.insert(0, inner_out)

        # P2 Cross-Scale Residual Guidance Injection into P3 (80x80)
        if self.use_p2_guidance and self.p2_refinement is not None and p2_feat is not None:
            if self.p2_version == "v10_1":
                inner_outs[0] = self.p2_refinement(inner_outs[0], p2_feat, depth=depth)
            else:
                inner_outs[0] = self.p2_refinement(inner_outs[0], p2_feat)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            fused_feat = (downsample_feat + feat_height).contiguous() \
                if self.fuse_op == 'sum' else torch.concat([downsample_feat, feat_height], dim=1).contiguous()
            out = self.pan_blocks[idx](fused_feat)
            outs.append(out)

        # V1创新038: PBD (Peak-Background Decoupling) for P3
        if getattr(self, 'use_pbd', False) and getattr(self, 'pbd', None) is not None:
            outs[0] = self.pbd(outs[0])

        # V1-011: StarConv (Microsoft Star Operation, CVPR 2024) for P4
        if getattr(self, 'use_starconv', False) and getattr(self, 'starconv', None) is not None and len(outs) > 1:
            outs[1] = self.starconv(outs[1])

        return outs
