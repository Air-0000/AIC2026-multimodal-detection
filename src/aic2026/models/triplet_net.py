"""
Track 2: 三模态扩展模型
=========================
基于 M2I2HA 的 M2I2HANet_v8，扩展为 RGB + Depth + IR 三输入。

架构概览:
```
RGB ──→ backbone ──→ Intra-HG ──┐
D   ──→ backbone ──→ Intra-HG ──┼──→ Inter-HG(3模态) ──→ 检测头
T   ──→ backbone ──→ Intra-HG ──┘
                               ↑
                         质量评估模块 (TODO)
```

当前状态: 骨架代码，具体实现在复现 baseline 后讨论。
"""

from typing import Literal
import torch
import torch.nn as nn


class AICTripletNet(nn.Module):
    """RGB + Depth + IR 三模态检测网络骨架。"""

    def __init__(
        self,
        nc: int = 12,
        imgsz: int = 640,
        net_scale: Literal["n", "s"] = "n",
        channels: int = 3,
    ):
        super().__init__()
        self.nc = nc
        self.imgsz = imgsz
        self.net_scale = net_scale

        # TODO: Phase 2 — 实现三路 backbone
        # 参考 M2I2HANet_v8: img_backbone + ir_backbone + depth_backbone
        # 三路均用 ImageNet 权重初始化

        # TODO: Phase 2 — 实现三模态 Intra-Hypergraph Enhancement
        # 参考 IntraHyperEnhance，每个模态独立一个

        # TODO: Phase 2 — 实现三模态质量评估模块
        # Uncertainty Head (σ_i) + Quality Head (q_i)
        # 见 pretext.md §4 方案 A+B

        # TODO: Phase 2 — 实现三模态 Inter-Hypergraph Fusion
        # 扩展 IntreHyperFusion 从双输入 → 三输入

        # TODO: Phase 2 — 实现检测头
        # 参考 DetectV8

    def forward(self, rgb, depth, ir):
        # TODO
        raise NotImplementedError("三模态网络实现在 baseline 复现后继续")


class UncertaintyHead(nn.Module):
    """异方差不确定性头 (Kendall et al., 2018)

    输入: 模态特征图 [B, C, H, W]
    输出: σ_i ∈ (0, ∞) — 输入依赖的不确定性
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # TODO: Phase 2
        # self.head = nn.Sequential(...)
        raise NotImplementedError


class QualityHead(nn.Module):
    """特征统计质量评估头

    输入: 模态特征图 [B, C, H, W]
    输出: q_i ∈ [0, 1] — 模态质量分数
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # TODO: Phase 2
        # 1. per-channel 统计量 (mean, var, max, min, energy)
        # 2. 2-layer MLP → sigmoid
        raise NotImplementedError
