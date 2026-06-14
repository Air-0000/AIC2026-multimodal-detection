"""
Track 2: 三模态数据集 + Depth 预处理
======================================
扩展 M2I2HA 的 MultiModalYoloDataset，支持 Depth 第三模态。
"""

from typing import Any
import torch
import numpy as np


def depth_log_normalize(depth_mm: np.ndarray, max_mm: int = 19999) -> np.ndarray:
    """Depth log 归一化: d_log = log(d+1) / log(max+1)

    Args:
        depth_mm: 原始 depth 值 (mm), shape [H, W], dtype uint16
        max_mm: depth 最大有效值

    Returns:
        log-normalized depth, shape [3, H, W] (复制3通道)
    """
    d_norm = depth_mm.astype(np.float32)
    d_log = np.log(d_norm + 1) / np.log(max_mm + 1)
    d_log = np.clip(d_log, 0, 1)
    # 单通道复制成 3 通道
    return np.stack([d_log] * 3, axis=0)


def depth_linear_normalize(depth_mm: np.ndarray, max_mm: int = 19999) -> np.ndarray:
    """Depth 线性归一化 (备选方案)"""
    d_norm = depth_mm.astype(np.float32) / max_mm
    d_norm = np.clip(d_norm, 0, 1)
    return np.stack([d_norm] * 3, axis=0)


class AICTripletDataset:
    """三模态数据加载器骨架。

    TODO: Phase 1 后实现
    继承 M2I2HA 的 MultiModalYoloDataset 并添加 depth 模态支持。
    """

    def __init__(self, *args, **kwargs):
        # TODO: Phase 2
        # 1. 调用 super().__init__(modalities=['img', 'depth', 'ir'])
        # 2. 加载 depth 目录下的图像列表
        # 3. 注册 depth 预处理函数 (log_transform)
        raise NotImplementedError
