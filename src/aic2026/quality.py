"""
Track 2: 质量评估模块训练
=========================
包含 Quality Head 的预文本任务（Stage 1）和联合训练（Stage 2）。
"""

from typing import Literal
import torch
import torch.nn as nn


class ModalityCorruptor:
    """模态损坏策略（用于 Quality Head 预文本任务）"""

    CORRUPTIONS = ["gaussian_noise", "zero_out", "bad_pixels", "contrast_compress"]

    @staticmethod
    def gaussian_noise(features: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        return features + torch.randn_like(features) * std

    @staticmethod
    def zero_out(features: torch.Tensor, ratio: float = 0.3) -> torch.Tensor:
        mask = torch.rand_like(features) < ratio
        features = features.clone()
        features[mask] = 0
        return features

    # TODO: Phase 2 — 补充 bad_pixels, contrast_compress


def pretext_training_step(model, batch, modality: Literal["rgb", "depth", "ir"]):
    """Quality Head 预文本任务训练一步

    TODO: Phase 2 实现
    """
    raise NotImplementedError


def joint_training_step(model, batch):
    """检测 + 质量评估联合训练一步

    TODO: Phase 2 实现
    """
    raise NotImplementedError
