"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
from ..core import register
from .vit_tiny import VisionTransformer
from .dinov3 import DinoVisionTransformer


class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16, in_chans=3):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(in_chans, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4


@register()
class DINOv3STAs(nn.Module):
    def __init__(
        self,
        name=None,
        weights_path=None,
        interaction_indexes=[],
        finetune=True,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        use_sta=True,
        conv_inplane=16,
        hidden_dim=None,
        in_chans=3,
        input_norm='none',
        input_norm_mean=0.45,
        input_norm_std=0.225,
    ):
        super(DINOv3STAs, self).__init__()
        if 'dinov3' in name:
            self.dinov3 = DinoVisionTransformer(name=name, in_chans=in_chans)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                sd = torch.load(weights_path)
                if in_chans != 3:
                    # expand the 3-channel patch projection to `in_chans` channels:
                    # new channels are initialized as the mean of the 3 RGB channels
                    # so the pretrained features are preserved as much as possible.
                    w = sd.get('patch_embed.proj.weight')
                    if w is not None and w.shape[1] == 3 and w.shape[1] != in_chans:
                        mean_w = w.mean(dim=1, keepdim=True)  # [D,1,K,K]
                        extra = mean_w.expand(-1, in_chans - 3, -1, -1)
                        sd['patch_embed.proj.weight'] = torch.cat([w, extra], dim=1)
                        print(f'[dinov3] expanded patch_embed proj 3 -> {in_chans} ch')
                self.dinov3.load_state_dict(sd)
            else:
                print('Training DINOv3 from scratch...')
        else:
            self.dinov3 =  VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                self.dinov3._model.load_state_dict(torch.load(weights_path))
            else:
                print('Training ViT-Tiny from scratch...')

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        # optional per-image illumination normalisation of the RGB channels
        self.input_norm = input_norm
        self.input_norm_mean = input_norm_mean
        self.input_norm_std = input_norm_std

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # init the feature pyramid
        self.use_sta = use_sta
        if use_sta:
            print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

    def _normalize_input(self, x):
        """Per-image illumination normalisation of the RGB channels.

        A night frame has a far lower mean/variance than the daytime statistics
        that the fixed ImageNet ``Normalize`` and the pretrained DINOv3 were
        built on, so the stem sees an out-of-distribution input. Re-standardising
        the first 3 channels per image removes the global brightness/contrast
        component before it reaches the backbone (aux IR/Depth, which are already
        illumination-invariant, are left untouched), then rescales back onto the
        ImageNet-ish statistics the pretrained weights expect.
        """
        rgb = x[:, :3]
        aux = x[:, 3:]
        m = rgb.mean(dim=(1, 2, 3), keepdim=True)
        s = rgb.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
        rgb = (rgb - m) / s * self.input_norm_std + self.input_norm_mean
        if aux.shape[1] == 0:
            return rgb
        return torch.cat([rgb, aux], dim=1)

    def forward(self, x):
        if self.input_norm == 'instance':
            x = self._normalize_input(x)
        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:    # repeat the same layer for all the three scales
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]
        
        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)

        # fusion
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4