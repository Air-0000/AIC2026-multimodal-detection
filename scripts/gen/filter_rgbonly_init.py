#!/usr/bin/env python
"""把 vitl16_rdttrack_scratch.pth 收敛成 "只含官方 RGB 主干" 的版本。

为什么必须这样做
----------------
aux 流用的注册名是 `dinov3_vits16plus`, 其 ffn_layer 是 **swiglu**
(blocks.N.mlp.w1/w2/w3), 而本地能拿到的干净 ViT-S 权重是 `dinov3_vits16`
(普通 MLP: fc1/fc2)。二者结构不兼容:

    Load model.state_dict, {'missed': ['backbone.ir_dinov3.blocks.0.mlp.w1.weight', ...]}

结果是一个半吊子混杂态 —— patch_embed / attn 用了 vits16 的值, SwiGLU mlp 全部随机。
**这比全随机更糟**: patch_embed 与前层输出的统计量不匹配, 反而给后面的 block 喂噪声。

而 DinoVisionTransformer.__init__ 只从 configs 里读 `pretrained` / `weights` 这两个
字段, **并不会真的去下载或加载权重**(真正加载只发生在 multimodal_backbones.py 里
`weights_path` 非空时)。所以不给 aux 权重文件时, aux 就是干净的全随机初始化。

结论: 诚实的选择是让 aux 主干全随机 —— aux 本来也**没有**红外预训练权重可用,
从头学正是 "一次性训练" 的定义。Stage A (RGB-only) 里 aux 甚至收不到梯度,
Stage B 才轮到它。
"""
import os
import sys

import torch

DEIM = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/d"
SRC = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/ckpts/dinov3/vitl16_rdttrack_scratch.pth"
DST = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/ckpts/dinov3/vitl16_rdttrack_scratch_rgbonly.pth"


def strip(ck):
    for k in ("model", "state_dict"):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict) and len(ck[k]):
            return strip(ck[k])
    if isinstance(ck, dict) and "ema" in ck and isinstance(ck["ema"], dict):
        e = ck["ema"]
        return strip(e.get("module", e))
    return ck


def main():
    sd = strip(torch.load(SRC, map_location="cpu"))
    keep = {k: v for k, v in sd.items() if k.startswith("backbone.rgb_dinov3.")}
    dropped = len(sd) - len(keep)
    torch.save({"model": keep}, DST)
    n = sum(v.numel() for v in keep.values())
    print(f"source : {SRC}")
    print(f"  total keys={len(sd)}  ->  kept {len(keep)} (dropped {dropped} aux keys)")
    print(f"saved  : {DST}")
    print(f"  keys={len(keep)}  params={n/1e6:.1f}M")
    print("\n加载时预期:")
    print("  rgb_dinov3        -> 全部命中 (官方 DINOv3 ViT-L 预训练)")
    print("  ir/depth_dinov3   -> 全随机初始化 (干净, 无半吊子混杂)")
    print("  encoder/decoder/convs/prompt/sta -> 全随机初始化")


if __name__ == "__main__":
    main()
