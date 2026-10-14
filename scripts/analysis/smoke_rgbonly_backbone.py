#!/usr/bin/env python
"""rgb_only 冒烟: 实例化 -> 断言无 aux 模块 -> CPU 前向 -> ckpt 键匹配统计."""
import sys

import torch

sys.path.insert(0, "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")

from engine.backbone.multimodal_backbones import DINOv3STAsRDTTrack  # noqa: E402

m = DINOv3STAsRDTTrack(
    name="dinov3_vitl16",
    interaction_indexes=[11, 17, 23],
    conv_inplane=32,
    hidden_dim=256,
    rgb_only=True,
    unfreeze_rgb_stages=2,
    llrd_top_lr=3e-5,
    llrd_decay=1.0,
    input_norm="instance",
)

keys = list(m.state_dict().keys())
bad = [k for k in keys if any(
    s in k for s in ("ir_dinov3", "depth_dinov3", "ir_norms", "depth_norms",
                     "dt_ort", "prompt", "ir_aux_proj", "depth_aux_proj"))]
assert not bad, f"aux 模块残留: {bad[:8]}"
print(f"[OK] state_dict 无任何 aux/prompt/dt_ort 键 (共 {len(keys)} 键)")

n_all = sum(p.numel() for p in m.parameters()) / 1e6
n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
print(f"[OK] params: total {n_all:.1f}M, trainable {n_tr:.1f}M")

m.eval()
with torch.no_grad():
    out = m(torch.randn(2, 6, 64, 64))
print(f"[OK] forward 6ch -> {[tuple(t.shape) for t in out]}")
assert out[0].shape[1] == 256

# ckpt 匹配统计 (模拟 load_tuning_state 的 _matched_state)
sd = torch.load("/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/ckpts/dinov3/vitl16_rdttrack_scratch_full.pth",
                map_location="cpu")["model"]
msd = m.state_dict()
matched = [k for k in sd if k.startswith("backbone.") and k[9:] in msd
           and sd[k].shape == msd[k[9:]].shape]
unmatched = [k for k in sd if k.startswith("backbone.") and (k[9:] not in msd)]
print(f"[OK] ckpt backbone keys={sum(1 for k in sd if k.startswith('backbone.'))}, "
      f"matched={len(matched)}, ignored(unexpected)={len(unmatched)}")
print("SMOKE_OK")
