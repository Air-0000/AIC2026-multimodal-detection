#!/usr/bin/env python
"""rgbonly 训练配置全链路冒烟: YAMLConfig -> model 构建 -> optimizer 分组."""
import sys

sys.path.insert(0, "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")

from engine.core.yaml_config import YAMLConfig  # noqa: E402

CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2/deimv2_dinov3_l_aic_vitl16_l4like_640_rgbonly.yml"

cfg = YAMLConfig(CFG)
m = cfg.model
n_all = sum(p.numel() for p in m.parameters()) / 1e6
n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
bb = m.backbone
assert getattr(bb, "rgb_only", False), "backbone.rgb_only 未生效!"
print(f"[OK] model={type(m).__name__}, backbone.rgb_only=True")
print(f"[OK] whole model: total {n_all:.1f}M, trainable {n_tr:.1f}M")

opt = cfg.optimizer
groups = [(len(g["params"]), g.get("lr"), g.get("initial_lr")) for g in opt.param_groups]
print(f"[OK] optimizer groups ({len(groups)}):")
for i, (n, lr, ilr) in enumerate(groups):
    print(f"   g{i}: n={n} lr={lr} initial_lr={ilr}")
assert all(n > 0 for n, _, _ in groups), "存在空参数组!"
print("CFG_SMOKE_OK")
