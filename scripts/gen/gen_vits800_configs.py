#!/usr/bin/env python3
"""生成 vits800 的受控对照配置 (ViT-S @800 + 同增强)。

vits800 与 vitl800 的唯一差别是 backbone 规模:
  vitl800: RGB 主干 dinov3_vitl16 (1024-dim, 24 层)
  vits800: RGB 主干 dinov3_vits16plus (384-dim, 12 层)

其余 (800 分辨率 / 同光度+几何+暗光增强 / 同解冻比例 25% / 同 LLRD / 同 batch)
完全照搬 vitl800, 以便把 "换大 backbone" 隔离成单变量。

由于 ViT-S 是 12 层 (vitl 是 24 层), 两处必须按比例缩放, 否则会:
  - interaction_indexes 若沿用 [11,17,23] 会越界 / 只抽前半段
  - unfreeze_rgb_stages 若沿用 6 会解冻一半 (50%) 而非 25%

热启动: 直接继承 l4u3 的 best_stg1.pth (已是微调好的 ViT-S 检测器),
因为 vits800 主干就是 vits16plus, -t 走 _matched_state 会全量继承
(主干 + ir/depth + decoder), 起点 = 已训 ViT-S 检测器。
这正好和 vitl800 (官方 ViT-L 起点 + best_stg1 的 head) 构成对称对照:
两者同设置训 50 epoch, 唯一变量是 backbone 容量。

用法:
    python3 scripts/gen/gen_vits800_configs.py
"""
import glob
import os

BASE = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
TRAIN_SRC = f"{BASE}/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug.yml"
TRAIN_DST = f"{BASE}/deimv2_dinov3_l_aic_vits800_rdttrack_deepir_aug.yml"


def patch(text: str, is_train: bool) -> str:
    # 1) 主干: ViT-L -> ViT-S (只改 RGB 主干那一行, 不影响 aux_name)
    text = text.replace("name: dinov3_vitl16", "name: dinov3_vits16plus")
    # 2) 采样层: 24 层 [11,17,23] -> 12 层 [5,8,11] (按层数等比)
    text = text.replace(
        "interaction_indexes: [11, 17, 23]", "interaction_indexes: [5, 8, 11]"
    )
    # 3) 解冻比例保持 25%: 24 层解冻 6 -> 12 层解冻 3
    text = text.replace("unfreeze_rgb_stages: 6", "unfreeze_rgb_stages: 3")
    # 4) 所有 vitl800 字样 -> vits800 (output_dir / 注释 / 文件名引用)
    text = text.replace("vitl800", "vits800")
    if is_train:
        text = text.replace(
            "由 -t ckpts/dinov3/vitl16_rdttrack_init.pth 热启动",
            "由 -t l4u3 的 best_stg1.pth (ViT-S 已训检测器) 热启动",
        )
    return text


def main():
    # 训练配置
    t = open(TRAIN_SRC).read()
    open(TRAIN_DST, "w").write(patch(t, True))
    print(f"wrote {TRAIN_DST}")

    # 评测配置 (orig + 9 档暗光)
    n = 0
    for src in sorted(glob.glob(f"{BASE}/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_*.yml")):
        dst = src.replace("vitl800", "vits800")
        open(dst, "w").write(patch(open(src).read(), False))
        n += 1
    print(f"wrote {n} val configs")


if __name__ == "__main__":
    main()
