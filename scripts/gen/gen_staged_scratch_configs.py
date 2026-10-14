#!/usr/bin/env python
"""从 vitl800_clean.yml 派生三阶段 (curriculum) 训练配置。

原则: 一次性训大 backbone 时, 让模型"先学会一件事再学下一件", 而不是三流 + 融合
一起随机起步互相干扰。

  Stage A  RGB-only         : aux 恒为零 -> encoder/decoder/RGB 末层先学会检测
  Stage B  模态注入, 冻 RGB  : 放开 IR/Depth 及其注入通路, RGB 全冻结 -> 学会用多模态补盲
  Stage C  联合微调         : 全开低 lr -> end-to-end 收敛融合

Stage A 为什么能"零改动"实现纯 RGB:
  multimodal_backbones.py forward 里
      ir_feats = [torch.zeros_like(f) for f in ir_feats]
  用 zeros_like 造的是**全新张量**, 与 ir_dinov3 的计算图断开 -> aux 主干收不到任何
  梯度。信息论上网络此时只能依赖 RGB, 等价于纯 RGB 训练, 无需改模型代码。
"""
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
CFGDIR = os.path.join(ROOT, "vendor/DEIMv2/configs/deimv2")
SRC = os.path.join(CFGDIR, "deimv2_dinov3_l_aic_vitl800_clean.yml")

# 干净 warm-start 权重: 三条流全部是官方 DINOv3 预训练 (rgb=ViT-L, aux=ViT-S+),
# encoder/decoder/融合通路为空 -> 随机初始化。绝不使用 AIC 上训练过的权重。
CKPT = "ckpts/dinov3/vitl16_rdttrack_scratch_full.pth"

STAGES = {
    "A": dict(
        desc="Stage A / RGB-only detector",
        out="vitl800_scratch_stageA_rgb",
        ir_drop="1.0", depth_drop="1.0",
        unfreeze="6", llrd="1.0e-4", epoches="40", flat_epoch="20",
    ),
    "B": dict(
        desc="Stage B / modality injection with RGB frozen",
        out="vitl800_scratch_stageB_mod",
        ir_drop="0.15", depth_drop="0.15",
        unfreeze="0", llrd="1.0e-4", epoches="20", flat_epoch="10",
    ),
    "C": dict(
        desc="Stage C / joint fine-tune",
        out="vitl800_scratch_stageC_fuse",
        ir_drop="0.15", depth_drop="0.15",
        unfreeze="6", llrd="2.0e-5", epoches="40", flat_epoch="20",
    ),
}


def rewrite(text, cfg):
    """行级替换, 保留原文件注释与结构。"""
    out_lines = []
    touched = set()
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        indent = ln[: len(ln) - len(ln.lstrip())]

        if s.startswith("output_dir:") and "output_dir" not in touched:
            out_lines.append(f"{indent}output_dir: ./outputs/{cfg['out']}")
            touched.add("output_dir")
            continue
        if s.startswith("epoches:"):
            out_lines.append(f"{indent}epoches: {cfg['epoches']}")
            touched.add("epoches")
            continue
        if s.startswith("flat_epoch:"):
            out_lines.append(f"{indent}flat_epoch: {cfg['flat_epoch']}")
            touched.add("flat_epoch")
            continue
        if s.startswith("ir_drop_prob:"):
            out_lines.append(f"{indent}ir_drop_prob: {cfg['ir_drop']}")
            touched.add("ir_drop")
            continue
        if s.startswith("depth_drop_prob:"):
            out_lines.append(f"{indent}depth_drop_prob: {cfg['depth_drop']}")
            touched.add("depth_drop")
            continue
        if s.startswith("weights_path:") and "weights_path" not in touched:
            out_lines.append(f"{indent}weights_path: null           "
                             f"# aux/rgb 官方权重由 -t {CKPT} 一次性加载")
            touched.add("weights_path")
            continue
        if s.startswith("unfreeze_rgb_stages:"):
            out_lines.append(f"{indent}unfreeze_rgb_stages: {cfg['unfreeze']}")
            out_lines.append(f"{indent}# the new configurable LLRD knobs "
                             f"(see multimodal_backbones.py)")
            out_lines.append(f"{indent}llrd_top_lr: {cfg['llrd']}")
            out_lines.append(f"{indent}llrd_decay: 0.85")
            touched.add("unfreeze")
            continue
        out_lines.append(ln)

    missed = {"output_dir", "epoches", "ir_drop", "depth_drop",
              "unfreeze", "flat_epoch"} - touched
    return "\n".join(out_lines) + "\n", missed


def main():
    with open(SRC) as f:
        base = f.read()
    header = (
        "# =========================================================================\n"
        "# {desc}\n"
        "# 由 scripts/gen/gen_staged_scratch_configs.py 从 vitl800_clean.yml 派生\n"
        "# =========================================================================\n"
    )
    for tag, cfg in STAGES.items():
        body, missed = rewrite(base, cfg)
        dst = os.path.join(CFGDIR, f"deimv2_dinov3_l_aic_{cfg['out']}.yml")
        with open(dst, "w") as f:
            f.write(header.format(desc=cfg["desc"]))
            f.write(body)
        print(f"[{tag}] {os.path.basename(dst)}")
        print(f"      ir_drop={cfg['ir_drop']} depth_drop={cfg['depth_drop']} "
              f"unfreeze={cfg['unfreeze']} llrd_top_lr={cfg['llrd']} "
              f"epoches={cfg['epoches']}")
        if missed:
            print(f"      [warn] 未替换的字段: {sorted(missed)}")


if __name__ == "__main__":
    main()
