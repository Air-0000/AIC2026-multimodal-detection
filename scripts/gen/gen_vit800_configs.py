#!/usr/bin/env python3
"""生成「大 backbone @高分辨率 + 增强」实验的训练配置 + 10 档评测配置。

参数化的两个维度(其余保持一致以便归因):
  --backbone vitb16 | vitl16    RGB 主干 (辅助 IR/Depth 流始终留在 vits16plus)
  --size     800 (可改)          训练/验证分辨率

三件事(用户要求):
  1. 更大 backbone : RGB 主干 dinov3_vits16plus(384) -> vitb16(768) / vitl16(1024)
                     辅助 IR/Depth 流留在 vits16plus(384), 用 aux_name 指定,
                     并由 ir_aux_proj/depth_aux_proj (1x1) 升维到主 embed_dim。
  2. 更高分辨率     : 640 -> --size (train/val Resize + base 的 eval_spatial_size)
  3. 更多数据增强   : 强光度 + 几何扰动 RandomAffine + 暗光退化 RandomRGBDegrade5ch
                     (该算子原本在 6ch 下因通道守卫 (3,4,5) 静默失效, 已修)

用法:
    python3 scripts/gen/gen_vit800_configs.py                       # ViT-B @800
    python3 scripts/gen/gen_vit800_configs.py --backbone vitl16     # ViT-L @800
产物:
    vendor/DEIMv2/configs/deimv2/deimv2_dinov3_l_aic_{tag}{size}_rdttrack_deepir_aug.yml
    ..._val_{orig,nightXX}.yml
"""
import argparse
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
CFG_DIR = os.path.join(ROOT, "vendor/DEIMv2/configs/deimv2")

LEVELS = ["night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]

# 主 backbone 规格。辅助流固定 vits16plus(384)。
# interaction_indexes 与 unfreeze_rgb_stages 按「层数的比例」对齐 12 层的 [5,8,11] / 3,
# 否则 24 层的 ViT-L 只会在前半段抽特征、解冻比例也腰斩。
SPECS = {
    "vitb16": dict(tag="vitb", embed=768, depth=12, heads=12,
                   interaction="[5, 8, 11]", unfreeze=3,
                   accum=5, tbs=10, ckpt="ckpts/dinov3/vitb16_rdttrack_init.pth"),
    "vitl16": dict(tag="vitl", embed=1024, depth=24, heads=16,
                   interaction="[11, 17, 23]", unfreeze=6,
                   accum=5, tbs=10, ckpt="ckpts/dinov3/vitl16_rdttrack_init.pth"),
}

HEADER = """__include__: [
  '../dataset/aic_detection.yml',
  '../runtime.yml',
  '../base/dataloader.yml',
  '../base/optimizer.yml',
  '../base/deimv2.yml',
]

# =============================================================================
# {tag} @{size} + 增强  (基线: ..._ufall_x_deepir_midboost_l4u3, AP=0.5372 @640/ViT-S)
# -----------------------------------------------------------------------------
# 三处改动, 其余尽量与基线保持一致以便归因:
#   1) backbone: RGB 主干 dinov3_vits16plus(384-dim) -> dinov3_{name}({embed}-dim,
#      {depth}层/{heads}头)。权重来自官方 facebook/dinov3-{name}-pretrain-lvd1689m
#      (gated, 走未受限镜像 Fanqi-Lin-IR/dinov3_{name}), 超参从 ckpt 反推校验,
#      已注册进 vision_transformer.py 的 'dinov3_{name}'。
#      interaction_indexes={interaction} / unfreeze_rgb_stages={unfreeze}: 按
#      {depth} 层的比例对齐 12 层 ViT 的 [5,8,11]/3 —— 若直接沿用小模型数值,
#      24 层的 ViT-L 只会用前半段抽特征, 解冻比例也会腰斩。
#      辅助 IR/Depth 流保持 vits16plus(384) —— 它们本来就是用 RGB 权重通道复制
#      初始化的(不是红外预训练), 放大只会白增可训练参数; 用 aux_name 指定并由
#      1x1 proj 升维到 {embed} 后进入正交融合。
#   2) 分辨率: 640 -> {size} (以下三处必须一起改, 否则训练/验证尺度不一致)。
#   3) 增强: 光度幅度加大 + 新增几何扰动 RandomAffine + 新增暗光退化
#      RandomRGBDegrade5ch(此算子此前在 6ch 下因通道守卫 (3,4,5) 静默失效)。
#      注意: 不启用 IlluminationNorm —— 它会把暗帧重新拉回白天亮度, 正好抵消
#      夜间增强的效果(该算子是给"零重训探测"用的, 不是训练期增强)。
#      不启用 Mosaic / 不加大 mixup(按用户选择)。
#
# 显存: micro = total_batch_size/accum = {tbs}/{accum} = {micro}。
#   实测峰值 (RTX 5060 Ti 16.3 GiB, 桌面已占 ~2.7 GiB, 最坏尺度 {size}x{size}):
#     vitb800:  micro=4 -> 10.23 GiB (可用)   micro=6 -> OOM
#     vitl800:  micro=1 ->  5.94 GiB
#               micro=2 ->  8.56 GiB  <-- 本配置采用
#               micro=3 -> 11.29 GiB (余量仅 2.9 GiB, 偏紧)
#               micro=4 -> OOM
#   数字取自 scripts/train/probe_mem.py (真实训练 batch, bf16 autocast + GradScaler
#   + EMA, 对齐 det_engine.py 的训练路径)。注意必须加 --max-scale: 配置开了多尺度
#   训练, 尺寸是随机抽的 (480..800), 峰值随尺寸平方变化, 抽到 736 会低估 18%。
#   若 OOM, 按顺序退:
#     a) accum 翻倍 (micro 减半 —— 这是唯一能降峰值的旋钮)
#     b) use_modality_stem: True (辅助流退化成 3-conv 小 stem, 省一大截)
#     c) 分辨率降到 640
#     d) 换回 vitb16 (python3 scripts/gen/gen_vit800_configs.py --backbone vitb16)
# =============================================================================
output_dir: {output_dir}

# 必须覆盖 base/deimv2.yml 的 eval_spatial_size (默认 640)
eval_spatial_size: [{size}, {size}]

freeze_bn: false
accum_steps: {accum}          # 有效 batch = {tbs}
use_ema: True

DEIM:
  backbone: DINOv3STAsRDTTrack

DINOv3STAsRDTTrack:
  name: dinov3_{name}          # <- 更大 backbone ({embed}-dim)
  aux_name: dinov3_vits16plus  # <- 辅助流保持 384-dim, 由 1x1 proj 升维
  weights_path: null           # 由 -t {ckpt} 热启动
  interaction_indexes: {interaction}
  conv_inplane: 32
  hidden_dim: 256
  ir_drop_prob: 0.15
  depth_drop_prob: 0.15
  depth_in_chans: 1
  ir_in_chans: 2
  unfreeze_rgb_stages: {unfreeze}
  input_norm: instance

HybridEncoder:
  in_channels: [256, 256, 256]
  hidden_dim: 256
  dim_feedforward: 1024
  expansion: 1.25
  depth_mult: 1.37

DEIMTransformer:
  feat_channels: [256, 256, 256]
  hidden_dim: 256
  num_layers: 6
  eval_idx: -1
  dim_feedforward: 2048

epoches: 50
lrsheduler: flatcosine
lr_gamma: 0.5
warmup_iter: 400
flat_epoch: 25
no_aug_epoch: 3

optimizer:
  type: AdamW
  params:
    -
      # RGB 主干走 LLRD(层级衰减): 解冻的 block 各自带 llrd_lr 属性, 由
      # engine/core/yaml_config.py::get_optim_params 拆成独立组覆盖这里的 lr。
      # 若某 RGB 参数没被 LLRD 覆盖(不应发生), 也用保守的 2e-5 而非旧版 1e-4
      # —— 后者对 1800 张图上的 ViT-L 太激进, 会打散预训练先验。
      params: '^(?=.*.dinov3)(?!.*(?:ir_dinov3|depth_dinov3))(?!.*(?:norm|bn|bias)).*$'
      lr: 0.00002
    -
      params: '^(?=.*.dinov3)(?!.*(?:ir_dinov3|depth_dinov3))(?=.*(?:norm|bn|bias)).*$'
      lr: 0.00002
      weight_decay: 0.
    -
      params: '^(?=.*(?:ir_dinov3|depth_dinov3))(?!.*(?:norm|bn|bias)).*$'
      lr: 0.00015
    -
      params: '^(?=.*(?:ir_dinov3|depth_dinov3))(?=.*(?:norm|bn|bias)).*$'
      lr: 0.00015
      weight_decay: 0.
    -
      params: '^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.
  lr: 0.0003
  betas: [0.9, 0.999]
  weight_decay: 0.000125

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetectionMultiModalCached
    cache_img_folder: {root}/work_dir/deepir_cache/train
    img_folder: {root}/work_dir/deimv2_dataset/train/images
    ir_folder: {root}/work_dir/deimv2_heat/images
    raw_ir_folder: {root}/work_dir/deimv2_ir/images
    depth_folder: {root}/dataset/train/depth
    ann_file: {root}/work_dir/deimv2_dataset/train/annotations/instances_train.json
    n_channels: 6
    return_masks: False
    transforms:
      type: Compose
      ops:
        # --- 光度: 幅度较基线加大 (0.6-1.4 -> 0.5-1.5 等) ---
        - {{type: RandomPhotometricDistort5ch, p: 0.8, brightness: [0.5, 1.5], contrast: [0.4, 1.6], saturation: [0.4, 1.6], hue: [-0.15, 0.15]}}
        - {{type: RandomZoomOut, fill: 0}}
        - {{type: RandomIoUCrop, p: 0.8}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: RandomHorizontalFlip}}
        # --- 新增几何扰动: 旋转/平移/缩放 (纯空间操作, 6ch 全通道一致) ---
        - {{type: RandomAffine, degrees: 10, translate: [0.1, 0.1], scale: [0.9, 1.1], fill: 0}}
        - {{type: Resize, size: [{size}, {size}]}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        # --- 新增暗光退化 (此前在 6ch 下静默失效, 已修通道守卫) ---
        - {{type: RandomRGBDegrade5ch, p: 0.35, brightness: [0.05, 0.60], contrast: [0.30, 0.90], noise_std: [0.02, 0.12], log_brightness: true}}
        # --- 基线原有的 midboost 中段过采样夜间模拟 ---
        - {{type: RandomNight5ch, p: 0.5, level: [0.005, 0.45], photon_max: 70.0, read_sigma: 0.012, log_level: true, lamps: [1, 4]}}
        - {{type: Normalize, mean: [0.485, 0.456, 0.406, 0.5, 0.5, 0.5], std: [0.229, 0.224, 0.225, 0.5, 0.5, 0.5]}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
  shuffle: True
  num_workers: 8
  drop_last: True
  total_batch_size: {tbs}

  collate_fn:
    type: BatchImageCollateFunction
    mixup_epochs: [4, 34]
    stop_epoch: 60
    copyblend_epochs: [4, 60]
    base_size_repeat: 1

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetectionMultiModal
    img_folder: {val_img}
    ir_folder: {root}/work_dir/deimv2_heat/valid
    raw_ir_folder: {root}/work_dir/deimv2_ir/valid
    depth_folder: {root}/dataset/train/depth
    ann_file: {root}/work_dir/deimv2_dataset/valid/annotations/instances_val.json
    n_channels: 6
    return_masks: False
    transforms:
      type: Compose
      ops:
        - {{type: Resize, size: [{size}, {size}]}}
        - {{type: Normalize, mean: [0.485, 0.456, 0.406, 0.5, 0.5, 0.5], std: [0.229, 0.224, 0.225, 0.5, 0.5, 0.5]}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
  shuffle: False
  num_workers: 4
  drop_last: False
  total_batch_size: 4

DEIMCriterion:
  matcher:
    matcher_change_epoch: 50
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vitb16", choices=sorted(SPECS))
    ap.add_argument("--size", type=int, default=800)
    args = ap.parse_args()

    sp = SPECS[args.backbone]
    tag, size = sp["tag"], args.size
    name = f"deimv2_dinov3_l_aic_{tag}{size}_rdttrack_deepir_aug"
    fields = dict(sp, name=args.backbone, size=size,
                  micro=sp["tbs"] // sp["accum"])

    def render(output_dir, val_img):
        return HEADER.format(root=ROOT, output_dir=output_dir, val_img=val_img,
                            **fields)

    os.makedirs(CFG_DIR, exist_ok=True)

    train_path = os.path.join(CFG_DIR, name + ".yml")
    with open(train_path, "w") as f:
        f.write(render(output_dir="./outputs/" + name,
                       val_img=f"{ROOT}/work_dir/deimv2_dataset/valid/images"))
    print("train ->", train_path)

    for lv in ["orig"] + LEVELS:
        # output_dir 必须带实验名: 否则 vitb800 / vitl800 两套评测会写进同一个
        # 目录互相覆盖 (早期版本用的是 CFG_DIR/{lv}_out, 与实验名无关)。
        out_dir = f"./outputs/{name}_eval/{lv}"
        val_img = (f"{ROOT}/work_dir/deimv2_dataset/valid/images" if lv == "orig"
                   else f"{ROOT}/data_aux/aic_valid_rgb_{lv}")
        p = os.path.join(CFG_DIR, f"{name}_val_{lv}.yml")
        with open(p, "w") as f:
            f.write(render(output_dir=out_dir, val_img=val_img))
        print("val   ->", p)


if __name__ == "__main__":
    main()
