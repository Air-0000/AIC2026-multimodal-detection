#!/usr/bin/env python3
"""生成 ViT-B @800 实验的训练配置 + 10 档评测配置。

三件事(用户要求):
  1. 更大 backbone : RGB 主干 dinov3_vits16plus(384) -> dinov3_vitb16(768)
                     辅助 IR/Depth 流留在 vits16plus(384), 用新的 aux_name 指定,
                     并由 ir_aux_proj/depth_aux_proj (1x1) 升维到 768 再融合。
  2. 更高分辨率     : 640 -> 800 (train/val Resize + base 的 eval_spatial_size)
  3. 更多数据增强   : 强光度 + 新增几何扰动 RandomAffine + 新增暗光退化
                     RandomRGBDegrade5ch (该算子原本在 6ch 下静默失效, 已修)

用法:
    python3 scripts/gen/gen_vitb800_configs.py
产物:
    vendor/DEIMv2/configs/deimv2/deimv2_dinov3_l_aic_vitb800_rdttrack_deepir_aug.yml
    vendor/DEIMv2/configs/deimv2/deimv2_dinov3_l_aic_vitb800_rdttrack_deepir_aug_val_{orig,nightXX}.yml
"""
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
CFG_DIR = os.path.join(ROOT, "vendor/DEIMv2/configs/deimv2")
NAME = "deimv2_dinov3_l_aic_vitb800_rdttrack_deepir_aug"

LEVELS = ["night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]

HEADER = """__include__: [
  '../dataset/aic_detection.yml',
  '../runtime.yml',
  '../base/dataloader.yml',
  '../base/optimizer.yml',
  '../base/deimv2.yml',
]

# =============================================================================
# ViT-B @800 + 增强  (基线: ..._ufall_x_deepir_midboost_l4u3, AP=0.5372 @640/ViT-S)
# -----------------------------------------------------------------------------
# 三处改动, 其余尽量与基线保持一致以便归因:
#   1) backbone: RGB 主干 dinov3_vits16plus(384-dim) -> dinov3_vitb16(768-dim)。
#      权重来自官方 facebook/dinov3-vitb16-pretrain-lvd1689m (gated, 用镜像
#      Fanqi-Lin-IR/dinov3_vitb16), 超参从 ckpt 反推: 768/12层/12头/ffn_ratio4/
#      ffn=mlp/n_storage=4, 已注册进 vision_transformer.py 的 'dinov3_vitb16'。
#      辅助 IR/Depth 流保持 vits16plus(384) —— 它们本来就是用 RGB 权重通道复制
#      初始化的(不是红外预训练), 放大只会白增可训练参数; 用 aux_name 指定并由
#      1x1 proj 升维到 768 后进入正交融合。
#   2) 分辨率: 640 -> 800 (以下三处必须一起改, 否则训练/验证尺度不一致)。
#   3) 增强: 光度幅度加大 + 新增几何扰动 RandomAffine + 新增暗光退化
#      RandomRGBDegrade5ch(此算子此前在 6ch 下因通道守卫 (3,4,5) 静默失效)。
#      注意: 不启用 IlluminationNorm —— 它会把暗帧重新拉回白天亮度, 正好抵消
#      夜间增强的效果(该算子是给"零重训探测"用的, 不是训练期增强)。
#      不启用 Mosaic / 不加大 mixup(按用户选择)。
#
# 显存: 800 + ViT-B 远比基线重。micro=2(accum5 -> 有效bs10)。
#   若 OOM: 先把 accum_steps 调到 10 并把 total_batch_size 降到 2(micro=1);
#   再不行就 use_modality_stem: True (辅助流退化成 3-conv 小 stem, 省一大截)。
# =============================================================================
output_dir: {output_dir}

# 必须覆盖 base/deimv2.yml 的 eval_spatial_size (默认 640)
eval_spatial_size: [800, 800]

freeze_bn: false
accum_steps: 5          # micro=2, accum=5 -> 有效 batch 10 (与基线一致)
use_ema: True

DEIM:
  backbone: DINOv3STAsRDTTrack

DINOv3STAsRDTTrack:
  name: dinov3_vitb16          # <- 更大 backbone (768-dim)
  aux_name: dinov3_vits16plus  # <- 辅助流保持 384-dim, 由 1x1 proj 升维
  weights_path: null           # 由 -t ckpts/dinov3/vitb16_rdttrack_init.pth 热启动
  interaction_indexes: [5, 8, 11]
  conv_inplane: 32
  hidden_dim: 256
  ir_drop_prob: 0.15
  depth_drop_prob: 0.15
  depth_in_chans: 1
  ir_in_chans: 2
  unfreeze_rgb_stages: 3
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
      params: '^(?=.*.dinov3)(?!.*(?:ir_dinov3|depth_dinov3))(?!.*(?:norm|bn|bias)).*$'
      lr: 0.0001
    -
      params: '^(?=.*.dinov3)(?!.*(?:ir_dinov3|depth_dinov3))(?=.*(?:norm|bn|bias)).*$'
      lr: 0.0001
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
        - {{type: Resize, size: [800, 800]}}
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
  total_batch_size: 10

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
        - {{type: Resize, size: [800, 800]}}
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


def render(output_dir, val_img):
    return HEADER.format(root=ROOT, output_dir=output_dir, val_img=val_img)


def main():
    os.makedirs(CFG_DIR, exist_ok=True)

    # ---- 训练配置 ----
    train_path = os.path.join(CFG_DIR, NAME + ".yml")
    with open(train_path, "w") as f:
        f.write(render(output_dir="./outputs/" + NAME,
                       val_img=f"{ROOT}/work_dir/deimv2_dataset/valid/images"))
    print("train ->", train_path)

    # ---- 评测配置 (orig + 9 档暗光) ----
    for lv in ["orig"] + LEVELS:
        out_dir = os.path.join(CFG_DIR, f"{lv}_out")
        if lv == "orig":
            val_img = f"{ROOT}/work_dir/deimv2_dataset/valid/images"
        else:
            val_img = f"{ROOT}/data_aux/aic_valid_rgb_{lv}"
        p = os.path.join(CFG_DIR, f"{NAME}_val_{lv}.yml")
        with open(p, "w") as f:
            f.write(render(output_dir=out_dir, val_img=val_img))
        print("val   ->", p)


if __name__ == "__main__":
    main()
