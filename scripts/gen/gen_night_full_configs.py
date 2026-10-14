import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
CFG = os.path.join(ROOT, "vendor/DEIMv2/configs/deimv2")
ANN = f"{ROOT}/work_dir/deimv2_dataset/valid/annotations/instances_val.json"

# (level, 压暗RGB目录)
LEVELS = [
    ("night50", "aic_valid_rgb_night50"),
    ("night25", "aic_valid_rgb_night25"),
    ("night12", "aic_valid_rgb_night12"),
    ("night05", "aic_valid_rgb_night05"),
    ("night01", "aic_valid_rgb_night01"),
    ("night005", "aic_valid_rgb_night005"),
    ("night002", "aic_valid_rgb_night002"),
    ("night001", "aic_valid_rgb_night001"),
    ("night0005", "aic_valid_rgb_night0005"),
]

TEMPLATE = '''__include__: [
  "../dataset/aic_detection.yml",
  "../runtime.yml",
  "../base/dataloader.yml",
  "../base/optimizer.yml",
  "../base/deimv2.yml",
]

output_dir: ./outputs/{out}__{level}

DEIM:
  backbone: DINOv3STAs

DINOv3STAs:
  name: dinov3_vits16
  weights_path: ./ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
  interaction_indexes: [5, 8, 11]
  finetune: True
  conv_inplane: 32
  hidden_dim: 224
{inorm}

HybridEncoder:
  in_channels: [224, 224, 224]
  hidden_dim: 224
  dim_feedforward: 896

DEIMTransformer:
  feat_channels: [224, 224, 224]
  hidden_dim: 224
  num_layers: 4
  eval_idx: -1
  dim_feedforward: 1792

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {root}/work_dir/{rgbdir}
    ann_file: {ann}
    return_masks: False
    transforms:
      type: Compose
      ops:
        - {{type: Resize, size: [640, 640]}}
        - {{type: ConvertPILImage, dtype: "float32", scale: True}}
  shuffle: False
  num_workers: 2
  drop_last: False
  total_batch_size: 1

DEIMCriterion:
  matcher:
    matcher_change_epoch: 50
'''

# (tag, config stem, output_dir prefix, use_input_norm)
MODELS = [
    ("rgb3ch", "deimv2_dinov3_l_aic_val_rgb", "outputs/deimv2_dinov3_l_aic__rgb", False),
    ("nightsim", "deimv2_dinov3_l_aic_nightsim_inorm_val", "outputs/deimv2_dinov3_l_aic_nightsim_inorm", True),
    ("fused", "deimv2_dinov3_l_aic_fused_tardal_val", "outputs/deimv2_dinov3_l_aic_fused_tardal", False),
]

made = 0
for tag, stem, out, inorm in MODELS:
    for level, rgbdir in LEVELS:
        fn = f"{stem}_{level}.yml"
        path = os.path.join(CFG, fn)
        if os.path.exists(path):
            print("skip", fn)
            continue
        inl = "  input_norm: instance" if inorm else ""
        txt = TEMPLATE.format(out=out, level=level, inorm=inl, root=ROOT, rgbdir=rgbdir, ann=ANN)
        open(path, "w").write(txt)
        made += 1
        print("write", fn)
print("made", made)
