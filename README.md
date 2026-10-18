# AIC2026 城市场景视觉多模态目标检测

> **Multimodal Urban Object Detection (Visible + Infrared + Depth)**  
> 面向城市场景下的可见光、红外与深度三模态 12 类目标检测系统，涵盖数据配准、多尺度训练、测试时增强 (TTA) 与框级融合 (WBF)。

---

## 📌 项目概述

本项目为 AIC（AI City）城市场景视觉多模态目标检测竞赛完整工程。任务基于可见光（RGB）、红外（Infrared）与深度（Depth）三种模态图像，精准检测 12 类典型城市实体目标，并输出符合官方评测格式的检测结果。

### 12 类检测目标
`person`, `bicycle`, `car`, `tricycle`, `ball`, `seat`, `boat`, `uav`, `sign`, `light`, `animal`, `garbage can`

---

## 🏗️ 系统架构与流水线

整个检测流水线由数据预处理、双主干模型、多模态特征融合与后处理集成构成：

```
                    ┌── Visible (RGB) ──────┐
Input Modalities ───┼── Infrared (IR) ──────┼──► [ Multimodal Alignment & Preprocessing ]
                    └── Depth (D) ──────────┘                    │
                                                                 ▼
                              ┌──────────────────────────────────┴──────────────────────────────────┐
                              ▼                                                                     ▼
                   [ RF-DETR RGB Baseline ]                                              [ DEIMv2 Primary Model ]
                 (DINOv3 ViT-L/16 Backbone)                                            (HGNetv2-X / D-FINE Decoder)
                              │                                                                     │
                              ▼                                                                     ▼
                   [ Single-Modal Detection ]                                            [ RGB+Heat 4-Channel Fusion ]
                              │                                                                     │
                              └──────────────────────────────────┬──────────────────────────────────┘
                                                                 ▼
                                                  [ Test-Time Augmentation (TTA) ]
                                                    (Multi-scale / Flip-TTA)
                                                                 │
                                                                 ▼
                                                [ Weighted Boxes Fusion (WBF) ]
                                                                 │
                                                                 ▼
                                              [ Safe Formatting & Gate Check ]
                                                                 │
                                                                 ▼
                                              Official Submission (`.txt` per image)
```

### 核心模块实现
1. **RF-DETR 基线检测器**：集成 Meta DINOv3 ViT-L/16 预训练特征主干，构建高精度基准检测能力。
2. **DEIMv2 主干模型**：基于实时 DETR 改进架构，采用 HGNetv2-X 与 HybridEncoder，具备强大的小目标和密集目标判别能力。
3. **红外通道融合 (`prep_ir_heat.py`)**：将配准后的红外热力信息构建为附加特征输入通道，实现端侧多模态互补表征。
4. **测试时增强 (TTA, `tta_predict.py`)**：在推理阶段引入多尺度与水平翻转增强，大幅提高复杂光照与遮挡工况下的召回率。
5. **加权框融合 (WBF, `wbf_fuse.py`)**：对多模型与多阈值候选框进行置信度加权融合，滤除重叠误检，优化边界框坐标。

---

## 📂 仓库结构

```
├── check_submission.py       # 官方提交格式校验网关（检查列数、范围、CRLF等）
├── eval_submission.py        # 验证集 mAP 评测脚本
├── eval_threshold.py         # 置信度阈值扫描与最佳 F1/mAP 搜索
├── predict_deimv2.py         # DEIMv2 推理产出脚本
├── predict_rgb.py            # RF-DETR RGB 推理脚本
├── predict_tta.py            # TTA 推理与缓存脚本
├── prep_ir_heat.py           # 红外特征提取与四通道数据集构建
├── prep_depth.py             # 深度模态归一化与数据对齐
├── tta_predict.py            # 增强推理与预测产物生成
├── wbf_fuse.py               # 加权框融合（Weighted Boxes Fusion）后处理
├── yolo_to_coco.py           # YOLO / COCO 格式标注互相转换
├── vendor/                   # 包含 DEIMv2 引擎及模型定义实现
└── 预测结果格式范例.zip        # 官方提交示例与规范参考
```

---

## 🚀 快速开始

### 1. 数据准备
将竞赛提供的多模态数据集存放至 `dataset/` 目录：
```
dataset/
├── train/
│   ├── visible/
│   ├── infrared/
│   ├── depth/
│   └── labels/        # YOLO 格式标注: class_id cx cy w h
└── test/
    ├── visible/
    ├── infrared/
    └── depth/
```

执行数据软链接与验证集切分：
```bash
python prepare_dataset.py
```

### 2. 生成多模态融合输入
```bash
# 生成 RGB + 红外热力四通道数据集
python prep_ir_heat.py --src dataset/train --dst work_dir/deimv2_heat
```

### 3. 模型推理与 TTA 增强
```bash
# 执行 DEIMv2 推理
python predict_deimv2.py --config configs/deimv2.yml --weight weights/best.pth --out work_dir/pred_deimv2

# 执行带测试时增强 (TTA) 的推理
python tta_predict.py --input dataset/test --out work_dir/pred_tta
```

### 4. 框级融合 (WBF)
```bash
python wbf_fuse.py --inputs work_dir/pred_deimv2 work_dir/pred_tta --output work_dir/submission
```

### 5. 提交格式合规校验
在打包上传前，必须执行校验脚本保证零格式错误：
```bash
python check_submission.py work_dir/submission dataset/test/visible
```
校验通过后生成最终压缩包提交。

---

## 📋 提交格式规范

每张测试图片对应一个同名 `.txt` 文件，每行包含一个预测框（按空格分隔，CRLF 换行，每图最多 100 个框）：
```
<class_id> <cx> <cy> <w> <h> <confidence>
```
* `class_id`：整数，范围 `0 ~ 11`
* `cx, cy, w, h`：归一化中心点坐标与宽高，范围 `[0.0, 1.0]`
* `confidence`：预测置信度，范围 `[0.0, 1.0]`
