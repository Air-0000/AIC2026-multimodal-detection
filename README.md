# AIC 2026 — 面向城市场景的多模态目标检测

**AIC 2026 竞赛项目** — 基于 RGB + Depth + Infrared 三模态数据融合的视觉目标检测。

## 任务概述

- **任务**：RGB + Depth + Infrared 三模态融合目标检测（12 类）
- **数据**：训练集 2000 对，测试集 3000 对
- **评价指标**：mAP@50-95
- **场景**：城市场景多源异构数据

## 双 Track 策略

本项目采用 **两阶段推进** 策略：

### 🟢 Track 1: M2I2HA Baseline（当前阶段）

纯 M2I2HA 复现，RGB + IR 双模态检测。

| 组件 | 来源 | 状态 |
|------|------|------|
| 超图增强 (Intra-Hypergraph) | M2I2HA 官方代码 | ✅ 直接可用 |
| 跨模态融合 (Inter-Hypergraph) | M2I2HA 官方代码 | ✅ 直接可用 |
| 训练 Pipeline | M2I2HA Trainer | ✅ 直接可用 |
| AIC 数据集适配 | 自定义 YAML config | 🟡 需配置路径 |
| Baseline 训练 | `train_baseline.sh` | 🟡 需在服务器运行 |

### 🔵 Track 2: 三模态扩展（Baseline 后推进）

RGB + Depth + IR 三模态 + 质量评估模块。

| 组件 | 状态 |
|------|------|
| Depth 第三模态集成 | 🔲 待实现 |
| 三模态 Inter-Hypergraph Fusion | 🔲 待实现 |
| Uncertainty Head + Quality Head | 🔲 待实现 |
| Depth log 归一化预处理 | 🔲 待实现 |
| 三模态同步 Mosaic | 🔲 待实现 |

## 核心架构

```
RGB ─→ FEN-RGB ─→ Intra-Hypergraph ──┐
D   ─→ FEN-D   ─→ Intra-Hypergraph ──┼──→ 模态质量评估 → 双向融合 → 检测头
T   ─→ FEN-T   ─→ Intra-Hypergraph ──┘
```

### 关键创新

| 组件 | 参考 | 说明 |
|------|------|------|
| 超图增强 (HyperGraph) | M2I2HA (2024) | 模态内高阶关系建模 |
| 融合质量评估 | EvaNet (TPAMI 2026) | 评估各模态质量，指导置信度 |
| 双向融合 + 形变敏感损失 | InfoFusion 2025 | 小目标专用损失函数 |
| 通道切换 + 空间注意力 | CVPR 2023 | 模态间动态通道选择 |

## 项目结构

```
├── README.md              # 本文件
├── AGENTS.md              # 开发规范
├── architecture.md        # 完整技术方案
├── pretext.md             # 预训练策略论文依据
├── CHANGELOG.md           # 决策记录
├── pyproject.toml         # Python 包配置
├── requirements.txt       # 依赖
│
├── configs/
│   ├── aic_baseline.yaml  # Track 1: 双模态数据集配置
│   └── aic_triplet.yaml   # Track 2: 三模态数据集配置
│
├── src/
│   └── aic2026/
│       ├── __init__.py
│       ├── configs/
│       │   └── baseline.yaml   # Track 1 训练超参数
│       ├── models/
│       │   └── triplet_net.py  # Track 2 三模态网络骨架
│       ├── data.py             # Track 2 三模态数据加载
│       ├── quality.py          # Track 2 质量评估模块
│       └── scripts/
│           ├── train_baseline.sh  # Track 1 启动脚本
│           └── train_triplet.sh   # Track 2 启动脚本
│
└── data/                  # 数据集（实际路径在 yaml 中配置）
```

## 部署指南（实验室服务器）

### 🚀 一键部署（推荐）

```bash
# 克隆仓库
git clone https://github.com/Air-0000/AIC2026-multimodal-detection.git
cd AIC2026-multimodal-detection

# 一键部署（自动检测 CUDA、创建环境、安装依赖、配置数据）
bash setup_linux.sh --data-dir /path/to/aic2026/data

# 如果服务器有 Conda，推荐用 Conda 环境
bash setup_linux.sh --conda --data-dir /path/to/aic2026/data

# 如果显存较小 (<12GB)，脚本会自动调优 batch/scale
# 也可以手动指定 CUDA 版本
bash setup_linux.sh --cuda 12.1 --data-dir /path/to/aic2026/data
```

### 🔧 分步手动安装

如果你偏好手动操作：

```bash
# 1. 安装依赖
pip install -e .
pip install git+https://github.com/WSYANGSX/machine_learning.git

# 2. 准备数据
# 将竞赛数据按以下目录结构放置：
# /path/to/aic2026/data/
# ├── images/{train,val,test}/  ← RGB 图像
# ├── ir/{train,val,test}/      ← 红外图像
# ├── depth/{train,val,test}/   ← Depth 图像（三模态时）
# └── labels/{train,val,test}/  ← YOLO 格式标签

# 3. 修改配置
# 编辑 configs/aic_baseline.yaml，将 path 改为实际数据路径

# 4. 启动训练
bash src/aic2026/scripts/train_baseline.sh
```

### 🎮 训练参数

训练脚本支持命令行参数覆盖，无需改文件：

```bash
# 默认训练 (s 模型, batch=16)
bash src/aic2026/scripts/train_baseline.sh

# 低显存模式 (n 模型, batch=8, 开 AMP)
bash src/aic2026/scripts/train_baseline.sh --scale n --batch 8 --amp

# 多卡训练
bash src/aic2026/scripts/train_baseline.sh --device 0,1,2,3

# 自定义
bash src/aic2026/scripts/train_baseline.sh --device 0 --batch 32 --epochs 500
```

## 参考文献

| 文献 | 用途 |
|------|------|
| M2I2HA — Multi-modal Hypergraph Attention (2024) | 超图融合检测基础框架 |
| Channel Switching + Spatial Attention (CVPR 2023) | 多模态特征融合基础 |
| YOLO (CVPR 2016) | 基线框架设计 |
| EvaNet (TPAMI 2026) | 融合质量评估模块 |
| Bi-Directional Fusion (InfoFusion 2025) | 双向融合 + 形变敏感损失 |
| RGB-D-T Tracking (NeurIPS 2025 D&B) | 三模态任务设计规范 |

## 状态

🟢 **当前**：Track 1 — M2I2HA Baseline 代码就绪，等待部署训练
🔲 **下一步**：在实验室服务器运行 Baseline 训练，验证 mAP
🔲 **后续**：推进 Track 2 — 三模态扩展 + 质量评估
