# AIC 2026 — 面向城市场景的多模态目标检测

**AIC 2026 竞赛项目** — 基于 RGB + Depth + Infrared 三模态数据融合的视觉目标检测。

## 任务概述

- **任务**：RGB + Depth + Infrared 三模态融合目标检测（12 类）
- **数据**：训练集 2000 对，测试集 3000 对
- **评价指标**：mAP@50-95
- **场景**：城市场景多源异构数据

## 技术方案

### 核心架构

```
RGB ─→ FEN-RGB ─→ Intra-Hypergraph Enhancement ──┐
D   ─→ FEN-D   ─→ Intra-Hypergraph Enhancement ──┼──→ 模态质量评估 → 双向融合 → 检测头
T   ─→ FEN-T   ─→ Intra-Hypergraph Enhancement ──┘
```

### 关键创新

| 组件 | 参考 | 说明 |
|------|------|------|
| 超图增强 (HyperGraph) | CVPR 2023 | 模态内高阶关系建模 |
| 融合质量评估 | EvaNet (TPAMI 2026) | 评估各模态质量，指导置信度 |
| 双向融合 + 形变敏感损失 | InfoFusion 2025 | 小目标专用损失函数 |
| 通道切换 + 空间注意力 | CVPR 2023 | 模态间动态通道选择 |

### 参考文献

| 文献 | 用途 |
|------|------|
| Channel Switching + Spatial Attention (CVPR 2023) | 多模态特征融合基础 |
| YOLO (CVPR 2016) | 基线框架设计 |
| EvaNet (TPAMI 2026) | 融合质量评估模块 |
| Bi-Directional Fusion (InfoFusion 2025) | 双向融合 + 形变敏感损失 |
| RGB-D-T Tracking (NeurIPS 2025 D&B) | 三模态任务设计规范 |

## 目录结构

```
├── AGENTS.md        # 开发规范（代码规范、Git规范、PR规范）
├── architecture.md  # 技术方案架构设计（完整方案文档）
└── pretext.md       # 其他说明
```

## 状态

🟢 方案设计完成，代码实现中
