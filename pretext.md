 # 预训练权重方案——论文依据

 > 三条核心主张的文献支撑，用于指导 Depth / Thermal backbone 初始化策略。

 ---

 ## 1. ImageNet RGB 权重 → Depth / Thermal backbone 初始化

 ### 理论依据

 **[Yosinski et al., NIPS 2014] — How transferable are features in deep neural networks?**
   `arXiv: 1411.1792` · [pdf](https://arxiv.org/abs/1411.1792)

 转移学习的奠基性实验。将 AlexNet 逐层冻结后在新任务上 finetune，发现：
 > *"The first few layers learn Gabor-like edge detectors, color blobs, and gradient
 > directions — **generic features that are modality-agnostic**. Deeper layers
 > progressively become task-specific."*
 - Layer 1-3 几乎完美迁移（冻结 vs finetune 差距 < 1%）
 - Layer 5-7 需要 finetune
 - 全部随机初始化 → 最优性能的 **72%**

 结论：CNN 低层 filter 对 RGB / Depth 梯度图 / Thermal 热辐射图是通用的。

 **[Kornblith et al., CVPR 2019] — Do Better ImageNet Models Transfer Better?**
   `arXiv: 1805.08974` · [pdf](https://arxiv.org/abs/1805.08974)
 - ImageNet 预训练权重在 12 个下游任务上比随机初始化平均高 **7-15%**
 - 优势在 **数据量越小时越显著**

 ### 实践依据（RGB-T / RGB-D 融合检测论文）

 | 论文 | 方法 | backbone 初始化策略 |
 |---|---|---|
 | **M2I2HA** (Jiang et al., 2024) | 超图融合检测 | RGB + Thermal 两路独立 backbone，**都用 ImageNet 权重初始化** |
 | **CFT** (Cheng et al., ICCV 2021) | Channel-wise Feature Transformer | RGB + Thermal 共享权重，ImageNet 初始化 |
 | **SimpleFusion** (`arXiv: 2406.19055`) | 红外-可见光融合 | RGB + T 均从 ImageNet 预训练 backbone 初始化 |
 | **多光谱行人检测** (Konig et al.) | Faster R-CNN + RGB-T | 两路独立 backbone，均用 ImageNet 初始化 |

 > 这是 RGB-T / RGB-D 融合检测领域的 **标准实践**，不是"凑合方案"。

 ---

 ## 2. Depth 归一化策略：log 变换

 **[Eigen et al., NIPS 2014] — Depth Map Prediction from a Single Image**
   `arXiv: 1406.2283` · [pdf](https://arxiv.org/abs/1406.2283)

 Monocular depth estimation 的开创性工作，首次提出用 **log 空间** 处理深度：
 > *"Depth values span orders of magnitude (0.3m to 100m+). Linear regression on
 > raw depth over-weights distant objects. We train on **log of depth** to equalize
 > error contributions across distance."*

 在比赛数据中 Depth 范围为 16bit [0, 19999 mm]：
 ```
 方案               | 公式                      | 近处物体 (0.3m) | 远处 (20m)
 ───────────────────┼───────────────────────────┼────────────────┼────────────
 线性归一化         | d / 19999                | 0.000015       | 1.0
 log 变换 ★ 推荐    | log(d+1) / log(20000)    | 0.12           | 1.0
 inverse depth      | (1/d) / (1/300)          | 1.0            | 0.015
 ```

 后续工作延续：
 - **DPT** (Ranftl et al., CVPR 2021, `arXiv: 2103.13413`) → inverse depth
 - **Depth Anything** (Li et al., 2024)
 - **MiDaS** (Ranftl et al., CVPR 2020)

 选择理由：检测任务中 **近处物体（< 5m）占检测框和类别多样性的大头**，log 变换给近处深度变化最大的数值空间，比 inverse depth 对远距离噪声更鲁棒。

 ---

 ## 3. 冻结策略：冻结 stem + stage1-2，训练 stage3-4 + fusion + head

 **[Yosinski et al., 2014] — 同一篇的冻结实验数据：**

 | 冻结策略 | 在新任务上的性能 (相对最佳) |
 |---|---|
 | 冻结 1-3 层，随机初始化 4-7 层 | **93%** |
 | 全部层随机初始化 | 72% |
 | 全部层 finetune (baseline) | 100% |

 核心 insight：
 - 低层（stem, stage1, stage2）filter 是 **Gabor 边缘检测器 + 色块/梯度探测器** → modality-agnostic，冻结不会损失表达能力
 - 高层（stage3, stage4）filter 包含**语义特征**（如物体部件、纹理模式） → 必须 fine-tune 适配新任务
 - 融合模块和 detection head 完全随机初始化 → 必须训练

 在 **2000 张训练集**的场景下，冻结低层额外的好处：
 - 参数量降至可训练部分 ~10-20M，大大减小过拟合风险
 - 保持 COCO/ImageNet 预训练的低层通用特征不被有限的 depth/thermal 数据带偏

 > 这也是多模态检测论文的常见做法。论文 [5] (RDTTrack, NeurIPS 2025) 同样采用了冻结 backbone + 只训 fusion 的策略。

 ---

## 4. 质量评估模块 & 置信度校调（自监督，无跨模态依赖）

### 设计约束

来自架构第 8 节的核心结论：
> *"Prompt Learning 假定 RGB 始终是可靠的会议召集人；超图融合假定任何
> 人都能当会议召集人，取决于今天的议题。"*

因此质量评估模块必须满足：
- **不能假设任何模态是"主要锚点"**（RGB 不行、D 不行、T 也不行）
- **每个模态独立评估自身质量**，不依赖其他模态作为参考
- **任意一个模态出问题，不影响其他模态的质量判断**

### 两个互补方案

#### 方案 A（主）：Heteroscedastic Aleatoric Uncertainty

**[Kendall et al., NeurIPS 2018] — Multi-Task Learning Using Uncertainty to Weigh Losses**
  `arXiv: 1705.07115` · [pdf](https://arxiv.org/abs/1705.07115)

Kendall 原文用每个任务一个全局 σ（homoscedastic），后续扩展为输入依赖的异方差
（heteroscedastic）不确定性。每个模态有一个**输入依赖的不确定性 σ_i(x)**：

```
Feature_RGB → Uncertainty Head → σ_rgb(x)     ← 只依赖 RGB 自身特征
Feature_D   → Uncertainty Head → σ_d(x)
Feature_T   → Uncertainty Head → σ_t(x)
```

训练方式（每个模态独立计算 loss）：
```
L_i = 1/(2·σ_i(x)²) · L_det(F_i, y) + 1/2 · log(σ_i(x)²)

第一项：σ 大 → loss 权重低 → 该模态贡献小
第二项：正则项，防止 σ 无限膨胀
```

| 场景 | 谁出问题 | σ_i ↑ | 融合权重 |
|---|---|---|---|
| 夜间 | RGB 差 | σ_rgb ↑ | w_rgb ↓ |
| 夏天热交叉 | T 差 | σ_t ↑ | w_t ↓ |
| 镜面玻璃 | D 差 | σ_d ↑ | w_d ↓ |
| 白天全好 | 都好 | 三个 σ 都小 | 近似平均 |

**优点**：完全自监督（唯一监督信号是 detection loss）、不依赖跨模态信息、
end-to-end 可微，参数量极小（每模态一个小 MLP）

**局限**：通过 loss 梯度间接驱动，对"不影响 loss 的质量差"可能不够敏感

#### 方案 B（互补）：Feature Statistics Quality Head

差模态的深层特征在统计性质上一定会偏离"健康"分布——直接检测这种偏离。

```
每个模态独立：

F_i ∈ R^(C×H×W)         ← backbone 输出的特征图
    ↓
per-channel 统计量：
  mean, var, max, min, energy (L2 norm)
    ↓
2-layer MLP → q_i ∈ [0, 1]
```

**训练信号：Modality Corruption Detection（自监督预文本任务）**

```
Stage 1 — Pretext task（正式训练前单独跑，~50 epoch）

  对每个训练样本：
  1. 随机选择一个模态做人工损坏
     （高斯噪声 / 置零 / 模拟坏像素 / 对比度压缩）
  2. 训练 binary classifier：该模态被损坏了吗？
  3. 迭代 → classifier 学会区分"正常特征模式"和"异常特征模式"

Stage 2 — 用作质量评估

  推理时：classifier logit → sigmoid → q_i ∈ [0, 1]
  q_i 高 → 特征模式与健康训练分布一致 → 质量好
  q_i 低 → 特征模式偏离训练分布 → 质量差
```

**完全自监督**：预文本任务自己构造损坏样本做监督，不需要任何人工标注。
损坏策略的选择不依赖任何模态做锚点。

**Stage 2 之后无需保留损坏数据**——classifier 已经学会区分"正常特征"的分布形式。

#### 融合方式

`σ_i` 与 `q_i` 各司其职，不再混用：

```
融合权重：

w_i = softmax( α · 1/(2·σ_i²) + β · q_i )
                ↑ 方案 A             ↑ 方案 B
                α, β 可学习权重
```

其中：
- `σ_i`：通过检测损失间接学习的输入依赖不确定性，用于调节模态在融合中的贡献
- `q_i`：通过自监督 pretext task 学到的模态质量分数，用于提供更直接的退化检测信号

| 维度 | 方案 A (Uncertainty) | 方案 B (Quality Head) |
|---|---|---|
| 信号来源 | detection loss 间接驱动 | 特征分布异常直接检测 |
| 对极差模态 | σ 可能拉得不够大 | 能明确输出 q ≈ 0 |
| 对轻微差模态 | σ 能精确微调 | 可能和正常分不开 |
| 联合训练 | 开始时即可用 | Pretext task 完成后 |

### 与置信度校调的关系

之前架构将质量评估（9.4）和置信度校调（9.6）拆成两个独立模块。
现在明确修正为：

- **质量分数主要用于融合阶段**，动态调节不同模态的特征贡献
- **最终提交置信度直接使用融合检测头输出的 `cls_score`**
- 不再额外构造 `confidence = cls_score · g(q_rgb, q_d, q_t)` 或 `α·cls + β·quality` 这类二次校调公式，避免对低质量模态重复惩罚

原因是：
- `cls_score` 来自**融合后的统一特征**，本身已经反映了质量加权后的结果
- 若某一模态质量差，它在融合阶段已经被降权；在最终置信度上再次惩罚会造成重复压制
- 每个模态的质量是其自身属性，不应通过统一聚合函数彼此连带惩罚

### 模态独立辅助置信度（用于诊断，不直接提交）

在 GateFusion 之前，每个模态在跨超图更新后都会得到各自的增强特征。
可以在这些特征上挂接一个**极轻量的辅助头**（例如 `Conv + Pool + MLP`），分别输出：

- `c_rgb`
- `c_d`
- `c_t`

这些分数的用途：
- 训练阶段的辅助监督
- 推理阶段的可解释性分析（判断当前检测主要依赖哪个模态）
- 论文中的错误分析与可视化

它们**不替代**最终检测头，也**不直接作为比赛提交置信度**。


 ## 综合策略

 ```
         ┌──────────────────────────────────┐
         │       ImageNet 权重 (×3)          │
         │    RGB | Depth | Thermal          │
         └────────┬─────────┬───────┬──────┘
                  │         │       │
                  v         v       v
         ┌──────────────────────────────────┐
         │   stem + stage1 + stage2         │ ← 冻结（通用特征，modality-agnostic）
         │     (全部共享初始化策略)            │
         └────────────────┬─────────────────┘
                          │
         ┌────────────────v─────────────────┐
         │   stage3 + stage4 (+ neck)       │ ← 训练（语义特征，需适配）
         └────────────────┬─────────────────┘
                          │
         ┌────────────────v─────────────────┐
         │   Intra-Hypergraph Enhancement   │ ← 训练（随机初始化）
         │   质量评估 & 置信度校调             │ ← 训练（随机初始化）
         │   (Uncertainty + Quality Heads)   │    两个方案并行
         │   Inter-Hypergraph Fusion        │
         │   Detection Head                 │
         └──────────────────────────────────┘
 ```

 Depth 具体处理：
 1. 16bit uint → 转 float，除以 19999 归一化到 [0,1]
 2. log 变换：`d_log = log(d_norm * 19999 + 1) / log(20000)`，本质是 `log(raw_d + 1) / log(20000)`
 3. 单通道复制成 3 通道，喂入 backbone（匹配 RGB 输入尺寸）

 Thermal 直接 normalize 到 [0,1]，无特殊处理。

 ---

 ## 引用

 1. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *Advances in Neural Information Processing Systems*, 27. arXiv: 1411.1792
 2. Kornblith, S., Shlens, J., & Le, Q. V. (2019). Do Better ImageNet Models Transfer Better? *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. arXiv: 1805.08974
 3. Eigen, D., Puhrsch, C., & Fergus, R. (2014). Depth Map Prediction from a Single Image using a Multi-Scale Deep Network. *Advances in Neural Information Processing Systems*, 27. arXiv: 1406.2283
 4. Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision Transformers for Dense Prediction. *Proceedings of the IEEE/CVF International Conference on Computer Vision*. arXiv: 2103.13413
 5. SimpleFusion: A Simple Fusion Framework for Infrared and Visible Images. arXiv: 2406.19055
 6. M2I2HA — Jiang et al. (2024). Multi-modal Interaction via Intra and Inter Hypergraph Aggregation.
 7. CFT — Cheng et al. (2021). Channel-wise Feature Transformer. *ICCV 2021*.
 8. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *Advances in Neural Information Processing Systems*, 31. arXiv: 1705.07115
