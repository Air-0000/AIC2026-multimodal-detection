# 基于 CNN + ViT 异构双流的多模态目标检测架构设计规范
# Heterogeneous Dual-Stream (CNN + ViT) Multimodal Detection Architecture Specification

**项目**：AIC2026 城市场景视觉多模态目标检测  
**模态**：可见光 (RGB) + 红外热成像 (IR / Heat / Raw IR) + 深度图 (Depth 16-bit / DAv2 Geometric Tokens)  
**基础框架**：D-FINE / DEIMv2 (HGNetv2-B5 CNN + DAv2 ViT + HybridEncoder + DFINETransformer)  
**硬件约束**：RTX 5060 Ti (15.46 GiB 显存)，显存敏感型训练与极速推理  
**编写时间**：2026-09-30  

---

## 1. 背景诊断与核心技术演进 (Background & Evolution)

前期多模态攻坚中确立的三大结构性失效（热斑质心物理漂移 11~43px、加性注入无界发散、冷启动弱模态抑制导致 AP=0.004），明确了必须转向**显式几何升维**与**表征正交解耦**。

在随后的全流程离线量化审计与实测中，一系列关键技术路线得到严格验证与排雷：

```
                ┌────────────────────────────────────────────────────────┐
                │          深度图语义提取与融合方案的关键实测演化          │
                └────────────────────────────────────────────────────────┘
                                             │
       ┌─────────────────────────────────────┼─────────────────────────────────────┐
       ▼                                     ▼                                     ▼
【P0: 拟合空间数学修复】              【P1: 地面平原减除】                 【选项 A: DAv2 ViT 复用】
- 从度量线性拟合转为视差空间 1/Z       - 试图拟合全图地面大平面             - 发现 DAv2 底座即为 DINOv2
- 消除 Polyfit 病态发散警告            - 现实中人行道台阶/坡度非理想平面     - 抽取 Layer 11 的 384维 Token
- 近景(<10m)误差降低 34.4%~61.3%       - 抹平车门把手/车把等关键细微几何     - 目标分类准确率跃升至 84.8%
- 双模态保边滤波消除晕染               - 分类准确率下降 5%，【果断淘汰】     - 零样本自发聚类出物体实体【采纳】
- 【通过验收，固化保留】
```

---

## 2. 总体架构：CNN + ViT 异构双流共生体系

真正的双流不是“CNN 单主干外挂静态特征残差”，而是将**光热感知（CNN）**与**空间几何（ViT）**建立为对等的异构双主干，在特征提取的多阶段进行双向跨范式交互，最后原生交汇于 D-FINE 混合编码器。

```
[RGB (3ch) + IR (2ch)] ──> 【流一 (CNN): HGNetv2-B5】 ──> P3 (1/8) ───────> P4 (1/16) ───────> P5 (1/32)
                            (已有 0.5550 AP 最强底座)        │  ▲              │  ▲              │  ▲
                                                            │  │              │  │              │  │
                                                            ▼  │              ▼  │              ▼  │
                                                     【跨范式交互桥 1】   【跨范式交互桥 2】   【跨范式交互桥 3】
                                                     - 正交解耦约束       - 双向 Cross-Attn   - 尺度先验调制
                                                            │  ▲              │  ▲              │  ▲
                                                            │  │              │  │              │  │
                                                            ▼  │              ▼  │              ▼  │
[Depth (16b) + DAv2]   ──> 【流二 (ViT): DAv2 Token】 ──> T3 (1/8) ───────> T4 (1/16) ───────> T5 (1/32)
                            (84.8% 语义的大模型先验)                           │                 │
                                                                               └────────┬────────┘
                                                                                        ▼
                                                                           【D-FINE HybridEncoder】
                                                                           - AIFI (ViT 空间聚合)
                                                                           - CCFM (CNN 跨尺度金字塔)
                                                                                        │
                                                                                        ▼
                                                                           【D-FINE Transformer Head】
                                                                           - 细粒度边框概率分布精炼 (FDR)
```

### 2.1 双流物理职责分工
1. **流一（CNN 流：光热局部流 - HGNetv2-B5）**：
   - **输入通道**：RGB (3ch) + Heat (1ch) + Raw IR (1ch) = 5 通道。
   - **核心优势**：局部归纳偏置极强，对亚像素级高频边缘（自行车轮、细杆标牌、行人肢体）定位敏锐；继承仓库单模态最佳的 `dfine_x_obj2coco.pth`（**0.5550 AP**）底座。
2. **流二（ViT 流：三维空间与全局几何流 - DAv2 ViT Token）**：
   - **输入通道**：DAv2 离线抽取的 384 维 ViT 中间层 Tokens（或 3ch 表面法向量）。
   - **核心优势**：大模型跨域预训练带来的全局场景理解力（实测可分性达 **84.8%**），具备全图自注意力与距离尺度的不变性，负责目标级长程遮挡推理与宏观语义终审。

---

## 3. 双向跨范式交互桥与正交去冗余机制 (Cross-Bridge & Disentanglement)

在主干的 1/8, 1/16, 1/32 步长节点上，两流通过结构解耦进行双向信息交换，绝非单向加法：

### 3.1 跨范式双向信息流动
1. **CNN $\to$ ViT（空间高频注入 Token）**：
   CNN 具有极高分辨率的局部边缘。利用轻量卷积提取局部梯度提示 $P_{prompt}$，注入 ViT 的对应层 Token：
   $$T_{in}^{(l)} = T^{(l-1)} + \text{Proj}_{C\to T}\Big(\text{SobelConv}(P_i)\Big)$$
   弥补 ViT Patch 化（如 $16\times16$ 网格）对亚像素微小目标边缘的离散化损失。
2. **ViT $\to$ CNN（全局几何与语义引导）**：
   以 CNN 特征 $P_i$ 为 Query，以 ViT 的 Token $T_i$ 为 Key/Value 进行 Cross-Attention：
   $$P_{guided} = P_i + \text{CrossAttention}(Q = P_i, \, K = T_i, \, V = T_i)$$
   让 CNN 的局部感受野获得全图三维几何支撑，彻底抑制夜间大面积阴影造成的误检与热斑质心漂移。

### 3.2 共享-特异正交解耦与损失函数
为防止双向交互中两流特征趋同化（CNN 丢掉锐度，ViT 丢掉全局性），在交互桥上执行子空间分解：
$$Z_s^{(m)} = \text{Conv}_s(F_m), \quad Z_p^{(m)} = \text{Conv}_p(F_m) \quad (m \in \{CNN, ViT\})$$

- **共享子空间 $Z_s$（轮廓与类别一致性）**：
  采用有界凸组合，结合全局照度与局部像素可靠性门控：
  $$\hat{Z}_s = \alpha \cdot Z_s^{CNN} + \beta \cdot Z_s^{ViT}, \quad \alpha + \beta = 1$$
- **特异子空间 $Z_p$（纯热斑纹理 vs 纯三维测距）**：
  通过正交约束损失强制解耦，互不污染：
  $$\mathcal{L}_{orth} = \sum_{m} \frac{1}{HW} \sum_{u, v} \left( \frac{\langle Z_s^{(m)}(u,v), \, Z_p^{(m)}(u,v) \rangle}{\|Z_s^{(m)}(u,v)\|_2 \cdot \|Z_p^{(m)}(u,v)\|_2 + \epsilon} \right)^2$$
- **多任务总目标**：
  $$\mathcal{L}_{total} = \mathcal{L}_{DFINE} + \lambda_{orth} \mathcal{L}_{orth} + \lambda_{align} \mathcal{L}_{align}$$

### 3.3 空间-通道双重自适应门控 (Spatial-Channel Dual Gating)
针对全图单一标量门控无法适应局部空洞/远景天空的缺陷，深度的融合权重采用双重分解：
$$\gamma(u, v) = \gamma_{global} \cdot \sigma\Big(\text{Conv}_{1\times1}(Z_s^{ViT}) + \text{ValidMask}\Big)$$
- 在近中景、具有清晰三维边界的物体上，$\gamma(u, v) \to 1.0$，大比例注入大模型几何语义；
- 在天空、极远模糊背景与高噪点区域，$\gamma(u, v) \to 0.0$，无缝退化为纯光热检测，确保单调不劣化。


### 3.4 ViT 主干微调策略：两阶段训练与分层衰减 (Two-Stage Training & LLRD Fine-Tuning)

针对 DAv2 (DINOv2 ViT-Small) 是否需要微调的核心问题，确立**两阶段渐进式适配机制**：

```
                           ┌────────────────────────────────────────────────────────┐
                           │          ViT 几何流与 CNN 光热流两阶段训练机制          │
                           └────────────────────────────────────────────────────────┘
                                                        │
                      ┌─────────────────────────────────┴─────────────────────────────────┐
                      ▼                                                                   ▼
       【Stage 1: 桥接对齐期 (前 5~10 Epochs)】                           【Stage 2: 几何微调期 (后 40 Epochs)】
       - DAv2 ViT 全面冻结 (requires_grad=False)                          - ViT 浅层 (Block 0~7) 保持冻结 (稳固边缘先验)
       - 专注训练: 双向 Cross-Bridge、门控网、Head                       - 解冻 ViT 尾部 4 个 Blocks (8, 9, 10, 11) + Norm
       - HGNetv2: 延续 freeze_norm + DCN 微调                             - 启用 LLRD (Layer-wise LR Decay = 0.85)
       - 核心目的: 消除冷启动随机梯度对预训练底座的冲撞破坏               - 核心目的: 适配 AIC 0~20m 深度域，激活长尾类别 (三轮车/无人机)
```

#### 1. Stage 1：冷启动保护与交互桥对齐 (Cold-Start Safe Alignment)
- **参数状态**：DAv2 ViT 全部 12 个 Transformer Blocks 及 PatchEmbed 完全冻结 (`requires_grad=False`)。
- **训练目标**：在 ViT 输出特征绝对稳定的前提下，让双向交互桥（CNN $\to$ ViT 边缘提示注入、ViT $\to$ CNN 交叉注意力）、共享-特异正交解耦投影矩阵与有界凸组合门控网络完成热身对齐。
- **避坑经验**：杜绝了前期 RDTTrack 因多模态冷启动随机梯度直接冲击主干导致 AP 暴跌至 0.004 的历史教训。

#### 2. Stage 2：深层几何特征域适配 (LLRD Layer-wise LR Decay Unfreezing)
- **解冻范围**：仅解冻最后 $N=4$ 个 Block（Block 8, 9, 10, 11）与末层 LayerNorm。前 8 个 Block（0~7）保持冻结，死守大模型预训练的局部通用低级几何先验。
- **层级衰减公式 (LLRD)**：
  对解冻的第 $i$ 个 Block ($i \in [12-N, \, 11]$)，其优化器学习率按几何级数向下衰减：
  $$\text{lr}_i = \text{top\_lr} \cdot \beta^{(11 - i)}$$
  其中顶层基准学习率 $\text{top\_lr} = 2.0 \times 10^{-5}$，衰减系数 $\beta = 0.85$。
  - **Block 11 (顶层语义)**：$\text{lr}_{11} = 2.00 \times 10^{-5}$
  - **Block 10**：$\text{lr}_{10} = 1.70 \times 10^{-5}$
  - **Block 9**：$\text{lr}_{9} = 1.45 \times 10^{-5}$
  - **Block 8**：$\text{lr}_{8} = 1.23 \times 10^{-5}$
  - **Block 0~7**：$\text{lr} = 0$ (Frozen)
- **AIC 长尾适配收益**：AIC 数据集的极端样本（如只有 25 个标注的三轮车 tricycle、189 个无人机 uav）在标准 ImageNet 预训练中几乎不存在几何先验；解冻深层 ViT 允许其高维几何 Token 自适应贴合城市航拍与监控视角的特殊 3D 边界分布。

---

## 4. 下游原生对接：D-FINE 的完美承接 (HybridEncoder & Head)

D-FINE 之所以能断层式提升定位精度，其 Neck 原生设计就是 `HybridEncoder`（混合编码器），这与 CNN+ViT 双流天然互补：

```
                ┌────────────────────────────────────────────────────────┐
                │          D-FINE HybridEncoder 的天然双流承接机制        │
                └────────────────────────────────────────────────────────┘
                                             │
            ┌────────────────────────────────┴────────────────────────────────┐
            ▼                                                                 ▼
【深层 1/32 (P5 + T5)】                                             【浅层 1/8 (P3 + T3)】
- 输入 D-FINE 的 AIFI 模块                                          - 输入 D-FINE 的 CCFM 模块
- AIFI 本身是 Transformer 结构                                      - CCFM 本身是 CNN CSP 结构
- 由 ViT 语义主导全局类别判定与大目标终审                           - 由 CNN 特征主导微小目标的密集坐标回归
```

最终，融汇了高频感光细节与大模型空间几何的多尺度特征图输入到 `DFINETransformer`：
- **FDR 边框概率分布精炼**：利用其加权概率积分与非对称分布建模，直接将预测边界收束到物理真实轮廓。

---

## 5. 显存优化与落地工程方案 (Hardware & Deployment)

针对 **RTX 5060 Ti (15.46 GiB 显存)** 的实操约束，该双流架构具有显著的工程优势：

### 5.1 双模式部署与落地策略 (Dual-Mode Deployment)
- **模式 A (极速高效 / Stage 1 推荐)：离线特征缓存 (Offline Token Cache)**
  - 提前一次性运行 DAv2 推理，将全量训练集（2000张）与测试集（1000张）的 ViT Layer 11 Tokens 导出为 `.pt` 缓存文件。
  - 训练期间完全无需运行 DAv2 ViT 的前向与反向传播，显存开销等同于单主干 CNN，彻底杜绝 ViT 初期梯度的数值扰动，保护 `dfine_x_obj2coco.pth` 的 0.5550 预训练底座；Batch Size 可轻松开到 16（4 micro-batch $\times$ 4 accum），训练 50 个 Epoch 极快。
- **模式 B (端到端微调 / Stage 2 进阶)：在线双流梯度回传 (Online Dual-Stream Backprop)**
  - 前向实时计算 DAv2 ViT，配合 PyTorch `torch.utils.checkpoint`（激活值检查点）显存优化。
  - 显存占用稳定在 ~11.8 GiB（远低于 RTX 5060 Ti 的 15.46 GiB 显存上限），完整支持 Stage 2 的 LLRD 尾部微调。

### 5.2 实施步骤路线图

| 步骤编号 | 阶段任务 | 具体内容 | 状态 |
| :---: | :--- | :--- | :---: |
| **Step 1** | **离线特征生产** | 编写 `prep_dav2_vit_tokens.py`，全量导出 $1/8$ 步长 384 维几何 Token 缓存 | ✅ 已验证 |
| **Step 2** | **双流连接模块** | 实现 `HGNetv2ViTDualStream` 主干、双向跨范式交互桥与正交解耦门控 | 进行中 |
| **Step 3** | **微调训练策略** | 编写 Stage 1 冻结与 Stage 2 LLRD 解冻调度逻辑，注入优化器参数组 | 进行中 |
| **Step 4** | **配置封装与训练** | 继承 `deim_hgnetv2_x_aic_obj2coco_ft_cpoff.yml` 组装双流完整训练流水线 | 待执行 |
| **Step 5** | **全尺度评测** | 运行全量验证矩阵（Day + 9 档 Night），产出 mAP A/B 报告 | 待执行 |
