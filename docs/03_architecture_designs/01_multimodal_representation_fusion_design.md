# 基于表征学习的多模态融合方案设计规范
# Multimodal Representation Learning & Disentangled Fusion Architecture

**项目**：AIC2026 城市场景视觉多模态目标检测  
**模态**：可见光 (RGB) + 红外热成像 (IR / Heat / Raw IR) + 深度图 (Depth 16-bit)  
**基础框架**：DEIMv2 (DINOv3-ViT Backbone + HybridEncoder + RT-DETR/DFINE Style Decoder)  
**硬件约束**：RTX 5060 Ti (15.46 GiB 显存)，显存敏感型训练

---

## 1. 背景与核心问题诊断 (Background & Diagnostics)

在前期针对 AIC 数据集与现有模型的攻坚整合审计中（见 `docs/04_audits_and_ablations/AIC_IR融合攻坚整合报告.html`），确认了传统多模态融合在红外/深度/可见光场景下的三大失效根源：

```
                ┌────────────────────────────────────────────────────────┐
                │             传统多模态融合的三大结构性缺陷             │
                └────────────────────────────────────────────────────────┘
                                             │
      ┌──────────────────────────────────────┼──────────────────────────────────────┐
      ▼                                      ▼                                      ▼
【1. 物理语义错配 (Disparity)】        【2. 加性注入无界 (Unbounded)】         【3. 弱特征冷启动抑制】
- 传感器虽空间对齐 (残差<3px)          - PromptBlock: f_rgb + prompt          - 冻结强 RGB (DINOv3 先验)
- 但物理量不同：                        - 在 IR 为无效噪声处 (白天/背景)       - 对比从头训练弱 IR/Depth
  IR = 体内核心热源 (热斑)               仍强行累加，幅度无界破坏 RGB           - 逐像素内积初始 α=β=0.62
  RGB = 表面外轮廓与材质纹理             - AUX_GATE 整图标量无法处理局部        直接扼杀初期微弱梯度 (AP=0.004)
  热斑质心与框中心偏移 11~43px
```

### 关键结论：
1. **不是几何错位，而是语义错配**：红外热核质心与可见光轮廓中心存在姿态相关的物理偏移（中位数 11px，P90 达 43px），任何全局刚性或仿射单应均无法纠正。必须采用**局部自适应可变形重采样（Deformable Semantic Alignment）**。
2. **拒绝无界加法，拥抱有界凸组合**：加性残差在白天或背景遮挡处带来负收益。必须转向**有界凸组合门控（Bounded Convex Combination）**，满足 $\|f_{fused}\| \le \max(\|f_m\|)$，实现“昼隐红外、夜抑可见、边借几何”。
3. **共享与私有特征解耦（Shared-Private Disentanglement）**：红外与深度既包含跨模态一致的边界语义，又包含物理特有的温差与测距信号。不加区分的逐像素 Gram-Schmidt 内积会导致有用特征被误剔除。

---

## 2. 表征学习指导下的四层融合架构

```
[RGB Input (3ch)]       [IR Input (2ch)]       [Depth Input (1ch)]
        │                      │                       │
        ▼                      ▼                       ▼
[DINOv3 (Frozen/LoRA)]   [DINOv3 / ConvStem]     [DINOv3 / ConvStem]
        │                      │                       │
     f_rgb                   f_ir                    f_depth
        │                      │                       │
        │             ┌────────┴───────────────────────┘
        │             ▼
        │    【Layer 1: 语义形变空间自适应校准 (Deformable Alignment)】
        │    利用 f_rgb 高层语义引导 f_ir / f_depth 偏移采样，纠正热核质心漂移
        │             │
        ├─────────────┼────────────────────────────────┐
        ▼             ▼                                ▼
【Layer 2: 共享-特异空间表征解耦 (Shared-Private Subspace Disentanglement)】
  Z_s: 共享空间 (几何边界、目标类别一致性)  <-- InfoNCE 跨模态语义对比拉近
  Z_p: 特异空间 (IR热辐射体核、Depth空间测距) <-- 正交/独立性约束 (HSIC / Orthogonal)
        │             │                                │
        └─────────────┼────────────────────────────────┘
                      ▼
【Layer 3: 不确定性与环境感知有界门控 (Bounded Uncertainty Gating)】
  - 物理照度感知先验 (Raw RGB Luminance & Variance)
  - 特征不确定性与熵估计
  - 凸组合加权: f_shared = α·Z_s^rgb + β·Z_s^ir + γ·Z_s^depth, α+β+γ=1
  - 特异互补注入: f_priv = Tanh(W_p · [Z_p^ir, Z_p^depth]) * scale
                      │
                      ▼
【Layer 4: 分层特征重组与多尺度注入 (Multi-Scale Hybrid Injection)】
  c2, c3: 融合特征 + STA 空间细节流
  c4: 高层抽象语义流 (保持强先验或轻量注入)
```

---

## 3. 数学原理与技术规范

### 3.1 语义级空间自适应形变校准 (Layer 1)
给定输入特征 $f_{rgb} \in \mathbb{R}^{B \times C \times H \times W}$ 与辅助模态特征 $f_{aux} \in \mathbb{R}^{B \times C \times H \times W}$：
1. 预测双向空间偏移场 $\Delta p$ 与调制幅度 $m$：
   $$\Delta p, m = \text{Conv}_{3\times3}(\text{GELU}(\text{Conv}_{1\times1}([f_{rgb}, f_{aux}]))) \in \mathbb{R}^{B \times 3K \times H \times W}$$
2. 通过可变形采样生成对齐后的辅助特征 $\tilde{f}_{aux}$：
   $$\tilde{f}_{aux}(p) = \sum_{k=1}^K w_k \cdot \sigma(m_k) \cdot f_{aux}(p + p_k + \Delta p_k)$$
   在轻量化实现中，可采用由 RGB 特征引导的双线性重网格采样（Grid Sample），显存增加 $< 3\%$。

### 3.2 共享-私有表征解耦 (Layer 2)
对各模态特征映射到低维流形空间（$C_r = C / 4$）：
$$Z_s^{(m)} = \text{Conv}_s(f_m), \quad Z_p^{(m)} = \text{Conv}_p(f_m), \quad m \in \{rgb, ir, depth\}$$

**正交解耦损失（Subspace Orthogonality Regularization）**：
$$\mathcal{L}_{orth} = \sum_{m \in \{rgb, ir, depth\}} \frac{1}{HW} \sum_{u, v} \left( \frac{\langle Z_s^{(m)}(u,v), Z_p^{(m)}(u,v) \rangle}{\|Z_s^{(m)}(u,v)\|_2 \cdot \|Z_p^{(m)}(u,v)\|_2 + \epsilon} \right)^2$$
此项确保私有分支仅提取模态独特的物理表征，不与共享分支重复。

### 3.3 环境与特征不确定性感知有界门控 (Layer 3)
1. **环境先验得分（Illumination Prior）**：
   从反归一化的原始 RGB 计算全局亮度均值 $b \in [0, 1]$：
   $$g_{env} = \text{Sigmoid}(k \cdot (b_0 - b)) \quad (\text{暗光时 } g_{env} \to 1, \text{ 白天 } g_{env} \to 0)$$
2. **特征不确定性加权凸组合（Convex Combination）**：
   通过特征通道统计与环境先验联合生成三模态权重 $[\alpha, \beta, \gamma]$：
   $$[\hat{\alpha}, \hat{\beta}, \hat{\gamma}] = \text{Linear}(\text{GAP}([Z_s^{rgb}, Z_s^{ir}, Z_s^{depth}])) + \mathbf{w}_{prior} \cdot g_{env}$$
   $$[\alpha, \beta, \gamma] = \text{Softmax}([\hat{\alpha}, \hat{\beta}, \hat{\gamma}] / \tau)$$
   $$\hat{Z}_s = \alpha \cdot Z_s^{rgb} + \beta \cdot \tilde{Z}_s^{ir} + \gamma \cdot \tilde{Z}_s^{depth}$$
3. **私有特征有界注入**：
   $$f_{priv} = s \cdot \text{Tanh}(\text{Conv}_{1\times1}([Z_p^{ir}, Z_p^{depth}]))$$
   其中 $s$ 为可学习标量参数，初始化为 $0.05$。输出幅度被严格限制在 $[-s, +s]$ 内，绝不溢出。
4. **最终多模态输出**：
   $$F_{fused} = f_{rgb} + \text{Proj}_{out}(\hat{Z}_s) + f_{priv}$$

---

## 4. 与 DEIMv2 框架的集成与接口规范

- **命名规范**：注册为 `DINOv3STAsRepFusion`（表征解耦多模态主干），继承自 `nn.Module`，注册于 `engine.core.register`。
- **输出接口**：与现存所有主干保持严格一致，返回 `(c2, c3, c4)` 三层特征：
  - `c2`: 尺度 $1/8$，通道 $C$（如 ViT-S 为 384，ViT-L 为 1024）
  - `c3`: 尺度 $1/16$，通道 $C$
  - `c4`: 尺度 $1/32$，通道 $C$
- **兼容性**：
  - 支持 `ir_in_chans=1`（仅 heat）与 `ir_in_chans=2`（heat + raw_ir 6通道缓存）。
  - 支持 `depth_in_chans=1`（16-bit 真实深度）。
  - 完美兼容 `HybridEncoder`，无需修改任何下游代码。

---

## 5. 显存优化与训练策略 (VRAM & Training Strategy)

针对 RTX 5060 Ti (15.46 GiB) 的显存上限：
1. **通道降维比例（Reduction Factor）**：在解耦和门控投影层采用 `reduction=4`（如 ViT-S 384 通道降为 96 维），计算开销小于原模型的 $5\%$。
2. **冻结 RGB 先验**：RGB DINOv3 主干全程冻结（阶段一），只训练辅助分支与解耦融合层，训练参数量控制在 10M 级别。
3. **冷启动保护**：门控网络权重小方差初始化，私有注入层初始化增益设为 $0.05$，初始阶段等效于纯 RGB 加轻微辅助指导，规避梯度冲击。
4. **单机批次控制**：输入 $640 \times 640$ 时，`micro_batch_size=4` 稳态显存约为 10.8 GiB，留有充足安全余量。

---

## 6. 渐进式验证与消融实验矩阵 (Ablation Matrix)

| 实验编号 | 模块构成 | 预期解决的核心问题 | 重点评估指标 |
|---|---|---|---|
| **E-Base** | 原版 RDTTrack (dt_ort + PromptBlock) | 基线对照 (当前 best_stg2) | Val mAP: ~0.537 |
| **E-Gate** | 有界凸组合门控 (Bounded Gate) | 消除白天/背景区域的无界加性噪声 | 白天 mAP 提升，极亮场景不跌点 |
| **E-Align** | 有界门控 + 语义形变对齐 (Deformable) | 消除热斑体核与轮廓中心 11~43px 偏移 | 小目标 (UAV/Light) 与行人物体 AP 改善 |
| **E-Disentangle** | 有界门控 + 语义对齐 + 共享-私有解耦 | 彻底剥离热辐射物理量与可见光纹理 | 10 档暗光综合 AUC 与极端夜间 AP50 |
| **E-Full** | E-Disentangle + 辅助正交/对比正则化 | 解决辅助分支冷启动与特征表征崩塌 | 训练初期收敛速度与最终 mAP 上限 |
