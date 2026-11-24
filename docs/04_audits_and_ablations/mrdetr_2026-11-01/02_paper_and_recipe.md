# Mr. DETR / Mr. DETR++ 论文与训练配方提取（v4 / CVPR 2025 版）

> **任务** `task-2`（论文与配方/数字，不涉及代码实现细节）｜**论文** Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with Mixture-of-Experts，arXiv:2412.10028v4，2025-06-26，17 页（`/tmp/mrdetr_paper.pdf`）｜**源码快照** `archive/scratch/mrdetr_src/`（文件时间 2025-09-06，README 已含 2025-07-25 的 ++ 更新）
> **取证**：`pdftotext` 抽文本 + `mutool draw` 渲染 8 个表格页逐行看图核对（本 PDF 两栏排版，`-layout` 抽文本会串栏）；repo 配置用 `grep -n` 标注行号。
> **标记**：`[实测]` 有完整训练/明确数字；`[推断]` 由表格基线数字反推设置；`[未验证]` 论文没写/本地无法核对；`[repo]` 仅仓库有；`[v1]` = CVPR 2025 Mr. DETR（文献 [21]）数字，`[v4]` = 本论文新增；出处写 `Table N` / `§IV-A` / `path:line`。
> **三个前提**：① Table II 的 **“w/ Mr. DETR” 行 = v1 结果**（与 repo README 发布的 v1 权重 AP 完全一致），“w/ Mr. DETR++” 行才是 v4 新增，不要混；② 论文把基线写作 **“Deformable-DETR++ [11]”**（[11]=Deformable DETR）但全文未定义“++”，对照 repo `*_two_stage_*.py`（`with_box_refine/as_two_stage=True`）推断为两阶段+框精修版 [推断；命名来源未验证]；③ 论文**没有** ConvNeXt/EVA-01/FocalNet/InternImage 的数字（backbone 只有 R50 与 Swin-L），repo `projects/mr_detr_pp_align/configs/` 有这些 backbone 的**配置**但**无 AP** [repo；未验证]。

## 1. 方法：按论文原意

### 1.1 为什么把检测当 multi-task（o2o + o2m）

DETR 系用 one-to-one（o2o）稀疏监督，收敛慢；传统检测器用 one-to-many（o2m）+NMS，已有工作把 o2m 当“辅助训练”加入 DETR。本文的框架转换原话（Abstract）：“we treat the model as a **multi-task framework**, simultaneously performing one-to-one and one-to-many predictions.” 冲突来源（§III-B）：同一预测框可能在 o2m 里是正样本、在 o2o 里是负样本，全共享时 o2o 被拉垮——**Table I(2) Share All = 41.6，比纯 o2o 的 47.6 低 6.0**。且已有工作只在“单任务设定”下逐个诊断组件（§I 原话：“previous works examine the functions of each component in the transformer decoder for two training targets only in the single-task setting”），本文在 multi-task 框架下重做。

### 1.2 解码器组件研究（self-attn / cross-attn / FFN）与核心结论

研究 **SA、CA、FFN** 三个组件（Fig. 3(b–d) 分别独立其一）。**Table I（p.2）** 全 12 行（12ep、R50、900q；由基线 47.6 与 Table II 的 12ep/900q 基线相同反推 [推断]；o2m 列 = 辅助路 NMS 最高分）：

| No | 配置 | Routes | o2o AP | o2m AP |
|---|---|---|---|---|
| (1) | One-to-one only | 1 | 47.6 | – |
| (2) | Share All | 1 | 41.6 (−6.0) | 41.6 |
| (3) | Not shared Self-Attention | 2 | 49.7 (+2.1) | 50.3 |
| (4) | Not shared Cross-Attention | 2 | 49.2 (+1.6) | 50.0 |
| (5) | Not shared FFN | 2 | 49.6 (+2.0) | 50.1 |
| (6) | Shared SA（独立 CA+FFN） | 2 | 49.4 (+1.8) | 50.3 |
| (7) | Shared CA（独立 SA+FFN） | 2 | 49.4 (+1.8) | 50.0 |
| (8) | Shared FFN（独立 SA+CA） | 2 | 49.2 (+1.6) | 50.0 |
| (9) | (3)+(4) | 3 | 49.4 (+1.8) | 49.9 |
| (10) | (3)+(5) | 3 | **50.0 (+2.4)** | 50.8 |
| (11) | (4)+(5) | 3 | 49.0 (+1.4) | 49.6 |
| (12) | (3)+(4)+(5) | 4 | 49.6 (+2.0) | 50.2 |

四条结论（§III-B 原文编号）：(i) 全共享损害 o2o（Tab. I(2)）；(ii) **任一单独组件独立即显著提升 o2o（+2.1/+1.6/+2.0），即使其它组件共享** → Abstract & 贡献点原句：“**any independent component in the decoder can effectively learn both targets simultaneously, even when other components are shared**”；(iii) 两个组件独立（I(6–8)）不如单组件独立、却更多参数；(iv) 独立 SA+FFN 最优（I(10)=50.0），含独立 CA 的组合会掉（“likely due to the slow convergence of independent cross-attention”）→ 取 Fig. 3(e) 为设计起点。

### 1.3 Multi-route 路线定义（Fig. 4，§III-B）

三路共享 **object queries 与分类/回归检测头**；Route-2 主路，Route-1/3 只训练、推理丢弃：

| 路线 | 目标 | 参数 | 公式（原编号） |
|---|---|---|---|
| Route-2（主路） | o2o | 同 baseline：SA→CA→FFN_o2o | $\hat Q_2=(\mathrm{FFN}_{o2o}\circ CA\circ SA)(Q)$（Eq.3） |
| Route-1（辅助 A） | o2m | 共享 SA/CA，**独立 FFN_o2m** | $\hat Q_1=(\mathrm{FFN}_{o2m}\circ CA\circ SA)(Q)$（Eq.4） |
| Route-3（辅助 B） | o2m | **共享全部**，SA→instructive SA | $\hat Q_3=(\mathrm{FFN}_{o2o}\circ CA\circ InstructSA)(Q)$（Eq.5） |

o2o 损失 Eq.1：$\mathcal L=\sum_{i=0}^{t}\mathcal L_{cls}(s_{\sigma(i)},\bar s_i)+\mathcal L_{box}(b_{\sigma(i)},\bar b_i)$。o2m 匹配 Eq.2：$M_{ij}=\alpha s_i+(1-\alpha)\mathrm{IoU}(b_i,\bar b_j)$，取匹配分最高至多 **K** 个正样本、再滤 **IoU≥τ**；超参 **K=6, α=0.3, τ=0.4**（§IV-A、Fig. 11、App. D）。推理只留 Route-2，“auxiliary routes are discarded during inference”→ **Mr. DETR 推理开销与 baseline 相同**（v4 §I 仍这样写；++ 不成立，见 §5.2）。

### 1.4 Instructive self-attention（§III-C，Fig. 5，Eq.6）

**m 个可学习 instruction token** $Q_{ins}=\{q^{ins}_0..q^{ins}_{m-1}\}$ 与 object queries **拼接**进 self-attention（长度 m+n），SA 后**丢弃 instruction token 的输出**：
$\hat Q^{ins}=\{q^{ins}_0..q^{ins}_{m-1},q_0..q_{n-1}\}$，$\hat Q_3=(\mathrm{FFN}_{o2o}\circ CA\circ R\circ SA)(\hat Q^{ins})$（Eq.6，$R$ 丢弃 token 输出）。目的：保留“独立 SA”的效果（Tab. I(3)）但不加独立参数；token 只“提示”区分 o2o/o2m（Discussion 与 prompt tuning 的区别）。

- **m=10，六个 decoder 层全用**（§IV-A、App. B；Table XIIIa：1/5/10/50/100 → 50.2/50.1/**50.7**/50.4/50.5）。
- 实现方式对比（Table IX）：independent SA 50.0｜remove SA 49.9｜independent object queries 49.9｜addition 50.1｜**concatenation (ours) 50.7**。
- Fig. 8：300 queries 作 Query、10 tokens 作 Key 时几乎所有 query 强激活 → 信息确实传给 queries [单次可视化，无统计]。

### 1.5 Mr. DETR++：三块新增（§III-D，Fig. 6）

**(a) Route-aware MoE（解码器 FFN 替换，Eq.7–11）** 动机原话：Mr. DETR 的两个独立 FFN “can be regarded as distinct experts tailored to separate tasks, representing a special case of the MoE paradigm”→ 共享专家池+门控同时获得**知识共享**与**任务特异化**。三条路共享 t 个专家 $E=\{e_0..e_{t-1}\}$；**Route-2/3 共享门控 G(·)，Route-1 用独立 G′(·)**（§III-D 正文；Fig. 6 caption 写反，见 §5.6-2）。Eq.7 $\hat Q_1=(E\circ G'\circ CA\circ SA)(Q)$；Eq.8 $\hat Q_2=(E\circ G\circ CA\circ SA)(Q)$；Eq.9 $\hat Q_3=(E\circ G\circ CA\circ InstructSA)(Q)$；Eq.10 $T_k=\mathrm{TopK}(s,k)$（原文用 s，前文门控输出写 c=G(Q)，符号不统一）；Eq.11 $\hat Q=\sum_{i\in T_k}c_i e_i(Q)$。**专家数 t=4、top-k=2**（编/解码器都是；§IV-A、App. C、Table XIV）。论文称 MoE 用在**解码器最后三层**（§IV-F + Fig. 9 画 layer 3/4/5）。**aux loss：论文完全没提**负载均衡损失，而 repo `modeling/moe.py:33` 的 `MoEGate(aux_loss_alpha=0.01)` + `AddAuxiliaryLoss`（`moe.py:214-216`）在训练里加了这个 loss [repo，论文未写]。

**(b) Scale-aware MoE（编码器，Eq.12）** 动机：编码器多尺度长序列全 MoE 太贵 → 所有 token 走一个共享 expert，只在低尺度特征开 routed MoE：$\hat f'_j=\mathrm{FFN}(f_j)+\mathrm{MoE}(f_j)\,(j<\eta)$，否则 $\hat f'_j=\mathrm{FFN}(f_j)$。放置：按 Table X 只在 **x64（最粗尺度）**开（51.8 AP / 486.8 GFLOPs；x64+x32+x16 51.9 但 569.4 GFLOPs）。repo 一致：`dino_transformer.py:77` 编码器 FFN=`SparseMoeBlock(..., n_shared_experts=8, isdecoder=False)`，`moe.py` 里 `level_start_index[3]:` 之后的 token 才走 routed MoE（第 4 尺度）[repo]。注意 `n_shared_experts=8` 是**共享 expert 的宽度倍数**（intermediate=2048），不是 8 个专家。

**(c) Localization-aware score calibration（推理校准，Eq.13）** 动机：只按分类分排序会让低质量框排前面。学 **class-aware IoU 分**（VFL 损失，目标分 $s_{target}=\mathrm{iou}^{0.75}$，§III-D），**不参与标签分配**（区别于 Stable-DINO/Align-DETR，故可作 plug-in）。推理：$s_{calib}=s_{cls}^{\phi}s_{iou}^{1-\phi}$，**默认 φ=0**。小笔误：下一句把预测 IoU 分写成 $s_{pred}$。repo：`two_stage_criterion.py:30-64` 有 `get_vfl_loss`，但指数是 **0.9**（`target_score**0.9`），与论文 0.75 不符 [repo vs 论文，§5.6-5]。

### 1.6 v4 相比 v1 新增了什么

论文原话（§I 末）：“A preliminary version ... has been presented in [21]，which investigates the role of transformer decoder components in a multi-task framework and introduces a multi-route training mechanism with a novel instructive self-attention mechanism. In this work, we further propose ...”。新增：**route-aware MoE**（解码器 FFN→专家池+双门控）、**scale-aware MoE**（编码器低尺度）、**localization-aware calibration**（Eq.13/Table XV）；新任务/数据集：**Objects365（Table IV）、NuImages（Table V）、全景分割（Table VII，Mask DINO 底座）**；新分析：**probing（Table XII）、专家激活分布（Fig. 9）、中间层各路（Table XVI）、训练成本（Table XI）**、更多 baseline（Group-DETR/Salience/Cascade/Align/Rank/Stable-DINO/EASE/DDQ-DETR）与消融（Table VIII–XV）；标题加 “with Mixture-of-Experts”。v1 的 multi-route/instruction 结论与数字原样保留（`w/ Mr. DETR` 行）。v1 独有、v4 **没有**的数字：repo README 的 **Swin-L + Objects365 预训练 → COCO 61.8**（549K iter，batch 16）[repo only]。

## 2. 全部数字表

### 2.1 Table II：COCO 2017 val，R50（p.8）

> ✓ 列 = 在左侧 baseline 上叠加；“w/ Mr. DETR”= v1，“w/ Mr. DETR++”= v4 新增。

| 组 | 基准模型 | 叠加 | Epochs | Queries | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|---|---|---|---|
| 300q/12ep | Deformable-DETR++ [11] | – | 12 | 300 | 47.0 | 65.3 | 51.0 | 30.1 | 50.5 | 60.7 |
| | H-DETR [8] | – | 12 | 300 | 48.7 | 66.4 | 52.9 | 31.2 | 51.5 | 63.5 |
| | MS-DETR [14] | – | 12 | 300 | 48.8 | 66.2 | 53.2 | 31.5 | 52.3 | 63.7 |
| | Deformable-DETR++ | w/ Mr.DETR [v1] | 12 | 300 | 49.5 (+2.5) | 67.0 | 53.7 | 32.1 | 52.5 | 64.7 |
| | H-DETR | w/ Mr.DETR [v1] | 12 | 300 | 49.8 (+1.1) | 67.3 | 54.5 | 31.6 | 53.0 | 65.6 |
| | Deformable-DETR++ | w/ Mr.DETR++ [v4] | 12 | 300 | **51.0 (+4.0)** | 67.8 | 55.6 | 33.2 | 54.8 | 66.6 |
| 900q/12ep | Deformable-DETR++ | – | 12 | 900 | 47.6 | 65.8 | 51.8 | 31.2 | 50.6 | 62.6 |
| | DINO [9] | – | 12 | 900 | 49.0 | 66.6 | 53.5 | 32.0 | 52.3 | 63.0 |
| | MS-DETR | – | 12 | 900 | 50.0 | 67.3 | 54.4 | 31.6 | 53.2 | 64.0 |
| | DAC-DETR [13] | – | 12 | 900 | 49.3 | 66.5 | 53.8 | 31.4 | 52.4 | 64.1 |
| | Deformable-DETR++ | w/ Mr.DETR [v1] | 12 | 900 | 50.7 (+3.1) | 68.2 | 55.4 | 33.6 | 54.3 | 64.6 |
| | Deformable-DETR++ | w/ Mr.DETR++ [v4] | 12 | 900 | **51.8 (+4.2)** | 68.9 | 56.7 | 35.6 | 55.9 | 67.0 |
| 900q/24ep | Deformable-DETR++ | – | 24 | 900 | 49.8 | 67.0 | 54.2 | 31.4 | 52.8 | 64.1 |
| | DINO-DETR [9] | – | 24 | 900 | 50.4 | 68.3 | 54.8 | 33.3 | 53.7 | 64.8 |
| | MS-DETR | – | 24 | 900 | 50.9 | 68.4 | 56.1 | 34.7 | 54.3 | 65.1 |
| | DAC-DETR | – | 24 | 900 | 50.5 | 67.9 | 55.2 | 33.2 | 53.5 | 64.8 |
| | Deformable-DETR++ | w/ Mr.DETR [v1] | 24 | 900 | 51.4 (+1.6) | 69.0 | 56.2 | 34.9 | 54.8 | 66.0 |
| | Deformable-DETR++ | w/ Mr.DETR++ [v4] | 24 | 900 | **52.0 (+2.2)** | 69.1 | 56.4 | 35.4 | 55.5 | 66.6 |
| 对比A | DINO | – | 12 | 900 | 49.0 | 66.6 | 53.5 | 32.0 | 52.3 | 63.0 |
| | Salience-DETR [36] | – | 12 | 900 | 49.2 | 67.1 | 53.8 | 32.7 | 53.0 | 63.1 |
| | Group-DETR [16] | – | 12 | 900 | 49.8 | – | – | 32.4 | 53.0 | 64.2 |
| | MS-DETR | – | 12 | 900 | 50.3 | 67.4 | 55.1 | 32.7 | 54.0 | 64.6 |
| | DAC-DETR | – | 12 | 900 | 50.0 | 67.6 | 54.7 | 32.9 | 53.1 | 64.2 |
| | Cascade-DETR [28] | – | 12 | 900 | 49.7 | 67.1 | 54.1 | 32.4 | 53.5 | 65.1 |
| | Align-DETR [12] | – | 12 | 900 | 50.2 | 67.8 | 54.4 | 32.9 | 53.3 | 65.0 |
| | Rank-DETR [20] | – | 12 | 900 | 50.4 | 67.9 | 55.2 | 33.6 | 53.8 | 64.2 |
| | Stable-DINO [19] | – | 12 | 900 | 50.4 | 67.4 | 55.0 | 32.9 | 54.0 | 65.5 |
| | EASE-DETR [29] | – | 12 | 900 | 49.7 | 67.5 | 54.3 | 32.7 | 52.9 | 64.1 |
| | DINO | w/ Mr.DETR [v1] | 12 | 900 | 50.9 (+1.9) | 68.4 | 55.6 | 34.6 | 53.8 | 65.2 |
| | DINO | w/ Mr.DETR++ [v4] | 12 | 900 | **52.2 (+3.2)** | 69.7 | 56.3 | 35.9 | 55.8 | 67.2 |
| | Align-DETR | w/ Mr.DETR [v1] | 12 | 900 | 51.4 (+1.2) | 68.6 | 55.7 | 33.8 | 54.7 | 66.3 |
| | Align-DETR | w/ Mr.DETR++ [v4] | 12 | 900 | **52.2 (+2.0)** | 69.7 | 56.8 | 34.8 | 55.8 | 66.7 |
| 对比B | DINO | – | 24 | 900 | 50.4 | 68.3 | 54.8 | 33.3 | 53.7 | 64.8 |
| | Align-DETR | – | 24 | 900 | 51.3 | 68.2 | 56.1 | 35.5 | 55.1 | 65.6 |
| | MS-DETR | – | 24 | 900 | 51.7 | 68.7 | 56.5 | 34.0 | 55.4 | 65.5 |
| | DDQ-DETR [37] | – | 24 | 900 | 52.0 | 69.5 | 57.2 | 35.2 | 54.9 | 65.9 |
| | Stable-DINO | – | 24 | 900 | 51.5 | 68.5 | 56.3 | 35.2 | 54.7 | 66.5 |
| | DAC-DETR | – | 24 | 900 | 51.2 | 68.9 | 56.0 | 34.0 | 54.6 | 65.4 |
| | DINO | w/ Mr.DETR [v1] | 24 | 900 | 51.7 (+1.3) | 69.2 | 56.4 | 34.1 | 55.1 | 65.8 |
| | DINO | w/ Mr.DETR++ [v4] | 24 | 900 | **52.3 (+1.9)** | 69.9 | 56.7 | 36.3 | 56.1 | 66.7 |
| | Align-DETR | w/ Mr.DETR [v1] | 24 | 900 | 52.3 (+1.0) | 69.5 | 56.7 | 35.2 | 56.0 | 67.0 |
| | Align-DETR | w/ Mr.DETR++ [v4] | 24 | 900 | **52.4 (+1.1)** | 70.1 | 56.9 | 36.3 | 55.9 | 66.9 |

> ⚠️ Table II 同设置重复行数字不一致：MS-DETR 12ep/900q = 50.0 vs 50.3；DAC-DETR 12ep/900q = 49.3 vs 50.0；MS-DETR 24ep = 50.9 vs 51.7；DAC-DETR 24ep = 50.5 vs 51.2（疑引用不同论文/协议，论文未说明）[未验证]。Fig. 1（p.1）的 +4.0/+4.2/+2.2/+3.2/+1.9/+2.0/+1.1 与上表 7 组 “++ vs baseline” 一一对应。

### 2.2 Table III：COCO val，Swin-L，12ep（p.9；底座 = DINO + IA-BCE loss [12]，900q，H-DETR 例外 300q）

| 模型 | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| H-DETR [8]（300q） | 55.9 | 75.2 | 61.0 | 39.1 | 59.9 | 72.2 |
| DINO [9] | 56.8 | 75.4 | 62.3 | 41.1 | 60.6 | 73.5 |
| Co-DETR [57] | 56.9 | 75.5 | 62.6 | 40.1 | 61.2 | 73.3 |
| Salience-DETR [36] | 56.5 | 75.0 | 61.5 | 40.2 | 61.2 | 72.8 |
| Rank-DETR [20] | 57.6 | 76.0 | 63.4 | 41.6 | 61.4 | 73.8 |
| DAC-DETR [13] | 57.3 | 75.7 | 62.7 | 40.1 | 61.5 | 74.4 |
| EASE-DETR [29] | 57.8 | 76.7 | 63.3 | 40.7 | 61.9 | 73.7 |
| Stable-DINO [19] | 57.7 | 75.7 | 63.4 | 39.8 | 62.0 | 74.7 |
| Relation-DETR [30] | 57.8 | 76.1 | 62.9 | 41.2 | 62.1 | 74.4 |
| **Mr. DETR (ours) [v1]** | **58.4** | 76.3 | 63.9 | 40.8 | 62.8 | 75.3 |
| **Mr. DETR++ (ours) [v4]** | **58.7** | 76.5 | 64.0 | 42.2 | 62.9 | 75.2 |

### 2.3 Table IV（Objects365）/ V（NuImages）/ VI（实例分割）/ VII（全景分割）

**Table IV**（Objects365，R50，900q；正文=只训 **4 epoch**，lr 2e-4 第 3 epoch 衰减 [实测，短日程]）与 **Table V**（NuImages，R50，12ep，900q [实测]）：

| 数据集 | 模型 | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|---|
| O365 | Baseline [11] | 30.4 | 40.8 | 33.1 | 16.1 | 30.1 | 39.1 |
| O365 | w/ Mr. DETR [v1] | 32.7 (+2.3) | 42.7 | 35.8 | 17.1 | 32.3 | 42.6 |
| O365 | w/ Mr. DETR++ [v4] | **34.9 (+4.5)** | 45.4 | 38.2 | 18.3 | 34.3 | 45.7 |
| NuImages | Baseline [11] | 48.5 | 75.1 | 52.3 | 27.4 | 46.1 | 63.0 |
| NuImages | w/ Mr. DETR [v1] | 51.2 (+2.7) | 76.4 | 55.8 | 28.3 | 48.1 | 66.4 |
| NuImages | w/ Mr. DETR++ [v4] | **52.3 (+3.8)** | 75.8 | 56.5 | **25.5** | 49.0 | 69.0 |

> ++ 在 NuImages 的 AP50（75.8<76.4）与 APs（25.5<27.4）低于 Mr.DETR/baseline，论文未讨论。另：**数据集名/引用错**——正文 §IV-A 写 “NuScenes [23]”，Table V/§IV-B.3/App. A 写 “NuImages [23]”，而文献 [23] 是 **nuScenes**；统计（90,000 图、67,279/14,772、1600×900）是 nuImages 量级 → 实际用 nuImages。

**Table VI**（实例分割，Deformable-DETR++ 底座、300q）：

| Epochs | 模型 | Mask AP | Box AP |
|---|---|---|---|
| 12 | Deformable-DETR++ | 32.4 | 46.5 |
| 12 | w/ Mr. DETR [v1] | 36.0 (+3.6) | 49.5 (+3.0) |
| 12 | w/ Mr. DETR++ [v4] | **37.7 (+5.3)** | **50.8 (+4.3)** |
| 24 | Deformable-DETR++ | 35.1 | 48.6 |
| 24 | w/ Mr. DETR [v1] | 37.6 (+2.5) | 50.3 (+1.7) |
| 24 | w/ Mr. DETR++ [v4] | **39.0 (+3.9)** | **51.1 (+2.5)** |

**Table VII**（全景分割，Mask DINO [97] 底座，R50，12ep；o2m 对 ‘stuff’ 标签重复 3 次、不用 calibration）：

| 模型 | Epochs | Queries | PQ | PQth | PQst | APbox | APmask |
|---|---|---|---|---|---|---|---|
| DETR [1] | 500 | 100 | 43.4 | 48.2 | 36.3 | – | 31.1 |
| Mask2Former [98] | 12 | 100 | 46.9 | 52.5 | 38.4 | – | 37.2 |
| Panoptic Segformer [99] | 12 | 353 | 48.0 | 52.3 | 41.5 | – | – |
| Mask DINO [97] | 12 | 300 | 49.0 | 54.8 | 40.2 | 43.2 | 40.4 |
| Mask DINO w/ Mr. DETR [v1] | 12 | 300 | 50.2 (+1.2) | 56.6 (+1.8) | 40.4 (+0.2) | 45.4 (+2.2) | 42.0 (+1.6) |
| Mask DINO w/ Mr. DETR++ [v4] | 12 | 300 | **51.2 (+2.2)** | 57.5 (+2.7) | 41.7 (+1.5) | 46.0 (+2.8) | 42.5 (+2.1) |

### 2.4 Table VIII：路线消融（R50，12ep，900q；由基线 47.6 = Table II 12ep/900q 反推 [推断]）

| 变体 | R-1(FFN) | R-2 | R-3 | R-1(MoE) | MoE-Enc | Calib | AP | AP50 | AP75 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | | ✔ | | | | | 47.6 | 65.8 | 51.8 |
| + Route-1 | ✔ | ✔ | | | | | 49.6 (+2.0) | 67.4 | 54.2 |
| + Route-3 | | ✔ | ✔ | | | | 50.4 (+2.8) | 67.9 | 55.3 |
| **Mr. DETR** | ✔ | ✔ | ✔ | | | | **50.7 (+3.1)** | 68.2 | 55.4 |
| ++ 路线 | | ✔ | ✔ | ✔ | | | 51.0 (+3.4) | 68.7 | 56.1 |
| + calibration | | ✔ | ✔ | ✔ | | ✔ | 51.3 (+3.7) | 68.5 | 55.9 |
| + MoE-encoder | | ✔ | ✔ | ✔ | ✔ | | 51.4 (+3.8) | 68.8 | 56.3 |
| **Mr. DETR++** | | ✔ | ✔ | ✔ | ✔ | ✔ | **51.8 (+4.2)** | 68.9 | 56.7 |

### 2.5 Table IX / X

**Table IX（instruction 实现，R50 12ep 900q，Mr.DETR 设定）**：(a) Independent SA 50.0/67.7/54.8｜(b) Remove SA 49.9/67.3/54.7｜(c) Independent object queries 49.9/67.5/54.6｜(d) Instruction by addition 50.1/67.6/54.7｜**(e) by concatenation (ours) 50.7/68.2/55.4**（AP/AP50/AP75）。

**Table X（encoder scale-aware MoE 尺度选择；++ 设定，12ep 900q）**：

| x64 | x32 | x16 | x8 | AP | AP50 | AP75 | GFLOPs |
|---|---|---|---|---|---|---|---|
| | | | | 51.3 | 68.5 | 55.9 | 424.1 |
| ✔ | | | | **51.8** | 68.9 | 56.7 | 486.8 |
| ✔ | ✔ | | | 51.7 | 68.7 | 56.4 | 503.4 |
| ✔ | ✔ | ✔ | | **51.9** | **69.0** | 56.7 | 569.4 |
| ✔ | ✔ | ✔ | ✔ | 51.5 | 68.6 | 56.2 | 832.5 |

> 选 = 仅 x64（+0.5 AP，+62.7 GFLOPs/+14.8%）；全开 +96.3% GFLOPs 反掉到 51.5。GFLOPs 测量协议（分辨率/是否含 backbone）论文未说明 [未验证]。

### 2.6 Table XIII / XIV / XV + Fig. 11（App. B–E）

**Table XIII(a) instruction token 数**：1→50.2，5→50.1，**10→50.7**，50→50.4，100→50.5。**Table XIII(b) 使用层**（0=仅第 0 层，0~5=全 6 层）：0→50.3，0~1→50.2，0~2→50.4，0~3→50.3，0~4→50.4，**0~5→50.7**。Fig. 10：10 个 token 两两余弦相似度 0.25–0.51 → 数量不敏感（作者解释）。

**Table XIV(a) 编码器 MoE**（AP/AP50/AP75/APs/APm/APl）：

| experts | top-k | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 51.7 | 68.8 | 56.5 | 35.3 | 55.3 | 66.1 |
| 4 | 1 | **51.9** | 68.9 | 56.6 | 35.0 | 55.9 | 67.2 |
| 4 | 2 | 51.8 | 68.9 | 56.7 | 35.6 | 55.9 | 67.0 |
| 4 | 3 | 51.8 | 68.9 | 56.7 | 35.0 | 55.6 | 67.0 |
| 8 | 2 | 51.7 | 68.7 | 56.4 | 35.8 | 55.3 | 66.9 |
| 8 | 4 | 51.7 | 68.7 | 56.5 | 34.9 | 55.6 | 67.0 |

**Table XIV(b) 解码器 MoE**：

| experts | top-k | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 51.4 | 68.5 | 56.0 | 34.3 | 55.2 | 66.1 |
| 4 | 1 | 51.7 | 68.8 | 56.3 | 35.4 | 55.5 | 66.7 |
| 4 | 2 | **51.8** | 68.9 | 56.7 | 35.6 | 55.9 | 67.0 |
| 4 | 3 | 51.8 | 68.9 | 56.7 | 34.4 | 55.5 | 66.9 |
| 8 | 2 | 51.5 | 68.6 | 56.3 | 34.2 | 55.5 | 66.6 |
| 8 | 4 | 51.5 | 68.6 | 56.1 | 34.1 | 55.0 | 67.0 |

> 作者选择：编码器 4e/top-1 AP 最高但 APs 比 4e/top-2 低 0.6 → 编码器 **4e/top-2**；解码器 **4e/top-2** 最好。

**Table XV（校准系数 φ；φ=0 默认）**：

| φ | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AP | **51.8** | 51.8 | 51.8 | 51.8 | 51.8 | 51.7 | 51.7 | 51.6 | 51.6 | 51.6 | 51.5 |
| AP50 | 68.9 | 69.1 | 69.2 | 69.3 | 69.3 | 69.3 | **69.4** | **69.4** | 69.3 | 69.3 | 69.3 |
| AP75 | **56.7** | 56.8 | 56.8 | 56.8 | 56.8 | 56.7 | 56.7 | 56.6 | 56.6 | 56.5 | 56.5 |

**Fig. 11（o2m 超参，曲线图无表）**：K 峰 **6**（AP 50.7）、K>7 掉；α 峰 **0.3**（50.7）；τ 峰 **0.4**（50.7）。

### 2.7 Table XI（训练时间）/ XII（probing）

**Table XI**（8×RTX 3090、batch 16、300q、12ep、每 epoch 分钟）：Deformable-DETR++ 84 AP47.0｜H-DETR 104(+20) 48.7(+1.7)｜MS-DETR 96(+12) 48.8(+1.8)｜**Mr. DETR 101(+17) 49.5(+2.5)**｜**Mr. DETR++ 127(+43) 51.0(+4.0)**。

**Table XII（probing：冻结预训练 Mr.DETR，冻结各路线，只训一个随机初始化分类层）**：

| 模型 | w/o NMS | w/ NMS |
|---|---|---|
| Deformable-DETR++ | 47.6 | – |
| Mr. DETR Route-2 | 50.7 | 50.8 |
| Mr. DETR Route-2 + Probing | 50.3 | 50.5 |
| Mr. DETR Route-1 | 12.8 | 51.0 |
| Mr. DETR Route-1 + Probing | 19.3 | 49.0 |
| Mr. DETR Route-3 | 12.8 | 51.1 |
| Mr. DETR Route-3 + Probing | 20.8 | 48.9 |
| Training with o2m only | 13.3 | 50.2 |
| + Probing for o2o | 13.3 | 50.1 |

> §V 解读：Route-1/3 的 query 已几乎不含 o2o 信息（probe ~20 AP）→ 非共享组件“主动丢弃”o2o 信息以避免冲突；Route-2 probe 50.3 ≈ oracle 50.7。

### 2.8 Table XVI：各路线中间层 Box AP（24ep，实例分割，300q，Deformable-DETR++ 底座；括号=相对对应主路/基线）

| Layer | Deformable++ | Mr.DETR R1 | Mr.DETR R2 | Mr.DETR R3 | ++ R1 | ++ R2 | ++ R3 |
|---|---|---|---|---|---|---|---|
| 0 | 41.3 | 48.2 (+5.6) | 42.6 | 48.2 (+5.6) | 48.7 (+4.9) | 43.8 | 48.8 (+5.0) |
| 1 | 45.1 | 49.6 (+2.5) | 47.1 | 49.7 (+2.6) | 49.8 (+1.6) | 48.2 | 49.9 (+1.7) |
| 2 | 47.2 | 50.1 (+0.9) | 49.2 | 50.2 (+1.0) | 50.3 (+0.2) | 50.1 | 50.6 (+0.5) |
| 3 | 48.1 | 50.2 (+0.2) | 50.0 | 50.3 (+0.3) | 50.9 (+0.0) | 50.9 | 51.1 (+0.2) |
| 4 | 48.6 | 50.5 (+0.2) | 50.3 | 50.4 (+0.1) | 51.1 (+0.0) | 51.1 | 51.2 (+0.1) |
| 5 | 48.6 | 50.4 (+0.1) | 50.3 | 50.4 (+0.1) | 51.1 (+0.0) | 51.1 | 51.0 (−0.1) |

> 浅层（layer 0）辅助路比主路高 5.6 AP，越深差距越小 → 作者论证“主路需更多层才追平带 NMS 的辅助路”。

### 2.9 消融增量汇总（自建）

| # | 消融项 | 对比 | ΔAP | 来源 |
|---|---|---|---|---|
| 1 | 全共享 vs 纯 o2o | Tab.I(2) vs (1) | **−6.0** | Table I |
| 2 | + 独立 SA 辅助路 | 47.6→49.7 | +2.1 | I(3) |
| 3 | + 独立 CA 辅助路 | 47.6→49.2 | +1.6 | I(4) |
| 4 | + 独立 FFN 辅助路 | 47.6→49.6 | +2.0 | I(5) |
| 5 | 两组件独立（共享 SA） | 47.6→49.4 | +1.8 | I(6) |
| 6 | 最优组合 独立 SA+FFN | 47.6→50.0 | +2.4 | I(10) |
| 7 | Route-1（独立 FFN）单独 | 47.6→49.6 | +2.0 | VIII |
| 8 | Route-3（instructive SA）单独 | 47.6→50.4 | **+2.8** | VIII |
| 9 | **Mr. DETR（R1+R3）** | 47.6→50.7 | **+3.1** | VIII |
| 10 | Route-1 FFN→route-aware MoE | 50.7→51.0 | +0.3 | VIII |
| 11 | + calibration | 51.0→51.3 | +0.3 | VIII |
| 12 | + scale-aware MoE（x64） | 51.0→51.4（单独）；51.3→51.8（Tab.X 口径） | +0.4/+0.5 | VIII/X |
| 13 | **Mr. DETR++ 总计** | 47.6→51.8 | **+4.2** | VIII |
| 14 | instruction 数 1→10 / 层 仅0→全6层 | 50.2→50.7 / 50.3→50.7 | +0.5/+0.4 | XIII |
| 15 | instruction 设计：add→concat；独立 SA→instr.SA；去 SA | 50.1→50.7；50.0→50.7；49.9→50.7 | +0.6/+0.7/+0.8 | IX |
| 16 | 编码器开 x64 MoE / 开 x64+x32+x16 | 51.3→51.8 / 51.3→51.9 | +0.5（+14.8% GF）/ +0.6（+34.3%） | X |
| 17 | 编码器 MoE 4e/2t vs 4e/1t | | −0.1 AP / +0.6 APs | XIVa |
| 18 | 校准 φ 0→1.0 | AP 51.8→51.5；AP50 68.9→69.3；AP75 56.7→56.5 | −0.3/+0.4/−0.2 | XV |
| 19 | O365（4ep）/ NuImages（12ep） | +2.3/+4.5；+2.7/+3.8 | | IV/V |
| 20 | 实例分割 mask AP（12/24ep）/ 全景 PQ | +3.6/+5.3；+1.2/+2.2 | | VI/VII |
| 21 | 训练成本（300q 12ep） | 84→101→127 min/ep | +20.2%/+51.2% | XI |

## 3. 训练配方

### 3.1 论文写明的设置（§IV-A/B、App. B–D）

| 项 | 值 |
|---|---|
| Epochs/iters | 12ep=90,000 iter、24ep=180,000 iter @ batch 16；Objects365=4ep |
| Optimizer / lr / wd | AdamW；**2e-4 / 1e-4**（注意与 repo 配置不符，见 3.2） |
| LR 衰减 | ×0.1：12ep 第 11 epoch 后、24ep 第 20 epoch 后；O365 第 3 epoch |
| Backbone | ImageNet 预训练 R50 / Swin-L |
| 数据增强 | 与 H-DETR/DINO/Deformable/Align-DETR 相同（标准 DETR aug） |
| Queries / 评测 | 300 或 900；取 top-100（300q）/ top-300（900q） |
| instruction tokens | **10 个，全 6 个 decoder 层** |
| MoE | **4 experts / top-2**（编+解码器）；解码器“最后三层” |
| o2m 赋配 | **K=6、α=0.3、τ=0.4** |
| DN（DINO 底座） | **100 个 contrastive denoising queries** |
| 校准 φ | 默认 **0** |
| 两阶段 | 基线 = Deformable-DETR++（两阶段+框精修）[推断] |
| 损失权重（论文无表，repo）：o2o 1/5/2、o2m 1/5/2、sep 1/5/2、VFL 1，所有 aux 层同权重 | repo `configs/models/*.py` |

### 3.2 repo 配置逐项（相对 `archive/scratch/mrdetr_src/`）

**公共**：`configs/common/data/coco_detr.py`（RandomFlip + ResizeShortestEdge 480…800/max1333，12ep 版加 RandomCrop 384–600；`total_batch_size=16`）；`coco_schedule.py:82-83`（`lr_multiplier_24ep=default_coco_scheduler(24,20,0)`、`lr_multiplier_12ep=default_coco_scheduler(12,11,0)` → MultiStep [1.0,0.1]、milestone 11/12ep 或 20/24ep、**无 warmup**，与论文“第 11/20 epoch 后 ×0.1”一致）；`train.py:27` **`seed=-1`（不固定随机种子）**。

**v1 Deformable 配方**：`projects/mr_detr_deformable/configs/deformable_detr_r50_12ep.py`：`lr=1e-4`(:35)、`wd=1e-4`(:37)、`batch=16`(:46)、`max_iter=90000`(:14)、`clip_grad max_norm=0.1`(:26-28)；base model `configs/models/deformable_detr_r50.py`：`num_queries=300`、`feedforward_dim=1024`、dropout 0.1、无 two-stage、`select_box_nums_for_evaluation=300`(:109)。`..._two_stage_12ep_plusplus.py`（=900q Mr.DETR）：`with_box_refine/as_two_stage=True`(:5-6)、feedforward 2048(:9-10)、dropout=0(:12-15)、`mixed_selection=True`(:17-18)、`look_forward_twice=True`(:19)、`num_queries=900`(:26)、top-k=300（与论文 900q 口径一致）。`..._plusplus_300q.py`：同上但 300q，**`select_box_nums_for_evaluation=300`（论文说 300q 用 top-100）**。实例分割 `projects/mr_detr_deformable_ins_seg/configs/..._12ep_plusplus.py`：two-stage、`select_box_nums=100`(:24)、`mask_on=True`(:30-31)（与论文 top-100 一致）。Objects365 `..._plusplus_objects365.py:439` `max_iter=int(1728778*4/16-1)=432,194`、`:478` `custom_objects365_pretrain_scheduler(epochs=4, decay_epochs=3, batch=16)`（与正文 4ep 一致、与表题 12ep 矛盾）。v1 Deformable 的 `modeling/` **没有 dn_criterion**（无 DN），与“DINO-based 才用 DN”一致；DINO/Align 路线 `projects/mr_detr_dino/modeling/deformable_detr.py:170` `self.dn_number=100`，`configs/models/deformable_detr_r50.py:115-116` `dn_label_noise_ratio=0.5, dn_box_noise_scale=0.4`。

**++ 配方**（`projects/mr_detr_pp_align/`，仅 Align-DETR 底座）：`configs/dino-swin/dino_swin_large_384_4scale_12ep.py`：`max_iter=90000`(:16)、`lr=1e-4`(:31)、`wd=1e-4`(:33)、`batch=16`(:42)、`amp.enabled=False`(:85，README 建议 CLI 开)、`select_box_nums=300`(:90)、`num_queries=900`(:91)、encoder/backbone `use_checkpoint=True`(:92-93)、**`box_noise_scale=0.4`(:94)、`dn_number=300`(:95)**（:47 先设 100 后被覆盖；README 注“more dn queries, set 300 here”）。Swin-L 实际用 `custom_train_net_swinL.py`：**非 backbone lr=2e-4(:227)、backbone lr=2e-5(:235)、reference_points/sampling_offsets lr=2e-5(:244)、AdamW wd=0.05(:247)**（v1 同款 `projects/mr_detr_align/train_net_swin.py:183,203`）。R50 ++ 配置 `configs/dino-resnet/dino_r50_4scale_12ep.py`：lr/wd 1e-4、900q（model config `dino_r50.py:77`）、`dn_number=100`(:111)、`box_noise_scale=1.0`(:113)、top-k=300。后处理 **model mean**：`projects/modelmean_12ep.py`（0.8·89999 + 0.15·29999 + 0.05·59999）、`modelmean_24ep.py`（0.8·179999 + 0.1·149999 + 0.05·119999 + 0.05·89999）；README 称用 `meanmodel.pth` 评测、**+0.1~0.2%** [repo 自述无表]，**论文没提**。

### 3.3 与 detrex 官方 DINO/Deformable 基线的差异（能核对的部分）

> 官方 detrex 的 `projects/dino`、`projects/deformable_detr` 配置**不在本地快照**（只有库代码 + `configs/common`），逐行对比官方配置**无法核对 [未验证]**；下表对比 repo 内随附的 detrex 风格基线。

| 项 | detrex 风格基线（`deformable_detr_r50_12ep.py` + model config） | Mr.DETR / ++（论文数字配置） | 差异 |
|---|---|---|---|
| 结构 | 单 decoder 1 条路、300q、FFN 1024、dropout 0.1 | 解码器 2–3 条路、900q/300q、FFN 2048、dropout 0 | 结构+参数 |
| 两阶段/框精修 | False | True（+mixed_selection、look_forward_twice） | 结构 |
| 监督 | 纯 o2o | o2o + 2×o2m（K=6,α=0.3,τ=0.4） | 损失 |
| 额外模块 | – | instruction tokens(10)、route-aware MoE(4e/2t)、scale-aware MoE(x64)、VFL IoU 分支(φ 校准) | 参数/计算 |
| lr/wd | 1e-4/1e-4（R50）；Swin-L 2e-4、backbone 2e-5、wd 0.05 | 论文写 2e-4/1e-4；R50 配置 1e-4/1e-4 | **论文≠repo** |
| batch/日程 | 16；90k/180k iter；MultiStep 11/20ep、无 warmup | 同 | 一致 |
| 梯度裁剪 | 0.1 | 0.1 | 一致 |
| AMP | 默认 False，README 训练命令开 True | 论文未提；Swin-L ++ 配置 False | 论文缺失 |
| grad checkpoint | encoder（Swin-L 加 backbone） | 同 | 一致 |
| DN | Deformable 无；DINO 系 100 | 论文 DINO 系 100；Swin-L ++ 配置 300 + box noise 0.4 | **论文≠repo** |
| 评测 top-k | 900q→300；300q→100 | 论文 300q→100；检测 300q 配置→300 | **协议不符** |
| 种子/后处理 | 未固定；无 model mean | 同 + model mean（0.8 末权重+尾部加权） | 论文未写 |

### 3.4 “Objects365 预训练”的两种含义（勿混）

① **论文 Table IV**：Objects365 **训练并评测**（R50/900q/仅 4ep、lr 2e-4 第 3ep 衰减）→ 30.4/32.7/34.9 [实测，短日程]。② **repo README**：Swin-L O365 预训练 **549K iter（batch 16）**→ COCO 5-scale 微调 → **COCO AP 61.8**（79.0/67.6/47.7/65.6/75.7）；配置 `projects/mr_detr_align/configs/deformable_detr_swinl_two_stage_12ep_plusplus_5scale_fintuning.py`（amp=True:149、max_iter=90000*4:150、5-scale、O365 schedule）。**② 是 v1 的 repo 扩展结果，v4 论文里没有** [repo only]。

## 4. 与 one2many 相关工作的定位与数字对比

**定位原话（§I、§II-B）**：DN-DETR/Group-DETR/DINO “utilize **multiple groups of parallel auxiliary queries that share the same transformer decoder** with the primary object query”；H-DETR 用匈牙利匹配**复制 GT** 构造 o2m；DETA 发现 SA 对 o2o 必要、对 o2m 非必要，**DAC-DETR 与 MS-DETR** 据此“incorporate one-to-many auxiliary training **by explicitly constraining self-attention and cross-attention**”；**Co-DETR** “adopts **multiple query groups with different matching strategies** to offer various supervision signals”；本文“Different from previous arts, we discuss the roles of each component ... in the multi-task framework ... propose an **instructive multi-route training** mechanism”。与 Group-DETR 的关系：都“多路并行”，但本文每路是**不同组件分工**的独立前向（Route-2/3 共享参数、Route-1 独立 FFN）且推理丢弃辅助路；Group-DETR 是复制 query 组共享 decoder。

| 方法 | 设定 | AP | 出处 |
|---|---|---|---|
| DN-DETR [15] | – | 论文未给数字（只在 Related Work 提） | §II-B |
| Group-DETR [16] | R50 12ep 900q | 49.8（AP50/75 原表 “-”） | Table II |
| H-DETR [8] | R50 12ep 300q / Swin-L 12ep 300q | 48.7（+Mr.DETR 49.8）/ 55.9 | Table II/III |
| Co-DETR [57] | Swin-L 12ep | 56.9 | Table III |
| MS-DETR [14] | R50 12ep/24ep 900q | 50.0–50.3 / 50.9–51.7（表内重复，见 §2.1） | Table II |
| DAC-DETR [13] | R50 12ep/24ep 900q | 49.3–50.0 / 50.5–51.2（重复） | Table II |
| DINO [9] | R50 12ep/24ep 900q | 49.0→50.9→52.2 / 50.4→51.7→52.3 | Table II |
| Align-DETR [12] | R50 12ep/24ep 900q | 50.2→51.4→52.2 / 51.3→52.3→52.4 | Table II |
| Deformable-DETR++ | R50 12ep 300q/900q、24ep 900q | 47.0→49.5→51.0 / 47.6→50.7→51.8 / 49.8→51.4→52.0 | Table II |
| Mask DINO [97] | R50 12ep 300q（全景） | PQ 49.0→50.2→51.2 | Table VII |

> 论文措辞：300q 上 Mr.DETR “surpasses ... H-DETR, MS-DETR and DAC-DETR by approximately 0.7% mAP”；900q/12ep 的 DINO 系 “surpassing variants such as Group-DETR, DAC-DETR and MS-DETR”；Co-DETR 只在 Swin-L 表里比（56.9 vs 58.4/58.7）。

## 5. 成本与可复现性

### 5.1 论文报了什么成本

| 成本项 | 论文是否给 | 内容 |
|---|---|---|
| 训练时间 | ✅ 仅 Table XI | 每 epoch 分钟（8×3090、batch 16、300q）：84/104/96/101/127 |
| 训练总 GPU 小时 | ❌ | 需自行折算（见 5.2，标注派生） |
| 模型 FLOPs | ⚠️ 仅 Table X | **只给 encoder MoE 尺度消融的 GFLOPs**：424.1/486.8/503.4/569.4/832.5；协议未说明 |
| 参数量 | ❌ | 全文无 params 表 |
| 推理 FLOPs/吞吐/FPS | ❌ | 全文无 |
| 训练卡型（主实验） | ⚠️ 只在 Table XI 说 8×3090 | 正式 COCO/Swin-L 训练用什么卡未说 |

### 5.2 派生估算（**非论文数字**，由 Table XI 折算）

每 epoch GPU·h ≈ 分钟×8/60：baseline **11.2**、Mr.DETR **13.5**、++ **16.9**；12ep 总 ≈ **134/162/203 GPU·h**；24ep ≈ **269/323/406 GPU·h**；相对开销 **+20.2% / +51.2%**（单 epoch）。推理：**Mr.DETR 辅助路丢弃 → 与 baseline 相同**（论文声明）；**++ 不成立**——encoder MoE（x64）+ decoder MoE（论文称最后三层）+ IoU 分支都留在推理图，Table X 自报最高 **+14.8% GFLOPs**（424.1→486.8）→ Abstract/§I “auxiliary routes are discarded → inference time consistent” 对 ++ 只能理解为“辅助路丢弃”，但 **++ 推理确实更贵**，且论文无实测延迟。

### 5.3 代码齐全性（repo 快照 2025-09-06）

| 项目 | 状态 |
|---|---|
| v1 Mr. DETR | ✅ `projects/mr_detr_deformable`（R50 300q/900q/12ep/24ep/O365）、`projects/mr_detr_dino`、`projects/mr_detr_align`（Swin-L 12/24ep、5-scale、O365 finetune）、`projects/mr_detr_deformable_ins_seg` |
| ++ 代码 | ⚠️ 仅 `projects/mr_detr_pp_align`（**Align-DETR 底座**；含 `modeling/moe.py`、`dino_transformer.py`、`two_stage_criterion.py`；配置覆盖 R50/Swin-L 12ep、Swin-L 36ep、ConvNeXt/EVA-01/FocalNet/InternImage/ViTDet，但**只有配置、无 AP**）；**没有 Deformable-DETR 底座的 ++ 项目/配置** → Table II 中 Deformable++ R50 三行（51.0/51.8/52.0）无对应项目；DINO/Align 两列 ++ 行与 pp_align 配置的逐条映射论文/README 未说明 [未验证] |
| modelmean 脚本 | ✅ `projects/modelmean_12ep.py`、`modelmean_24ep.py`（权重平均，见 3.2） |
| converter | ✅ v1 各项目有 `converter.py` + `convert_two_stage.py`；❌ `mr_detr_pp_align/` 没有 converter |
| 训练入口 | `projects/train_net.py` + `projects/mr_detr_pp_align/custom_train_net_swinL.py`（Swin-L 专用 lr 分组）；README 的 Train 段**只写 v1 命令**，无 ++ 训练说明 |
| Table III 的 IA-BCE loss [12] | ❌ 本地快照 `grep ia_bce` 无结果，`configs/models/dino_r50.py:106` 仍是 `focal_loss` → 该表损失实现**未随仓库提供 [未验证]** |
| 推理/FLOPs/FPS 脚本 | 只有 detrex 通用 `tools/benchmark.py`；无 FPS/吞吐评测 |
| 种子 | ❌ 未固定（`configs/common/train.py:27` `seed=-1`） |

### 5.4 权重获取（repo README）

入口 `https://huggingface.co/allencbzhang/Mr.DETR/tree/main`；demo `https://huggingface.co/spaces/allencbzhang/Mr.DETR`。**v4/++ 只发布 1 个权重**：`MrDETR_pp_align_swinL_12ep_900q.pth`（Swin-L/900q/12ep/58.7，配置 `projects/mr_detr_pp_align/configs/dino-swin/dino_swin_large_384_4scale_12ep.py`）。v1 权重：Deformable R50 300q 49.5 / 900q 50.7 / 24ep 51.4；DINO R50 12ep 50.9；Align R50 12ep 51.4 / 24ep 52.3；Align Swin-L 12ep 58.4；Align Swin-L+O365 61.8；实例分割 R50 300q 12/24ep（box 49.5/50.3、mask 36.0/37.6）。**未发布**：Table II 的 ++ R50/DINO/Align 权重（除 Swin-L 58.7 外）、Table VII 全景权重。

### 5.5 单 seed / 方差

论文**通篇没有** seed、重复次数、标准差/误差棒（`grep -i "seed|variance|std"` 只命中致谢 “HKU Seed Fund”）。所有表格、Fig. 7 收敛曲线、Fig. 11 超参曲线、Table XI 计时、Table XII probing、Fig. 8/9/10 都是**单次运行** [论文未声明]；repo `seed=-1` → 数字不可位级复现；同配置两次训练可能差 ~0.1–0.3 AP（本报告推断，非论文结论）。Table XI 计时也是单次、无方差。

### 5.6 论文内部 / 论文↔repo 不一致清单（重要）

1. 基线名 “Deformable-DETR++” 全文未定义，疑似=两阶段+框精修 Deformable DETR（Table II/IV 表题 + repo `*_two_stage_*.py`）。
2. **Fig. 6 caption 把 G/G′ 写反**（caption 说“G governs Route-1, while G′ is shared between Route-2 and Route-3”，但 §III-D 正文与 Eq.7–9、图上标注都是 Route-1 用 G′、Route-2/3 共享 G）。
3. lr/wd：论文 2e-4/1e-4；repo R50 配置 1e-4/1e-4，Swin-L 脚本 2e-4/backbone 2e-5/wd 0.05。
4. 论文“decoder 最后三层用 route-aware MoE”（Fig. 9 画 layer 3/4/5）；repo `dino_transformer.py:188` `custom_layers_index=[5]` 只有最后一层 [需代码同学二次确认]。
5. VFL 目标分：论文 $iou^{0.75}$ vs repo `two_stage_criterion.py:55` `**0.9`。
6. α=0.3（匹配分含分类置信度）：repo `mr_detr_pp_align/modeling/one2manyMatcherDAC.py:275` 是 `C = 1.0*cost_box + 0.0*cost_class`（v1 `mr_detr_deformable` 同函数是 `0.7*box+0.3*class`）[需代码同学确认]。
7. **Table IV 表题写“12 epochs”，正文写“4 epochs only”**（repo 配置也是 4ep）。
8. Table II 同设置重复行数字不同（MS-DETR/DAC-DETR，见 §2.1 注）。
9. 数据集名/引用错：正文 “NuScenes [23]” vs Table V “NuImages [23]”，而 [23] 是 nuScenes。
10. 评测协议：论文 300q 用 top-100，检测 300q 配置用 top-300（`..._plusplus_300q.py:24`）。
11. DN 数量：论文说 DINO 系用 100；Swin-L ++ 配置 `dn_number=300`（`dino_swin_large_384_4scale_12ep.py:95`）。
12. 论文称辅助路丢弃→推理时间与 baseline 一致；但 ++ 的 MoE/校准留在推理图，Table X 自报 +14.8% GFLOPs。
13. 符号小错：Eq.10 用 `TopK(s,k)` 但门控输出写作 c=G(Q)；Eq.13 用 $s_{iou}$、下一句写 $s_{pred}$。
14. MoE 的负载均衡 aux loss（repo `moe.py:33` `aux_loss_alpha=0.01`）论文未提。
15. model mean 后处理（+0.1~0.2 AP，repo 自述）论文未提。

## 6. 可信度分级

- **A. 完整训练实测（最高；但单 seed）**：Table I、II、III（Swin-L 12ep 全量）、VI（12/24ep）、VIII–X（12ep 900q 全量消融）、XIII–XV、XVI（24ep 实例分割）、Fig. 7（12/24ep 收敛）、Fig. 1（7 组 baseline↔++）。判定依据：epoch/query/backbone/评测口径明确、配置与描述大体自洽（例外见 §5.6）。仍**全部单次运行、无方差**；Table VIII–XV 未在表题写 backbone/epochs，本报告按基线数字反推 12ep/900q [推断]。
- **B. 训练实测但日程/协议偏离常规（引用须注明）**：Table IV（Objects365 **仅 4ep**，表题却写 12ep）、Table V（NuImages 12ep）、Table VII（Mask DINO 底座 12ep；非本文行来自文献）、Table XI（8×3090 单次计时）、repo README 的 Swin-L O365→COCO 61.8（549K iter，非论文）、model mean +0.1~0.2%（自述无表）。
- **C. 机理/探查证据（结论是推断，不是训练收益）**：Table XII probing（冻结模型+只训探针分类层；“非共享组件丢弃 o2o 信息”是推断）、Fig. 8 attention 图、Fig. 9 专家激活分布（只 layer 3/4/5，无负载/多样性指标）、Fig. 10 token 余弦相似度。支持“为什么”，不是 AP 收益证据。
- **D. 作者自述/无数据/未验证**：“any independent component can learn both targets”：只有 **AP(+2.1/+1.6/+2.0)** 与 NMS 版 o2m 支持，缺“两目标都学会”的独立度量（probing 只测 o2o 信息残留）；MoE“知识共享 vs 冲突缓解”只有 Fig. 9 + 0.3–0.5 AP，无共享/专门化分别的消融；encoder scale-aware 的省算力动机只有 GFLOPs、无实测速度/显存；“training cost effective trade-off”只有 127 vs 84 min/ep，无“达到同 AP 所需时间”；与 Group-DETR/Co-DETR 只有文字定位、无同设定逐一复现；复现性：seed 不固定、无 IA-BCE 实现、++ 只发布 Swin-L 一个权重、Table VIII–XV 未写设定 → 复现 ++ R50 51.8 需自行改码/猜配置 [未验证]。

## 7. 出处索引

**论文表/图→PDF 页**：Table I p.2｜Fig.3 p.2｜Fig.4 p.5｜Eq.3–6 p.5–6｜Fig.5 p.6｜Eq.7–13+Fig.6 p.7｜Table II p.8｜Table III–VI p.9｜Table VII+Fig.8 p.10｜Table VIII–X+Fig.9 p.11｜Table XI/XII p.12｜Table XIII/XIV+Fig.10/11 p.15｜Table XV p.16｜Table XVI+Fig.12 p.17。

**repo 关键路径（相对 `archive/scratch/mrdetr_src/`）**：v1 = `projects/mr_detr_deformable/`（`configs/deformable_detr_r50_12ep.py`、`..._two_stage_12ep_plusplus.py`、`..._plusplus_300q.py`、`..._plusplus_objects365.py`）、`projects/mr_detr_dino/`、`projects/mr_detr_align/`（`train_net_swin.py`、`objects365_schedule.py`、5-scale/O365 finetune 配置）、`projects/mr_detr_deformable_ins_seg/`；++ = `projects/mr_detr_pp_align/`（`configs/dino-swin/dino_swin_large_384_4scale_12ep.py`、`custom_train_net_swinL.py`、`modeling/{moe,dino_transformer,two_stage_criterion}.py`，其余 backbone 配置在 `configs/dino-{resnet,swin,convnext,eva-01,focal,internimage,vitdet}/`，只有配置无 AP）；公共 = `configs/common/{data/coco_detr.py,optim.py,coco_schedule.py,train.py}`、`projects/modelmean_{12,24}ep.py`、`projects/train_net.py`；权重 = `https://huggingface.co/allencbzhang/Mr.DETR/tree/main`（++ 仅 `MrDETR_pp_align_swinL_12ep_900q.pth`）。
