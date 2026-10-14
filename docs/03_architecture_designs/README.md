# 03. 模型架构与算法演进蓝图 (Architecture Designs)

本目录归档 AIC 项目中经历的核心模型架构设计规范，完整记录了从早期加性注入缺陷攻坚，到异构双流工程化，再到 Sovereign V23 零畸变超高清旗舰的**九个阶段架构演进**。

---

## 架构演进九阶段 (Nine Phases of Architectural Evolution)

> 每阶段一篇权威规范，**文件编号即阅读顺序**；指标总账见 [`../MODEL_RECORDS.md`](../MODEL_RECORDS.md)。

| 阶段 | 纲领文件 | 代表模型 | 核心机制 |
| :--- | :--- | :---: | :--- |
| 第 1 部 · 表征解耦 | [01_multimodal_representation_fusion_design.md](./01_multimodal_representation_fusion_design.md) | V1~V3 | 双边池化正交分解 / 动态模态置信门控 / 频域残差傅里叶 |
| 第 2 部 · 异构双流 | [02_cnn_vit_dual_stream_multimodal_design.md](./02_cnn_vit_dual_stream_multimodal_design.md) | V4~V5 | CNN (HGNetv2) + ViT (DAv2) 双流 / 深度图 P0~P3 几何中台 |
| 第 3 部 · 高维手术 | [03_sovereign_v6_structural_optimization_blueprint.md](./03_sovereign_v6_structural_optimization_blueprint.md) | V6~V9 | P2 稀疏微残差直通桥 / Mamba 2D-SSM 颈 / TDFG 逆向反哺门控 / SNR-FDR 锐化头 |
| 第 4 部 · 频域小波 | [04_sovereign_v10_p2_csrg_structural_improvement_ideas.md](./04_sovereign_v10_p2_csrg_structural_improvement_ideas.md) | V10~V11 | Haar 小波 MWHL / 11×1 纵向条带卷积 / CSR 空间显著性门控 / 内生 CAA-SAHI |
| 第 5 部 · 顶会合流 | [05_sovereign_v12_full_blueprint_empirical_validation_report.md](./05_sovereign_v12_full_blueprint_empirical_validation_report.md) | V12 | MHLA (ICLR 2026) / DIAM 交互注意力 / FCR 频域对比损失 / Batch 24 |
| 第 6 部 · 十模块验证 | [06_sovereign_v13_ten_modules_validation_report.md](./06_sovereign_v13_ten_modules_validation_report.md) | V13 | 十模块解耦先锋验证 / LFP 低频引导净化 / PBD 峰值背景解耦 / MixStyle |
| 第 7 部 · 异构解耦 | [07_sovereign_v14_heterogeneous_paradigm_validation_report.md](./07_sovereign_v14_heterogeneous_paradigm_validation_report.md) | V14 | StarConv 乘法星操作 / Spatial-PBD 水空门控 / Point-Preserving LFP |
| 第 8 部 · 连续流形 | [08_sovereign_v20_master_architecture_spec.md](./08_sovereign_v20_master_architecture_spec.md) | V20 | FCSPM 频域锁相 / 3D-PPR 局部突出度 / 保边深度修复 / 52:48 参数流形汤 |
| 第 9 部 · 零畸变合流 | [09_sovereign_v23_master_architecture_spec.md](./09_sovereign_v23_master_architecture_spec.md) 👑 | V23~V28 | 1152×2048 零插值畸变 / TADL 任务对齐解耦 / NWD-QFL 连续损失 / **两段式训练**（补标 72 轮预训练 → 旧标注 TADL 精调 21 轮） |

> `09_sovereign_v23_master_architecture_spec.md` 为软链，正文唯一权威版本是 [`../SOVEREIGN_V23_DETAILED_TECHNICAL_SPEC.md`](../SOVEREIGN_V23_DETAILED_TECHNICAL_SPEC.md)。

---

## 文档详情与核心贡献

### 1. [01_multimodal_representation_fusion_design.md](./01_multimodal_representation_fusion_design.md)
- **主题**：基于表征学习的多模态解耦融合规范
- **解决痛点**：
  - 传统 PromptBlock 简单的加性注入造成特征无界发散；
  - 白天/弱热斑场景下强行注入 IR 噪声导致严重负迁移；
  - 冻结强模态（DINOv3）与从头训练弱模态导致的冷启动抑制（AP 从 0.52 跌至 0.004）。
- **核心机制**：
  - 双边池化正交分解（Bilateral Pooling & Disentangled Representation）；
  - 模态有效性门控（Modality Confidence Gate）；
  - 频域残差傅里叶引导（RFGM）。

### 2. [02_cnn_vit_dual_stream_multimodal_design.md](./02_cnn_vit_dual_stream_multimodal_design.md)
- **主题**：基于 CNN + ViT 异构双流的多模态目标检测架构设计规范
- **解决痛点**：
  - 单纯依赖 ViT 或 CNN 的感受野与边缘定位矛盾；
  - 深度图（Depth）直接当图像吃导致空间几何信息丢失。
- **核心机制**：
  - HGNetv2-B5 高效局部骨干与 DAv2 ViT 全局几何表征深度协作；
  - P0~P3 深度优化中台（拟合空间修复、地面平原减除、高斯法向量场计算、几何 Token 缓存）；
  - 最终落地成成就 56.77% 历史战果的 **Sovereign V5** 架构基石。

### 3. [03_sovereign_v6_structural_optimization_blueprint.md](./03_sovereign_v6_structural_optimization_blueprint.md)
- **主题**：Sovereign V6 架构缺陷解剖与下一代结构优化技术蓝图
- **解决痛点**：
  - 小目标空间与语义断层（P2-Stride 4 被完全丢弃，导致 $AP_{small} \approx 29\%$）；
  - 融合阶段纯单向“盲前馈”（缺乏深层 Query 语义逆流反哺）；
  - D-FINE 边界概率分布在极端暗区低信噪比下的熵增发散。
- **核心机制（4 大结构手术）**：
  - **MSRB（Micro-residual Sparse Residual Bridge）**：P2 稀疏微残差直通桥；
  - **Mamba-CCFM**：2D-SSM 线性全局感受野混血特征颈；
  - **TDFG（Top-Down Feedback Gate）**：深层 Query 逆向感知门控；
  - **SNR-Aware FDR Head**：暗区信噪比感知的分布锐化检测头。

### 4. [04_sovereign_v10_p2_csrg_structural_improvement_ideas.md](./04_sovereign_v10_p2_csrg_structural_improvement_ideas.md)
- **主题**：Sovereign V10/V11 (P2-CSRG+) 结构进阶与顶会即插即用改进日志
- **解决痛点**：
  - 交通标牌 11.8% 细长高耸立柱（$w/h \approx 0.20 \sim 0.35$, 1000px+）在 P3 尺度物理塌陷至 0.125px，造成立柱断层与严重漏检；
  - 水面微型弱小船只受到波纹反光与漫反射伪边缘干扰；
  - 传统对称池化在天空与平坦水面各向同性平滑注入高频噪声。
- **核心机制（4 大顶会即插即用手术）**：
  - **MWHL 小波下采样 (V1-246)**：Haar DWT 提取 HL 垂直高频子带注入，阻断立柱信号衰减；
  - **非对称垂直条纹卷积 (V1-058)**：$11\times1$ 垂直大核感知长程纵向立柱连续性；
  - **CSR 空间显著性门控 (V1-248)**：根据空间频率显著性动态调节高频注入，平坦水面与背景强制截断；
  - **PFM 傅里叶相位滤波 (V1-249)**：利用相位谱刚体几何不变性滤除水面反光杂波。

### 5. [05_sovereign_v12_full_blueprint_empirical_validation_report.md](./05_sovereign_v12_full_blueprint_empirical_validation_report.md)
- **主题**：Sovereign V12 全链路顶会架构合流与终极实证报告
- **核心指标突破**：
  - 验证集 **mAP 达到 58.29%（0.58288）历史最高纪录**！
  - **AP50 达到 86.60%**，**AP75 达到 63.68%**。
  - **小目标 $AP_{small}$ 突破至 30.84%**，彻底攻克小目标断层历史瓶颈。
- **核心机制与系统集成**：
  - **V1-218 MHLA 多头线性注意力 (ICLR 2026)**：正向特征映射 $\phi(x)=\text{ELU}(x)+1.0$ 实现 $O(N)$ 复杂度，与 Mamba 2D-SSM 并联构成双核超长程特征颈；
  - **V1-153 DIAM 可变形交互注意力 (一区TOP 2025)**：Query 语义与几何宽高比 $w/h$ 联合调制，将采样点紧锁于立柱骨架与刚体龙骨；
  - **V1-054 傅里叶对比正则化损失 (ECCV 2024)**：cuFFT 1D 离散傅里叶频域对比约束 FDR 边界分布，推理零开销；
  - **天水深度梯度去噪器**：依据垂直坐标 $Y$ 与深度梯度 $|\nabla Z|$ 将天空与水面高频增益强行归零；
  - **CAA-SAHI 3-View 内生化**：将切片与逆映射完全嵌入 `DEIM.forward()`，彻底告别外部 Python 脚本；
  - **Batch 24 极致性能释放**：RTX PRO 6000 96GB 显存占用 70.2GB，8 轮训练仅耗时 15 分 55 秒。

### 6. [07_sovereign_v14_heterogeneous_paradigm_validation_report.md](./07_sovereign_v14_heterogeneous_paradigm_validation_report.md)
- **主题**：Sovereign V13/V14 异构融合与解耦先锋验证
- **核心机制**：
  - StarConv 乘法星操作高阶特征交互先验；
  - Spatial-PBD 高度水空门控与 Point-Preserving LFP 低频引导净化；
  - 终极模型汤融合，小目标 AP 首次单模破 31.12%。

### 7. [08_sovereign_v20_master_architecture_spec.md](./08_sovereign_v20_master_architecture_spec.md)
- **主题**：Sovereign V20 全物理要素与跨模态连续流形终极合流架构规范
- **全项目历史极值突破**：
  - **官方 AP75 冲至 0.6577** 创全项目历史最高纪录（打破 0.6570 物理天花板 🏆🏆）；
  - **微小目标精度 $AP_{small}$ 达到 0.3127 ~ 0.3131** 历史最高峰；
  - **微小目标召回率 $AR_{small}$ 达到 0.4343 ~ 0.4612** 全域第一；
  - **置信度 $\ge 0.50$ 硬过滤 mAP 达 0.5296，AP50 达 0.7595 全线夺冠**；
  - **单图目标密度压至 6.75**，物理归零路面、倾斜草坡与水面反光虚警！
- **三大物理基石**：
  - **FCSPM (频域相位锁定)**：傅里叶正交频域将红外完整幅值与可见光高频相位正交结合，彻底消除空域晕影；
  - **3D-PPR (局部流形突出度残差)**：拟合局部 3D 背景平面切空间，计算欧氏垂直距离，地表完全归零；
  - **RGB 引导保边深度修复**：废除 Telea/NS 假斜坡，保边扩展修复传感器反光空洞；
  - **Pure-LMM 52:48 连续流形融合**：权重超平面鞍点融合，零 Hook 纯原生前传。

### 8. [09_sovereign_v23_master_architecture_spec.md](./09_sovereign_v23_master_architecture_spec.md) 👑👑👑
- **主题**：Sovereign V23 零插值畸变超高清 2048 物理流形与全链路终极白皮书
- **全链路革新与官方实战成果**：
  - **零插值畸变空间对齐（Zero Distortion $1152 \times 2048$）**：废除一切破坏微小物体物理边缘的多次插值缩放，单次双线性重映射严格对准相机传感器。
  - **自主高阶物理多模态表征**：6 通道输入（RGB 3 通道 + 各向异性去光晕 IR 1 通道 + LBG-IR 局域背景引导 1 通道 + P0 连续公制深度 1 通道），坚守自主技术路线，彻底拒绝竞对粗糙残差。
  - **NWD-QFL 高重叠连续损失优化器**：Wasserstein 距离度量亚像素级小目标微弱交并比，联合 Quality Focal Loss 将分类置信度与定位质量连续绑定，根治硬阈值二值截断。
  - **两段式训练（交付版为第二段）**：Stage A「补标口径预训练」按 72 轮排程（前 52 轮满负荷增广，第 52 轮冻结 Mixup/CopyBlend，第 58 轮切除扰动）；Stage B「官方旧标注 TADL 精调」从 Stage A 的 `best_stg2.pth` 热启动，**24 轮排程实跑 21 轮**、末 4 轮关增广 —— 这才是 0.8964 的交付权重。
  - **Compact 压框法则实战登顶**：官方复赛 4 次提交实测验证，以 7,169 框（单图 7.17 框，精准对齐官方 7.09 真实密度）一举斩获 **53.555 分锁定官方天梯榜前列第 3/4 名**！
  - **96GB 极限算力释放**：物理 Batch 8 稳态占用 85.6GB / 96GB（89.2% 满载利用），单步 1.38 秒，单轮 ~5.8 分钟高速收敛，直奔 58 ~ 60+ 巅峰！

---

## 演进里程碑对照一览

| 阶段 | 架构代号 | 核心算子 | 验证集 mAP | 验证集 AP50 | 关键性能突破点 |
| :---: | :---: | :--- | :---: | :---: | :--- |
| **阶段一** | Sovereign V1~V3 | 双边池化正交投影 + DCDA + DOG + 3D-Lift | 56.12% ~ 56.40% | 84.20% ~ 84.82% | 克服加性无界发散，融入法向量与深度阶跃 |
| **阶段二** | Sovereign V4~V5 | RFGM 频域相位残差 + MAL2 + GIoU 3.5x | 56.67% ~ 56.77% | 85.10% ~ 85.60% | 细颗粒度单图精炼，`sign` 跃升至 43.11% |
| **阶段三** | Sovereign V6~V9 | Mamba 2D-SSM + MSRB 直通桥 + SNR-FDR | 57.08% ~ 57.73% | 85.92% ~ 86.42% | 引入长程状态空间扫描，解决极端暗区熵增 |
| **阶段四** | Sovereign V10~V11| P2-CSRG+ + Haar 小波 + 11x1 纵向核 + 内生 SAHI | 57.29% ~ 58.10% | 86.10% ~ 86.40% | 立柱断层修复，天水去噪，SAHI 迁入模型内 |
| **阶段五** | Sovereign V12 | MHLA (ICLR26) + DIAM + 傅里叶损失 + Batch 24 | 58.29% | 86.60% | 全链路顶会技术合流，小目标 30.84% |
| **阶段六** | Sovereign V14 | StarConv + Spatial-PBD + LFP + 双模模型汤 | 58.30% | 86.53% | 乘法星操作高阶特征交互，小目标单模 31.12% |
| **阶段七** | Sovereign V20 | FCSPM 锁相 + 3D-PPR 局部突出度 + 保边深度 + 52/48 连续流形 | 0.5845 / 0.5835<br>(0.5过滤 0.5296) | 0.8616 / 0.8604<br>(0.5过滤 0.7595) | AP75 冲至 0.6577 创全域历史最高峰！小目标 APs 0.3127！0.5过滤下 6.75 单图检出，地表虚警归零！ |
| **阶段八** | **Sovereign V23** 👑👑👑 | **2048 零插值畸变 + 各向异性去光晕 6 通道 + NWD-QFL + 两段式(72轮预训练→21轮TADL精调) + Compact 压框** | **0.5408** (仅E6初收敛)<br>目标 **0.585 ~ 0.60+** | **87.35%** (人)<br>**90.79%** (车) | **官方实战锁定 53.555（前三甲/第四）！零畸变重构，85.6GB/96GB 极限吞吐，全速冲击 60+！** |

