# Mr. DETR / Mr. DETR++：代码拆解 + 对本项目 DEIMv2 的可迁移性评估（总报告）

**日期**：2026-11-01 · **Lead 合成稿** · **对象**：`https://github.com/Visual-AI/Mr.DETR`（CVPR 2025；v4 = Mr. DETR++，arXiv 2412.10028v4）
**本地材料**：源码快照 `archive/scratch/mrdetr_src/`（main@`287c6935` tarball）· 论文文本 `paper_text.txt` · 详版报告见同目录 `01/02/03_*.md`
**证据分级**：`[实测]` 论文有完整训练数字 · `[代码]` 本机代码直读/仿真 · `[推断]` 结构推算 · `[待验证]` 必须跑实验
**边界**：本报告只做评估与方案，未训练、未改 `vendor/` 与 `docs/` 任何既有文件。

---

## 0. 结论摘要（TL;DR）

1. **是什么**：CVPR 2025，把 DETR 训练当"o2o + o2m 多任务"来做——同一套 backbone/encoder 之上**训练期多跑几条 decoder 路线**（主路 o2o + 独立 FFN 的辅助路 + 指令自注意力的辅助路），推理只保留主路。COCO R50/12ep/900q：**47.6 → 50.7（Mr.DETR，+3.1）→ 51.8（++，+4.2）**；Route-3（指令自注意力单路）单独 **+2.8**，是全表性价比最高的单项。[实测]
2. **与我们的关系**：我们的 decoder 内**没有任何多路/辅助解码支路**，而 o2m 监督只存在于 encoder 侧的外部卷积头（Co-ATSS）。所以 Mr.DETR 补的正是我们的空档，且**训练多路、推理单路 = 单模型单结构，合规**（细则十一（一）第 3 条禁的是"多个不同结构/训练阶段模型"的投票/平均集成）。
3. **唯一推荐试的原子单元**：**Route-3 移植** = 训练期第二条 o2m 解码路线 + self-attn 里的 10 个 instruction token（两者不可拆：只加第二遍 decoder 而组件全共享，就是论文 TABLE I(2) `Share All = 41.6 (−6.0)` 的已知负收益配置）。新增参数约 **33,792**（6 层 ×(2×10×256+2×256)，按论文实现"每层各一份"，不是 03 报告里的全局共享 5,632），**推理零改动**，训练成本按论文 +20%。
4. **诚实预期**：COCO 上 +2.8 AP 是 900 query、两阶段 Deformable 底座、12ep 全量训练的结果；我们是 300 query、D-FINE 底座、已带 Co-ATSS 密集监督与蒸馏、1800 张训练图——**同机制增益可能显著缩水，必须按 20ep 短预算 pilot 判定，不可直接承诺涨点**。[待验证]
5. **不建议做**：route-aware MoE（+0.3 AP / 训练 +31% / 唯一改推理图）、encoder scale-aware MoE（我们的 encoder 仅 1 层且仅 P5，机制位错位）、900 query、零新参数"全共享"接法、搬 DAC rank loss（与 TADL 冲突）、推理期多路融合（不合规）。
6. **附带发现（本次拆解的副产品）**：论文/代码有 15 处不一致（全篇单 seed、无方差；++ 只发布 1 个权重；VFL 指数 0.75 vs 0.9；decoder MoE "最后三层" vs 代码仅最后一层；Fig.6 caption 把 G/G′ 写反等），复现 `++ R50 51.8` 没有现成配置——**引用数字必须带这些限定**。

---

## 1. 方法骨架（论文口径）

检测被当作 **multi-task**：o2o 主任务 + o2m 辅助任务。§III-B 的核心实证（TABLE I）：全共享会**掉 6.0 AP**（47.6→41.6）；而"SA / CA / FFN 任一个独立"就涨 **+2.1 / +1.6 / +2.0**，即使其余组件共享。由此设计三条训练路线（Fig.4）：

| 路线 | 目标 | 参数 | 推理 |
|---|---|---|---|
| Route-2 主路 | o2o（匈牙利） | 与 baseline 相同 | **保留** |
| Route-1 辅助 | o2m（top-k/GT） | 共享 SA/CA，**独立 FFN** | 丢弃 |
| Route-3 辅助 | o2m | **全共享**，self-attn 加 10 个 instruction token | 丢弃 |

- **instructive self-attention**（§III-C, Eq.6）：10 个可学习 token 与 object query **concat** 进 self-attention（长度 m+n），SA 后**丢弃 instruction 位置的输出**。消融（TABLE IX）：独立 SA 50.0｜无 SA 49.9｜独立 query 49.9｜相加 50.1｜**concat 50.7**；token 数 1/5/10/50/100 → 50.2/50.1/**50.7**/50.4/50.5；**全 6 层都用**最好（TABLE XIII）。
- **o2m 指派**：$M=\alpha s+(1-\alpha)\mathrm{IoU}$，每 GT 取至多 K 个正样本、IoU≥τ；**K=6, α=0.3, τ=0.4**。损失归一化 `num_boxes*3`（三路合计≈主路 1 倍）。
- **Mr.DETR++ 三块新增**：① route-aware MoE（替换 decoder FFN，4 experts/top-2，Route-2/3 共享 gate、Route-1 独立 gate，**推理图内生效**）；② encoder scale-aware MoE（只对 x64 尺度，+0.4~0.5 AP、推理 +14.8% GFLOPs）；③ localization-aware score calibration（$s_{cls}^{\phi}s_{iou}^{1-\phi}$，φ=0 默认，**实测不涨总 AP**，只把 AP50 抬到 69.4）。

**关键结论：Mr.DETR（不含 ++）推理与 baseline 同构同开销**。++ 的"推理一致"只在"辅助路丢弃"意义上成立，MoE/校准留在推理图（TABLE X 自报 +14.8% GFLOPs）——论文这句话的适用范围要打折。

---

## 2. 代码实际做了什么（与论文的偏差，迁移前必须知道）[代码]

1. **推理并不裁路**：`deformable_transformer.py:553-595` 无 `if self.training`，官方代码 eval 时**仍跑 3 遍 decoder**（外加两套两阶段 proposal 头），只是结果只取 o2o 最后一层（`deformable_detr.py:405-418`）。要"推理零开销"必须自己加训练期开关——这正好是我们方案的设计点，不是负担。
2. **instruction 是"每层各一份"**：每个 decoder 层的 `CustomMultiheadAttention` 各自持有 `instruction_o2m` / `instruction_o2m_pos`（`transformer.py:72-78`），共 6×(2×10×256)=30,720 参数；K 用 `weight+pos`、**V 只用 weight**（不对称是有意的）；输出丢弃前 10 个 token（`:140-168`）。
3. **魔法常数 `10` / `-900` 绑死**：mask 构造 `[-900:, :10]` 假设"真实 query 是最后 900 行"（`:152-155`）。我们 `num_queries=300`，照抄**不报错但语义静默走样**（负索引被 clamp）。迁移必须写成变量，或按我们的无 DN 配置绕开。
4. **`-900` 那套只在有 DN 时必要**：我们 `num_denoising: 0`，可直接照 `mr_detr_deformable`（无 DN 版）迁移，绕开全部 DN mask 坑。
5. **两阶段强耦合**：o2m 路在 align 里另有一套 proposal 头（`class_embed_o2m_encoder`），且变量只在 `as_two_stage` 分支定义，非两阶段会 `UnboundLocalError`。我们的 D-FINE 是 topk query selection，**不能直接抄**，要自己决定 o2m 路的初始 query 从哪来。
6. **仓库自身缺陷**（不影响我们，但影响"复现论文"判断）：`use_checkpoint` 全部无效（rebind 而非赋值）；encoder scale-aware MoE 的"只对最高 level 生效"是死代码（实际对全部 token 生效）；`output["group"]` 命名误导（实际是 o2m 路）。

---

## 3. 数字与可信度（引用前必读）

| 设定（COCO val） | 基线 | Mr.DETR | Mr.DETR++ |
|---|---|---|---|
| R50 12ep 900q | 47.6 | **50.7 (+3.1)** | 51.8 (+4.2) |
| R50 12ep 300q | 47.0 | 49.5 (+2.5) | 51.0 (+4.0) |
| R50 24ep 900q | 49.8 | 51.4 (+1.6) | 52.0 (+2.2) |
| DINO R50 12ep 900q | 49.0 | 50.9 (+1.9) | 52.2 (+3.2) |
| Swin-L 12ep | 56.8 (DINO) | 58.4 | 58.7 |

- 训练成本（TABLE XI，8×3090 batch16 300q，每 epoch 分钟）：84 → **101（+20%）** → 127（+51%）。论文**无参数量、无推理 FLOPs/吞吐、无总 GPU 时**；唯一 FLOPs 是 encoder MoE（424.1→486.8 GFLOPs）。
- **可信度边界**：全部单 seed、无方差、`seed=-1`；++ 只发布 Swin-L 58.7 一个权重，TABLE II 的 ++ R50 行**无对应配置/权重**；TABLE III 用的 IA-BCE loss 仓库里没有实现；VFL 目标指数论文 0.75 / 代码 0.9；decoder MoE 论文"最后三层" / 代码仅最后一层；α=0.3 在 `pp_align` 赋配代码里是 0.0。→ **把 +3.1/+4.2 当"机制上限"可以，当"我们能拿到的数"不行。**
- 机理证据（probing TABLE XII、attention Fig.8、专家分布 Fig.9）只支持"为什么"，不是收益证据；"any independent component can learn both targets"缺独立度量。

---

## 4. 映射到我们的 DEIMv2：重叠与差集

**我们已有**（逐条 `file:line` 见 03 报告 §1）：

| 已有件 | 落点 | 与 Mr.DETR 的关系 |
|---|---|---|
| Co-ATSS 辅助头（o2m 密集监督） | `co_aux_head.py:67/163`；`deim.py:559` 仅训练期；`deim_criterion.py:727-733` | **重叠在"监督目标"，不重叠在"作用位置"**：它在 encoder 特征挂卷积头，梯度不进 decoder query |
| FDR/go-LSD/DDF | `dfine_decoder.py:476-478`、`deim_criterion.py:594-613` | 无重叠；`loss_local` 的隐式缓存是迁移最大坑 |
| DN（已实现但关闭） | `denoising.py`；`yml:244 num_denoising: 0` | 无重叠；比新 route 更廉价的备选，但会是另一个独立议题 |
| 逐层 aux loss / encoder aux / 第 0 层 pre-head | `dfine_decoder.py:980-983`、`deim_criterion.py:740-794` | 无重叠（都是 o2o 系） |
| one2many matcher / `query_select_method=='one2many'` | `matcher.py:141-161`、`dfine_decoder.py:902-904` | **死代码，全仓库无调用**；把它接到现有 o2o 输出上就是 TABLE I(2) 的 −6.0 错路 |

**差集（唯一值得搬的）**：decoder 内的第二条 o2m 解码路线 + 可学习 instruction token（Route-3），以及可选的独立 FFN 辅助路（Route-1）。

**原子性警告**：Route-3 里"第二遍 decoder"与"instruction token"必须一起上。只加第二遍、组件全共享时，o2m 路输出与 o2o 路完全相同（同一权重同一输入），等价于论文里 −6.0 AP 的 `Share All`。**instruction token 就是那个"独立组件"，不是可选装饰。**

---

## 5. 落地方案（修正 03 报告后的版本）

### 5.1 改动清单（`use_o2m_route` 开关，默认 false）

| # | 文件 | 改动 | 备注 |
|---|---|---|---|
| 1 | `dfine_decoder.py` `TransformerDecoderLayer.__init__` | 每层加 `instruction_o2m` + `instruction_o2m_pos`（各 `(10,256)`）与 `norm_o2m`（`LayerNorm(256)`） | **每层一份**（对齐论文实现）；6 层共 33,792 参数 |
| 2 | 同文件 `TransformerDecoderLayer.forward` | 新增 `instr` 分支：`q=k=cat([ins, q])`、`v=cat([ins_w, target])`、出 `a[:,10:]` + residual `+norm_o2m`；`instr is None` 时**逐字走原路径** | o2o 路径 bit-level 不变是硬约束 |
| 3 | 同文件 `TransformerDecoder.forward` | 把 `value_op`（`:423` 对 75,600×384 memory 的投影）**上提到两条路之外共用**；透传 `o2m/instr` | 否则第二遍白付 ~7.4 GMAC/图 |
| 4 | 同文件 `DFINETransformer.forward` | `if self.training and self.use_o2m_route:` 再调一次 decoder，输出键名 **必须含 `aux`**（`aux_o2m_outputs`，否则 criterion 的 `'aux' not in k` 过滤会把 list 送进 matcher 崩） | 推理路径 0 改动 |
| 5 | `deim_criterion.py` | 新增 o2m 指派（K=6/α=0.3/τ=0.4）+ 损失块，插在 enc_aux 之后、DN 之前；**只接 `tadl` 与 `boxes`，绝不接 `local`**（`loss_local` 首次调用即缓存 FGL 目标与 num_pos，按 `'is_dn'` 区分两套 → o2m 复用会拿到 o2o 目标，loss 有限、`stop_on_nonfinite` 不报 = 静默错） | 归一化 `num_boxes*3` |
| 6 | `configs/deimv2/<new_arm>.yml` | 加开关 + 在 `weight_dict` 登记新损失名 | **criterion 对未登记 loss 静默丢弃**（`deim_criterion.py:716` `if k in self.weight_dict`），漏登记 = 白跑 |

### 5.2 验收门禁（缺一不可）

1. `CUDA_VISIBLE_DEVICES='' $PY scripts/diagnostics/test_clean_baseline_contract.py` 全绿（CPU、不更新参数）。
2. **零步等价**：同一 checkpoint、eval 模式、3 张 val 图，`use_o2m_route` true/false 的 `pred_logits/pred_boxes` **最大绝对差 = 0**（先例：irbind pilot 的零差分验收）。
3. 新写 `test_o2m_route_contract.py`：① `aux_o2m_outputs` 不出现在 `outputs_without_aux`；② eval 逐位等价；③ `loss_local` 缓存不被 o2m 触发。
4. `ROOT=$PWD GPUS="0" REQUIRE_ECC=1 BENCH=1 bash scripts/train/preflight_cloud.sh` 报实测显存/step。

### 5.3 实验设计（受控 A/B）

- **A0 对照**：现 `aic_clean_baseline_v1` 原样（`use_o2m_route=false`），同 seed 42、同 20 ep、同 1800/200 划分。
- **A1 处理**：只差一个开关。
- **预算**：2 臂 × 20 ep ≈ **9.4 h**（RTX PRO 6000 Blackwell 96GB，0.76 s/step、900 step/ep、约 14 min/ep 含验证，据 MODEL_RECORDS §四十二–四十三）。V100 纯 FP32 约 44 h，不建议；本地 15.46 GiB 不能训练（v1 峰值 21.06 GiB）。
- **判据**：主判据配对按图 bootstrap 的 **ΔwAP ≥ +0.4 pp 且 95% CI 不含 0**（噪声下限见 MODEL_RECORDS §三十七）；同时官方 mAP 不低于 A0−0.5 pp、AP75 不低于 A0−2.5 pp；step 中位耗时增幅 ≤25%。
- **证伪**：ΔwAP ≤ 0 或 CI 跨 0；或 mAP 掉 >1 pp；或 `loss_tadl_o2m` 占总 loss >30%（o2m 反客为主）。
- **回滚**：`use_o2m_route: false`，配置一行；新参数不进推理、不需要改 checkpoint。
- **口径红线**：20 ep 结论不能外推 50 ep；阳性后必须 50 ep 复跑（+11 h）才能进提交候选。

---

## 6. 对 03 号报告的两处 Lead 校验订正

1. **参数增量 5,632 → 33,792**：03 号把 instruction embedding 设计成全局共享一份；论文/代码实际是**每层各一份**（`transformer.py:72-78`），6 层 ×(2×10×256 + 2×256) = 33,792。仍可忽略，但实现要按每层写，否则与"论文验证过的配置"不是一回事。
2. **"Co-ATSS 很可能是训练能耗最大单项"不成立（至少待验证）**：冻结的 **DINOv3-ViT-L 教师**每步都在前向（`deim.py:547-550`、`dinov3_teacher.py:117-136`，RGB 下采样 0.5 → 约 2.3k–3.6k token × 24 层），估算 **~2.3 TMAC/step**，高于 Co-ATSS 四层 384→384 3×3 卷积在 P3/P4/P5 上的 **~0.8 TMAC/step**（batch 2，本机静态乘法计数）。因此"关掉 Co-ATSS 可省一半算力"的说法要下调；但它仍是**值得归因**的项（若关掉不掉点，可省 10–25% step 时间）。两者都需 `BENCH=1` profile 实锤。

---

## 7. 不建议做清单（含理由）

| 不做 | 理由 |
|---|---|
| route-aware MoE（decoder） | 论文 +0.3 AP 换 +31% 训练时间、+15.7M 参数，且是唯一改推理图的项（checkpoint/EMA/BF16 数值面最大）；1800 图撑不起"专家分工" |
| encoder scale-aware MoE | 我们 encoder 仅 1 层且仅 P5，机制位错位；放到 P2/P3 是 230K/57.6K token，训练成本不可接受 |
| 900 query 的 o2m 路 | 300q/900q 收益同阶（+2.5/+3.1、+4.0/+4.2），900q 让每层匹配与注意力 ×3，而我们没有匹配开销余量 |
| 零新参数"全共享"接法（含接线 `matcher.py` 的 o2m 死代码） | = TABLE I(2) 41.6 (−6.0)，已知负收益 |
| 搬 DAC rank 重加权 / VFL 换 TADL | 与 TADL（正 target=1.0 + q^β）语义冲突；同时改损失+指派不可归因 |
| 推理期多路平均/NMS/WBF；V23 Stage-A/B 权重融合 | 违反细则十一（一）第 3 条"禁止不同结构/训练阶段模型简单集成" |
| 把 DN 与 o2m route 同时开 | 两个训练期机制一起上，归因归零；DN 是独立议题 |

---

## 8. 合规判定（一句话版）

**训练多路、推理单路 = 合规**（同一 checkpoint、同一优化器、单次前向、无跨模型输出合并；先例 Co-ATSS 训练挂载/推理切除已进纯净基准）。**MoE = 合规**（单模型内条件计算）。**不合规 = 推理期多路输出投票/平均/NMS、两阶段两套权重融合**。依据：细则十一（一）第 3 条原文（`docs/01_competition/` PDF 第 6 页）+ `AGENTS.md` 红线 8 + `PROJECT_SPEC.md` §7.2。

---

## 9. 未验证 / 待确认

1. 我们数据上 o2m route 是否涨点：**零实验证据**，本报告所有收益预期均来自 COCO。
2. 03 的"Co-ATSS ≈4.0×10¹¹ MAC/图"是静态计数（未上机 profile）；教师前向成本同样需实测。
3. 论文数字均单 seed；我们 200 张 val 的噪声（mAP ±1 pp、wAP ±0.4 pp）可能**大于**该机制在小数据上的真实效应——这是本方案最大的"做了也可能测不出"的风险。
4. `detrex/` submodule 在本 tarball 中不完整，无法给"相对 detrex baseline 的逐行 diff"；01 报告用结构级 + 可读文件 diff 替代。
5. 论文未定义 "Deformable-DETR++"，按 repo 配置推断为"两阶段+框精修"版。

---

## 10. 附录：三份详版报告与文件

| 文件 | 内容 | 作者 |
|---|---|---|
| `01_method_code_analysis.md`（448 行） | 代码级拆解：3 路结构/分叉/参数共享矩阵、instruction SA 的 q/k/v 与 mask 构造、Stage2Assigner、MoE 放置、18 条实现陷阱、DEIMv2 落点 | method-code-analyst |
| `02_paper_and_recipe.md`（437 行） | 论文方法原意 + TABLE I–XVI 全部数字 + 训练配方（论文 vs repo）+ 15 条论文↔代码不一致 + 可信度分级 | paper-recipe-analyst |
| `03_aic2026_transfer_plan.md`（291 行） | 现状盘点（13 项）、逐机制映射、合规判定、3 个候选实验、10 项不建议、R1–R12 风险 | transfer-analyst |
| `paper_text.txt` | arXiv 2412.10028v4 抽取文本（1378 行，供复核） | — |

**复核入口**：源码 `archive/scratch/mrdetr_src/`（tarball，主分支 `287c6935`）；本报告所有 `file:line` 均可直接 grep 复核。
