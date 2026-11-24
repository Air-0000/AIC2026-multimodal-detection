# Mr.DETR → AIC2026 单模型 DEIMv2：机制映射、合规判定与受控实验方案

**日期**：2026-11-01 · **作者**：transfer-analyst（task-3）· **对象**：`vendor/DEIMv2` 纯净基准 v1（`configs/deimv2/aic_clean_baseline_v1.yml`）
**源码**：`archive/scratch/mrdetr_src`（Mr.DETR / Mr.DETR++） · **论文文本**：`outputs/reports/mrdetr_2026-11-01/paper_text.txt`
**证据等级**：`[验证]`=本仓库实测（MODEL_RECORDS/CLAUDE 有记录）· `[论文]`=CVPR2025 论文自报（COCO/118k，不可直接搬）· `[推断]`=读代码/静态计数 · `[待验证]`=必须跑实验才能确定。
**边界**：本文件只做分析与方案，**不训练**、**不改 `vendor/` 与 `docs/` 任何文件**。撰写时同伴产出 `01_method_code_analysis.md` / `02_paper_and_recipe.md` 尚未落盘，若后到则以它们为准的源码/论文细节，本文件负责**迁移映射 + 合规 + 实验设计**。

---

## 0. 结论摘要（TL;DR）

1. **我们的 o2m 监督在 encoder 锚点侧，不在 decoder query 侧**：Co-ATSS 是挂在 encoder 输出（P3/P4/P5）上的独立卷积头，`deim.py:559-561` 只在 `training` 时调用，梯度进主干/encoder，**不进 decoder query**。Mr.DETR 的 (a)(b)(c) 全部作用在 decoder 内。这就是两者**不重叠**的部分，也是唯一值得搬的东西。
2. **仓库里已有的 one2many 机件是死代码**：`matcher.py:141-161`（`get_top_k_matches`/`indices_o2m`）与 `dfine_decoder.py:902-904`（`query_select_method=='one2many'`）全仓库无调用，任何配置也都没开（`aic_clean_baseline_v1.yml:255` 用 `default`）。**不要用它**——把它接到现有 o2o 输出上，正是论文 TABLE I 的 `(2) Share All = 41.6 (−6.0)` 已知负收益配置。
3. **成本对比反转了直觉**：现有 Co-ATSS 在本分辨率下估算 **≈4.0×10¹¹ 乘加/图（前向，静态计数）**，很可能是训练能耗最大单项（需 profile 确认）；而新增一条 300-query o2m 解码路线只有 **≈2×10⁹ 乘加（<1% 总前向乘加）**、`+5,632` 参数、推理零开销。**贵的不是 Mr.DETR，是我们已有的辅助头。**
4. **合规**：训练多路/推理单路 **合规**；MoE **合规**（单模型单前向、一个 checkpoint）；**不合规**的是"推理期把多路输出平均/投票/NMS 合并"与"Stage-A/Stage-B 两套权重融合"（细则十一（一）第 3 条 + AGENTS 红线 8）。
5. **性价比排序**：先做 **(b) 指令自注意力单路（Route-3，+5,632 参数）**；同批做 **Co-ATSS 归因闸门**（可能直接省掉一半训练算力）；**(d) MoE 与 900-query 一律不做**。三个实验共用的对照臂只跑一次，3 臂 ≈ 14 h（Blackwell 96GB）。
6. **最大风险不是显存，是 `loss_local` 的隐式缓存**（`deim_criterion.py:490-497`、`:523-525`）：o2m 输出若复用 `local` 损失会拿到 o2o 的 FGL 目标，**静默错**且 loss 有限、守卫不报。

---

## 1. 现状盘点：我们已有的"one2many / 辅助监督"件

| # | 件 | 落点（`path:line`） | 一句话作用 | 与 Mr.DETR 的重叠 |
|---|---|---|---|---|
| 1 | **Co-ATSS 辅助头** | `engine/deim/co_aux_head.py:67`（类）、`:163`（forward）、`:233-285`（每 GT 每层 3×3 候选 + ATSS `mean+std` 阈值）、`:296-307`（冲突消解：最高 IoU 独占、分类回归同一 owner）、`:329-337`（`num_pos_norm=max(num_pos,4)`） | 在 encoder 输出 P3/P4/P5 上做 one-to-many 密集正样本监督，训练期挂载、推理期切除 | **对应 (a) 的"监督目标"**，但作用面在 encoder 锚点、且与 decoder 解耦；Mr.DETR 的 o2m 是 decoder 内的第二条 route |
| 2 | Co-ATSS 接线 | `engine/deim/deim.py:478-488`（构造，`in_channels=encoder.hidden_dim=384`）、`:559-561`（`if self.training and use_co_aux and targets`）、`deim_criterion.py:727-733`（`loss_co_cls`×1.0 / `loss_co_reg`×2.0）；配置 `yml:184-186,281-282`，lr 组 `yml:146-147` | 逐机制接线范式：**只在训练期进 outputs、由 criterion 统一乘权** | Mr.DETR 的辅助损失接线方式与我们相同，可直接复用这套范式 |
| 3 | one2many matcher（**死代码**） | `matcher.py:59`（`return_topk` 形参）、`:141-142`（返回 `indices_o2m`）、`:146-161`（`get_top_k_matches` 迭代 Hungarian + `1e6` 屏蔽） | 名义上提供"第 2..k 个 Hungarian 匹配"作为 o2m 指派 | 对应 (a) 的**指派器**，但 (i) 无调用者、(ii) 用的是 Hungarian 二次求解而非论文的 greedy ATSS/`M=α·s+(1-α)·IoU`，语义不同 |
| 4 | `query_select_method=='one2many'`（**死代码**） | `dfine_decoder.py:902-904`（`topk_ind = topk//num_classes`）；配置一律 `default`（`yml:255`） | 名义上的 o2m 版 query 选择 | 与 Mr.DETR 无关（它们两路共用 object queries，见论文 Fig.4 "We share object queries"） |
| 5 | FDR / FGL / DDF（细粒度定位） | `dfine_decoder.py:476-478`（`pred_corners` 逐层残差 + `distance2bbox`）、`deim_criterion.py:478`（`loss_local` 同时算 FGL+DDF）；权重 `yml:278-279`（0.25 / 1.5） | 把框回归改成 4×33 分布 + 分布蒸馏 | 无重叠；但它**占用 `local` 损失名并写缓存**，是 o2m 接线的最大坑（见 R5） |
| 6 | go-LSD（层间统一匹配集） | `deim_criterion.py:594-613`（`_get_go_indices`）、`:41`（`use_uni_set=True`）、`:711-713/:746-748/:762-763`（仅对 `boxes`/`local` 用并集 indices）、`:615`（`loss_secl`，配置关 `yml:316`） | 让 box 损失用各层匹配并集，训练早期更稳 | 与 Mr.DETR 的 (c) 共享策略同属"多路监督融合"，但 go-LSD 只融合**匹配集**不新增前向，成本≈0，**优先级高于任何新 route**（已启用） |
| 7 | DN（去噪查询） | `denoising.py:12`（`get_contrastive_denoising_training_group`）、`dfine_decoder.py:929-940`（构造）、`:961-970`（与主 query 切分）、criterion `:799+`；**配置 `yml:244` `num_denoising: 0` → 关闭** | 加噪 GT 查询做辅助回归/分类 | 与 Mr.DETR 的辅助路是同类"训练期加一路监督"；**DN 已实现但关闭**，是比新 route 更便宜的备选（若要做，属于另一种机制） |
| 8 | encoder 级辅助损失 | `dfine_decoder.py:982`（`enc_aux_outputs`）、`deim_criterion.py:773-794`（class-agnostic 目标、`_enc_i` 后缀）；encoder 只 1 层（`yml:215-217`，仅 P5，`hybrid_encoder.py:756-768`） | 在 query 选择前对 topk 打分做监督 | 与 Mr.DETR 的 o2m **encoder 侧**监督部分重叠（我们是 class-agnostic + Hungarian，它们是 anchor ATSS） |
| 9 | 逐层 aux loss | `dfine_decoder.py:980`（`aux_outputs`，5 层）、`deim_criterion.py:740-755`（每层单独 Hungarian，`:685`） | 每层独立匹配监督 | 无重叠（这是 o2o 的深层监督） |
| 10 | D-FINE `pre_outputs` | `dfine_decoder.py:983`、`:470-473`（第 0 层 `pre_bbox_head`）；criterion `:757-771`（`_pre` 后缀） | 第 0 层的传统头辅助监督 | 论文 TABLE XVI 显示 o2m route 在**浅层**领先最多（layer0 +5.6 AP），我们第 0 层已有独立头——**部分替代关系，需在实验里作为混杂因素记录** |
| 11 | query 选择（P3 配额） | `dfine_decoder.py:858-922`（`_select_topk`）、`:888`（`p3_k = int(topk*0.35)`，`yml:261`）；SDQS/distinct **fail-closed**（`:628-631`） | 35% 查询保底给 P3 + 全局补齐 | 与 Mr.DETR 的两阶段 proposal 生成（`deformable_transformer.py:475-537`）功能位对应但实现完全不同（我们是 encoder topk 单阶段） |
| 12 | TADL 标签指派损失 | `deim_criterion.py:245-346`（正样本 target=1.0、`q^β` 质量加权、稀有类负样本抑制）；权重 `yml:284`（2.0）；NDSL 关 `yml:302` | 自定义正/负加权 BCE（VFL 变体） | 与 (a)(b) 无重叠，但**决定了 o2m 损失必须怎么接**：TADL 与指派无关（任意 `(fg,gt)` 对都成立）→ 可直接复用于 o2m indices（见 §3.1） |
| 13 | LQE 分数精修 | `dfine_decoder.py:314`（类）、`:387`（每层一个）、`:482`（训练/推理都用） | 用 corner 分布精修分类分数 | 与 Mr.DETR 的 (d) 子项 **localization-aware score calibration** 功能重叠 → 该子项对我们价值最低（见 §6） |

**没有的东西（与 Mr.DETR 的差集）**：decoder 内的 o2m route；可学习 instruction token；route 级参数共享策略；decoder/encoder 的 MoE；GT 复制式 dense O2O（`yml:17-49` 只有翻转/缩放/清洗/归一化，无 copy-paste/mosaic）；DETR 意义的两阶段/级联（V23 的"两阶段"是**两次训练**，见 `MODEL_RECORDS.md:874` 与 AGENTS §1，不是检测级两阶段）。

---

## 2. Mr.DETR 机制摘要（只留迁移需要的）

三条训练 route（论文 Fig.4、`paper_text.txt:420-450`）；**推理只留 Route-2，辅助路丢弃**（`paper_text.txt:449`）：

- **Route-2**（主路，o2o）：标准 decoder 前向，与 baseline 完全相同。
- **Route-1**（辅助，o2m）：共享 SA/CA、**独立 FFN**。源码 `transformer.py:371-382`（`CustomLinear.linear_sep`）+ `:340-343`（`sep=True` 路由）。
- **Route-3**（辅助，o2m）：共享**全部**参数，只加 **instruction token**。源码 `transformer.py:74-75`（`nn.Embedding(10, embed_dim)` ×2）、`:140-168`（concat → SA → 丢掉前 10 个 → `norm_o2m`；value 用 weight 而非 weight+pos）。
- o2m 指派：`one2manyMatcherDAC.py:223-302`（`Stage2Assigner(num_queries=900, max_k=6)`，`Matchers([0.4])`，成本 `0.8·IoU+0.2·p`），损失归一化 `num_boxes*3`（`deformable_criterion.py:115-128`），正样本按同 GT 内 rank 再乘 `exp(-rank/1.5)`（`:199-205`）。
- 超参（论文）：instruction token **m=10**（`paper_text.txt:681`）、o2m 指派 **K=6, α=0.3, τ=0.4**（`:685`）、Mr.DETR++ MoE **4 experts / top-2 / 2 shared**（`:682`）。

**论文量化（300 query Deformable-DETR++，12 ep；`paper_text.txt:209-232, 854-869, 945-957`）**：

| 形态 | AP | 训练时间/epoch | 我们关心的点 |
|---|---|---|---|
| (1) 只 o2o | 47.6 | 84 min | 现状对应物 |
| (2) **全共享 + o2m 损失** | **41.6 (−6.0)** | — | ⚠️ **零新参数的最省事接法会掉 6 个点** |
| (3) 独立 SA | 49.7 (+2.1) | — | |
| (5) 独立 FFN | 49.6 (+2.0) | — | = Route-1 |
| Route-3（指令 SA）单路 | **50.4 (+2.8)** | ~93 min（推算 +9） | **性价比最高** |
| Route-1 + Route-3（Mr.DETR） | 50.7 (+3.1) | 101 min (+17) | |
| + route-aware MoE | 51.0 (+3.4) | 127 min (+43) | MoE 相对 Mr.DETR 仅 **+0.3**，却多 **+26 min** |
| + calib / + encoder-MoE | 51.3 / 51.4 | — | 各 +0.3 / +0.4 |

> 口径注：AP 取 TABLE I/VIII（其 o2o-only 基行为 47.6），训练时间取 TABLE XI（该表基行为 47.0，+84 min/epoch）——论文内部这 0.6 AP 的基行差异未解释；本表只做**增量**解读，绝对值不可与 TABLE II 混读。

---

## 3. 逐机制映射（落点 / 最小 diff / shape / 参数 / 接线）

统一前提：`hidden_dim=256`、`num_queries=300`、`num_layers=6`、`dim_feedforward=1024`、`reg_max=32`（`yml:239-248`）；1440×2560 下 encoder 输出 **P3 (B,384,180,320) / P4 (B,384,90,160) / P5 (B,384,45,80)**（`hybrid_encoder.py:790-798`），decoder memory 展平后 **75,600 token**（57,600+14,400+3,600）。

### 3.1 (a) 辅助 o2m 解码路线 —— 复用同一份 encoder top-300 查询

**落点**：`engine/deim/dfine_decoder.py:924`（`DFINETransformer.forward`）+ `:406`（`TransformerDecoder.forward`）。**不动** `_get_decoder_input`（`:807`）——直接用现有 `init_ref_contents`/`init_ref_points_unact`（`:942-943`），对应论文"We share object queries"。

```python
# dfine_decoder.py:946-959 之后（o2o 前向完全不变）
if self.training and self.use_o2m_route:
    with torch.autocast('cuda', enabled=False):          # 与 criterion 一致：FP32
        o2m = self.decoder(init_ref_contents, init_ref_points_unact, memory, spatial_shapes,
                           self.dec_bbox_head, self.dec_score_head, self.query_pos_head,
                           self.pre_bbox_head, self.integral, self.up, self.reg_scale,
                           o2m=True,                       # 新增：进入指令自注意力分支
                           instr=self.instruction_o2m)     # 新增：nn.Embedding(10,256)
    out['aux_o2m_outputs'] = [{'pred_logits': l, 'pred_boxes': b, 'pred_corners': c,
                               'ref_points': r} for l, b, c, r in zip(*o2m)]   # 6 层
```

- **shape**：`init_ref_contents (2,300,256)`、`init_ref_points_unact (2,300,4)`；o2m 每层 `pred_logits (2,300,12)`、`pred_boxes (2,300,4)`、`pred_corners (2,300,132)`、`ref_points (2,300,4)`。
- **参数增量**：**0**（头与层全共享）。
- **耗时增量 `[推断]`**：decoder FFN+SA+头 在 300 query 上 ≈ **1.8×10⁹ 乘加/图/路**（对照 Co-ATSS ≈4.0×10¹¹），**不到 step 的 1%**；但**必须复用 `value`**（`dfine_decoder.py:423` 的 `value_op` 会对 75,600×256 memory 再做一次投影 ≈5 G MAC）→ 把 `value_op` 提到 `TransformerDecoder.forward` 外层，两条 route 共用（见 R3）。
- **显存增量 `[推断]`**：300q×6 层激活 + 310² 注意力 ≈ **+0.2–0.5 GiB（B=2）**；900q 约 +0.5–1.0 GiB。
- **接线（criterion，唯一有坑的地方）**：在 `deim_criterion.py:794`（enc_aux 之后）、`:799`（DN）之前插入
  ```python
  if 'aux_o2m_outputs' in outputs:                      # 键名必须含 'aux'，见 R4
      for i, o in enumerate(outputs['aux_o2m_outputs']):
          idx = self.o2m_assigner(o, targets)           # greedy，无需 Hungarian
          for loss in ('tadl', 'boxes'):                # 绝不要用 'local'，见 R5
              ld = self.get_loss(loss, o, targets, idx, num_boxes * 3)   # 论文归一化
              losses.update({k + f'_o2m_{i}': v * 0.5 for k, v in ld.items()})
  ```
  `o2m_assigner` = 论文 `M=0.3·sigmoid+0.7·IoU` → 过滤 `IoU<0.4` → 每 GT top-6（`α=0.3, τ=0.4, K=6`）。**TADL 与指派无关**（`deim_criterion.py:245-346` 只用 `(fg,gt)` 对算 IoU/NWD 权重），可直接吃多对一 indices；`weight_dict` 只需加 `loss_tadl_o2m_*` / `loss_bbox_o2m_*`（`yml:274-284`），或沿用 `get_loss` 返回的原名并让 `:716` 的 `weight_dict` 过滤生效（需在 `yml` 里登记名字，否则**被静默丢弃**）。

### 3.2 (b) instruction token / instructive self-attention —— 推荐的唯一必做件

**落点**：`dfine_decoder.py:165-244`（`TransformerDecoderLayer`）+ `:522`（`DFINETransformer.__init__`）。

```python
# __init__（DFINETransformer，:695 之前）
self.instruction_o2m     = nn.Embedding(10, hidden_dim)   # 2,560
self.instruction_o2m_pos = nn.Embedding(10, hidden_dim)   # 2,560
self.norm_o2m            = nn.LayerNorm(hidden_dim)       # +512
self.use_o2m_route       = bool(use_o2m_route)            # 默认 False

# TransformerDecoderLayer.forward（:215-244），仅 instr 非空时走
if instr is None:                                     # o2o 路径逐字不变
    ... 原 223-228 行 ...
else:
    m   = instr.shape[1]                                                 # 10
    ins = self._instr_sum                                        # (B,10,256) = w + pos
    q   = torch.cat([ins, self.with_pos_embed(target, query_pos_embed)], 1)  # (B,310,256)
    v   = torch.cat([instr_w, target], 1)                                 # (B,310,256)
    a,_ = self.self_attn(q, q, value=v)                                   # 无 mask（DN 关闭）
    target = target + self.dropout1(a[:, m:])                             # 丢掉前 10 个输出
    target = self.norm_o2m(target)          # ← 必须独立 LN；改 norm1 会让 o2o 路不再逐位一致
```

- **shape**：`(2,310,256)` → 丢弃前 10 → `(2,300,256)`；`norm_o2m` 作用于 `(2,300,256)`。
- **参数增量 `[推断]`**：2,560 + 2,560 + 512 = **5,632**（= 论文 m=10 × d=256 × 2 + LN）。
- **显存/耗时**：自注意力从 300² 变 310²，可忽略（<0.05 GiB）。
- **与现有损失/匹配器接线**：同 3.1；指令 token 本身**不需要**新损失，也不参与匹配。
- **关键约束**：`norm_o2m` 必须是**新模块**、`instruction_*` 必须**新初始化**，这样 `use_o2m_route=false` 与 `true` 在 `eval()` 下前向**逐位相同**（新增参数在推理路径上完全不被读取）→ 可做"零步等价"验收（§8）。

### 3.3 (c) 路由模块共享 / 独立策略

| 形态 | 独立件 | 我们参数增量 `[推断]` | 训练增量 `[论文]` | 12ep AP `[论文]` | 建议 |
|---|---|---|---|---|---|
| 只 o2o（现状） | — | 0 | 0 | 47.6 | — |
| **全共享 + o2m 损失** | 无 | 0 | 小 | **41.6 (−6.0)** | **禁止**（见 §6.3） |
| Route-3 | instr token + LN | **+5,632** | ~+9%（推算） | **50.4 (+2.8)** | **首选** |
| Route-1 | o2m FFN `linear1'/linear2'` | +525,568×6 = **+3.15 M** | ~+10%（推算） | 49.6 (+2.0) | 条件性（Route-3 阳性后再叠） |
| Route-1+3 | 两者 | +3.16 M | +17 min (+20%) | 50.7 (+3.1) | 二期 |

**要点**：论文 `(2) Share All` 是唯一负收益形态，而它恰好是"不动结构、只加损失"的写法。仓库里 `matcher.py:141-161` + `dfine_decoder.py:902-904` 这两块死代码指向的就是这条错路——**迁移前必须先在文档里钉死"至少一个独立件"这条约束**。共享策略本身不引入新损失，与匹配器无接线改动。

### 3.4 (d) Mr.DETR++ MoE —— 映射清楚，但不建议做

**落点**（若做）：`dfine_decoder.py:212-213`（`forward_ffn`）+ `:240-242`（FFN 调用处），移植 `moe.py:133`（`SparseMoeBlock`）、`:31`（`MoEGate`）、`:121`（`MoeMLP`）。

```python
# TransformerDecoderLayer.__init__：self.moe_ffn = SparseMoeBlock(256, mlp_ratio=4,
#     num_experts=4, num_experts_per_tok=2, n_shared_experts=2, isdecoder=True)
# forward：tgt2 = self.moe_ffn(tgt, SEP=(route_name != 'o2o'))   # gate / gate_sep (:156-163)
# 辅助负载均衡损失经 AddAuxiliaryLoss(:101-118) 注入，需在 criterion 增加 loss_moe_aux 权重
```

- **参数增量 `[推断]`**：4 experts × `MoeMLP(256,1024)` = 3.15 M + 2 shared (interm 512) 0.26 M + 2 gates ≈ **3.41 M/层 → 6 层 +15.7 M**；若用 `moe.py:133` 的默认 `num_experts=16` → **+75 M**。
- **耗时 `[论文]`**：+26 min/epoch（+31%，127 vs 101），换 **+0.3 AP**；MoE **留在推理图内**（Route-2 = `E∘G∘CA∘SA`，论文 Eq.7-8），推理侧每层 FFN 由 1 个变 top-2 专家。
- **encoder scale-aware MoE**：论文只对低分辨率 token 加 MoE（`paper_text.txt:596-610`）。我们的 encoder **只有 1 层且只在 P5**（`yml:215-217`、`hybrid_encoder.py:756-768`），机制位错位；P2/P3 上加 MoE 是 230K/57.6K token，训练成本不可接受。
- **结论**：`+0.3 AP / +31% 训练 / 15.7 M 参数 / 唯一的推理期架构改动` —— 性价比全场最差，**不做**（§6.1）。

---

## 4. 合规判定：训练多路、推理单路是否违反"禁止集成"

**条款原文**（`docs/01_competition/【竞赛细则】面向城市场景的视觉多模态目标检测.pdf`，第 6 页，**十一、赛题约束条件 →（一）算法约束**第 3 条）：
> 「禁止将多个不同结构或训练阶段的模型进行简单集成，直接采用投票法、平均法等方式决策。」

项目内的解释与执行口径（三处一致）：`AGENTS.md:83`（红线 8）、`docs/02_collaboration_and_specs/PROJECT_SPEC.md:214`（§7.2 合规红线）、`README.md:74`；历史 WBF 计划已标注"从未执行、不得用于提交"（`docs/03_architecture_designs/SOVEREIGN_V23_DETAILED_TECHNICAL_SPEC.md:297`）。

| 对象 | 判定 | 依据 |
|---|---|---|
| **辅助 o2m 路（Route-1/Route-3）** | ✅ **合规** | 条款禁的是"多个…**模型**"被**集成决策**。辅助 route 与主路共享同一组参数、同一优化器、同一 checkpoint，是**一个模型的一张训练图**；推理时只前向 Route-2，**没有任何跨模型/跨路输出被合并**。项目内已有同构先例：Co-ATSS 训练期挂载、推理切除（`deim.py:559`；`MODEL_RECORDS.md:1056-1069` §十五 明写"推理阶段零开销原生切除辅助头"），并已进入纯净基准（`yml:184`）。 |
| **MoE（route-aware / scale-aware）** | ✅ **合规** | 专家在**一次**训练里联合优化，产出**一个**权重文件；推理是单次前向内的条件计算（top-2 专家按 gating 加权求和），不是"不同结构模型"的投票/平均。与 `AGENTS.md:83` 允许的"单模型内生化"同类。 |
| **推理期把 Route-1/2/3 输出做平均/NMS/WBF** | ❌ **不合规** | 落在"平均法"字面，且属于跨路决策融合；细则未区分"同一 checkpoint 的不同分支"，风险高。**不做**。 |
| **把 V23 Stage-A（72ep 补标预训练）与 Stage-B（21ep 旧标注精调）两套权重融合** | ❌ **不合规** | 同时踩"不同结构/不同训练阶段"+"集成"。V23 交付用的是**Stage-B 单个权重**（`MODEL_RECORDS.md:874`、AGENTS §1）。 |
| 多阶段**顺序训练**（预训练→精调，只有一个最终权重） | ✅ 合规 | 这是常规训练流程；V23/V28-A 均如此且被接受。 |

**其它合规边界（本方案不触碰）**：逐图独立推理、无跨图状态（`CLEAN_BASELINE.md:11`）；仅用官方数据 + 允许的 COCO/ImageNet/Objects365 预训练权重（细则十一（二））；`dataset/test` 不读标注、不用于训练（`AGENTS.md:81` 红线 6）。

---

## 5. 价值排序与候选实验（可直接执行）

以下三项机制均为 `[待验证]`（本仓库无训练证据）；实验设计与判据按 `MODEL_RECORDS.md` §三十七 的噪声标定设定。

**预算基准 `[验证]`**：v1 在 RTX PRO 6000 Blackwell 96GB（BF16 autocast + TF32，`num_workers:8`）**0.76–0.82 s/step**、900 step/epoch；含每轮 200 图验证约 **14 min/epoch**（10.7 h / 46 ep，`MODEL_RECORDS.md:2165-2204` §四十二–四十三）。**20 ep ≈ 4.7 h/臂，50 ep ≈ 11 h/臂**。V100 纯 FP32 为 56 h/50ep（`MODEL_RECORDS.md:2028` §三十五），且 ailab40 两张卡 ECC 关闭、GPU1 静默错误未修（§四十），**不作为首选**。本地 RTX 5060 Ti 15.46 GiB **不能训练**（v1 训练峰值 21.06 GiB、B=2；§四十一），只能做 CPU 契约测试与单图 FP32 前向（§四十七 已在本机跑过 14 条件剖面）。

### 实验 1（推荐·性价比第一）：Route-3 最小迁移 = 指令自注意力 + o2m 解码路线

- **改动点**：`dfine_decoder.py` 三处（`__init__` 加 5,632 参数/开关；`TransformerDecoderLayer.forward` 加 instr 分支 + `norm_o2m`；`TransformerDecoder.forward` 透传 `o2m/instr` 并复用 `value_op`）+ `DFINETransformer.forward` 加训练分支 + `deim_criterion.py` 加 `o2m_assigner` 与 `aux_o2m_outputs` 损失块 + `yml` 加开关/权重。**推理路径 0 改动**。
- **对照臂**：**A0 = 现 v1 配置原样**（`use_o2m_route=false`），同 seed 42、同 20 ep、同 COCO `tuning`（`yml:359-361`）、同 1800/200 划分；两臂仅差一个开关。
- **预算**：2 臂 × 20 ep ≈ **9.4 h**（Blackwell 96GB）；若只有 V100 纯 FP32 ≈ 44 h（不建议）。
- **成功判据**：val 200 张上 **ΔwAP(A2−A0) ≥ +0.4 pp 且配对按图 bootstrap 95% CI 不含 0**（噪声下限见 `MODEL_RECORDS.md:2076-2087` §三十七）；同时官方口径 mAP 不低于 A0−0.5 pp、AP75 不低于 A0−2.5 pp；step 中位耗时增幅 ≤25%。
- **证伪判据**：ΔwAP ≤ 0 或 CI 跨 0；或 mAP 掉 >1 pp；或 `loss_tadl_o2m` 占总 loss >30%（o2m 反客为主）。
- **回滚**：`use_o2m_route: false`（默认 false）即回滚，**新参数不进推理、不需改 checkpoint**；配置层一行。
- **注意**：20 ep 结论只对"同预算 20 ep 点"有效，**不能外推到 50 ep**；若阳性，再跑 50 ep 全量臂（+11 h）才能进提交候选。

### 实验 2（闸门·性价比第二，与实验 1 共享 A0）：Co-ATSS 归因

- **改动点**：**纯配置**——`use_co_aux: false`（`yml:184`），并把 `loss_co_cls/loss_co_reg` 从 `weight_dict` 注释掉（`yml:281-282`）。criterion 用 `if 'loss_co_cls' in outputs` 容错（`deim_criterion.py:727-733`），**无需改代码**。
- **对照臂**：直接复用实验 1 的 **A0**（增量成本 = 1 臂）。
- **预算**：+1 臂 × 20 ep ≈ **4.7 h**（总计 A0+A1+A2 三臂 ≈ 14 h）。
- **成功判据（判为"可下线"）**：ΔwAP(A1−A0) ≥ −0.4 pp 且 CI 含 0，同时 mAP 不降 >1 pp → Co-ATSS 可下线，**回收约一半训练前向乘加**，并让实验 1 的"第二个 o2m"变成"唯一 o2m"。
- **证伪判据**：ΔwAP ≤ −0.4 pp 且 CI 不含 0 → Co-ATSS 承重，不可下线；此时实验 1 的收益必须解释为"叠加"而非"替代"。
- **回滚**：配置即回滚。
- **为何排第二**：它花同样的钱，但可能**一次性减掉本项目最贵的单项训练开销**；且它决定实验 1 的结论该怎么读。代价是它本身不产生精度收益。

### 实验 3（廉价 pilot·性价比第三）：localization-aware score calibration

- **改动点**：decoder 最后一层加 class-aware IoU 头 `nn.Linear(256, 12)`（**+3,084 参数**），目标分数 `iou^0.75`（论文 `paper_text.txt:615-640`），损失复用现有 `loss_labels_vfl`（`deim_criterion.py:163`）；推理 `s_calib = s_cls^φ · s_iou^(1−φ)`（论文 Eq.13）。**无新前向、无新匹配**。
- **对照臂**：v1 `best.pth`（epoch 37，`MODEL_RECORDS.md:2275-2281` §四十五）同权重、同 200 张 val 的配对比较（注意：该权重是在同一 val 上选出来的，属**探索性**，见 §四十四）。
- **预算**：1 臂 × 12 ep 热启动（v1 best 用 `-t` 载入，`irbind` 同款 `tuning_key_remap` 机制，`MODEL_RECORDS.md:2316` §四十七）≈ **2.8 h** + val 网格搜 φ（11 点）+ 配对评测 ≈ **0.5 h**，合计 **≈3.3 h**。
- **成功判据**：φ 由 val 选定后 ΔwAP ≥ +0.4 pp 且 CI 不含 0，或 ΔAP75 ≥ +2.5 pp。
- **证伪判据**：φ∈[0,1] 全网格下 wAP/mAP 变化均在噪声内 → 收益已被现有 LQE（`dfine_decoder.py:314/482`）与 TADL（`deim_criterion.py:245`）吃掉，结案。
- **回滚**：φ=1 使 `s_calib = s_cls` **精确退化**为现状；新头不写入主配置即完全无影响。

---

## 6. 不建议做清单（附理由）

| # | 不做 | 理由 |
|---|---|---|
| 6.1 | **Mr.DETR++ route-aware MoE（decoder）** | +15.7 M 参数、+31% 训练时间（论文 +26/84 min），相对 Mr.DETR 只 **+0.3 AP**（`paper_text.txt:854-869`）；且是**唯一改推理图**的项（checkpoint/EMA/BF16 数值面最大）。1800 图无法支撑"专家分工"假设。 |
| 6.2 | **encoder scale-aware MoE** | 论文 +0.4 AP；我们的 encoder 仅 **1 层、仅 P5**（`yml:215-217`），机制位错位；放到 P2/P3（230K/57.6K token）训练成本不可接受。 |
| 6.3 | **零新参数接法**（把 o2m 损失/`matcher.py:141-161` 直接接到现有 o2o 输出） | 就是论文 TABLE I 的 `(2) Share All = 41.6 (−6.0)`，**已知负收益**，且会抹掉我们能归因的前提。 |
| 6.4 | **900 query 的 o2m 路** | 论文 300q/900q 两档收益同阶（+2.5 vs +3.1 / +4.0 vs +4.2）；我们 300q 已够（`yml:178,243`）。900q 让每层 Hungarian/指派与注意力矩阵 ×3，而我们**没有**匹配开销的余量（`matcher.py:132-136` 在 CPU 上跑 `linear_sum_assignment`，×6 层 ×(o2o+o2m)）。 |
| 6.5 | **GT 复制式 dense O2O / copy-paste** | 与 Co-ATSS 的"每 GT 每层 3×3 候选 + ATSS 阈值"（`co_aux_head.py:233-285`）功能重叠；每图仅 ~7 GT（1523 GT/200 图，`MODEL_RECORDS.md:2061`），正样本密度不是瓶颈 `[推断]`；且会与 §四十七 的模态 pilot 争同一实验预算。 |
| 6.6 | **搬 Mr.DETR 的 focal/VFL + DAC `rank_weight`（`deformable_criterion.py:199-213`）** | 与 TADL 的正样本 target=1.0 + `q^β` 质量加权（`deim_criterion.py:245-346`）语义冲突；等于同时改损失 + 改指派，不可归因。 |
| 6.7 | **instruction token 用于 o2o 路** | 作者自己已注释掉（`transformer.py:170-177`），论文无验证。 |
| 6.8 | **推理期多 route 平均/NMS 融合** | 触碰细则"平均法"字面，合规风险高（§4）；与"单模型内生"红线冲突。 |
| 6.9 | **用 V23 式两阶段（两套权重）模拟多路** | 直接落在"不同训练阶段的模型集成"禁令内（§4）。 |
| 6.10 | **临时打开 DN（`yml:244` 0→100）与 o2m route 同时上** | 两个训练期辅助机制同时引入，attribution 归零；DN 是独立议题，应单独对照。 |

---

## 7. 风险清单

| # | 风险 | 触发点 / 后果 | 缓解 |
|---|---|---|---|
| **R1** | **跨架构精度不可比** | V100 必须纯 FP32+禁 cuDNN，Blackwell 走 BF16 autocast+TF32（`CLAUDE.md` §3、`CLEAN_BASELINE.md:21`）；对照臂跨架构 = 无效对比 | 同组对照**必须同卡同精度**；本地只做 CPU 契约 + 单图 FP32 前向（本地 15.46 GiB 无法训练，v1 峰值 21.06 GiB，§四十一） |
| **R2** | 显存 | o2m 路 300q ≈ +0.2–0.5 GiB；900q +0.5–1.0 GiB；Co-ATSS 的 P3 特征 180×320×384×4B ≈ **88 MB/张**已是既存大项 | 开跑前用 `BENCH=1` 的 `preflight_cloud.sh` 实数；`accum_steps` 只拆 micro，不改有效 batch（`CLAUDE.md` §3） |
| **R3** | **二次 value 投影** | 直接调两次 `TransformerDecoder.forward` 会在 `:423` 对 75,600×256 memory 重做投影（≈5 G MAC/图），白付 2–5% step | 把 `value_op` 上提到两条 route 之外，`value` 传入 |
| **R4** | **输出键名会撞 criterion 的过滤** | `deim_criterion.py:672` 用 `if 'aux' not in k` 构造 `outputs_without_aux`；键名若不含 `aux`，list 会被送进 `matcher` → **KeyError 崩**（阳性：不会静默错） | 键名固定 `aux_o2m_outputs`；CPU 契约测试里断言 `outputs_without_aux` 不含它 |
| **R5** | **`loss_local` 隐式缓存（最危险）** | `deim_criterion.py:490-497` 首次调用即缓存 `fgl_targets`、`:523-525` 缓存 `num_pos/num_neg`，仅用 `'is_dn' in outputs` 区分两套；o2m 输出若跑 `local` 损失会**复用 o2o 的 FGL 目标** → 目标错、loss 有限、`stop_on_nonfinite` 不报 = **静默错误** | o2m 路只用 `('tadl','boxes')`；或给缓存加 route key 并在契约测试中覆盖 |
| **R6** | eval 分支行为 | `TransformerDecoder.forward` 在 `not training` 时于 `eval_idx` `break` 且不产 aux（`:494-510`）；o2m 路必须 `if self.training` 才走，否则推理变慢/变值 | 在合成 config 上做 eval 前向断言：`use_o2m_route=true` 的推理输出与 `false` **逐位相同** |
| **R7** | **checkpoint / EMA 兼容** | 新参数使 v1 checkpoint 无法 `-r`（`strict_resume: true`，`yml:353` 缺键即失败）；EMA `decay 0.9999/warmup 1000`（`yml:131-136`）会从初值开始平均新参数 | 从 COCO `tuning` 全新跑（实验 1 设计即如此）；若要热启动 v1，用 `-t` + `tuning_key_remap`/`init_*_after_tuning` 机制（`MODEL_RECORDS.md:2316` §四十七 先例） |
| **R8** | 数值契约 | `stop_on_nonfinite: true`（`yml:352`）+ `require_finite` → 新增 BCE/`exp` 在 BF16 autocast 边界溢出会直接中断 | criterion 整体关闭 autocast（既有行为），新损失在 FP32 下算；`co_aux_head.py:192` 的 `clamp(-8,5)` 是同类先例 |
| **R9** | **strict_inputs 数据契约** | 本方案**不改数据**；若为给 o2m 更多正样本而改标注/增强，会破 6 通道、raw16、`strict_inputs`（`yml:50-58`；`coco_dataset_mm.py:84-94`，`:175-176` 明确禁止 `ABLATE` 覆盖） | 数据块一行不动；改动只在模型/损失 |
| **R10** | 选点与噪声 | 同一 200 张 val 既选点又评估（§四十四/§四十五）；mAP ±1 pp、AP75 ±2.5 pp、wAP ±0.4 pp 内不可分辨（§三十七） | 判据一律用**配对按图 bootstrap**，主判据用 wAP；同时报逐 GT 配对计数 |
| **R11** | 小数据过拟合 / 长跑退化 | V28-B 72 轮全解冻单调退化 0.888→0.778、`best_stg2` 落在最差轮（`MODEL_RECORDS.md:1005-1018` §十三.3） | 新机制臂**固定短预算**（20 ep），不拉长轮数；同时记 train/val gap；不从 best 回滚（clean lifecycle） |
| **R12** | 结论外推与提交口径 | 20 ep 结论不能外推 50 ep；Phase 3 ≠ Phase 2（`MODEL_RECORDS.md:2283` §四十六） | 阳性后必须 50 ep 复跑；提交前跑 `scripts/eval/check_submission.py` 与工作台导入（`AGENTS.md:83` 红线 11） |

---

## 8. 验收与回放（本方案要求的门禁顺序）

```bash
PY=/home/constant/miniconda3/envs/deimv2/bin/python
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:${PYTHONPATH:-}"

# ① CPU 契约（不训练、不更新参数）：先证 key 过滤/缓存/开关 fail-closed 都没被破坏
CUDA_VISIBLE_DEVICES='' $PY scripts/diagnostics/test_clean_baseline_contract.py

# ② 零步等价（本地 RTX 5060 Ti 单图 FP32，1440x2560）：use_o2m_route true/false
#    同一 checkpoint、eval 模式、3 张 val 图，pred_logits/pred_boxes 最大绝对差必须为 0
#    先例：§四十七 irbind pilot "3 张验证图上 pred_logits/pred_boxes 最大差均为 0"

# ③ 换卡自检 + 30 步测时（必须真实报 R2 的显存/step 数）
ROOT=$PWD GPUS="0" REQUIRE_ECC=1 BENCH=1 bash scripts/train/preflight_cloud.sh

# ④ 训练（cd vendor/DEIMv2；全新 output_dir；V100 追加 -u use_amp=False）
cd vendor/DEIMv2 && $PY train.py -c configs/deimv2/<new_arm>.yml

# ⑤ 评测/配对（官方 NMS0.60、每图 ≤100；主判据 wAP + 配对 bootstrap）
$PY scripts/analysis/compare_detectors_paired.py --annotations work_dir/deimv2_dataset/valid/annotations/instances_val.json \
    --a A0=<a0_preds.json> --b A2=<a2_preds.json> --out work_dir/paired_o2m_e1
```

新增的 2 个 CPU 契约测试（本任务不写，留给实现阶段）：`test_o2m_route_contract.py`（①键过滤 ②eval 逐位等价 ③缓存不复用）——放 `scripts/diagnostics/`，遵循 `CLAUDE.md` §2 的 unittest 惯例。

---

## 附录：引用速查

- **我们的代码**：`vendor/DEIMv2/engine/deim/deim.py:478-488,559-561`（Co-ATSS 接线）· `co_aux_head.py:67,163,233-337` · `deim_criterion.py:41,163,245,478,490-497,523-525,594,615,672,675,679-691,710-755,757-794,799` · `matcher.py:59,132-161` · `dfine_decoder.py:215-244,314,387,406-518,522-631,858-922,924-992` · `denoising.py:12,99` · `hybrid_encoder.py:727-808` · `data/dataset/coco_dataset_mm.py:39-94,120,175-176` · `configs/deimv2/aic_clean_baseline_v1.yml:8,55-58,127-138,178,184-186,215-218,226-228,243-255,261,274-302,316,350-361`
- **Mr.DETR 源码**：`archive/scratch/mrdetr_src/projects/mr_detr_align/modeling/{transformer.py:74-75,78,140-193,340-343,371-382;deformable_transformer.py:475-537,553-609;one2manyMatcherDAC.py:223-302;deformable_criterion.py:53-81,115-128,199-213;dn_criterion.py:28-57}`、`projects/mr_detr_pp_align/modeling/moe.py:31,101-118,121,133,156-233`
- **论文**：`outputs/reports/mrdetr_2026-11-01/paper_text.txt:209-232`（TABLE I）· `:420-450`（三条 route / 推理丢弃）· `:596-640`（scale-MoE、calibration Eq.12-13）· `:630-690`（TABLE II、超参 m=10/K=6/α=0.3/τ=0.4、4 experts top-2）· `:854-869`（TABLE VIII）· `:870-880`（TABLE IX）· `:945-957`（TABLE XI 训练时间 / TABLE XII probing）
- **项目历史**：`docs/MODEL_RECORDS.md:874`（§十一.3 V23 两段式）· `:1005-1018`（§十三.3 V28-B 退化红线）· `:1056-1069`（§十五 V31 Co-ATSS）· `:2028`（§三十五 V100 56h）· `:2076-2087`（§三十七 噪声下限）· `:2143-2204`（§四十一天 Blackwell 迁移/§四十二 Turbo 0.76s step/§四十三 10.7h 46ep）· `:2229-2244`（§四十四 选点偏差）· `:2275-2281`（§四十五 终止状态）· `:2283-2290`（§四十六 Phase 3 = 51.1）· `:2292-2318`（§四十七 模态剖面 + irbind pilot 的零差分验收）· `docs/02_collaboration_and_specs/CLEAN_BASELINE.md:11,21,79-93`（逐图独立、精度、未确定收益项）· `docs/01_competition/【竞赛细则】…pdf` 第 6 页 十一（一）· `AGENTS.md:83` · `docs/02_collaboration_and_specs/PROJECT_SPEC.md:214` · `README.md:74`
