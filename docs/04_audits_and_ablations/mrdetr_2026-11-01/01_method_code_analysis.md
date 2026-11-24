# Mr. DETR / Mr. DETR++ 代码级拆解（CVPR'25 + v4 MoE）

> 拆解对象：`archive/scratch/mrdetr_src/`（`Visual-AI/Mr.DETR`，main@`287c6935`，tarball 无 `.git`）。
> 标注约定：**【确认】**＝代码直读 / 本地仿真验证；**【推断】**＝代码结构可推断但未运行训练验证；**【未验证】**＝本机无法验证（论文数字、外部仓库实现、硬件实测）。
> 证据格式 `path:line`，Mr.DETR 侧路径相对 `archive/scratch/mrdetr_src/`，DEIMv2 侧路径相对 AIC 仓库根。
> 本报告不做论文综述；只回答"代码到底怎么写的、迁移要改哪几行"。

---

## 0. 一句话机制 + 给迁移的 5 条判断

**【确认】** 训练期同一套 backbone/encoder/decoder 权重被**串行跑 3 遍 decoder**：`o2o`（匈牙利一对一）、`o2m`（额外一套 top-k-per-GT 一对一多指派 + self-attn 里注入 10 个 learnable instruction token）、`sep`（用 o2o 的 query 再跑一遍、只把 decoder FFN 换成第二套权重）。推理时**只取 o2o 路的输出**，但当前代码**不裁剪** o2m/sep 两路，推理仍旧跑 3 遍 decoder（`projects/mr_detr_align/modeling/deformable_detr.py:396-418`、`projects/mr_detr_align/modeling/deformable_transformer.py:553-595`）。Mr. DETR++ 在此基础上把 encoder FFN 与 decoder 最后一层 FFN 换成 MoE（`projects/mr_detr_pp_align/modeling/dino_transformer.py:76-79,205-208`），并把 DINO 的 7 层 encoder 中间输出按层路由给 decoder。

给迁移的 5 条：

1. **最值得迁移的是"第二遍 decoder 的 o2m 监督"，不是 instruction token**：它与我们已有的 Co-ATSS 外部 head（`vendor/DEIMv2/engine/deim/co_aux_head.py:67`）正交——Co-ATSS 在 encoder 特征上挂卷积头，Mr.DETR 在 decoder 里用**同一套 head 权重**再跑一遍、用 top-k 指派拿密集梯度。两者可叠加，但共享 head 这一条决定了它不能直接复用 Co-ATSS 的 loss 结构。
2. **instruction token 极便宜**：每 decoder 层 +2×10×C 参数（R50 版共 30,720），只改 self-attn 的 K/V，不改 head、不改 cross-attn；训练期生成"跨 query 通信 token"。是否涨点无法从代码判断【未验证】，但实现成本极低（见 §3）。
3. **推理成本可回归基线**：o2m/sep 路只贡献训练梯度；把 `self.decoder(...)` 两次调用用 `if self.training` 包起来即可让推理回到 baseline（§5.2），但**没有现成代码做这件事**，出厂模型推理是 3 遍 decoder。
4. **最大的移植坑是魔法常数 `10` / `-900`**（`projects/mr_detr_align/modeling/transformer.py:152-155`）：它同时绑定"instruction token 数=10""num_queries=900""DN token 位于 query 块前部"三个假设。我们的 DEIMv2 是 `num_queries=300`、`num_denoising=0`（`vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml:178,243-244`），照抄不会报错但 attention 语义会静默走样（本地仿真已复现，§3.4）。
5. **MoE 那套不建议一起迁**：代码里 encoder MoE 的"只对第 4 个 level 生效"分支是**死代码**（§6.4），实际是对全部 22k encoder token 做 2-of-4 专家 + shared 专家，MoE 的 FFN MACs 约为 baseline FFN 的 4.5×（估算，§6.5），且 `use_checkpoint` 全部失效（§7-T4）。收益/成本比在单卡竞赛场景不划算【推断】。

---

## 1. 仓库地图与四个 project 的真实差异

### 1.1 目录与技术栈

- `projects/mr_detr_align/modeling/`、`projects/mr_detr_dino/modeling/`、`projects/mr_detr_deformable/modeling/`：**Deformable-DETR 系**（detrex 的 project 拷贝，`deformable_transformer.py` 里同时含 encoder/decoder/transformer 类）。
- `projects/mr_detr_pp_align/modeling/`：**DINO 系重写**（`dino.py` + `dino_transformer.py` + `moe.py` + `two_stage_criterion.py`），不是前者的增量修改。
- **detrex 基线实现不在本 tarball 内**：`detrex/modeling/` 只有 `backbone/criterion/losses/matcher/neck`，没有 `deformable_detr.py`/`dino.py`（`detrex/` 是 submodule，tarball 只带了被主仓直接提交修改过的文件；`detrex/layers/denoising.py` 甚至在 269 行被截断）。因此**无法做真正的"逐行 diff vs detrex baseline"**；§1.3 用"可验证的 detrex 侧文件 diff + 新增代码块定位"替代。

### 1.2 四个 project 的差异矩阵（来自 `diff -u`，【确认】）

| project | 基础配方 | DN | 分类 loss | o2m/sep loss 归一化 | 其它 |
|---|---|---|---|---|---|
| `mr_detr_deformable` | Deformable-DETR | **无 DN** | focal（detrex 原生 `loss_labels`） | `num_boxes` | enc loss 用 MS-DETR 的 anchor/pred 交换 trick |
| `mr_detr_dino` | DINO 配方（DN + focal） | **有**（3 路都注 DN） | focal | `num_boxes*3` | 与 align 的 `deformable_detr.py`/`deformable_transformer.py`/`transformer.py` **字节相同** |
| `mr_detr_align` | Align-DETR 配方 | **有**（3 路都注 DN） | IoU-aware BCE + rank 重加权 | `num_boxes*3` | 24ep 配置额外 `pos_embed.temperature=20, offset=0` |
| `mr_detr_pp_align` | DINO 架构 + MoE | **仅在 o2o 路** | IoU-aware BCE（rank 被注释）+ 独立 VFL IoU 头 | `num_boxes*3` | 7 个 encoder 记忆、`class_embed_iou`、MoE |

关键 diff 证据：
- `mr_detr_dino` 与 `mr_detr_align` 的 `modeling/deformable_transformer.py` / `deformable_detr.py` / `transformer.py` 三文件 diff 为空；唯一差异在 `deformable_criterion.py`（align 增加 `loss_labels/loss_boxes` override 与 `userank=True`）与 configs。
- `mr_detr_deformable` → `mr_detr_align` 的 DN 增量：`deformable_transformer.py` 在两阶段分支里给两条路都拼 DN reference/query（align `deformable_transformer.py:493-498,524-526,547-549`），`deformable_detr.py` 追加 160 行 `prepare_for_cdn`（align `:484-627`）与 `dn_post_process`（`:629-641`）。
- 结论：**"align" 修的是 loss，"dino"/"deformable" 修的是 baseline 配方（DN/pos-embed）；多路结构与 instruction 在三个 deformable 系 project 里是同一份代码。**

### 1.3 与 detrex 基线的 diff（结构级，可验证部分）

detrex 侧完整可读的只有 `detrex/layers/transformer.py`（原始 `BaseTransformerLayer`）与 `detrex/layers/attention.py`（原始 `MultiheadAttention`）。项目侧把它们**复制并改写**成 `projects/*/modeling/transformer.py`：

| 基线（detrex） | 项目侧（Mr.DETR） | 差异性质 |
|---|---|---|
| `detrex/layers/attention.py:28` `MultiheadAttention.forward`，输出 `identity + proj_drop(out)`（`:138`） | `projects/mr_detr_align/modeling/transformer.py:31` `CustomMultiheadAttention` | **新增** instruction embedding（`:72-75`）、`norm_o2o/norm_o2m`（`:77-78`）、o2m 分支（`:140-168`）；输出变成 `norm(identity+drop(out))`——**比基线多了一层 attention 内 LayerNorm**（连 o2o 路也受影响） |
| `detrex/layers/transformer.py:188-190` `ffn: self.ffns[..](query, identity)` | `projects/mr_detr_align/modeling/transformer.py:339-344` | **新增** `sep` 时向 FFN 传 `**kwargs`，用于选第二套 FFN 权重 |
| 无 | `transformer.py:371-478` `CustomLinear/CustomBlock/DynamicFFN` | **新增**：每个 decoder FFN 变成 `linear`+`linear_sep` 双份；`k_groups=16` 是死参数（未使用） |
| `detrex/layers/transformer.py:195` `TransformerLayerSequence` 均匀复制 | `transformer.py:482-520` `CustomTransformerLayerSequence` | **新增** `custom_layers_index` + `transformer_layers_custom`（pp 用它只把第 6 层换成 MoE） |
| `detrex/layers/denoising.py:244` `GenerateCDNQueries`（文件被截断） | `deformable_detr.py:484-627` 内联 `prepare_for_cdn` | **未使用 detrex 的 DN 模块**，抄的是 DINO 官方实现（含 `.cuda()` 硬编码） |

Deformable-DETR 主体（`DeformableDetrTransformer`/`DeformableDetrTransformerDecoder`）在 detrex 侧不可读，无法给行号级 diff；但从结构上，Mr.DETR 只在 `DeformableDetrTransformer.forward` 的两阶段分支后追加了 o2m proposal 与第 2/3 次 decoder 调用（§2.3），其余部分与 detrex Deformable-DETR 两阶段实现同构（`look_forward_twice`、`mixed_selection`、`with_box_refine` 开关都在）。

---

## 2. 多路结构：3 条路、query 数、共享/独立参数、分叉点

### 2.1 三条路的定义（`mr_detr_align`，【确认】）

| 路 | 代码开关 | 初始 query 来源 | query 数 | 匹配器 | 用途 |
|---|---|---|---|---|---|
| **o2o** | 无额外 kwarg（`deformable_transformer.py:553-564`） | 两阶段 top-k proposals（o2o 专用头 `class_embed[6]`/`bbox_embed[6]`，`:480-507`） | 900 + DN(~200) | `HungarianMatcher`（cost 2/5/2） | 推理唯一输出 |
| **o2m** | `o2m=True`（`:566-578`） | 两阶段 top-k proposals（**独立头** `decoder.class_embed_o2m_encoder` / `decoder.bbox_embed_enc_o2m`，`:511-537`） | 900 + DN(~200) | `Stage2Assigner(max_k=6)` 每 GT 取 top-6 | 训练监督 |
| **sep** | `sep=True`（`:580-595`） | 直接复用 o2o 的 `query/query_pos/reference_points`（`:580-584`，`sep_query = query` 等） | 900 + DN(~200) | 同 `Stage2Assigner` | 训练监督（FFN 参数独立） |


> **代码命名 ≠ 论文 Route 编号（别被 `o2m` 这个词骗）**：论文（v4 正文 `paper_text.txt:420-520`）里 Route-2 = primary o2o、Route-3 = 带 instructive self-attention 的辅助路、Route-1 = 独立 FFN 的辅助路。代码里 `o2m=True` 的路 = 论文 **Route-3**；`sep=True` 的路 = 论文 **Route-1**；`o2o` = Route-2。pp 的 `output["o2m"]` 也对应 Route-3（不是常说的 one-to-many group）。

DN 总量公式（`deformable_detr.py:510-517`）：`dn_number=100 →*2 → //(max_gt*2)`，`pad_size = max_gt*2*dn_number`，结果 ≈ 2×`dn_number`（=200，floor 抖动 ±几个 token），与 batch 内 max_gt 基本无关。因此**每路 self-attn 长度 ≈ 1100**（不是 900）。

### 2.2 前向伪代码（align，含 DN）

```python
# projects/mr_detr_align/modeling/deformable_transformer.py:418-624
feats, masks, pos = flatten(multi_level_feats...)          # :428-455
memory = self.encoder(...)                                 # :461-472  只跑一次
if as_two_stage:
    ref_o2o, qpos_o2o, q_o2o = two_stage_propose(memory, heads.class_embed[6], heads.bbox_embed[6])   # :480-507
    ref_o2m, qpos_o2m, q_o2m = two_stage_propose(memory, decoder.class_embed_o2m_encoder,
                                                         decoder.bbox_embed_enc_o2m)                  # :511-537
    ref_*  = cat([dn_bbox, ref_*], 1)   # :493-498, 524-526  DN reference 前置
if dn_label is not None:
    q_o2o = cat([dn_label, q_o2o], 1);  q_o2m = cat([dn_label, q_o2m], 1)   # :547-549
st_o2o,  ref_o2o  = self.decoder(query=q_o2o,  reference_points=ref_o2o,  dn_attn_mask=M, **kw)   # :553
st_o2m,  ref_o2m  = self.decoder(query=q_o2m,  reference_points=ref_o2m,  o2m=True, **kw)        # :566 o2m 路自-attn 走 instruction 分支
st_sep,  ref_sep  = self.decoder(query=q_o2o,  reference_points=ref_o2o,  sep=True, **kw)         # :583 sep 路只换 FFN 权重
```

### 2.3 分叉点

**在 decoder 的第 0 层之前分叉，不在层内分叉。** decoder 层本身是同一个 `nn.ModuleList`（三条路共享全部层参数），区别全部由"输入 query/reference + kwargs 里的 `o2m`/`sep` 标志"决定；层内唯一按 flag 变化的只有 self-attn（instruction 分支，`transformer.py:140`）与 FFN（`linear_sep`，`transformer.py:371-383`）。三次调用顺序执行 → **训练 3 份 decoder 激活保活**，显存≈3×decoder 部分。

### 2.4 参数共享矩阵（【确认】）

| 组件 | o2o | o2m | sep | 代码 |
|---|---|---|---|---|
| backbone / encoder / level_embeds | 共享 | 共享 | 共享 | `deformable_transformer.py:461-472` |
| decoder 全部 self-attn / cross-attn 权重 | 共享 | 共享 | 共享 | 同一 `self.layers` |
| decoder FFN | `linear` | `linear` | **`linear_sep`** | `transformer.py:371-383` |
| decoder bbox refine head | 共享 | 共享 | 共享 | `deformable_transformer.py:233-243` |
| 分类/回归输出头 `class_embed[lvl]/bbox_embed[lvl]` | 共享 | 共享 | 共享 | `deformable_detr.py:299-300,320-321` |
| 两阶段 proposal 头 | 专用 | **专用一套**（`class_embed_o2m_encoder` + `bbox_embed_enc_o2m`） | 无（复用 o2o） | `deformable_detr.py:148-156` |
| instruction embedding / `norm_o2m` | 无 | 有（每层各一份） | 无 | `transformer.py:72-78` |
| `norm_o2o` | 有 | 无（走 `norm_o2m`） | 有 | `transformer.py:167-168,177` |

> **注**：即便完全不走 o2m/sep，o2o 路相对 detrex baseline 也已多了一层 self-attn 内 LayerNorm（`norm_o2o`）。所以"删掉两路就退回 baseline"**不成立**，只是计算量退回。

---

## 3. instructive self-attention：shape、拼接、mask

### 3.1 参数 shape / 数量

```python
# projects/mr_detr_align/modeling/transformer.py:72-78
# self.instruction_o2o = nn.Embedding(10, embed_dim)      # ← 被注释掉（o2o 路没有 instruction）
# self.instruction_o2o_pos = nn.Embedding(10, embed_dim)
self.instruction_o2m     = nn.Embedding(10, embed_dim)    # (10, 256)
self.instruction_o2m_pos = nn.Embedding(10, embed_dim)    # (10, 256)
self.norm_o2o = nn.LayerNorm(embed_dim)
self.norm_o2m = nn.LayerNorm(embed_dim)
```

- 每个 decoder 层的 **self-attn** 各有一份（层定义 `deformable_transformer.py:151-178` + `CustomTransformerLayerSequence` 的 `deepcopy`），共 6 层 × 2 个 embedding × (10×256) = **30,720 参数**；`10` 与「全部 6 层都加」与论文附录 B 一致（`paper_text.txt:1253-1254`：apply instruction tokens across all six decoder layers and set their number to 10）。
- **cross-attn 没有 instruction**：decoder `operation_order=("self_attn","cross_attn","norm","ffn","norm")`（`deformable_transformer.py:147`），cross-attn 是 `MultiScaleDeformableAttention`，token 维度上无法 concat。

### 3.2 如何 concat 进 q/k/v

```python
# transformer.py:140-168（o2m 分支）
if "o2m" in list(kwargs.keys()) and kwargs["o2m"]:
    instruction = self.instruction_o2m.weight + self.instruction_o2m_pos.weight   # (10, C)
    instruction = instruction[None].repeat(query.shape[0], 1, 1)                  # (B, 10, C)  ← query.shape[0] 是 batch，依赖 batch_first=True
    query = torch.cat([instruction, query], dim=1)                                # (B, 10+N, C)
    key   = torch.cat([instruction, key],   dim=1)                                # K = [instr(+pos); q+pos]
    value = torch.cat([self.instruction_o2m.weight[None].repeat(B,1,1), value], 1) # V 只用 weight（不加 pos！）
    ...
    out = self.attn(query=query, key=key, value=value, attn_mask=new_attn_mask,
                    key_padding_mask=key_padding_mask)[0]
    out = out[:, 10:]                       # 丢弃 instruction 位置的输出
    out = self.proj_drop(out) + identity    # residual 用加 pos 之前的 query
    out = self.norm_o2m(out)
```

- **K 用 `weight+pos`、V 只用 `weight`**：不对称，且 instruction 的输出被丢弃（`out[:, 10:]`）→ instruction 只作为"被真实 query 读取的 K/V 记忆槽"起作用；`instruction_o2m_pos` 实际只影响 K。论文 Eq.(6) 与 Fig.5(c) 对应同一实现：选 concat 方案（addition 方案要求 token 数=query 数），输出丢弃（`paper_text.txt:486-520`）。
- `key_padding_mask` 没有为多出的 10 个 key 扩展——因为本项目里 decoder self-attn 的 `query_key_padding_mask` 恒为 `None`（`deformable_transformer.py:553-595` 都没传），一旦引入 padding 会 shape 崩。

### 3.3 attn_mask 如何从 `dn_attn_mask` 构造（含 10 / 900 偏移）

```python
# transformer.py:150-157
if kwargs['dn_attn_mask'] is not None:
    dn_attn_mask = kwargs['dn_attn_mask']                     # (pad+900, pad+900) bool，True=屏蔽
    new_attn_mask = torch.ones(query.shape[1], query.shape[1],
                               device=query.device, dtype=dn_attn_mask.dtype)   # (10+pad+900)² 全 True(=全屏蔽)
    new_attn_mask[10:, 10:] = dn_attn_mask                    # 真实 query 块继承 DN 规则
    new_attn_mask[:10, -900:] = False                         # instruction 行 → 最后 900 个真实 query 可见
    new_attn_mask[-900:, :10] = False                         # 最后 900 个真实 query 行 → instruction 可见
else:
    new_attn_mask = attn_mask                                 # 推理（无 DN）时 = None
```

本地用 torch 复现该构造（pad=200, num_queries=900, instr=10，`True=屏蔽`），得到的 3×3 块矩阵（行=query 角色，列=key 角色，数值=块内屏蔽比例，**仿真结果**）：

| 行 \ 列 | instr(10) | DN(pad≈200) | real(900) |
|---|---|---|---|
| **instr(10)** | 1.00（不可见自己/彼此） | 1.00（看不到 DN） | **0.00（可见）** |
| **DN(pad)** | 1.00（看不到 instruction） | 依 `dn_attn_mask`（简化仿真未建模组间屏蔽） | 0.00（可见） |
| **real(900)** | **0.00（可见）** | 1.00（匹配 query 看不到 DN，来自 `deformable_detr.py:601`） | 0.00（可见） |

**为什么 `new_attn_mask[:10, -900:] = False` 是必须的**：`torch.ones` 初始化把 instruction 行也设成全屏蔽，若不打开某列，该行 softmax 全 `-inf` → NaN。所以这行首先是**数值必需**；instruction→real 方向对最终输出无贡献（instruction 位置输出被丢弃），真正的语义只有 `[-900:, :10] = False`（真实 query 能读取 instruction）。【确认：仿真 + `torch.nn.MultiheadAttention` 语义】

### 3.4 `-900` 的语义与 300q 复现

`-900` 与总长 `10+pad+900` 配合，等价于"真实 query 块 = 最后 900 行/列"。**换成 `num_queries=300` 不会报错**：Python 负索引越界会被 clamp 到 0，于是 `[:10, -900:]` 变成"整行 0 列起全 False"、`[-900:, :10]` 变成"所有行对 instr 可见"——DN 与 instruction 之间的隔离被静默破坏（本地仿真复现【确认】）。`mr_detr_deformable` 版根本没有这段 mask 代码（只在 align 系里新增，见 §1.3 表格），所以无 DN 迁移时天然安全。

### 3.5 推理时该分支如何处理

- `o2m` kwarg 由 `DeformableDetrTransformer.forward` 硬编码传入：`self.decoder(..., o2m=True, **kwargs)`（`deformable_transformer.py:576`）→ `BaseTransformerLayer.forward(**kw)` → self-attn `transformer.py:140`。**没有"eval 时关掉"的开关**。
- 推理时 `dn_query_embeds=(None,None)`、`dn_attn_mask=None`（`deformable_detr.py:235-237`）→ o2m 分支走 `new_attn_mask=None`，**完全不建 mask**。由于 instruction 输出被丢弃、而真实 query 在 train/eval 下可见的 key 集合都是 `{instr, real}`，**两种模式的 o2o/o2m 输出理论等价**（DN 只在训练出现）【推断，静态推演】。
- **eval 时仍走全部 3 条路**，只是结果字典里只取 o2o：`output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}`（`deformable_detr.py:361`、`:405-409`）。
- 去掉 instruction 的后果代码无法回答：只能是"o2m 路的 self-attn 少 10 个 K/V token"，对 AP 的影响【未验证】；ablation 在论文里，本仓库无对应 config。

---

## 4. 三路的 matcher 与 loss

### 4.1 o2o 路：标准 Hungarian

`projects/mr_detr_align/configs/models/deformable_detr_r50.py`：`HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, cost_class_type="focal_loss_cost")`；两阶段 proposal 用同一个 matcher 对 `bin_targets`（labels 全 0，`deformable_criterion.py:134-140`）做匹配。`criterion.enc_matcher` 虽然被传入并保存（`:55`），但 **align/dino 版本从未使用**（`grep enc_matcher` 只在 `mr_detr_deformable` 用）。

### 4.2 o2m / sep 路：`Stage2Assigner`（文件头注明来自 DAC-DETR）

`projects/mr_detr_align/modeling/one2manyMatcherDAC.py`：

```python
class Stage2Assigner(nn.Module):                       # :223
    def __init__(self, num_queries, max_k=6, allow_low_quality_matches=True):   # :224
        self.positive_fraction = 0.25                  # :226
        self.batch_size_per_image = num_queries        # :228  ← 硬编码 900（criterion 里 :54）
        self.proposal_matcher = Matcher(thresholds=[0.4], labels=[0,1],
                                        allow_low_quality_matches=True)          # :229
        self.k = max_k                                 # :230
    def get_cost_matrix(self, pred_logits, pred_boxes, gt_classes, gt_boxes):    # :268
        cost_box   = IoU(pred, gt)                     # :273
        cost_class = sigmoid(cls)[:, gt_classes]       # :274
        return (0.8 * cost_box + 0.2 * cost_class).T   # :275-277  ← IoU 主导的"相似度"
    def forward(...):
        matched_idxs, matched_labels = self.proposal_matcher(cost_matrix)        # :291 每个 query 取 max-cost 的 GT，cost>=0.4 记正
        sampled_idxs, ... = self._sample_proposals(...)                          # :292 随机子采样 pos:0.25 / neg
        pos_pr_inds, pos_gt_inds = self.postprocess_indices(...)                 # :297
        return sample_topk(pr_inds, gt_inds, iou, k=6)                           # :304 → 每个 GT 最多 6 个正样本
```

- **DAC 的含义**：文件头 `# DAC-DETR Copyright (c) 2023 University of Technology Sydney & Baidu Inc & Zhejiang University`（`:1-5`）——`Stage2Assigner` 这套"top-k per GT + 动态采样"指配器直接取自 DAC-DETR 仓库（NeurIPS 2023, *Divide the Attention Layers and Conquer*）；代码里 **DAC 没有展开成算法缩写**，只作出处标记【确认：只确认出处，不确认论文内部对应哪一节】。
- 与 Hungarian 的本质差别：**不是二分匹配**，同一 query 只挂 1 个 GT、每个 GT 的正样本数 = 通过 cost≥0.4 的候选数（再由 `sample_topk` 截到最多 6 个）；`allow_low_quality_matches=True` 保证每个 GT 至少补一个最高 cost 的 query（`Matcher.set_low_quality_matches_`，`:128-151`，此处被改写成 topk 版本）。
- **注意随机性**：`subsample_labels` 用 `randperm` 采正样本（`:195-199`），正样本上限 `0.25*900=225`；COCO 常见 <20 个 GT 时不会触发截断，但**目标密集的 AIC 场景（每图 ~7 个框，最多几十）仍可能触发**；若一个 batch 里多 GT 图很多，正样本会被随机丢弃 → 训练不稳定来源之一【推断】。

### 4.3 loss 项目、权重、归一化（align，【确认】）

`deformable_criterion.py:83-164` + config 的 `weight_dict`：

| 路 | loss key | 权重 | 归一化 | 备注 |
|---|---|---|---|---|
| o2o 主 + aux×5 | `loss_class/loss_bbox/loss_giou` + `_0.._4` | 1 / 5 / 2 | `num_boxes` | 最后一层同时就是主输出 |
| o2m（输出键名 **`group`**） | `loss_*_group_{i}`，i=0..5（**含最后一层**） | 1 / 5 / 2 | **`num_boxes*3`** | `userank=True` |
| sep | `loss_*_sep_{i}`，i=0..5 | 1 / 5 / 2 | **`num_boxes*3`** | `userank=True` |
| DN（三条路各算一份） | `loss_*_dn`、`loss_*_dn_o2m`、`loss_*_dn_sep` + aux `_dn_{k}` | 1 / 5 / 2 | `num_boxes*dn_num` | 见 §4.5 |
| enc（两阶段 proposal） | `loss_*_enc`、`loss_*_enc_o2m` | 1 / 5 / 2 | `num_boxes` | labels 全 0（二分类） |

`num_boxes*3` 的效果：每条 o2m/sep 路的分类/回归 loss 量级 ≈ 主路 1/3，三路合计 ≈ 主路 1 倍。DN 则没有 ×3 → **DN 的监督总强度是 baseline 的 3 倍**（三条路各一份 DN loss，归一化相同）【确认：`dn_criterion.py:50-57`】。

### 4.4 分类 loss 不是 focal：Align-style IoU-aware BCE + rank 重加权

`deformable_criterion.py:166-215`（align 独有；dino/deformable 版是 detrex 原生 focal）：

```python
pred_score = sigmoid(src_logits)
t = pred_score[pos].pow(0.25) * ious.pow(0.75)         # :196  Align-DETR 的 IoU-aware 目标
t = clamp(t, 0.01).detach()
if userank:                                            # :199  仅 o2m/sep 路
    rank = self.get_local_rank_o2m(t, indices)         # 同一 GT 的多个正样本按 t 排序
    rank_weight = exp(-rank / 1.5)                     # :201  τ=1.5，秩 0/1/2… 衰减
    t = t * rank_weight
pos_weights[pos] = t;  neg_weights = pred^2            # :191,205-207
loss_ce = -pos_w*log(p) - neg_w*log(1-p)               # :210   sum()/num_boxes
```

- `t→0.01` 的 floor 使低质量重复正样本仍有梯度；rank 衰减是**抑制同一 GT 的重复正样本互相"抢分类分"**（H-DETR 用 hybrid matching 分数、这里用 loss 权重）【推断】。
- **`loss_boxes` 里 GIoU 也乘同一个 `rank_weight`**（`:239`），而 `rank_weight` 是**实例状态**（`self.rank_weight`，`:213`），依赖 `get_loss` 的调用顺序（`losses=["class","boxes"]`，`:243-249`）→ 顺序一变就串味（§7-T5）。

### 4.5 DN 在各路的处理

- **align**：DN 的 query/reference 同时拼进 3 条路（`deformable_transformer.py:493-498,524-526,547-549`），三条路的输出各自 `dn_post_process` 剥离 DN 前缀（`deformable_detr.py:345-358`，`dn_meta` 用 `deepcopy` 三份），DN loss 三份独立计算（`dn_criterion.py:50-57`），后两份 key 改名为 `*_dn_o2m` / `*_dn_sep`。
- **pp_align**：DN **只拼进 o2o 路**（`dino_transformer.py:507-508,519-520`；o2m/sep 的 DN concat 被注释 `:537-538,550-551`），因此 pp 的 DN loss 只有一份（`dn_criterion.py:50-57` 同名实现）。**这条差异意味着 align 的 `output["group"]` 张量前 pad 个 token 是 DN、pp 不是**，跨 project 复用时不能直接对齐。
- 两阶段 enc 头在 align 里也是两套（o2o/enc_o2m），DN 不涉及 enc。

### 4.6 总 loss 的死键 / 命名错位（陷阱，§7-T10 详述）

`configs/models/deformable_detr_r50.py` 的 `weight_dict` 里：`_o2m_{i}`、`_enc_sep`、`loss_class_dn_group` 等键在 criterion 里**不产生**（criterion 产出的是 `_group_{i}`/`_sep_{i}`/`_dn_o2m`/`_dn_sep`），被 `deformable_detr.py:401-403` 的 `if k in weight_dict` 静默丢弃；反过来 pp 的 `aux_outputs` 含最后一层（`dino.py:342-346`），而 swin config 的 aux 权重只生成到 `_4`（`dino_swin_large_384_4scale_12ep.py:77`）→ 最后一层 aux 被静默丢弃。

---

## 5. 推理路径与真实成本

### 5.1 推理实际执行什么（【确认】）

`deformable_detr.py:405-418`：`box_cls/box_pred` 只取 o2o 最后一层 → `inference()`（top-300，`:430-466`）→ `detector_postprocess`。但在那之前：
- encoder 跑 1 次；
- **decoder 跑 3 次**（`deformable_transformer.py:553-595`，无 `if self.training`）；
- 两套两阶段 proposal 头都跑（o2o/o2m）；
- o2m/sep 的输出头前向（`deformable_detr.py:291-330`）也照算，只是不进结果；
- **论文声称 auxiliary routes 在推理时被丢弃、推理时间与 baseline 一致**（`paper_text.txt:449,1025`），但当前代码只丢弃**输出**、不丢弃**计算**：三条路的 decoder 前向在 eval 下照跑。想要真正零开销必须自己加 `if self.training`（§5.2）。
- **eval-only 的分类混合**：o2o 每层 `p = (p_last^2.001 + p_prev^1.853)/2` 再 `sqrt`（`deformable_detr.py:280-285`），pp 版还多一个 `sqrt(cls*iou)`（`dino.py:266-277,389`，iou 来自独立 `class_embed_iou`）。这是**训练时不存在的推理增强**，复现报数必须保留；代码注释原话 "This can improve AP by about 0.1 ~ 0.2%"（`mr_detr_deformable/modeling/deformable_detr.py:239-245`），自己重写 head 会丢这部分差异【推断】。

### 5.2 多余参数能否丢弃

**代码层面没有 deploy/裁剪逻辑**（`grep convert_to_deploy|del self|prune` 在 `projects/` 无命中）。但如果自己改推理，可安全跳过：o2m/sep 的 decoder 调用、`class_embed_o2m_encoder`/`bbox_embed_enc_o2m`、`instruction_o2m*`、`norm_o2m`、`linear_sep`。o2o 路不引用它们。最小改法：

```python
# 训练保留三路，推理只跑 o2o（需同时删掉 output["group"]/["sep"] 的 head 循环）
st, ref = self.decoder(query=q, ...)                       # 原 :553-564
if self.training:
    st_o2m, ref_o2m = self.decoder(..., o2m=True, **kw)    # 原 :566-578
    st_sep, ref_sep = self.decoder(..., sep=True, **kw)    # 原 :583-595
```

注意 `norm_o2o` 不能删（o2o 路在用），`DynamicFFN` 只能把 `linear_sep` 删掉、`linear` 必须留。

### 5.3 参数量增量（按 config 手算，【确认：按模块形状计算】）

**align R50 版**（`ff=2048, C=256, 6 层, 80 类, 7 个 head`）：

| 新增组件 | 参数量 |
|---|---|
| decoder FFN 第二套 `linear_sep`（6 层 × 1,050,880） | **6,305,280** |
| `class_embed_o2m_encoder` + `bbox_embed_enc_o2m` | 153,172 |
| `instruction_o2m(+pos)` 6 层 × 2 × 10×256 | 30,720 |
| `norm_o2o/norm_o2m` 6 层 × 2 × 512 | 6,144 |
| DN `label_enc`（80×256） | 20,480 |
| **合计** | **≈ 6.52 M** |

baseline（R50 两阶段 4scale）≈ 43.1 M（backbone 23.5 + neck 1.44 + transformer 17.1 + heads 1.07…）→ **+15%**。注意 box-refine 的 7 套 head 是 baseline 就有的，不算增量。

**pp_align Swin-L 版**：

| 新增组件 | 参数量 |
|---|---|
| encoder MoE × 6 层（4 专家 + shared + gate，替换原 FFN，净增 6,814,464/层） | **40,886,784** |
| decoder 5 层 `linear_sep` | 5,254,400 |
| decoder 第 6 层 MoE（替换 FFN，另加 `gate_sep`） | +5,242,624 |
| 7 个 encoder 记忆映射 `SiLUFFN`（1792→256） | 3,216,640 |
| `class_embed_iou`（7×分类头） | 143,920 |
| o2m proposal 头（class+bbox） | 153,172 |
| **合计** | **≈ 54.9 M** |

Swin-L 全模型 ≈ 222 M → **+25%**。参数集中在 encoder MoE（41 M）。

### 5.4 FLOPs / 显存（估算，公式化；未上机 profile）

输入 800×1333、4 个 level（stride 8/16/32/64）、encoder token ≈ 22.2k、decoder 900 query（训练 +~200 DN）：

| 组件 | baseline MACs | Mr.DETR align | pp_align |
|---|---|---|---|
| encoder FFN（6 层） | ≈140 GMAC | 140（不变） | **≈629 GMAC**（2 专家+shared×4.5 倍，见 §6） |
| decoder（每路 6 层） | ≈10.6 GMAC | ×3 = **31.9**（+21.3） | ×3 = 31.9（且第 6 层是 MoE，再略高） |
| 增量相对"backbone+全部" | — | 约 **+8~12%** 训练 FLOPs；推理若裁掉两路 = 0 | encoder 增量约 **+480 GMAC**，即使裁掉两路仍在 |

- **训练**：3 路 decoder 串行 → decoder 激活 ≈3 份；DN 让每路 token 从 900→~1100（self-attn 矩阵 ×1.49）。align 的主要成本是**激活显存**（≈+2 份 decoder 层激活），不是 FLOPs。
- **推理**：align 若不改代码，decoder 计算 3×（对小 backbone 更痛）；改代码后 = baseline。**pp 的 MoE 在推理端固定生效**（encoder 全 token 2-of-4 专家 + shared；decoder 第 6 层 2 专家），属于"结构性成本"，与 multi-route 无关。
- 以上为**按 config 手算的 MACs 估算**，无硬件实测；我们 DEIMv2 是 ViT-L 级 backbone + 300 query，decoder 增量会明显小于上表。

---

## 6. Mr. DETR++ 的 MoE

### 6.1 放在哪

- **encoder：每一层的 FFN 全是 MoE**（替换 `FFN`，`dino_transformer.py:76-79`；`BaseTransformerLayer` 的 `ffn` 槽位，`operation_order=("self_attn","norm","ffn","norm")`:81）。
- **decoder：只有最后一层（index 5）的 FFN 是 MoE**（`custom_layers_index=[5]`，`dino_transformer.py:188`；custom 层 `ffn=SparseMoeBlock(...)`，`:205-208`），其余 5 层是 `DynamicFFN`（有 `linear/linear_sep`）。
- **self-attn / cross-attn 完全没有 MoE**：`moe.py` 没有 attention 相关代码；`SparseMoeBlock` 只出现在 ffn 槽位。

### 6.2 专家 / topk / router / shared（`moe.py`）

```python
# moe.py:180-192
SparseMoeBlock(embed_dim=256, mlp_ratio=8, num_experts=4, num_experts_per_tok=2,
               pretraining_tp=1, n_shared_experts=8, isdecoder=False)
# moe.py:121-155  MoeMLP = SwiGLU 风格：gate/up/down 三个 Linear(bias=False)，act=SiLU
#   → 每个专家 = 3×256×2048 = 1,572,864 参数
# moe.py:32-95   MoEGate：softmax 打分（:58-59），top-k 不归一化（:64-69），kaiming 初始化（:48-50）
# moe.py:72-87   aux loss：Pi=mean(scores), fi=count*E → (Pi*fi).sum()*alpha, alpha=0.01（默认参数 :33）
# moe.py:100-117 AddAuxiliaryLoss：forward 原样返回 x，backward 给 loss 注入梯度 1（标量不返回、不进 loss_dict）
```

- 专家数 4、top-2、encoder 另有 **1 个 shared expert**（`n_shared_experts=8` 表示 shared 的中层宽度 = `8*256 = 2048`，不是 8 个专家）；decoder 层 `n_shared_experts=None` → 输出是 `y + identity`（残差，`:224-227`）。
- **load-balance aux loss 走的是"隐式加法"**：`y = AddAuxiliaryLoss.apply(y, aux_loss)`（`moe.py:213`）→ 总 loss 里看不见这个标量，但梯度进了 total（`alpha=0.01` 缩放）；日志里没有它 → 调试时容易误以为没有 aux loss【确认】。
- **训练/推理都用 top-2**：训练按 expert 分桶（`moe.py:206-213`），推理 `moe_infer` 用 argsort + `scatter_reduce_(reduce='sum')`（`moe.py:231-250`）。

### 6.3 与 multi-route 的耦合

- **`gate_sep`**：只有 decoder 层（`isdecoder=True` 时构造，`moe.py:191-192`），sep 路 `SparseMoeBlock.forward` 里检测 `SEP` kwarg 走第二套 router（`:199-200`）；专家与 shared 专家**共享**，只有 router 权重独立。o2m 路（`GROUP=True`）用普通 router。与论文 Route-aware MoE 定义一致：三条路共享专家集，Route-2/Route-3 共享 gating `G`、Route-1 用独立 gating `G'`（`paper_text.txt:484-500`）。
- **7 个 encoder 记忆**：encoder 每层输出拼成 `(B, 22k, 7*256)`，再经 7 个 `SiLUFFN`（实为 `nn.Linear`，SiLU 被注释，`dino_transformer.py:34-45`）投回 256 维 → `memory_list[0..6]`；decoder 第 i 层用 `key[memory_list[i]]`（`:263`），`memory_list[6]` 只用于两阶段 proposal（`:480`）。这是"层级特征路由"，与 multi-route 正交但叠在同一次前向里。

### 6.4 与 MoE 相关的两个硬事实（重要）

1. **`ENCODER` / `level_start_index[3]` 分支是死代码**：MoE 的"只对最高 level 生效"逻辑（`moe.py:196-197,217-222`）依赖 `kwargs["ENCODER"]`，但 `BaseTransformerLayer` 调 FFN 时不透传 kwargs（`projects/mr_detr_pp_align/modeling/transformer.py:333`，只有 `SEP` 分支透传 `:331`）→ **encoder 的每一层 MoE 都对全部 token 生效**，走 `y = moe(x) + shared(x)`（`moe.py:224-225`）。发布权重正是按这个行为训的；若"修好"该分支，行为与开源权重不一致。成本上：按原意只对 1/4 的 token 用专家，实际对 22k token 全部用 2 个专家（§6.5）。
2. **`use_checkpoint` 全部无效**：`for layer in self.layers: layer = checkpoint_wrapper(layer)`（`dino_transformer.py:94-96,104-106,222-225`；align 版 `deformable_transformer.py:76-78`）只是把局部变量重绑定，没有写回 `self.layers[i]` → **梯度检查点从未生效**，README 里 "save gpu memory" 的说法不成立。pp 的 encoder 又必须保留 7 份层输出（拼接用），激活显存很大（22k×7×256×dtype）——这也是为什么 pp 在 swin 配置里开了 `use_checkpoint=True` 却没用（`dino_swin_large_384_4scale_12ep.py:92`）。

### 6.5 MoE 的 FLOPs 代价（估算）

encoder 每 token：2 专家 × 1.57M + shared 1.57M ≈ **4.72 M MACs**，baseline FFN 1.05 M → **×4.5**；6 层 ≈ 629 GMAC vs 140 GMAC。若死代码分支生效（只最后一个 level 用专家），≈213 GMAC（×1.5）【推断：按代码结构算，未实测】。

### 6.6 论文 v4 正文 vs 开源代码：3 处实质不一致（对复现影响大）

| # | 论文怎么写 | 代码实际做了什么 | 证据 |
|---|---|---|---|
| D1 | encoder 的 scale-aware MoE 只作用于 low-scale（x64）tokens，其余 scale 只过 shared FFN；Table X：x64-only = 51.8 AP / 486.8 GFLOPs，全 scale = 51.5 AP / 832.5 GFLOPs | 该分支**永远不触发**（`ENCODER` kwargs 未透传给 FFN），MoE 对**全部 4 个 level** 生效 → 等价全 scale 行；发布权重实际是"全 scale"版本 | 论文 `paper_text.txt:577-611,900-935`；代码 `pp/moe.py:196-197` + `pp/transformer.py:333` |
| D2 | Route-aware MoE 放在 decoder **最后 3 层**（layer 3/4/5，Fig.9 也是 3/4/5） | 只有**最后一层**（index 5）是 MoE，其余 5 层是 `DynamicFFN`（两套 FFN） | 论文 `paper_text.txt:933-934`；代码 `pp/dino_transformer.py:188,205-208` |
| D3 | Invariance calibration 的 VFL target 抬升为 `iou^0.75` | 代码里是 `target_score ** 0.9` | 论文 `paper_text.txt:581`；代码 `pp/two_stage_criterion.py:55` |

> 结论：**复现报数必须用"代码行为"而不是论文描述**；我们要借鉴时，D1 说明 "scale-aware" 在开源版没落地（推理代价比论文大），D2 说明 "route-aware MoE 只加 1 层" 未经论文验证，不能当作论文结论【推断】。

---

## 7. 实现陷阱清单（按迁移风险排序）

| # | 陷阱 | 证据 | 影响 / 对策 |
|---|---|---|---|
| T1 | **魔法常数 10 / -900** 绑定 instruction 数、`num_queries=900`、DN 前置布局 | `align/transformer.py:152-155` | 换 300q 不报错但 attention 语义静默错（§3.4）；必须改成 `n_instr` + `num_queries` 变量 |
| T2 | **强依赖两阶段 + box refine**：o2m proposal 变量只在 `as_two_stage` 分支定义，非两阶段会在 `if dn_query_embeds[0] is not None: o2m_query=...` 处 UnboundLocalError；`look_forward_twice=True` 且 `bbox_embed=None` 时 `new_reference_points` 未定义 | `align/deformable_transformer.py:475-549,613-624`；`align/deformable_detr.py:128-145` | 我们的 DEIMv2 是两阶段式 query 选择（D-FINE），结构不同，需自己接 o2m proposal 头或复用 topk |
| T3 | **`Stage2Assigner(num_queries=900, max_k=6)` 硬编码**：900 只用于正样本上限 225，300q 时须改 | `align/deformable_criterion.py:54`；`one2manyMatcherDAC.py:224-230` | 迁移必须参数化；`max_k=6` 与 `positive_fraction=0.25` 需重调 |
| T4 | **梯度检查点全部无效果**（rebind 而非赋值） | `align/deformable_transformer.py:76-78`；`pp/dino_transformer.py:94-96,104-106,222-225` | README 的省显存承诺是假的；pp 训练显存高 |
| T5 | **`self.rank_weight` 实例状态泄漏 + loss 顺序依赖** | `align/deformable_criterion.py:199-206,213,239` | 改 `losses` 顺序或单独求 box loss 会用上一条路的 rank 权重；多 route 串行时也会串 |
| T6 | **`.cuda()` 硬编码**（DN 生成、rank 计算、`_sample_proposals`） | `align/deformable_detr.py:511,551,580`；`one2manyMatcherDAC.py:195-196`；`deformable_criterion.py:268` | CPU 测试 / 单卡以外环境直接崩；我们做 CPU 契约测试会踩 |
| T7 | **`dn_number=100` 写死在 `DeformableDETR.__init__`**，`dn_label_noise_ratio/dn_box_noise_scale` 是构造参数 | `align/deformable_detr.py:170,179-180` | 想改 DN 组数只能改 config 属性（LazyCall 后赋值）或改源码 |
| T8 | **eval-only 分类混合**（align 的 2.001/1.853；pp 的 `sqrt(cls*iou)`） | `align/deformable_detr.py:280-285`；`pp/dino.py:266-277,389` | 训练/推理不一致；复现 AP 必须保留，迁移到我们模型时要么带上、要么重做基线对比 |
| T9 | **pp aux 层数不一致**：`aux_outputs` 含最后一层（6 个），swin config 只给 `_0.._4` 权重 | `pp/dino.py:342-346`；`configs/dino-swin/...12ep.py:77` | 最后一层"重复监督"被静默丢弃；r50 config 用 `range(num_layers)` 则最后一层被算两次 |
| T10 | **loss 键死键/命名错位**：criterion 产 `_group_`/`_sep_`，config 里写 `_o2m_`；`_enc_sep`、`loss_class_dn_group` 无产出 | `align/deformable_criterion.py:115-128`；`align/configs/models/deformable_detr_r50.py`（`weight_dict` 尾部） | 改权重时一定对着 criterion 实际 `k+="_xxx"` 命名 |
| T11 | **MoE ENCODER 分支死代码**（§6.4） | `pp/moe.py:196-197`；`pp/transformer.py:333` | 想"修好"会与开源权重行为不一致；迁移时先决定要哪种 |
| T12 | **pp 的 `class_embed_iou` 只在 eval 用于 proposal 排序**（train 用 `class_embed`） | `pp/dino_transformer.py:487-492` | train/eval proposal 选择不一致（有意为之，RT-DETR IA-QS 思路）；迁移后若不复刻，AP 会掉 |
| T13 | **modelmean 权重平均**：手写脚本按固定 iter 文件 0.8/0.15/0.05 或 0.8/0.1/0.05/0.05 平均，非 EMA、路径写死 | `projects/modelmean_12ep.py`、`modelmean_24ep.py` | 复现报数必须用它；文件名绑定 12ep/24ep 的 iter 数 |
| T14 | **两阶段 proposal 头初始化不对称**：o2m enc bbox 头在 zero-init 之前 deepcopy，保留 `bias[2:]=-2`，o2o 被置 0 | `align/deformable_detr.py:139,150-153` | 复现细节；重写建议统一 |
| T15 | **命名混淆**：`output["group"]` 实际是 o2m 路（不是 Group DETR） | `align/deformable_detr.py:366-375` | 读代码/改权重时极易误解；pp 里改叫 `output["o2m"]` |
| T16 | `SiLUFFN` 实为 `nn.Linear`（激活被注释） | `pp/dino_transformer.py:34-45` | 名字误导，实际是 7×256→256 线性投影 |
| T17 | `dn_query_embeds` 从 kwargs 取，缺 key 直接 KeyError；o2o 分支 `kwargs['dn_attn_mask']` 同理 | `align/deformable_transformer.py:494,547`；`align/transformer.py:150,178` | 迁移时若只拷贝部分代码会 runtime 崩 |
| T18 | **`DynamicFFN` 使 decoder FFN 参数翻倍**，只为了 sep 路；不用 sep 时应退回 `FFN` | `align/transformer.py:403-478`；`deformable_transformer.py:167-174` | 增量参数的大头（6.31 M / 6.52 M ≈ 97%）来自它 |

---

## 8. 与 Group DETR / H-DETR / Co-DETR / RT-DETR 的 one-many 实现差别

**本仓库能直接确认的只有 Mr.DETR 侧**；对其它方法只能按公开实现对比，标【未验证】。

| 方法 | 实现方式 | 与 Mr.DETR 的差别（代码级） |
|---|---|---|
| **Group DETR** | 单次 decoder 前向内把 K 组 query 拼在序列里，用 block-diagonal mask 让组内互看；每组的匈牙利 loss 乘 1/K | Mr.DETR 是**串行 K 次独立 decoder pass + 共享权重**，不是拼接、不用 group mask；`output["group"]` 是命名误导（T15）。block mask 会拉长 sequence，Mr.DETR 不会，但 3 次 pass 的串行延迟更高【未验证：Group DETR 侧】 |
| **H-DETR / hybrid matching** | 在同一前向内加一组 one2many query，**把 o2o 与 o2m 的匹配索引合并后一起监督**；推理用 hybrid score 抑制重复框 | Mr.DETR 的 o2m 是**独立第二遍**；而且**没有 merge**：`indices_merge()` 写了但从未被调用（`align/deformable_criterion.py:59-81`；`mr_detr_deformable/deformable_criterion.py:114-117` 里调用被注释）。重复框靠 loss 里的 rank 衰减（§4.4）而不是 hybrid score |
| **Co-DETR** | 外挂多个 one2many assigner 的辅助头（ATSS/FCOS/Faster R-CNN），每个头独立参数、独立 loss，推理丢弃 | Mr.DETR 的 o2m **共享**主检测头与 decoder 参数（只多 o2m proposal 头），supervision 在 decoder 最后一层而非外部 head。我们 DEIMv2 已经有 Co-ATSS 版本（`vendor/DEIMv2/engine/deim/co_aux_head.py:67`），二者可叠加：Co-ATSS 补 encoder 特征梯度，Mr.DETR 补 decoder query 梯度 |
| **RT-DETR / D-FINE 的 one2many** | RT-DETR 本身没有 one2many 路；相关的是 IoU-aware query selection（IA-QS）与 VFL 监督 | pp_align **借用了 RT-DETR 的 IA-QS 思路**：eval 时 proposal 排序用 `sqrt(cls*iou)`，并把 IoU 头用 VFL 单独监督（`pp/dino_transformer.py:487-492`；`pp/two_stage_criterion.py:30-64`）。这是唯一"与 RT-DETR 同源"的部分，与 o2m 无关 |

> 上表中 Group DETR / H-DETR / Co-DETR / RT-DETR 的具体实现细节来自公开实现，本机无网络与源码可核对【未验证】；Mr.DETR 侧全部行号可查。

**DEIMv2 侧现有可对接的钩子（给实现者）**：
- `vendor/DEIMv2/engine/deim/matcher.py:141-144` 已有 `return_topk → {"indices_o2m": ...}`，`get_top_k_matches`（`:146-163`）是"迭代 k 次匈牙利 + 已匹配位置置 1e6"的 Group-DETR 式 k 重匹配，**但目前无任何调用者**（`grep indices_o2m` 只有 matcher 内部）→ 可以直接接线成 o2m 指派，不需要搬 `Stage2Assigner`。
- `vendor/DEIMv2/engine/deim/deim_criterion.py:665` 的 `forward` 已有 main/aux/enc/dn 四套 loss 分发，o2m loss 挂在 `aux_outputs` 之后即可。
- 我们的 DN 是关的（`aic_clean_baseline_v1.yml:244 num_denoising: 0`），因此可直接照 `mr_detr_deformable`（无 DN 版）而不是 align 版迁移，**省掉所有 `-900`/dn mask 的坑**。

---

## 9. 迁到 DEIMv2 的代码级落点（最小改动）

现状（`vendor/DEIMv2/engine/deim/dfine_decoder.py`）：decoder 层 self-attn 是 `nn.MultiheadAttention`，`q=k=with_pos_embed(target, query_pos_embed)`、`value=target`、`attn_mask` 由 DN 生成（`:218-226`）；decoder 循环在 `:446-468`；`convert_to_deploy`（`:400-404`）已有"推理裁层"的先例。

| 改动 | 落点 | 说明 |
|---|---|---|
| A. o2m 第二遍 decoder | 在 `DFINETransformer.forward` 的 decoder 调用（`dfine_decoder.py:946-960`）之后再调一次 `self.decoder(...)`，inputs 换成第二套 proposal（复用 `_get_decoder_input`，把 `class_embed` 换成新 o2m 头） | Mr.DETR 的 o2m 路最小必需件：多一套两阶段 proposal 头 + 第二次 decoder 前向 |
| B. o2m 指派 | `matcher.py:141-144` 接线 `return_topk=6`；loss 侧 `deim_criterion.py` 加 `loss_*_o2m` | 可先用 DEIMv2 现成的 k 重匹配，成功率最高的最小改法 |
| C. instruction token | `dfine_decoder.py:214-230`（`DFINEDecoderLayer.forward` 的 self-attn 段） | 需要把 `q=k` 拼 10 个 token、`value` 用 instruction V、出 R 后丢掉前 10 个；mask 用 `(n_instr + L)` 重新拼（不要照抄 -900） |
| D. sep 路 / 推理裁剪 | sep 路只在我们想验证"参数独立的第二条 o2m 监督"时才要（decoder FFN 参数翻倍）；推理裁剪可仿 `dfine_decoder.py:400-404 convert_to_deploy` | 建议顺序 A+B → C → D，并给推理加 `if self.training` 只跑 o2o |

**风险提示**：DEIMv2 的两阶段 query 选择（`_get_decoder_input` + `query_select_method`，`dfine_decoder.py:860-925`）与 loss（`tadl/boxes/local`，`aic_clean_baseline_v1.yml:274-288`）都与 Mr.DETR 的 class/bbox/giou 不对齐——**A+B 的 loss 必须按我们现有 loss 重配**。这是本报告认为"迁移可行但必须重写 loss 接线"的核心原因【推断】。

---

## 10. 未验证 / 存疑清单

1. **论文侧数字**（58.7/51.4/50.9 等）来自 README Model Zoo（`README.md:34-53`）；论文 v4 正文有队友抽取的 `paper_text.txt`（同目录）可交叉核对术语/结构，但**未核对 PDF 排版与表号**，表格数字未逐一复核。
2. `detrex/` submodule 不完整：无法给出"相对 detrex baseline 的逐行 diff"；§1.3 是结构级 + detrex 侧可读文件（`detrex/layers/transformer.py`、`detrex/layers/attention.py`）的真实 diff。
3. 三个 deformable project 的 `focal` vs `Align loss` 命名是我从 `diff` 推出的（README 没解释 "Align" 指 Align-DETR 的 loss）——【推断】。
4. §5.4 / §6.5 的 MACs 为按 config 与模块形状手算（未跑 `thop/calflops`，未上机 profile）；显存只给结构推断，无实测。
5. `Stage2Assigner` 与 DAC-DETR 论文内部哪个模块对应、rank 衰减 `exp(-rank/1.5)` 的出处：代码无注释，仅文件头出处可确认（论文正文只在 related work 引用 DAC-DETR [13]，`paper_text.txt:189,350,641`；具体模块对应未验证）。
6. pp_align 的 `n_shared_experts=8` 我按 `intermediate_size = embed_dim*8` 解释（`moe.py:187-189`），即"1 个 shared 专家、宽度 8×"，不是 8 个专家——从代码可确认宽度，但设计意图【推断】。
7. 我们 AIC 数据上 o2m 路是否有效：本报告只做代码拆解，未做任何实验。
