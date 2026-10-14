# 权重溯源报告：为什么 vitl800_clean 能在 6 epoch 到 0.513

> **文档性质**: 模型热启动权重继承机制深度解剖与参数溯源报告  
> **归档位置**: `docs/04_audits_and_ablations/weight_provenance_report.md`  
> **生成时间**: 2026-09-25 · 脚本：`scripts/analysis/provenance_init.py`、`provenance_confirm.py`

---

## 一、结论先行

**`vitl800_clean` 不是"一次性训练出来"的。它热启动的权重里有 55.15M（20.5%）是另一个模型
（`midboost_l4u3`）在 AIC 数据集上训练出来的成品，逐字节相同。**

继承来的这 20.5%，恰好是检测器的**全部核心部件**：三条流的特征提取网络、decoder、融合投影。
ViT-L 分支提供的只是一部分 RGB 特征。所以 epoch 6 的 0.513 里，绝大部分能力不是这次训练出来的。

---

## 二、`vitl16_rdttrack_init.pth` 的真实构成

由 `vendor/DEIMv2/prepare_dinov3_for_rdttrack.py` 生成。该脚本有两个可选参数
`--inherit-aux`、`--inherit-encoder`，一旦给出，就把旧模型的对应模块整个拷进新权重。

| 模块 | 参数量 | 占比 | 来源 |
|---|---:|---:|---|
| `backbone.rgb_dinov3` | 303.23M | 79.5% | 官方 DINOv3 ViT-L 预训练（唯一"干净"的部分） |
| `backbone.ir_dinov3` | 28.61M | 7.5% | **midboost_l4u3 训练好的** |
| `backbone.depth_dinov3` | 28.51M | 7.5% | **midboost_l4u3 训练好的** |
| `encoder` | 11.23M | 2.9% | **midboost_l4u3 训练好的** |
| `decoder` | 9.80M | 2.6% | **midboost_l4u3 训练好的** |

溯源验证（逐 key 做 shape + tensor 指纹比对，`encoder.` / `decoder.` /
`backbone.ir_dinov3.` / `backbone.depth_dinov3.` 共 1159 个 key）：

```
outputs/.../midboost_l4u3/best_stg1.pth          1159/1159 相同   完全继承
outputs/.../midboost_combo/best_stg1.pth           154/1159 相同   部分（巧合/未变前的余留）
ckpts/deimv2_x_for_rdttrack_deepir.pth              53/1159 相同   部分
```

---

## 三、整条训练链：项目里没有任何一次"一次性训练"

每次训练都用 `-t`（tuning/热启动）继承上一个模型的成果：

| 实验 | 起点 `-t` | 继承范围 | 结果 |
|---|---|---|---|
| `..._x_deepir` | （更早的链） | — | 上游 |
| **midboost_combo** | `..._x_deepir/best_stg2.pth` | 全模型 | 0.5222 @49ep |
| **midboost_l4u3** | combo / x_deepir 系 | 全模型 | 0.5372 @38ep |
| **vitl800_clean** | `vitl16_rdttrack_init.pth`（← l4u3 全继承 + 官方 ViT-L） | 混合 | **ep6 0.513** |

`logs/midboost_combo_train.log` 铁证：

```
'tuning': 'outputs/..._x_deepir/best_stg2.pth'
Tuning checkpoint from outputs/..._x_deepir/best_stg2.pth
Load model.state_dict, {'missed': [], 'unmatched': []}
```

`missed` 与 `unmatched` 均为空 ⇒ **整个模型（含 encoder/decoder）被完整继承**。
这也解释了为什么 combo 的 epoch 0 就已经是 0.502：它一开始就不是从零在学。

---

## 四、重新解释 "6 epoch 就 0.513"

前一轮把原因归结为"2.9M 随机融合模块的修复期"——那是真的，但**只是一半**，且是较小的那一半。
完整解释是：

1. `encoder` / `decoder` / `ir_dinov3` / `depth_dinov3`（55.15M）**本来就是 l4u3 在 AIC 上训了几十
   epoch 的成品**，出框能力是现成的。
2. 只有 `convs` / `aux_proj` / `sta.conv2-4` / `prompt` / `dt_ort`（2.9M）是随机的，随机 prompt
   往冻结主干里灌噪声 ⇒ epoch 0 塌到 0.085。
3. 这 2.9M 又短又浅，几千步对齐后噪声消失，借用来的检测能力恢复 ⇒ 快速回到 0.50+。

**所以 0.513 是"借用能力的恢复"，不是"训练出来的能力"。**

更值得注意的数字：**供体 l4u3 在 val 上是 0.537，而继承它全套大脑的 vitl800_clean 只有 0.513，
低了 2.4 个点**（虽然 ep6 尚未收敛，不能直接定性）。

---

## 五、"一次性训练"怎么做

### 1. 重新生成干净的热启动权重（不继承任何 detector 权重）

```bash
cd vendor/DEIMv2
python prepare_dinov3_for_rdttrack.py \
    --dinov3 ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --out    ckpts/dinov3/vitl16_rdttrack_scratch.pth
# 关键：不传 --inherit-aux，也不传 --inherit-encoder
```

产出只有三样东西：
- `backbone.rgb_dinov3` ← 官方 DINOv3 ViT-L 预训练
- `backbone.ir_dinov3` ← 官方权重的 patch_embed 通道复制（3→2）
- `backbone.depth_dinov3` ← 官方权重的 patch_embed 通道复制（3→1）

于是 `encoder`、`decoder`、`convs`、`aux_proj`、`sta`、`prompt`、`dt_ort`
**全部随机初始化**，训练才真正是"一次跑出来的"。

> 说明：主干用官方 DINOv3 预训练是通用做法（等价于 ImageNet 预训练），不算"用训练好的权重"。
> 真正该排斥的是继承**在目标数据集/任务上已收敛的 detector 权重**。

### 2. 基线必须同口径重训，否则数字无意义

现有 combo（0.5222）也是继承来的，不能作为"一次性训练"的对照。要公平回答
"换 ViT-L 到底涨不涨点"，必须：

| 实验 | 主干 | 起点 | 其余完全一致 |
|---|---|---|---|
| A：scratch 基线 | ViT-S/16plus | 官方 ViT-S 预训练，**随机 head** | 分辨率、增强、lr、epoch |
| B：scratch 目标 | ViT-L/16 | 官方 ViT-L 预训练，**随机 head** | 同上 |

### 3. 顺带必须修掉的 bug

LLRD 被硬编码（`multimodal_backbones.py` 第 1250 / 1546 / 1675 行，三处调用都不传
`top_lr` / `decay`），yml 里的 `lr: 0.0001` 被静默作废，实际只有 7.5e-6 ~ 2e-5。
需加 `llrd_top_lr` / `llrd_decay` 配置项（默认值保持 2e-5 / 0.85 以向后兼容）。

### 4. 时间代价（RTX 5060 Ti 16G，实测）

- 随机 head + 预训练 backbone，DETR 类通常需要 **~100 epoch** 才能收敛到位，
  现有 50 epoch 配置偏紧。
- vitl800_clean 目前约 6 epoch / 35 分钟（ViT-L 三流 @800，batch 12 / accum 4）
  ⇒ 100 epoch 约 **10 小时**；A+B 两组串行约 **15 小时**。
- 建议：先跑 B（ViT-L scratch），再排 A（ViT-S scratch），两者都不中断。

---

## 六、待确认

1. 是否启动「scratch 双实验」（最科学，约 15 小时）？还是只跑 ViT-L scratch（约 10 小时，但缺同口径对照）？
2. epoch 数定多少：50（沿用现配置，可能欠收敛）还是 100（推荐）？
3. 分辨率：两者都用 800？还是完整沿用现有 640 / 800 的设定？
