# Mr. DETR-Align* Swin-L（COCO 61.8）在 AIC2026 12 类数据上的微调 + 推理结果

**日期**：2026-11-01 · **执行**：Lead（环境由 teammate `env-builder` 搭建）· **状态**：✅ 已完成（训练 + 验证推理）
**目标权重**：`MrDETR_align_swinl_12ep_900q_objects365_processed.pth`（HF 官方，903,827,174 B，md5 `a2ad7c3979a78bcd28e4c885fb4e4c85`）
**配置**：`projects/mr_detr_align/configs/aic_swinl_finetune.py`

---

## 1. 结论

| 指标（200 张 AIC 验证集） | 值 |
|---|---|
| **mAP@[.5:.95]（bbox）** | **43.55** |
| AP50 | 66.89 |
| AP75 | 45.60 |
| APs（小目标） | 21.98 |
| APm（中目标） | 38.19 |
| APl（大目标） | 59.30 |
| 训练时长 | **1 小时 37 分**（6300 iter，~0.83 s/iter，峰值显存 ~33 GB） |

起点是 COCO AP 61.8 的 Mr. DETR-Align\*（Swin-L + Objects365 预训练 + 5-scale），在**仅 RGB、12 类、1800 张训练图、7 epoch** 条件下微调到 **43.55 mAP**。
该数字**不能**与 DEIMv2 主线的 val mAP 55.94 直接排名：DEIMv2 v1 是**六通道（RGB+IR+depth）**、输入 1440×2560、50 epoch、且用本项目自有评测脚本；本实验是**单 RGB 通道**、输入长边 1792、7 epoch。差距主要来自模态与训练预算，而非检测器本身。

## 2. 训练曲线（每 2 epoch 验证一次）

| eval 点 | iter | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|---|
| epoch 2 | 1800 | 19.68 | 30.84 | 21.02 | 7.54 | 18.48 | 26.64 |
| epoch 4 | 3600 | 36.78 | 58.29 | 38.26 | 16.68 | 33.63 | 49.60 |
| epoch 6 | 5400 | 43.47 | 66.44 | 45.01 | 22.20 | 37.34 | 59.88 |
| **epoch 7（最终）** | **6300** | **43.55** | **66.89** | **45.60** | **21.98** | **38.19** | **59.30** |

曲线在最后 2 个 epoch 已接近饱和（43.47 → 43.55），说明 7 epoch 基本到位；若延长日程预计还能小幅提升。

## 3. 最终逐类 AP（epoch 7，括号内为验证集实例数）

| 类别 | AP | 类别 | AP | 类别 | AP |
|---|---|---|---|---|---|
| person (574) | 53.64 | boat (14) | 51.22 | animal (352) | 61.26 |
| seat (73) | 61.68 | sign (76) | 26.99 | bicycle (74) | 52.43 |
| car (156) | 48.19 | ball (13) | 45.91 | light (135) | 42.64 |
| garbage can (35) | 28.20 | uav (19) | 50.39 | tricycle (2) | **0.00** |

- **tricycle = 0**：验证集仅 2 个实例，属小样本噪声，不能据此判定该类别能力。
- 弱项：**sign 26.99、garbage can 28.20**；若要继续优化，优先针对这两类做类别重采样/损失加权，而不是拉长训练。

## 4. 做法与口径

| 项 | 值 |
|---|---|
| 框架 | detrex 0.3.0 + detectron2 0.6（pinned `9604f59`），PyTorch 2.12.1+cu130，Python 3.12 |
| 模型 | Swin-L backbone + 5-scale ChannelMapper + 6 层 Deformable-DETR encoder/decoder + **900 query** + two-stage + box refine，**Mr.DETR 三路训练**（o2o 主路 / o2m 指令自注意力路 / sep 独立 FFN 路），推理只走 o2o 主路 |
| 数据 | AIC 官方训练集 1800 图训练 / 200 图验证，12 类；**仅 RGB**（原仓库不支持 6 通道） |
| 初始化 | 官方 COCO 权重：784 个 key 中 **753 个加载成功，31 个 80→12 类形状不符被自动跳过并重初始化**（30 个分类头 + `label_enc`），0 missing |
| 训练 | batch 2 + 梯度累积 8（等效 16）；lr 5e-5（backbone 5e-6、reference_points/sampling_offsets 5e-6），AdamW wd 5e-2；FP16 AMP；7 epoch = 6300 iter；LR 在 80%/95% 处各降一次 |
| 分辨率 | 训练短边 480–1152（步长 64）、长边封顶 **1792**；验证短边 1152、长边封顶 1792 |
| 硬件 | 1× RTX PRO 6000 Blackwell 96 GB；~0.83 s/iter、~25 min/epoch、峰值 ~33 GB |
| 截止约束 | 原计划 12 epoch / 长边 2048；为满足 20:00 截止改为 **7 epoch / 1792**（训练 1h37m） |

## 5. 产物位置（远端节点 `/home/...` 侧的绝对路径）

| 产物 | 路径 |
|---|---|
| 最终权重（2.6 GB） | `/root/aic_run/aic_swinl_12ep/model_final.pth`，另已备份到 `/autodl-fs/data/mrdetr/run/aic_swinl_final/model_final.pth` |
| 中间 checkpoint | `model_0001799/0002699/0003599/0004499/0005399/0006299.pth`（约 2.6 GB/份） |
| 验证集推理结果（200 图 COCO json，8.7 MB） | `/root/aic_run/aic_swinl_12ep/coco_instances_results.json` |
| 训练指标 | `/root/aic_run/aic_swinl_12ep/metrics.json`、`log.txt` |
| 训练日志 | `/root/aic_run/train7b_175023.log` |

## 6. 复现命令

```bash
# 节点上
cd /root/build_mrdetr/mrdetr_src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 训练（7 epoch，batch2 × 梯度累积8）
/root/miniconda3/bin/python projects/mr_detr_align/custom_train_net_swinl_gradaccumulate_fintuning.py \
  --config-file projects/mr_detr_align/configs/aic_swinl_finetune.py \
  --num-gpus 1 --grad-accum-steps 8

# 单独推理/评测（任意 checkpoint）
/root/miniconda3/bin/python projects/mr_detr_align/custom_train_net_swinl_gradaccumulate_fintuning.py \
  --config-file projects/mr_detr_align/configs/aic_swinl_finetune.py \
  --num-gpus 1 --eval-only \
  --opts train.init_checkpoint /root/aic_run/aic_swinl_12ep/model_final.pth
```

## 7. 已知限制与后续建议

1. **仅 RGB**：IR/Depth 完全未用。若要和 DEIMv2 主线公平对比，需要把 Swin-L stem 改成 6 通道（新增通道权重初始化 + 数据集 mapper 改造），工作量约 0.5–1 天。
2. **分辨率**：长边 1792（官方 5-scale 配置可到 2400），小目标 APs 仅 21.98，提升分辨率是最直接的涨点方向。
3. **轮次**：7 epoch 是截止约束下的取舍；曲线末段已趋平，12 epoch 预计 +0.5–1.5 mAP（约 3h）。
4. **类别不平衡**：tricycle/boat/ball 等长尾类样本 <20，建议改用类别均衡采样或分类损失类别加权。
5. **推理口径**：本次"推理"= 200 张验证集 COCO 评测；若要产出**比赛提交包**，还需把 COCO json 转成逐图 txt（class_id/cx/cy/w/h/conf、CRLF、≤100 框），仓库本身没有该转换脚本。
