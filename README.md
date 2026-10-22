# AIC2026 城市场景视觉多模态目标检测
## Multimodal Urban Object Detection (Visible + Infrared + Depth)

[![Framework](https://img.shields.io/badge/Framework-DEIMv2%20%7C%20D--FINE-blue.svg)](vendor/DEIMv2)
[![Flagship](https://img.shields.io/badge/Flagship-Sovereign%20V23%20(1152x2048%20Zero--Distort)-brightgreen.svg)](docs/03_architecture_designs/SOVEREIGN_V23_CORE_SPEC.md)
[![Official Best](https://img.shields.io/badge/Official%20Phase%202-53.555%20(best%20submitted)-gold.svg)](docs/MODEL_RECORDS.md)
[![Hardware](https://img.shields.io/badge/Hardware-RTX%20PRO%206000%2096GB-orange.svg)](docs/02_collaboration_and_specs/PROJECT_SPEC.md)

> **一句话**：用可见光 + 红外 + 深度三模态，在 AIC2026 城市场景赛道上做 12 类目标检测，产出官方格式提交。
> **本文档是全仓唯一入口索引**；动手前先读 [`AGENTS.md`](AGENTS.md)（环境 / 红线 / 命令）。

---

## 一、速览

| 项 | 内容 |
| :--- | :--- |
| **任务** | 三模态 12 类检测；提交为逐图 `.txt`（`class cx cy w h conf`，归一化，CRLF，≤100 框） |
| **12 类** | person, bicycle, car, tricycle, ball, seat, boat, uav, sign, light, animal, garbage can |
| **数据** | 训练 2,000 组（14,432 框，7.22 框/图）；复赛测试 1,000 组**无标签**，官方先验 **7,086 框 / 7.09 框每图** |
| **框架** | **DEIMv2**（DINOv3-ViT / HGNetv2-X + HybridEncoder + D-FINE 式解码器），`vendor/DEIMv2/` |
| **官方最高分** | **Sovereign V23 TADL**：Phase 2 实测 **53.555**（τ=0.45，7,459 框）；历史参照标杆 F5 = 58.105 |
| **最新进展** | 当前主线是**纯净基准 v1**（`vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml`），尚无成绩；V24–V37 的演进与现状以 [`AGENTS.md`](AGENTS.md) §1 为准 |
| **首选可用权重** | `outputs/sovereign_v27_pure_unbiased_2048/best_stg2.pth`（V27）｜AP75 最优：`.../sovereign_v28_disparity_awakening_2048/best_stg1.pth` |
| **环境** | conda `deimv2`；本地 RTX 5060 Ti 调试，远端 `seeta` 96GB 训练 |

**技术主线**（详见各架构规范与编年史）：物理配准与深度治理 → 6 通道正交多模态输入 → 双核长程特征颈（Mamba 2D-SSM + MHLA）→ 骨架交互可变形注意力（DIAM）→ 频域对比正则（FCR）→ TADL 置信度解耦 → 模型内生化后处理（CAA-SAHI / Flip-TTA）。

---

## 二、文档地图（唯一索引）

| 我要… | 读这篇 | 一句话 |
| :--- | :--- | :--- |
| **动手干活**（环境 / 命令 / 红线 / 坑点） | [`AGENTS.md`](AGENTS.md) | Agent 与开发者速查，先读这个 |
| **工程规范 + 全链路流程** | [`docs/02_collaboration_and_specs/PROJECT_SPEC.md`](docs/02_collaboration_and_specs/PROJECT_SPEC.md) | 目录职责、脚本分工、数据契约、六阶段流程、排错索引 |
| **查成绩 / 权重来源** | [`docs/MODEL_RECORDS.md`](docs/MODEL_RECORDS.md) | 唯一实验真相源：表A 训练模型、表B 暗光评测、V20~V28 总账 |
| **V23 旗舰架构（官方最高分）** | [`SOVEREIGN_V23_CORE_SPEC.md`](docs/03_architecture_designs/SOVEREIGN_V23_CORE_SPEC.md) · [`SOVEREIGN_V23_DETAILED_TECHNICAL_SPEC.md`](docs/03_architecture_designs/SOVEREIGN_V23_DETAILED_TECHNICAL_SPEC.md) | V23 核心规范 + 全链路白皮书 |
| **上一代旗舰** | [`SOVEREIGN_V20_CORE_SPEC.md`](docs/03_architecture_designs/SOVEREIGN_V20_CORE_SPEC.md) · [`SOVEREIGN_V20_DETAILED_TECHNICAL_SPEC.md`](docs/03_architecture_designs/SOVEREIGN_V20_DETAILED_TECHNICAL_SPEC.md) | V20 连续流形合流（AP75 0.6577 历史极值） |
| **架构演进全九阶段** | [`docs/03_architecture_designs/`](docs/03_architecture_designs/) | README 总表即阅读顺序：表征解耦 → 双流 → V6 → V10 → V12 → V13 → V14 → V20 → V23（V20 / V23 规范也在此目录） |
| **全周期历程 / 天梯榜** | [`docs/02_collaboration_and_specs/AIC_FULL_CONVERSATION_CHRONICLE.md`](docs/02_collaboration_and_specs/AIC_FULL_CONVERSATION_CHRONICLE.md) | 从零到登顶的决策与突破复盘 |
| **数据预处理 / 对齐 / 深度中台 / 配置字典** | [`docs/02_collaboration_and_specs/AIC_MASTER_COLLABORATION_REGISTRY.md`](docs/02_collaboration_and_specs/AIC_MASTER_COLLABORATION_REGISTRY.md) | 技术中台总册 |
| **攻坚审计与消融定论** | [`docs/04_audits_and_ablations/`](docs/04_audits_and_ablations/) | 红外对齐审计、权重溯源、CP 增强废止定论 |
| **后处理 / 提交合规** | [`docs/05_postprocessing_and_inference/SAFE_POSTPROCESSING_GUIDELINES.md`](docs/05_postprocessing_and_inference/SAFE_POSTPROCESSING_GUIDELINES.md) | 内生化范式、H-Flip 共识、红线禁令 |
| **测试集先验（门限设定依据）** | [`docs/01_competition/SUBMISSION_CONFIDENCE_THRESHOLD_PRIOR.md`](docs/01_competition/SUBMISSION_CONFIDENCE_THRESHOLD_PRIOR.md) | 7,086 目标 / 7.09 密度 / 尺寸分布 |
| **官方赛题材料** | [`docs/01_competition/`](docs/01_competition/) | 竞赛细则 PDF、提交格式范例、综述 |
| **半决赛提交** | [`docs/06_semifinal_submission/`](docs/06_semifinal_submission/) | 技术报告草稿骨架、代码 / 模型 / 环境提交清单与已知风险 |
| **纯净基准 / 任务地图** | [`docs/02_collaboration_and_specs/CLEAN_BASELINE.md`](docs/02_collaboration_and_specs/CLEAN_BASELINE.md) · [`AIC_TASK_MAP.md`](docs/02_collaboration_and_specs/AIC_TASK_MAP.md) | 当前主线的定义、14 项修复与任务完成度 |
| **仓库整理 / 找已删旧文件** | [`docs/02_collaboration_and_specs/REPO_CLEANUP_LOG.md`](docs/02_collaboration_and_specs/REPO_CLEANUP_LOG.md) | 2026-10-22 删除 1,356 个历史实验文件的依据、清单与取回方法 |

---

## 三、快速上手（三条命令）

```bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:${PYTHONPATH:-}"
PY=/home/constant/miniconda3/envs/deimv2/bin/python

# 训练纯净基准（必须 cd vendor/DEIMv2；不加 --use-amp；换卡流程见 CLOUD_MIGRATION_RUNBOOK.md）
cd vendor/DEIMv2 && $PY train.py -c configs/deimv2/aic_clean_baseline_v1.yml

# 推理产提交包（统一入口，参数与示例见脚本头部；--zip 时自动做格式校验）
$PY scripts/inference/predict.py --config CFG --checkpoint CKPT --images <RGB_DIR> ...

# 提交前格式校验（必跑）
$PY scripts/eval/check_submission.py <SUBMISSION_DIR> dataset/test/AIC2026_PHASE_2_1000/visible
```

---

## 四、合规红线

1. **禁止多模型集成**：赛题第十一条禁止不同结构/不同训练阶段模型的投票、平均、WBF 等简单集成 —— 原 `scripts/archive_legacy_postprocess/` 下的这类脚本已于 2026-10-22 删除，不得恢复用于提交。
2. **只允许单模型内生化**：CAA-SAHI、内生化 Flip-TTA、同模型多尺度推断。
3. **禁止测试集投机**：不得读取/推断测试集标注，不得做全局配额排序外挂或测试集截断。
4. **禁止外部数据进正式训练**：`data_aux/`（BDD100K、FLIR、LLVIP 等）仅用于泛化/消融实验。

---

## 五、目录结构

```
docs/            唯一文档中心（顶层仅 MODEL_RECORDS.md；01 赛题 / 02 规范与中台 / 03 架构 / 04 审计 / 05 后处理 / 06 半决赛）
vendor/DEIMv2/   核心框架与训练入口 train.py
scripts/         prep 数据准备 · train 训练驱动 · inference 推理提交 · eval 评测把关 · analysis 分析 · diagnostics 契约测试与诊断 · viz 工作台 · lib 公共库
dataset/         原始三模态数据（只读）      work_dir/  运行时产物（submissions/audits/visualizations/archives 分层）
weight/ weights/ 权重中心                    logs/      唯一日志落点
outputs/         分析产物（analysis/figures/metrics/reports/submissions 分层）  archive/  历史草稿、快照与备份
AGENTS.md        环境 / 红线 / 命令速查        MODEL_RECORDS.md  实验成绩真相源（软链）
```
