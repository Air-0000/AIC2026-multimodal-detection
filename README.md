# AIC2026 城市场景视觉多模态目标检测
## Multimodal Urban Object Detection (Visible + Infrared + Depth)

[![Framework](https://img.shields.io/badge/Framework-DEIMv2%20%7C%20D--FINE-blue.svg)](vendor/DEIMv2)
[![Flagship](https://img.shields.io/badge/Flagship-Sovereign%20V23%20(1152x2048%20Zero--Distort)-brightgreen.svg)](docs/03_architecture_designs/SOVEREIGN_V23_CORE_SPEC.md)
[![Official Best](https://img.shields.io/badge/Official%20Phase%202-53.555%20(best%20submitted)-gold.svg)](docs/MODEL_RECORDS.md)
[![Hardware](https://img.shields.io/badge/Hardware-RTX%20PRO%206000%2096GB-orange.svg)](docs/02_collaboration_and_specs/PROJECT_SPEC.md)
[![Repo](https://img.shields.io/badge/Repo-3rd%20cleanup%20%282026--11--24%29-lightgrey.svg)](docs/02_collaboration_and_specs/REPO_CLEANUP_LOG.md)
[![Branch](https://img.shields.io/badge/Branch-codino--rgb--ceiling%20%28Co--DINO%20gate%29-blueviolet.svg)](https://github.com/Air-0000/AIC2026-multimodal-detection/tree/codino-rgb-ceiling)

> **一句话**：用可见光 + 红外 + 深度三模态，在 AIC2026 城市场景赛道上做 12 类目标检测，产出官方格式提交。
> **本文档是全仓唯一入口索引**；动手前先读 [`AGENTS.md`](AGENTS.md)（环境 / 红线 / 命令）。

---

## 一、速览

| 项 | 内容 |
| :--- | :--- |
| **任务** | 三模态 12 类检测；提交为逐图 `.txt`（`class cx cy w h conf`，归一化，CRLF，≤100 框） |
| **12 类**（官方 class_id 0–11 顺序） | person, boat, animal, seat, sign, bicycle, car, ball, light, garbage can, uav, tricycle |
| **数据** | 训练 2,000 组（官方统计 14,432 框、7.22 框/图）；测试两组各 1,000 组**无标签**：Phase 2 旧布局 `dataset/test/{visible,depth,infrared}`、Phase 3 `dataset/test/AIC2026_PHASE_3_1000/`；官方先验 7,086 框 / 7.09 框每图（Phase 2） |
| **框架** | **DEIMv2**（DINOv3-ViT / HGNetv2-X + HybridEncoder + D-FINE 式解码器），`vendor/DEIMv2/` |
| **官方最高分** | **Sovereign V23 TADL**：Phase 2 实测 **53.555**（τ=0.45，7,459 框）；历史参照标杆 F5 = 58.105；决赛 Phase 3 上纯净基准 v1 首个官方成绩 **51.1**（不同测试集，不可横比）。截至 2026-11-24 无新增官方提交 |
| **最新进展** | 当前主线仍是**纯净基准 v1**（`vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml`）：seeta 训练停在 epoch 45、验证集最佳 epoch 37（mAP 55.94），Phase 3 官方 **51.1**（MODEL_RECORDS §四十六）；10-26 完成模态鲁棒性剖面与红外绑定 pilot 配置（§四十七，仅配置未训练）；11-01～02 两条支线（§四十九 / §五十）只做评估与机制 pilot，**未改变主线、未用于提交**。V24–V37 的演进与现状以 [`AGENTS.md`](AGENTS.md) §1 为准 |
| **首选可用权重** | 本机：`weights/sovereign_v23_best_stg2_ep71.pth`（V23，官方最高分）、`weights/sovereign_v23_tadl_best_stg2.pth`（TADL 变体）、`weight/sovereign_v23_best_stg1_e33.pth`（Stage A）、`weights/F5.pth`（标杆 58.105）、`outputs/checkpoints/aic_clean_baseline_v1_seeta/best.pth`（当前主线，Phase 3 提交所用）、`outputs/checkpoints/s0918_gate_20261101/epoch_4.pth`（11 月 Co-DINO 门控 pilot，911 MiB）。V27 / V28-A 权重本机已不存在（历史台账在远端） |
| **环境** | conda `deimv2`（训练 / 评测 / 推理）；本地 RTX 5060 Ti 调试，远端 `seeta` RTX PRO 6000 96GB 训练 |

### 近期变更（2026-11-24）

- **第三轮仓库清理完成**：Git 跟踪文件 **318 → 325**（本轮不删跟踪文件），本地一次性产物清理 ≈**69.3 MB**，根目录 `index.html`（46.9 MB）移入 `outputs/reports/`；清理前快照 tag `pre-cleanup-20261124`（= `6e56b74`）已推送 GitHub。依据、清单与取回方式见 [`REPO_CLEANUP_LOG.md`](docs/02_collaboration_and_specs/REPO_CLEANUP_LOG.md)。
- **11 月支线入库**：Mr.DETR / Mr.DETR++ 拆解评估 6 份报告 → [`docs/04_audits_and_ablations/mrdetr_2026-11-01/`](docs/04_audits_and_ablations/mrdetr_2026-11-01/00_REPORT.md)；Co-DINO「冻结检测器 + 夜间亮度门控残差分支」脚本 24 个 → 分支 `codino-rgb-ceiling` 的 `scripts/train/codino/swinl_rgb_full/` 与 `scripts/train/codino/aic_mm/`（含 README）。
- **两个 11 月数据点（都不是主线成绩）**：Mr.DETR-Align Swin-L 仅 RGB 微调在 AIC 200 图上 **mAP 43.55**（7 epoch，不能与六通道 v1 的 55.94 排名）；Co-DINO 门控 pilot 契约全部通过（正常光与底座逐位一致、只训练分支、Phase 3 门控触发 0/1000），但**无训练收益证据、未用于提交**。记录见 [`MODEL_RECORDS.md`](docs/MODEL_RECORDS.md) §四十九 / §五十。
- **文档脱敏**：11-21 已从 master 隐去云主机 SSH 地址 / 端口；11-24 归档分支 tip 同步合并了已脱敏的 master（merge `e130b7b`），不再露出旧主机信息（历史对象与原 `pre-cleanup-*` tag 保持原样，不改写）。

**技术主线**（详见各架构规范与编年史）：物理配准与深度治理 → 6 通道正交多模态输入 → 双核长程特征颈（Mamba 2D-SSM + MHLA）→ 骨架交互可变形注意力（DIAM）→ 频域对比正则（FCR）→ TADL 置信度解耦 → 模型内生化后处理（CAA-SAHI / Flip-TTA）。

---

## 二、文档地图（唯一索引）

| 我要… | 读这篇 | 一句话 |
| :--- | :--- | :--- |
| **动手干活**（环境 / 命令 / 红线 / 坑点） | [`AGENTS.md`](AGENTS.md) | Agent 与开发者速查，先读这个 |
| **工程规范 + 全链路流程** | [`docs/02_collaboration_and_specs/PROJECT_SPEC.md`](docs/02_collaboration_and_specs/PROJECT_SPEC.md) | 目录职责、脚本分工、数据契约、六阶段流程、排错索引 |
| **查成绩 / 权重来源** | [`docs/MODEL_RECORDS.md`](docs/MODEL_RECORDS.md) | 唯一实验真相源：表A 训练模型、表B 暗光评测、V20~V37 + 纯净基准 v1 + 11 月支线（§三十五–五十）总账 |
| **V23 旗舰架构（官方最高分）** | [`SOVEREIGN_V23_CORE_SPEC.md`](docs/03_architecture_designs/SOVEREIGN_V23_CORE_SPEC.md) · [`SOVEREIGN_V23_DETAILED_TECHNICAL_SPEC.md`](docs/03_architecture_designs/SOVEREIGN_V23_DETAILED_TECHNICAL_SPEC.md) | V23 核心规范 + 全链路白皮书 |
| **上一代旗舰** | [`SOVEREIGN_V20_CORE_SPEC.md`](docs/03_architecture_designs/SOVEREIGN_V20_CORE_SPEC.md) · [`SOVEREIGN_V20_DETAILED_TECHNICAL_SPEC.md`](docs/03_architecture_designs/SOVEREIGN_V20_DETAILED_TECHNICAL_SPEC.md) | V20 连续流形合流（AP75 0.6577 历史极值） |
| **架构演进全九阶段** | [`docs/03_architecture_designs/`](docs/03_architecture_designs/) | README 总表即阅读顺序：表征解耦 → 双流 → V6 → V10 → V12 → V13 → V14 → V20 → V23（01–08 按编号，第 9 部为 `SOVEREIGN_V23_*`；V20 / V23 规范文件也在本目录） |
| **全周期技术综述（历史）** | [`AIC_RAW_DATA_PROCESSING_AND_COMPLETE_MODEL_IMPROVEMENTS_MASTER.md`](docs/02_collaboration_and_specs/AIC_RAW_DATA_PROCESSING_AND_COMPLETE_MODEL_IMPROVEMENTS_MASTER.md) | 数据全流程 + 模型改进的整合视角，与架构白皮书互补 |
| **全周期历程 / 天梯榜** | [`docs/02_collaboration_and_specs/AIC_FULL_CONVERSATION_CHRONICLE.md`](docs/02_collaboration_and_specs/AIC_FULL_CONVERSATION_CHRONICLE.md) | 从零到登顶的决策与突破复盘 |
| **数据预处理 / 对齐 / 深度中台 / 配置字典** | [`docs/02_collaboration_and_specs/AIC_MASTER_COLLABORATION_REGISTRY.md`](docs/02_collaboration_and_specs/AIC_MASTER_COLLABORATION_REGISTRY.md) | 技术中台总册 |
| **攻坚审计与消融定论** | [`docs/04_audits_and_ablations/`](docs/04_audits_and_ablations/) | 红外对齐审计、权重溯源、CP 增强废止定论、Mr.DETR++ 评估与纯 RGB 上限实测（`mrdetr_2026-11-01/`） |
| **Co-DINO 夜间门控支线代码（分支）** | [`codino-rgb-ceiling: swinl_rgb_full/`](https://github.com/Air-0000/AIC2026-multimodal-detection/tree/codino-rgb-ceiling/scripts/train/codino/swinl_rgb_full) | 11 月 G0 / G1 两代门控实现、契约与评测脚本、README；未用于提交（记录见 MODEL_RECORDS §五十） |
| **后处理 / 提交合规** | [`docs/05_postprocessing_and_inference/SAFE_POSTPROCESSING_GUIDELINES.md`](docs/05_postprocessing_and_inference/SAFE_POSTPROCESSING_GUIDELINES.md) | 内生化范式、H-Flip 共识、红线禁令 |
| **测试集先验（门限设定依据）** | [`docs/01_competition/SUBMISSION_CONFIDENCE_THRESHOLD_PRIOR.md`](docs/01_competition/SUBMISSION_CONFIDENCE_THRESHOLD_PRIOR.md) | 7,086 目标 / 7.09 密度 / 尺寸分布 |
| **官方赛题材料** | [`docs/01_competition/`](docs/01_competition/) | 竞赛细则 PDF、提交格式范例、综述 |
| **半决赛提交** | [`docs/06_semifinal_submission/`](docs/06_semifinal_submission/) | 技术报告草稿骨架、代码 / 模型 / 环境提交清单与已知风险 |
| **纯净基准 / 任务地图** | [`docs/02_collaboration_and_specs/CLEAN_BASELINE.md`](docs/02_collaboration_and_specs/CLEAN_BASELINE.md) · [`AIC_TASK_MAP.md`](docs/02_collaboration_and_specs/AIC_TASK_MAP.md) | 当前主线的定义、14 项修复与任务完成度 |
| **仓库整理 / 找已删旧文件** | [`docs/02_collaboration_and_specs/REPO_CLEANUP_LOG.md`](docs/02_collaboration_and_specs/REPO_CLEANUP_LOG.md) | 2026-10-22（1,356 文件）、2026-10-31（134 文件 + 约 87 GB 本地产物）、2026-11-24（11 月支线入库 + 约 69 MB 本地产物）三轮清理的依据、清单与 tag 取回方法 |

---

## 三、快速上手（三条命令）

```bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:${PYTHONPATH:-}"
PY=/home/constant/miniconda3/envs/deimv2/bin/python

# 训练纯净基准（必须 cd vendor/DEIMv2；SM<8.0 的 V100 必须加 -u use_amp=False，其它架构按配置走 BF16+TF32；换卡流程见 CLOUD_MIGRATION_RUNBOOK.md）
cd vendor/DEIMv2 && $PY train.py -c configs/deimv2/aic_clean_baseline_v1.yml

# 推理产提交包（统一入口，参数与示例见脚本头部；--zip 时自动做格式校验）
$PY scripts/inference/predict.py --config CFG --checkpoint CKPT --images <RGB_DIR> ...

# 提交前格式校验（必跑；Phase 3 用 AIC2026_PHASE_3_1000，Phase 2 旧布局用 dataset/test/visible）
$PY scripts/eval/check_submission.py <SUBMISSION_DIR> dataset/test/AIC2026_PHASE_3_1000/visible
```

> 训练派生数据（1800 / 200 划分）与夜化验证集不在 git 里：分别用 `scripts/prep/prepare_dataset.py` + `scripts/prep/yolo_to_coco.py`、`scripts/prep/prep_nightxx_server.py` 重新生成。

---

## 四、合规红线

1. **禁止多模型集成**：赛题第十一条禁止不同结构/不同训练阶段模型的投票、平均、WBF 等简单集成 —— 原 `scripts/archive_legacy_postprocess/` 下的这类脚本已于 2026-10-22 删除，不得恢复用于提交。
2. **只允许单模型内生化**：CAA-SAHI、内生化 Flip-TTA、同模型多尺度推断。
3. **禁止测试集投机**：不得读取/推断测试集标注，不得做全局配额排序外挂或测试集截断。
4. **禁止外部数据进正式训练**：BDD100K / FLIR / LLVIP / M3FD 等外部集已于 2026-10-31 清出本机（曾仅用于泛化/消融实验）；二轮清理后 `data_aux/` 已不存在，模态鲁棒性评测所需的 9 档夜化验证集可用 `scripts/prep/prep_nightxx_server.py` 重新生成。

---

## 五、目录结构

```
docs/            唯一文档中心（顶层仅 MODEL_RECORDS.md；01 赛题 / 02 规范与中台 / 03 架构 / 04 审计 / 05 后处理 / 06 半决赛）
vendor/DEIMv2/   核心框架与训练入口 train.py（outputs/ 训练产物，git 忽略）
scripts/         prep 数据准备 · train 训练驱动 · inference 推理提交 · eval 评测把关 · analysis 分析 · diagnostics 契约测试与诊断 · viz 工作台 · lib 公共库
dataset/         原始三模态数据（只读）：train 2,000 组；test 为 Phase 2 旧布局 {visible,depth,infrared} 与 Phase 3 AIC2026_PHASE_3_1000/，各 1,000 组无标签
work_dir/        运行时产物（submissions/audits/visualizations/archives/transfer）；训练派生数据与夜化集可重新生成（见第三节）
weight/ weights/ 权重中心（V23 / F5 / RealESRGAN / DINOv3 backbone）
outputs/         分析产物（analysis/figures/metrics/reports/submissions/checkpoints/official_submission_package）
logs/            唯一日志落点
archive/         历史草稿、快照与备份（含三轮清理前快照备份）
AGENTS.md        环境 / 红线 / 命令速查        MODEL_RECORDS.md  实验成绩真相源（软链）
CLAUDE.md        补充约定（精度 / 测试 / 坑点）  CODEBUDDY.md      旧入口（软链 → AGENTS.md）
```

---

## 六、分支与 tag

| ref | 用途 |
| :--- | :--- |
| `master` | 主线：DEIMv2 纯净基准 v1 + 里程碑（V23 / F5 / V27 / V28-A / V34–V37）+ 11 月支线记录 |
| `codino-rgb-ceiling` | Co-DINO 支线归档：10 月 Co-DINO 环境 / 配置 / 推理脚本 + 11 月夜间门控 G0 / G1 代码（24 文件 + README）；底座权重不在本仓、**未用于提交** |
| `master_backup_20261022` | 2026-10-22 清理前备份（仅本地，未推送） |
| tag `pre-cleanup-20261022` / `pre-cleanup-20261031` / `pre-cleanup-20261124` | 三轮清理前快照（分别 = `4d71413` / `2c170ef` / `6e56b74`），均已推送 GitHub；旧文件用对应 tag 取回 |

```bash
git fetch --tags
git show pre-cleanup-20261124:docs/MODEL_RECORDS.md     # 清理前文档快照
git checkout codino-rgb-ceiling -- scripts/train/codino  # 取 Co-DINO 支线代码
```
