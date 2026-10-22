# 仓库整理记录

> 历次整理的完整记录。`AGENTS.md` 只保留一句指针；本文件按时间倒序。

## 2026-10-22：删除历史实验文件，去掉文档软链层

**起点**：commit `4d71413`（已在 `origin/master`），本地 annotated tag `pre-cleanup-20261022` 指向它。
**结果**：跟踪文件 1,812 → 约 440；`scripts/` 约 750 → 160；`vendor/DEIMv2/configs/deimv2` 782 → 65。`engine/` 一个文件都没动。

### 保留了什么

保留规则：从「当前主线 + 里程碑 + 框架」出发，沿脚本里的路径与 import、配置里的 `__include__` 做传递闭包，闭包内的文件全部保留。

| 类别 | 内容 |
|---|---|
| 框架 | `vendor/DEIMv2/` 的 `engine/`、`tools/`、`train.py`、`configs/{base,deim_dfine,deim_rtdetrv2}/`、上游 COCO 配置、`LICENSE.md` / `README.md` / `requirements.txt` |
| 当前主线 | 纯净基准 v1 三臂配置（`aic_clean_baseline_v1{,_rgb_control,_night}.yml`）、启动与迁移脚本、`scripts/inference/predict.py`、契约测试、模态鲁棒性评测、配对比较、V36 平移诊断全套（`scripts/diagnostics/`、`scripts/analysis/`） |
| 里程碑（仍可运行） | V23（官方最高 53.555）、F5（`deim_hgnetv2_x_aic_obj2coco_ft.yml`）、V27、V28-A、V34–V37（含 V36 机制 / 注入热启动子目录）的配置、启动脚本与推理脚本 |
| 数据管线 | `prepare_dataset.py` + `yolo_to_coco.py`（1800/200 划分，seed=42；后者从 `scripts/eval/` 移到 `scripts/prep/`）、`batch_generate_server_ir_lbg.py` + `align_ir_pixel_pipeline_v3.py`（训练集配准 / LBG 红外）、`gen_ir_like_train.py` + `align_ir_pixel_pipeline.py`（测试集同法重生成）、`prep_ir_localbg.py`、`prep_nightxx_server.py`（`eval_modality_robustness.py` 用的 9 档暗光验证集）、`pack_training_inputs.py` |
| 工具 | `check_submission.py`、`eval_submission.py`、`official_competition_evaluator.py`、`collect_model_records.py`、可视化工作台（`import_submission_to_workstation.py`、`workstation_server.py` 及其依赖） |

### 删除了什么

1,356 个文件，完整清单：[`REPO_CLEANUP_20261022_removed.txt`](REPO_CLEANUP_20261022_removed.txt)。

| 目录 | 删除数 | 主要内容 |
|---|---:|---|
| `vendor/DEIMv2/configs/deimv2` | 717 | V1–V22、V24–V26、V29–V33 的训练配置；DINOv3-L / RDTTrack / RDTDiffuse / 双流 / LLVIP / FLIR / BDD 系列；按模板批量生成的 `_val_night*` 验证配置；V28-B 72 轮配置（已判定禁止使用） |
| `scripts/analysis` | 193 | 一次性分析、画图、扫描日志、`eval_exp*` / `eval_round*` 系列 |
| `scripts/train` | 178 | 上述实验的启动 / 排队 / 续训脚本 |
| `scripts/eval` | 61 | 旧版评测、夜间套件、模型汤、旧 `predict_*` / `tta_*`（多数依赖已废弃的 `rfdetr` 环境）、`predict_mm_tta.py`（旧通道布局 ch3=heat，只服务旧 l4u3 模型；统一入口已是 `predict.py`） |
| `scripts/gen` | 49 | 旧实验配置生成器（整个目录） |
| `scripts/prep` | 37 | 旧深度精修、外部数据集（LLVIP / FLIR / M3FD / BDD）准备、deepir 缓存、v2 配准管线 |
| `scripts/viz` | 36 | 针对单张图 / 单个版本对比的面板与画廊 |
| `scripts/inference` | 23 | V18–V22、V24–V26、V29–V31 推理与打包脚本 |
| `vendor/DEIMv2/configs/dataset` | 22 | 只被已删配置 include 的 `aic_detection_{night*,dndt_*,cp*,nightify}.yml`、`aic_heat.yml` |
| `scripts/lib` | 11 | 旧后处理（共面过滤、无人机验证、UAV 热精修）、SAM 掩码库构建、冒烟 / 校验脚本、`prep_night_real.py`（被 `prep_nightxx_server.py` 取代） |
| `scripts/archive_legacy_postprocess` | 9 | 多模型 WBF、外部 CAA-SAHI 切片、多阈值投票——赛题第十一条禁止的集成手段，整个目录删除 |
| 其他 | 20 | `vendor/DEIMv2/` 顶层的 `diag_nan*.py`、`prep_deepir_cache*.py`、`prepare_*_for_rdttrack.py`；`scripts/{utils,audit}/`；V25–V31 时期的 4 个诊断脚本与 `test_all_targeted_repairs.py`（它是 V26 测试集 GPU 后处理实验，不是契约测试）；`scripts/test_v28_72ep_dryrun.py`；`scripts/tta_eval.py` 软链 |

### 文档层

- 去掉全部 21 个快捷软链（其中 3 个已断：`docs/training_records*.md`、`docs/llvip_dl.md` 指向不入库的 `docs/archive/`）：`docs/` 顶层 13 个扁平软链、`docs/02_…/{CODEBUDDY,MODEL_RECORDS}.md`、`docs/05_…/SUBMISSION_CONFIDENCE_THRESHOLD_PRIOR.md`、`docs/03_…/09_sovereign_v23_master_architecture_spec.md`、根目录 5 个规范快捷方式。根目录只留 `MODEL_RECORDS.md`、`CODEBUDDY.md` 两个软链。
- 4 份 V20 / V23 规范从 `docs/` 顶层移到 `docs/03_architecture_designs/`（文件名不变）；`AIC_RAW_DATA_PROCESSING_AND_COMPLETE_MODEL_IMPROVEMENTS_MASTER.md` 移到 `docs/02_collaboration_and_specs/`。`docs/` 顶层现在只有 `MODEL_RECORDS.md`（`collect_model_records.py` 的写入目标）和 6 个分类目录。
- 运营文档（`AGENTS.md`、`CLAUDE.md`、`README.md`、`PROJECT_SPEC.md`、架构 README）里指向已删 / 已移文件的引用已改写。`MODEL_RECORDS.md`、编年史、中台总册、架构设计等**历史叙述**里提到的旧脚本 / 配置名保持原样，按上面的清单到 tag 里取。

### 怎么取回

```bash
git show pre-cleanup-20261022:scripts/train/run_sovereign_v29.sh           # 只看内容
git checkout pre-cleanup-20261022 -- vendor/DEIMv2/configs/deimv2/X.yml     # 恢复到工作树
```

tag 只在本地；其他机器没有这个 tag 时把 `pre-cleanup-20261022` 换成 `4d71413`。Linux 端 `git pull` 后这些文件会从工作树消失，`work_dir/`、`outputs/`、`logs/`、权重不受影响。

## 2026-10-07：位置归位与文档归纳

仓库做过一次结构整理，**流程与数据零改动，仅位置归位**。完整日志：`logs/cleanup_20261007.log`；回滚快照：`archive/backups/`。（以下为当时记录，其中提到的部分软链与脚本已在 2026-10-22 删除。）

| 原位置 | 新位置 | 说明 |
|---|---|---|
| 根目录 `align_viz_*.jpg`（1000 张） | `work_dir/test_align_viz/` | 配准可视化面板，回归脚本默认 `--viz-dir` 与审计文档所述位置 |
| 根目录 `test_aligned_ir.tar` / `test_depth_p0.tar` | `work_dir/` | 测试集派生缓存归档（`test_aligned_ir` 已解包同目录） |
| 根目录 `F5_all2000_*_fp16.pth` | 删除（保留 `weights/F5.pth`） | 二者 md5 完全一致，脚本引用已改指 `weights/F5.pth` |
| 根目录 `index.html` | `outputs/reports/audit_workbench_superres_collision.html` | 生成式审计工作台报告 |
| 根目录 `docs.zip` / `logs.zip` | `archive/snapshots/` | 过期快照（新快照在 `archive/backups/`） |
| 根目录 `run_sovereign_v27/v28*.sh` | `scripts/train/` | 训练驱动归位 |
| 根目录 `test_vram_measure.py` | `scripts/diagnostics/` | 探针脚本归位 |
| 根目录 `scratch/` | `archive/scratch/`（目录已删） | 一次性草稿 |
| 根目录 `cv-model-improvement-master/`（616MB） | `vendor/` | 外部资源合集，非本项目流程依赖 |
| 根目录 `AIC2026_migration_20261003/` | `archive/` | 跨机迁移历史快照 |
| `docs/README (2).md` | 根目录 `README.md` | 本身就是项目门面文档，此前放错位置 |
| `docs/02_collaboration_and_specs/MODEL_RECORDS.md`（重复正本） | 软链 → `../MODEL_RECORDS.md` | 唯一正本为 `docs/MODEL_RECORDS.md`（`collect_model_records.py` 的写入目标） |
| `docs/03_architecture_designs/08_sovereign_v23_master_architecture_spec.md`（重复正本） | 改名 `09_…` 并设为软链 → `../../SOVEREIGN_V23_DETAILED_TECHNICAL_SPEC.md` | 消除与 V23 详细白皮书的内容重复 |
| `06_sovereign_v14_heterogeneous_…` / `07_sovereign_v20_master_…` | `07_…` / `08_…` | 修复两个 `06_` 编号冲突，编号 01→09 无断档 |
| `docs/AIC_MASTER_COLLABORATION_REGISTRY.md`、`docs/02_…/CODEBUDDY.md` | 修正软链目标 | 此前为断链 |
| `work_dir/{SAFE_POSTPROCESSING_GUIDELINES,SUBMISSION_CONFIDENCE_THRESHOLD_PRIOR,CP_tricycle_verdict}.md` | `archive/migrated_docs_from_work_dir/` | 已迁入 `docs/` 的三份文档副本，正本在 `docs/01_*`、`docs/04_*`、`docs/05_*` |

### 文档归纳（同日第二步）

原则：**每类信息只留一处正本**，导航层收敛为「根 `README.md` 索引 + `AGENTS.md` 速查 + `PROJECT_SPEC.md` 规范」三层。
被归并文件全部移入 `archive/merged_docs_20261007/`（未删除），快照 `archive/backups/docs_before_merge_20261007.tar.gz`。

| 被归并文件 | 去处 |
|---|---|
| `docs/README.md` + `docs/{01,02,04,05}/README.md` + `docs/archive/README.md`（6 份分类索引） | 内容并入根 `README.md`「文档地图」；`03_architecture_designs/README.md` 因含九阶段总表而保留 |
| `docs/CODEBUDDY.md` | 并入 `AGENTS.md`；根 `CODEBUDDY.md` 软链改指 `AGENTS.md`（旧路径仍可用） |
| `docs/02_.../PROJECT_WORKFLOW.md` | 并入 `PROJECT_SPEC.md` §7 全链路流程 / §8 最小闭环命令 / §9 排错索引 |
| `docs/02_.../aic_full_chat_chronicle_and_technical_master_record.md`（996 行，止于 V14） | 已被 `AIC_FULL_CONVERSATION_CHRONICLE.md`（含 V16~V23）取代，扁平软链改指正本 |
| `docs/05_.../SUBMISSION_CONFIDENCE_THRESHOLD_PRIOR.md`（与 01 目录同文） | 改为软链 → `../01_competition/SUBMISSION_CONFIDENCE_THRESHOLD_PRIOR.md` |

### 子目录分层（同日第三步：work_dir / outputs）

**work_dir 顶层 52 → 14 项；outputs 顶层 117 → 9 项**，共移动 154 项，全部为 `mv`（可回滚）。

| 位置 | 归入 | 内容 |
|---|---|---|
| `work_dir/submissions/` | 15 项 | `submission_*` / `submissions_v22_*` / `test1_predictions_*` / `v23_*submission` / `v23_unfiltered_top100` |
| `work_dir/audits/` | 17 项 | `audit_*`(8) / `f5_reproduce` / `tadl` / `test_align_probe` / `test_compact` / `masklib_*` / `vis_*` / `logs_text` |
| `work_dir/visualizations/` | 6 项 | `v23_test_vis_all1000(+zip)` / `figures` / `fusion_viz` / `heat_overlay` / `test_align_viz` |
| `work_dir/archives/` | 4 项 | 两个测试集 tar + `cached_*_preds.pt`（第二步的两个 tar 已进一步归入此处） |
| `work_dir/` 原位保留 | — | **训练/推理输入，代码与 config 直接引用**：`deimv2_dataset` `test_aligned_ir` `test_lbg_ir` `rfdetr_dataset` `LLVIP` `M3FD_raw` `FLIR_raw` `da2_weights` `dav2_token_cache` `assets` |
| `outputs/analysis/` | 25 项 | `discrepancies_f5_vs_*` / `*_analysis_*` / `visualizations_*` / `_inspect` `_dets*` / `nightxx_eval` `crops` `audit*` 等 |
| `outputs/{figures,metrics,reports}/` | 按类型 | 散落的 `*.png` / `*.json|npz|txt` / `*.html|md|py` |
| `outputs/submissions/` | 6 项 | `submissions_v23_native`…`v28` 并入既有 `submissions/` |
| `outputs/checkpoints/` | 1 项 | `sovereign_v28_full72ep_zoom_awakening_8gpu` |
| `outputs/` 原位保留 | — | `official_submission_package/`（官方提交包，勿动）、`README.md` |

- **引用同步**：对 `scripts/` 与 `vendor/DEIMv2/configs` 的 **1361 个文件**应用 **462 条路径替换规则**（含 `a/b/c`、`os.path.join('a','b')`、绝对路径三种写法）；校验结果：旧路径残留 0、脚本残留引用 0。
- **修复 2 个因移动失效的相对软链**：`outputs/reports/weight_provenance_report.md`、`outputs/analysis/dataset`。
- 移动清单（回滚用）：`archive/backups/move_manifest_20261007.tsv`；日志 `logs/cleanup_subdirs_20261007.log`。
- ⚠️ **遗留（非本次造成）**：`work_dir/LLVIP` 有 30,976 个断链（指向已删除的 `work_dir/LLVIP_zip/`），`data_aux/` 有 38,477 个断链——均为外部/消融数据集，不影响主流程。
