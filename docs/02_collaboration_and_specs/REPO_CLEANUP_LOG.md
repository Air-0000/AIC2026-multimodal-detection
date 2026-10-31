# 仓库整理记录

> 历次整理的完整记录。`AGENTS.md` 只保留一句指针；本文件按时间倒序。

## 2026-10-31：第二轮清理（已结案实验代码 + 约 87 GB 本地产物）

**起点**：commit `2c170ef`（= 当时 `origin/master`），annotated tag `pre-cleanup-20261031` 指向它，**已推送到 GitHub**（`git ls-remote --tags origin` 可见；其他机器 `git fetch --tags` 后即可取回，不再有 2026-10-22 条目中"tag 只在本地"的限制）。
**结果**：Git 跟踪文件 451 → 317（删除 135 个文件，新增 1 份本清单）；`scripts/` 168 → 74；`vendor/DEIMv2/configs/deimv2` 67 → 31；`docs/` 41 → 41（删 1 份生成式 HTML、增 1 份清单）。本地被 ignore 的过期实验产物清理约 87 GB（磁盘已用 94% → 84%）。`engine/` 一个文件都没动，`dataset/` 未动。
> 注：清理提交 rebase 到 `origin/master` 后，远端另有 `7539f1d`（红外数据核查，MODEL_RECORDS §四十八）新增 `scripts/analysis/audit_ir_alignment.py`，最终跟踪文件为 318。

### 为什么清

- 2026-10-22 之后又有一批实验结案：V36 平移 / 归一化 / light 诊断全部为负结果并固化进 `CLEAN_BASELINE.md`；A01（机制 pilot）与 A02（注入位置）已结案并写入 `AIC_TASK_MAP.md`；V34 thr / 去重打包探针只服务当时工作台对比。用户授权"已结案或一次性使用的文件删除，拿不准的靠 tag 取回"。
- 本地磁盘 94% 满：`work_dir/` 49G、`vendor/DEIMv2/outputs/` 30G、`outputs/` 16G、`data_aux/` 22G 中大量是旧实验检查点、旧分析图与外部数据集。

### 删除了什么（Git 跟踪：134 个代码/配置 + 1 份文档）

| 目录 | 删除数 | 主要内容 |
|---|---:|---|
| `scripts/analysis` | 35 | V36 冻结权重的平移 / 归一化 / light / query 诊断与 V35 适配评测 |
| `scripts/diagnostics` | 47 | 平移干预 / 追踪 / 归一化 30 + A01/A02 消融与事故诊断 17 |
| `scripts/eval` | 3 | V34 零训练消融、V36 机制消融驱动、路由配额扫描 |
| `scripts/inference` | 3 | V34 thr0.10 / 0.30 与 V34+V35 NMS 打包探针 |
| `scripts/train` | 3 | A01/A02 消融队列与注入热启动启动器 |
| `scripts/viz` | 3 | V34/V35 对比面板、归一化 / 平移证据图 |
| `vendor/DEIMv2/configs/deimv2` | 36 | `v36_mechanism_*` 5 目录（25）+ `v36_injection_warmstart_20261019*` 3 目录（11） |
| `vendor/DEIMv2` 顶层 | 4 | `hf_models.ipynb`、`example.jpg`、`figures/*.png`（上游演示资产） |
| `docs/04_audits_and_ablations` | 1 | `AIC_IR融合攻坚整合报告.html`（生成式 HTML，本次唯一删除的文档） |

完整逐文件清单见 [`REPO_CLEANUP_20261031_removed.txt`](REPO_CLEANUP_20261031_removed.txt)。**未删除**：当前主线三臂 / v2 / 红外绑定 pilot、V23 / F5 / V27 / V28-A / V34–V37 的配置+启动+推理脚本、数据管线、全部 CPU 契约测试、工作台 6 个脚本、框架 `engine/` / `tools/` / `train.py` / 上游 COCO 配置；唯一保留的 `package_*` 是单模型内部去重的 `package_v34_distinct_dedup.py`。

### 本地被 ignore 产物的清理（约 87 GB，不进 tag、不可回滚）

| 类别 | 回收 | 内容 |
|---|---:|---|
| 旧实验审计 | ≈13 GB | `work_dir/audits/v36_injection_warmstart_20261019*`（12.9G；warmstart 权重可由保留的 `v36_frozen.pth` + `prepare_v36_injection_stages.py` 再生） |
| 旧实验检查点 | ≈27 GB | `vendor/DEIMv2/outputs/` 中 V1–V33 旧实验目录（保留 V28-A、V34、V36/V37 summary）；V30 与 `outputs/checkpoints` 的 md5 副本去重 |
| 外部数据集 | ≈18 GB | `work_dir/{M3FD_raw,FLIR_raw,LLVIP}`、`data_aux/BDD100K*`、`aic_nightify`（赛题禁外部数据、全库 0 引用；9 档夜化验证集 `aic_valid_rgb_night*` 保留） |
| 旧测试派生数据 | ≈7.6 GB | `work_dir/{test_aligned_ir,test_lbg_ir}`（AGENTS 明示旧配准器产物）、`work_dir/archives` 与 dataset 同源的两个 tar、旧缓存 pt |
| 过期中间产物 | ≈7 GB | `v2x_input_check_20261026`、`v22_phase3_20261026`、`v31_infer`、`rfdetr_dataset`、`dav2_token_cache` 等 2026-10 一次性审计 |
| 旧 outputs 分析 | ≈7.2 GB | `outputs/analysis/discrepancies_f5_vs_v23..v28`、UAV / ball 专项、可视化、V28-B（已判退化禁用）检查点、2 份与工作台首页 md5 相同的 HTML |
| 权重 / 杂物 | ≈2.5 GB | V21/V22 旧权重、重复 hgnetv2 权重、`NaN.pth`、`*_out/`、`__pycache__`、`work_dir` 误落产物 |

保留的关键本地产物：`dataset/`（红线）、`work_dir/{deimv2_dataset,train_lbg_ir,test_ir_like_train,test3_ir_like_train,test1_ir_like_train}`、`outputs/checkpoints/aic_clean_baseline_v1_seeta/best.pth`（当前主线唯一权重）、`outputs/checkpoints/v36_mechanism_20261018/v36_frozen.pth`、`outputs/official_submission_package/`、`work_dir/submissions/`（工作台 registry 引用）、`work_dir/audits/{aic_clean_baseline_v1_20261021,v36_mechanism_20261018}` 等证据目录、`weights/{F5,V23,RealESRGAN}`、`vendor/DEIMv2/ckpts/{dinov3,dfine_x_obj2coco}.pth`、9 档夜化验证集、`logs/`、`archive/`。执行明细与删除前快照：`archive/backups/cleanup_20261031/`（`local_delete_manifest.tsv`、`local_snapshot_{before,after}.txt`）；被删大目录里的轻量文本证据（manifest / json / README）另存 `local_evidence/`。

### 文档层

- 新增本条目与 `REPO_CLEANUP_20261031_removed.txt`；`AGENTS.md` §3/§6/§7/§8 与 `README.md` 更新目录数量、跟踪范围、tag、现状（Phase 3 51.1、pilot 未训练）、data_aux / 权重路径；`CLAUDE.md` §0 区分 macOS 镜像与 Linux 开发机、§5 提交框数；`PROJECT_SPEC.md` 修正 60 框 / 7,000–7,200 旧规、`test_aligned_ir`、`rfdetr_dataset` 示例、tag 取回方式与 scripts 分工表。
- 历史文档加"已撤回 / 历史快照"标注而不删：`SAFE_POSTPROCESSING_GUIDELINES.md`、`SUBMISSION_CONFIDENCE_THRESHOLD_PRIOR.md`、`SOVEREIGN_V23_DETAILED_TECHNICAL_SPEC.md` §十（WBF 禁止执行）、`AIC_TASK_MAP.md`、`AIC_MASTER_COLLABORATION_REGISTRY.md`、`docs/03_architecture_designs/README.md`；`SUBMISSION_CHECKLIST.md` / `TECHNICAL_REPORT_DRAFT.md` / `QUESTIONS_TO_ORGANIZERS.md` 修正过期指引。
- 本地的 `docs/archive/{training_records.md,training_records_mAP.md,llvip_dl.md}`（未跟踪、tag 不保护）已移到 `archive/backups/cleanup_20261031/docs_archive_local/`。

### 怎么取回

```bash
git fetch --tags                                    # 其他机器先取 tag
git show pre-cleanup-20261031:scripts/analysis/analyze_translation_phases.py
git checkout pre-cleanup-20261031 -- <path>         # 恢复到工作树
```

2026-10-22 及更早删除的文件用 `pre-cleanup-20261022`（= `4d71413`）。两个 tag 均已推送到 GitHub。

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
