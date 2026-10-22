# AGENTS.md — AIC2026 多模态目标检测项目

> 本文件是面向 AI 编码 Agent 的项目入口速查。**唯一文档中心是 `docs/`**，本文件只做导航与红线提示，不复制技术细节。
> **跑通一次实验先读**：`docs/02_collaboration_and_specs/PROJECT_SPEC.md`（工程规范 + 全链路流程 + 排错索引）。

---

## 1. 项目是什么

**AIC2026 城市场景视觉多模态目标检测竞赛**仓库。任务：从 **可见光 (RGB) + 红外 (IR) + 深度 (Depth)** 三模态图像中检测 **12 类**目标，产出官方格式提交文件。

- **12 类固定顺序**（class_id 0–11）：person, boat, animal, seat, sign, bicycle, car, ball, light, garbage can, uav, tricycle
- **单一主干框架**：**DEIMv2**（DINOv3-ViT / HGNetv2-X Backbone + HybridEncoder + D-FINE/RT-DETR 式解码器），vendored 在 `vendor/DEIMv2/`。RF-DETR 路线已废弃。
- **数据**：`dataset/train/` 2000 张（visible/infrared/infrared_aligned/depth/labels）；`dataset/test/AIC2026_PHASE_2_1000/` 1000 张**无标签**。
- **版本现状（2026-10-22）**：官方最高分仍是 **V23 TADL 53.555**（τ=0.45，7,459 框；F5 标杆 58.105）。V23 为**两段式**：Stage A 补标口径 72 轮预训练 → Stage B 旧标注口径 TADL 精调 **21 轮**（单卡约 2h），详见 `docs/MODEL_RECORDS.md` §11.5。此后 V24–V36 均已跑过：**V27 纯净无偏**（val mAP 0.8959 / AP50 0.9638）与 **V28-A 视差觉醒**（val AP75 0.9502）是历史 val 高点；**V28-B 72 轮长跑单调退化（0.888→0.778），`best_stg2` 落在最差轮次，禁止使用**；V29 循环蒸馏、V30 CSSP、V31 Co-ATSS、V32 PIG 有记录；V33 失败结案；**V36**（HGNetv2-B5 六通道 + P2-v10 + DINOv3 教师蒸馏 + Co-ATSS）在 ailab40 完成 50 轮纯 FP32 训练并产出 1000 图提交包。⚠️ V23/V27/V28 与 V34–V36 的验证集输入、标注和来源不同，**不能拼成统一排名**。
- **当前主线：纯净基准 v1**（`vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml`，修复 14 项通用正确性缺陷后的完全展开配置），2026-10-22 起在 ailab40 GPU1 跑 50 轮，**尚无成绩**。定义与修复清单见 `docs/02_collaboration_and_specs/CLEAN_BASELINE.md`，任务完成度见同目录 `AIC_TASK_MAP.md`，记录见 `docs/MODEL_RECORDS.md` §三十五。

## 2. 环境与运行约定

只认 **两个 conda 环境**（`rfdetr` / `torch25` 已不存在，勿引用）：

| 环境 | 路径 | 用途 |
|---|---|---|
| `deimv2` | `/home/constant/miniconda3/envs/deimv2/bin/python` | 训练、评测、推理、数据准备（几乎全部） |
| `aic-2` | `/home/constant/miniconda3/envs/aic-2/bin/python` | 少量辅助 / 夜化预处理 |

标准前缀（**缺了不会报错，只会静默降级**：`engine/data/transforms/_transforms.py` 对 `nightify_aic` 的 import 包了 try/except，找不到时 `nightify=None`，夜化增强变成 no-op；非 strict 数据集的 IR destripe 也会静默回退。只有 `strict_inputs: true` 的 config（如纯净基准）会直接报错。所以"能跑但结果不对"时先查这一项）：

```bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- **训练入口** `vendor/DEIMv2/train.py` —— **必须 `cd vendor/DEIMv2` 执行**，`-c` 配置路径与 `output_dir` 都相对该目录（`configs/deimv2/<exp>.yml`）。
- **推理/分析脚本** —— 从仓库根目录驱动；脚本顶部普遍硬编码 `ROOT = "/home/constant/CodeBuddy/Dev/AIC"`，且 `scripts/eval/predict_*.py` 会 `os.chdir(vendor/DEIMv2)`（chdir 后相对路径含义变化，`predict_mm_tta.py` 会把 `--out` resolve 回 ROOT）。
- **本地 GPU 仅 RTX 5060 Ti 15.46 GiB**（开发调试）。远端训练节点：
  - **ailab40（当前主力，纯净基准 v1 / V36 均在此训练）**：实验室服务器，Tesla V100 32GB（GPU 0、1；纯净基准用 GPU 1）。项目根 `/home/ailab40/hejianzhong/AIC_CLEAN_BASELINE_20261021`，python 为 `/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python`（**不是** `deimv2` 环境）；启动脚本 `scripts/train/run_clean_baseline_v1_ailab40.sh`。V100（SM_70）上**必须纯 FP32、禁用 cuDNN、不加 `--use-amp`**——V36 早期 AMP 运行因 FP16 溢出有 92.9% 的步被 NaN-guard 跳过。
  - 以下两个 seetacloud 节点路径根均为 `/root/autodl-tmp/AIC`：
  - **8 卡节点（V28 时期主力）**：`ssh -p 32426 root@connect.westb.seetacloud.com`（`~/.ssh/config` 别名 `seeta8`），8 × RTX PRO 6000 Blackwell。
  - **96GB 单卡节点（V23 主训练）**：`ssh -p 33715 root@connect.westd.seetacloud.com`（别名 `seeta`），RTX 6000 Ada / RTX PRO 6000。
  - ⚠️ 容器可能未挂载 GPU（`v28_infer.log` 曾报 `Found no NVIDIA driver`），跑推理前先 `nvidia-smi -L` 确认。

## 3. 目录职责

```
docs/            唯一文档中心（01_competition / 02_collaboration_and_specs / 03_architecture_designs
                 / 04_audits_and_ablations / 05_postprocessing_and_inference / archive）
  └ 02_collaboration_and_specs/PROJECT_SPEC.md  ★ 工程规范 + 全链路流程 + 数据契约 + 排错索引
vendor/DEIMv2/   核心框架：train.py、engine/、configs/、outputs/（训练检查点）
vendor/Depth-Anything-V2/  深度补全代码与权重
vendor/cv-model-improvement-master/  外部资源合集（仅参考，不参与流程）
scripts/         prep(42) / train(166) / eval(67) / inference(18) / gen(49) / analysis(185)
                 / viz(33) / lib(14) / audit / diagnostics / utils / collect_model_records.py
dataset/         原始数据（只读软链，git 忽略）
work_dir/        运行时产物（已分层）：deimv2_dataset 训练数据 · test_aligned_ir/test_lbg_ir 测试缓存
                 · submissions/ 提交包 · audits/ 审计 · visualizations/ 可视化 · archives/ 归档 · 外部集(LLVIP/M3FD_raw/FLIR_raw)
data_aux/        外部辅助数据（BDD100K、夜化、black 变体，git 忽略）
outputs/         分析产物（已分层）：analysis/ · figures/ · metrics/ · reports/ · submissions/ · checkpoints/
                 · official_submission_package/（官方提交包，勿动）
weight/ weights/ 权重中心（根目录保留软链兼容旧配置）
logs/            唯一日志落点
work/            红外配准攻坚探针库（详见 work/README.md，日常流程勿用）
archive/         历史草稿与废弃脚本，含 backups/（整理前快照）、snapshots/、dedup_docs/、scratch/
AGENTS.md        本文件（Agent 入口速查）
README.md        项目门面（竞赛简介、旗舰与战绩徽章）
CODEBUDDY.md -> AGENTS.md                        兼容旧入口（内容已并入 AGENTS.md）
MODEL_RECORDS.md -> docs/MODEL_RECORDS.md           ★ 唯一实验真相源
SOVEREIGN_V23_CORE_SPEC.md / SOVEREIGN_V20_DETAILED_TECHNICAL_SPEC.md -> docs/…   旗舰规范快捷方式
```

## 4. 红线（务必遵守）

1. **根目录整洁**：禁止在根目录新增临时脚本、日志、权重、图表。临时代码 → `archive/scratch/`，日志 → `logs/<exp>.log`，图表 → `outputs/`（报告 → `outputs/reports/`）。
2. **深度图 16-bit 陷阱**：depth 是 16-bit 毫米值，必须按 `raw16` 约定读取（mode `I;16`/`I` → `/19999`，否则 `/255`）。**绝不能用 `PIL convert("L")`** —— 会把毫米值截断成二值掩码（历史 bug，见 commit `53c6577`，并已校正 `depth_invert` 方向）。
3. **6 通道布局必须与训练一致**：`ch0-2=RGB`、`ch3=对齐 IR`、`ch4=LBG IR`、`ch5=depth`；归一化 mean/std 见 `PROJECT_SPEC.md` §4.1。`n_channels` 必须与 config 的 `ir_in_chans`/`depth_in_chans` 对齐；通道装配由各实验 config 的 `dataset` 块显式指定，**不是数据集默认行为**。
4. **提交格式**：逐图一个 `.txt`，`<class_id> <cx> <cy> <w> <h> <confidence>`，空格分隔，坐标归一化 [0,1]，**CRLF**，**每图 ≤100 框**，按 confidence 降序，无检测写空文件。**提交前必跑** `scripts/eval/check_submission.py`。
5. **实验记录只进 `MODEL_RECORDS.md`**，不要另建分散记录；跑完新实验重跑 `python3 scripts/collect_model_records.py`。
6. **`dataset/test` 无标签**，只能做格式校验，不能算 mAP；不要读取/推断测试集标注。
7. **不要删 `dataset/`，不要破坏 `work_dir/*_dataset` 软链**；不要动竞赛合规红线（禁外部数据 / 禁集成，见 `docs/01_competition/`）。
8. **提交合规**：赛题第十一条**禁止不同结构或不同训练阶段模型的简单集成（投票/平均/WBF）** —— `scripts/archive_legacy_postprocess/` 下的 `wbf_fuse.py`、外部 CAA-SAHI 切片、外部多阈值投票**均已废弃且不得用于提交**。只允许**单模型**的模型内生化后处理（CAA-SAHI、内生化 Flip-TTA）与同模型多尺度推断。
9. **检查点为混合键**：兼容 EMA（`ck["ema"]["module"]`）与 `ck["model"]`，一般 `strict=False`；Combo3 权重宣称 strict=True 通过。
10. **文档是软链体系**：`docs/` 下的扁平文件与根目录快捷方式多为**符号链接**，唯一正本在分类子目录（或 `docs/` 顶层规范）。改动前先 `ls -la` 确认，**不要用 `cp` 覆盖软链**（会退化成两份正本）。
11. **产出提交包自动导入可视化工作台（强制）**：凡产出测试集预测或提交包（`.zip`），生成脚本末尾必须自动调用 `scripts.viz.import_submission_to_workstation` 中的 `auto_import_submission()`，将模型注册入工作台列表并自动触发编译、确保守护进程运行。用户可随时在 `http://localhost:8088/` 实时核验全量 1000 张图检测、超分对比与全模型差异矩阵。

## 5. 常用命令

```bash
PY=/home/constant/miniconda3/envs/deimv2/bin/python

# 提交包自动导入网页工作台 / 服务启动
$PY scripts/viz/import_submission_to_workstation.py --zip work_dir/submissions/<exp>/submission.zip --id <id> --name "<name>"
$PY scripts/viz/workstation_server.py --port 8088  # 访问 http://localhost:8088/

# 训练（cd vendor/DEIMv2；-t 热启动权重）
cd vendor/DEIMv2 && $PY train.py -c configs/deimv2/<exp>.yml -t outputs/<src>/best_stg2.pth --use-amp
# 评测（--test-only + -r 指向待评权重）
cd vendor/DEIMv2 && $PY train.py -c configs/deimv2/<exp>_val_<level>.yml -r "$CKPT" --use-amp --test-only

# 6 通道离线缓存（避免每 epoch 重复读 4 路图；须 cd vendor/DEIMv2）
cd vendor/DEIMv2 && $PY prep_deepir_cache.py

# 推理 + 同模型多尺度 TTA（通用）
$PY scripts/eval/predict_mm_tta.py --config CFG --ckpt CKPT \
    --test-root dataset/test/AIC2026_PHASE_2_1000 --out work_dir/submission_x --scales 640,800 --thresh 0.3

# 逐代原生推理（当前主力，直接产提交包）
$PY scripts/inference/infer_test2_v28_disparity_awakening.py --config CFG --checkpoint CKPT

# 提交格式校验 / mAP 评测 / 记录重建
$PY scripts/eval/check_submission.py <SUB_DIR> dataset/test/AIC2026_PHASE_2_1000/visible
$PY scripts/eval/eval_submission.py <SUB_DIR> <ANN_JSON>
$PY scripts/collect_model_records.py
```

## 6. 关键机制与坑点

- **Config 体系**：`configs/{base,dataset,deimv2}/`；实验 config 用 `__include__` 深合并，**后 include 者覆盖前者**（`engine/core/yaml_utils.py`）。`dataset/` 下 `aic_detection.yml`(RGB 基线)、`aic_heat.yml`(heat)、`aic_detection*_nightXX.yml`(9 档暗光 night50→night0005)；6 通道数据块写在**各实验 config 内**。
- **多模态 backbone** 由 config 的 `DEIM.backbone:` 字段选择（**不是 `fusion:`**），另配同名参数块（如 `HGNetv2MultiModal:`）。**主方案 `HGNetv2MultiModal` 在 `vendor/DEIMv2/engine/backbone/hgnetv2_mm.py`**（DCDA 视差对齐 + 3D-Lift + DOG 深度阶跃门控 + 双边池化 + 浅层边缘融合）；`multimodal_backbones.py`（约 2400 行）放的是下一条列出的 `DINOv3STAs*` 历史变体与 CSSA/BiDir/RDT 等融合件。新 backbone 须在 `engine/backbone/__init__.py` 中 import 才会注册。
- **可选 backbone 变体**（历史/消融）：`DINOv3STAs`(纯 RGB 基线) · `…CSD`(通道互补+CBAM) · `…BiDir`(RGB↔IR 双向交叉注意力) · `…DiffFuse`(模态主导度加权) · `…TriModal`(RGB+heat 共 stem) · `…PureIR`(单红外) · **`…RDTTrack`**(正交去冗余 + Fovea/PromptBlock 注入冻结 RGB，前期主方案) · `…RDT`/`…RDTDiffuse`(RGB-D 对照/差异融合注入) · `…BiDirPaper`(Tang'24 复现)。
- **deepir** = `work_dir/deepir_cache/train/*.pt` 6 通道离线缓存 + `CocoDetectionMultiModalCached`；**midboost** = 暗光亮度采样聚焦中段（`level: [0.005,0.45]` + log-uniform 过采样）的微调策略。
- **训练用 COCO 划分**：`work_dir/deimv2_dataset/{train 1800, valid 200}`（seed=42）。标注两个口径：`instances_train.json`（官方老标注，V23 起主线）与 `instances_train_quality1998.json`（质量补标，**会产生官方判定 FP，已弃用**）。
- **`interaction_indexes` 三流共用**：RGB 与 aux 的 ViT 层数不等（ViT-L 24 层 + ViT-S 12 层）会越界断言，已由 `_scale_interaction_indexes()` 按 depth 比例缩放；新增 backbone 组合时留意。
- **桌面合成器抢显存**：Wayland/niri 曾独占 ~12 GiB，导致"只有 13GB 可用"的误解（卡本身 15.46 GiB）。`accum_steps` 就是为此加的。
- **`.gitignore`** 忽略 `*.pth/*.pt/*.log/*.pkl/*.out`、`/dataset`（只锚定仓库根目录的原始数据，目录或软链都匹配）、`/data_aux/`、`**/work_dir/`、`archive/`、`vendor/Depth-Anything-V2/`。**不要改回不带 `/` 前缀的 `dataset/`**：它会连带忽略 `vendor/DEIMv2/` 下 `engine/data/dataset/`、`configs/dataset/`、`tools/dataset/` 三个代码目录，新建文件被静默忽略（`95b2090` 修过一次，`bc383e2` 又改回，2026-10-22 恢复）。2026-10-22 修正时发现有 21 个 `configs/dataset/aic_detection_{night*,dndt_*,cp*,nightify}.yml` 因此从未入库（被 120 个已入库实验 config `__include__`），已于同日补交入库；补交后所有已入库 config 的 `__include__` 目标均在库。
- **git 跟踪范围**：`docs/` 与 `scripts/` 已入库（约 1,770 个跟踪文件：`vendor/DEIMv2` ~960、`scripts/` ~750、`docs/` ~50），`docs/` 的改动可用 git 追溯与回滚。`dataset/`、`work_dir/`、`outputs/`、`logs/`、`archive/` 与权重仍被 gitignore，对它们的破坏性操作前仍要先备份到 `archive/backups/`。
- **历史遗留混淆项**：`scripts/` 下大量 `predict_*`/`tta_*` 老脚本依赖已废弃的 `rfdetr` 环境，当前不可运行；日常请用 `predict_mm_tta.py` 与 `scripts/inference/`。

## 7. 我该看哪篇文档

| 任务 | 文档 |
|---|---|
| **工程规范 + 全链路流程 + 数据契约 + 排错** | `docs/02_collaboration_and_specs/PROJECT_SPEC.md` ★ |
| 文档总索引 / 项目门面 | `README.md`（根目录） |
| 实验成绩单一真相源 | `docs/MODEL_RECORDS.md`（表A 训练模型总览，表B 暗光评测，第十一~十三节 V20~V28 演进总账） |
| 最新旗舰架构 | `docs/SOVEREIGN_V23_CORE_SPEC.md` + `docs/SOVEREIGN_V23_DETAILED_TECHNICAL_SPEC.md` |
| 架构演进九阶段 | `docs/03_architecture_designs/`（01→09 即阅读顺序，README 有总表） |
| 全周期演进史 / 天梯榜 | `docs/02_collaboration_and_specs/AIC_FULL_CONVERSATION_CHRONICLE.md` |
| 数据预处理 / 对齐 / 深度优化中台 | `docs/02_collaboration_and_specs/AIC_MASTER_COLLABORATION_REGISTRY.md` |
| 提交前把关 / 后处理准则 | `docs/05_postprocessing_and_inference/SAFE_POSTPROCESSING_GUIDELINES.md` |
| 测试集先验（目标数 7,086 / 密度 7.09） | `docs/01_competition/SUBMISSION_CONFIDENCE_THRESHOLD_PRIOR.md` |
| 半决赛技术报告骨架与提交清单 | `docs/06_semifinal_submission/`（`TECHNICAL_REPORT_DRAFT.md`、`SUBMISSION_CHECKLIST.md`） |
| 历史审计（对齐/权重溯源/CP 消融） | `docs/04_audits_and_ablations/` |

## 8. 整理记录（2026-10-07）

仓库做过一次结构整理，**流程与数据零改动，仅位置归位**。完整日志：`logs/cleanup_20261007.log`；回滚快照：`archive/backups/`。

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

原则：**每类信息只留一处正本**，导航层收敛为「根 `README.md` 索引 + 本文件速查 + `PROJECT_SPEC.md` 规范」三层。
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
