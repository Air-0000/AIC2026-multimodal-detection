# AGENTS.md — AIC2026 多模态目标检测项目

> 本文件是面向 AI 编码 Agent 的项目入口速查。**唯一文档中心是 `docs/`**，本文件只做导航与红线提示，不复制技术细节。
> **跑通一次实验先读**：`docs/02_collaboration_and_specs/PROJECT_SPEC.md`（工程规范 + 全链路流程 + 排错索引）。

---

## 1. 项目是什么

**AIC2026 城市场景视觉多模态目标检测竞赛**仓库。任务：从 **可见光 (RGB) + 红外 (IR) + 深度 (Depth)** 三模态图像中检测 **12 类**目标，产出官方格式提交文件。

- **12 类固定顺序**（class_id 0–11）：person, boat, animal, seat, sign, bicycle, car, ball, light, garbage can, uav, tricycle
- **单一主干框架**：**DEIMv2**（DINOv3-ViT / HGNetv2-X Backbone + HybridEncoder + D-FINE/RT-DETR 式解码器），vendored 在 `vendor/DEIMv2/`。RF-DETR 路线已废弃。
- **数据**：`dataset/train/` 2000 张（visible/infrared/infrared_aligned/depth/labels）；`dataset/test/AIC2026_PHASE_2_1000/` 复赛 1000 张**无标签**；`dataset/test/AIC2026_PHASE_3_1000/` 决赛 1000 张**无标签**（Phase 3 与 Phase 2 不可比）。
- **版本现状（2026-10-31）**：官方最高分仍是 **V23 TADL 53.555**（τ=0.45，7,459 框；F5 标杆 58.105）。V23 为**两段式**：Stage A 补标口径 72 轮预训练 → Stage B 旧标注口径 TADL 精调 **21 轮**（单卡约 2h），详见 `docs/MODEL_RECORDS.md` §11.5。此后 V24–V36 均已跑过：**V27 纯净无偏**（val mAP 0.8959 / AP50 0.9638）与 **V28-A 视差觉醒**（val AP75 0.9502）是历史 val 高点；**V28-B 72 轮长跑单调退化（0.888→0.778），`best_stg2` 落在最差轮次，禁止使用**；V29 循环蒸馏、V30 CSSP、V31 Co-ATSS、V32 PIG 有记录；V33 失败结案；**V36**（HGNetv2-B5 六通道 + P2-v10 + DINOv3 教师蒸馏 + Co-ATSS）在 ailab40 完成 50 轮纯 FP32 训练并产出 1000 图提交包。⚠️ V23/V27/V28 与 V34–V36 的验证集输入、标注和来源不同，**不能拼成统一排名**。
- **当前主线：纯净基准 v1**（`vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml`，修复 14 项通用正确性缺陷后的完全展开配置），2026-10-22 起在 ailab40 GPU1 跑 50 轮，10-23 在第 6 轮因该卡显存硬件错误中断（`docs/MODEL_RECORDS.md` §四十）；同日按 `docs/02_collaboration_and_specs/CLOUD_MIGRATION_RUNBOOK.md` 迁到 seeta RTX PRO 6000 Blackwell 从头重训（BF16 autocast + TF32，§四十一–四十二），10-24 跑完第 46 轮（epoch 45）后实例被外部停止，没有 `final.pth`，已决定不再续跑；验证集最佳为 epoch 37 的 mAP 55.94（单次运行、在同一 200 张验证集上选点，口径见 §四十四、§四十五）。定义与修复清单见 `docs/02_collaboration_and_specs/CLEAN_BASELINE.md`，任务完成度见同目录 `AIC_TASK_MAP.md`，记录见 `docs/MODEL_RECORDS.md` §三十五。**10-26 决赛测试集（Phase 3）首个官方成绩 51.1**（§四十六；Phase 3 ≠ Phase 2，不能与 53.555 / 58.105 比较）；同日完成 v1 模态鲁棒性剖面与**红外绑定 pilot 配置** `aic_irbind_pilot_ws_v1.yml`（仅配置与热启动契约，未训练，§四十七）。

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
- **推理/分析脚本** —— 从仓库根目录驱动；较早的脚本顶部硬编码 `ROOT = "/home/constant/CodeBuddy/Dev/AIC"`，较新的按 `Path(__file__).parents[2]` 定位仓库根。多数推理 / 评测 / 诊断脚本加载配置时会 `os.chdir(vendor/DEIMv2)`，chdir 后相对路径含义变化，传路径参数时用绝对路径或按脚本说明。
- **本地 GPU 仅 RTX 5060 Ti 15.46 GiB**（开发调试）。远端训练节点：
  - **ailab40（V36 与中断的纯净基准 v1 在此训练）**：⚠️ 两张卡 ECC 均关闭，GPU1 显存静默出错（自检 88/720 次读校验不一致），不要再用它训练；任何新卡开训前先跑 `scripts/train/preflight_cloud.sh`。实验室服务器，Tesla V100 32GB（GPU 0、1；纯净基准用 GPU 1）。项目根 `/home/ailab40/hejianzhong/AIC_CLEAN_BASELINE_20261021`，python 为 `/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python`（**不是** `deimv2` 环境）；启动脚本 `scripts/train/run_clean_baseline_v1_ailab40.sh`。V100（SM_70）上**必须纯 FP32、禁用 cuDNN**——V36 早期 AMP 运行因 FP16 溢出有 92.9% 的步被 NaN-guard 跳过。纯净基准三臂配置现为 `use_amp: true`，在 V100 上会被 `train.py` 拒绝，须加 `-u use_amp=False`（`run_clean_arm.sh` 用 `USE_AMP=0`；两个 ailab40 启动脚本已带上该覆盖；精度规则见 CLAUDE.md §3）。
  - 以下两个 seetacloud 节点路径根均为 `/root/autodl-tmp/AIC`：
  - **8 卡节点（V28 时期主力）**：`ssh -p 32426 root@connect.westb.seetacloud.com`（`~/.ssh/config` 别名 `seeta8`），8 × RTX PRO 6000 Blackwell。
  - **96GB 单卡节点（V23 主训练）**：`ssh -p 33715 root@connect.westd.seetacloud.com`（别名 `seeta`），RTX 6000 Ada / RTX PRO 6000。
  - ⚠️ 容器可能未挂载 GPU（`v28_infer.log` 曾报 `Found no NVIDIA driver`），跑推理前先 `nvidia-smi -L` 确认。

## 3. 目录职责

```
docs/            唯一文档中心：顶层只有 MODEL_RECORDS.md，其余按 01_competition / 02_collaboration_and_specs
                 / 03_architecture_designs / 04_audits_and_ablations / 05_postprocessing_and_inference
                 / 06_semifinal_submission 分类（docs/archive/ 只在 Linux 本地，不入库）
  └ 02_collaboration_and_specs/PROJECT_SPEC.md  ★ 工程规范 + 全链路流程 + 数据契约 + 排错索引
vendor/DEIMv2/   核心框架：train.py、engine/、configs/、outputs/（训练检查点）
                 configs/deimv2/ 只留纯净基准三臂、里程碑（V23 / F5 / V27 / V28-A / V34–V37）与上游 COCO 配置
vendor/Depth-Anything-V2/  深度补全代码与权重
vendor/cv-model-improvement-master/  外部资源合集（仅参考，不参与流程）
scripts/         prep(10) / train(19) / eval(7) / inference(10) / analysis(4) / viz(6) / lib(4)
                 / diagnostics(14) / collect_model_records.py；历史实验脚本已于 2026-10-22 与 2026-10-31 两轮删除（见 §8）
dataset/         原始数据（只读软链，git 忽略）
work_dir/        运行时产物（已分层）：deimv2_dataset 训练数据 · test_ir_like_train/ 与 test3_ir_like_train/ 测试红外
                 （按训练同法，§三十九；旧 test_aligned_ir/test_lbg_ir 已于 2026-10-31 删除）
                 · submissions/ 提交包 · audits/ 审计 · visualizations/ 可视化 · archives/ 归档 · transfer/ 迁移清单
data_aux/        外部辅助数据（现仅保留 9 档夜化验证集 aic_valid_rgb_night{50…0005}，供模态鲁棒性评测；
                 BDD100K/FLIR/LLVIP、aic_nightify 与 black 变体已于 2026-10-31 清理；git 忽略）
outputs/         分析产物（已分层）：analysis/ · figures/ · metrics/ · reports/ · submissions/ · checkpoints/
                 · official_submission_package/（官方提交包，勿动）
weight/ weights/ 权重中心（根目录保留软链兼容旧配置）
logs/            唯一日志落点
work/            红外配准攻坚探针库（详见 work/README.md，日常流程勿用）
archive/         历史草稿与废弃脚本，含 backups/（整理前快照）、snapshots/、dedup_docs/、scratch/
AGENTS.md        本文件（Agent 入口速查）
README.md        项目门面（竞赛简介、旗舰与战绩徽章）
CODEBUDDY.md -> AGENTS.md                        兼容旧入口（内容已并入 AGENTS.md）
MODEL_RECORDS.md -> docs/MODEL_RECORDS.md           ★ 唯一实验真相源（仓库里仅剩这两个软链）
```

## 4. 红线（务必遵守）

1. **根目录整洁**：禁止在根目录新增临时脚本、日志、权重、图表。临时代码 → `archive/scratch/`，日志 → `logs/<exp>.log`，图表 → `outputs/`（报告 → `outputs/reports/`）。
2. **深度图 16-bit 陷阱**：depth 是 16-bit 毫米值，必须按 `raw16` 约定读取（mode `I;16`/`I` → `/19999`，否则 `/255`）。**绝不能用 `PIL convert("L")`** —— 会把毫米值截断成二值掩码（历史 bug，见 commit `53c6577`，并已校正 `depth_invert` 方向）。
3. **6 通道布局必须与训练一致**：`ch0-2=RGB`、`ch3=对齐 IR`、`ch4=LBG IR`、`ch5=depth`；归一化 mean/std 见 `PROJECT_SPEC.md` §4.1。`n_channels` 必须与 config 的 `ir_in_chans`/`depth_in_chans` 对齐；通道装配由各实验 config 的 `dataset` 块显式指定，**不是数据集默认行为**。
4. **提交格式**：逐图一个 `.txt`，`<class_id> <cx> <cy> <w> <h> <confidence>`，空格分隔，坐标归一化 [0,1]，**CRLF**，**每图 ≤100 框**，按 confidence 降序，无检测写空文件。**提交前必跑** `scripts/eval/check_submission.py`。
5. **实验记录只进 `MODEL_RECORDS.md`**，不要另建分散记录；跑完新实验重跑 `python3 scripts/collect_model_records.py`。
6. **`dataset/test` 无标签**，只能做格式校验，不能算 mAP；不要读取/推断测试集标注。
7. **不要删 `dataset/`，不要破坏 `work_dir/*_dataset` 软链**；不要动竞赛合规红线（禁外部数据 / 禁集成，见 `docs/01_competition/`）。
8. **提交合规**：赛题第十一条**禁止不同结构或不同训练阶段模型的简单集成（投票/平均/WBF）** —— 多模型 `wbf_fuse.py`、外部 CAA-SAHI 切片、外部多阈值投票（原 `scripts/archive_legacy_postprocess/`）已于 2026-10-22 从仓库删除，**不得恢复用于提交**。只允许**单模型**的模型内生化后处理（CAA-SAHI、内生化 Flip-TTA）与同模型多尺度推断。
9. **检查点为混合键**：兼容 EMA（`ck["ema"]["module"]`）与 `ck["model"]`，一般 `strict=False`；Combo3 权重宣称 strict=True 通过。
10. **每份文档只有一处正本**：`docs/` 下已无软链；仓库里只剩根目录 `MODEL_RECORDS.md`、`CODEBUDDY.md` 两个软链。**不要用 `cp` 覆盖这两个软链**（会退化成两份正本），也不要再新建快捷软链。
11. **产出提交包自动导入可视化工作台（强制）**：凡产出测试集预测或提交包（`.zip`），生成脚本末尾必须自动调用 `scripts.viz.import_submission_to_workstation` 中的 `auto_import_submission()`，将模型注册入工作台列表并自动触发编译、确保守护进程运行。用户可随时在 `http://localhost:8088/` 实时核验全量 1000 张图检测、超分对比与全模型差异矩阵。

## 5. 常用命令

```bash
PY=/home/constant/miniconda3/envs/deimv2/bin/python

# 提交包自动导入网页工作台 / 服务启动
$PY scripts/viz/import_submission_to_workstation.py --zip work_dir/submissions/<exp>/submission.zip --id <id> --name "<name>"
$PY scripts/viz/workstation_server.py --port 8088  # 访问 http://localhost:8088/

# 训练（cd vendor/DEIMv2；新实验用全新 output_dir；CLI 语义与精度规则见 CLAUDE.md §3）
# SM >= 8.0 的卡按配置走 BF16 autocast + TF32；V100 必须追加 -u use_amp=False（纯 FP32），否则 train.py 报错
cd vendor/DEIMv2 && $PY train.py -c configs/deimv2/aic_clean_baseline_v1.yml
# 换卡训练（自检 → 打包输入 → 上线前检查 → 启动），见 docs/02_collaboration_and_specs/CLOUD_MIGRATION_RUNBOOK.md
ROOT=$PWD GPUS="0 1" REQUIRE_ECC=1 BENCH=1 bash scripts/train/preflight_cloud.sh

# 统一推理入口（训练同款加载器、严格模式、单次前向；--zip 时自动校验格式并导入工作台）
$PY scripts/inference/predict.py --config vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml \
    --checkpoint CKPT --images dataset/test/AIC2026_PHASE_2_1000/visible \
    --rgb-dir dataset/test/AIC2026_PHASE_2_1000/visible --ir-dir work_dir/test_ir_like_train/infrared_aligned \
    --lbg-dir work_dir/test_ir_like_train/lbg_ir --depth-dir dataset/test/AIC2026_PHASE_2_1000/depth \
    --score-thr 0.45 --out-dir work_dir/submissions/x/txt --zip work_dir/submissions/x/submission.zip

# 验证集模态鲁棒性剖面（clean / night* / rgb_black / ir_off / depth_off）与配对比较
$PY scripts/eval/eval_modality_robustness.py --config CFG --checkpoint CKPT --out work_dir/robustness_x

# 提交格式校验 / mAP 评测 / 记录重建
$PY scripts/eval/check_submission.py <SUB_DIR> dataset/test/AIC2026_PHASE_2_1000/visible
$PY scripts/eval/eval_submission.py <SUB_DIR> <ANN_JSON>
$PY scripts/collect_model_records.py
```

## 6. 关键机制与坑点

- **Config 体系**：`configs/{base,dataset,deimv2}/`；实验 config 用 `__include__` 深合并，**后 include 者覆盖前者**（`engine/core/yaml_utils.py`）。`dataset/` 下只剩 `aic_detection.yml`（RGB，V23 / F5 用）与上游数据集配置；6 通道数据块写在**各实验 config 内**。纯净基准刻意不用 include。9 档暗光验证集（`data_aux/aic_valid_rgb_night{50…0005}`）由 `scripts/prep/prep_nightxx_server.py` 生成，`eval_modality_robustness.py` 直接读图，不再需要逐档的 val 配置。
- **多模态 backbone** 由 config 的 `DEIM.backbone:` 字段选择（**不是 `fusion:`**），另配同名参数块（如 `HGNetv2MultiModal:`）。**主方案 `HGNetv2MultiModal` 在 `vendor/DEIMv2/engine/backbone/hgnetv2_mm.py`**（DCDA 视差对齐 + 3D-Lift + DOG 深度阶跃门控 + 双边池化 + 浅层边缘融合）；`multimodal_backbones.py`（约 2400 行）放的是下一条列出的 `DINOv3STAs*` 历史变体与 CSSA/BiDir/RDT 等融合件。新 backbone 须在 `engine/backbone/__init__.py` 中 import 才会注册。
- **可选 backbone 变体**（历史/消融）：`DINOv3STAs`(纯 RGB 基线) · `…CSD`(通道互补+CBAM) · `…BiDir`(RGB↔IR 双向交叉注意力) · `…DiffFuse`(模态主导度加权) · `…TriModal`(RGB+heat 共 stem) · `…PureIR`(单红外) · **`…RDTTrack`**(正交去冗余 + Fovea/PromptBlock 注入冻结 RGB，前期主方案) · `…RDT`/`…RDTDiffuse`(RGB-D 对照/差异融合注入) · `…BiDirPaper`(Tang'24 复现)。
- **历史术语**：**deepir** = `work_dir/deepir_cache/train/*.pt` 6 通道离线缓存 + `CocoDetectionMultiModalCached`（缓存构建脚本已删，数据集类仍在 `engine/`，现存配置都不用它）；**midboost** = 暗光亮度采样聚焦中段（`level: [0.005,0.45]` + log-uniform 过采样）的微调策略。
- **训练用 COCO 划分**：`work_dir/deimv2_dataset/{train 1800, valid 200}`（seed=42）。标注两个口径：`instances_train.json`（官方老标注，V23 起主线）与 `instances_train_quality1998.json`（质量补标，**会产生官方判定 FP，已弃用**）。
- **`interaction_indexes` 三流共用**：RGB 与 aux 的 ViT 层数不等（ViT-L 24 层 + ViT-S 12 层）会越界断言，已由 `_scale_interaction_indexes()` 按 depth 比例缩放；新增 backbone 组合时留意。
- **桌面合成器抢显存**：Wayland/niri 曾独占 ~12 GiB，导致"只有 13GB 可用"的误解（卡本身 15.46 GiB）。`accum_steps` 就是为此加的。
- **`.gitignore`** 忽略 `*.pth/*.pt/*.log/*.pkl/*.out`、`/dataset`（只锚定仓库根目录的原始数据，目录或软链都匹配）、`/data_aux/`、`**/work_dir/`、`archive/`、`vendor/Depth-Anything-V2/`。**不要改回不带 `/` 前缀的 `dataset/`**：它会连带忽略 `vendor/DEIMv2/` 下 `engine/data/dataset/`、`configs/dataset/`、`tools/dataset/` 三个代码目录，新建文件被静默忽略（`95b2090` 修过一次，`bc383e2` 又改回，2026-10-22 恢复）。当时补交的 21 个 `configs/dataset/aic_detection_{night*,dndt_*,cp*,nightify}.yml` 只被旧实验 config include，已随它们在同日整理中删除；现存所有 config 的 `__include__` 目标均在库。
- **git 跟踪范围**：约 318 个跟踪文件（`vendor/DEIMv2` ~196、`scripts/` 75、`docs/` 41），可用 git 追溯与回滚；两轮清理前的完整历史分别在 tag `pre-cleanup-20261022`（= `4d71413`，已推 GitHub）与 `pre-cleanup-20261031`（= `2c170ef`，已推 GitHub）。`dataset/`、`work_dir/`、`outputs/`、`logs/`、`archive/` 与权重仍被 gitignore，对它们的破坏性操作前仍要先备份到 `archive/backups/`；2026-10-31 已清理约 87 GB 过期本地实验产物（见 REPO_CLEANUP_LOG）。
- **旧脚本去哪了**：依赖 `rfdetr` 环境的 `predict_*` / `tta_*`、V1–V33 的启动 / 分析 / 可视化脚本，以及 2026-10-31 删除的 V36 平移/机制诊断与已结案消融脚本，清单见 `docs/02_collaboration_and_specs/REPO_CLEANUP_20261022_removed.txt` 与同目录 `REPO_CLEANUP_20261031_removed.txt`，按 tag 取回。推理用 `scripts/inference/predict.py`；里程碑版本仍用各自的 `scripts/inference/infer_test2_v*.py`。

## 7. 我该看哪篇文档

| 任务 | 文档 |
|---|---|
| **工程规范 + 全链路流程 + 数据契约 + 排错** | `docs/02_collaboration_and_specs/PROJECT_SPEC.md` ★ |
| 文档总索引 / 项目门面 | `README.md`（根目录） |
| 实验成绩单一真相源 | `docs/MODEL_RECORDS.md`（表A 训练模型总览，表B 暗光评测，V20~V37 + 纯净基准 v1 演进总账） |
| V23 旗舰架构（官方最高分） | `docs/03_architecture_designs/SOVEREIGN_V23_CORE_SPEC.md` + `SOVEREIGN_V23_DETAILED_TECHNICAL_SPEC.md`（同目录；§十 WBF 计划已废弃，禁止执行） |
| 架构演进九阶段 | `docs/03_architecture_designs/`（01→08 按编号阅读，第 9 部为 `SOVEREIGN_V23_*`，README 有总表） |
| 当前主线定义 / 任务完成度 | `docs/02_collaboration_and_specs/CLEAN_BASELINE.md` · `AIC_TASK_MAP.md`（后者为 2026-10-21 历史快照） |
| 全周期技术综述（历史，与架构白皮书互补） | `docs/02_collaboration_and_specs/AIC_RAW_DATA_PROCESSING_AND_COMPLETE_MODEL_IMPROVEMENTS_MASTER.md` |
| 全周期演进史 / 天梯榜 | `docs/02_collaboration_and_specs/AIC_FULL_CONVERSATION_CHRONICLE.md` |
| 数据预处理 / 对齐 / 深度优化中台 | `docs/02_collaboration_and_specs/AIC_MASTER_COLLABORATION_REGISTRY.md` |
| 提交前把关 / 后处理准则 | `docs/05_postprocessing_and_inference/SAFE_POSTPROCESSING_GUIDELINES.md` |
| 测试集先验（目标数 7,086 / 密度 7.09） | `docs/01_competition/SUBMISSION_CONFIDENCE_THRESHOLD_PRIOR.md` |
| 半决赛技术报告骨架与提交清单 | `docs/06_semifinal_submission/`（`TECHNICAL_REPORT_DRAFT.md`、`SUBMISSION_CHECKLIST.md`） |
| 训练迁到新 GPU（自检、数据打包、上线前检查、启动） | `docs/02_collaboration_and_specs/CLOUD_MIGRATION_RUNBOOK.md` |
| 历史审计（对齐/权重溯源/CP 消融） | `docs/04_audits_and_ablations/` |
| 仓库整理记录 / 找已删除的旧文件 | `docs/02_collaboration_and_specs/REPO_CLEANUP_LOG.md` |

## 8. 整理记录

- **2026-10-31**：第二轮清理。Git 跟踪文件 451 → 317（删 135、增 1 份清单）：删除 134 个已结案代码/配置（scripts 94：V36 平移/归一化/light 诊断 65、A01/A02 消融 17、eval/inference/train/viz 12；configs 36：v36_mechanism 25 + injection_warmstart 11；vendor 上游演示资产 4）与 1 份生成式 HTML 审计报告；`scripts/` 168→74、`configs/deimv2/` 67→31；同时清理约 87 GB 过期本地实验产物（旧检查点/旧 outputs 分析/外部数据集/旧测试派生数据等，`dataset/` 与主线权重未动）。清理前完整快照 tag `pre-cleanup-20261031`（= `2c170ef`）**已推送到 GitHub**；逐文件删除清单 `REPO_CLEANUP_20261031_removed.txt`（同目录），依据与本地清理明细见 `REPO_CLEANUP_LOG.md`。
- **2026-10-22**：删除 1,356 个历史实验文件（V1–V33 配置 / 启动 / 分析 / 可视化脚本、禁用的集成后处理），去掉文档软链层，V20 / V23 规范归入 `docs/03_architecture_designs/`。保留与删除的依据、各目录数量、取回方法见 `docs/02_collaboration_and_specs/REPO_CLEANUP_LOG.md`；逐文件清单 `REPO_CLEANUP_20261022_removed.txt`（同目录）。
- **2026-10-07**：根目录 / `work_dir` / `outputs` 位置归位与文档归纳（只移动不改流程），原表格已移入同一份 `REPO_CLEANUP_LOG.md`。
