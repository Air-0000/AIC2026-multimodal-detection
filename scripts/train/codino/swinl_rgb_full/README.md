# Co-DINO 夜间门控支线（G0 纯 RGB / G1「744」冻结底座）

> 归档于分支 `codino-rgb-ceiling` ｜ 实验日期 2026-10-30 ～ 2026-11-02 ｜ 状态：**机制 pilot，未用于提交**
> 详细实验记录见 **`master` 分支**的 `docs/MODEL_RECORDS.md` §五十（该记录在本分支之后合入主线）；本目录只归档脚本、配置与证据，不归档权重与数据。

## 1. 本目录是什么

`scripts/train/codino/swinl_rgb_full/` 是 2026-10-30～11-02 Co-DINO 支线的「**冻结检测器 + 夜间亮度门控残差分支**」实验脚本与配置归档。
底座是外部协作方的 Co-DINO 5-scale Swin-L 三模态融合主线（代号 **s0918**）——按固定公式把 RGB / 红外 / 深度融合成 3 通道 PNG 后送入检测器，**本地没有该底座权重**，本仓也不归档它。本目录只保留门控分支代码、契约校验脚本、训练/评测编排脚本与训练配置证据；框架源码（onedl-mmdetection 3.5.1）、wheels、数据、标注与预测 json 均不入库。

## 2. 底座与合规背景

- **底座**：外部协作方 Co-DINO 5-scale Swin-L 三模态融合主线 s0918。检测器（Swin-L backbone + Co-DETR 头）与融合后的 3 通道输入一起作为固定底座使用。
- **思路**：冻结检测器全部参数，只训练 `data_preprocessor.fuse.*` 的小分支（5 层卷积、hidden 32、末层零初始化），以
  `rgb' = rgb + G·S·net(Y, T, W, D, V, G)` 只加一个亮度残差。图像级门 `G` 由整图 mean-luma 决定：≤ lo(0.02) 为 1、≥ hi(0.05) 为 0、中间线性；`S` 是热窗口软化、`V` 是深度有效掩码。
- **正常光与底座逐位一致**：`G = 0` 时输入原样通过且不调用 net（eval 模式直接返回原 uint8 张量），所以正常光下检测器输入与输出都与 s0918 底座逐位相同；只有夜间（mean-luma 低）才真正加上亮度残差。
- **合规背景**：原始设计意图是"先训纯 RGB，再用一个训练期生效、正常光推理时不激活的亮度门控分支接入热/深信息"，在满足三模态规则的同时保持纯 RGB 能力。它属于单个检测器内部的输入侧分支，不做多模型融合或投票，不是赛题禁止的集成。但本支线只到机制 pilot 阶段，**未用于任何提交，也没有涨点证据**。

## 3. 两代文件分组

### G0（2026-10-30）：纯 RGB Swin-L 底座 + 门控

| 文件 | 作用 |
|---|---|
| `aic_nightgate.py` | 门控主体：`LoadAICGateInputs`（6 通道 BGR+热/窗/深）、`AICRandomNightBGR`（训练夜化增强）、`AICNightProbeBGR` / `AICZeroAux`（评测探针）、`AICNightGateDataPreprocessor`（G·S·net 残差）、`AICFrozenCoDETR`（冻结检测器、backbone 常驻 eval） |
| `build_nightgate_cfg.py` | 由纯 RGB 配置生成门控微调配置（换 loader、插夜化增强、冻结与可训练参数、lr/epoch 覆盖） |
| `predict_nightgate_txt.py` | 比赛格式 TXT 推理（每图 top 100 框）；支持 `--night` 探针、`--zero-aux` 关辅助通道、`--ir-dir/--depth-dir` 覆盖，并输出逐图 `gates.json` |
| `test_nightgate_contract.py` | G0 契约测试：门函数数值、loader 前 3 通道逐位一致、正常光预测逐位一致、夜间门开/零初始化 net 不改变输出、24 图 smoke 后仅 fuse 张量变化 |
| `chain_nightgate.sh` | 纯 RGB 训练结束后的串联编排：Phase 3 TXT → 门控微调 → 门控 TXT 与逐字节对比 → split-A 留出集探针与 mAP |
| `codino_aic_full_rgb_s20260918.py` | 纯 RGB Swin-L 训练配置（s0918 配方去掉融合，seed 20260918） |
| `run_swinl_rgb_full.sh` | 纯 RGB Swin-L 训练启动脚本 |
| `setup_venv.sh` | 协作方卡 A 同款环境搭建（`venv_codino`，torch 2.10.0+cu129 / onedl-mmcv 2.3.7.post1 等 pin） |
| `dump_rgb_images.py` | 导出 visible-only PNG 数据集（与融合数据集同名同标注，校验 JPEG 解码口径） |

### G1「744」（2026-11-01/02）：冻结 s0918 融合底座 + 9 通道门控输入

| 文件 | 作用 |
|---|---|
| `aic_744_nightgate.py` | 744 版主体：`Load744GateInputs`（9 通道 = 原融合 BGR + 热/窗/深 + 原 BGR）、`RandomNight744`（只夜化可见通道、保留原热/深贡献）、`DataPreprocessor744`（G=0 直接返回原融合图）、`Finite744GradientHook`（分支梯度非有限即报错） |
| `build_744_gate_cfg.py` | 从 s0918 配置生成冻结分支实验配置（4 epoch、lr 1e-3、split A 训练、`Load744GateInputs`、关闭中途验证） |
| `check_744_gate.py` | 训练前 fail-closed 契约：12 张原图融合像素 SHA256、正常光 744 预处理器逐位旁路、3 张图预测逐位一致、仅 `data_preprocessor.fuse.*` 可训练、24 图 smoke 后非分支张量逐位不变 |
| `eval_744_cpu.py` | CPU 探针评测（normal / night01 及其 base、aux-off 反事实；可断点续跑，pycocotools 汇总 mAP） |
| `verify_744_checkpoint.py` | 训练后核验：检测器全部张量与底座逐位相同、分支全部有限且确实训练过；记录 checkpoint/base 的 sha256 到 `*.frozen_verified.json` |
| `run_744_gate.sh` | 744 部署根一键流程：环境校验 → GPU 自检 → 数据 sha256 → 生成配置 → 契约检查 → 训练 → 核验 |
| `run_744_cpu_local.sh` | 本地 CPU 复算入口（macOS 本地 venv，300 图 night01） |
| `chain_744_cpu_local.sh` | CPU 复算串联（等 pilot 结束后跑 300 图 night01） |
| `cpu_attention_chunks.py` | CPU 参考 MSDA 按查询位置分块，限制临时显存/内存；安装前用逐位相等的参考测试 |
| `compare_ckpts.py` | mmengine checkpoint 并排解析与比对（meta、张量数、差集、逐位相同数、最大差） |
| `744_gate_generated.py` | `build_744_gate_cfg.py` 在部署根 `/root/744_gate` 生成的训练配置（生成物留档，绝对路径为远端部署路径，仅作训练配置证据） |

### 公共依赖

- `aic_mm/`（位于 `scripts/train/codino/aic_mm/`）：**夜化仿真与热信号分解依赖**。
  - `night.py`：夜间仿真（低频照度场 × 冷白平衡 × 泊松光子噪声 + 高斯读出噪声）、`AICNightRGB` 确定性探针、`AICRandomNightRGB` 训练增强、`probe_seed`。
  - `ir_signal.py`：从组织方 IR 文件反解热信号（高/低频估计 RGB 泄漏系数 0.15 并扣除，得到 T + 热窗掩码）；`decompose_thermal` 供数据 pipeline 使用。
  - 两个模块互不依赖，只依赖 numpy / OpenCV。
- `codino_aic_hooks.py`：来自原 **codino_fused 仓**的 `scripts/cloud/newbase/codino/codino_aic_hooks.py`，提供 `AicParamCheckHook`（载入后检查参数有限）、`AicIterRecorderHook`（逐步 JSONL 训练记录，容忍 AMP 偶发非有限步）、`AicEarlyGateHook`（验证指标门控）。
- 所有脚本按部署根写死路径，彼此通过 `PYTHONPATH` 与 `custom_imports` 关联，见第 7 节。

## 4. 已通过的契约（2026-11-01，GPU）

「744」/ G1 的契约由 `check_744_gate.py` 与 `verify_744_checkpoint.py` 在 GPU 上实测通过：

1. **融合像素逐位一致**：12 张原图的 744 融合像素 SHA256 与基线（s0918 的融合公式）完全一致。
2. **正常光预测逐位一致**：3 张图上底座与 744 门控模型的 `bboxes / scores / labels` 逐位相同（此时门 `G = 0`，预处理器原样旁路）。
3. **只有分支可训练**：除 `data_preprocessor.fuse.*` 外，全部检测器参数 `requires_grad = False`。
4. **24 图 smoke 训练后冻结不变**：除分支外的 checkpoint 张量逐位不变，分支确实更新。
5. **训练后核验**：`verify_744_checkpoint.py` 断言检测器全部张量与底座逐位不变、分支张量全部有限且已训练。

G0 对应契约脚本为 `test_nightgate_contract.py`（门函数数值、6 通道 loader 前 3 通道逐位一致、正常光预测逐位一致、夜间门开且零初始化 net 不改变输出、24 图 smoke 冻结），与上表同一套判定思路。

## 5. 训练与评测口径

### 5.1 训练（G1「744」）

- 部署根 `/root/744_gate`；冻结底座 `/autodl-fs/data/pretrained_models/teammate_s0918/epoch_12.pth`（s0918 主线第 12 轮权重），配置模板 `/autodl-fs/data/pretrained_models/teammate_s0918/codino_aic_full.py`。
- **4 epoch**、batch 1、num_workers 2、AdamW lr 1e-3（200 iter linear warmup + cosine 到 5%），`val_interval=5` 所以训练中途不验证；split A 训练集 **1700 张**（`splitA_train.json`），训练期 `RandomNight744`（prob 1.0、level 0.003–0.08 log-uniform、photon 70 / read 0.012），热/深通道保持原值。
- 产物 `epoch_4.pth` 已拷回本地（gitignore）：`outputs/checkpoints/s0918_gate_20261101/epoch_4.pth`，大小 **955,233,051 B**，sha256 `3cf15f3a211975227ae8eadab151d1428f834ed1fd2d3269e4b491c854d5e907`。

### 5.2 评测尝试（都不是有效成绩）

- **GPU pilot（1 图）**：normal AP 0.40（单图，无意义）、night01 及其变体 0.00；路径 `outputs/analysis/s0918_gate_workstation/gpu_pilot/metrics.json`。
- **CPU 300 图 night01（合成夜化）**：AP 0.059 / AP50 0.112。CPU 推理且本机 mmcv **LineNMS 未编译**（日志明确 `Line NMS is not compiled`），只能证明链路跑通，**不能当成绩**。
- **Phase 3 门控扫描**：1000 张测试图最暗 mean-luma **0.0637**（`shuming_860_00000209`），高于门上限 hi=0.05 → **triggered 0/1000**，即当前阈值下门控在 Phase 3 完全不会打开；扫描结果 `outputs/analysis/s0918_gate_workstation/gate_test_phase3_scan.json`。

### 5.3 状态

机制 pilot：契约（逐位一致 / 冻结 / 只有分支训练）已通过，但**没有任何训练收益证据，未用于任何提交**。门控阈值之外的调参、真实夜间数据验证与大规模评测都未做。

## 6. 运行假设

- 这些脚本按**远端部署根**写死（`/root/744_gate`、`/autodl-fs/data/...`、`/root/tm/...`、`/root/rgb_all2000`、`/root/autodl-tmp/AIC/dataset/...` 等），另有本地 CPU 复算脚本引用 macOS 本地路径。本仓**只做归档，不改写这些路径**；换机器运行需自行替换部署根。
- 复现需要自备：onedl-mmdetection **3.5.1**（Co-DETR 的 mmdet 3.x 移植）+ onedl-mmcv（2.3.7.post1）+ 对应 CUDA/torch 环境（G1 用 torch 2.10.0+cu129 / Python 3.12，见 `setup_venv.sh` 的 pin）+ **s0918 底座权重** + AIC 数据集（visible/infrared/depth 与标注）。
- 注释中的「MODEL_RECORDS §五十八」是**原 codino_fused 项目**的记录编号；本仓（`master` 分支）对应编号为 §五十，`aic_nightgate.py` 与 `aic_mm/night.py` 里已相应改写，其余内容保持归档时原样。
