# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Overview

**AIC（AI City）城市场景视觉多模态目标检测竞赛**仓库。任务：从可见光 / 红外 / 深度三模态图像中检测 **12 类**目标，输出官方提交格式的预测。

- 训练数据：`dataset/train/{visible,infrared,depth,depth_color}` + `dataset/train/labels/*.txt`（YOLO 归一化 `class_id cx cy w h`）。2000 张图 + 2000 个标签。
- 测试数据：`dataset/test/AIC2026_PHASE_2_1000/{visible,infrared,depth,depth_color}`，1000 张、**无标签**。
- 单一主干框架：**DEIMv2**（vendored 在 `vendor/DEIMv2/`）。早期用过的 RF-DETR 路线已废弃，`vendor/` 下只剩 `DEIMv2` 和 `Depth-Anything-V2`。
- 提交格式（逐图一个 `.txt`）：`<class_id> <cx> <cy> <w> <h> <confidence>`，空格分隔，坐标归一化到 [0,1]，**CRLF** 行尾，**每图 ≤100 框**，按 confidence 降序。无检测时写空文件。由 `scripts/eval/check_submission.py` 把关。

## Environment

只有两个 conda 环境是活的，**不要引用 `rfdetr` / `torch25`**（已不存在）：

| 环境 | Python 路径 | 用途 |
|---|---|---|
| `deimv2` | `/home/constant/miniconda3/envs/deimv2/bin/python` | 训练、评测、推理、数据准备（几乎所有脚本） |
| `aic-2` | `/home/constant/miniconda3/envs/aic-2/bin/python` | 少量辅助 / 夜化预处理脚本 |

约定前缀（几乎每个 `scripts/train/*.sh` 都有）：
```bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
`scripts/lib/` 里有 `nightify_aic.py`、`prep_night_real.py` —— DEIMv2 的 `engine/data/transforms` 会 `import nightify_aic`（顶层模块名），缺了 PYTHONPATH 会直接 ImportError。

几乎所有 `scripts/**/*.py` 顶部硬编码 `ROOT = "/home/constant/CodeBuddy/Dev/AIC"`；`scripts/eval/predict_*.py` 会 `os.chdir(vendor/DEIMv2)`。**从仓库根目录运行**，并注意 chdir 之后相对路径的含义会变（`predict_mm_tta.py` 因此主动把 `--out` resolve 回 ROOT）。

## Repository Layout

```
scripts/prep/      数据准备（RGB/IR/depth/heat/外部数据集/深度补全）
scripts/train/     训练与评测的 shell 驱动（run_*.sh）+ 少量 smoke 测试
scripts/eval/      推理（predict_*）、TTA、WBF 融合、格式校验、mAP 评测
scripts/gen/       批量生成实验 config 与训练脚本（消融实验矩阵）
scripts/analysis/  画图、扫描 log、统计（plot_*.py / scan_*.py）
scripts/lib/       共享库（nightify_aic 等）
vendor/DEIMv2/     DEIMv2 框架：train.py、engine/、configs/、outputs/（366 个实验目录）
vendor/Depth-Anything-V2/  深度补全权重与代码
dataset/           竞赛原始数据（gitignore）
work_dir/          运行时产物：数据集、缓存、submission、图表
data_aux/          派生的辅助数据集（BDD100K、nightify、black 变体等）
logs/              全部训练/评测日志（约 376 个）
MODEL_RECORDS.md   实验结果的单一真相源
```

## Pipeline Architecture

### 1. 训练入口

真实入口是 `vendor/DEIMv2/train.py`（不是 `engine/`）。参数：`-c/--config`、`-t/--tuning`（热启动权重）、`-r/--resume`、`--use-amp`、`--test-only`、`-u/--update`、`--seed`、`--output-dir`。

**必须 `cd vendor/DEIMv2` 后运行**，且 config 路径相对于该目录、`output_dir` 也落在该目录下。

```bash
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
# 训练（从 best_stg2 热启动）
$PY train.py -c configs/deimv2/<exp>.yml -t outputs/<src>/best_stg2.pth --use-amp
# 评测（--test-only + -r 指向待评测权重）
$PY train.py -c configs/deimv2/<exp>_val_<level>.yml -r "$CKPT" --use-amp --test-only
```

### 2. Config 体系（`vendor/DEIMv2/configs/`）

- `base/` —— 模型骨架与优化器默认值（`deimv2.yml` 定义 task/model/criterion/postprocessor、`optimizer.yml`、`dataloader.yml`）。
- `dataset/` —— 数据路径 + `num_classes: 12`。`aic_detection.yml`（RGB）、`aic_heat.yml`（heat）、`aic_detection_nightXX.yml` / `aic_detection_dndt_nightXX.yml`（9 档暗光：night50 → night0005）。
- `deimv2/` —— 每个实验一份，通过 `__include__` 列表继承（`engine/core/yaml_utils.py`，深合并，**后 include 者覆盖前者**）。
- `deim_dfine/`、`deim_rtdetrv2/` —— 上游框架自带，本项目不用。

### 3. 多模态融合方案

全部实现在 **`vendor/DEIMv2/engine/backbone/multimodal_backbones.py`**（约 1800 行）。**由 config 的 `DEIM.backbone:` 字段选择**（不是 `fusion:`），另配同名的参数块 `DINOv3STAsXXX:`。当前使用分布：RDTTrack 227、DINOv3STAs 102、RDTDiffuse 49、BiDirPaper 10、DiffFuse 8、CSD 3。

| Backbone 名 | 思路 |
|---|---|
| `DINOv3STAs` | 纯 RGB 基线；`in_chans≠3` 时 patch_embed 由 3 通道均值初始化扩展 |
| `DINOv3STAsCSD` | CSSA（Cao'23）：ECA 通道互补软开关 + CBAM 空间注意力 |
| `DINOv3STAsBiDir` | RGB↔IR 双向交叉注意力（SDPA + PVT 式 kv 降采样防 OOM） |
| `DINOv3STAsDiffFuse` | 由 RGB/IR 差异估计逐位置模态主导度，加权双向融合 |
| `DINOv3STAsTriModal` | RGB + heat 共用一个 `in_chans=4` 的 DINOv3 stem |
| `DINOv3STAsPureIR` | 单红外主干（`ir_in_chans` 1=heat / 2=heat+配准原始 IR） |
| **`DINOv3STAsRDTTrack`** | **主方案**（RDTTrack, NeurIPS'25）：IR 与 Depth 先经 `OrthogonalProjectionFusion` 去冗余，再用 `Fovea` + `PromptBlock` 以 prompt 形式注入冻结的 RGB 主干 |
| `DINOv3STAsRDT` | 去掉 IR 的 RGB-D 对照版 |
| `DINOv3STAsRDTDiffuse` | 保留正交去冗余，注入方式改为差异加权融合（`inject: prompt\|diffuse\|both`） |
| `DINOv3STAsBiDirPaper` | Tang'24 忠实复现：RGB 冻结 + IR 可训练，渐进双向融合，无注意力无深度 |

另有两个**不是网络模块**的关键概念：
- **deepir** —— `work_dir/deepir_cache/train/*.pt` 的 **6 通道离线缓存**（避免每 epoch 重复读 4 路图），配合 `CocoDetectionMultiModalCached`（`engine/data/dataset/coco_dataset_mm.py`）。生成脚本是 **`vendor/DEIMv2/prep_deepir_cache.py`**（须 `cd vendor/DEIMv2` 运行，支持断点续跑）；`scripts/prep/prep_deepir_cache.py` 是带 CLI 的副本。
- **midboost** —— 训练策略：把 `RandomNight5ch` 的亮度采样聚焦中段（`level: [0.005, 0.45]` + log-uniform 过采样），低 LR 微调 `best_stg2`。L4 = midboost + 光度扰动加强（p 0.5→0.8）。

### 4. 数据准备（`scripts/prep/`，输入 → 输出）

| 脚本 | 输出 |
|---|---|
| `prepare_dataset.py` | `work_dir/rfdetr_dataset/{train,valid}/{images,labels}` + `data.yaml`（**symlink**，1800/200 划分，seed=42） |
| `prep_ir_heat.py --mode heat\|raw` | heat → `work_dir/deimv2_heat/`；raw → `work_dir/deimv2_ir/`。核心 `heat_residual()` = `clip(align_ir(IR) - RGB_gray, 0, 255)`（整数平移 + FFT 亚像素配准，质量门限回退） |
| `prep_depth_refine.py` | `work_dir/deimv2_depth/images`（8-bit PNG） |
| `refine_full.py` / `depth_refine*.py` / `depth_to_color.py` | `work_dir/depth_refined_*` 系列深度补全/去噪变体 |
| `prepare_flir.py` / `prepare_llvip.py` / `prepare_m3fd.py` / `prepare_bdd100k.py` | `work_dir/FLIR`、`work_dir/LLVIP`、`work_dir/M3FD`、`data_aux/BDD100K`（外部数据集，用于泛化/消融） |
| `prep_night_robust.py`、`make_black_ir.py` | `data_aux/aic_valid_*_night*`、`aic_valid_{ir,depth,rgb}_black` 鲁棒性评测集 |

### 5. 推理与提交（`scripts/eval/`）

| 脚本 | 用法 |
|---|---|
| **`predict_mm_tta.py`** | **当前主力**。6 通道 + TTA（hflip × multiscale）+ WBF：`python predict_mm_tta.py --config CFG --ckpt CKPT --test-root dataset/test/AIC2026_PHASE_2_1000 --out work_dir/submission_... --scales 640,800 --thresh 0.3` |
| `predict_deimv2.py` | 单尺度 RGB/heat 推理：`[CKPT] [OUT_DIR] [THRESH=0.3] [SIZE=640]` |
| `wbf_fuse.py` | `[A_DIR] [B_DIR] [OUT_DIR] [IOU_THR=0.55]`，纯 numpy，融合多模型预测 |
| `check_submission.py` | `python3 check_submission.py [SUB_DIR] [TEST_VISIBLE_DIR]`，纯 stdlib，**提交前必跑** |
| `eval_submission.py` | `[SUB_DIR] [ANN_JSON]`，pycocotools 算 mAP |
| `eval_mm_tta_val.py` / `predict_val.py` / `predict_tta.py` / `tta_*.py` / `predict_multi_thr.py` / `predict_rgb.py` | 依赖已废弃的 `rfdetr` 环境，**目前不可运行** |

**6 通道布局必须与训练一致**（`predict_mm_tta.py` 头部注释 + `CocoDetectionMultiModal.load_item`）：`ch0-2=RGB`、`ch3=heat`、`ch4=raw_ir`、`ch5=depth`。归一化用 6 通道的 `MEAN/STD`。

### 6. 结果记录

**`MODEL_RECORDS.md` 是唯一真相源**，不要再新建分散的实验记录。表A = 训练模型总览（按模态/数据集分组），表B = night/暗光评测索引。跑完新实验后重跑：

```bash
python3 scripts/collect_model_records.py
```

当前最优：`logs/midboost_l4u3_train.log` 中 `best_stat: {'epoch': 38, 'coco_eval_bbox': 0.5372}`（表A 尚未收录，需重跑采集脚本）。

## Common Commands

```bash
# 训练一个实验（进 DEIMv2 目录，日志统一写 logs/）
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
/home/constant/miniconda3/envs/deimv2/bin/python train.py \
  -c configs/deimv2/<exp>.yml -t outputs/<src>/best_stg2.pth --use-amp \
  > /home/constant/CodeBuddy/Dev/AIC/logs/<exp>_train.log 2>&1

# 跑一整套「训练 + 10 档暗光评测」（脚本内含 GPU 排队逻辑）
bash scripts/train/run_l4u3.sh

# 生成 6 通道缓存（改了数据准备后必跑）
cd vendor/DEIMv2 && /home/constant/miniconda3/envs/deimv2/bin/python prep_deepir_cache.py

# 生成 phase2 提交（TTA 版）
/home/constant/miniconda3/envs/deimv2/bin/python scripts/eval/predict_mm_tta.py \
  --config <cfg> --ckpt <ckpt> \
  --test-root dataset/test/AIC2026_PHASE_2_1000 \
  --out work_dir/submission_phase2_l4u3_tta --scales 640,800 --thresh 0.3

# 提交前格式校验
python3 scripts/eval/check_submission.py work_dir/submission_phase2_l4u3_tta \
  dataset/test/AIC2026_PHASE_2_1000/visible

# 重建实验记录表
python3 scripts/collect_model_records.py
```

`scripts/train/run_*.sh` 普遍带 **GPU 排队**（`nvidia-smi` 判空闲，如 `run_l4u3.sh` 要求 ≥12000 MiB 空闲），因为与另一个 mask2former 任务抢卡。

## GPU / 显存

显卡是 **RTX 5060 Ti，16596140032 B = 15.46 GiB**。训练/推理前先估显存：

```bash
# 有 GPU: 跑 2-3 iter 测真实稳态峰值; --auto 自动二分搜最大可行 micro
cd vendor/DEIMv2
/home/constant/miniconda3/envs/deimv2/bin/python \
  /home/constant/CodeBuddy/Dev/AIC/scripts/train/probe_mem.py \
  -c configs/deimv2/<cfg>.yml --auto

# 无 GPU: 用实测锚点做一阶外推(±20%)
... -c <cfg>.yml --estimate
```

- **实测锚点**：基线 640×640 / ViT-S 三流 / micro=5 / amp+ema → **峰值 12228 MiB**（`logs/midboost_l4u3_train.log` 的 `max mem`）。
- **静态显存公式**：`参数×4 + 可训练×4(梯度) + 可训练×8(AdamW) + EMA(总参数×4) + AMP bf16 cache(总参数×2)`。EMA 是大头之一，紧张时可关。
- 估算值（未实测）：vitb800 @micro=2 ≈ **11.0 GiB**（余量 4.5G，较安全）；vitl800 @micro=1 ≈ **13.0 GiB**（余量 2.5G，偏紧）。

⚠️ **会话里看不到 GPU，但卡是真的在**（已取证，不要再重复排查）：

- 硬件和驱动都正常：`lspci` 有卡、`/proc/driver/nvidia/gpus/*/information` 可读、`lsmod` 有 nvidia/nvidia_uvm、`/proc/devices` 有 nvidia=195 / nvidia-uvm=507。
- 但 **WorkBuddy 进程跑在 systemd scope `app-gnome-workbuddy-*.scope` 里**（`/proc/self/cgroup`），该 scope 带设备沙箱：`/dev` 被一个 10MB 的 **tmpfs 覆盖挂载**，只剩 `null/zero/random/urandom/full/tty` 六个节点，没有 `/dev/nvidia*`。
- 于是 `nvidia-smi` 报 "couldn't communicate with driver"、`torch.cuda.is_available()` 为 False —— libcuda 找不到可以 `open()` 的设备文件。
- **手工补建节点不可行**：`mknod` 一律 `EPERM`。实测已排除两个常见误判 —— capability **有** `cap_mknod`（`CapEff=0x1ffffffffff`）、文件系统也可写（`/dev` 上 `touch` 成功）。真正拦住的是 **seccomp 过滤器**（`/proc/self/status` 的 `Seccomp: 2`），在 ext4/overlay 上 `mknod` 同样 EPERM。另外 D-Bus 不可用，`systemd-run -p PrivateDevices=no` 也起不来。
- **结论**：会话内不可能访问 GPU，别再试 mknod / mount / systemd-run。要么让用户在**普通系统终端**（不在该 scope 内）跑，要么用 `--estimate` 外推。

## Key Constraints / Gotchas

- **桌面合成器会抢显存**：`engine/solver/det_engine.py:61` 的注释记录了 Wayland 合成器 niri 曾独占 **~12 GiB**，这才是历史上「只有 ~13GB 可用」说法的来源（卡本身是 15.46 GiB）。梯度累积（`accum_steps`）就是为此加的。跑长训练前先确认没有桌面合成器在吃显存。
- **`interaction_indexes` 三流共用**：RGB 与 aux(ir/depth) 的 ViT 层数不等时（如 RGB=ViT-L 24 层 + aux=ViT-S 12 层），索引会越界并在 `vision_transformer.py:451` 断言 `only N / 3 blocks found`。已由 `_scale_interaction_indexes()` 按 depth 比例自动缩放，新增 backbone 组合时留意。
- **12 类，顺序固定**（`scripts/prep/prepare_dataset.py:CLASSES`）：person, boat, animal, seat, sign, bicycle, car, ball, light, garbage can, uav, tricycle。class_id 0-11。
- **深度图 16-bit 陷阱**：depth 以 **16-bit 毫米**存储，读取必须走 `depth_read='raw16'` 约定（mode `I;16`/`I` → `/19999`，否则 `/255`）。**绝不能用 `PIL convert("L")`**，那会把毫米值截断成二值掩码 —— 这是历史上「depth 是死权重」的 bug（见 git `53c6577`）。同时 `depth_invert` 方向也已校正，改动前先看该 commit。
- **`dataset/test` 无标签**，只能做格式校验，不能算 mAP。
- Symlink 数据集：不要删除 `dataset/`，也不要破坏 `work_dir/*_dataset` 的软链。
- Checkpoint 加载要同时兼容 EMA（`ck["ema"]["module"]`）和 `ck["model"]` 两种键；`strict=False`。
- 训练时 `n_channels`（4/5/6/7）必须与数据准备产出的通道数一致，config 的 `ir_in_chans` / `depth_in_chans` 与 backbone 实现一一对应。
- `logs/` 是唯一日志落点（旧脚本有的还写 `/tmp/*.log`）。
- `.gitignore` 忽略 `*.pth`/`*.log`/`work_dir/`/`/dataset/`，但特意放行了 `vendor/` 内框架自带的 dataset 代码目录（见 commit `95b2090`），改 gitignore 时别再误伤。
