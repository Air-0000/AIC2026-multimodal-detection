# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

> 上面导入的 `AGENTS.md` 是环境、版本现状、红线、常用命令与文档导航的正本（`CODEBUDDY.md` 也软链到它）。本文件只补充 AGENTS.md 没写的内容；纯净基准的细节以 `docs/02_collaboration_and_specs/CLEAN_BASELINE.md` 为准。

## 0. 这个 checkout 是什么

- `~/Desktop/Project/AIC2026-multimodal-detection` 是 macOS 上的 **git 镜像**：只有 `vendor/DEIMv2/`、`scripts/`、`docs/`；没有 `dataset/ work_dir/ outputs/ weights/ logs/ archive/`，本机也没有 torch / conda 环境。这里只能读改代码与文档，**不能训练、推理或跑测试**——需要运行时，给出在 Linux 开发机或远端节点执行的命令。
- 脚本与文档里的 `ROOT = "/home/constant/CodeBuddy/Dev/AIC"`、`/root/autodl-tmp/AIC`、`/home/ailab40/...` 等绝对路径指 Linux 机器，**不要把它们"修"成本机路径**。
- 2026-10-22 起仓库只保留主线、里程碑（V23 / F5 / V27 / V28-A / V34–V37）与框架；`MODEL_RECORDS.md` 等历史叙述里提到但仓库里找不到的脚本 / 配置，查 `docs/02_collaboration_and_specs/REPO_CLEANUP_20261022_removed.txt`，用 `git show pre-cleanup-20261022:<path>`（无 tag 时用 `4d71413`）取回，不要凭记录重写。

## 1. 工作顺序（用户定下的规则）

- 先修通用正确性缺陷、固定纯净基准，再逐项做同起点、同预算的受控对照。不要把未验证机制（V37 平移/抗混叠、查询配额 Routing、Aux 分类减权、DOG 回退等）塞进基准，也不要把 V34–V36 旧权重/成绩当成纯净基准的结果。
- 比赛约束：**逐图独立推理**，不引入跨图状态。

## 2. 测试

没有 lint 或 CI。"测试"是 `scripts/diagnostics/` 下的 CPU 契约测试（unittest，不训练、不更新参数，需要 `deimv2` 环境）：

```bash
CUDA_VISIBLE_DEVICES='' $PY scripts/diagnostics/test_clean_baseline_contract.py
```

```bash
CUDA_VISIBLE_DEVICES='' $PY scripts/diagnostics/test_clean_baseline_contract.py CleanContracts.test_yaml_loads_are_independent
```

- 改动基准用到的 `engine/` 代码后至少重跑 `test_clean_baseline_contract.py`。其他 CPU 契约测试：`test_rgb_control_contract.py`、`test_night_aug_contract.py`（RGB 对照 / 暗光两臂）、`test_light_val_analysis.py`、`test_cssp_module.py`、`test_irbind_pilot_contract.py`（红外绑定试验：热启动键重映射、学习率分组、严格暗光增强）、`check_v36_*_contract.py`；`test_predict_contract.py` 是 `predict.py` 的验证集推理契约（要 GPU 与数据）。
- GPU 端只读验收用 `scripts/diagnostics/audit_clean_baseline.py`（完整命令见 `CLEAN_BASELINE.md`；`--out` 目录必须不存在，先确认 GPU 空闲）。

## 3. 训练入口补充（对 AGENTS.md §5 的修正）

- 基准训练：`cd vendor/DEIMv2 && $PY train.py -c configs/deimv2/aic_clean_baseline_v1.yml`。新实验必须用全新 `output_dir`。
- **精度随 GPU 架构变化（2026-10-23 起，commit `c8657a6`）**：三臂配置都是 `use_amp: true`。
  - SM ≥ 8.0（Ampere / Ada / Hopper / Blackwell）：`train.py` 打开 TF32（matmul 与 cuDNN）；训练前向在 BF16 autocast 下运行（`det_engine.py` 按架构选 dtype），criterion 关闭 autocast 按 FP32 算；`GradScaler` 仍启用；验证 `evaluate()` 不走 autocast，FP32 权重 + TF32 matmul。纯净基准 v1 的 seeta 运行（§四十一–四十三）就是这种设置。
  - SM < 8.0（V100）：TF32 关闭、cuDNN 关闭；`train.py` 与 `clean_lifecycle.py` 对 `use_amp: true` 直接报错，所以在 V100 上必须加 `-u use_amp=False`（用 `scripts/train/run_clean_arm.sh` 时设 `USE_AMP=0`）才能按纯 FP32 跑。
  - 结果因此依赖 GPU 架构：同一份配置在 V100 是 FP32，在 Blackwell 是 BF16 + TF32。跨架构的运行不能当作同一设置比较；同一组对照（v1 / RGB 对照 / 暗光臂）要在同一架构上跑。
- 续训：`-r <ckpt> -u tuning=null`。`strict_resume` 缺 model/EMA/optimizer/RNG 状态或 shape 不符即失败——这是设计，不要放宽。
- CLI 语义：`-t` 只加载权重热启动；`-r` 完整恢复；`-u key=value` 覆盖 YAML；`--seed` 仅在显式传入时覆盖 YAML seed。
- 有效 batch 就是 `total_batch_size`（三臂均为 2）；`accum_steps` 只把一个 loader batch 拆成 micro-batch，**不相乘**（旧文档里的 effective batch 乘法已作废）。三臂现为 `accum_steps: 1`、训练 `num_workers: 8` + `persistent_workers`（V100 时期为 2 与 0；配置文件第一行注释仍写着 `accum_steps=2`，以字段为准）。
- `clean_baseline: true` 时 `DetSolver.fit()` 改走 `engine/solver/clean_lifecycle.py::fit_clean`（epoch 单调推进、不从 best 回滚、原子保存 last/final/best）；其他旧 config 仍走带回滚等历史行为的老路径。

## 4. 架构大图（需要跨文件才看得懂的部分）

- **注册与装配**：`engine/core/workspace.py` 的 `@register()` 把类登记进 `GLOBAL_CONFIG`。YAML 顶层 `model: DEIM`，`DEIM:` 块里 `backbone/encoder/decoder` 的值是**类名**，各类的构造参数写在同名顶层块（`HGNetv2MultiModal:`、`HybridEncoder:`、`DFINETransformer:`…）。
- **`__include__`**：`engine/core/yaml_utils.py::load_config` 递归深合并，后 include 者覆盖前者；每次 load 独立（共享 dict 污染已修）。纯净基准刻意不用 include。
- **数据**：`engine/data/dataset/coco_dataset_mm.py` 的 `CocoDetectionMultiModal`（及 `…Cached`）把 RGB + 对齐 IR + LBG IR + depth 拼成 6 通道，来源由 config 的 `ir_folder / raw_ir_folder / depth_folder / depth_read / n_channels` 决定。`strict_inputs: true` 时缺通道、非 raw16、设置了 `ABLATE` 环境变量、destripe 依赖缺失都会直接报错；非 strict 时会静默降级。
- **颈 / 头 / 损失**：`engine/deim/hybrid_encoder.py`（P2 引导）、`dfine_decoder.py`（`DFINETransformer`；查询选择为 P3 配额 35% + 全局补齐，SDQS / distinct 等开关）、`co_aux_head.py`（Co-ATSS）、`deim_criterion.py`、`postprocessor.py`；教师蒸馏 `engine/backbone/dinov3_teacher.py`（`require_teacher_weights` + SHA256 校验，教师始终 eval）。
- **数值契约**：`engine/misc/numerical_contract.py::require_finite`；`stop_on_nonfinite` 下 NaN/Inf 直接抛错，不得改回"清零继续"。新增但未实现或未接线的开关要 fail-closed（参照 SDQS 的处理），不能静默无效。

## 5. 其他坑点

- **`MODEL_RECORDS.md` 的写法**：`collect_model_records.py` 只重写 `<!-- AUTO_GENERATED_TABLES_START/END -->` 之间的表格，保留其后的手写章节（写前自动备份到 `archive/backups/`，手写部分字符数回退会硬断言中止）。手写结论加在 END 标记之后，不要手改自动表格。
- **提交框数**：`check_submission.py` 强制每图 ≤100 框。`PROJECT_SPEC.md` §6 的"每图 ≤60 / 总数 7,000–7,200"是 V23 时期的做法；无标签测试集的总框数不能证明优劣，阈值只在有标签的 200 张验证集上定。
- **结论措辞**：历史文档中大量"必涨 / 根治 / 实锤"式叙述已被撤回。写记录时区分完整训练实测、零训练 / pilot、诊断观察与模拟数据；一次功能验收不等于可复现或有收益。
