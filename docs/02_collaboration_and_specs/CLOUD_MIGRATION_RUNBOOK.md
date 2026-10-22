# 训练迁移到新 GPU（云端或其他机器）操作手册

背景：ailab40 两张 V100 的 ECC 关闭，GPU1 显存静默出错，纯净 v1 因此中断（MODEL_RECORDS §四十）。本手册把 v1 与纯 RGB 对照（以及之后的暗光臂、最终训练）迁到新卡，与平台无关。

## 0. 选卡

- **两张同型号的卡**，v1 与 RGB 对照各占一张并行跑；之后的暗光臂与最终训练也用同型号、同环境，否则对比不公平。
- 每卡显存 ≥24 GB：实测峰值 v1 18.5 GiB、RGB 对照 16.1 GiB（FP32，ailab40 V100，`bench_clean_arm.py`）；16 GB 的卡两条线都会 OOM（5060 Ti 实测）。
- 耗时参考：ailab40 V100（无 cuDNN）v1 每步 4.46 s（其中等数据 0.51 s），RGB 对照 3.80 s（0.17 s），50 轮不含验证约 56 h / 48 h。配置为 `num_workers: 0`，数据在主线程串行读取，换快卡也省不掉这部分（v1 约 900 步 × 50 轮 × 0.5 s ≈ 6 h，还取决于云机的 CPU 与磁盘）；实际耗时以第 4 步的 `BENCH=1` 为准。
- 优先选支持并已开启 ECC 的数据中心卡（A100、H100、L40S、RTX 6000 Ada、RTX PRO 6000 等）；消费卡（4090、5090）没有 ECC，只能靠第 4 步的自检把关。
- 架构与 PyTorch：Ampere / Ada / Hopper 可用 torch 2.6 + CUDA 12.4（ailab40 训练环境同版本）；Blackwell（RTX PRO 6000、5090）需要 CUDA 12.8 以上的构建（torch ≥2.7，本地工作站为 torch 2.13 + CUDA 13.0）。两条线用同一个环境即可。
- 非 V100 的卡会自动启用 cuDNN（train.py 只在 Volta 上关闭），比 ailab40 快很多；数值与 ailab40 不同，因此 v1 **从头重训**，不接 ailab40 的 `last.pth`。
- 磁盘：每份检查点约 3.66 GB，每 5 轮存一份加 last、best，每条线约 44 GB；两条线加数据（15.4 GB）与环境，建议 ≥150 GB。

## 1. 云端：代码与环境

```bash
git clone git@github.com:Air-0000/AIC2026-multimodal-detection.git AIC && cd AIC
git checkout <本次迁移固定的提交>
```

依赖参考 `docs/06_semifinal_submission/env/`：torch / torchvision（按卡选版本）、faster-coco-eval、pycocotools、opencv-python-headless、scipy、pillow、PyYAML、tensorboard、numpy。记下实际安装的版本（第 4 步会自动导出）。

## 2. 源端（Linux 工作站）：打包输入

工作站上 8000 个输入文件与 v1 训练时逐字节一致（2026-10-23 用下面的 `--reference` 复核，证据在工作站 `work_dir/audits/cloud_migration_tooling_20261023/`），LBG 红外在审计目录下：

```bash
python scripts/prep/pack_training_inputs.py --out /tmp/aic_stage --lbg work_dir/audits/v36_mechanism_20261018/train_lbg_ir --reference work_dir/audits/aic_clean_baseline_v1_20261021/acceptance_final/inputs.json
```

```bash
tar -chf aic_inputs.tar -C /tmp/aic_stage .
```

得到约 15.4 GB 的 `aic_inputs.tar`（2000 组四路图像、两份标注、两份预训练权重，以及 `work_dir/transfer/MANIFEST.sha256`），用 rsync / scp / 平台网盘传到云端。

## 3. 云端：解包

```bash
tar -xf aic_inputs.tar -C AIC
```

文件会落到配置期望的位置：`dataset/train/{visible,infrared_aligned,depth}`、`work_dir/train_lbg_ir`、`work_dir/deimv2_dataset/{train,valid}/annotations`、`vendor/DEIMv2/ckpts/`。

## 4. 云端：上线前检查（顺序固定，失败即停）

```bash
ROOT=$PWD/AIC GPUS="0 1" PY=python REQUIRE_ECC=1 BENCH=1 bash AIC/scripts/train/preflight_cloud.sh
```

依次执行：环境快照 → 输入哈希核对 → 每张卡 GPU 自检（显存反复读校验、矩阵乘与卷积可重复性、ECC）→ CPU 契约测试（纯净基准 14 项、RGB 对照 7 项、暗光臂 3 项）→ 两条线各 30 步基准测时（`BENCH=1`，在第一张卡上依次跑）。消费卡没有 ECC 时去掉 `REQUIRE_ECC=1`。自检的判别力已在坏卡上验证：ailab40 GPU1 显存读校验 88/720 次不一致而判失败，5060 Ti 330 次全部一致。全部结果写在 `logs/preflight_<时间>/`，最后一行为 `PREFLIGHT_PASS` 才继续。

## 5. 启动

```bash
ROOT=$PWD/AIC ARM=v1 GPU=0 PY=python nohup bash AIC/scripts/train/run_clean_arm.sh > /dev/null 2>&1 &
```

```bash
ROOT=$PWD/AIC ARM=rgb GPU=1 PY=python nohup bash AIC/scripts/train/run_clean_arm.sh > /dev/null 2>&1 &
```

日志在 `AIC/logs/<配置名>_train.log`，权重在 `AIC/vendor/DEIMv2/outputs/<配置名>/`。中断后续训：加 `RESUME=.../last.pth`（脚本会带上 `-u tuning=null`）。输出目录已存在时脚本拒绝启动；若在第一个检查点之前就失败，确认原因后删掉该输出目录再启动。

## 6. 运行中与结束后

- 盯每轮耗时、验证 mAP 与是否出现 `nonfinite`；严格守卫触发时先在该卡上重跑第 4 步的 GPU 自检。
- 结束后把 `best.pth`、`final.pth`、`log.txt`、训练日志与 `logs/preflight_<时间>/` 拉回工作站；用 `scripts/eval/eval_modality_robustness.py`、`scripts/analysis/compare_detectors_paired.py` 评估两条线。
- 记入 MODEL_RECORDS：卡型号、环境、preflight 结果、两条线的实测与配对比较。
