# 运行环境快照（2026-10-23）

只读导出，用于选定半决赛提交环境。完整依赖见同目录两份 `pip freeze`。

| 项 | 训练：ailab40 `aic2026` | 本地工作站：`deimv2` |
|---|---|---|
| 用途 | 纯净 v1 及后续 50 轮训练 | 推理、评估、契约测试 |
| 系统 | Ubuntu 18.04.6，内核 5.4 | Ubuntu 24.04.5，内核 6.8 |
| GPU / 驱动 | Tesla V100-SXM2-32GB / 530.41.03 | RTX 5060 Ti 16GB / 595.91.07 |
| Python | 3.12.4 | 3.11.15 |
| torch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 | 2.13.0+cu130 / 0.28.0+cu130 |
| CUDA / cuDNN | 12.4 / 9.1.0 | 13.0 / 9.20 |
| faster-coco-eval | 1.8.0 | 1.7.2 |
| pycocotools | 2.0.11 | 2.0.11 |
| numpy | 2.5.3 | 2.4.6 |
| opencv-python-headless | 5.0.0.93 | 5.0.0.93 |
| pillow / PyYAML | 12.3.0 / 6.0.3 | 12.3.0 / 6.0.3 |

- 两个环境的 torch 主版本不同。提交前需选定一个环境，并确认最终权重在该环境下的验证集输出与训练环境一致或差异可忽略（契约测试 `scripts/diagnostics/test_predict_contract.py` 可直接用于此）。
- 训练环境从本地 wheel 安装（`pip freeze` 中的 `file:///...wheels_cu124/...` 行），提交时应改为公开源的版本号或附上 wheel。
- 工作站上 OpenCV 5.0.0 重新生成训练集红外与原文件逐字节一致（MODEL_RECORDS §三十九），红外预处理不受环境差异影响。
