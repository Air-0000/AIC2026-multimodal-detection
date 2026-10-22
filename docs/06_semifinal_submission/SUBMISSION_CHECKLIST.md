# 半决赛提交清单（代码 / 模型 / 环境 / 结果 / 报告）

> 依据：竞赛细则§十三。半决赛须提交完整算法代码（数据预处理、训练、推理）、测试结果文件、模型文件（下载链接）与加载说明及运行环境、PDF 技术报告；模型须能在指定测试环境中运行并输出结果。
> 状态：☐ 未开始　◐ 进行中　☑ 完成。截止日期见 `docs/02_collaboration_and_specs/AIC_TASK_MAP.md` 排期（截止前约 5 天冻结）。

## A. 待向组委会确认（越早越好）

- ☐ 半决赛具体截止时间与提交方式。
- ☐ “指定测试环境”的硬件、CUDA / PyTorch 版本、是否联网、显存上限。
- ☐ DINOv3-ViT-L（LVD-1689M 预训练，仅训练期作为蒸馏教师）是否属于允许的“公开预训练权重”。细则举例为 ImageNet、COCO、Objects365。
- ☐ 半决赛测试集是否同样提供配准红外，或需选手自行配准（影响预处理代码是否必须提交并可运行）。

## B. 代码

- ☐ 确定最小代码集，只含最终模型用到的部分：
  - `vendor/DEIMv2/`：`train.py`、`engine/`、最终配置文件（完全展开、无 `__include__`）；
  - `scripts/lib/`：`destripe_1d.py` 等推理与训练依赖的公共模块；
  - 预处理：红外配准（`scripts/prep/align_ir_pixel_pipeline_v3.py`）与 LBG 红外（`scripts/prep/prep_ir_localbg.py`），以最终训练数据实际使用的版本为准；
  - 推理入口：新写一个统一的 `predict.py`（见 E），不复用历次 `infer_test2_v*.py`；
  - 格式校验：`scripts/eval/check_submission.py`。
- ☐ 去掉硬编码路径：仓库脚本普遍写死 `/home/constant/CodeBuddy/Dev/AIC`、`/root/autodl-tmp/AIC` 等绝对路径，提交版必须全部改为命令行参数或相对路径。
- ☐ 说明文档：从原始数据到提交 zip 的一条命令序列（预处理 → 训练 → 推理 → 校验），附各步预计耗时与显存。
- ☐ 在一个全新克隆里按说明从零走通一遍（至少推理部分）。

## C. 测试集预处理

- ☐ 测试集的配准红外与 LBG 红外必须用与训练数据**相同版本**的脚本和参数生成。记录：两台机器生成的 IR/LBG 曾经不一致（MODEL_RECORDS §三十一）。
- ☐ 深度使用与训练一致的来源：纯净 v1 训练使用原始 `depth`（raw16），不是 `depth_p0`。

## D. 模型文件

- ☐ 最终权重（EMA）上传与下载链接，附 SHA256 与文件大小。
- ☐ 推理只需学生网络：导出时去掉 `dinov3_distiller.*` 与 `co_aux_head.*`，并用 strict=True 重载验证。
- ☐ 训练所用预训练权重的来源与 SHA256：COCO D-FINE-X `39361855c21f8c4757a9bb777a7ff134ae8aa110d0db67a374f081e44c2ce04b`；DINOv3-ViT-L `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`。

## E. 推理与结果文件

- ☑ 统一推理入口 `scripts/inference/predict.py`：复用配置自身的 `CocoDetectionMultiModal`（严格模式，缺文件即报错）与 `PostProcessor`，参数化输入目录、权重、输出目录与阈值；输出逐图 txt（CRLF、坐标归一化、按置信度降序、每图 ≤100 框、空图写空文件），可打包 zip，目录模式下调用格式校验与工作台注册。
- ☑ 契约测试 `scripts/diagnostics/test_predict_contract.py`（2026-10-22，冻结 V36 + `run_1.yml`）：200 张验证图输入张量与训练验证加载器逐位相同；mAP 0.5200238929 与记录一致；txt 通过 `check_submission.py`，读回误差 ≤5e-7。最终模型确定后须用其配置与权重重跑一次。
- ☐ 置信度阈值与每图框数只在有标注的验证集上确定，不依据测试集总框数（AIC_TASK_MAP 已撤回旧说法）。
- ☐ 跑 `scripts/eval/check_submission.py`；产出提交包后按 AGENTS.md 红线 11 导入可视化工作台。

## F. 运行环境

- ☐ 统一一个提交环境并导出依赖清单（`conda env export` 或 `pip freeze`）。现状：训练在 ailab40 的 `aic2026` 环境（torch 2.6.0+cu124，V100），本地工作站为 `deimv2` 环境（Python 3.11.15、torch 2.13.0+cu130、torchvision 0.28.0、faster_coco_eval 1.7.2、numpy 2.4.6）。须确认最终权重在提交环境下的验证集输出与训练环境一致或差异可忽略。
- ☐ 说明推理显存需求（1440×2560、FP32、batch 1）。
- ☐ 注意：DEIMv2 的 `engine/data/dataset/coco_dataset.py` 导入时会用 `faster_coco_eval.init_as_pycocotools()` 全局替换 pycocotools；评测口径须在报告中写明。

## G. 技术报告

- ☐ 按 [TECHNICAL_REPORT_DRAFT.md](TECHNICAL_REPORT_DRAFT.md) 填写并导出 PDF。
- ☐ 架构图。
- ☐ 主结果表与鲁棒性表全部引用 MODEL_RECORDS 中同口径的实测数字。

## H. 已发现的风险（来自历史推理脚本 `scripts/inference/infer_test2_v36_p2_direct_injection.py`，2026-10-22 审阅）

| # | 问题 | 影响 | 处置 |
|---|---|---|---|
| 1 | 测试时 LBG 红外未去条纹，而训练加载器对两路红外都去条纹 | 训练与测试输入分布不一致 | `predict.py` 复用训练加载类 |
| 2 | 测试集存在 `depth_p0` 时优先使用它，而纯净 v1 / V36 50 轮训练使用原始 `depth` | 训练与测试深度来源不一致（V36 对深度不敏感，影响可能小，未量化） | 同上，并在说明中固定深度来源 |
| 3 | 找不到配准红外时静默改用未配准的原始红外；找不到 LBG 时静默复制红外 | 结果可能悄悄退化且无报错 | 缺文件直接报错 |
| 4 | 手写 6 通道加载与缩放（`F.interpolate`），与训练的 torchvision `Resize` 实现不同 | 可能存在细微数值差异，未验证 | 契约测试逐像素比对 |
| 5 | 打包时使用固定类别阈值（稀有类 0.45、常见类 0.48）与每图 60 框上限 | 阈值来源需在验证集上重新确定 | 见 E |
