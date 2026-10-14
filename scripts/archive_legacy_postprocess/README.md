# Archived Legacy Postprocessing Scripts (Deprecated)

> **归档说明**：  
> 本目录下的所有外部后处理脚本（包含外部多尺度切片、外部 WBF 框融合、人工类别差异化门限、外部多阈值投票等）已被**永久废弃并归档**。

### 为什么废弃？
1. **统计与工程脆弱性**：外部后处理脚本依赖复杂的外部 multiprocessing 切片和重叠拼接，不仅单图推理时间暴涨至数秒甚至数十秒，而且在边缘接缝处极易引发假阳性聚类。
2. **内生化全面碾压**：自 Sovereign V11/V12/V14 起，CAA-SAHI 3-View 高清切片、接缝自适应衰减、专属类别过滤以及 GPU Batched NMS（IoU=0.65）已经**100% 完整内生化在 `DEIM.forward()` 的底层计算流中**。
3. **全链路统一**：模型输入多模态原图张量，模型输出最终干净的绝对 BBox 预测张量，推理全链路严格由统一引擎接管，零外部手搓脚本依赖。

### 官方推荐推断入口
统一使用：
- 推理生成提交文件：`scripts/inference/predict_sovereign_native.py`
- 验证集权威评估：`python3 train.py -c configs/... --test-only -r ...`
