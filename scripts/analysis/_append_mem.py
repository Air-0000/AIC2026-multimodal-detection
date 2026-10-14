mem = "/home/constant/CodeBuddy/Dev/AIC/.workbuddy/memory/2026-09-26.md"
note = """

## 09-26 04:38 收尾：Stage C 完成态 + SWA + 损失地形 bug 修复

- **Stage C 已全跑完**（ep39 + 脚本自带 10 档 night 评测已出）。最终 best **0.5265@ep35**（AP75 0.7775，本轮最实改善）；day mAP=0.525（checkpoint0039=last 口径）；10 档 night 曲线白天端平稳（night50 0.503）、**night05→night01 断崖 0.422→0.128**（RGB 压暗到阈值后融合也救不回）。产物：outputs/metrics/stageC_results.json、outputs/reports/stageC_report.html（内联 SVG，含逐 epoch 曲线 + 鲁棒性图 + 明细表）。
- **SWA 权重已构建** outputs/vitl800_scratch_stageC_fuse/swa_stgC_ep24_39.pth（对 ep24/29/34/39 收敛轨道平均；直接平均 state['model']，落盘 1.5GB）。理由：§5.6 奇偶留出证明 argmax-best 在选择上追噪声，SWA/末轮更稳。
- **损失地形 bug 修复**：loss_landscape.py:83 在 filter_normalized_dir 里 w=p.detach().float() 是 CUDA 张量而随机方向 d 在 CPU，导致 wf.norm()/df.norm() 设备不一致 -> RuntimeError。改为 w=...float().cpu()，方向与 base/buf 全程 CPU，仅在 set_params 时 .copy_ 回 CUDA 参数。修复后重跑成功（pid 76419，GPU 3.2GB）。
- **后台串联任务 ARXw4T**（scripts/analysis/run_post_training_analysis.sh）：① 重跑 loss_landscape（random+PCA 双切片 + 插值势垒）-> outputs/loss_landscape_stageC.{json,png}；② 对 best/last/swa 三模型落 val det（outputs/analysis/_dets/*.pkl）；③ bootstrap_val_ci.py --iters 200 配对 bootstrap -> outputs/val_ci_bootstrap.{json,npz}（回答「SWA 是否显著优于 argmax-best」）。预计 ~20-30min，完成后补报。
- 坑：脚本里相对路径 scripts/analysis/... 在 cd vendor/DEIMv2 后失效（脚本实际在 $REPO/scripts/analysis），改用绝对路径 + cd $REPO 修复。
"""
with open(mem, "a", encoding="utf-8") as f:
    f.write(note)
print("appended", len(note), "bytes")
