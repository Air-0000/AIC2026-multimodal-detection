mem = "/home/constant/CodeBuddy/Dev/AIC/.workbuddy/memory/2026-09-26.md"
note = """

## 09-26 04:45 用户追问「和纯rgb比呢」+ 排队配对 bootstrap
- 用户要的是 vitl800 三阶段管线内 纯 RGB vs 融合的对比。纯 RGB 对应 Stage A（vitl800_scratch_stageA_rgb/best_stg1.pth，同 ViT-L/800，forward 内 ir/depth 置零）。
- 同协议口径（同 val200、测试集口径）：A 0.4990 vs C 0.5265@ep35 → 融合 +2.75pt。
- 跨模型历史（midboost，§5.2）：l4-纯RGB +0.0244 [0.0002,0.0535] 勉强显著；l4u3-纯RGB +0.0285 [-0.0011,0.0534] 跨 0 → 融合增益量级 +2.4~2.9pt 但并非铁证。
- 口径风险：Stage A 训练用「白天优先」增强（与 C 原配方刻意不同），+2.75pt 含部分 recipe 差异；val200 sd≈2pt，单点差 ~1.4x 噪声。
- 已排队 run_rgb_vs_fuse_ci.sh（task Gqm2hJ）：等 ARXw4T 让 GPU → dump 纯 RGB dets → 与 Stage C best/last/swa 在 _dets_rgb_vs_fuse/ 做聚焦配对 bootstrap → outputs/metrics/val_ci_rgb_vs_fuse.json。两侧 val 指向同一 instances_val.json + 同一 valid/images + 同 Normalize，干净受控。
"""
with open(mem, "a", encoding="utf-8") as f:
    f.write(note)
print("appended", len(note), "bytes")
