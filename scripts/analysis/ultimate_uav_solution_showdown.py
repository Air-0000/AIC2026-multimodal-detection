#!/usr/bin/env python3
"""
Ultimate Head-to-Head Showdown:
Comparing All Solutions for Sovereign V29:
1. F5 Benchmark (58.105 baseline)
2. Cycle 5 Peak Cap60 (Baseline model)
3. Cycle 5 Peak + Fusion 1C (免训练方案)
4. Cycle 8 Aligned Cap60 (微调方案 - 模型原生)
5. Cycle 8 Aligned + Fusion 1C (微调 + 融合集大成)
"""

import sys
import os
import torch
import math

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, ROOT)
from scripts.analysis.analyze_f5_overlap_sweep import parse_zip, to_xyxy, iou_xyxy, match_predictions
from scripts.analysis.benchmark_uav_calibration_variants import get_variant_1c

f5_zip = os.path.join(ROOT, "work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip")
f5_data = parse_zip(f5_zip)

c5_raw = torch.load(os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/raw_detections_v29_c5_peak.pt"), weights_only=False)
c8_raw = torch.load(os.path.join(ROOT, "work_dir/submissions/v29_cycle8_align/raw_detections.pt"), weights_only=False)

c5_cap60 = parse_zip(os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/submission_sovereign_v29_c5_peak_cap60.zip"))
c5_1c = parse_zip(os.path.join(ROOT, "work_dir/submissions/v29_cycle5_peak/submission_v29_c5_fusion_dual_gate_thr35.zip"))
c8_cap60 = parse_zip(os.path.join(ROOT, "work_dir/submissions/v29_cycle8_align/submission_sovereign_v29_c5_peak_cap60.zip"))
c8_1c = get_variant_1c(c8_raw, uav_small_thr=0.35)

models = [
    ("1. F5 Official Benchmark", f5_data),
    ("2. V29 Cycle 5 Peak Cap60", c5_cap60),
    ("3. V29 Cycle 5 + Fusion 1C (免训练)", c5_1c),
    ("4. V29 Cycle 8 Aligned Cap60 (微调)", c8_cap60),
    ("5. V29 Cycle 8 + Fusion 1C (微调+融合)", c8_1c),
]

print("=" * 115)
print(" HEAD-TO-HEAD ULTIMATE SHOWDOWN: Cycle 5 vs Cycle 8 vs Fusion 1C vs F5 Benchmark ".center(115, "="))
print("=" * 115)
header = f"{'Model Variant':<38} | {'Total':<6} | {'Consensus':<9} | {'F5 Recall':<10} | {'Cand Prec':<10} | {'UAV Recall (out of 123)'}"
print(header)
print("-" * 115)

for name, d in models:
    if "F5" in name:
        print(f"{name:<38} | 7097   | 7097      |    100.0% |    100.0% | 123/123 (100.0%) [Total: 123]")
        continue
    m = match_predictions(f5_data, d, iou_thr=0.50)
    u_f5 = m["per_class_ref"][10]
    u_mat = m["per_class_matched"][10]
    u_tot = m["per_class_cand"][10]
    print(f"{name:<38} | {m['total_cand']:<6} | {m['matched_same']:<9} | {m['recall_ref']:>8.2f}% | {m['precision_cand']:>8.2f}% | {u_mat}/{u_f5} ({u_mat/u_f5*100:.1f}%) [Total: {u_tot}]")

print("=" * 115)

# Per-test image forensics on 00000511, 00000573, 00000637, 00000648
test_stems = ["00000511", "00000573", "00000637", "00000648"]
print("\n" + "=" * 115)
print(" FORENSICS ON CITATION IMAGES: 00000511, 00000573, 00000637, 00000648 ".center(115, "="))
print("=" * 115)

for s in test_stems:
    print(f"\n--- Image {s} ---")
    f5_u = [(round(b[5], 3), round(b[1], 3), round(b[2], 3), round(b[3], 3), round(b[4], 3)) for b in f5_data.get(s, []) if b[0] == 10]
    c5_u = [(round(b[5], 3), round(b[1], 3), round(b[2], 3), round(b[3], 3), round(b[4], 3)) for b in c5_cap60.get(s, []) if b[0] == 10]
    c5_1c_u = [(round(b[5], 3), round(b[1], 3), round(b[2], 3), round(b[3], 3), round(b[4], 3)) for b in c5_1c.get(s, []) if b[0] == 10]
    c8_u = [(round(b[5], 3), round(b[1], 3), round(b[2], 3), round(b[3], 3), round(b[4], 3)) for b in c8_cap60.get(s, []) if b[0] == 10]
    c8_1c_u = [(round(b[5], 3), round(b[1], 3), round(b[2], 3), round(b[3], 3), round(b[4], 3)) for b in c8_1c.get(s, []) if b[0] == 10]
    
    print(f"  F5 Benchmark ({len(f5_u)}):       {f5_u}")
    print(f"  C5 Peak Cap60 ({len(c5_u)}):      {c5_u}")
    print(f"  C5 + Fusion 1C ({len(c5_1c_u)}):     {c5_1c_u}")
    print(f"  C8 Aligned Cap60 ({len(c8_u)}):   {c8_u}")
    print(f"  C8 + Fusion 1C ({len(c8_1c_u)}):     {c8_1c_u}")
