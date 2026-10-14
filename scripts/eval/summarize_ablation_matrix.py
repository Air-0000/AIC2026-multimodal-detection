#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_ablation_matrix.py - 自动解析并汇总 2x2 全因子消融实验 + Adaptive MAL 结果
"""

import os
import re
import json

REPO = "/home/constant/CodeBuddy/Dev/AIC"
LOG_DIR = os.path.join(REPO, "logs")

EXPERIMENTS = [
    {
        "key": "ft",
        "name": "1. 纯微调基线 (ft: 无 DCN, MAL=1)",
        "dcn": "No",
        "mal": "1.0",
        "log": os.path.join(LOG_DIR, "dfine_x_ft_eval_day.log"),
    },
    {
        "key": "dcn",
        "name": "2. DCNv2 形变卷积 (dcn: 有 DCN, MAL=1)",
        "dcn": "Yes (stages 2,3)",
        "mal": "1.0",
        "log": os.path.join(LOG_DIR, "pair_dcn47_orig.log"),
    },
    {
        "key": "puremal",
        "name": "3. 纯 MAL2 消融 (puremal: 无 DCN, MAL=2)",
        "dcn": "No",
        "mal": "2.0",
        "log": os.path.join(LOG_DIR, "dfine_x_puremal_eval_orig.log"),
    },
    {
        "key": "dcn_mal2",
        "name": "4. DCN + MAL2 结合 (mal2: 有 DCN, MAL=2)",
        "dcn": "Yes (stages 2,3)",
        "mal": "2.0",
        "log": os.path.join(LOG_DIR, "dfine_x_mal2_eval_orig.log"),
    },
    {
        "key": "dcn_adpmal",
        "name": "5. DCN + 自适应 MAL (adpmal: 有 DCN, 类别特异 MAL)",
        "dcn": "Yes (stages 2,3)",
        "mal": "Class-Adaptive (1.0~2.0)",
        "log": os.path.join(LOG_DIR, "dfine_x_adpmal_eval_orig.log"),
    },
]

def parse_log(path):
    if not os.path.exists(path):
        return None
    content = open(path).read()
    
    m_ap = re.search(r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0\.50:0\.95\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)", content)
    m_50 = re.search(r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0\.50\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)", content)
    m_75 = re.search(r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0\.75\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)", content)
    m_s = re.search(r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0\.50:0\.95\s+\|\s+area=\s+small\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)", content)
    m_m = re.search(r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0\.50:0\.95\s+\|\s+area=medium\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)", content)
    m_l = re.search(r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0\.50:0\.95\s+\|\s+area=\s+large\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)", content)
    
    # per-category
    per_cat = {}
    for line in content.split("\n"):
        m_cat = re.match(r"^\s+([a-zA-Z\s]+?)\s+AP=([\d\.]+)\s+AP50=([\d\.]+)", line)
        if m_cat:
            cname = m_cat.group(1).strip()
            per_cat[cname] = (float(m_cat.group(2)), float(m_cat.group(3)))
            
    if not m_ap:
        return None
    return {
        "mAP": float(m_ap.group(1)),
        "AP50": float(m_50.group(1)) if m_50 else None,
        "AP75": float(m_75.group(1)) if m_75 else None,
        "APs": float(m_s.group(1)) if m_s else None,
        "APm": float(m_m.group(1)) if m_m else None,
        "APl": float(m_l.group(1)) if m_l else None,
        "per_cat": per_cat
    }

def main():
    results = {}
    for exp in EXPERIMENTS:
        data = parse_log(exp["log"])
        results[exp["key"]] = {
            "name": exp["name"],
            "dcn": exp["dcn"],
            "mal": exp["mal"],
            "data": data
        }
        
    out_json = os.path.join(REPO, "work_dir/ablation_mal_dcn_matrix.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 95)
    print("AIC 2026 核心消融矩阵: [DCN] x [MAL] 对比分析")
    print("=" * 95)
    header = f"{'模型/实验':<35} | {'DCN':<10} | {'MAL权重':<12} | {'mAP@50:95':<10} | {'AP50':<8} | {'AP75':<8}"
    print(header)
    print("-" * 95)
    for exp in EXPERIMENTS:
        r = results[exp["key"]]
        d = r["data"]
        if d:
            print(f"{r['name']:<35} | {r['dcn']:<10} | {r['mal']:<12} | {d['mAP']:<10.4f} | {d['AP50']:<8.4f} | {d['AP75']:<8.4f}")
        else:
            print(f"{r['name']:<35} | {r['dcn']:<10} | {r['mal']:<12} | {'(训练/排队中)':<10} | {'-':<8} | {'-':<8}")
    print("=" * 95)

if __name__ == "__main__":
    main()
