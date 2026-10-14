#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_training_curves.py - 解析并绘制多组消融模型的 Epoch 训练评测指标折线图
对比模型:
1. ft: 纯微调基线 (无 DCN, MAL=1)
2. dcn: DCNv2 形变卷积 (stages 2,3, MAL=1)
3. dcn + mal2: DCN + 全局加倍 MAL (MAL=2)
4. dcn + adpmal: DCN + 类别自适应 MAL (当前训练中)
"""

import os
import re
import shutil
import matplotlib.pyplot as plt
import numpy as np

REPO = "/home/constant/CodeBuddy/Dev/AIC"
LOG_DIR = os.path.join(REPO, "logs")
OUT_DIR = os.path.join(REPO, "work_dir/visualizations/figures")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    {
        "id": "ft",
        "name": "ft (Baseline: No DCN, MAL=1.0)",
        "log": os.path.join(LOG_DIR, "dfine_x_ft_train.log"),
        "color": "#7f7f7f",
        "linestyle": "--",
        "linewidth": 2.0,
        "marker": "o",
        "ms": 4,
    },
    {
        "id": "dcn",
        "name": "dcn (DCNv2 stages 2,3, MAL=1.0)",
        "log": os.path.join(LOG_DIR, "dfine_x_dcn_train.log"),
        "color": "#1f77b4",
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "s",
        "ms": 4,
    },
    {
        "id": "mal2",
        "name": "dcn + mal2 (Global MAL=2.0)",
        "log": os.path.join(LOG_DIR, "dfine_x_mal2_train.log"),
        "color": "#ff7f0e",
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "^",
        "ms": 4,
    },
    {
        "id": "adpmal",
        "name": "dcn + adpmal (Class-Adaptive MAL)",
        "log": os.path.join(LOG_DIR, "dfine_x_adpmal_train.log"),
        "color": "#d62728",
        "linestyle": "-",
        "linewidth": 2.5,
        "marker": "*",
        "ms": 6,
    },
]

def parse_full_log(path):
    if not os.path.exists(path):
        return None
    
    with open(path) as f:
        text = f.read()

    # 按 Test 块划分
    blocks = text.split("Test: Total time:")
    data = {
        "epochs": [],
        "mAP": [],
        "AP50": [],
        "AP75": [],
        "APs": [],
        "APm": [],
        "APl": [],
        "per_cat": {}
    }

    cur_epoch = 0
    lines = text.split("\n")
    
    # 逐行扫描保持同步
    i = 0
    while i < len(lines):
        line = lines[i]
        m_ep = re.search(r"Epoch:\s*\[(\d+)\]", line)
        if m_ep:
            cur_epoch = int(m_ep.group(1))

        if "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]" in line:
            m_val = re.search(r"=\s*([\d\.]+)$", line.strip())
            if m_val:
                data["epochs"].append(cur_epoch)
                data["mAP"].append(float(m_val.group(1)))
                
                # 连续抓取接下来的指标
                for j in range(i+1, min(i+15, len(lines))):
                    ln = lines[j]
                    if "Average Precision  (AP) @[ IoU=0.50      | area=   all" in ln:
                        m_50 = re.search(r"=\s*([\d\.]+)$", ln.strip())
                        if m_50: data["AP50"].append(float(m_50.group(1)))
                    elif "Average Precision  (AP) @[ IoU=0.75      | area=   all" in ln:
                        m_75 = re.search(r"=\s*([\d\.]+)$", ln.strip())
                        if m_75: data["AP75"].append(float(m_75.group(1)))
                    elif "area= small | maxDets=100" in ln and "Average Precision" in ln:
                        m_s = re.search(r"=\s*([\d\.]+)$", ln.strip())
                        if m_s: data["APs"].append(float(m_s.group(1)))
                    elif "area=medium | maxDets=100" in ln and "Average Precision" in ln:
                        m_m = re.search(r"=\s*([\d\.]+)$", ln.strip())
                        if m_m: data["APm"].append(float(m_m.group(1)))
                    elif "area= large | maxDets=100" in ln and "Average Precision" in ln:
                        m_l = re.search(r"=\s*([\d\.]+)$", ln.strip())
                        if m_l: data["APl"].append(float(m_l.group(1)))
                
                # 抓取每类指标
                for k in range(i+1, min(i+40, len(lines))):
                    ln = lines[k]
                    m_cat = re.match(r"^\s+([a-zA-Z\s]+?)\s+AP=([\d\.]+)\s+AP50=([\d\.]+)", ln)
                    if m_cat:
                        cname = m_cat.group(1).strip()
                        cap = float(m_cat.group(2))
                        if cname not in data["per_cat"]:
                            data["per_cat"][cname] = []
                        # 确保与当前 epoch 对齐
                        if len(data["per_cat"][cname]) < len(data["epochs"]):
                            data["per_cat"][cname].append(cap)
        i += 1
    return data

def main():
    parsed_models = {}
    for m in MODELS:
        print(f"Parsing log: {m['id']} -> {m['log']}")
        d = parse_full_log(m["log"])
        if d and len(d["epochs"]) > 0:
            parsed_models[m["id"]] = (m, d)
            print(f"  Loaded {len(d['epochs'])} points. Epochs {d['epochs'][0]}~{d['epochs'][-1]}, Best mAP: {max(d['mAP']):.4f}")

    # 绘图配置
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200)
    fig.patch.set_facecolor("#fafafa")

    # Subplot 1: mAP@50:95
    ax1 = axes[0, 0]
    ax1.set_title("Overall Validation mAP@50:95 across Epochs", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("mAP@50:95", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: AP50
    ax2 = axes[0, 1]
    ax2.set_title("Overall Validation AP50 across Epochs", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("AP50", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: boat AP
    ax3 = axes[1, 0]
    ax3.set_title("Vulnerable Class: 'boat' AP (DCN Boost vs MAL2 Suppression vs AdpMAL Rescue)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Boat AP@50:95", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: ball AP
    ax4 = axes[1, 1]
    ax4.set_title("Frequent FP Class: 'ball' AP (False Positive Glare Suppression)", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Epoch", fontsize=11)
    ax4.set_ylabel("Ball AP@50:95", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.5)

    for mid, (m, d) in parsed_models.items():
        eps = d["epochs"]
        
        # 1. mAP
        if len(d["mAP"]) == len(eps):
            ax1.plot(eps, d["mAP"], label=m["name"], color=m["color"],
                     linestyle=m["linestyle"], linewidth=m["linewidth"],
                     marker=m["marker"], markersize=m["ms"], alpha=0.9)
            # 标注最高点
            best_idx = int(np.argmax(d["mAP"]))
            ax1.scatter([eps[best_idx]], [d["mAP"][best_idx]], color=m["color"], s=60, zorder=5)
            ax1.annotate(f"{d['mAP'][best_idx]:.3f}", (eps[best_idx], d["mAP"][best_idx]),
                         textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, fontweight="bold", color=m["color"])

        # 2. AP50
        if len(d["AP50"]) == len(eps):
            ax2.plot(eps, d["AP50"], label=m["name"], color=m["color"],
                     linestyle=m["linestyle"], linewidth=m["linewidth"],
                     marker=m["marker"], markersize=m["ms"], alpha=0.9)

        # 3. boat
        if "boat" in d["per_cat"] and len(d["per_cat"]["boat"]) == len(eps):
            ax3.plot(eps, d["per_cat"]["boat"], label=m["name"], color=m["color"],
                     linestyle=m["linestyle"], linewidth=m["linewidth"],
                     marker=m["marker"], markersize=m["ms"], alpha=0.9)

        # 4. ball
        if "ball" in d["per_cat"] and len(d["per_cat"]["ball"]) == len(eps):
            ax4.plot(eps, d["per_cat"]["ball"], label=m["name"], color=m["color"],
                     linestyle=m["linestyle"], linewidth=m["linewidth"],
                     marker=m["marker"], markersize=m["ms"], alpha=0.9)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(loc="lower right" if ax != ax3 else "upper left", fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc")
        ax.set_xlim(-1, 51)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "ablation_mAP_training_curves.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {out_png}")

    # 拷贝到脑图/artifact目录以便前端直显
    artifact_dir = "/home/constant/.gemini/antigravity/brain/6131717e-d78a-4d2b-947e-c99ae4e56221"
    os.makedirs(artifact_dir, exist_ok=True)
    dst_png = os.path.join(artifact_dir, "ablation_mAP_training_curves.png")
    shutil.copyfile(out_png, dst_png)
    print(f"Copied to artifact dir: {dst_png}")

if __name__ == "__main__":
    main()
