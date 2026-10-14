#!/usr/bin/env python3
"""
Sovereign V31 Ablation Showdown Analyzer:
Compares Baseline (Sovereign V30, purely 1-to-1) vs Sovereign V31 (Co-ATSS One-to-Many Auxiliary Supervision).
"""

import os
import sys
import json
import re

ROOT = "/home/constant/CodeBuddy/Dev/AIC"

V30_LOG = "outputs/sovereign_v30_cssp_v29peak_2560/log.txt"
V31_LOG = "outputs/sovereign_v31_ablation_probe_2560/log.txt"
V30_CONSOLE = "outputs/sovereign_v30_cssp_v29peak_2560/train_console.log"
V31_CONSOLE = "outputs/sovereign_v31_ablation_probe_2560/train_console.log"

CATEGORIES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage can", "uav", "tricycle"
]


def parse_log_txt(path):
    epochs_data = {}
    if not os.path.exists(path):
        return epochs_data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                ep = d.get("epoch")
                bbox = d.get("test_coco_eval_bbox")
                if ep is not None and bbox:
                    epochs_data[ep] = {
                        "mAP": bbox[0],
                        "AP50": bbox[1],
                        "AP75": bbox[2],
                        "AP_s": bbox[3],
                        "AP_m": bbox[4],
                        "AP_l": bbox[5],
                        "train_loss": d.get("train_loss", 0.0),
                    }
            except Exception:
                pass
    return epochs_data


def parse_per_category(console_path):
    cat_by_epoch = {}
    if not os.path.exists(console_path):
        return cat_by_epoch
    with open(console_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    blocks = re.findall(r"Epoch:\s*\[(\d+)\].*?per-category \(area=all, maxDets=100\):(.*?)(?:best_stat:|Epoch:|$)", text, re.DOTALL)
    for ep_str, blk in blocks:
        ep = int(ep_str)
        cats = {}
        for line in blk.strip().split("\n"):
            line = line.strip()
            for cat in CATEGORIES:
                if line.startswith(cat):
                    # parse AP=... AP50=...
                    m = re.search(r"AP=([\d\.]+)\s+AP50=([\d\.]+)", line)
                    if m:
                        cats[cat] = (float(m.group(1)), float(m.group(2)))
        if cats:
            cat_by_epoch[ep] = cats
    return cat_by_epoch


def generate_report():
    v30_ep = parse_log_txt(V30_LOG)
    v31_ep = parse_log_txt(V31_LOG)
    v30_cats = parse_per_category(V30_CONSOLE)
    v31_cats = parse_per_category(V31_CONSOLE)

    print("=" * 80)
    print("=== Sovereign V31 Controlled Ablation Showdown Report ===")
    print("=" * 80)
    print("Baseline (Control):      Sovereign V30 (Pure 1-to-1 Hungarian Matching)")
    print("Experiment (Test):       Sovereign V31 (Co-ATSS One-to-Many Auxiliary Supervision)")
    print("Strict Parity:           Exact same seed=42, 2.5K (1440x2560), same lr, same base checkpoint")
    print("-" * 80)

    all_eps = sorted(list(set(v30_ep.keys()) | set(v31_ep.keys())))
    if not all_eps:
        print("[-] No completed evaluation epochs found yet.")
        return

    print(f"{'Epoch':<6} | {'Metric':<8} | {'V30 Baseline':<14} | {'V31 Co-ATSS':<14} | {'Delta':<10}")
    print("-" * 65)

    for ep in all_eps:
        d30 = v30_ep.get(ep, {})
        d31 = v31_ep.get(ep, {})

        for m in ["mAP", "AP50", "AP75", "AP_s"]:
            val30 = d30.get(m, None)
            val31 = d31.get(m, None)
            s30 = f"{val30:.4f}" if val30 is not None else "N/A"
            s31 = f"{val31:.4f}" if val31 is not None else "In Prog"
            if val30 is not None and val31 is not None:
                delta = val31 - val30
                s_del = f"{delta:+.4f}"
            else:
                s_del = "-"
            print(f"{ep:<6} | {m:<8} | {s30:<14} | {s31:<14} | {s_del:<10}")
        print("-" * 65)

        # Per category for UAV and Tricycle
        c30 = v30_cats.get(ep, {})
        c31 = v31_cats.get(ep, {})
        for cat in ["uav", "tricycle", "sign", "ball"]:
            if cat in c30 or cat in c31:
                u30_ap, u30_ap50 = c30.get(cat, (None, None))
                u31_ap, u31_ap50 = c31.get(cat, (None, None))
                s30 = f"{u30_ap:.3f} / {u30_ap50*100:.1f}%" if u30_ap is not None else "N/A"
                s31 = f"{u31_ap:.3f} / {u31_ap50*100:.1f}%" if u31_ap is not None else "In Prog"
                print(f"  {cat:<12} (AP/AP50) | {s30:<14} | {s31:<14} |")
        print("=" * 80)


if __name__ == "__main__":
    generate_report()
