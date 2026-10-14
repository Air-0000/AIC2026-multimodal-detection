#!/usr/bin/env python3
"""Parses Day + 9 Night evaluation logs for HGNetv2ViTDualStream and prints summary table."""
import os
import re

LOG_DIR = "/home/constant/CodeBuddy/Dev/AIC/logs"
LEVELS = [
    ("day", "Day (1.0x)"),
    ("night50", "Night 0.50x"),
    ("night25", "Night 0.25x"),
    ("night12", "Night 0.12x"),
    ("night05", "Night 0.05x"),
    ("night01", "Night 0.01x"),
    ("night005", "Night 0.005x"),
    ("night002", "Night 0.002x"),
    ("night001", "Night 0.001x"),
    ("night0005", "Night 0.0005x"),
]

def parse_log(filepath):
    if not os.path.exists(filepath):
        return None, None, None
    ap, ap50, ap75 = None, None, None
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]" in line:
                m = re.search(r"=\s*([0-9\.]+)", line)
                if m:
                    ap = float(m.group(1))
            elif "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]" in line:
                m = re.search(r"=\s*([0-9\.]+)", line)
                if m:
                    ap50 = float(m.group(1))
            elif "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]" in line:
                m = re.search(r"=\s*([0-9\.]+)", line)
                if m:
                    ap75 = float(m.group(1))
    return ap, ap50, ap75

def main():
    print("\n" + "=" * 70)
    print("=== HGNetv2 + ViT Dual-Stream Evaluation Matrix (Day + 9 Night) ===")
    print("=" * 70)
    print(f"{'Condition':<18} | {'AP (0.50:0.95)':<15} | {'AP50':<10} | {'AP75':<10}")
    print("-" * 70)

    aps = []
    for tag, desc in LEVELS:
        log_path = os.path.join(LOG_DIR, f"dfine_x_vit_dualstream_eval_{tag}.log")
        ap, ap50, ap75 = parse_log(log_path)
        if ap is not None:
            aps.append(ap)
            print(f"{desc:<18} | {ap:<15.4f} | {ap50:<10.4f} | {ap75:<10.4f}")
        else:
            print(f"{desc:<18} | {'Pending/Missing':<15} | {'-':<10} | {'-':<10}")

    print("-" * 70)
    if aps:
        print(f"Mean mAP across all tested levels: {sum(aps) / len(aps):.4f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
