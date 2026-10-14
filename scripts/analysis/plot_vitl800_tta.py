#!/usr/bin/env python3
"""Plot vitl800 (polluted best_stg1.pth) night inference: no-TTA vs TTA.

Reads:
  outputs/metrics/vitl800_best_night_infer.txt        (no-TTA)
  outputs/metrics/vitl800_best_night_infer_tta.txt    (TTA: scales 640,800 + hflip, NMS-merged)
Writes outputs/vitl800_night_tta_compare.png
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/vitl800_night_tta_compare.png"
NOTTA = "outputs/metrics/vitl800_best_night_infer.txt"
TTA = "outputs/metrics/vitl800_best_night_infer_tta.txt"

ORDER = ["orig", "night0005", "night001", "night002", "night005",
         "night01", "night05", "night12", "night25", "night50"]


def parse(path):
    d = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*(\S+)\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
            if m and m.group(1) in ORDER:
                d[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return d


def main():
    nt = parse(NOTTA)
    tt = parse(TTA)
    conds = [c for c in ORDER if c in nt and c in tt]
    if not conds:
        print("missing data; nt=%s tt=%s" % (list(nt), list(tt)))
        return
    ap_nt = [nt[c][0] for c in conds]
    ap_tt = [tt[c][0] for c in conds]
    ap50_nt = [nt[c][1] for c in conds]
    ap50_tt = [tt[c][1] for c in conds]

    x = range(len(conds))
    w = 0.2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax1.bar([i - w for i in x], ap_nt, w, label="no-TTA", color="#4C72B0")
    ax1.bar([i + w for i in x], ap_tt, w, label="TTA (640,800+flip)", color="#DD8452")
    ax1.set_ylabel("AP @ IoU=0.50:0.95")
    ax1.set_title("vitl800 (polluted best) — night inference: no-TTA vs TTA")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(ap_nt, ap_tt)):
        ax1.text(i - w, a + 0.005, f"{a:.3f}", ha="center", fontsize=7)
        ax1.text(i + w, b + 0.005, f"{b:.3f}", ha="center", fontsize=7)

    ax2.bar([i - w for i in x], ap50_nt, w, label="no-TTA", color="#4C72B0")
    ax2.bar([i + w for i in x], ap50_tt, w, label="TTA", color="#DD8452")
    ax2.set_ylabel("AP50")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(conds, rotation=30, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(ap50_nt, ap50_tt)):
        ax2.text(i - w, a + 0.005, f"{a:.3f}", ha="center", fontsize=7)
        ax2.text(i + w, b + 0.005, f"{b:.3f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("saved", OUT)

    # delta summary
    print("\ncond        AP(noTTA)  AP(TTA)   ΔAP    AP50(noTTA) AP50(TTA)  ΔAP50")
    for c in conds:
        d_ap = tt[c][0] - nt[c][0]
        d_50 = tt[c][1] - nt[c][1]
        print(f"{c:11s} {nt[c][0]:.3f}     {tt[c][0]:.3f}   {d_ap:+.3f}   "
              f"{nt[c][1]:.3f}      {tt[c][1]:.3f}    {d_50:+.3f}")


if __name__ == "__main__":
    main()
