#!/usr/bin/env python3
"""
Plots 11-level NightXX comparison:
  1. Sovereign V24 + Calibrated MidBoost (New King)
  2. Sovereign V24 Native (TADL Robustness)
  3. Sovereign V23 Native (Daytime Baseline)
  4. F5 Real Tri-Modal RAW (Historical Baseline)
  5. rgb3ch (Pure RGB Baseline)
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Setup font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

levels = [
    "orig", "night50", "night25", "night12", "night05",
    "night01", "night005", "night002", "night001", "night0005", "black"
]
x_indices = np.arange(len(levels))

# Data
v24_midboost_map = [0.8924, 0.6133, 0.5249, 0.4342, 0.2293, 0.0923, 0.0839, 0.0796, 0.0762, 0.0830, 0.0834]
v24_native_map   = [0.8924, 0.6133, 0.5249, 0.4342, 0.2287, 0.0110, 0.0037, 0.0001, 0.0001, 0.0001, 0.0009]
v23_native_map   = [0.8950, 0.5472, 0.4550, 0.3294, 0.1754, 0.0010, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
f5_raw_map       = [0.7371, 0.5994, 0.4922, 0.3506, 0.1070, 0.0059, 0.0050, 0.0052, 0.0051, 0.0043, 0.0025]
rgb3ch_map       = [0.5110, 0.4500, 0.3950, 0.3080, 0.0710, 0.0020, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

v24_midboost_ap50 = [0.9649, 0.8697, 0.7807, 0.6765, 0.3988, 0.1768, 0.1648, 0.1738, 0.1673, 0.1660, 0.1716]
v24_native_ap50   = [0.9649, 0.8697, 0.7807, 0.6765, 0.4005, 0.0305, 0.0100, 0.0000, 0.0000, 0.0000, 0.0023]
f5_raw_ap50       = [0.9566, 0.8542, 0.7109, 0.5244, 0.1902, 0.0111, 0.0090, 0.0085, 0.0080, 0.0070, 0.0045]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

# Subplot 1: mAP@50:95
ax1.plot(x_indices, v24_midboost_map, 'o-', color='#D90429', linewidth=3.2, markersize=8, label='Sovereign V24 + Calibrated MidBoost (Ours)', zorder=5)
ax1.plot(x_indices, v24_native_map, 's--', color='#FF9F1C', linewidth=2.2, markersize=6, label='Sovereign V24 Native (No Gating)', zorder=4)
ax1.plot(x_indices, v23_native_map, '^-.', color='#2EC4B6', linewidth=1.8, markersize=5, label='Sovereign V23 TADL (Baseline)', zorder=3)
ax1.plot(x_indices, f5_raw_map, 'd:', color='#3A86FF', linewidth=2.0, markersize=6, label='F5 Tri-Modal RAW (He Jianzhong)', zorder=2)
ax1.plot(x_indices, rgb3ch_map, 'x:', color='#8D99AE', linewidth=1.5, markersize=4, label='rgb3ch Pure Visible Baseline', zorder=1)

ax1.set_title('AIC2026 NightXX 11-Level Robustness (mAP@50:95)', fontsize=14, fontweight='bold', pad=12)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(levels, rotation=35, ha='right', fontsize=10)
ax1.set_ylabel('COCO mAP@50:95', fontsize=12, fontweight='bold')
ax1.set_ylim(-0.02, 0.95)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)

# Annotate the dramatic jump in pitch black
ax1.annotate('Emergency Thermal Takeover\n0.0009 -> 0.0834 (92.7x)',
             xy=(10, v24_midboost_map[-1]), xytext=(7.5, 0.18),
             arrowprops=dict(facecolor='#D90429', shrink=0.08, width=1.5, headwidth=7),
             fontsize=10, fontweight='bold', color='#D90429',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F3', edgecolor='#D90429', lw=1.2))

# Subplot 2: mAP@50
ax2.plot(x_indices, v24_midboost_ap50, 'o-', color='#D90429', linewidth=3.2, markersize=8, label='Sovereign V24 + Calibrated MidBoost', zorder=5)
ax2.plot(x_indices, v24_native_ap50, 's--', color='#FF9F1C', linewidth=2.2, markersize=6, label='Sovereign V24 Native', zorder=4)
ax2.plot(x_indices, f5_raw_ap50, 'd:', color='#3A86FF', linewidth=2.0, markersize=6, label='F5 Tri-Modal RAW', zorder=2)

ax2.set_title('AIC2026 NightXX Auxiliary Metric (mAP@50)', fontsize=14, fontweight='bold', pad=12)
ax2.set_xticks(x_indices)
ax2.set_xticklabels(levels, rotation=35, ha='right', fontsize=10)
ax2.set_ylabel('mAP@50', fontsize=12, fontweight='bold')
ax2.set_ylim(-0.02, 1.02)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)

ax2.annotate('AP50 Rescued to 17.2%\nin Pitch Black',
             xy=(10, v24_midboost_ap50[-1]), xytext=(7.8, 0.30),
             arrowprops=dict(facecolor='#D90429', shrink=0.08, width=1.5, headwidth=7),
             fontsize=10, fontweight='bold', color='#D90429',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F3', edgecolor='#D90429', lw=1.2))

plt.tight_layout()
out_fig = Path("/home/constant/CodeBuddy/Dev/AIC/work_dir/visualizations/figures/nightxx_v24_midboost_breakthrough_comparison.png")
out_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_fig, dpi=300)
print(f"[+] Saved breakthrough comparison figure to {out_fig}")
