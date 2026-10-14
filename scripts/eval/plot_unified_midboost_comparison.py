#!/usr/bin/env python3
"""
Plots 11-level NightXX Master Comparison:
  1. Sovereign V24 + Unified MidBoost (Early-Dark Booster + Pitch-Black Takeover)
  2. Sovereign V24 Native (No Gating)
  3. Sovereign V23 Native (Old Baseline)
  4. F5 Tri-Modal RAW (He Jianzhong Baseline)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

levels = [
    "orig", "night50", "night25", "night12", "night05",
    "night01", "night005", "night002", "night001", "night0005", "black"
]
x_indices = np.arange(len(levels))

# Data
v24_unified_map = [0.8924, 0.6135, 0.5268, 0.4373, 0.2397, 0.1084, 0.1029, 0.0766, 0.1010, 0.0889, 0.0763]
v24_native_map  = [0.8924, 0.6133, 0.5249, 0.4342, 0.2287, 0.0110, 0.0037, 0.0001, 0.0001, 0.0001, 0.0009]
v23_native_map  = [0.8950, 0.5472, 0.4550, 0.3294, 0.1754, 0.0010, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
f5_raw_map      = [0.7371, 0.5994, 0.4922, 0.3506, 0.1070, 0.0059, 0.0050, 0.0052, 0.0051, 0.0043, 0.0025]

v24_unified_ap50 = [0.9649, 0.8698, 0.7885, 0.6819, 0.4123, 0.1967, 0.1962, 0.1737, 0.1883, 0.1743, 0.1664]
v24_native_ap50  = [0.9649, 0.8697, 0.7807, 0.6765, 0.4005, 0.0305, 0.0100, 0.0000, 0.0000, 0.0000, 0.0023]
f5_raw_ap50      = [0.9566, 0.8542, 0.7109, 0.5244, 0.1902, 0.0111, 0.0090, 0.0085, 0.0080, 0.0070, 0.0045]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

# Subplot 1: mAP@50:95
ax1.plot(x_indices, v24_unified_map, 'o-', color='#D90429', linewidth=3.4, markersize=8, label='Sovereign V24 + Unified MidBoost (All-Range King)', zorder=5)
ax1.plot(x_indices, v24_native_map, 's--', color='#FF9F1C', linewidth=2.2, markersize=6, label='Sovereign V24 Native (No Gating)', zorder=4)
ax1.plot(x_indices, v23_native_map, '^-.', color='#2EC4B6', linewidth=1.8, markersize=5, label='Sovereign V23 TADL (Baseline)', zorder=3)
ax1.plot(x_indices, f5_raw_map, 'd:', color='#3A86FF', linewidth=2.0, markersize=6, label='F5 Tri-Modal RAW (He Jianzhong)', zorder=2)

ax1.set_title('AIC2026 NightXX 11-Level Robustness (mAP@50:95)', fontsize=14, fontweight='bold', pad=12)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(levels, rotation=35, ha='right', fontsize=10)
ax1.set_ylabel('COCO mAP@50:95', fontsize=12, fontweight='bold')
ax1.set_ylim(-0.02, 0.95)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)

# Annotation for early darkness gain
ax1.annotate('Early Dark Boost:\nnight05: 0.2287 -> 0.2397\nnight12: 0.4342 -> 0.4373',
             xy=(4, v24_unified_map[4]), xytext=(2.6, 0.33),
             arrowprops=dict(facecolor='#D90429', shrink=0.08, width=1.4, headwidth=6),
             fontsize=9.5, fontweight='bold', color='#D90429',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F3', edgecolor='#D90429', lw=1.2))

# Annotation for pitch black breakthrough
ax1.annotate('Emergency Thermal Takeover:\nnight01: 0.0110 -> 0.1084 (9.8x)\nblack: 0.0009 -> 0.0763 (85x)',
             xy=(5, v24_unified_map[5]), xytext=(5.6, 0.20),
             arrowprops=dict(facecolor='#D90429', shrink=0.08, width=1.4, headwidth=6),
             fontsize=9.5, fontweight='bold', color='#D90429',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F3', edgecolor='#D90429', lw=1.2))

# Subplot 2: mAP@50
ax2.plot(x_indices, v24_unified_ap50, 'o-', color='#D90429', linewidth=3.4, markersize=8, label='Sovereign V24 + Unified MidBoost', zorder=5)
ax2.plot(x_indices, v24_native_ap50, 's--', color='#FF9F1C', linewidth=2.2, markersize=6, label='Sovereign V24 Native', zorder=4)
ax2.plot(x_indices, f5_raw_ap50, 'd:', color='#3A86FF', linewidth=2.0, markersize=6, label='F5 Tri-Modal RAW', zorder=2)

ax2.set_title('AIC2026 NightXX Auxiliary Metric (mAP@50)', fontsize=14, fontweight='bold', pad=12)
ax2.set_xticks(x_indices)
ax2.set_xticklabels(levels, rotation=35, ha='right', fontsize=10)
ax2.set_ylabel('mAP@50', fontsize=12, fontweight='bold')
ax2.set_ylim(-0.02, 1.02)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)

ax2.annotate('AP50 Reaches 19.7%\nin Deep Night',
             xy=(5, v24_unified_ap50[5]), xytext=(4.5, 0.32),
             arrowprops=dict(facecolor='#D90429', shrink=0.08, width=1.4, headwidth=6),
             fontsize=9.5, fontweight='bold', color='#D90429',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F3', edgecolor='#D90429', lw=1.2))

plt.tight_layout()
out_fig = Path("/home/constant/CodeBuddy/Dev/AIC/work_dir/visualizations/figures/nightxx_v24_unified_midboost_master_comparison.png")
out_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_fig, dpi=300)
print(f"[+] Saved master comparison figure to {out_fig}")
