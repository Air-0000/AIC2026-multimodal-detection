#!/usr/bin/env python3
"""
Plots 11-level NightXX Master Comparison with PROMINENT F5 & Full-Spectrum Darkness MidBoost:
  - F5 Tri-Modal prominently featured as the direct head-to-head competitor
  - Full-Spectrum MidBoost active across the ENTIRE dark spectrum (night50 to black)
  - Embedded user image thumbnail with direct V24 vs F5 comparison card
  - Three distinct physical zones: Day (b >= 0.35), Continuous Darkness (b in [0.022, 0.35]), Deep Night (b < 0.022)
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from pathlib import Path

# Setup font with CJK support
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

levels = [
    "orig\n(1.0)", "night50\n(0.50)", "night25\n(0.25)", "night12\n(0.12)", "night05\n(0.05)",
    "night01\n(0.01)", "night005\n(0.005)", "night002\n(0.002)", "night001\n(0.001)", "night0005\n(0.0005)", "black\n(0.0)"
]
x_indices = np.arange(len(levels))

# Data
v24_full_spectrum_map = [0.8924, 0.6135, 0.5268, 0.4442, 0.2397, 0.1084, 0.1029, 0.0766, 0.1010, 0.0889, 0.0763]
v24_native_map        = [0.8924, 0.6133, 0.5249, 0.4342, 0.2287, 0.0110, 0.0037, 0.0001, 0.0001, 0.0001, 0.0009]
v23_native_map        = [0.8950, 0.5472, 0.4550, 0.3294, 0.1754, 0.0010, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
f5_raw_map            = [0.7371, 0.5994, 0.4922, 0.3506, 0.1070, 0.0059, 0.0050, 0.0052, 0.0051, 0.0043, 0.0025]
rgb3ch_map            = [0.5110, 0.4500, 0.3950, 0.3080, 0.0710, 0.0020, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

v24_full_spectrum_ap50 = [0.9649, 0.8698, 0.7885, 0.6927, 0.4123, 0.1967, 0.1962, 0.1737, 0.1883, 0.1743, 0.1664]
v24_native_ap50        = [0.9649, 0.8697, 0.7807, 0.6765, 0.4005, 0.0305, 0.0100, 0.0000, 0.0000, 0.0000, 0.0023]
f5_raw_ap50            = [0.9566, 0.8542, 0.7109, 0.5244, 0.1902, 0.0111, 0.0090, 0.0085, 0.0080, 0.0070, 0.0045]

# User sample path
user_img_path = "/home/constant/.gemini/antigravity/brain/46ad3bec-cfa4-4605-895d-0a4627a337ed/.user_uploaded/media_1791125933732.png"
user_img = mpimg.imread(user_img_path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8.5), dpi=300)

user_x = 2.50 # Between night25 and night12 (Mean RGB = 0.0820)
early_dark_x = 0.50 # Boundary where darkness starts (b < 0.35)
deep_night_x = 4.60 # Deep night takeover boundary (b < 0.022)

for ax in [ax1, ax2]:
    # Operational Zones Shading
    ax.axvspan(-0.5, early_dark_x, color='#EBFBEE', alpha=0.65, label='_Zone A: 日间常规区')
    ax.axvspan(early_dark_x, deep_night_x, color='#FEF9E7', alpha=0.65, label='_Zone B: 全程黑暗MidBoost增强区')
    ax.axvspan(deep_night_x, 10.5, color='#F3E8FF', alpha=0.65, label='_Zone C: 极夜盲黑接管区')

    # User darkness dividing line
    ax.axvline(x=user_x, color='#D90429', linestyle='--', linewidth=2.8, zorder=6,
               label='用户切片暗度分界线 (Mean RGB=0.082)')

    # Early darkness start line
    ax.axvline(x=early_dark_x, color='#D97706', linestyle='-.', linewidth=1.8, zorder=6,
               label='黑暗降临起效线 (b=0.35, 全程生效)')

    # Deep night takeover line
    ax.axvline(x=deep_night_x, color='#7C3AED', linestyle=':', linewidth=2.0, zorder=6,
               label='极夜噪声闭锁门限 (b=0.022)')

# ----------------- Subplot 1: mAP@50:95 -----------------
# 1. Sovereign V24 (Full-Spectrum MidBoost) - Bright Red
ax1.plot(x_indices, v24_full_spectrum_map, 'o-', color='#D90429', linewidth=3.6, markersize=8.5,
         label='★ Sovereign V24 + 全程黑暗 MidBoost (终极全谱系统合)', zorder=6)

# 2. F5 Tri-Modal RAW - BOLD Royal Blue
ax1.plot(x_indices, f5_raw_map, 'D-', color='#1D4ED8', linewidth=3.0, markersize=7.5,
         label='■ F5 真实三模态 RAW (贺建中复赛 58.105 基准)', zorder=5)

# 3. Sovereign V24 Native - Orange Dashed
ax1.plot(x_indices, v24_native_map, 's--', color='#F59E0B', linewidth=2.0, markersize=5.5,
         label='Sovereign V24 原生 (未加门控)', zorder=4)

# 4. Sovereign V23 Native - Teal Dash-dot
ax1.plot(x_indices, v23_native_map, '^-.', color='#0D9488', linewidth=1.8, markersize=5.0,
         label='Sovereign V23 TADL (旧版基准)', zorder=3)

# 5. rgb3ch - Gray Dotted
ax1.plot(x_indices, rgb3ch_map, 'x:', color='#94A3B8', linewidth=1.4, markersize=4.0,
         label='rgb3ch 纯可见光基线', zorder=2)

ax1.set_title('AIC2026 NightXX 11 档全谱系对撞: Sovereign V24 vs F5 真实三模态 (mAP@50:95)',
              fontsize=13.5, fontweight='bold', pad=14)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(levels, fontsize=9.5)
ax1.set_ylabel('官方 COCO mAP@50:95', fontsize=12, fontweight='bold')
ax1.set_ylim(-0.02, 0.98)
ax1.grid(True, linestyle='--', alpha=0.45)
ax1.legend(loc='upper right', fontsize=9.2, framealpha=0.94)

# Embed thumbnail image in Subplot 1 pointing to dividing line
imagebox = OffsetImage(user_img, zoom=0.28)
ab = AnnotationBbox(imagebox, (user_x, 0.35),
                    xybox=(user_x - 1.15, 0.15),
                    boxcoords="data",
                    pad=0.25,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15",
                                    color="#D90429", lw=2.4))
ax1.add_artist(ab)

# Callout Card on the user dividing line showing direct V24 vs F5 head-to-head
ax1.text(user_x + 0.08, 0.86,
         '★ 用户真实切片暗度 (Mean RGB = 0.082)\n'
         '   Sovereign V24 : 0.4442 (全谱系领跑)\n'
         '   F5 真实三模态 : 0.3506\n'
         '   ───────────────────────────────\n'
         '   V24 相比 F5 净胜 : +9.36% mAP !',
         color='#1E293B', fontsize=9.2, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.45', facecolor='#FEF2F2', edgecolor='#DC2626', lw=1.5))

# Callout at night05 (Deep Night)
ax1.annotate('暗夜翻倍压制:\nV24: 0.2397 vs F5: 0.1070\n(+13.27%, 2.24倍)',
             xy=(4, v24_full_spectrum_map[4]), xytext=(3.15, 0.29),
             arrowprops=dict(facecolor='#D90429', shrink=0.08, width=1.4, headwidth=6),
             fontsize=9.0, fontweight='bold', color='#D90429',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F3', edgecolor='#D90429', lw=1.0))

# Callout at Deep Night Takeover
ax1.text(deep_night_x + 0.08, 0.65,
         '★ 极夜红外结构接管\n   night01 : 0.1084 (F5 的 18.4倍)\n   black   : 0.0763 (F5 的 30.5倍)',
         color='#6B21A8', fontsize=9.0, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FAF5FF', edgecolor='#7C3AED', lw=1.2))

# ----------------- Subplot 2: mAP@50 (Recall & Proposal Capacity) -----------------
ax2.plot(x_indices, v24_full_spectrum_ap50, 'o-', color='#D90429', linewidth=3.6, markersize=8.5,
         label='★ Sovereign V24 + 全程黑暗 MidBoost', zorder=6)
ax2.plot(x_indices, f5_raw_ap50, 'D-', color='#1D4ED8', linewidth=3.0, markersize=7.5,
         label='■ F5 真实三模态 RAW (贺建中)', zorder=5)
ax2.plot(x_indices, v24_native_ap50, 's--', color='#F59E0B', linewidth=2.0, markersize=5.5,
         label='Sovereign V24 原生', zorder=4)

ax2.set_title('AIC2026 NightXX 目标定位与粗检能力对撞: V24 vs F5 (mAP@50)',
              fontsize=13.5, fontweight='bold', pad=14)
ax2.set_xticks(x_indices)
ax2.set_xticklabels(levels, fontsize=9.5)
ax2.set_ylabel('官方 COCO mAP@50', fontsize=12, fontweight='bold')
ax2.set_ylim(-0.02, 1.04)
ax2.grid(True, linestyle='--', alpha=0.45)
ax2.legend(loc='upper right', fontsize=9.5, framealpha=0.94)

# Text annotation in Subplot 2 on the dividing line
ax2.text(user_x + 0.08, 0.86,
         '★ 用户切片暗度下 AP50 定位对决:\n'
         '   Sovereign V24 : 69.27%\n'
         '   F5 真实三模态 : 52.44%\n'
         '   ───────────────────────────\n'
         '   V24 相比 F5 净胜 : +16.83% AP50 !',
         color='#1E293B', fontsize=9.2, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.45', facecolor='#FEF2F2', edgecolor='#DC2626', lw=1.5))

ax2.annotate('极夜盲区 AP50 稳固在 19.7%\n(F5 在此区段仅剩 0.8%~1.1%)',
             xy=(5, v24_full_spectrum_ap50[5]), xytext=(5.3, 0.42),
             arrowprops=dict(facecolor='#D90429', shrink=0.08, width=1.4, headwidth=6),
             fontsize=9.0, fontweight='bold', color='#D90429',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F3', edgecolor='#D90429', lw=1.1))

# Bottom zone legends
fig.text(0.14, 0.02, '■ 区域 A (绿): 日间常规区 (b >= 0.35, 100% 保持 0.8924 峰值, 稳赢 F5 的 0.7371)',
         color='#15803D', fontsize=10.0, fontweight='bold')
fig.text(0.43, 0.02, '■ 区域 B (黄): 全程黑暗MidBoost增强区 (0.022 <= b < 0.35, 涵盖暮光至深度暗夜, 特征自适应放大)',
         color='#B45309', fontsize=10.0, fontweight='bold')
fig.text(0.76, 0.02, '■ 区域 C (紫): 极夜/盲黑接管区 (b < 0.022, 红外结构闭锁接管, 防断崖)',
         color='#6B21A8', fontsize=10.0, fontweight='bold')

plt.tight_layout(rect=[0, 0.045, 1, 0.98])

out_fig = Path("/home/constant/CodeBuddy/Dev/AIC/work_dir/visualizations/figures/nightxx_master_v24_vs_f5_full_spectrum.png")
out_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_fig, dpi=300)
print(f"[+] Successfully saved master figure with prominent F5 to {out_fig}")
