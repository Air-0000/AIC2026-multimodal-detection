#!/usr/bin/env python3
"""
Plots 11-level NightXX comparison with User Darkness Level Dividing Line:
  - User uploaded night scene: Mean RGB = 0.0820 (~ Night12 - Night25 real urban night)
  - Clear dividing line marking this real-world night baseline
  - Embedded image thumbnail with bounding boxes
  - Highlighted operational zones: Day/Dusk, Early Darkness (MidBoost active), Deep Night Takeover
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
v24_unified_map = [0.8924, 0.6135, 0.5268, 0.4373, 0.2397, 0.1084, 0.1029, 0.0766, 0.1010, 0.0889, 0.0763]
v24_native_map  = [0.8924, 0.6133, 0.5249, 0.4342, 0.2287, 0.0110, 0.0037, 0.0001, 0.0001, 0.0001, 0.0009]
v23_native_map  = [0.8950, 0.5472, 0.4550, 0.3294, 0.1754, 0.0010, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
f5_raw_map      = [0.7371, 0.5994, 0.4922, 0.3506, 0.1070, 0.0059, 0.0050, 0.0052, 0.0051, 0.0043, 0.0025]
rgb3ch_map      = [0.5110, 0.4500, 0.3950, 0.3080, 0.0710, 0.0020, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

v24_unified_ap50 = [0.9649, 0.8698, 0.7885, 0.6819, 0.4123, 0.1967, 0.1962, 0.1737, 0.1883, 0.1743, 0.1664]
v24_native_ap50  = [0.9649, 0.8697, 0.7807, 0.6765, 0.4005, 0.0305, 0.0100, 0.0000, 0.0000, 0.0000, 0.0023]
f5_raw_ap50      = [0.9566, 0.8542, 0.7109, 0.5244, 0.1902, 0.0111, 0.0090, 0.0085, 0.0080, 0.0070, 0.0045]

# User sample path
user_img_path = "/home/constant/.gemini/antigravity/brain/46ad3bec-cfa4-4605-895d-0a4627a337ed/.user_uploaded/media_1791125933732.png"
user_img = mpimg.imread(user_img_path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)

user_x = 2.50 # Between night25 and night12, matching Mean RGB = 0.0820
takeover_x = 4.60 # Deep night threshold b < 0.022

for ax in [ax1, ax2]:
    # Operational Zones Shading
    ax.axvspan(-0.5, 1.5, color='#E8F8F5', alpha=0.6, label='_Zone A: 日间与暮光常规区')
    ax.axvspan(1.5, takeover_x, color='#FFF9E6', alpha=0.6, label='_Zone B: 早期夜间/路灯增强区 (MidBoost Aux活跃)')
    ax.axvspan(takeover_x, 10.5, color='#F5EEF8', alpha=0.6, label='_Zone C: 极夜/盲黑应急接管区 (Thermal Takeover)')

    # User darkness dividing line
    ax.axvline(x=user_x, color='#D90429', linestyle='--', linewidth=2.8, zorder=6,
               label='用户实测暗度分界线 (Mean RGB=0.082)')

    # Deep night takeover line
    ax.axvline(x=takeover_x, color='#7209B7', linestyle=':', linewidth=2.0, zorder=6,
               label='极夜噪声闭锁门限 (b=0.022)')

# ----------------- Subplot 1: mAP@50:95 -----------------
ax1.plot(x_indices, v24_unified_map, 'o-', color='#D90429', linewidth=3.4, markersize=8, label='Sovereign V24 + 统合 MidBoost (全谱系王者)', zorder=5)
ax1.plot(x_indices, v24_native_map, 's--', color='#FF9F1C', linewidth=2.2, markersize=6, label='Sovereign V24 原生 (未加门控)', zorder=4)
ax1.plot(x_indices, v23_native_map, '^-.', color='#2EC4B6', linewidth=1.8, markersize=5, label='Sovereign V23 TADL (旧版基准)', zorder=3)
ax1.plot(x_indices, f5_raw_map, 'd:', color='#3A86FF', linewidth=2.0, markersize=6, label='F5 真实三模态 RAW (贺建中)', zorder=2)
ax1.plot(x_indices, rgb3ch_map, 'x:', color='#8D99AE', linewidth=1.4, markersize=4, label='rgb3ch 纯可见光基准', zorder=1)

ax1.set_title('AIC2026 NightXX 11 档全谱系鲁棒性与暗度分界线 (mAP@50:95)', fontsize=13, fontweight='bold', pad=14)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(levels, fontsize=9.5)
ax1.set_ylabel('官方 COCO mAP@50:95', fontsize=12, fontweight='bold')
ax1.set_ylim(-0.02, 0.98)
ax1.grid(True, linestyle='--', alpha=0.45)
ax1.legend(loc='upper right', fontsize=9.2, framealpha=0.92)

# Embed thumbnail image in Subplot 1 pointing to dividing line
imagebox = OffsetImage(user_img, zoom=0.28)
ab = AnnotationBbox(imagebox, (user_x, 0.35),
                    xybox=(user_x - 1.15, 0.16),
                    boxcoords="data",
                    pad=0.25,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15",
                                    color="#D90429", lw=2.2))
ax1.add_artist(ab)

# Text annotation on the dividing line
ax1.text(user_x + 0.08, 0.88, '★ 用户上传暗度分界线\n   (真实城市路灯/阴影夜景)\n   Mean RGB = 0.0820\n   介于 night25 与 night12 之间',
         color='#D90429', fontsize=9.5, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F3', edgecolor='#D90429', lw=1.3))

ax1.text(takeover_x + 0.08, 0.65, '★ 极夜闭锁门限\n   光子死灭/噪声突变\n   红外结构应急接管',
         color='#7209B7', fontsize=9, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5EEF8', edgecolor='#7209B7', lw=1.1))

# Annotate Early Darkness Lift
ax1.annotate('早期黑暗 Aux 增益区\nnight12: 0.4342 -> 0.4373\nnight05: 0.2287 -> 0.2397',
             xy=(4, v24_unified_map[4]), xytext=(2.9, 0.28),
             arrowprops=dict(facecolor='#D90429', shrink=0.08, width=1.4, headwidth=6),
             fontsize=9, fontweight='bold', color='#D90429',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F3', edgecolor='#D90429', lw=1.0))

# ----------------- Subplot 2: mAP@50 -----------------
ax2.plot(x_indices, v24_unified_ap50, 'o-', color='#D90429', linewidth=3.4, markersize=8, label='Sovereign V24 + 统合 MidBoost', zorder=5)
ax2.plot(x_indices, v24_native_ap50, 's--', color='#FF9F1C', linewidth=2.2, markersize=6, label='Sovereign V24 原生', zorder=4)
ax2.plot(x_indices, f5_raw_ap50, 'd:', color='#3A86FF', linewidth=2.0, markersize=6, label='F5 真实三模态 RAW', zorder=2)

ax2.set_title('AIC2026 NightXX 辅助指标与暗度分界线 (mAP@50)', fontsize=13, fontweight='bold', pad=14)
ax2.set_xticks(x_indices)
ax2.set_xticklabels(levels, fontsize=9.5)
ax2.set_ylabel('mAP@50 (定位召回能力)', fontsize=12, fontweight='bold')
ax2.set_ylim(-0.02, 1.04)
ax2.grid(True, linestyle='--', alpha=0.45)
ax2.legend(loc='upper right', fontsize=9.5, framealpha=0.92)

# Text annotation on the dividing line in Subplot 2
ax2.text(user_x + 0.08, 0.88, '★ 此暗度下 V24 保持 68.2% AP50\n   远超 F5 (52.4%) 与旧版 V23 (32.9%)',
         color='#D90429', fontsize=9.5, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F3', edgecolor='#D90429', lw=1.3))

ax2.annotate('极暗自愈区:\nAP50 达到 19.7%\n全黑稳定在 16.6%',
             xy=(5, v24_unified_ap50[5]), xytext=(5.6, 0.40),
             arrowprops=dict(facecolor='#D90429', shrink=0.08, width=1.4, headwidth=6),
             fontsize=9, fontweight='bold', color='#D90429',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F3', edgecolor='#D90429', lw=1.1))

# Bottom zone legends
fig.text(0.18, 0.02, '■ 区域 A (绿): 日间/暮光常规区 (光学主导)', color='#0B6E4F', fontsize=10.5, fontweight='bold')
fig.text(0.42, 0.02, '■ 区域 B (黄): 早期夜间增强区 (当前图片所在暗度, MidBoost Aux 放大起效)', color='#B7791F', fontsize=10.5, fontweight='bold')
fig.text(0.74, 0.02, '■ 区域 C (紫): 极夜/全黑应急接管区 (红外闭锁防断崖)', color='#6B21A8', fontsize=10.5, fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 0.98])

out_fig = Path("/home/constant/CodeBuddy/Dev/AIC/work_dir/visualizations/figures/nightxx_user_darkness_threshold_master.png")
out_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_fig, dpi=300)
print(f"[+] Successfully saved figure with user darkness threshold line to {out_fig}")
