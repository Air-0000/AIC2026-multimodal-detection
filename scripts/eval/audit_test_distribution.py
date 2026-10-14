#!/usr/bin/env python3
"""
Test Set vs Train/Val Distribution Audit Script
Analyzes image resolution, illumination, thermal dynamics, depth validity,
and cross-modal alignment offset between Visible and Infrared.
"""

import os
import glob
import json
from PIL import Image
import numpy as np

def audit():
    test_root = "/root/autodl-tmp/AIC/dataset/test/AIC2026_PHASE_2_1000"
    vis_dir = os.path.join(test_root, "visible")
    ir_dir = os.path.join(test_root, "infrared")
    ir_aligned_dir = "/root/autodl-tmp/AIC/work_dir/test_aligned_ir"
    depth_dir = os.path.join(test_root, "depth_p0")

    vis_files = sorted(glob.glob(os.path.join(vis_dir, "*.jpg")) + glob.glob(os.path.join(vis_dir, "*.png")))
    total_imgs = len(vis_files)
    print(f"Total test images found: {total_imgs}")

    resolutions = {}
    lum_list = []
    ir_mean_list = []
    ir_std_list = []
    ir_high_list = []
    depth_valid_list = []
    water_scene_count = 0

    for idx, p in enumerate(vis_files):
        stem = os.path.splitext(os.path.basename(p))[0]
        with Image.open(p) as img:
            sz = img.size
            resolutions[sz] = resolutions.get(sz, 0) + 1
            arr_vis = np.array(img.convert("L"), dtype=np.float32)
            lum_list.append(arr_vis.mean())

        ir_p = os.path.join(ir_dir, stem + ".jpg")
        if not os.path.exists(ir_p): ir_p = os.path.join(ir_dir, stem + ".png")
        if os.path.exists(ir_p):
            with Image.open(ir_p) as img:
                arr_ir = np.array(img.convert("L"), dtype=np.float32)
                ir_mean_list.append(arr_ir.mean())
                ir_std_list.append(arr_ir.std())
                ir_high_list.append((arr_ir > 240).mean())

        dp_p = os.path.join(depth_dir, stem + ".png")
        if os.path.exists(dp_p):
            with Image.open(dp_p) as img:
                arr_d = np.array(img)
                depth_valid_list.append((arr_d > 0).mean())
                # Water scene check (bottom half depth std)
                h = arr_d.shape[0]
                bottom_d = arr_d[int(h * 0.5):, :]
                valid_d = bottom_d[bottom_d > 0]
                if len(valid_d) > 0 and (valid_d.std() / (65535.0 if arr_d.max() > 255 else 255.0)) > 0.09:
                    water_scene_count += 1

    lum_arr = np.array(lum_list)
    ir_mean_arr = np.array(ir_mean_list)
    ir_std_arr = np.array(ir_std_list)
    ir_high_arr = np.array(ir_high_list)
    depth_valid_arr = np.array(depth_valid_list)

    print("\n" + "="*50)
    print("      AIC 2026 Phase 2 Test Set Profile (1000 Images)")
    print("="*50)

    print("\n1. 分辨率分布 (Resolutions):")
    for sz, count in sorted(resolutions.items(), key=lambda x: -x[1]):
        print(f"   {sz[0]}x{sz[1]} ({sz[0]/sz[1]:.2f}:1): {count} 张 ({count/total_imgs*100:.1f}%)")

    print("\n2. 光照环境与昼夜分布 (Illumination):")
    night = (lum_arr < 35).sum()
    low = ((lum_arr >= 35) & (lum_arr < 75)).sum()
    day = (lum_arr >= 75).sum()
    print(f"   极暗夜景 (Luminance < 35):     {night} 张 ({night/total_imgs*100:.1f}%)")
    print(f"   黄昏/弱光 (35 <= Lum < 75):    {low} 张 ({low/total_imgs*100:.1f}%)")
    print(f"   日间/正常 (Luminance >= 75):   {day} 张 ({day/total_imgs*100:.1f}%)")
    print(f"   平均可见光亮度:                {lum_arr.mean():.1f} (std: {lum_arr.std():.1f})")

    print("\n3. 红外物理特性 (Infrared Dynamics):")
    print(f"   平均红外灰度值:                {ir_mean_arr.mean():.1f} (std: {ir_mean_arr.std():.1f})")
    print(f"   红外对比度 (平均内部方差):     {ir_std_arr.mean():.1f}")
    print(f"   高温热斑饱和像素占比 (>240):   {ir_high_arr.mean()*100:.2f}%")

    print("\n4. 深度图特性 (Depth p0):")
    print(f"   有效深度像素覆盖率:            {depth_valid_arr.mean()*100:.1f}%")
    print(f"   判定为水面场景数 (Water Prior):{water_scene_count} 张 ({water_scene_count/total_imgs*100:.1f}%)")

    # Alignment audit
    align_summary_p = "/root/autodl-tmp/AIC/work_dir/alignment_summary_test_aligned_ir.json"
    if os.path.exists(align_summary_p):
        with open(align_summary_p) as f:
            data = json.load(f)
        print("\n5. 多模态视场配准审计 (Cross-Modal Alignment):")
        print(f"   全局几何失准样本 (Global Misalignment):  {data.get('global_applied_count')} 张 ({data.get('global_applied_rate')*100:.1f}%)")
        print(f"   局部运动目标漂移 (Local Displaced):     {data.get('local_applied_count')} 张图像，共 {data.get('total_local_targets')} 个目标")
        print(f"   配准后平均结构相关度增益 (Delta CC):    +{data.get('avg_delta_cc'):.4f}")

if __name__ == "__main__":
    audit()
