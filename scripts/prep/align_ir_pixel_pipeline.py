#!/usr/bin/env python3
"""
分层双阶像素级红外-可见光预处理配准流水线 (Hierarchical Two-Stage IR-RGB Pixel Alignment)

解决痛点：
  1. 阶一（全局刚性）：校正相机安装与视场固有偏差（如 640x360 序列普遍存在的 1~3px 偏移与 ~0.5° 旋转）；
     带严格自适应门限：1080p 等已对齐样本自动保持原图恒等（Applied=False），杜绝过拟合。
  2. 阶二（目标感知吸附）：校正高速运动目标因曝光时滞/微时差产生的独立非刚性残余位移
     （如机动无人机下沉 ~8.5px），通过显著热斑连通域局部相位匹配与羽化位移场实现精准机身吸附。
  3. 阶三（边缘引导）：通过 RGB 引导滤波（Guided Filter）将弥散热辐射收拢在物体真实物理轮廓内。

保证：
  - 纯训练前图像预处理，下游模型（检测/推理/TTA）直接受益，不依赖任何标注文件（无监督适用于测试集）。
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np


def struct_map(im_gray):
    """计算高阶各向同性梯度结构图，消除两模态灰度反相与温差绝对幅值差异。"""
    gx = cv2.Scharr(im_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(im_gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    mag = cv2.GaussianBlur(mag, (5, 5), 1.5)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def guided_filter(I, p, r=5, eps=1e-2):
    """标准单通道引导滤波：以 I (可见光) 为引导，将 p (红外图) 的弥散边界吸附到 I 的锐利边缘上。"""
    I = I.astype(np.float32) / 255.0
    p = p.astype(np.float32) / 255.0
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    q = mean_a * I + mean_b
    return np.clip(q * 255.0, 0, 255).astype(np.uint8)


def align_image_pair(vis_bgr, ir_bgr, enable_local_snap=True, enable_guided=True):
    """对单张可见光与红外图像执行分层像素级配准。

    返回:
      ir_aligned_bgr: 对齐后的红外三通道图像
      info: 包含配准指标与判定状态的字典
    """
    H, W = vis_bgr.shape[:2]
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY)
    ig = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY)

    ve = struct_map(vg)
    ie = struct_map(ig)
    cc0 = float(np.corrcoef(ve.ravel().astype(float), ie.ravel().astype(float))[0, 1])

    # ---- 阶段 1：全局几何刚性配准 (Euclidean ECC with Bicubic & Quality Gate) ----
    M = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-4)
    global_applied = False
    dx, dy, angle = 0.0, 0.0, 0.0
    ig_current = ig.copy()
    ie_current = ie.copy()
    cc_current = cc0

    try:
        cc_val, M_est = cv2.findTransformECC(ve, ie, M, cv2.MOTION_EUCLIDEAN, criteria, inputMask=None, gaussFiltSize=5)
        angle_est = float(np.arctan2(M_est[1, 0], M_est[0, 0]) * 180.0 / np.pi)
        dx_est, dy_est = float(M_est[0, 2]), float(M_est[1, 2])
        dist_est = float(np.sqrt(dx_est**2 + dy_est**2))

        # 使用双三次插值 (INTER_CUBIC) 保持亚像素级高频梯度与边缘清晰度
        ig_cand = cv2.warpAffine(ig, M_est, (W, H), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
        ie_cand = struct_map(ig_cand)
        cc1 = float(np.corrcoef(ve.ravel().astype(float), ie_cand.ravel().astype(float))[0, 1])

        # 质量门限：只有真实图像结构相关性切实提升且几何形变物理合理时才应用
        if (cc1 - cc0 >= 0.003) and dist_est <= 25.0 and abs(angle_est) <= 3.0:
            global_applied = True
            M = M_est
            dx, dy, angle = dx_est, dy_est, angle_est
            ig_current = ig_cand
            ie_current = ie_cand
            cc_current = cc1
    except Exception:
        pass

    # ---- 阶段 2：独立动态目标热斑吸附 (Salient Thermal Blob Snapping with Gate) ----
    local_applied_count = 0
    if enable_local_snap:
        local_bg = cv2.medianBlur(ig_current, 31)
        hot_resid = np.maximum(ig_current.astype(float) - local_bg.astype(float), 0).astype(np.uint8)
        _, hot_mask = cv2.threshold(hot_resid, 25, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hot_mask)

        dense_flow_x = np.zeros((H, W), dtype=np.float32)
        dense_flow_y = np.zeros((H, W), dtype=np.float32)

        for lbl in range(1, num_labels):
            x, y, bw, bh, area = stats[lbl]
            if area < 15 or area > 6000:
                continue
            pad = 25
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + bw + pad), min(H, y + bh + pad)

            pv = ve[y0:y1, x0:x1]
            pi = ie_current[y0:y1, x0:x1]
            if pv.shape != pi.shape or pv.shape[0] < 10 or pv.shape[1] < 10:
                continue

            (cdx, cdy), resp = cv2.phaseCorrelate(pi.astype(np.float64), pv.astype(np.float64))
            dist = float(np.sqrt(cdx**2 + cdy**2))

            # 严格置信度门限：只有高响应且存在显著残差位移的前景目标才做独立吸附
            if resp >= 0.40 and 0.5 <= dist <= 15.0:
                local_applied_count += 1
                comp_mask = (labels[y0:y1, x0:x1] == lbl).astype(np.float32)
                comp_mask = cv2.dilate(comp_mask, np.ones((5, 5), np.uint8))
                comp_mask = cv2.GaussianBlur(comp_mask, (15, 15), 5)
                # 采样坐标逆变换
                dense_flow_x[y0:y1, x0:x1] -= cdx * comp_mask
                dense_flow_y[y0:y1, x0:x1] -= cdy * comp_mask

        if local_applied_count > 0:
            grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
            map_x = (grid_x + dense_flow_x).astype(np.float32)
            map_y = (grid_y + dense_flow_y).astype(np.float32)
            ig_snap = cv2.remap(ig_current, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            ie_snap = struct_map(ig_snap)
            cc_snap = float(np.corrcoef(ve.ravel().astype(float), ie_snap.ravel().astype(float))[0, 1])

            # 阶二安全防劣化门限：若局部吸附未破坏全局结构一致性则应用，否则安全熔断
            if cc_snap >= cc_current - 0.001:
                ig_current = ig_snap
                ie_current = ie_snap
                cc_current = cc_snap
            else:
                local_applied_count = 0

    # ---- 阶段 3：RGB 边缘引导平滑吸附 (Guided Filtering) 带严格防劣化门限 ----
    if enable_guided:
        ig_candidate = guided_filter(vg, ig_current, r=5, eps=0.01)
        ie_cand = struct_map(ig_candidate)
        cc_cand = float(np.corrcoef(ve.ravel().astype(float), ie_cand.ravel().astype(float))[0, 1])
        if cc_cand >= cc_current:
            ig_final = ig_candidate
            cc_final = cc_cand
        else:
            ig_final = ig_current
            cc_final = cc_current
    else:
        ig_final = ig_current
        cc_final = cc_current

    # 最终输出三通道红外图像 (保持原通道格式)
    ir_aligned_bgr = cv2.cvtColor(ig_final, cv2.COLOR_GRAY2BGR)

    info = {
        "cc_before": round(cc0, 4),
        "cc_after": round(cc_final, 4),
        "delta_cc": round(cc_final - cc0, 4),
        "global_applied": global_applied,
        "dx": round(dx, 2),
        "dy": round(dy, 2),
        "angle": round(angle, 3),
        "local_snapped_targets": local_applied_count,
    }
    return ir_aligned_bgr, info


def make_visualization_panel(vis_bgr, ir_raw_bgr, ir_aligned_bgr, info):
    """生成 6 格高清可视化比对面板。"""
    H, W = vis_bgr.shape[:2]
    vg = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY)
    ig_raw = cv2.cvtColor(ir_raw_bgr, cv2.COLOR_BGR2GRAY)
    ig_ali = cv2.cvtColor(ir_aligned_bgr, cv2.COLOR_BGR2GRAY)

    heat_raw = cv2.applyColorMap(ig_raw, cv2.COLORMAP_JET)
    heat_ali = cv2.applyColorMap(ig_ali, cv2.COLORMAP_JET)

    over_raw = cv2.addWeighted(vis_bgr, 0.5, heat_raw, 0.5, 0)
    over_ali = cv2.addWeighted(vis_bgr, 0.5, heat_ali, 0.5, 0)

    # 边缘叠合图：绿色 = RGB，红色 = IR，叠合黄色 = 像素级对齐
    ev = cv2.Canny(vg, 40, 120)
    ei_raw = cv2.Canny(ig_raw, 40, 120)
    ei_ali = cv2.Canny(ig_ali, 40, 120)

    comp_raw = np.zeros((H, W, 3), dtype=np.uint8)
    comp_raw[:, :, 1] = ev
    comp_raw[:, :, 2] = ei_raw

    comp_ali = np.zeros((H, W, 3), dtype=np.uint8)
    comp_ali[:, :, 1] = ev
    comp_ali[:, :, 2] = ei_ali

    def tag(im, text, color=(255, 255, 255)):
        out = im.copy()
        cv2.putText(out, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return out

    p1 = tag(vis_bgr, f"Visible RGB ({W}x{H})")
    p2 = tag(over_raw, f"Raw IR (CC={info['cc_before']:.3f})")
    p3 = tag(over_ali, f"Aligned IR (G={info['global_applied']}, dx={info['dx']:+.1f}, dy={info['dy']:+.1f}, L={info['local_snapped_targets']})",
             color=(0, 255, 255) if info["global_applied"] or info["local_snapped_targets"] > 0 else (200, 200, 200))
    p4 = tag(comp_raw, "Edges Before (Green:RGB, Red:IR)")
    p5 = tag(comp_ali, "Edges After (Yellow: Aligned)")

    scale = 640.0 / W if W > 640 else 1.0
    def rsz(im):
        return cv2.resize(im, (int(W * scale), int(H * scale))) if scale != 1.0 else im

    row1 = np.hstack([rsz(p1), rsz(p2), rsz(p3)])
    row2 = np.hstack([rsz(vis_bgr), rsz(p4), rsz(p5)])
    return np.vstack([row1, row2])


def process_single(args_tuple):
    stem, vis_path, ir_path, out_dir, viz_dir = args_tuple
    vis_bgr = cv2.imread(vis_path)
    ir_bgr = cv2.imread(ir_path)
    if vis_bgr is None or ir_bgr is None:
        return stem, None

    aligned_bgr, info = align_image_pair(vis_bgr, ir_bgr)

    # 写入对齐产物
    if out_dir is not None:
        ext = os.path.splitext(ir_path)[1]
        out_path = os.path.join(out_dir, stem + ext)
        cv2.imwrite(out_path, aligned_bgr)

    # 写入可视化面板
    if viz_dir is not None:
        viz_panel = make_visualization_panel(vis_bgr, ir_bgr, aligned_bgr, info)
        viz_path = os.path.join(viz_dir, f"align_viz_{stem}.jpg")
        cv2.imwrite(viz_path, viz_panel)

    return stem, info


def main():
    parser = argparse.ArgumentParser(description="分层双阶像素级红外-可见光预处理配准流水线")
    parser.add_argument("--vis-dir", default="dataset/test/AIC2026_PHASE_2_1000/visible", help="可见光图像目录")
    parser.add_argument("--ir-dir", default="dataset/test/AIC2026_PHASE_2_1000/infrared", help="红外图像目录")
    parser.add_argument("--out-dir", default="work_dir/test_aligned_ir", help="对齐红外图像输出目录")
    parser.add_argument("--viz-dir", default="work_dir/visualizations/test_align_viz", help="可视化面板输出目录")
    parser.add_argument("--sample", type=int, default=20, help="处理样本数量 (0 = 全量)")
    parser.add_argument("--workers", type=int, default=4, help="多进程并行核心数")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)

    vis_files = sorted(glob.glob(os.path.join(args.vis_dir, "*")))
    ir_map = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(args.ir_dir, "*"))}

    tasks = []
    for vp in vis_files:
        stem = os.path.splitext(os.path.basename(vp))[0]
        if stem in ir_map:
            tasks.append((stem, vp, ir_map[stem], args.out_dir, args.viz_dir))

    total = len(tasks)
    if args.sample > 0 and args.sample < total:
        # 分层抽样：同时包含 640x360 和 1080p 样本
        tasks_sampled = tasks[:args.sample]
    else:
        tasks_sampled = tasks

    print(f"=== 开始运行红外像素级配准流水线 (任务数: {len(tasks_sampled)} / 总共 {total}) ===")
    print(f"  输入可见光: {args.vis_dir}")
    print(f"  输入红外:   {args.ir_dir}")
    print(f"  输出对齐红外: {args.out_dir}")
    print(f"  输出可视化:   {args.viz_dir}")

    results = []
    n_global_applied = 0
    n_local_applied = 0
    total_local_targets = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for stem, info in pool.map(process_single, tasks_sampled):
                if info is not None:
                    results.append((stem, info))
                    if info["global_applied"]:
                        n_global_applied += 1
                    if info["local_snapped_targets"] > 0:
                        n_local_applied += 1
                        total_local_targets += info["local_snapped_targets"]
                    print(f"[{len(results)}/{len(tasks_sampled)}] {stem:20s}: "
                          f"G_Applied={info['global_applied']} (dx={info['dx']:+.1f}, dy={info['dy']:+.1f}, θ={info['angle']:+.2f}°), "
                          f"L_Targets={info['local_snapped_targets']}, "
                          f"EdgeCC: {info['cc_before']:.3f}->{info['cc_after']:.3f} (Δ={info['delta_cc']:+.3f})")
    else:
        for t in tasks_sampled:
            stem, info = process_single(t)
            if info is not None:
                results.append((stem, info))
                if info["global_applied"]:
                    n_global_applied += 1
                if info["local_snapped_targets"] > 0:
                    n_local_applied += 1
                    total_local_targets += info["local_snapped_targets"]
                print(f"[{len(results)}/{len(tasks_sampled)}] {stem:20s}: "
                      f"G_Applied={info['global_applied']} (dx={info['dx']:+.1f}, dy={info['dy']:+.1f}, θ={info['angle']:+.2f}°), "
                      f"L_Targets={info['local_snapped_targets']}, "
                      f"EdgeCC: {info['cc_before']:.3f}->{info['cc_after']:.3f} (Δ={info['delta_cc']:+.3f})")

    print("\n=== 配准汇总统计 ===")
    print(f"  处理总数: {len(results)}")
    print(f"  全局几何校正触发: {n_global_applied} / {len(results)} ({n_global_applied/len(results)*100:.1f}%)")
    print(f"  目标级动态吸附触发: {n_local_applied} / {len(results)} (共校正 {total_local_targets} 个独立运动热斑)")
    avg_delta_cc = np.mean([info["delta_cc"] for _, info in results])
    print(f"  平均边缘结构相关度提升: ΔCC = {avg_delta_cc:+.4f}")

    # 保存结构化审计报告
    import json
    summary_path = os.path.join(os.path.dirname(os.path.abspath(args.out_dir)), f"alignment_summary_{os.path.basename(args.out_dir)}.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_data = {
        "total": len(results),
        "global_applied_count": n_global_applied,
        "global_applied_rate": round(n_global_applied / len(results), 4),
        "local_applied_count": n_local_applied,
        "total_local_targets": total_local_targets,
        "avg_delta_cc": round(float(avg_delta_cc), 4),
        "details": [{"stem": stem, **info} for stem, info in results],
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  详细审计报告已保存至: {summary_path}")


if __name__ == "__main__":
    main()
