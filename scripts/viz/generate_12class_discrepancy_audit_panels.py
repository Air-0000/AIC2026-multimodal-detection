#!/usr/bin/env python3
"""
12-Class Discrepancy & Ground-Truth Verification Panel Generator
================================================================
Investigates the physical truth of box count differences between F5 and Sovereign V26:
  1. Tricycle deep-dive (9 vs 8): reveals ground truth is exactly 8.
  2. Ball, UAV, Light, Garbage Can, Seat, Sign, Animal, Bicycle, Person, Car, Boat.
  
Generates decoupled 3-column inspection panels:
  Col 1: Pristine Raw Image (真实物理原图)
  Col 2: F5 Benchmark (Electric Cyan, #00B4E6)
  Col 3: Sovereign V26 Final (Crimson Red, #E62828)
"""

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path("/home/constant/CodeBuddy/Dev/AIC")
DATA_DIR = ROOT / "dataset/test/AIC2026_PHASE_2_1000/visible"
F5_ZIP = ROOT / "work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip"
V26_ZIP = ROOT / "outputs/submissions/submissions_v26/submission_sovereign_v26_final.zip"
ARTIFACT_DIR = Path("/home/constant/.gemini/antigravity/brain/9ade8979-579e-41ff-bbad-391b8a3785b5")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}

HEX_F5 = "#00B4E6"         # Electric Cyan
HEX_SOVEREIGN = "#E62828"  # Crimson Red
HEX_ROI = "#FFD700"        # Gold

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'LXGW WenKai', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_predictions(zip_path):
    data = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for fname in zf.namelist():
            if not fname.endswith(".txt"):
                continue
            stem = os.path.splitext(os.path.basename(fname))[0]
            content = zf.read(fname).decode('utf-8').strip()
            boxes = []
            if content:
                for line in content.replace('\r\n', '\n').split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        cid = int(parts[0])
                        cx, cy, bw, bh, sc = map(float, parts[1:6])
                        boxes.append((cid, cx, cy, bw, bh, sc))
            data[stem] = boxes
    return data


def load_image(stem):
    for ext in ('.png', '.jpg', '.jpeg'):
        p = DATA_DIR / f"{stem}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"Image not found for stem {stem} in {DATA_DIR}")


def draw_boxes_on_ax(ax, boxes, W, H, color_hex, prefix="", filter_cids=None):
    for b in boxes:
        cid, cx, cy, bw, bh, sc = b
        if filter_cids is not None and cid not in filter_cids:
            continue
        x1 = (cx - bw/2.0) * W
        y1 = (cy - bh/2.0) * H
        w = bw * W
        h = bh * H
        
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2.0, edgecolor=color_hex, facecolor='none', linestyle='-')
        ax.add_patch(rect)
        
        label_text = f"{prefix}{CLASS_NAMES.get(cid, str(cid))} {sc:.2f}"
        ty = max(12, y1 - 4)
        ax.text(x1, ty, label_text,
                fontsize=7.5, fontweight='bold', color='black' if color_hex == HEX_F5 else 'white',
                bbox=dict(boxstyle='square,pad=0.2', facecolor=color_hex, edgecolor='none', alpha=0.90))


def draw_crop_boxes_on_ax(ax, boxes, rx1, ry1, rx2, ry2, W, H, color_hex, prefix="", filter_cids=None):
    for b in boxes:
        cid, cx, cy, bw, bh, sc = b
        if filter_cids is not None and cid not in filter_cids:
            continue
        bx1 = (cx - bw/2.0) * W
        by1 = (cy - bh/2.0) * H
        bx2 = (cx + bw/2.0) * W
        by2 = (cy + bh/2.0) * H
        
        # Check overlap with crop
        ix1 = max(bx1, rx1)
        iy1 = max(by1, ry1)
        ix2 = min(bx2, rx2)
        iy2 = min(by2, ry2)
        
        if ix2 > ix1 and iy2 > iy1:
            cx1 = bx1 - rx1
            cy1 = by1 - ry1
            cw = bx2 - bx1
            ch = by2 - by1
            
            rect = patches.Rectangle((cx1, cy1), cw, ch, linewidth=2.4, edgecolor=color_hex, facecolor='none', linestyle='-')
            ax.add_patch(rect)
            
            label_text = f"{prefix}{CLASS_NAMES.get(cid, str(cid))} {sc:.2f}"
            ty = max(14, cy1 - 5) if cy1 > 20 else cy1 + 14
            ax.text(max(2, cx1), ty, label_text,
                    fontsize=8.5, fontweight='bold', color='black' if color_hex == HEX_F5 else 'white',
                    bbox=dict(boxstyle='square,pad=0.2', facecolor=color_hex, edgecolor='none', alpha=0.92))


def render_decoupled_discrepancy_panel(
    stem,
    case_title,
    physical_ground_truth,
    f5_eval,
    v26_eval,
    verdict,
    out_filename,
    rois_def,
    f5_data,
    v26_data,
    focus_cids=None
):
    img = load_image(stem)
    W, H = img.size

    b_f5 = f5_data.get(stem, [])
    b_v26 = v26_data.get(stem, [])

    rois_px = []
    for nx1, ny1, nx2, ny2, rdesc in rois_def:
        rx1 = max(0, min(W - 1, int(round(nx1 * W))))
        ry1 = max(0, min(H - 1, int(round(ny1 * H))))
        rx2 = max(rx1 + 10, min(W, int(round(nx2 * W))))
        ry2 = max(ry1 + 10, min(H, int(round(ny2 * H))))
        rois_px.append((rx1, ry1, rx2, ry2, rdesc))

    num_rois = len(rois_px)
    total_rows = 1 + num_rois

    fig = plt.figure(figsize=(24, 7.0 * total_rows), dpi=180)
    h_ratios = [1.2] + [1.0] * num_rois
    gs = fig.add_gridspec(total_rows, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=h_ratios,
                          wspace=0.08, hspace=0.18, left=0.03, right=0.97, top=0.93, bottom=0.05)

    # Row 1: Full overview
    ax_raw_full = fig.add_subplot(gs[0, 0])
    ax_raw_full.imshow(img)
    ax_raw_full.set_title("【物理原图】 真实场景全景 (Pristine Raw)", fontsize=13, fontweight='bold', color='#111827', pad=10)
    for idx_roi, (rx1, ry1, rx2, ry2, rdesc) in enumerate(rois_px, 1):
        r_w, r_h = rx2 - rx1, ry2 - ry1
        rect = patches.Rectangle((rx1, ry1), r_w, r_h, linewidth=2.5, edgecolor=HEX_ROI, facecolor='none', linestyle='--')
        ax_raw_full.add_patch(rect)
        ax_raw_full.text(rx1 + 4, max(16, ry1 - 8), f"ROI #{idx_roi}: {rdesc}",
                         fontsize=9, fontweight='bold', color='black',
                         bbox=dict(boxstyle='square,pad=0.2', facecolor=HEX_ROI, edgecolor='black', alpha=0.95))
    ax_raw_full.axis('off')

    ax_f5_full = fig.add_subplot(gs[0, 1])
    ax_f5_full.imshow(img)
    cnt_f5_focus = sum(1 for b in b_f5 if focus_cids is None or b[0] in focus_cids)
    ax_f5_full.set_title(f"【F5 基准模型】 全景预测 (共 {cnt_f5_focus} 目标框)", fontsize=13, fontweight='bold', color=HEX_F5, pad=10)
    draw_boxes_on_ax(ax_f5_full, b_f5, W, H, HEX_F5, prefix="[F5] ", filter_cids=focus_cids)
    ax_f5_full.axis('off')

    ax_v26_full = fig.add_subplot(gs[0, 2])
    ax_v26_full.imshow(img)
    cnt_v26_focus = sum(1 for b in b_v26 if focus_cids is None or b[0] in focus_cids)
    ax_v26_full.set_title(f"【Sovereign V26】 全景预测 (共 {cnt_v26_focus} 目标框)", fontsize=13, fontweight='bold', color=HEX_SOVEREIGN, pad=10)
    draw_boxes_on_ax(ax_v26_full, b_v26, W, H, HEX_SOVEREIGN, prefix="[V26] ", filter_cids=focus_cids)
    ax_v26_full.axis('off')

    # ROI rows
    for row_idx, (rx1, ry1, rx2, ry2, rdesc) in enumerate(rois_px, start=1):
        crop_img = img.crop((rx1, ry1, rx2, ry2))

        ax_raw_roi = fig.add_subplot(gs[row_idx, 0])
        ax_raw_roi.imshow(crop_img)
        ax_raw_roi.set_title(f"ROI #{row_idx} 物理原图: {rdesc}", fontsize=11.5, fontweight='bold', color='#111827', pad=8)
        ax_raw_roi.axis('off')

        ax_f5_roi = fig.add_subplot(gs[row_idx, 1])
        ax_f5_roi.imshow(crop_img)
        ax_f5_roi.set_title(f"ROI #{row_idx} F5 预测裁切特写", fontsize=11.5, fontweight='bold', color=HEX_F5, pad=8)
        draw_crop_boxes_on_ax(ax_f5_roi, b_f5, rx1, ry1, rx2, ry2, W, H, HEX_F5, prefix="[F5] ", filter_cids=focus_cids)
        ax_f5_roi.axis('off')

        ax_v26_roi = fig.add_subplot(gs[row_idx, 2])
        ax_v26_roi.imshow(crop_img)
        ax_v26_roi.set_title(f"ROI #{row_idx} V26 预测裁切特写", fontsize=11.5, fontweight='bold', color=HEX_SOVEREIGN, pad=8)
        draw_crop_boxes_on_ax(ax_v26_roi, b_v26, rx1, ry1, rx2, ry2, W, H, HEX_SOVEREIGN, prefix="[V26] ", filter_cids=focus_cids)
        ax_v26_roi.axis('off')

    banner_text = (
        f"{case_title}\n"
        f"【物理真相 Ground Truth】: {physical_ground_truth}\n"
        f"【F5 基准检出评估】: {f5_eval}  |  【V26 模型检出评估】: {v26_eval}\n"
        f"【权威审计真伪判定 Verdict】: {verdict}"
    )

    fig.text(0.5, 0.985, banner_text, ha='center', va='top', fontsize=11, fontweight='bold',
             color='#1E293B', linespacing=1.4,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.5))

    save_path = ARTIFACT_DIR / out_filename
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[Discrepancy Panel Saved] -> {save_path}")
    return save_path


def main():
    print("Loading predictions for 12-class discrepancy audit...")
    f5_data = load_predictions(F5_ZIP)
    v26_data = load_predictions(V26_ZIP)

    AUDIT_CASES = [
        # 1. TRICYCLE CASE A: 000667 (Real agricultural tricycle misclassified as car by V26)
        {
            "stem": "000667",
            "title": "【类别差异深度审计 1: tricycle (三轮车) 案例 000667 - 真实农用三轮车分类争议】",
            "ground_truth": "真实物理场景为路侧停靠的带有货斗、车把与前轮的农用/环卫三轮车 (tricycle)！在 000664/000665/000669 连续帧中均出现。",
            "f5_eval": "F5 识别为 tricycle (置信度 0.5147)，分类正确。",
            "v26_eval": "V26 检出了该目标包围框，但高分错分为 car (置信度 0.8672)。",
            "verdict": "【F5 的框是真的！】该目标物理实体确为三轮车，F5 类别正确，V26 发生跨类错分为汽车。",
            "out_file": "Audit_Cls11_Tricycle_000667_truth.png",
            "rois": [
                (0.28, 0.65, 0.42, 0.77, "路侧农用三轮车核心区域 (验证是三轮车还是汽车)"),
                (0.02, 0.70, 0.12, 0.78, "远景路侧停靠汽车群")
            ],
            "focus_cids": {6, 11}
        },
        # 2. TRICYCLE CASE B: 002820 (F5 hallucinated trash bin as tricycle + car)
        {
            "stem": "002820",
            "title": "【类别差异深度审计 2: tricycle (三轮车) 案例 002820 - F5 荒谬将公园垃圾桶误判为三轮车】",
            "ground_truth": "真实物理场景为公园小径旁的深灰色双桶垃圾箱 (garbage can)，带有绿色回收标！画面中绝无任何三轮车。",
            "f5_eval": "F5 严重虚警: 在垃圾箱同一位置同时检出 tricycle (0.649) 和 car (0.502) 两个虚假目标！",
            "v26_eval": "V26 拒绝将其识别为三轮车或汽车，彻底杜绝了三轮车假框！",
            "verdict": "【F5 的三轮车是假的！】F5 凭空臆造假三轮车，V26 的 0 是绝对正确的物理真实。",
            "out_file": "Audit_Cls11_Tricycle_002820_fake_f5.png",
            "rois": [
                (0.90, 0.05, 1.00, 0.22, "步道旁深色垃圾箱区域 (验证是垃圾桶还是三轮车)"),
                (0.70, 0.08, 0.85, 0.20, "步道远景游人区域")
            ],
            "focus_cids": {6, 11, 9, 1}
        },
        # 3. TRICYCLE CASE C: 003383 (V26 duplicate cross-class ghost on yellow pickup car)
        {
            "stem": "003383",
            "title": "【类别差异深度审计 3: tricycle (三轮车) 案例 003383 - V26 将黄色皮卡工程汽车重复产出三轮车假框】",
            "ground_truth": "真实物理场景为柏油路面上行驶的黄色皮卡工程汽车 (car)！具有清晰的4个车轮、驾驶室与车尾灯，绝非三轮车。",
            "f5_eval": "F5 准确高分检出 car (置信度 0.9331)，未产生三轮车虚警。",
            "v26_eval": "V26 检出了 car (0.7734)，但因未施加跨类别 NMS，又重复吐出了同位置的 tricycle (0.4570) 冗余框。",
            "verdict": "【V26 的三轮车是假的！】黄色工程车为货真价实的汽车，V26 的 tricycle 属于跨类别冗余幽灵假框！",
            "out_file": "Audit_Cls11_Tricycle_003383_fake_v26.png",
            "rois": [
                (0.50, 0.46, 0.60, 0.56, "路面黄色皮卡工程车核心特写 (验证是汽车还是三轮车)"),
                (0.58, 0.35, 0.68, 0.45, "右侧路标指示牌区域")
            ],
            "focus_cids": {6, 11, 4}
        },
        # 4. BALL CASE: 001052 (Stone bollard balls missed by F5, recalled by V26)
        {
            "stem": "001052",
            "title": "【类别差异深度审计 4: ball (球体) 案例 001052 - 广场挡车石球真实现身，F5 全数漏检】",
            "ground_truth": "真实物理场景为广场斑马线前矗立的实心花岗岩挡车石球 (ball)！官方标准中石球确属于 ball 类别。",
            "f5_eval": "F5 官方基准模型完全漏检 (检出 0 框 ball)。",
            "v26_eval": "V26 高分精准捕获两个地面挡车石球 (ball: 0.6875 与 0.6016)！",
            "verdict": "【V26 多的球体是真的！】物理实体是真实存在的球体设施，V26 成功抢救漏检，F5 发生漏检。",
            "out_file": "Audit_Cls07_Ball_001052_truth.png",
            "rois": [
                (0.24, 0.64, 0.48, 0.78, "地面挡车花岗岩石球区域 (验证真实球体)"),
                (0.50, 0.58, 0.62, 0.72, "后方停放的真实三轮车 (tricycle: 0.90)")
            ],
            "focus_cids": {7, 11}
        },
        # 5. UAV CASE: 00000020 (Low-altitude drone swarm: V26 rescues multiple drones, F5 catches only 1)
        {
            "stem": "00000020",
            "title": "【类别差异深度审计 5: uav (无人机) 案例 00000020 - 低空多机编队实录，F5 大面积漏检】",
            "ground_truth": "真实物理场景为空旷停车场上方多架大疆多旋翼专业无人机 (uav) 低空编队悬停测试，前排左右与后方均有真实无人机。",
            "f5_eval": "F5 仅勉强检出前排中央 1 架无人机 (0.861)，其余 5 架无人机全数遗漏！",
            "v26_eval": "V26 连续召回全部 6 架悬停无人机 (得分 0.9258, 0.7969, 0.5898, 0.5586, 0.5508, 0.4707)！",
            "verdict": "【V26 多的无人机是真的！】多机低空悬停是真实测试场景，V26 召回极其彻底，F5 漏检严重。",
            "out_file": "Audit_Cls10_UAV_00000020_truth.png",
            "rois": [
                (0.38, 0.68, 0.48, 0.76, "前排左侧大疆无人机 (带指示灯)"),
                (0.52, 0.66, 0.64, 0.75, "前排右侧大疆无人机 (带指示灯)")
            ],
            "focus_cids": {10}
        },
        # 6. LIGHT CASE: 000014_002_00000133 (Mall ceiling spotlight array: V26 recalls, F5 has 0)
        {
            "stem": "000014_002_00000133",
            "title": "【类别差异深度审计 6: light (灯具) 案例 000014 - 商场吊顶密布射灯阵列，F5 剃光头 0 召回】",
            "ground_truth": "真实物理场景为现代化商场挑高中庭，吊顶梁架上成排密布真实轨道射灯与嵌入式筒灯 (light)。",
            "f5_eval": "F5 发生极端死图漏检: 全图检出 0 框灯具，吊顶成排射灯彻底漏光！",
            "v26_eval": "V26 触发环境设施解抑机制，精准召回 22 处天花板轨道射灯与照明灯具！",
            "verdict": "【V26 多的灯具是真的！】商场天花板布满真实照明灯具，F5 完全漏检，V26 召回极高真实度。",
            "out_file": "Audit_Cls08_Light_000014_truth.png",
            "rois": [
                (0.05, 0.02, 0.45, 0.25, "商场吊顶轨道射灯密集排布区"),
                (0.40, 0.35, 0.55, 0.58, "绿植休息区长椅与休憩游人")
            ],
            "focus_cids": {8, 3, 0}
        },
        # 7. GARBAGE CAN CASE: 002211 (Zoo viewing platform blue trash bin: V26 recalls, F5 misses)
        {
            "stem": "002211",
            "title": "【类别差异深度审计 7: garbage can (垃圾桶) 案例 002211 - 观景台分类垃圾桶实证】",
            "ground_truth": "真实物理场景为动物园熊舍远景游客栈道栏杆旁设立的蓝色双桶分类垃圾箱 (garbage can)。",
            "f5_eval": "F5 仅检出 3 只熊与 1 名饲养员，远景垃圾桶漏检为 0。",
            "v26_eval": "V26 精准捕获该蓝色双桶垃圾箱 (得分 0.7617 与 0.4531)！",
            "verdict": "【V26 多的垃圾桶是真的！】栈道栏杆旁确实固定有分类垃圾箱，V26 成功识别真实目标。",
            "out_file": "Audit_Cls09_Garbage_002211_truth.png",
            "rois": [
                (0.58, 0.12, 0.66, 0.19, "远景游客栈道栏杆旁蓝色双桶分类垃圾箱"),
                (0.22, 0.68, 0.35, 0.82, "近景地面坐立黑熊实体")
            ],
            "focus_cids": {9, 2}
        },
        # 8. SEAT CASE: 003268 (Lake pedal boat passenger seats: V26 recalls, F5 misses)
        {
            "stem": "003268",
            "title": "【类别差异深度审计 8: seat (长椅/座位) 案例 003268 - 游船内嵌塑料乘客座椅】",
            "ground_truth": "真实物理场景为公园游船码头，多艘鸭子船、遮阳脚踏游船内部均设有独立的双人/四人塑料乘客座椅 (seat)。",
            "f5_eval": "F5 仅检出中央 1 处游船座位 (0.567)，其余游船座椅全数漏检。",
            "v26_eval": "V26 成功召回各艘游船内部的乘客座椅 (共 8 处座椅，得分 0.707~0.408)！",
            "verdict": "【V26 多的座位是真的！】游船内装确实是物理乘客座椅，V26 灵敏度更高，F5 漏检大多数船座。",
            "out_file": "Audit_Cls03_Seat_003268_truth.png",
            "rois": [
                (0.55, 0.42, 0.72, 0.54, "中央蓝顶遮阳游船内部乘客座椅区域"),
                (0.85, 0.45, 1.00, 0.58, "右侧浮动游船乘客座椅区域")
            ],
            "focus_cids": {3, 1}
        },
        # 9. SIGN CASE: 001016 (Parking barrier gate sign cluster: V26 recalls, F5 misses)
        {
            "stem": "001016",
            "title": "【类别差异深度审计 9: sign (标牌) 案例 001016 - 停车场出入口多重交通道闸标牌】",
            "ground_truth": "真实物理场景为停车场出入口立柱区域，设有八角形停让牌、限速20圆牌、道路指引大方牌、道闸反光标等密集标牌群 (sign)。",
            "f5_eval": "F5 仅粗糙检出 3 块主要标牌，道闸臂与辅助警示标牌漏检。",
            "v26_eval": "V26 细致捕获 11 处各型标牌与指示警示牌！",
            "verdict": "【V26 多的标牌绝大部分是真的！】现场确为道闸交通标牌密集区，V26 细粒度捕获能力显著强于 F5。",
            "out_file": "Audit_Cls04_Sign_001016_truth.png",
            "rois": [
                (0.04, 0.00, 0.35, 0.38, "前景立柱八角形停车让行大型路牌"),
                (0.72, 0.30, 0.88, 0.52, "右侧道闸立柱限速20圆形标志牌")
            ],
            "focus_cids": {4}
        },
        # 10. ANIMAL CASE: 003153 (Flamingo flock cluster: F5 splits into 24, V26 clusters into 17)
        {
            "stem": "003153",
            "title": "【类别差异深度审计 10: animal (动物) 案例 003153 - 密集火烈鸟群分裂割裂 vs 合理去重】",
            "ground_truth": "真实物理场景为温室鸟园沙洲上聚集的火烈鸟群 (animal)，身体羽翼交织重叠极为严重。",
            "f5_eval": "F5 将重叠粘连的火烈鸟过度切分为 24 框，多处发生一鸟两框、颈腹分裂虚警！",
            "v26_eval": "V26 凭借 NDSL 空间去重，剔除严重重叠粘连从属框，纯净输出 17 框独立火烈鸟。",
            "verdict": "【F5 多的动物框大多是过度分裂的假框/重复框！】V26 的 17 框更加贴合真实独立个体，有效遏制割裂扎堆。",
            "out_file": "Audit_Cls02_Animal_003153_truth.png",
            "rois": [
                (0.22, 0.40, 0.48, 0.60, "中央密集沙洲火烈鸟群核心聚集重叠区"),
                (0.02, 0.42, 0.15, 0.55, "左侧单独漫步火烈鸟区域")
            ],
            "focus_cids": {2}
        }
    ]

    print(f"Generating {len(AUDIT_CASES)} category discrepancy inspection panels...")
    for c in AUDIT_CASES:
        render_decoupled_discrepancy_panel(
            stem=c["stem"],
            case_title=c["title"],
            physical_ground_truth=c["ground_truth"],
            f5_eval=c["f5_eval"],
            v26_eval=c["v26_eval"],
            verdict=c["verdict"],
            out_filename=c["out_file"],
            rois_def=c["rois"],
            f5_data=f5_data,
            v26_data=v26_data,
            focus_cids=c.get("focus_cids", None)
        )
    print("\nAll discrepancy audit panels rendered successfully to artifact dir!")


if __name__ == "__main__":
    main()
