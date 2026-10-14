#!/usr/bin/env python3
"""
Multimodal Detection Universal Comparator & Audit Engine
========================================================
A generic, reusable evaluation, discrepancy matching, and visualization engine
for multimodal object detection models (RGB + IR + Depth).

Core Capabilities:
1. Flexible Ingestion:
   - Compares ANY Baseline Model (A) against ANY Candidate Model (B).
   - Ingests prediction files from directories of .txt or .zip archives.
   - Supports YOLO format (<class_id> <cx> <cy> <w> <h> <score>).
   - Configurable per-class or flat confidence thresholds, top-k caps.

2. Bipartite Discrepancy Matching:
   - IoU-based Hungarian / Greedy bipartite matching.
   - Partitions detections into:
     * Exact Consensus (双方共识真框, IoU >= thr)
     * Category Conflict (同目标类别冲突, IoU >= thr, class mismatch)
     * Baseline Only (Candidate 遗漏 Baseline 真框 / 漏检)
     * Candidate Only (Candidate 独有多报伪框 / 虚警)
   - Classifies images into:
     * pure_miss (纯漏检无多报)
     * pure_fp (纯多报无漏检)
     * mixed (既有漏检又有多报)
     * conflict (存在类别冲突)
     * identical (100% 同框)

3. Visual Artifact Rendering:
   - Multi-process parallel rendering of high-res 3-column inspection panels:
     Col 1: Physical Reference (Visible image + IR thermal PIP thumbnail)
     Col 2: Baseline Model Detections (Consensus in Green, Baseline-only in Cyan)
     Col 3: Candidate Model Detections (Consensus in Green, Candidate-only in Red)
   - Support for specific category focus (e.g., ball, uav micro-crops).

4. Production-Ready Web Gallery & Audit Report:
   - Generates interactive, standalone HTML inspection gallery.
   - Generates structured JSON metadata & Markdown audit reports.

CLI Usage:
    python multimodal_comparator.py \
        --baseline work_dir/f5/submission.zip \
        --candidate work_dir/v23/submission.zip \
        --name-baseline "F5 (57.266)" \
        --name-candidate "V23 (53.00)" \
        --vis-dir dataset/test/visible \
        --ir-dir dataset/test/infrared \
        --output outputs/analysis/audit_f5_vs_v23 \
        --render-panels
"""

import os
import sys
import json
import zipfile
import shutil
import argparse
import concurrent.futures
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont

# Default AIC 2026 Class Names
DEFAULT_CLASSES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}
DEFAULT_ZH_SHORT = {
    0: '人', 1: '船', 2: '动物', 3: '座椅', 4: '标牌', 5: '单车',
    6: '汽车', 7: '球', 8: '路灯', 9: '垃圾桶', 10: '无人机', 11: '三轮车'
}
DEFAULT_RARE_CLASSES = {1, 7, 9, 10, 11}

# Visual Styling
C_EXACT    = (46, 204, 113)    # Emerald Green (Consensus)
C_CONF     = (255, 140, 0)     # Bright Orange (Conflict)
C_A_ONLY   = (0, 195, 255)     # Electric Cyan (Baseline / F5 Only)
C_B_ONLY   = (255, 45, 85)     # Crimson Magenta (Candidate / V23 Only)
C_BG       = (18, 20, 24)      # Dark Background
C_BANNER   = (10, 12, 16)      # Header
C_CARD     = (26, 30, 36)
C_TEXT     = (240, 245, 250)
C_MUTED    = (160, 168, 180)

FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'


class MultimodalComparator:
    def __init__(
        self,
        baseline_path,
        candidate_path,
        name_baseline="Baseline",
        name_candidate="Candidate",
        vis_dir="/home/constant/CodeBuddy/Dev/AIC/dataset/test/AIC2026_PHASE_2_1000/visible",
        ir_dir="/home/constant/CodeBuddy/Dev/AIC/dataset/test/AIC2026_PHASE_2_1000/infrared",
        iou_exact=0.25,
        iou_conflict=0.30,
        rare_classes=None,
        class_names=None,
        class_zh=None,
        thr_baseline=None,
        thr_candidate=None,
        max_boxes=60,
        focus_class=None
    ):
        self.baseline_path = str(baseline_path)
        self.candidate_path = str(candidate_path)
        self.name_baseline = name_baseline
        self.name_candidate = name_candidate
        self.vis_dir = str(vis_dir)
        self.ir_dir = str(ir_dir)
        self.iou_exact = iou_exact
        self.iou_conflict = iou_conflict
        self.max_boxes = max_boxes
        self.focus_class = focus_class

        self.class_names = class_names or DEFAULT_CLASSES
        self.class_zh = class_zh or DEFAULT_ZH_SHORT
        self.rare_classes = rare_classes if rare_classes is not None else DEFAULT_RARE_CLASSES

        # Threshold rules: flat float, or default (rare 0.45, common 0.50)
        self.thr_baseline = thr_baseline
        self.thr_candidate = thr_candidate

        self.baseline_data = {}
        self.candidate_data = {}
        self.all_stems = []
        self.diff_records = []
        self.global_meta = {}

    def _get_threshold(self, class_id, rule):
        if rule is None:
            return 0.45 if class_id in self.rare_classes else 0.50
        if isinstance(rule, (int, float)):
            return float(rule)
        if isinstance(rule, dict):
            return rule.get(class_id, 0.45 if class_id in self.rare_classes else 0.50)
        return 0.45

    def _load_prediction_source(self, path, thr_rule):
        d = {}
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if not fname.endswith('.txt'): continue
                stem = fname.rsplit('.', 1)[0]
                with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f if l.strip()]
                boxes = self._parse_lines(lines, thr_rule)
                d[stem] = boxes[:self.max_boxes]
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, 'r') as zf:
                for name in sorted(zf.namelist()):
                    if not name.endswith('.txt'): continue
                    stem = os.path.basename(name).rsplit('.', 1)[0]
                    content = zf.read(name).decode('utf-8', errors='ignore').strip()
                    lines = [l.strip() for l in content.split('\n') if l.strip()]
                    boxes = self._parse_lines(lines, thr_rule)
                    d[stem] = boxes[:self.max_boxes]
        else:
            raise FileNotFoundError(f"Prediction path not found or invalid: {path}")
        return d

    def _parse_lines(self, lines, thr_rule):
        boxes = []
        for l in lines:
            parts = l.split()
            if len(parts) < 6: continue
            cid = int(parts[0])
            if self.focus_class is not None and cid != self.focus_class:
                continue
            cx, cy, w, h, sc = map(float, parts[1:6])
            thr = self._get_threshold(cid, thr_rule)
            if sc >= thr:
                boxes.append((cid, cx, cy, w, h, sc))
        boxes.sort(key=lambda x: x[5], reverse=True)
        return boxes

    @staticmethod
    def compute_iou(b1, b2):
        # b: (cid, cx, cy, w, h, sc)
        x1_1, x2_1 = b1[1] - b1[3]/2.0, b1[1] + b1[3]/2.0
        y1_1, y2_1 = b1[2] - b1[4]/2.0, b1[2] + b1[4]/2.0
        x1_2, x2_2 = b2[1] - b2[3]/2.0, b2[1] + b2[3]/2.0
        y1_2, y2_2 = b2[2] - b2[4]/2.0, b2[2] + b2[4]/2.0
        ix = max(0.0, min(x2_1, x2_2) - max(x1_1, x1_2))
        iy = max(0.0, min(y2_1, y2_2) - max(y1_1, y1_2))
        inter = ix * iy
        union = b1[3]*b1[4] + b2[3]*b2[4] - inter
        return inter / union if union > 0 else 0.0

    def match_boxes(self, bf, bg):
        pairs_exact = []
        for i, b_f in enumerate(bf):
            for j, b_g in enumerate(bg):
                if b_f[0] == b_g[0]:
                    iou = self.compute_iou(b_f, b_g)
                    if iou >= self.iou_exact:
                        pairs_exact.append((iou, i, j))
        pairs_exact.sort(key=lambda x: x[0], reverse=True)

        matched_f = {}
        matched_g = {}
        for iou, i, j in pairs_exact:
            if i not in matched_f and j not in matched_g:
                matched_f[i] = ('exact', j, iou)
                matched_g[j] = ('exact', i, iou)

        pairs_conflict = []
        for i, b_f in enumerate(bf):
            if i in matched_f: continue
            for j, b_g in enumerate(bg):
                if j in matched_g: continue
                if b_f[0] != b_g[0]:
                    iou = self.compute_iou(b_f, b_g)
                    if iou >= self.iou_conflict:
                        pairs_conflict.append((iou, i, j))
        pairs_conflict.sort(key=lambda x: x[0], reverse=True)

        for iou, i, j in pairs_conflict:
            if i not in matched_f and j not in matched_g:
                matched_f[i] = ('conflict', j, iou)
                matched_g[j] = ('conflict', i, iou)

        return matched_f, matched_g

    def run_audit(self):
        print(f"[{self.name_candidate} vs {self.name_baseline}] Ingesting predictions...")
        self.baseline_data = self._load_prediction_source(self.baseline_path, self.thr_baseline)
        self.candidate_data = self._load_prediction_source(self.candidate_path, self.thr_candidate)

        self.all_stems = sorted(set(self.baseline_data.keys()) | set(self.candidate_data.keys()))
        print(f"Total images evaluated: {len(self.all_stems)}")

        cls_missed = Counter()
        cls_fp = Counter()
        cls_consensus = Counter()
        cls_conflict = Counter()

        total_a_boxes = 0
        total_b_boxes = 0
        total_consensus = 0
        total_conflict = 0
        total_missed = 0
        total_fp = 0

        self.diff_records = []

        for stem in self.all_stems:
            bf = self.baseline_data.get(stem, [])
            bg = self.candidate_data.get(stem, [])
            total_a_boxes += len(bf)
            total_b_boxes += len(bg)

            mf, mg = self.match_boxes(bf, bg)

            num_conflicts = sum(1 for v in mf.values() if v[0] == 'conflict')
            num_exact = sum(1 for v in mf.values() if v[0] == 'exact')
            total_consensus += num_exact
            total_conflict += num_conflicts

            for i, b in enumerate(bf):
                if i in mf:
                    if mf[i][0] == 'exact':
                        cls_consensus[b[0]] += 1
                    else:
                        cls_conflict[b[0]] += 1

            # Baseline Only (Candidate missed true boxes)
            a_only = []
            for i, b in enumerate(bf):
                if i not in mf:
                    a_only.append({
                        'cid': b[0],
                        'cname': self.class_names.get(b[0], str(b[0])),
                        'czh': self.class_zh.get(b[0], str(b[0])),
                        'score': round(b[5], 3),
                        'box': [round(b[1], 4), round(b[2], 4), round(b[3], 4), round(b[4], 4)]
                    })
                    cls_missed[b[0]] += 1
                    total_missed += 1

            # Candidate Only (Candidate over-predicted false positives)
            b_only = []
            for j, b in enumerate(bg):
                if j not in mg:
                    b_only.append({
                        'cid': b[0],
                        'cname': self.class_names.get(b[0], str(b[0])),
                        'czh': self.class_zh.get(b[0], str(b[0])),
                        'score': round(b[5], 3),
                        'box': [round(b[1], 4), round(b[2], 4), round(b[3], 4), round(b[4], 4)]
                    })
                    cls_fp[b[0]] += 1
                    total_fp += 1

            conflicts = []
            for i, v in mf.items():
                if v[0] == 'conflict':
                    j, iou = v[1], v[2]
                    conflicts.append({
                        'a_cid': bf[i][0],
                        'a_cname': self.class_names.get(bf[i][0], str(bf[i][0])),
                        'a_czh': self.class_zh.get(bf[i][0], str(bf[i][0])),
                        'a_score': round(bf[i][5], 3),
                        'b_cid': bg[j][0],
                        'b_cname': self.class_names.get(bg[j][0], str(bg[j][0])),
                        'b_czh': self.class_zh.get(bg[j][0], str(bg[j][0])),
                        'b_score': round(bg[j][5], 3),
                        'iou': round(iou, 2)
                    })

            delta = len(a_only) + len(b_only)
            if num_conflicts == 0 and delta == 0:
                continue

            summary_parts = []
            if num_conflicts > 0:
                for c in conflicts:
                    summary_parts.append(f"⚠️ 冲突: {self.name_baseline}:{c['a_czh']} ⚔️ {self.name_candidate}:{c['b_czh']} (IoU={c['iou']:.2f})")
            if len(a_only) > 0:
                summary_parts.append(f"🟦 遗漏真框 {len(a_only)}个")
            if len(b_only) > 0:
                summary_parts.append(f"🟥 多报伪框 {len(b_only)}个")

            has_a = len(a_only) > 0
            has_b = len(b_only) > 0

            self.diff_records.append({
                'stem': stem,
                'a_cnt': len(bf),
                'b_cnt': len(bg),
                'exact_cnt': num_exact,
                'conf_cnt': num_conflicts,
                'delta': delta,
                'missed_cnt': len(a_only),
                'fp_cnt': len(b_only),
                'missed_boxes': a_only,
                'fp_boxes': b_only,
                'conflicts': conflicts,
                'missed_cids': sorted(list(set(x['cid'] for x in a_only))),
                'fp_cids': sorted(list(set(x['cid'] for x in b_only))),
                'has_missed': has_a,
                'has_fp': has_b,
                'is_pure_miss': has_a and not has_b and num_conflicts == 0,
                'is_pure_fp': has_b and not has_a and num_conflicts == 0,
                'is_mixed': has_a and has_b,
                'max_miss_score': max([x['score'] for x in a_only]) if a_only else 0.0,
                'max_fp_score': max([x['score'] for x in b_only]) if b_only else 0.0,
                'summary': " | ".join(summary_parts)
            })

        self.global_meta = {
            'name_baseline': self.name_baseline,
            'name_candidate': self.name_candidate,
            'total_images': len(self.all_stems),
            'diff_images': len(self.diff_records),
            'identical_images': len(self.all_stems) - len(self.diff_records),
            'baseline_total_boxes': total_a_boxes,
            'candidate_total_boxes': total_b_boxes,
            'consensus_boxes': total_consensus,
            'conflict_boxes': total_conflict,
            'missed_boxes': total_missed,
            'fp_boxes': total_fp,
            'cls_missed': {cid: cls_missed[cid] for cid in range(12)},
            'cls_fp': {cid: cls_fp[cid] for cid in range(12)},
            'cls_consensus': {cid: cls_consensus[cid] for cid in range(12)},
            'class_names': self.class_names,
            'class_zh': self.class_zh
        }

        print(f"Audit completed: {len(self.diff_records)} diff images, {total_missed} missed boxes, {total_fp} false positive boxes.")
        return self.diff_records, self.global_meta

    def render_panels(self, output_dir, max_workers=8, skip_existing=True):
        out_img_dir = os.path.join(output_dir, 'all_discrepancies')
        os.makedirs(out_img_dir, exist_ok=True)

        tasks = []
        for r in self.diff_records:
            stem = r['stem']
            target_path = os.path.join(out_img_dir, f"{stem}.png")
            if skip_existing and os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
                continue
            bf = self.baseline_data.get(stem, [])
            bg = self.candidate_data.get(stem, [])
            mf, mg = self.match_boxes(bf, bg)
            tasks.append((stem, bf, bg, mf, mg, target_path))

        if not tasks:
            print("All panels already exist. Skipping rendering.")
            return

        print(f"Rendering {len(tasks)} comparison panels using {max_workers} CPU workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._render_single_panel, t) for t in tasks]
            done = 0
            for f in concurrent.futures.as_completed(futures):
                f.result()
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"  Rendered {done}/{len(tasks)} ({done*100.0/len(tasks):.1f}%)")

    def _render_single_panel(self, task):
        stem, bf, bg, mf, mg, out_path = task
        vis_path = os.path.join(self.vis_dir, f"{stem}.png")
        if not os.path.exists(vis_path): vis_path = os.path.join(self.vis_dir, f"{stem}.jpg")
        ir_path = os.path.join(self.ir_dir, f"{stem}.png")
        if not os.path.exists(ir_path): ir_path = os.path.join(self.ir_dir, f"{stem}.jpg")

        if not os.path.exists(vis_path):
            return

        img_vis = Image.open(vis_path).convert('RGB')
        img_ir = Image.open(ir_path).convert('RGB') if os.path.exists(ir_path) else None

        W_PANEL = 800
        H_PANEL = 450
        img_vis_r = img_vis.resize((W_PANEL, H_PANEL), Image.Resampling.LANCZOS)

        font_title = ImageFont.truetype(FONT_PATH, 22)
        font_sub = ImageFont.truetype(FONT_PATH, 15)
        font_box = ImageFont.truetype(FONT_PATH, 13)
        font_badge = ImageFont.truetype(FONT_PATH, 11)

        MARGIN = 16
        PAD = 14
        CANVAS_W = MARGIN * 2 + W_PANEL * 3 + PAD * 2
        HEADER_H = 100
        TITLE_BAR_H = 36
        CANVAS_H = HEADER_H + TITLE_BAR_H + H_PANEL + MARGIN * 2

        canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), C_BG)
        draw = ImageDraw.Draw(canvas)

        draw.rectangle([0, 0, CANVAS_W, HEADER_H], fill=C_BANNER)
        draw.line([(0, HEADER_H), (CANVAS_W, HEADER_H)], fill=(45, 50, 60), width=1)

        num_exact = sum(1 for v in mf.values() if v[0] == 'exact')
        title_str = f"【图号 {stem}】 {self.name_baseline} vs {self.name_candidate} 深度审计面板"
        sub_str = f"{self.name_baseline}: {len(bf)}框 | {self.name_candidate}: {len(bg)}框 | 共识: {num_exact}框"
        draw.text((MARGIN, 16), title_str, fill=C_TEXT, font=font_title)
        draw.text((MARGIN, 48), sub_str, fill=C_MUTED, font=font_sub)

        # Col 1: Raw + IR PIP
        x_col1 = MARGIN
        y_panels = HEADER_H + TITLE_BAR_H
        draw.rectangle([x_col1, HEADER_H + 4, x_col1 + W_PANEL, HEADER_H + TITLE_BAR_H - 2], fill=C_CARD)
        draw.text((x_col1 + 10, HEADER_H + 8), "物理原图 (右上角为嵌入式 IR 热成像)", fill=(120, 220, 255), font=font_sub)

        p1 = img_vis_r.copy()
        if img_ir:
            w_thumb = 240
            h_thumb = int(w_thumb * H_PANEL / W_PANEL)
            ir_thumb = img_ir.resize((w_thumb, h_thumb), Image.Resampling.LANCZOS)
            p1.paste(ir_thumb, (W_PANEL - w_thumb - 10, 10))
            d1 = ImageDraw.Draw(p1)
            d1.rectangle([W_PANEL - w_thumb - 10, 10, W_PANEL - 10, 10 + h_thumb], outline=(255, 255, 255), width=2)
            d1.rectangle([W_PANEL - w_thumb - 10, 10, W_PANEL - w_thumb + 85, 30], fill=(0, 0, 0))
            d1.text((W_PANEL - w_thumb - 6, 12), "IR 热成像", fill=(255, 215, 0), font=font_badge)
        canvas.paste(p1, (x_col1, y_panels))

        # Helper to draw boxes
        def draw_boxes(base_img, boxes, match_dict, is_baseline):
            p = base_img.copy()
            dp = ImageDraw.Draw(p)
            for idx, b in enumerate(boxes):
                cid, cx, cy, bw, bh, sc = b
                x1 = (cx - bw/2.0) * W_PANEL
                y1 = (cy - bh/2.0) * H_PANEL
                x2 = (cx + bw/2.0) * W_PANEL
                y2 = (cy + bh/2.0) * H_PANEL

                if idx in match_dict:
                    m_type = match_dict[idx][0]
                    if m_type == 'exact':
                        color = C_EXACT
                        tag = f"{self.class_names.get(cid, str(cid))} {sc:.2f}"
                    else:
                        color = C_CONF
                        tag = f"[冲突] {self.class_names.get(cid, str(cid))} {sc:.2f}"
                else:
                    color = C_A_ONLY if is_baseline else C_B_ONLY
                    tag = f"[{'遗漏真框' if is_baseline else '多报伪框'}] {self.class_names.get(cid, str(cid))} {sc:.2f}"

                dp.rectangle([x1, y1, x2, y2], outline=color, width=3)
                bbox = font_box.getbbox(tag)
                tw = bbox[2] - bbox[0] + 8
                th = bbox[3] - bbox[1] + 6
                ty = max(0, y1 - th)
                dp.rectangle([x1, ty, x1 + tw, ty + th], fill=color)
                dp.text((x1 + 4, ty + 2), tag, fill=(255, 255, 255), font=font_box)
            return p

        # Col 2: Baseline
        x_col2 = x_col1 + W_PANEL + PAD
        draw.rectangle([x_col2, HEADER_H + 4, x_col2 + W_PANEL, HEADER_H + TITLE_BAR_H - 2], fill=C_CARD)
        draw.text((x_col2 + 10, HEADER_H + 8), f"标杆模型: {self.name_baseline} (共 {len(bf)} 框)", fill=(0, 215, 255), font=font_sub)
        p2 = draw_boxes(img_vis_r, bf, mf, is_baseline=True)
        canvas.paste(p2, (x_col2, y_panels))

        # Col 3: Candidate
        x_col3 = x_col2 + W_PANEL + PAD
        draw.rectangle([x_col3, HEADER_H + 4, x_col3 + W_PANEL, HEADER_H + TITLE_BAR_H - 2], fill=C_CARD)
        draw.text((x_col3 + 10, HEADER_H + 8), f"对比模型: {self.name_candidate} (共 {len(bg)} 框)", fill=(255, 80, 110), font=font_sub)
        p3 = draw_boxes(img_vis_r, bg, mg, is_baseline=False)
        canvas.paste(p3, (x_col3, y_panels))

        canvas.save(out_path, format='PNG', quality=95)

    def export_all(self, output_dir, render_panels=False):
        os.makedirs(output_dir, exist_ok=True)
        self.run_audit()

        if render_panels:
            self.render_panels(output_dir)

        # Export JSON metadata
        json_path = os.path.join(output_dir, 'discrepancy_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'items': self.diff_records, 'meta': self.global_meta}, f, ensure_ascii=False, indent=2)
        print(f"Exported JSON metadata: {json_path}")

        # Export HTML gallery
        html_path = os.path.join(output_dir, 'index.html')
        html_content = self.generate_html_content()
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Exported Interactive Gallery: {html_path}")

    def generate_html_content(self):
        json_str = json.dumps({'items': self.diff_records, 'meta': self.global_meta}, ensure_ascii=False)
        
        name_a = self.name_baseline
        name_b = self.name_candidate

        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name_a} vs {name_b} 多模态检测差异审阅大盘</title>
    <style>
        :root {{
            --bg-dark: #0f1115;
            --bg-card: #181b21;
            --border: #282e3b;
            --text-main: #f3f4f6;
            --text-dim: #94a3b8;
            --f5-color: #06b6d4;
            --v23-color: #f43f5e;
            --consensus-color: #10b981;
            --primary: #3b82f6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", sans-serif; }}
        body {{ background: var(--bg-dark); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }}
        header {{ background: #090a0d; border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }}
        .brand-title {{ font-size: 17px; font-weight: 700; color: #fff; }}
        .kpi-bar {{ background: #12151b; border-bottom: 1px solid var(--border); padding: 10px 24px; display: flex; gap: 16px; overflow-x: auto; }}
        .kpi-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 6px 14px; white-space: nowrap; }}
        .kpi-val {{ font-size: 16px; font-weight: 700; }}
        .kpi-lbl {{ font-size: 12px; color: var(--text-dim); }}
        .controls-section {{ background: #14171e; border-bottom: 1px solid var(--border); padding: 10px 24px; display: flex; flex-direction: column; gap: 10px; }}
        .tab-group {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .tab-btn {{ background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }}
        .tab-btn.active {{ background: #2563eb; color: #fff; border-color: #3b82f6; }}
        .sub-filters {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 12px; }}
        .chip {{ background: #1e222a; border: 1px solid #2d3340; color: var(--text-dim); padding: 3px 9px; border-radius: 4px; cursor: pointer; }}
        .chip.active {{ background: #374151; color: #fff; font-weight: 600; }}
        main {{ flex: 1; padding: 16px 24px; display: flex; flex-direction: column; align-items: center; gap: 14px; }}
        .info-card {{ width: 100%; max-width: 1920px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 20px; }}
        .stage-container {{ width: 100%; max-width: 1920px; background: #08090b; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
        .stage-container img {{ width: 100%; height: auto; display: block; }}
        .box-breakdown {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px; }}
        .box-list-card {{ background: #121418; border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; }}
        .tag-f5 {{ background: rgba(6,182,212,0.15); border: 1px solid rgba(6,182,212,0.4); color: var(--f5-color); padding: 3px 8px; border-radius: 4px; font-size: 12px; margin: 2px; display: inline-block; }}
        .tag-v23 {{ background: rgba(244,63,94,0.15); border: 1px solid rgba(244,63,94,0.4); color: var(--v23-color); padding: 3px 8px; border-radius: 4px; font-size: 12px; margin: 2px; display: inline-block; }}
        .nav-btn {{ background: #2563eb; color: #fff; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; }}
    </style>
</head>
<body>
    <header>
        <div class="brand-title">AIC 2026: {name_a} ⚔️ {name_b} 对比工作台</div>
        <div style="display:flex;gap:8px;align-items:center;">
            <button class="nav-btn" onclick="prevImg()">◀ 上一张 (←)</button>
            <span id="curIndexLabel" style="font-weight:700;font-size:13px;min-width:70px;text-align:center;">1 / 0</span>
            <button class="nav-btn" onclick="nextImg()">下一张 (→) ▶</button>
        </div>
    </header>

    <div class="kpi-bar">
        <div class="kpi-card" style="border-left: 3px solid #3b82f6;">
            <div class="kpi-val" style="color:#60a5fa;" id="kpiTotalDiff">0 张</div>
            <div class="kpi-lbl">存在差异图像</div>
        </div>
        <div class="kpi-card" style="border-left: 3px solid var(--f5-color);">
            <div class="kpi-val" style="color:var(--f5-color);" id="kpiMissed">0 框</div>
            <div class="kpi-lbl">📉 {name_b} 遗漏 {name_a} 真框</div>
        </div>
        <div class="kpi-card" style="border-left: 3px solid var(--v23-color);">
            <div class="kpi-val" style="color:var(--v23-color);" id="kpiFP">0 框</div>
            <div class="kpi-lbl">📈 {name_b} 独有多报伪框</div>
        </div>
        <div class="kpi-card" style="border-left: 3px solid var(--consensus-color);">
            <div class="kpi-val" style="color:var(--consensus-color);" id="kpiConsensus">0 框</div>
            <div class="kpi-lbl">🟩 双方共识检测框</div>
        </div>
    </div>

    <div class="controls-section">
        <div class="tab-group">
            <button class="tab-btn active" onclick="setMainMode('all')">🌐 全量差异</button>
            <button class="tab-btn" onclick="setMainMode('missed')">📉 {name_b} 遗漏真框</button>
            <button class="tab-btn" onclick="setMainMode('fp')">📈 {name_b} 多报伪框</button>
            <button class="tab-btn" onclick="setMainMode('pure_miss')">🎯 纯遗漏</button>
            <button class="tab-btn" onclick="setMainMode('pure_fp')">🚨 纯多报</button>
            <button class="tab-btn" onclick="setMainMode('mixed')">⚖️ 交织差异</button>
            <button class="tab-btn" onclick="setMainMode('conflict')" style="color:#fbbf24;">⚠️ 类别冲突</button>
        </div>
        <div class="sub-filters">
            <span style="color:var(--text-dim);font-weight:600;">类别过滤：</span>
            <span class="chip active" onclick="setClassFilter(-1)" id="chipAllCls">全部类别</span>
            <span class="chip" onclick="setClassFilter(0)">人</span>
            <span class="chip" onclick="setClassFilter(2)">动物</span>
            <span class="chip" onclick="setClassFilter(8)">路灯</span>
            <span class="chip" onclick="setClassFilter(6)">汽车</span>
            <span class="chip" onclick="setClassFilter(4)">标牌</span>
            <span class="chip" onclick="setClassFilter(3)">座椅</span>
            <span class="chip" onclick="setClassFilter(9)">垃圾桶</span>
            <span class="chip" onclick="setClassFilter(5)">单车</span>
            <span class="chip" onclick="setClassFilter(1)">船</span>
            <span class="chip" onclick="setClassFilter(10)">无人机</span>
            <span class="chip" onclick="setClassFilter(7)">球</span>
            <span class="chip" onclick="setClassFilter(11)">三轮车</span>
            <div style="margin-left:auto;display:flex;align-items:center;gap:6px;">
                <span>高置信(≥0.75):</span>
                <input type="checkbox" id="chkHigh" onchange="toggleHigh()" />
                <input type="text" id="searchBox" placeholder="搜索图号..." oninput="onSearch()" style="background:#1a1e26;border:1px solid var(--border);color:#fff;padding:3px 8px;border-radius:4px;font-size:12px;width:120px;" />
            </div>
        </div>
    </div>

    <main>
        <div class="info-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div id="stemTitle" style="font-size:18px;font-weight:700;">【图号】</div>
                <div id="stemSummaryText" style="color:#fbbf24;font-size:13px;font-weight:600;"></div>
            </div>
            <div class="box-breakdown">
                <div class="box-list-card" style="border-left: 3px solid var(--f5-color);">
                    <div style="font-size:13px;font-weight:700;color:var(--f5-color);margin-bottom:6px;">📉 {name_b} 遗漏的 {name_a} 真框</div>
                    <div id="f5OnlyTags"></div>
                </div>
                <div class="box-list-card" style="border-left: 3px solid var(--v23-color);">
                    <div style="font-size:13px;font-weight:700;color:var(--v23-color);margin-bottom:6px;">📈 {name_b} 独有多报的伪框</div>
                    <div id="v23OnlyTags"></div>
                </div>
            </div>
        </div>

        <div class="stage-container">
            <img id="mainStageImg" src="" alt="面板加载中..." />
        </div>
    </main>

    <script>
        const DATA = {json_str};
        const items = DATA.items;
        const meta = DATA.meta;

        document.getElementById('kpiTotalDiff').innerText = `${{meta.diff_images}} 张`;
        document.getElementById('kpiMissed').innerText = `${{meta.missed_boxes}} 框`;
        document.getElementById('kpiFP').innerText = `${{meta.fp_boxes}} 框`;
        document.getElementById('kpiConsensus').innerText = `${{meta.consensus_boxes}} 框`;

        let curMode = 'all';
        let curCid = -1;
        let isHigh = false;
        let query = '';
        let filtered = [];
        let curIdx = 0;

        function apply() {{
            filtered = items.filter(it => {{
                if (curMode === 'missed' && !it.has_missed) return false;
                if (curMode === 'fp' && !it.has_fp) return false;
                if (curMode === 'pure_miss' && !it.is_pure_miss) return false;
                if (curMode === 'pure_fp' && !it.is_pure_fp) return false;
                if (curMode === 'mixed' && !it.is_mixed) return false;
                if (curMode === 'conflict' && it.conf_cnt === 0) return false;

                if (curCid !== -1) {{
                    if (curMode === 'missed' || curMode === 'pure_miss') {{
                        if (!it.missed_cids.includes(curCid)) return false;
                    }} else if (curMode === 'fp' || curMode === 'pure_fp') {{
                        if (!it.fp_cids.includes(curCid)) return false;
                    }} else {{
                        if (!it.missed_cids.includes(curCid) && !it.fp_cids.includes(curCid)) return false;
                    }}
                }}

                if (isHigh) {{
                    if (curMode === 'missed' && it.max_miss_score < 0.75) return false;
                    if (curMode === 'fp' && it.max_fp_score < 0.75) return false;
                }}

                if (query.trim() && !it.stem.toLowerCase().includes(query.trim().toLowerCase())) return false;
                return true;
            }});
            curIdx = 0;
            render();
        }}

        function render() {{
            const lbl = document.getElementById('curIndexLabel');
            if (filtered.length === 0) {{
                lbl.innerText = '0 / 0';
                document.getElementById('stemTitle').innerText = '无匹配样本';
                document.getElementById('mainStageImg').src = '';
                return;
            }}
            lbl.innerText = `${{curIdx + 1}} / ${{filtered.length}}`;
            const it = filtered[curIdx];
            document.getElementById('stemTitle').innerText = `【图号 ${{it.stem}}】 (${{it.a_cnt}}框 vs ${{it.b_cnt}}框)`;
            document.getElementById('stemSummaryText').innerText = it.summary;
            document.getElementById('f5OnlyTags').innerHTML = it.missed_boxes.map(b => `<span class="tag-f5">🔷 ${{b.czh}} (${{b.score.toFixed(2)}})</span>`).join('') || '<span style="color:#64748b;font-size:12px;">无</span>';
            document.getElementById('v23OnlyTags').innerHTML = it.fp_boxes.map(b => `<span class="tag-v23">🔴 ${{b.czh}} (${{b.score.toFixed(2)}})</span>`).join('') || '<span style="color:#64748b;font-size:12px;">无</span>';
            document.getElementById('mainStageImg').src = `all_discrepancies/${{it.stem}}.png`;
        }}

        function setMainMode(m) {{
            curMode = m;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            apply();
        }}

        function setClassFilter(c) {{
            curCid = c;
            document.querySelectorAll('.chip').forEach(ch => ch.classList.remove('active'));
            event.target.classList.add('active');
            apply();
        }}

        function toggleHigh() {{
            isHigh = document.getElementById('chkHigh').checked;
            apply();
        }}

        function onSearch() {{
            query = document.getElementById('searchBox').value;
            apply();
        }}

        function prevImg() {{
            if (filtered.length === 0) return;
            curIdx = (curIdx - 1 + filtered.length) % filtered.length;
            render();
        }}

        function nextImg() {{
            if (filtered.length === 0) return;
            curIdx = (curIdx + 1) % filtered.length;
            render();
        }}

        window.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'INPUT') return;
            if (e.key === 'ArrowLeft') prevImg();
            else if (e.key === 'ArrowRight') nextImg();
        }});

        document.addEventListener('DOMContentLoaded', () => apply());
    </script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description="Multimodal Detection Universal Comparator")
    parser.add_argument("--baseline", required=True, help="Baseline predictions (.zip or directory)")
    parser.add_argument("--candidate", required=True, help="Candidate predictions (.zip or directory)")
    parser.add_argument("--name-baseline", default="Baseline", help="Display name for baseline")
    parser.add_argument("--name-candidate", default="Candidate", help="Display name for candidate")
    parser.add_argument("--vis-dir", default="/home/constant/CodeBuddy/Dev/AIC/dataset/test/AIC2026_PHASE_2_1000/visible")
    parser.add_argument("--ir-dir", default="/home/constant/CodeBuddy/Dev/AIC/dataset/test/AIC2026_PHASE_2_1000/infrared")
    parser.add_argument("--output", default="outputs/analysis/audit_comparison", help="Output directory")
    parser.add_argument("--iou-exact", type=float, default=0.25)
    parser.add_argument("--iou-conflict", type=float, default=0.30)
    parser.add_argument("--focus-class", type=int, default=None, help="Optional class ID focus (e.g. 7 for ball)")
    parser.add_argument("--render-panels", action="store_true", help="Render 3-column PNG panels")

    args = parser.parse_args()

    comp = MultimodalComparator(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        name_baseline=args.name_baseline,
        name_candidate=args.name_candidate,
        vis_dir=args.vis_dir,
        ir_dir=args.ir_dir,
        iou_exact=args.iou_exact,
        iou_conflict=args.iou_conflict,
        focus_class=args.focus_class
    )
    comp.export_all(args.output, render_panels=args.render_panels)


if __name__ == '__main__':
    main()
