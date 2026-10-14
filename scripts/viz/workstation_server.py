#!/usr/bin/env python3
"""
AIC 2026 Multimodal Workstation HTTP Server
===========================================
Enhanced local web server providing:
1. High-speed static file serving with CORS & proper cache-control headers.
2. Real-time On-Demand Neural Super-Resolution API (/api/sr?stem=...):
   - Generates Real-ESRGAN 4x enhanced visible and infrared images on the fly if not yet cached.
3. Bidirectional Human Audit Sync API (/api/audit/save, /api/audit/load):
   - Auto-saves human annotations to `outputs/analysis/audit/human_audits.json`.
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import zipfile

ROOT_DIR = '/home/constant/CodeBuddy/Dev/AIC'
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

AUDIT_FILE = os.path.join(ROOT_DIR, 'outputs/analysis/audit/human_audits.json')
MANUAL_ANNOTATIONS_FILE = os.path.join(ROOT_DIR, 'outputs/analysis/audit/manual_annotations.json')
ALIGN_SUMMARY_FILE = os.path.join(ROOT_DIR, 'outputs/metrics/test_alignment_summary.json')
VIS_DIR = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/visible')
IR_DIR = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/infrared')
VIS_SR_DIR = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/visible_sr4x')
IR_SR_DIR = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/infrared_sr4x')
F5_ZIP_PATH = os.path.join(ROOT_DIR, 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip')
V26_ZIP_PATH = os.path.join(ROOT_DIR, 'outputs/submissions/submissions_v26/submission_sovereign_v26_rare45_common50_clean7241.zip')

CLASS_NAMES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}
CLASS_ZH = {
    0: '人', 1: '船', 2: '动物', 3: '座椅', 4: '标牌', 5: '单车',
    6: '汽车', 7: '球', 8: '路灯', 9: '垃圾桶', 10: '无人机', 11: '三轮车'
}

_f5_cache = None
_v26_cache = None

def get_v26_preds():
    global _v26_cache
    if _v26_cache is None:
        if os.path.exists(V26_ZIP_PATH):
            _v26_cache = parse_boxes_from_zip(V26_ZIP_PATH)
        else:
            _v26_cache = {}
    return _v26_cache

def generate_gemini_prompt(stem, manual_boxes, f5_boxes, v26_boxes, align_info, img_w, img_h):
    align_desc = "原生几何对齐"
    if align_info and (align_info.get('global_applied') or align_info.get('targets', 0) > 0):
        align_desc = f"ECC偏移: dx={align_info.get('dx', 0)}px, dy={align_info.get('dy', 0)}px | 目标吸附: {align_info.get('targets', 0)}个 | ΔCC={align_info.get('delta_cc', 0)}"

    lines = [
        f"# 🤖 AIC 2026 多模态目标检测异常诊断报告 (由工作台人工标注触发)",
        f"",
        f"- **图像帧编号**: `{stem}` (图像分辨率: `{img_w} × {img_h} px`)",
        f"- **多模态传感器状态**: 包含可见光 (RGB)、红外 (IR)、几何深度 (Depth)",
        f"- **红外配准遥测**: `{align_desc}`",
        f"",
        f"## 一、专家人工真值纠偏 (Ground Truth & Human Feedback)",
    ]
    if manual_boxes:
        for i, mb in enumerate(manual_boxes, 1):
            cid = mb.get('class_id', 0)
            cname = CLASS_NAMES.get(cid, 'unknown')
            czh = CLASS_ZH.get(cid, str(cid))
            b = mb.get('box', [0, 0, 0, 0])
            tag = mb.get('tag_zh', mb.get('tag', 'GT新增真值'))
            note = mb.get('note', '').strip() or '无专家额外描述'
            lines.append(f"{i}. **[{tag}] {czh} ({cname})**")
            lines.append(f"   - 归一化中心与宽高 `[cx, cy, w, h]`: `[{b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}, {b[3]:.4f}]`")
            lines.append(f"   - 像素坐标 `[x1, y1, x2, y2]`: `[{int((b[0]-b[2]/2)*img_w)}, {int((b[1]-b[3]/2)*img_h)}, {int((b[0]+b[2]/2)*img_w)}, {int((b[1]+b[3]/2)*img_h)}]`")
            lines.append(f"   - 专家复核诊断说明: *\"{note}\"*")
    else:
        lines.append("*(当前帧尚未手动勾选新增标注，请依据基准与候选模型的分歧目标进行推断)*")

    lines.extend([
        f"",
        f"## 二、模型检出对比 (Top-Performing Models Detection)",
        f"- **F5 标杆基准 (57.266)**: 检出 {len(f5_boxes)} 个目标" + (f" (前置类别: {', '.join(CLASS_ZH.get(b[0], str(b[0])) for b in f5_boxes[:6])}...)" if f5_boxes else " (无检出)"),
        f"- **Sovereign V26 黄金候选 (58.10+)**: 检出 {len(v26_boxes)} 个目标" + (f" (前置类别: {', '.join(CLASS_ZH.get(b[0], str(b[0])) for b in v26_boxes[:6])}...)" if v26_boxes else " (无检出)"),
        f"",
        f"## 三、请求 Gemini 视觉大模型进行专家级多模态认知研判:",
        f"1. **【真实物理存在性验证】**：请仔细对比红外热辐射、可见光边缘与深度通道，确认人工标注区域是否存在真实有效物理目标？是否存在低光照隐蔽、伪装或热扩散反差？",
        f"2. **【算法漏检/误报归因】**：为什么主流检测器（如 DFINE / DeIMv2 / RF-DETR）在此处产生漏检（FN）或分类混淆？是否属于小尺度密集重叠、强光眩光干扰、或跨模态通道融合冲突？",
        f"3. **【模型迭代行动指南】**：请给出针对该类难例的特征提取与训练重加权建议（例如：调整多模态交叉注意力权重、针对特定类别进行红外自适应对比度增强、或扩充小目标合成数据）。",
    ])
    return "\n".join(lines)

def parse_boxes_from_zip(zip_p):
    d = {}
    with zipfile.ZipFile(zip_p, 'r') as zf:
        for fname in sorted(zf.namelist()):
            if not fname.endswith('.txt'): continue
            stem = os.path.basename(fname).rsplit('.', 1)[0]
            content = zf.read(fname).decode('utf-8', errors='ignore').strip()
            boxes = []
            if content:
                for line in content.split('\n'):
                    p = line.strip().split()
                    if len(p) >= 6:
                        cid = int(p[0])
                        cx, cy, w, h, sc = map(float, p[1:6])
                        boxes.append([cid, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4), round(sc, 3)])
            boxes.sort(key=lambda x: x[5], reverse=True)
            d[stem] = boxes[:60]
    return d

def get_f5_preds():
    global _f5_cache
    if _f5_cache is None:
        if os.path.exists(F5_ZIP_PATH):
            _f5_cache = parse_boxes_from_zip(F5_ZIP_PATH)
        else:
            _f5_cache = {}
    return _f5_cache

def compute_iou(b1, b2):
    x1_1, x2_1 = b1[1] - b1[3]/2, b1[1] + b1[3]/2
    y1_1, y2_1 = b1[2] - b1[4]/2, b1[2] + b1[4]/2
    x1_2, x2_2 = b2[1] - b2[3]/2, b2[1] + b2[3]/2
    y1_2, y2_2 = b2[2] - b2[4]/2, b2[2] + b2[4]/2
    ix = max(0.0, min(x2_1, x2_2) - max(x1_1, x1_2))
    iy = max(0.0, min(y2_1, y2_2) - max(y1_1, y1_2))
    inter = ix * iy
    union = b1[3]*b1[4] + b2[3]*b2[4] - inter
    return inter / union if union > 0 else 0.0

def quick_match(bf, bg, iou_exact=0.25, iou_conf=0.30):
    matched_f, matched_g = {}, {}
    pairs_exact = []
    for i, f in enumerate(bf):
        for j, g in enumerate(bg):
            if f[0] == g[0]:
                iou = compute_iou(f, g)
                if iou >= iou_exact:
                    pairs_exact.append((iou, i, j))
    pairs_exact.sort(key=lambda x: x[0], reverse=True)
    for iou, i, j in pairs_exact:
        if i not in matched_f and j not in matched_g:
            matched_f[i] = ('exact', j, iou)
            matched_g[j] = ('exact', i, iou)

    pairs_conf = []
    for i, f in enumerate(bf):
        if i in matched_f: continue
        for j, g in enumerate(bg):
            if j in matched_g or f[0] == g[0]: continue
            iou = compute_iou(f, g)
            if iou >= iou_conf:
                pairs_conf.append((iou, i, j))
    pairs_conf.sort(key=lambda x: x[0], reverse=True)
    for iou, i, j in pairs_conf:
        if i not in matched_f and j not in matched_g:
            matched_f[i] = ('conflict', j, iou)
            matched_g[j] = ('conflict', i, iou)

    n_exact = sum(1 for v in matched_f.values() if v[0] == 'exact')
    n_conf = sum(1 for v in matched_f.values() if v[0] == 'conflict')
    n_missed = len(bf) - len(matched_f)
    n_fp = len(bg) - len(matched_g)
    return n_exact, n_conf, n_missed, n_fp

def scan_available_zips():
    candidates = []
    search_dirs = [
        os.path.join(ROOT_DIR, 'outputs'),
        os.path.join(ROOT_DIR, 'work_dir'),
        '/home/constant/Downloads'
    ]
    seen = set()
    for sdir in search_dirs:
        if not os.path.exists(sdir): continue
        for root, dirs, files in os.walk(sdir):
            rel = os.path.relpath(root, sdir)
            if rel.count(os.sep) > 3:
                continue
            for f in sorted(files):
                if f.endswith('.zip') and ('submission' in f.lower() or 'pred' in f.lower()):
                    full_p = os.path.join(root, f)
                    if full_p not in seen:
                        seen.add(full_p)
                        try:
                            sz = os.path.getsize(full_p) / (1024 * 1024)
                            candidates.append({
                                'path': full_p,
                                'name': f,
                                'folder': os.path.basename(root),
                                'size_mb': round(sz, 2)
                            })
                        except:
                            pass
    candidates.sort(key=lambda x: x['name'])
    return candidates

# Global enhancer instance (lazy-loaded on first request to avoid startup delay)
_enhancer = None
_enhancer_lock = threading.Lock()

def get_enhancer():
    global _enhancer
    with _enhancer_lock:
        if _enhancer is None:
            from scripts.viz.neural_superres_engine import RealESRGANEnhancer
            _enhancer = RealESRGANEnhancer()
        return _enhancer

class WorkstationRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

    def end_headers(self):
        # Enable CORS and disable overly aggressive caching for dynamic development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # Route: /api/sr_status?stem=00000010
        if parsed.path == '/api/sr_status':
            query = urllib.parse.parse_qs(parsed.query)
            stem = query.get('stem', [''])[0]
            if not stem:
                self.send_json_response({'error': 'Missing stem'}, 400)
                return
                
            vis_sr_path = os.path.join(VIS_SR_DIR, f"{stem}.jpg")
            ir_sr_path = os.path.join(IR_SR_DIR, f"{stem}.jpg")
            ready = os.path.exists(vis_sr_path)
            
            self.send_json_response({
                'stem': stem,
                'ready': ready,
                'vis_url': f"/dataset/test/AIC2026_PHASE_2_1000/visible_sr4x/{stem}.jpg" if ready else None,
                'ir_url': f"/dataset/test/AIC2026_PHASE_2_1000/infrared_sr4x/{stem}.jpg" if os.path.exists(ir_sr_path) else None
            })
            return

        # Route: /api/generate_sr?stem=00000010
        if parsed.path == '/api/generate_sr':
            query = urllib.parse.parse_qs(parsed.query)
            stem = query.get('stem', [''])[0]
            if not stem:
                self.send_json_response({'error': 'Missing stem'}, 400)
                return
                
            vis_sr_path = os.path.join(VIS_SR_DIR, f"{stem}.jpg")
            ir_sr_path = os.path.join(IR_SR_DIR, f"{stem}.jpg")
            
            try:
                enhancer = get_enhancer()
                os.makedirs(VIS_SR_DIR, exist_ok=True)
                os.makedirs(IR_SR_DIR, exist_ok=True)
                
                # Check visible
                vis_src = os.path.join(VIS_DIR, f"{stem}.jpg")
                if not os.path.exists(vis_src):
                    vis_src = os.path.join(VIS_DIR, f"{stem}.png")
                    
                if not os.path.exists(vis_sr_path) and os.path.exists(vis_src):
                    enhancer.enhance_file(vis_src, vis_sr_path)

                # Check infrared
                ir_src = os.path.join(IR_DIR, f"{stem}.jpg")
                if not os.path.exists(ir_src):
                    ir_src = os.path.join(IR_DIR, f"{stem}.png")
                    
                if not os.path.exists(ir_sr_path) and os.path.exists(ir_src):
                    enhancer.enhance_file(ir_src, ir_sr_path)

                self.send_json_response({
                    'status': 'ok',
                    'stem': stem,
                    'vis_url': f"/dataset/test/AIC2026_PHASE_2_1000/visible_sr4x/{stem}.jpg",
                    'ir_url': f"/dataset/test/AIC2026_PHASE_2_1000/infrared_sr4x/{stem}.jpg" if os.path.exists(ir_sr_path) else None
                })
            except Exception as e:
                self.send_json_response({'error': str(e)}, 500)
            return

        # Route: /api/depth_view?stem=00000010&cmap=turbo
        if parsed.path == '/api/depth_view':
            query = urllib.parse.parse_qs(parsed.query)
            stem = query.get('stem', [''])[0]
            cmap = query.get('cmap', ['turbo'])[0]
            if not stem:
                self.send_json_response({'error': 'Missing stem'}, 400)
                return

            depth_dir = os.path.join(ROOT_DIR, 'dataset/test/AIC2026_PHASE_2_1000/depth')
            dep_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                p = os.path.join(depth_dir, f"{stem}{ext}")
                if os.path.exists(p):
                    dep_path = p
                    break

            if not dep_path:
                self.send_json_response({'error': f'Depth not found for {stem}'}, 404)
                return

            try:
                import cv2
                from scripts.lib.depth_utils import colorize_depth
                colored_bgr = colorize_depth(dep_path, colormap=cmap)
                success, enc = cv2.imencode('.jpg', colored_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not success:
                    self.send_json_response({'error': 'Failed to encode depth image'}, 500)
                    return

                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(enc)))
                self.send_header('Cache-Control', 'public, max-age=86400')
                self.end_headers()
                self.wfile.write(enc.tobytes())
                return
            except Exception as e:
                self.send_json_response({'error': str(e)}, 500)
                return

        # Route: /api/audit/load
        if parsed.path == '/api/audit/load':
            if os.path.exists(AUDIT_FILE):
                try:
                    with open(AUDIT_FILE, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.send_json_response(data)
                    return
                except Exception as e:
                    pass
            self.send_json_response({})
            return

        # Route: /api/manual_annotations/load
        if parsed.path == '/api/manual_annotations/load':
            if os.path.exists(MANUAL_ANNOTATIONS_FILE):
                try:
                    with open(MANUAL_ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.send_json_response(data)
                    return
                except:
                    pass
            self.send_json_response({'annotations': []})
            return

        # Route: /api/gemini/diagnostic_context?stem=00000076
        if parsed.path == '/api/gemini/diagnostic_context':
            query = urllib.parse.parse_qs(parsed.query)
            stem = query.get('stem', [''])[0]
            if not stem:
                self.send_json_response({'error': 'Missing stem'}, 400)
                return

            # 1. Load manual annotations for this stem
            man_boxes = []
            if os.path.exists(MANUAL_ANNOTATIONS_FILE):
                try:
                    with open(MANUAL_ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
                        m_data = json.load(f)
                        man_boxes = [x for x in m_data.get('annotations', []) if x.get('stem') == stem]
                except:
                    pass

            # 2. Load model predictions
            f5_all = get_f5_preds()
            v26_all = get_v26_preds()
            f5_boxes = f5_all.get(stem, [])
            v26_boxes = v26_all.get(stem, [])

            # 3. Load image dimensions
            img_w, img_h = 640, 360
            for ext in ['.jpg', '.png']:
                vp = os.path.join(VIS_DIR, f"{stem}{ext}")
                if os.path.exists(vp):
                    try:
                        import cv2
                        im = cv2.imread(vp)
                        if im is not None:
                            img_h, img_w = im.shape[:2]
                    except:
                        pass
                    break

            # 4. Load alignment info
            align_info = None
            if os.path.exists(ALIGN_SUMMARY_FILE):
                try:
                    with open(ALIGN_SUMMARY_FILE, 'r', encoding='utf-8') as f:
                        al_doc = json.load(f)
                        for it in al_doc.get('details', []):
                            if it.get('stem') == stem:
                                align_info = it
                                break
                except:
                    pass

            # 5. Format prompt
            prompt_md = generate_gemini_prompt(stem, man_boxes, f5_boxes, v26_boxes, align_info, img_w, img_h)

            self.send_json_response({
                'status': 'ok',
                'stem': stem,
                'width': img_w,
                'height': img_h,
                'manual_annotations': man_boxes,
                'f5_count': len(f5_boxes),
                'v26_count': len(v26_boxes),
                'alignment_info': align_info,
                'prompt_markdown': prompt_md,
                'vis_url': f"/dataset/test/AIC2026_PHASE_2_1000/visible/{stem}.jpg",
                'ir_url': f"/dataset/test/AIC2026_PHASE_2_1000/infrared_aligned/{stem}.jpg",
                'depth_url': f"/dataset/test/AIC2026_PHASE_2_1000/depth/{stem}.jpg"
            })
            return

        # Route: /api/model/list_available_zips
        if parsed.path == '/api/model/list_available_zips':
            zips = scan_available_zips()
            self.send_json_response({'status': 'ok', 'zips': zips})
            return

        # Standard static file serving
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # Route: /api/audit/save
        if parsed.path == '/api/audit/save':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length).decode('utf-8')
                data = json.loads(body)
                os.makedirs(os.path.dirname(AUDIT_FILE), exist_ok=True)
                with open(AUDIT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self.send_json_response({'status': 'saved', 'count': len(data)})
            except Exception as e:
                self.send_json_response({'error': str(e)}, 500)
            return

        # Route: /api/manual_annotations/save
        if parsed.path == '/api/manual_annotations/save':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length).decode('utf-8')
                data = json.loads(body)
                os.makedirs(os.path.dirname(MANUAL_ANNOTATIONS_FILE), exist_ok=True)
                with open(MANUAL_ANNOTATIONS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self.send_json_response({'status': 'saved', 'count': len(data.get('annotations', []))})
            except Exception as e:
                self.send_json_response({'error': str(e)}, 500)
            return

        # Route: /api/model/parse_and_add
        if parsed.path == '/api/model/parse_and_add':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length).decode('utf-8')
                data = json.loads(body)

                model_id = data.get('id', '').strip()
                model_name = data.get('name', '').strip() or model_id
                model_score = data.get('score', '').strip() or '-'
                zip_path = data.get('zip_path', '').strip()

                if not model_id:
                    self.send_json_response({'error': 'Missing model id'}, 400)
                    return
                if not zip_path or not os.path.exists(zip_path):
                    self.send_json_response({'error': f'Zip file not found: {zip_path}'}, 400)
                    return

                new_preds = parse_boxes_from_zip(zip_path)
                f5_preds = get_f5_preds()
                all_stems = sorted(list(set(list(f5_preds.keys()) + list(new_preds.keys()))))
                if not all_stems:
                    all_stems = [f"{i:08d}" for i in range(1000)]

                diff_list = []
                for stem in all_stems:
                    bf = [b for b in f5_preds.get(stem, []) if b[5] >= 0.45]
                    bg = [b for b in new_preds.get(stem, []) if b[5] >= 0.45]
                    n_exact, n_conf, n_missed, n_fp = quick_match(bf, bg)
                    delta = len(bg) - len(bf)
                    diff_list.append({
                        'stem': stem,
                        'f_cnt': len(bf),
                        'c_cnt': len(bg),
                        'exact': n_exact,
                        'conf': n_conf,
                        'missed': n_missed,
                        'fp': n_fp,
                        'delta': delta,
                        'is_pure_miss': n_missed > 0 and n_fp == 0 and n_conf == 0,
                        'is_pure_fp': n_fp > 0 and n_missed == 0 and n_conf == 0,
                        'is_mixed': n_missed > 0 and n_fp > 0,
                        'is_conf': n_conf > 0
                    })

                total_boxes = sum(len(bs) for bs in new_preds.values())
                colors = ['#38bdf8', '#a855f7', '#ec4899', '#f97316', '#14b8a6', '#6366f1']
                chosen_color = colors[abs(hash(model_id)) % len(colors)]

                self.send_json_response({
                    'status': 'ok',
                    'model': {
                        'id': model_id,
                        'name': model_name,
                        'score': model_score,
                        'badge': f"{total_boxes:,} 框 / 动态接入",
                        'path': zip_path,
                        'color': chosen_color
                    },
                    'predictions': new_preds,
                    'diff_summary': diff_list
                })
            except Exception as e:
                self.send_json_response({'error': str(e)}, 500)
            return

        self.send_json_response({'error': 'Not found'}, 404)

    def send_json_response(self, obj, status_code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def run_server(port=8088):
    server = HTTPServer(('0.0.0.0', port), WorkstationRequestHandler)
    print(f"============================================================")
    print(f"AIC 2026 Multimodal Workstation Server running at:")
    print(f"  http://localhost:{port}/")
    print(f"  Neural Super-Resolution API ready (/api/generate_sr)")
    print(f"============================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server...")
        server.server_close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="AIC 2026 Workstation Server")
    parser.add_argument("pos_port", nargs="?", type=int, default=None, help="Positional port number")
    parser.add_argument("--port", "-p", type=int, default=8088, help="Port number")
    args = parser.parse_args()
    port = args.pos_port if args.pos_port is not None else args.port
    run_server(port)
