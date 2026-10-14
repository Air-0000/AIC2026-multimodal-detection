#!/usr/bin/env python3
"""
Generate Interactive Discrepancy Reviewer for Sovereign V28 vs F5 Gold Benchmark
Outputs: outputs/analysis/discrepancies_f5_vs_v28/index.html
"""

import os
import json
import zipfile

import sys
ROOT = '/home/constant/CodeBuddy/Dev/AIC'
F5_ZIP  = os.path.join(ROOT, 'work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip')
V28_ZIP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'outputs/submissions/submissions_v28/submission_sovereign_v28_3d_consolidated.zip')
OUTPUT_ROOT = os.path.join(ROOT, 'outputs/analysis/discrepancies_f5_vs_v28')

CLASS_NAMES = {
    0: 'person', 1: 'boat', 2: 'animal', 3: 'seat', 4: 'sign', 5: 'bicycle',
    6: 'car', 7: 'ball', 8: 'light', 9: 'garbage can', 10: 'uav', 11: 'tricycle'
}


def load_submission_zip(zip_path):
    d = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith('.txt'):
                continue
            stem = os.path.splitext(os.path.basename(name))[0]
            content = zf.read(name).decode('utf-8').strip()
            boxes = []
            if content:
                for line in content.replace('\r\n', '\n').split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        cid = int(parts[0])
                        cx, cy, w, h, sc = map(float, parts[1:6])
                        boxes.append((cid, cx, cy, w, h, sc))
            boxes.sort(key=lambda x: x[5], reverse=True)
            d[stem] = boxes[:60]
    return d


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


def match_boxes(bf, bg):
    pairs_exact = []
    for i, b_f in enumerate(bf):
        for j, b_g in enumerate(bg):
            if b_f[0] == b_g[0]:
                iou = compute_iou(b_f, b_g)
                if iou >= 0.25:
                    pairs_exact.append((iou, i, j))
    pairs_exact.sort(key=lambda x: x[0], reverse=True)
    
    matched_f, matched_g = {}, {}
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
                iou = compute_iou(b_f, b_g)
                if iou >= 0.30:
                    pairs_conflict.append((iou, i, j))
    pairs_conflict.sort(key=lambda x: x[0], reverse=True)
    
    for iou, i, j in pairs_conflict:
        if i not in matched_f and j not in matched_g:
            matched_f[i] = ('conflict', j, iou)
            matched_g[j] = ('conflict', i, iou)
            
    return matched_f, matched_g


def build_viewer():
    print(f"[+] Loading predictions...")
    f5_data = load_submission_zip(F5_ZIP)
    v28_data = load_submission_zip(V28_ZIP)

    stems = sorted(set(list(f5_data.keys()) + list(v28_data.keys())))
    
    report_dict = {
        'all': [],
        'conflict': [],
        'large': [],
        'medium': [],
        'single': []
    }

    for stem in stems:
        bf = f5_data.get(stem, [])
        bg = v28_data.get(stem, [])
        mf, mg = match_boxes(bf, bg)
        num_conflicts = sum(1 for v in mf.values() if v[0] == 'conflict')
        num_exact = sum(1 for v in mf.values() if v[0] == 'exact')
        num_f_only = len(bf) - len(mf)
        num_g_only = len(bg) - len(mg)
        delta = num_conflicts + num_f_only + num_g_only

        if delta == 0:
            continue

        summary_parts = []
        if num_conflicts > 0:
            for i, v in mf.items():
                if v[0] == 'conflict':
                    j, iou = v[1], v[2]
                    summary_parts.append(f"⚠️ 冲突: F5:{CLASS_NAMES[bf[i][0]]} ⚔️ V28:{CLASS_NAMES[bg[j][0]]} (IoU={iou:.2f})")
        if num_f_only > 0:
            f_only_names = [f"{CLASS_NAMES[bf[i][0]]}" for i in range(len(bf)) if i not in mf]
            summary_parts.append(f"🟦 F5独有 {num_f_only}框 ({', '.join(f_only_names[:4])})")
        if num_g_only > 0:
            g_only_names = [f"{CLASS_NAMES[bg[j][0]]}" for j in range(len(bg)) if j not in mg]
            summary_parts.append(f"🟥 V28独有 {num_g_only}框 ({', '.join(g_only_names[:4])})")

        item_meta = {
            'stem': stem,
            'f5_cnt': len(bf),
            'v28_cnt': len(bg),
            'exact_cnt': num_exact,
            'conf_cnt': num_conflicts,
            'delta': delta,
            'summary': " | ".join(summary_parts)
        }

        report_dict['all'].append(item_meta)
        if num_conflicts > 0:
            report_dict['conflict'].append(item_meta)
        elif delta >= 5:
            report_dict['large'].append(item_meta)
        elif delta >= 2:
            report_dict['medium'].append(item_meta)
        elif delta == 1:
            report_dict['single'].append(item_meta)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIC2026 - F5 (58.105) vs Sovereign V28 (Disparity Awakening) 差异审阅器</title>
    <style>
        :root {{
            --bg-dark: #121418;
            --bg-card: #1c1f26;
            --border: #2d3340;
            --text-main: #f0f3f8;
            --text-dim: #9aa5b8;
            --primary: #6366f1;
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #10b981;
            --cyan: #06b6d4;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Noto Sans CJK SC", sans-serif; }}
        body {{ background: var(--bg-dark); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }}
        header {{ background: #0b0d10; border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }}
        .logo-title {{ display: flex; align-items: center; gap: 12px; }}
        .title {{ font-size: 18px; font-weight: 700; color: #fff; }}
        .stats-badge {{ background: #232732; border: 1px solid var(--border); padding: 4px 10px; border-radius: 6px; font-size: 13px; color: var(--text-dim); }}
        .controls {{ display: flex; gap: 8px; align-items: center; }}
        button {{ background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.15s; }}
        button:hover {{ background: #282d38; border-color: var(--primary); }}
        button.active {{ background: var(--primary); border-color: var(--primary); color: #fff; }}
        .filter-group {{ display: flex; gap: 6px; }}
        .nav-btn {{ background: #4f46e5; color: #fff; border: none; font-weight: 600; padding: 6px 16px; }}
        .nav-btn:hover {{ background: #4338ca; }}
        main {{ flex: 1; padding: 20px; display: flex; flex-direction: column; align-items: center; }}
        .info-panel {{ width: 100%; max-width: 1920px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 20px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
        .img-meta {{ font-size: 15px; font-weight: 600; }}
        .img-diff-summary {{ font-size: 14px; color: var(--warning); font-weight: 500; }}
        .image-container {{ width: 100%; max-width: 1920px; display: flex; justify-content: center; background: #08090b; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
        .image-container img {{ width: 100%; height: auto; display: block; object-fit: contain; }}
        .keyboard-hint {{ margin-top: 12px; font-size: 12px; color: var(--text-dim); text-align: center; }}
        kbd {{ background: #232732; border: 1px solid #3c4456; border-radius: 4px; padding: 2px 6px; font-size: 11px; }}
        .hub-link {{ color: #818cf8; text-decoration: none; font-size: 13px; font-weight: 600; padding: 6px 12px; border: 1px solid #4338ca; border-radius: 6px; }}
        .hub-link:hover {{ background: #4338ca; color: #fff; }}
    </style>
</head>
<body>
    <header>
        <div class="logo-title">
            <div class="title">AIC2026: F5 (58.105) vs Sovereign V28 差异审阅器</div>
            <div class="stats-badge" id="statsBadge">加载中...</div>
            <a href="../index.html" class="hub-link">← 返回统一对比工作站</a>
        </div>
        <div class="controls">
            <div class="filter-group">
                <button class="active" onclick="setFilter('all')">全部差异 (<span id="cntAll">0</span>)</button>
                <button onclick="setFilter('conflict')" style="color:#fb923c;">⚠️ 类别冲突 (<span id="cntConf">0</span>)</button>
                <button onclick="setFilter('large')" style="color:#f87171;">重大差异(Δ≥5) (<span id="cntLarge">0</span>)</button>
                <button onclick="setFilter('medium')">中度差异(Δ2~4) (<span id="cntMed">0</span>)</button>
                <button onclick="setFilter('single')">单一微差(Δ1) (<span id="cntSingle">0</span>)</button>
            </div>
            <div style="width: 1px; height: 24px; background: var(--border); margin: 0 4px;"></div>
            <button class="nav-btn" onclick="prevImg()">◀ 上一张 (←)</button>
            <span id="curIndex" style="font-size: 13px; font-weight: 600; min-width: 60px; text-align: center;">1 / 0</span>
            <button class="nav-btn" onclick="nextImg()">下一张 (→) ▶</button>
        </div>
    </header>

    <main>
        <div class="info-panel">
            <div class="img-meta" id="imgMeta">载入中...</div>
            <div class="img-diff-summary" id="imgDiffSummary"></div>
        </div>
        <div class="image-container">
            <img id="mainImg" src="" alt="对比面板" />
        </div>
        <div class="keyboard-hint">
            快捷键提示：键盘左右键 <kbd>←</kbd> <kbd>→</kbd> 快速切换图片，<kbd>1</kbd> 查看全量，<kbd>2</kbd> 查看类别冲突，<kbd>3</kbd> 查看重大差异
        </div>
    </main>

    <script>
        const report = {json.dumps(report_dict, ensure_ascii=False)};
        let currentFilter = 'all';
        let filteredList = [];
        let currentIndex = 0;

        function init() {{
            document.getElementById('cntAll').innerText = report.all.length;
            document.getElementById('cntConf').innerText = report.conflict.length;
            document.getElementById('cntLarge').innerText = report.large.length;
            document.getElementById('cntMed').innerText = report.medium.length;
            document.getElementById('cntSingle').innerText = report.single.length;
            setFilter('conflict');
        }}

        function setFilter(type) {{
            currentFilter = type;
            document.querySelectorAll('.filter-group button').forEach(b => b.classList.remove('active'));
            if(type === 'all') document.querySelectorAll('.filter-group button')[0].classList.add('active');
            if(type === 'conflict') document.querySelectorAll('.filter-group button')[1].classList.add('active');
            if(type === 'large') document.querySelectorAll('.filter-group button')[2].classList.add('active');
            if(type === 'medium') document.querySelectorAll('.filter-group button')[3].classList.add('active');
            if(type === 'single') document.querySelectorAll('.filter-group button')[4].classList.add('active');

            filteredList = report[type];
            currentIndex = 0;
            renderCurrent();
        }}

        function renderCurrent() {{
            if(filteredList.length === 0) {{
                document.getElementById('imgMeta').innerText = "当前筛选类别下无差异图片";
                document.getElementById('imgDiffSummary').innerText = "";
                document.getElementById('mainImg').src = "";
                document.getElementById('curIndex').innerText = "0 / 0";
                return;
            }}
            const item = filteredList[currentIndex];
            document.getElementById('curIndex').innerText = `${{currentIndex + 1}} / ${{filteredList.length}}`;
            document.getElementById('statsBadge').innerText = `当前类别: ${{currentFilter}} (共 ${{filteredList.length}} 张)`;
            document.getElementById('imgMeta').innerText = `【图号 ${{item.stem}}】 F5预测: ${{item.f5_cnt}} 框 | Sovereign V28预测: ${{item.v28_cnt}} 框 | 共识一致: ${{item.exact_cnt}} 框`;
            document.getElementById('imgDiffSummary').innerText = item.summary;
            document.getElementById('mainImg').src = 'all_discrepancies/' + item.stem + '.png';
        }}

        function nextImg() {{
            if(currentIndex < filteredList.length - 1) {{
                currentIndex++;
                renderCurrent();
            }}
        }}

        function prevImg() {{
            if(currentIndex > 0) {{
                currentIndex--;
                renderCurrent();
            }}
        }}

        window.addEventListener('keydown', (e) => {{
            if(e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') nextImg();
            if(e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') prevImg();
            if(e.key === '1') setFilter('all');
            if(e.key === '2') setFilter('conflict');
            if(e.key === '3') setFilter('large');
            if(e.key === '4') setFilter('medium');
            if(e.key === '5') setFilter('single');
        }});

        init();
    </script>
</body>
</html>
"""
    out_html = os.path.join(OUTPUT_ROOT, 'index.html')
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[✔] Successfully generated: {out_html}")


if __name__ == '__main__':
    build_viewer()
