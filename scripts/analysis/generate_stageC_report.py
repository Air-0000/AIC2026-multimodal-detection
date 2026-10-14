#!/usr/bin/env python3
"""Generate a self-contained HTML report (inline SVG, no CDN) for Stage C results."""
import json, math, datetime

SRC = "/home/constant/CodeBuddy/Dev/AIC/outputs/metrics/stageC_results.json"
OUT = "/home/constant/CodeBuddy/Dev/AIC/outputs/reports/stageC_report.html"

d = json.load(open(SRC))
curve = d["curve"]
night = d["night"]

# ---------- mAP curve ----------
xs = [c["epoch"] for c in curve]
ys = [c["mAP"] for c in curve]
y50 = [c["AP50"] for c in curve]
y75 = [c["AP75"] for c in curve]
xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys + y50 + y75), max(ys + y50 + y75)
ymin = math.floor((ymin - 0.01) * 100) / 100
ymax = math.ceil((ymax + 0.01) * 100) / 100

W, H = 760, 320
ML, MR, MT, MB = 56, 16, 18, 40
def sx(x):
    return ML + (x - xmin) / (xmax - xmin) * (W - ML - MR)
def sy(y):
    return MT + (ymax - y) / (ymax - ymin) * (H - MT - MB)

def poly(vals, color, width=2.0, dash=None):
    pts = " ".join("%.1f,%.1f" % (sx(xs[i]), sy(vals[i])) for i in range(len(xs)))
    da = ' stroke-dasharray="5 4"' if dash else ""
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{da} points="{pts}"/>'

best = max(curve, key=lambda c: c["mAP"])
bx, by = sx(best["epoch"]), sy(best["mAP"])

# y gridlines
ygrid = ""
yt = ymin
step = 0.02
while yt <= ymax + 1e-9:
    yv = round(yt, 2)
    yg = sy(yv)
    ygrid += f'<line x1="{ML}" y1="{yg:.1f}" x2="{W-MR}" y2="{yg:.1f}" stroke="#e3e6ea" stroke-width="1"/>'
    ygrid += f'<text x="{ML-6}" y="{yg+3:.1f}" text-anchor="end" font-size="10" fill="#6b7280">{yv:.2f}</text>'
    yt += step

# x ticks every 5
xticks = ""
for x in range(0, xmax + 1, 5):
    xticks += f'<text x="{sx(x):.1f}" y="{H-MB+16}" text-anchor="middle" font-size="10" fill="#6b7280">{x}</text>'

curve_svg = f'''
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  {ygrid}
  <line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H-MB}" stroke="#9aa3af" stroke-width="1"/>
  <line x1="{ML}" y1="{H-MB}" x2="{W-MR}" y2="{H-MB}" stroke="#9aa3af" stroke-width="1"/>
  {xticks}
  {poly(y50, "#9ca3af", 1.4, dash=True)}
  {poly(y75, "#60a5fa", 1.6, dash=True)}
  {poly(ys, "#2563eb", 2.2)}
  <circle cx="{bx:.1f}" cy="{by:.1f}" r="4.5" fill="#dc2626"/>
  <text x="{bx:.1f}" y="{by-9:.1f}" text-anchor="middle" font-size="10.5" fill="#dc2626">best ep{best["epoch"]} {best["mAP"]:.4f}</text>
  <text x="{W-MR}" y="{MT+4}" text-anchor="end" font-size="10" fill="#2563eb">mAP</text>
  <text x="{W-MR}" y="{MT+15}" text-anchor="end" font-size="10" fill="#60a5fa">AP75(虚)</text>
  <text x="{W-MR}" y="{MT+26}" text-anchor="end" font-size="10" fill="#9ca3af">AP50(点)</text>
</svg>'''

# ---------- night robustness (log scale) ----------
lv = night
vals = [n["mAP"] for n in lv]
labels = [n["level"] for n in lv]
lo_v, hi_v = 0.003, 0.6
def ly(v):
    return MT + (math.log10(hi_v) - math.log10(max(v, lo_v))) / (math.log10(hi_v) - math.log10(lo_v)) * (H - MT - MB)
NW = 760
bw = (NW - ML - MR) / len(lv) * 0.62
bars = ""
for i, (lab, v) in enumerate(zip(labels, vals)):
    cx = ML + (i + 0.5) * (NW - ML - MR) / len(lv)
    top = ly(v)
    col = "#16a34a" if i < 5 else ("#f59e0b" if i < 6 else "#dc2626")
    bars += f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{H-MB-top:.1f}" fill="{col}" rx="2"/>'
    bars += f'<text x="{cx:.1f}" y="{top-5:.1f}" text-anchor="middle" font-size="9.5" fill="#374151">{v:.3f}</text>'
    bars += f'<text x="{cx:.1f}" y="{H-MB+15}" text-anchor="middle" font-size="9.5" fill="#374151">{lab}</text>'
# log gridlines
lgr = ""
for e in [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0]:
    v = 10 ** e
    if lo_v <= v <= hi_v:
        yg = ly(v)
        lgr += f'<line x1="{ML}" y1="{yg:.1f}" x2="{NW-MR}" y2="{yg:.1f}" stroke="#e3e6ea"/>'
        lgr += f'<text x="{ML-6}" y="{yg+3:.1f}" text-anchor="end" font-size="9.5" fill="#6b7280">{v:.3f}</text>'

night_svg = f'''
<svg viewBox="0 0 {NW} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  {lgr}
  <line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H-MB}" stroke="#9aa3af"/>
  <line x1="{ML}" y1="{H-MB}" x2="{NW-MR}" y2="{H-MB}" stroke="#9aa3af"/>
  {bars}
  <text x="{ML}" y="{MT-4}" font-size="10.5" fill="#374151">mAP@0.5:0.95 （对数纵轴；绿=白天端→红=深夜端）</text>
</svg>'''

# ---------- table ----------
rows = "".join(
    f"<tr><td>{c['epoch']}</td><td>{c['mAP']:.4f}</td><td>{c['AP50']:.4f}</td><td>{c['AP75']:.4f}</td>"
    f"<td>{c['lr']:.2e}</td></tr>" for c in curve)
night_rows = "".join(
    f"<tr><td>{n['level']}</td><td>{n['mAP']:.4f}</td></tr>" for n in night)

gen = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

html = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage C 最终评测报告</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;color:#1f2937;background:#fff}}
 h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:22px 0 8px;color:#111827}}
 .sub{{color:#6b7280;font-size:12px;margin-bottom:14px}}
 .card{{border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px;margin:10px 0;background:#fafbfc}}
 table{{border-collapse:collapse;font-size:12px;margin-top:6px}} th,td{{border:1px solid #e5e7eb;padding:3px 8px;text-align:center}}
 th{{background:#f3f4f6}} .note{{font-size:12px;color:#374151;line-height:1.6}}
 .tag{{display:inline-block;background:#ecfdf5;color:#047857;border:1px solid #a7f3d0;border-radius:6px;padding:2px 8px;font-size:12px;margin:2px 4px 2px 0}}
 .warn{{background:#fffbeb;color:#92400e;border-color:#fde68a}}
</style></head><body>
<h1>Stage C 三阶段训练 — 最终评测报告</h1>
<div class="sub">生成于 {gen} · 模型 vitl800 (DINOv3-L/16 + RGB→IR 融合) · 评测口径 = val200 (200 张/1523 框, 12 类)</div>

<div class="card">
  <h2>1. 逐 epoch 验证 mAP 曲线</h2>
  <div class="note">flatcosine 调度：warmup→flat(恒 lr 8.87e-6)→cosine 衰减(ep20–37)→no-aug(ep37–39)。
  最佳 <b>0.5265 @ ep35</b>（AP75 从 ep25 的 0.727 升到 0.777，是本轮最实的改善）。</div>
  {curve_svg}
</div>

<div class="card">
  <h2>2. 10 档亮度鲁棒性（checkpoint0039 = last, ep39）</h2>
  <div class="note">合成降亮度压力测试（测试集本身全白天，见 MEMORY §1）。
  <b>白天端增益≈0（IR 冗余）</b>；在 night05→night01 之间出现断崖（0.422 → 0.128）——RGB 被压暗到一定程度后融合也救不回。</div>
  {night_svg}
</div>

<div class="card">
  <h2>3. 关键结论与产物</h2>
  <div class="note">
    <span class="tag">训练完成</span> Stage C ep39 跑完，checkpoint0039 = last。
    <span class="tag">10 档 night 评测</span> 已随脚本自动跑完。
    <span class="tag">SWA 权重已构建</span> <code>swa_stgC_ep24_39.pth</code>（对 ep24/29/34/39 收敛轨道平均，384.2M 参数）。
    <br><br>
    <b>口径警告</b>（MEMORY §5.2）：历史 val 数不可复现，跨模型比较必须用同一条冻结路径重测；best 0.5265 与基线 combo 0.5241 的差距（0.24pt）落在 boot CI 噪声内，<b>不能断言 Stage C 显著优于基线</b>。
    <br><br>
    <b>argmax-best 在选择上不可靠</b>（MEMORY §5.6 奇偶留出：Stage B 选中的最佳反而最差）。因此本报告优先采用 <b>SWA / 末轮</b> 而非峰值快照。
  </div>
</div>

<div class="card">
  <h2>4. 进行中的 GPU 分析（后台任务 ARXw4T）</h2>
  <div class="note">
    <span class="tag">损失地形</span> loss_landscape.py 已修 device bug 并重跑（random + PCA 双切片 + 插值势垒）。
    <span class="tag">配对 bootstrap</span> 对 best / last / SWA 三模型做 200 次配对重采样，给出差值的 95% CI（直接回答「SWA 是否显著优于 argmax-best」）。
    <br>完成后产物：<code>outputs/loss_landscape_stageC.png/.json</code>、<code>outputs/metrics/val_ci_bootstrap.json</code>。
  </div>
</div>

<div class="card">
  <h2>5. 逐 epoch 明细</h2>
  <table><tr><th>epoch</th><th>mAP</th><th>AP50</th><th>AP75</th><th>lr</th></tr>{rows}</table>
</div>

<div class="card">
  <h2>6. Night 鲁棒性明细</h2>
  <table><tr><th>level</th><th>mAP@0.5:0.95</th></tr>{night_rows}</table>
</div>
</body></html>'''

open(OUT, "w", encoding="utf-8").write(html)
print("wrote", OUT, len(html), "bytes")
