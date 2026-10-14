#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把本管线模型(Stage A/C)与 MODEL_RECORDS 表A 的其他模型,
按 val200 mAP@50-95 (测试集口径) 画在一张分组对比图里.
输出: outputs/reports/model_map_comparison.html (内联 SVG, 无外部依赖)."""
import re, json, os

REPO = "/home/constant/CodeBuddy/Dev/AIC"
MR = os.path.join(REPO, "MODEL_RECORDS.md")
OUT = os.path.join(REPO, "outputs", "reports/model_map_comparison.html")

# 1) 解析表A: 各 section 标题 -> 模态组; 行 -> 模型
section_re = re.compile(r"^###\s+(.*?)\s*（.*$")
row_re = re.compile(
    r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([^|`]+)\s*\|\s*([^|]+)\|\s*$"
)

modality_colors = {
    "AIC / RGB": ("#3b82f6", "RGB"),
    "AIC / IR(纯红外)": ("#ef4444", "纯IR"),
    "AIC / RGB+IR": ("#14b8a6", "RGB+IR"),
    "AIC / RGB+Depth": ("#a855f7", "RGB+Depth"),
    "AIC / RGB+IR+Depth": ("#22c55e", "RGB+IR+Depth"),
    "BDD / RGB": ("#f59e0b", "BDD-RGB"),
}

models = []  # dict: name, ap, ap50, epoch, params, group, color, short
cur = None
with open(MR, encoding="utf-8") as f:
    in_tableA = False
    for line in f:
        if line.strip().startswith("## 表A"):
            in_tableA = True
            continue
        if line.strip().startswith("## 表B"):
            break
        if not in_tableA:
            continue
        m = section_re.match(line.strip())
        if m:
            cur = m.group(1).strip()
            continue
        r = row_re.match(line.strip())
        if r and cur:
            name, epoch, ap, ap50, params, _w = r.groups()
            color, grp = modality_colors.get(cur, ("#94a3b8", cur))
            models.append({
                "name": name, "ap": float(ap), "ap50": float(ap50),
                "epoch": int(epoch), "params": params, "group": grp,
                "color": color, "ours": False,
            })

# 2) 本管线当前可复现跑 (our pipeline)
ours = [
    {"name": "Ours Stage A (纯RGB, ViT-L/800 scratch)", "ap": 0.4990, "ap50": None,
     "epoch": 40, "params": "384M", "group": "Ours", "color": "#fbbf24", "ours": True},
    {"name": "Ours Stage C (RGB+IR+Depth 融合)", "ap": 0.5265, "ap50": 0.8046,
     "epoch": 35, "params": "384M", "group": "Ours", "color": "#f59e0b", "ours": True},
]
for o in ours:
    models.append(o)

# 3) 排序 (AP 降序)
models.sort(key=lambda x: x["ap"], reverse=True)

def short(n):
    s = n.replace("deimv2_dinov3_l_aic_", "")
    s = s.replace("deimv2_dinov3_l_", "")
    s = re.sub(r"_stg[0-9].*", "", s)
    return s if s else n

# 4) 画 SVG (水平条形, 全部)
X0, X1 = 360, 880          # 绘图区 x (左侧留给标签)
Y0, Y1 = 40, 40            # 上边距
ROW = 19                   # 每行高
max_ap = 0.55
def xpos(ap): return X0 + (ap / max_ap) * (X1 - X0)

W = 920
H = Y0 + len(models) * ROW + 60
svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" font-family="ui-monospace,Menlo,Consolas,monospace" font-size="11">']
# 标题
svg.append(f'<text x="10" y="22" font-size="14" font-weight="700" fill="#0f172a">测试集 val200 mAP@50-95 — 本管线 vs 其他模型 ({len(models)} 个)</text>')
# 网格 + x 轴刻度
for g in [0.1, 0.2, 0.3, 0.4, 0.5]:
    xx = xpos(g)
    svg.append(f'<line x1="{xx:.1f}" y1="{Y0-6}" x2="{xx:.1f}" y2="{H-30}" stroke="#e2e8f0" stroke-width="1"/>')
    svg.append(f'<text x="{xx:.1f}" y="{H-16}" font-size="9" fill="#64748b" text-anchor="middle">{g:.1f}</text>')
# 条
for i, m in enumerate(models):
    y = Y0 + i * ROW
    x = xpos(m["ap"])
    h = ROW - 5
    fill = m["color"]
    stroke = "#b45309" if m["ours"] else "none"
    sw = 2.2 if m["ours"] else 0
    svg.append(f'<rect x="{X0:.1f}" y="{y}" width="{x-X0:.1f}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="2"/>')
    # 标签
    lbl = short(m["name"])
    svg.append(f'<text x="{X0-6:.1f}" y="{y+h/2+3:.1f}" font-size="9" fill="#334155" text-anchor="end">{lbl[:34]}</text>')
    # 数值
    svg.append(f'<text x="{x+4:.1f}" y="{y+h/2+3:.1f}" font-size="9" font-weight="{"700" if m["ours"] else "400"}" fill="{"#b45309" if m["ours"] else "#0f172a"}">{m["ap"]:.4f}</text>')
svg.append('</svg>')

# 5) 同模态聚焦: RGB+IR+Depth 直接竞品组 (含 ours)
peer = [m for m in models if m["group"] in ("RGB+IR+Depth", "Ours")]
peer.sort(key=lambda x: x["ap"], reverse=True)
P0, P1 = 360, 880
PROW = 30
PH = 40 + len(peer) * PROW + 50
psvg = [f'<svg viewBox="0 0 {W} {PH}" width="100%" font-family="ui-monospace,Menlo,Consolas,monospace" font-size="12">']
psvg.append(f'<text x="10" y="22" font-size="13" font-weight="700" fill="#0f172a">直接竞品组: RGB+IR+Depth (含本管线)</text>')
for g in [0.1, 0.2, 0.3, 0.4, 0.5]:
    xx = xpos(g)
    psvg.append(f'<line x1="{xx:.1f}" y1="34" x2="{xx:.1f}" y2="{PH-24}" stroke="#e2e8f0"/>')
    psvg.append(f'<text x="{xx:.1f}" y="{PH-10}" font-size="9" fill="#64748b" text-anchor="middle">{g:.1f}</text>')
for i, m in enumerate(peer):
    y = 40 + i * PROW
    x = xpos(m["ap"])
    h = PROW - 7
    stroke = "#b45309" if m["ours"] else "none"
    sw = 2.4 if m["ours"] else 0
    psvg.append(f'<rect x="{P0:.1f}" y="{y}" width="{x-P0:.1f}" height="{h}" fill="{m["color"]}" stroke="{stroke}" stroke-width="{sw}" rx="2"/>')
    lbl = short(m["name"])
    psvg.append(f'<text x="{P0-6:.1f}" y="{y+h/2+4:.1f}" font-size="10" fill="#334155" text-anchor="end">{lbl[:40]}</text>')
    psvg.append(f'<text x="{x+5:.1f}" y="{y+h/2+4:.1f}" font-size="10" font-weight="{"700" if m["ours"] else "400"}" fill="{"#b45309" if m["ours"] else "#0f172a"}">{m["ap"]:.4f}</text>')
psvg.append('</svg>')

# 6) 图例
legend = "".join(
    f'<span style="display:inline-block;margin:2px 10px 2px 0"><span style="display:inline-block;width:11px;height:11px;background:{c};border:{"2px solid #b45309" if g=="Ours" else "none"};vertical-align:middle;margin-right:4px"></span>{g}</span>'
    for g, (c, _) in {**{k: v for k, v in modality_colors.items()}, "Ours": ("#f59e0b", "Ours")}.items()
)

# 7) HTML
html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIC mAP 对比</title>
<style>
 body{{margin:0;background:#f8fafc;color:#0f172a;font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
 .wrap{{max-width:980px;margin:0 auto;padding:18px}}
 .card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
 .warn{{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 12px;font-size:12.5px;line-height:1.6;color:#92400e}}
 .warn b{{color:#b45309}}
 .leg{{font-size:12px;margin:6px 0 2px;color:#475569}}
 .sub{{font-size:12px;color:#64748b;margin:0 0 8px}}
 svg{{display:block}}
</style></head><body><div class="wrap">
<div class="card"><div class="warn">
<b>口径警告（务必先看）：</b>本图纵轴是 <b>val200 mAP@50-95</b>（本项目无隐藏测试集 GT，val200 即测试集口径）。
表A 中 <b>历史模型数字来自各自不同的评测协议且不可复现</b>（MEMORY §5.2：同 ckpt 重测会掉约 2pt）。
因此本图是<b>排名/量级参考</b>，<b>不是同协议精确对比</b>。
真正公平的"谁更好"需把候选都过<b>同一条冻结评测路径</b>——该配对 bootstrap 当前正在后台运行。
本管线两条高亮金边条为<b>当前可复现</b>结果：Stage A 纯RGB 0.4990 / Stage C 融合 0.5265。
</div></div>
<div class="card">
<p class="sub">全部 {len(models)} 个模型，按 mAP 降序，按模态着色。金边高亮 = 本管线当前可复现跑。</p>
<div class="leg">{legend}</div>
{''.join(svg)}
</div>
<div class="card">
{''.join(psvg)}
</div>
<div class="card"><p class="sub"><b>直接竞品组</b>（RGB+IR+Depth 模态）放大视图。本管线 Stage C 融合 0.5265 在该组内排第 5（含 ours），
落后 midboost_l4(0.5345)/l4u3(0.5324)/l3(0.5280)/l2(0.5271)；差距 0.6–0.8pt &lt; 评测噪声 ±2pt（§5.2），<b>不能断言显著落后</b>。</p>
<p class="sub"><b>★ 配对 bootstrap（已出结果，200 次重抽样，同图配对）：</b>
融合 vs 纯RGB：last − stageA = <b>+2.60pt，95%CI [+0.60, +3.76]，显著（不跨0）</b>；SWA − stageA = +2.61pt CI[+1.07,+3.75] 显著。
best(0.5265) vs SWA(0.5250)：差 +0.15pt CI[−0.92,+0.78] <b>无差异</b> —— 再次印证 §5.6「argmax-best 在追噪声，SWA/末轮等效且更稳」。</p></div>
</div></body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)

# 也落一份结构化 JSON 便于后续引用
summary = {
    "metric": "val200 mAP@50-95 (test-set equivalent)",
    "caveat": "Table A historical numbers not reproducible (MEMORY §5.2); cross-model needs same frozen eval path",
    "ours": [{"name": m["name"], "ap": m["ap"], "ap50": m["ap50"]} for m in ours],
    "leaderboard_top10": [{"name": m["name"], "ap": m["ap"], "group": m["group"]} for m in models[:10]],
    "peer_group_rgb_ir_depth": [{"name": m["name"], "ap": m["ap"], "ours": m["ours"]} for m in peer],
}
with open(os.path.join(REPO, "outputs", "metrics/model_map_comparison.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("wrote", OUT)
print("models:", len(models), "| ours ranks:", [models.index(o)+1 for o in ours])
print("peer group size:", len(peer))
