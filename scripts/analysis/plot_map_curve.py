import json

BASE = '/home/constant/CodeBuddy/Dev/AIC'
LOG = f'{BASE}/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug/log.txt'
OUT = f'{BASE}/outputs/reports/vitl800_map_curve.html'

rows = []
for ln in open(LOG, encoding='utf-8'):
    ln = ln.strip()
    if not ln:
        continue
    try:
        d = json.loads(ln)
    except Exception:
        continue
    if 'epoch' in d and 'test_coco_eval_bbox' in d:
        try:
            rows.append((int(d['epoch']), float(d['test_coco_eval_bbox'][0])))
        except Exception:
            continue

rows.sort()
assert rows, 'no mAP rows found'

epochs = [r[0] for r in rows]
maps = [r[1] for r in rows]
best_ep, best_map = max(rows, key=lambda r: r[1])
ep0, ep1 = min(epochs), max(epochs)
mmin, mmax = min(maps), max(maps)
pad = (mmax - mmin) * 0.25 + 0.004
ylo, yhi = mmin - pad, mmax + pad

W, H = 760, 470
L, R, T, B = 66, 28, 54, 72
def X(e):
    if ep1 == ep0:
        return L + (W - L - R) / 2
    return L + (e - ep0) / (ep1 - ep0) * (W - L - R)
def Y(m):
    return T + (yhi - m) / (yhi - ylo) * (H - T - B)

parts = []
ny = 5
for i in range(ny + 1):
    m = ylo + (yhi - ylo) * i / ny
    y = Y(m)
    parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#e3e6ee" stroke-width="1"/>')
    parts.append(f'<text x="{L-10:.1f}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#555">{m:.3f}</text>')

nx = 7
step = max(1, (ep1 - ep0) // (nx - 1))
xticks = list(range(ep0, ep1 + 1, step))
if xticks[-1] != ep1:
    xticks.append(ep1)
for e in xticks:
    x = X(e)
    parts.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H-B}" stroke="#f0f2f7" stroke-width="1"/>')
    parts.append(f'<text x="{x:.1f}" y="{H-B+20:.1f}" text-anchor="middle" font-size="12" fill="#555">{e}</text>')

pts = ' '.join(f'{X(e):.1f},{Y(m):.1f}' for e, m in rows)
parts.append(f'<polyline fill="none" stroke="#2f7ed8" stroke-width="2.5" points="{pts}"/>')
for e, m in rows:
    parts.append(f'<circle cx="{X(e):.1f}" cy="{Y(m):.1f}" r="3" fill="#2f7ed8"/>')

bx, by = X(best_ep), Y(best_map)
parts.append(f'<line x1="{L}" y1="{by:.1f}" x2="{W-R}" y2="{by:.1f}" stroke="#e0584f" stroke-width="1.4" stroke-dasharray="6 4"/>')
parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="5.5" fill="none" stroke="#e0584f" stroke-width="2.5"/>')
parts.append(f'<text x="{bx+9:.1f}" y="{by-9:.1f}" font-size="13" font-weight="700" fill="#c0392b">best {best_map:.4f} @ ep{best_ep}</text>')

if ep0 <= 26 <= ep1:
    vx = X(26)
    parts.append(f'<line x1="{vx:.1f}" y1="{T}" x2="{vx:.1f}" y2="{H-B}" stroke="#9b59b6" stroke-width="1.2" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{vx+4:.1f}" y="{T+14:.1f}" font-size="11" fill="#8e44ad">lr cosine (ep26)</text>')

svg = f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>
<text x="{W/2:.0f}" y="26" text-anchor="middle" font-size="16" font-weight="700" fill="#1f2d3d">vitl800 - mAP / epoch (ViT-L @800, LLRD unfreeze6, val=200)</text>
{''.join(parts)}
<text x="{L}" y="{H-22:.0f}" font-size="12" fill="#7a8699">noise ~ +-2pt is NOT training instability (EMA=0.9999); it is the 200-img val + long-tail (tricycle=2 ann)</text>
<text x="{L}" y="{H-6:.0f}" font-size="12" fill="#7a8699">after ep26 lr enters cosine decay: trend flattens, noise shows -> best locked @ ep26; final uses best weights</text>
</svg>'''

html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>vitl800 mAP curve</title><style>body{{margin:0;background:#f5f6f8;display:flex;justify-content:center;padding:24px}} .card{{background:#fff;border:1px solid #e6e8ec;border-radius:12px;padding:12px 16px;box-shadow:0 2px 10px rgba(0,0,0,.06)}}</style></head><body><div class="card">{svg}</div></body></html>'''

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'OK points={len(rows)} ep0={ep0} ep1={ep1} best={best_map:.4f}@ep{best_ep}')
