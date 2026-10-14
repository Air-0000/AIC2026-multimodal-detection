import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = '/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug/log.txt'
OUT = '/home/constant/CodeBuddy/Dev/AIC/outputs/figures/vitl800_map_curve.png'

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
ep = [r[0] for r in rows]
mp = [r[1] for r in rows]
be, bm = max(rows, key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, mp, '-o', color='#2f7ed8', ms=4, lw=2, label='mAP (val=200 imgs)')
ax.axhline(bm, color='#e0584f', ls='--', lw=1.4)
ax.scatter([be], [bm], s=110, facecolors='none', edgecolors='#e0584f', lw=2.5, zorder=5)
ax.annotate(f'best {bm:.4f} @ ep{be}', (be, bm), xytext=(be + 0.6, bm + 0.0045),
            color='#c0392b', fontsize=11, fontweight='bold')
if min(ep) <= 26 <= max(ep):
    ax.axvline(26, color='#9b59b6', ls=':', lw=1.6)
    ax.text(26.3, min(mp) + 0.0008, 'lr cosine decay\nstarts @ ep26', color='#8e44ad',
            fontsize=9, va='bottom')
ax.set_xlabel('epoch')
ax.set_ylabel('mAP')
ax.set_title('vitl800 mAP / epoch  (ViT-L@800, LLRD unfreeze 6 layers)')
ax.grid(alpha=0.3)
ax.legend(loc='lower right')
fig.text(0.13, 0.02,
         'noise ~ +-2pt comes from 200-img val + long-tail classes (tricycle=2 ann), NOT training instability (EMA=0.9999)',
         fontsize=8.5, color='#7a8699')
fig.text(0.13, 0.045,
         f'after ep26 lr enters cosine decay: slow climb + noise -> NEW best @ ep{be} ({bm:.4f}); final uses best weights',
         fontsize=8.5, color='#7a8699')
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(OUT, dpi=130)
print(f'OK points={len(rows)} best={bm:.4f}@ep{be} -> {OUT}')
