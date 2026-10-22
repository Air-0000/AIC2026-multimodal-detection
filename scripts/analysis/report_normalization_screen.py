#!/usr/bin/env python3
"""Summarize completed frozen normalization interventions without selection-lock confounds."""
from pathlib import Path
import json,hashlib,html,base64
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/reports/normalization_screen_20261022';RAW=ROOT/'work_dir/audits/normalization_screen_20261022/raw'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((OUT/'analysis/results.json').read_text());pairs=json.loads((OUT/'analysis/paired_gt.json').read_text());by={x['ann_id']:x for x in pairs};m=json.loads((RAW/'manifest.json').read_text());assert m['runs'][0]['status']=='complete'
assert sha(RAW/'manifest.json')==r['manifest_sha256']
z=np.load(RAW/'candidate_outputs.npz');ann=json.loads((ROOT/'work_dir/deimv2_dataset/valid/annotations/instances_val.json').read_text());gt={x['id']:x for x in ann['annotations']};ims={x['id']:x for x in ann['images']};pos={int(v):i for i,v in enumerate(z['image_ids'])}
strongpath=ROOT/'outputs/reports/translation_cause_validation_20261021/iou_boundary_summary.json';strong=json.loads(strongpath.read_text())['SHIFT_X1']['low_gt_score_even_at_iou30'];assert len(strong)==13
labels={'BASELINE':'保留原归一化','D2_N1_OFF':'第2层 Norm1 跳过','D2_GATE_OFF':'第2层门控 Norm 跳过','D2_N3_OFF':'第2层 Norm3 跳过','D2_GATE_N3_OFF':'第2层门控 Norm、Norm3 均跳过','ALL_N3_OFF':'全部解码层 Norm3 跳过','D2_N3_NO_AFFINE':'第2层 Norm3 去 γ、β','D2_N3_NO_BETA':'第2层 Norm3 仅去 β','D2_N3_RMS':'第2层 Norm3 换 RMS（保留 γ、β）','D2_N3_EPS1E3':'第2层 Norm3 ε 改 1e-3'}
rows=[];cases=[]
for name,ss in r['candidate_stability'].items():
 shifted='SHIFT_X1' if name=='BASELINE' else name+'_SHIFT_X1';s=ss['interior'];a=r['metrics'][name];b=r['metrics'][shifted];ta=next(t for t in a['thresholds'] if t['scope']=='all' and t['threshold']==.48);tb=next(t for t in b['thresholds'] if t['scope']=='all' and t['threshold']==.48)
 row=dict(name=name,label=labels[name],original_mAP=100*a['metrics']['mAP'],shifted_mAP=100*b['metrics']['mAP'],original_AP75=100*a['metrics']['AP75'],shifted_AP75=100*b['metrics']['AP75'],original_APsmall=100*a['metrics']['AP_small'],shifted_APsmall=100*b['metrics']['AP_small'],original_TP=ta['tp'],original_FP=ta['fp'],shifted_TP=tb['tp'],shifted_FP=tb['fp'],interior=s,strong13_recovered=[gid for gid in strong if by[gid]['arms'][shifted]['detected']]);rows.append(row)
 for gid in [1296,1471,9]:
  g=gt[gid];im=ims[g['image_id']];i=pos[g['image_id']];bb=z[shifted+'_boxes'][i].astype(float).copy();bb[:,:2]-=bb[:,2:]/2;bb*=np.array([im['width'],im['height'],im['width'],im['height']]);x,y,w,h=g['bbox'];inter=np.maximum(0,np.minimum(x+w,bb[:,0]+bb[:,2])-np.maximum(x,bb[:,0]))*np.maximum(0,np.minimum(y+h,bb[:,1]+bb[:,3])-np.maximum(y,bb[:,1]));ov=inter/np.maximum(w*h+bb[:,2]*bb[:,3]-inter,1e-12);sc=1/(1+np.exp(-np.clip(z[shifted+'_logits'][i,:,g['category_id']].astype(float),-80,80)));cases.append(dict(name=name,gt_id=gid,actual_gt_detected=by[gid]['arms'][shifted]['detected'],max_score_iou30=float(np.max(np.where(ov>=.3,sc,0))),max_score_iou50=float(np.max(np.where(ov>=.5,sc,0)))))
summary=dict(source_results_sha256=sha(OUT/'analysis/results.json'),strong_cohort_source_sha256=sha(strongpath),rows=rows,cases=cases,scope='Historical V36 frozen inference, natural selection. No training; independent images, exploratory validation only. Does not establish the outcome of normalization-free training.')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['# 归一化配对验证（2026-10-22）','','历史V36 epoch39 EMA，完整200图/1523非crowd GT，10种配置×原图/右移1网络输入像素，共4000次前向。未修改基线400次与旧预测逐元素一致；纯净v1仍未训练。','', '## 完整验证结果','','| 修改 | 原图 mAP | 微移 mAP | 原图TP/FP | 微移TP/FP | 原19漏检救回 | 强低分13救回 | 原稳定目标新增丢失 | 共同原图TP→微移丢失 |','|---|---:|---:|---|---|---:|---:|---:|---|']
for v in rows:
 s=v['interior'];lines.append(f"| {v['label']} | {v['original_mAP']:.3f} | {v['shifted_mAP']:.3f} | {v['original_TP']}/{v['original_FP']} | {v['shifted_TP']}/{v['shifted_FP']} | {len(s['original19_recovered'])}/19 | {len(v['strong13_recovered'])}/13 | {len(s['original_stable_lost'])} | {s['shared_original_tp']} → {s['shared_original_lost_on_shift']} |")
lines+=['','TP/FP列覆盖全部1523 GT，NMS0.60、score0.48、IoU0.50，一对一匹配；恢复/稳定队列为排除边缘的1368 GT。稳定损失队列固定为原模型原图及微移均检出的目标；共同原图TP为原模型与修改模型原图均检出的目标，防止能力退化被误判稳定。AP75/AP-small、所有GT配对、逐类别原始预测与完整哈希见JSON/raw。','','## 归一化拆解','','同q77受控案例Norm3输入cos0.867671，去均值/除标准差后cos0.867815，乘γ后0.387391，加β后0.288951；输入std29.99/19.68，var/ε约8994万/3875万，不支持小方差除法是该案例主要来源。正标量标准差除法本身不改变两向量余弦；该位置绝对L2差从259.17降到10.55，不能称为绝对数值爆炸。该重算比较锁候选的未恢复/OTHERS_H前向，不能混同本轮自然选候选的性能评测。','','旧19图队列中，19漏检目标D2 Norm3前后cos差的中位0.09558，225稳定目标0.04214；描述性关联，不是随机试验或总体比例。','','推理时移除已训练模型的归一化会改变特征分布；本轮不能代替同起点、同种子、同预算的有/无归一化重新训练。没有启动训练、改动基准或产出提交包。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
parts=['<!doctype html><meta charset="utf-8"><title>归一化配对验证</title><style>body{font:17px/1.7 sans-serif;max-width:1500px;margin:40px auto;padding:0 24px;background:#f5f7fb;color:#172b44}img{width:100%;cursor:zoom-in}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ccd4dd}</style><h1>保留 / 跳过归一化：实际测试</h1><p>历史V36，200张验证图，10组配对，4000次独立前向。纯净v1未训练。点击图片放大。</p>']
for f in sorted(OUT.glob('*.png')):parts.append('<img onclick="this.style.width=this.style.width===\'160%\'?\'100%\':\'160%\'" src="data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode()+'">')
parts+=['<h2>完整结果与边界</h2><pre>'+html.escape('\n'.join(lines))+'</pre>'];(OUT/'index.html').write_text('\n'.join(parts))
print(json.dumps(rows,ensure_ascii=False,indent=2))
