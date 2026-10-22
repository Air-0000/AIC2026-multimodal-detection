#!/usr/bin/env python3
"""Scientific visualizations of normalization decomposition and paired inference screen."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/reports/normalization_screen_20261022'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'figure.facecolor':'#f5f7fb','axes.facecolor':'#f5f7fb','font.size':11})
d=json.loads((OUT/'norm3_decomposition.json').read_text());r=d['rows'][1]
fig,ax=plt.subplots(figsize=(12,5));fig.subplots_adjust(left=.09,right=.97,bottom=.25,top=.74)
fig.text(.05,.94,'Norm3 拆解：本例相似度主要在 γ、β 两步下降',size=21,weight='bold',color='#172b44')
fig.text(.05,.86,'历史V36 / 垃圾桶q77：同一微移输入，“不恢复”与“只恢复其他候选H”的两次实测前向',size=11,color='#43556d')
keys=['input','centered','unit','gamma','affine'];vals=[r['cosine'][k] for k in keys]
ax.plot(range(5),vals,'o-',color='#149ccc',lw=3,ms=9);ax.set_xticks(range(5),['Norm3输入','减均值','除标准差','乘逐通道 γ','加逐通道 β']);ax.set_ylim(0,1.03);ax.set_ylabel('两次前向特征的余弦相似度');ax.grid(axis='y',alpha=.2)
for i,v in enumerate(vals):ax.annotate(f'{v:.3f}',(i,v),xytext=(0,12),textcoords='offset points',ha='center',weight='bold',size=13)
fig.text(.05,.13,'输入标准差：29.99 / 19.68，均远离 ε 主导区间。仅除以各自的正标量标准差，不会改变两向量的余弦。',size=11,color='#43556d')
fig.text(.05,.065,'这解释的是该位置的相似度变化；是否改善检测，要由完整配对前向验证。不能据此直接宣布 γ、β 有害。',size=11,color='#43556d')
fig.savefig(OUT/'01_norm3_decomposition.png',dpi=170);plt.close(fig)
p=OUT/'analysis/results.json'
if p.exists():
 d=json.loads(p.read_text());names=list(d['candidate_stability']);labels={'BASELINE':'原模型','D2_N1_OFF':'第2层 Norm1 跳过','D2_GATE_OFF':'第2层 门控Norm 跳过','D2_N3_OFF':'第2层 Norm3 跳过','D2_GATE_N3_OFF':'第2层 门控Norm+Norm3 跳过','ALL_N3_OFF':'全部层 Norm3 跳过','D2_N3_NO_AFFINE':'第2层 Norm3 去γ和β','D2_N3_NO_BETA':'第2层 Norm3 仅去β','D2_N3_RMS':'第2层 Norm3 换RMS','D2_N3_EPS1E3':'第2层 Norm3 增大ε'}
 fig,axes=plt.subplots(1,2,figsize=(15,7.7),gridspec_kw={'width_ratios':[1.2,1]});fig.subplots_adjust(left=.24,right=.97,bottom=.19,top=.80,wspace=.27);fig.text(.035,.945,'保留 / 跳过归一化：完整200图配对结果',size=22,weight='bold',color='#172b44');fig.text(.035,.88,'冻结同一份历史V36；每组独立原图/右移1网络输入像素。纯净v1未训练；推理干预不等于无归一化训练。',size=11,color='#43556d')
 y=np.arange(len(names));orig=[];shift=[];rescues=[];damage=[]
 for n in names:
  s='SHIFT_X1' if n=='BASELINE' else n+'_SHIFT_X1';orig.append(d['metrics'][n]['metrics']['mAP']*100);shift.append(d['metrics'][s]['metrics']['mAP']*100);q=d['candidate_stability'][n]['interior'];rescues.append(len(q['original19_recovered']));damage.append(len(q['original_stable_lost']))
 axes[0].barh(y-.17,orig,.32,label='原图 mAP',color='#008b72');axes[0].barh(y+.17,shift,.32,label='右移1像素 mAP',color='#bd3c57');axes[0].set_yticks(y,[labels[n] for n in names]);axes[0].invert_yaxis();axes[0].set_xlabel('mAP（%）');axes[0].legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,fontsize=10);axes[0].grid(axis='x',alpha=.15)
 for i,(a,b) in enumerate(zip(orig,shift)):
  axes[0].text(a+.3,i-.17,f'{a:.2f}',va='center',size=8);axes[0].text(b+.3,i+.17,f'{b:.2f}',va='center',size=8)
 axes[0].set_xlim(0,max(orig+shift)+8)
 axes[1].set_xlim(0,2);axes[1].set_ylim(len(names)-.2,-.8);axes[1].set_xticks([]);axes[1].set_yticks([])
 axes[1].text(.25,1.025,'原19漏检中救回',transform=axes[1].transAxes,ha='center',color='#008b72',weight='bold')
 axes[1].text(.75,1.025,'原稳定目标新增丢失',transform=axes[1].transAxes,ha='center',color='#bd3c57',weight='bold')
 stable=d['candidate_stability']['BASELINE']['interior']['original_tp']-d['candidate_stability']['BASELINE']['interior']['lost']
 for i,(a,b) in enumerate(zip(rescues,damage)):
  axes[1].axhline(i+.5,color='#dce3ed',lw=.6)
  axes[1].text(.5,i,f'{a} / 19',va='center',ha='center',size=13,color='#008b72',weight='bold')
  axes[1].text(1.5,i,f'{b} / {stable}',va='center',ha='center',size=13,color='#bd3c57',weight='bold')
 axes[1].axvline(1,color='#dce3ed',lw=1);axes[1].set_xlabel('固定同一批目标逐个核对，数字越大含义分别为救回 / 损失')
 fig.text(.035,.095,'验收同时看精度与稳定性：不能以原图就检不出来为代价，获得更少的“微移后丢失”。',size=11,color='#43556d');fig.text(.035,.045,'实际TP采用NMS0.60、score≥0.48、IoU≥0.50与一对一匹配；所有候选的GT类别分数仅用于诊断。',size=10,color='#43556d');fig.savefig(OUT/'02_fullval_comparison.png',dpi=170);plt.close(fig)
print('Saved normalization figures')
