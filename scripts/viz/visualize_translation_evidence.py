#!/usr/bin/env python3
"""Render actual frozen-V36 evidence; no inference, training, or synthetic attention maps."""
import base64,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,patches,colors

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/reports/translation_evidence_visuals_20261022';OUT.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
FONT=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
plt.rcParams.update({'font.family':FONT,'axes.unicode_minus':False,'font.size':11,'axes.titlesize':14,'figure.facecolor':'#f4f6fa','axes.facecolor':'white','savefig.facecolor':'#f4f6fa'})
GREEN='#008568';RED='#c13b51';BLUE='#159ad6';GOLD='#f2b935';INK='#182a40'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCES={}
def readj(rel):
 p=ROOT/rel;SOURCES[rel]=sha(p);return json.loads(p.read_text())
def bundle(name):
 p=ROOT/'work_dir/audits'/name/'raw';m=readj(str((p/'manifest.json').relative_to(ROOT)));r=m['runs'][0];assert r['status']=='complete';assert sha(p/'intervention_outputs.npz')==r['intervention_outputs_sha256'];SOURCES[str((p/'intervention_outputs.npz').relative_to(ROOT))]=r['intervention_outputs_sha256'];return p,m,np.load(p/'intervention_outputs.npz')
ANN=readj('work_dir/deimv2_dataset/valid/annotations/instances_val.json');GT={g['id']:g for g in ANN['annotations']};IMS={i['id']:i for i in ANN['images']}
OP,OM,O=bundle('other_query_height_interventions_20261021');SP,SM,S=bundle('single_query_geometry_interventions_20261021')
assert OM['annotation_sha256']==SM['annotation_sha256']==sha(ROOT/'work_dir/deimv2_dataset/valid/annotations/instances_val.json');assert OM['inputs_sha256']==SM['inputs_sha256']
INPUTS=readj(str((OP/'val_inputs.json').relative_to(ROOT))); assert sha(OP/'val_inputs.json')==OM['inputs_sha256'];input_by_id={v['image_id']:v for v in INPUTS}
OD=readj('outputs/reports/other_query_height_interventions_20261021/gt_details.json');SD=readj('outputs/reports/single_query_geometry_interventions_20261021/gt_details.json')
CP=ROOT/'work_dir/audits/cross_query_component_trace_20261021/raw';CM=readj(str((CP/'manifest.json').relative_to(ROOT)));assert sha(CP/'query_attention_trace.npz')==CM['runs'][0]['query_attention_trace_sha256'];SOURCES[str((CP/'query_attention_trace.npz').relative_to(ROOT))]=sha(CP/'query_attention_trace.npz');T=np.load(CP/'query_attention_trace.npz');COMP=readj('outputs/reports/cross_query_component_trace_20261021/component_summary.json')
CASES=[]
def photo(gid):
 g=GT[gid];im=IMS[g['image_id']];p=ROOT/'dataset/train/visible'/im['file_name'];h=sha(p);assert h==input_by_id[im['id']]['files']['img_folder']['sha256'];SOURCES[str(p.relative_to(ROOT))]=h;img=np.asarray(Image.open(p).convert('RGB'));assert img.shape[:2]==(im['height'],im['width']);return g,im,img

def pos(z,iid):return int(np.where(z['image_ids']==iid)[0][0])
def scores(z,arm,i,c):return 1/(1+np.exp(-np.clip(z[arm+'_logits'][i,:,c].astype(float),-80,80)))
def boxes(z,arm,i,im):
 b=z[arm+'_boxes'][i].astype(float).copy();b*=np.array([im['width'],im['height'],im['width'],im['height']]);b[:,:2]-=b[:,2:]/2;return b

def initial(z,arm,i,im):
 b=1/(1+np.exp(-np.clip(z[arm+'_init_ref_unact'][i].astype(float),-80,80)));b[:,0]-=0 if arm=='BASELINE' or arm.startswith('BASELINE_') else 1/2560.;b*=np.array([im['width'],im['height'],im['width'],im['height']]);b[:,:2]-=b[:,2:]/2;return b

def ious(b,g):
 g=np.asarray(g,float);inter=np.maximum(0,np.minimum(b[:,:2]+b[:,2:],g[:2]+g[2:])-np.maximum(b[:,:2],g[:2])).prod(1);return inter/np.maximum(b[:,2:].prod(1)+g[2:].prod()-inter,1e-12)
def crop_limits(g,im,pad):
 x,y,w,h=g['bbox'];return max(0,x-pad),min(im['width'],x+w+pad),max(0,y-pad),min(im['height'],y+h+pad)
def background(ax,img,lim):
 ax.imshow(img,interpolation='nearest');ax.set_xlim(lim[0],lim[1]);ax.set_ylim(lim[3],lim[2]);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
def rect(ax,b,color,ls='-',lw=2.4):ax.add_patch(patches.Rectangle(b[:2],b[2],b[3],fill=False,edgecolor=color,linewidth=lw,linestyle=ls))
def save(fig,name):fig.savefig(OUT/name,dpi=160);plt.close(fig)
def case_figure(gid,z,details,arms,q,token,titles,name,subtitle,alternate=None):
 g,im,img=photo(gid);i=pos(z,im['id']);lim=crop_limits(g,im,80 if gid==1296 else 90);fig,axes=plt.subplots(1,len(arms),figsize=(15,6.3));fig.subplots_adjust(left=.025,right=.975,top=.72,bottom=.20,wspace=.07)
 fig.text(.035,.945,'垃圾桶：原查询不动，其他候选也能改变它的分数' if gid==1296 else '船：同一查询掉分，与整个目标漏检是两回事',fontsize=21,color=INK,weight='bold');fig.text(.035,.885,subtitle,fontsize=11,color='#43556d')
 for ax,arm,title in zip(axes,arms,titles):
  assert int(z[arm+'_ids'][i,q])==token;b=boxes(z,arm,i,im);s=scores(z,arm,i,g['category_id']);r=initial(z,arm,i,im);v=float(s[q]);ov=float(ious(b,g['bbox'])[q]);det=details[arm][str(gid)]['detected'];color=GREEN if v>=.48 else RED;background(ax,img,lim);rect(ax,g['bbox'],'white',':',3.8);rect(ax,r[q],BLUE,'--',2.2);rect(ax,b[q],color,'-',2.7)
  if alternate and arm in alternate:
   aq=alternate[arm];rect(ax,b[aq],GREEN,'-.',1.8);ax.text(.02,.025,f'q{aq} 补位：{s[aq]*100:.2f}%',transform=ax.transAxes,color='white',fontsize=11,bbox=dict(facecolor=GREEN,alpha=.92,pad=4))
  ax.set_title(title,fontsize=14,color=INK,pad=9);ax.text(.5,1.19,f'{v*100:.2f}%',transform=ax.transAxes,ha='center',color=color,fontsize=28,weight='bold');ax.text(.5,-.035,f'同一 q{q} 最终框 IoU {ov*100:.2f}%\n实际目标匹配：'+('检出' if det else '未检出'),transform=ax.transAxes,ha='center',va='top',fontsize=12,color=INK)
  CASES.append(dict(gt_id=gid,image_id=im['id'],arm=arm,query_index=q,token_id=token,query_score=v,query_final_iou=ov,actual_gt_detected=bool(det),initial_xywh_original_frame=r[q].tolist(),final_xywh_original_frame=b[q].tolist()))
 fig.text(.035,.105,'白点线：人工标注GT    蓝虚线：初始参考框    红/绿实线：该query回归框（低分候选也画出，≠已检出）',fontsize=10.5,color='#43556d')
 fig.text(.035,.047,'所有框已逆映射到原图坐标；背景复用同一张RGB原图。1像素指2560×1440网络输入像素；实际实验同时平移全部6通道。',fontsize=10,color='#43556d');save(fig,name)

case_figure(1296,O,OD,['BASELINE','SHIFT_X1_LOCKED','OTHERS_H'],77,19910,['原图','右移1像素＋固定原候选ID','相同微移输入，仅恢复其他51个H'],'01_garbage_recovery.png','后两列图像、候选ID、原q77的初始内容与四个参考框参数相同；第3列是诊断恢复，不是可部署修复。')
case_figure(1471,S,SD,['BASELINE','BASELINE_CY_TO_SHIFT','SHIFT_X1_LOCKED'],162,28072,['原图','原图不动，只改q162的初始Cy','右移1像素＋固定原候选ID'],'04_boat_query_handoff.png','第2列只改变q162一个初始坐标；q162掉分，但q188补位。第3列才发生整体漏检。诊断反事实，不是修复。',{'BASELINE_CY_TO_SHIFT':188})

# Which OTHER queries changed? Draw true reference centers, not fabricated pixel attention.
g,im,img=photo(1296);i=pos(O,170);a=O['SHIFT_X1_LOCKED_init_ref_unact'][i];b=O['OTHERS_H_init_ref_unact'][i];changed=np.where(np.any(a!=b,axis=1))[0];assert len(changed)==51 and 77 not in changed;assert np.array_equal(a[77],b[77]);assert np.array_equal(a[:,:3],b[:,:3]);r=initial(O,'SHIFT_X1_LOCKED',i,im);centers=r[:,:2]+r[:,2:]/2;delta=(1/(1+np.exp(-b[:,3].astype(float)))-1/(1+np.exp(-a[:,3].astype(float))))*1440.;lim=crop_limits(g,im,100);lim=(min(lim[0],centers[changed,0].min()-15),max(lim[1],centers[changed,0].max()+15),min(lim[2],centers[changed,1].min()-15),max(lim[3],centers[changed,1].max()+15));fig,axes=plt.subplots(1,2,figsize=(14,6.8),gridspec_kw={'width_ratios':[1.1,1]});fig.subplots_adjust(left=.04,right=.96,top=.79,bottom=.17,wspace=.28);fig.text(.035,.945,'变化的是51个候选，不是51个物体',fontsize=22,color=INK,weight='bold');fig.text(.035,.885,'这些候选集中在同一垃圾桶附近；原q77四个初始框参数均未改变。左图是候选位置，不是像素注意力热力图。',fontsize=11,color='#43556d');background(axes[0],img,lim);rect(axes[0],g['bbox'],'white',':',3);norm=colors.TwoSlopeNorm(vcenter=0,vmin=-np.max(np.abs(delta[changed])),vmax=np.max(np.abs(delta[changed])));sc=axes[0].scatter(centers[changed,0],centers[changed,1],c=delta[changed],cmap='coolwarm',norm=norm,s=50,edgecolors='white',linewidths=.55);axes[0].scatter(*centers[77],s=260,marker='*',c='#00f2dd',edgecolors='black',linewidth=.8);axes[0].annotate('q77：未改',centers[77],xytext=(18,-25),textcoords='offset points',color='white',fontsize=13,bbox=dict(facecolor='#14283b',alpha=.8,pad=3),arrowprops=dict(arrowstyle='->',color='white'));fig.colorbar(sc,ax=axes[0],fraction=.045,pad=.025,label='初始高度改变量（网络输入像素）');top=changed[np.argsort(np.abs(delta[changed]))[-12:]][::-1];axes[1].barh(np.arange(12),delta[top],color=[plt.cm.coolwarm(norm(v)) for v in delta[top]],height=.65);axes[1].set_yticks(np.arange(12),[f'q{q}' for q in top]);axes[1].invert_yaxis();axes[1].axvline(0,color='#7f8fa3',lw=1);axes[1].set_xlabel('恢复后H − 微移时H（网络输入像素）');axes[1].set_title('只列变化幅度最大的12个候选');axes[1].grid(axis='x',alpha=.15);fig.text(.035,.07,'候选的中心和宽度未改；只改其他候选的高度，最终使原q77从0.68%恢复到91.63%。数据来自已验证的同一次受控前向。',fontsize=10.5,color='#43556d');save(fig,'02_other_candidates.png')

# Attention and component changes: actual arrays, no image heatmaps.
left='SHIFT_X1_LOCKED';right='OTHERS_H';w0=T[left+'_L0_attention_weights'];w1=T[right+'_L0_attention_weights'];assert np.array_equal(T[left+'_L0_q'],T[right+'_L0_q']);assert np.array_equal(T[left+'_L0_v'],T[right+'_L0_v']);assert np.array_equal(T[left+'_L0_k'][77],T[right+'_L0_k'][77]);keys=np.any(T[left+'_L0_k']!=T[right+'_L0_k'],axis=1);assert np.array_equal(np.where(keys)[0],changed);top=np.argsort(np.abs(w1-w0))[-12:][::-1];stages=[('norm1','Norm1'),('cross_attn','跨注意力'),('gateway','门控后归一化'),('activation','ReLU'),('linear2','FFN线性输出'),('norm3','Norm3')];cs=[next(r['cosine'] for r in COMP['rows'] if r['decoder_layer']==2 and r['component']==name) for name,_ in stages];fig,axes=plt.subplots(1,3,figsize=(16,6),gridspec_kw={'width_ratios':[1.2,1.15,.65]});fig.subplots_adjust(left=.045,right=.975,top=.76,bottom=.25,wspace=.30);fig.text(.035,.95,'传播入口：其他候选的K改变；原查询的Q/K与所有V没变',fontsize=21,color=INK,weight='bold');fig.text(.035,.88,'同一垃圾桶、同一q77；图中为真实候选间注意力和模块输出。固定原候选ID，比较“未恢复”与“仅恢复其他51个H”。',fontsize=11,color='#43556d');x=np.arange(12);axes[0].bar(x-.2,w0[top]*100,.4,color=RED,label='未恢复：q77掉分');axes[0].bar(x+.2,w1[top]*100,.4,color=GREEN,label='只恢复其他H');axes[0].set_xticks(x,[f'q{q}' for q in top],rotation=65,ha='right',fontsize=9);axes[0].set_ylabel('q77给候选的注意力权重（%，多头平均）');axes[0].set_title('第1层：权重变化最大的12个候选',fontsize=12);axes[0].legend(fontsize=9,frameon=False);axes[1].plot(range(6),cs,'o-',color=BLUE,lw=2);axes[1].set_xticks(range(6),[v[1] for v in stages],rotation=35,ha='right',fontsize=10);axes[1].set_ylim(0,1.1);axes[1].set_ylabel('两次前向的特征余弦相似度');axes[1].set_title('第2层：越低，表示差异越大',fontsize=12)
for k,v in enumerate(cs):axes[1].annotate(f'{v:.3f}',(k,v),xytext=(0,9 if k%2==0 else -16),textcoords='offset points',ha='center',fontsize=10)
vals=[float(scores(O,left,i,9)[77]),float(scores(O,right,i,9)[77])];axes[2].bar([0,1],np.array(vals)*100,color=[RED,GREEN],width=.6);axes[2].axhline(48,color='#8995a7',linestyle=':');axes[2].set_xticks([0,1],['未恢复','其他H恢复']);axes[2].set_ylim(0,106);axes[2].set_ylabel('同q77最终正确类分数（%）');axes[2].set_title('最后：同一查询',fontsize=12)
for k,v in enumerate(vals):axes[2].text(k,v*100+3,f'{v*100:.2f}%',ha='center',fontsize=14,weight='bold')
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.text(.035,.105,'注意力图是query对query的权重，不是物体像素热图。子模块尺度/基底不同，不能把这些差异相除当成放大倍率。',fontsize=10.5,color='#43556d');fig.text(.035,.055,'只描述该历史V36案例；未执行的中间分类头不补造分数。此前直接删除/削弱自注意力的整体评测未通过。',fontsize=10.5,color='#43556d');save(fig,'03_attention_propagation.png')

# Counterexample: high class score but IoU crosses .50.
g,im,img=photo(9);i=pos(O,2);fig,axes=plt.subplots(1,2,figsize=(11,6.4));fig.subplots_adjust(left=.10,right=.9,top=.72,bottom=.22,wspace=.35);fig.text(.045,.945,'另一种丢检：分数没降，是框跨过IoU边界',fontsize=21,color=INK,weight='bold');fig.text(.045,.885,'light / GT9：选该目标附近最高分候选（IoU≥0.3），不把“低于IoU门槛”误画成“模型不认识”。',fontsize=10.5,color='#43556d');lim=crop_limits(g,im,18)
for ax,arm,label in zip(axes,['BASELINE','SHIFT_X1'],['原图','右移1网络输入像素']):
 b=boxes(O,arm,i,im);s=scores(O,arm,i,8);ov=ious(b,g['bbox']);q=int(np.argmax(np.where(ov>=.3,s,-1)));background(ax,img,lim);rect(ax,g['bbox'],'white',':',4);rect(ax,b[q],GREEN if ov[q]>=.5 else RED,'-',2.5);det=OD[arm]['9']['detected'];ax.set_title(label+f' · 候选q{q}',fontsize=14);ax.text(.5,1.16,f'light分数 {s[q]*100:.2f}%',transform=ax.transAxes,ha='center',fontsize=21,color=GREEN,weight='bold');ax.text(.5,-.035,f'IoU = {ov[q]:.4f}  '+('≥ 0.50' if ov[q]>=.5 else '< 0.50')+'\n实际匹配：'+('检出' if det else '未检出'),transform=ax.transAxes,ha='center',va='top',fontsize=14,color=GREEN if det else RED);CASES.append(dict(gt_id=9,arm=arm,query_index=q,token_id=int(O[arm+'_ids'][i,q]),query_score=float(s[q]),query_final_iou=float(ov[q]),actual_gt_detected=bool(det)))
fig.text(.045,.06,'白点线：GT；彩色实线：高分候选框。背景及框统一到原图坐标，局部放大使用最近邻显示；不是新的推理或修复。',fontsize=10,color='#43556d');save(fig,'05_light_iou_boundary.png')

# Standalone evidence gallery, no external scripts or remote dependencies.
sections=[('01_garbage_recovery.png','1. 垃圾桶：其他候选影响原查询','后两列是同一微移输入和同一候选ID。第三列只恢复其他候选的初始高度，属于使用原图内部状态的诊断，不能部署。'),('02_other_candidates.png','2. 到底动了哪些候选','不是51个物体，而是同一目标附近的51个候选。颜色表示初始框高度变化，不是注意力权重。'),('03_attention_propagation.png','3. 变化如何传播','第一层仅其他候选的K改变；原查询Q/K及所有V保持相同。第二层子模块出现更大特征分歧。'),('04_boat_query_handoff.png','4. 单个query掉分 ≠ 整个目标漏检','原图完全不动，只改q162初始Cy，它掉分，但q188补位；微移后才可能整目标漏检。'),('05_light_iou_boundary.png','5. 分数稳定也可能判为漏检','light分数仍高，但IoU跨过0.5，属于定位边界案例，不能混为分类崩落。')]
html=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>微平移失稳：实测证据可视化</title><style>body{margin:0;background:#edf1f7;color:#182a40;font:16px/1.7 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1500px;margin:auto;padding:32px}h1{font-size:32px;margin:0}p{max-width:1000px}.note{background:#fff4ce;padding:16px 22px;border-left:5px solid #d69a16;border-radius:8px}nav{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}a{color:#086baf}section{background:white;border-radius:16px;padding:22px;margin:24px 0;box-shadow:0 5px 25px #20354d0b}h2{margin:0;font-size:22px}img{display:block;width:100%;height:auto;cursor:zoom-in}summary{cursor:pointer;color:#086baf}code{font-size:13px}footer{color:#607086}button{cursor:pointer;border:1px solid #bcc9d8;border-radius:6px;background:#f4f7fb;padding:6px 12px}</style><main><h1>微平移失稳：把实测变化放回原图看</h1><p>历史 V36 冻结权重 · 2026-10-22 · 未启动新推理或训练；只使用既有原始记录绘图。</p><div class="note">诊断恢复 ≠ 可部署修复。为比较几何变化，所有预测框已逆映射到原图坐标，背景复用原始RGB。实际实验对2560×1440网络输入的全部6通道平移1像素。红色低分候选框仍会画出，不代表检测成功。图中实际匹配结果来自既有COCO评测。</div><nav>']
for n,(_,title,_) in enumerate(sections):html.append(f'<a href="#s{n}">{title.split("：")[0]}</a>')
html.append('</nav>')
for n,(file,title,note) in enumerate(sections):
 data=base64.b64encode((OUT/file).read_bytes()).decode();html.append(f'<section id="s{n}"><h2>{title}</h2><p>{note}</p><img alt="{title}" src="data:image/png;base64,{data}" onclick="this.style.width=this.style.width===\'160%\'?\'100%\':\'160%\'"><p><small>点击图片可放大／缩回。原始PNG：<a href="{file}">{file}</a></small></p></section>')
html.append('<details><summary>查看病例数值与来源范围</summary><pre style="overflow:auto">'+json.dumps(CASES,indent=2,ensure_ascii=False)+'</pre></details><footer>仅为已选病例的受控证据，不代表所有检测器或所有帧的共同根因。纯净v1仍未训练，未因本次可视化改动模型。候选间注意力为真实记录，不是生成的像素热图。</footer></main></html>');(OUT/'index.html').write_text('\n'.join(html));(OUT/'case_values.json').write_text(json.dumps(CASES,indent=2,ensure_ascii=False)+'\n')
clean=readj('work_dir/audits/aic_clean_baseline_v1_20261021/source_final_manifest.json');assert all(sha(ROOT/f)==h for f,h in clean.items());manifest=dict(scope='Visualization only. Frozen historical V36 synthetic-shift evidence; original RGB reused for inverse-coordinate overlays. No new inference or training; no fabricated attention. Displayed score is per-query GT-class score; actual GT matching separately provided.',script_sha256=sha(__file__),source_sha256=SOURCES,baseline124_unchanged=True,artifacts={f.name:sha(f) for f in OUT.iterdir() if f.is_file() and f.name!='manifest.json'});(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(json.dumps(dict(out=str(OUT),plots=5,cases=len(CASES),source_files=len(SOURCES),baseline124_unchanged=True)))
