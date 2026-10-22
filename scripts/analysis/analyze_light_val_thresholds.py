#!/usr/bin/env python3
"""CPU-only COCO matching, threshold sweep, and per-GT light diagnosis."""
import argparse
from collections import Counter, defaultdict
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def ious(box, boxes):
    if not len(boxes):
        return np.zeros(0)
    b = np.asarray(boxes, dtype=float)
    x,y,w,h = box
    inter = np.maximum(0,np.minimum(x+w,b[:,0]+b[:,2])-np.maximum(x,b[:,0])) * np.maximum(0,np.minimum(y+h,b[:,1]+b[:,3])-np.maximum(y,b[:,1]))
    return inter / np.maximum(w*h+b[:,2]*b[:,3]-inter, 1e-12)


def nms(rows, threshold=.6):
    # Use the same class-wise operator as the production PostProcessor.
    import torch
    from torchvision.ops import batched_nms
    if not rows:return []
    boxes=torch.tensor([r['bbox'] for r in rows],dtype=torch.float32)
    boxes[:,2:]+=boxes[:,:2]
    keep=batched_nms(boxes,torch.tensor([r['score'] for r in rows],dtype=torch.float32),
                     torch.tensor([r['category_id'] for r in rows]),threshold)
    return [rows[i] for i in keep.tolist()]


def evaluate(coco, predictions):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(predictions)
        e = COCOeval(coco,dt,'bbox')
        e.evaluate(); e.accumulate(); e.summarize()
    return e


def scan(e, light):
    rows=[]
    evals=[x for x in e.evalImgs if x is not None and x['aRng']==e.params.areaRng[0]]
    cached={}
    for cutoff in [.5,.75]:
        j=int(np.where(np.isclose(e.params.iouThrs,cutoff))[0][0])
        for scope in ['all','light']:
            ss=[];tt=[];ff=[];dd=[];ng=0
            for ev in evals:
                if scope=='light' and ev['category_id']!=light:continue
                ng+=int(np.count_nonzero(np.asarray(ev['gtIgnore'])==0))
                valid=~ev['dtIgnore'][j]
                tp=valid & (ev['dtMatches'][j]>0)
                fp=valid & (ev['dtMatches'][j]==0)
                dup=np.zeros(len(ev['dtIds']),dtype=bool)
                gt=[e.cocoGt.anns[g]['bbox'] for g,ig in zip(ev['gtIds'],ev['gtIgnore']) if not ig]
                for k in np.flatnonzero(fp):
                    dup[k]=np.any(ious(e.cocoDt.anns[ev['dtIds'][k]]['bbox'],gt)>=cutoff)
                ss.extend(ev['dtScores']);tt.extend(tp);ff.extend(fp);dd.extend(dup)
            cached[(cutoff,scope)]=(np.asarray(ss),np.asarray(tt),np.asarray(ff),np.asarray(dd),ng)
    for t in [round(.10+.01*i,2) for i in range(51)]:
        for cutoff in [.5,.75]:
            j=int(np.where(np.isclose(e.params.iouThrs,cutoff))[0][0])
            for scope in ['all','light']:
                ss,tt,ff,dd,ng=cached[(cutoff,scope)]
                use=ss>=t
                tp=int(np.count_nonzero(use & tt));fp=int(np.count_nonzero(use & ff))
                duplicate=int(np.count_nonzero(use & dd))
                precision=tp/(tp+fp) if tp+fp else 0.
                recall=tp/ng if ng else 0.
                rows.append(dict(threshold=t,iou=cutoff,scope=scope,tp=tp,fp=fp,fn=ng-tp,gt=ng,
                                 precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,duplicate_fp=duplicate))
    return rows


def diagnose(coco, e, raw, light, threshold=.48):
    gtlist=[a for a in coco.anns.values() if a['category_id']==light and not a.get('iscrowd',0) and not a.get('ignore',0)]
    by_img=defaultdict(list)
    for r in raw:
        by_img[r['image_id']].append(r)
    j=int(np.where(np.isclose(e.params.iouThrs,.5))[0][0])
    matched=set()
    for ev in e.evalImgs:
        if ev is None or ev['aRng']!=e.params.areaRng[0] or ev['category_id']!=light:
            continue
        for k,score in enumerate(ev['dtScores']):
            if score>=threshold and not ev['dtIgnore'][j,k] and ev['dtMatches'][j,k]>0:
                matched.add(int(ev['dtMatches'][j,k]))
    rows=[]
    for a in gtlist:
        prs=by_img[a['image_id']]
        same=[r for r in prs if r['category_id']==light]
        ov=ious(a['bbox'],[r['bbox'] for r in same])
        allov=ious(a['bbox'],[r['bbox'] for r in prs])
        good=[r for r,o in zip(same,ov) if o>=.5]
        score=max((r['score'] for r in good),default=0.)
        best_iou=float(ov.max()) if len(ov) else 0.
        if a['id'] in matched:
            bucket='detected'
        elif score>=threshold:
            bucket='matching_or_postprocess_loss'
        elif good:
            bucket='low_score'
        elif np.any(allov>=.5):
            bucket='class_confusion'
        elif best_iou>=.1:
            bucket='localization'
        else:
            bucket='no_candidate_top300'
        candidate=max(good,key=lambda r:r['score']) if good else (same[int(ov.argmax())] if len(ov) else None)
        im=coco.imgs[a['image_id']]
        area=a.get('area',a['bbox'][2]*a['bbox'][3])
        dense=sum(g['image_id']==a['image_id'] for g in gtlist)
        rows.append(dict(ann_id=a['id'],image_id=a['image_id'],file_name=im['file_name'],gt_bbox=a['bbox'],
                         size='small' if area<32**2 else 'medium' if area<96**2 else 'large',
                         lights_in_image=dense,bucket=bucket,best_same_iou=best_iou,
                         best_valid_score=score,candidate=candidate,threshold=threshold))
    return rows


def metrics(e, light):
    k=e.params.catIds.index(light)
    pr=e.eval['precision'][:,: ,k,0,-1]
    ap=float(pr[pr>=0].mean()) if np.any(pr>=0) else None
    return dict(mAP=float(e.stats[0]),AP50=float(e.stats[1]),AP75=float(e.stats[2]),AP_small=float(e.stats[3]),light_AP=ap)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--image-root',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.bundle/'manifest.json').read_text())
    assert sha(args.annotations)==manifest['annotation_sha256']
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO(str(args.annotations))
    light=next(c['id'] for c in coco.cats.values() if c['name']=='light')
    results={}; allscan=[]
    for run in manifest['runs']:
        assert run['status']=='complete' and set(run['evaluated_image_ids'])==set(coco.imgs)
        name=run['name']; predpath=args.bundle/(name+'_predictions.json')
        assert sha(predpath)==run['prediction_sha256']
        raw=json.loads(predpath.read_text())
        assert len(raw)==60000 and all(r['image_id'] in coco.imgs and 0<=r['score']<=1 and np.isfinite(r['bbox']).all() for r in raw)
        invalid=[r for r in raw if r['category_id'] not in coco.cats]
        by_img=defaultdict(list)
        for r in raw: by_img[r['image_id']].append(r)
        modes={}
        for mode in ['raw','nms60','nms60_cap100']:
            # maxDets=100 per category is standard COCO AP, not a global submission cap.
            preds=[r for prs in by_img.values() for r in ((nms(prs)[:100] if mode=='nms60_cap100' else nms(prs)) if mode.startswith('nms60') else prs)]
            e=evaluate(coco,preds)
            scans=scan(e,light)
            bad=[p for p in preds if p['category_id'] not in coco.cats]
            for r in scans:
                nbad=sum(p['score']>=r['threshold'] for p in bad) if r['scope']=='all' else 0
                den=r['tp']+r['fp']+nbad
                pvalid=r['tp']/den if den else 0.
                fvalid=2*pvalid*r['recall']/(pvalid+r['recall']) if pvalid+r['recall'] else 0.
                r.update(model=name,mode=mode,out_of_task_fp=nbad,precision_including_out_of_task=pvalid,f1_including_out_of_task=fvalid)
            allscan.extend(scans)
            diag=diagnose(coco,e,raw,light)
            diag30=diagnose(coco,e,raw,light,.30)
            dump(args.out/(name+'_'+mode+'_gt_diagnosis.json'),diag)
            dump(args.out/(name+'_'+mode+'_gt_diagnosis_thr30.json'),diag30)
            recovered=[b['ann_id'] for a,b in zip(diag,diag30) if a['bucket']!='detected' and b['bucket']=='detected']
            modes[mode]=dict(metrics=metrics(e,light),diagnosis_at_048=dict(Counter(r['bucket'] for r in diag)),
                             diagnosis_at_030=dict(Counter(r['bucket'] for r in diag30)),
                             recovered_gt_048_to_030=recovered,
                             diagnosis_by_size={s:dict(Counter(r['bucket'] for r in diag if r['size']==s)) for s in ['small','medium','large']},
                             dense_diagnosis=dict(Counter(r['bucket'] for r in diag if r['lights_in_image']>=5)),
                             operating_points=[r for r in scans if r['threshold'] in [.1,.3,.45,.48,.6]],
                             best_light_f1=max((r for r in scans if r['scope']=='light' and r['iou']==.5),key=lambda r:r['f1']))
            print(name,mode,json.dumps(modes[mode],ensure_ascii=False),flush=True)
        results[name]=dict(**modes, out_of_task_candidates=len(invalid),out_of_task_score_ge_01=sum(r['score']>=.1 for r in invalid),out_of_task_max_score=max((r['score'] for r in invalid),default=0.),head_classes=run['head_classes'])
    with (args.out/'threshold_scan.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allscan[0]));w.writeheader();w.writerows(allscan)
    dump(args.out/'summary.json',dict(models=results,scope='Validation descriptive analysis; thresholds selected on this same split are exploratory. Standard COCO maxDets=100 per category, not global TXT cap100. Diagnosis uses pre-NMS top300 candidates, not all decoder logits. Same source and inputs for all checkpoints; no causal attribution.',annotation_sha256=sha(args.annotations),manifest_sha256=sha(args.bundle/'manifest.json'),script_sha256=sha(__file__)))
    render_examples(args,coco)
    render_report(args,results,allscan)


def render_report(args,results,scans):
    lines=['# 验证集 light 阈值与漏检诊断（2026-10-21）','',
           '200 张固定验证图；三版 EMA 同一 ailab40 输入、FP32、1440×2560、无 TTA。每图导出分类/查询 Top-300，未作分数截断；候选缺失只指这批导出结果。',
           '','## 统一检查点评测','',
           '| 模型 | Decoder 类数 | mAP | AP50 | AP75 | AP_small | light AP |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        m=r['nms60']['metrics']
        lines.append('| '+name+' | '+str(r['head_classes'])+' | '+' | '.join(f"{100*m[k]:.2f}%" for k in ['mAP','AP50','AP75','AP_small','light_AP'])+' |')
    lines+=['','上表为同一 NMS 0.60、COCO 标准 maxDets=100/类/图；全图 cap100 结果另列于 JSON/CSV。V34/V35 保留检查点实际分类头，超出 0–11 的候选不参与标准 COCO 类别指标；未切片或重置权重。超范围候选仍保留参与 NMS/cap100，且在下方 operational precision 中额外计为 FP。','',
            '## light 阈值收益与代价','',
            '| 模型 | 阈值 | TP | FP | FN | Precision | Recall | F1 | 重复 FP |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in results:
        for t in [.1,.3,.45,.48,.6]:
            r=next(r for r in scans if r['model']==name and r['mode']=='nms60' and r['scope']=='light' and r['iou']==.5 and r['threshold']==t)
            lines.append(f"| {name} | {t:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['duplicate_fp']} |")
    lines+=['','## 全类别：降低全局阈值的代价（NMS 0.60 + 每图 cap100）','',
            '| 模型 | 阈值 | TP | 12类 FP | 超范围 FP | FN | Precision（含超范围） | Recall | F1（含超范围） |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in results:
        for t in [.3,.48]:
            r=next(r for r in scans if r['model']==name and r['mode']=='nms60_cap100' and r['scope']=='all' and r['iou']==.5 and r['threshold']==t)
            lines.append(f"| {name} | {t:.2f} | {r['tp']} | {r['fp']} | {r['out_of_task_fp']} | {r['fn']} | {r['precision_including_out_of_task']:.3f} | {r['recall']:.3f} | {r['f1_including_out_of_task']:.3f} |")
    compact={k:v for k,v in results['V34']['nms60'].items() if k!='operating_points'}
    lines+=['','## V34 在阈值 0.48 下的 light 诊断','',
            '桶使用标准 COCO 按分数一对一匹配判断已检出；未检出 GT 再检查原始候选。low_score 表示存在同类 IoU≥0.5 的低分框；localization 为最大同类 IoU 在 [0.1,0.5)；class_confusion 表示其他类有 IoU≥0.5；matching_or_postprocess_loss 表示有高分合格框却未匹配，需继续查 NMS/匹配竞争；no_candidate_top300 不等于网络内部没有响应。','',
            '```json',json.dumps(compact,ensure_ascii=False,indent=2),'```','',
            '## 解释边界与下一步','',
            '- light 共 135 个标注，仅 6 个 COCO small；尺寸切片只作描述，不能据此推广为稳定小目标收益。',
            '- 此处没有室内/室外标签，不宣称某类场景占比。dense 定义为单图 ≥5 个 light 标注。',
            '- 0.10–0.60（步长 0.01）中的最优 F1 来自同一验证集，只是探索性工作点；全类别与每图 cap100 代价见 threshold_scan.csv。未修改生产阈值、模型或提交包。',
            '- 各模型来自不同训练过程；分数差异不能归因于 P2 或 SDQS。历史 SDQS 开关未接线，统一使用实际历史 P3 35% 配额。',
            '- 若降低阈值只能救回少量 GT，应优先处理仍然缺失/偏移的候选；对 matching_or_postprocess_loss 先检查去重和匹配冲突。',
            '- 输入/权重/源码/配置哈希与严格加载记录在原始 bundle manifest.json；同轮三版可比，不能直接与别的节点或历史输入变体的绝对数值相减。']
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for name in results:
        rr=[r for r in scans if r['model']==name and r['mode']=='nms60' and r['scope']=='light' and r['iou']==.5]
        for ax,key in zip(axes,['precision','recall','f1']):
            ax.plot([r['threshold'] for r in rr],[r[key] for r in rr],label=name)
            ax.set(xlabel='Confidence threshold',ylabel=key.title(),ylim=(0,1.02))
            ax.axvline(.48,color='gray',linestyle=':',linewidth=1)
            ax.grid(alpha=.25)
    axes[0].legend();fig.suptitle('Validation light, IoU=0.50, same NMS=0.60 (135 GT)')
    fig.tight_layout();fig.savefig(args.out/'light_threshold_curves.png',dpi=160);plt.close(fig)


def render_examples(args,coco):
    from PIL import Image,ImageDraw
    rows=json.loads((args.out/'V34_nms60_gt_diagnosis.json').read_text())
    buckets=['low_score','localization','no_candidate_top300','class_confusion','matching_or_postprocess_loss']
    selected=[]
    for b in buckets:
        group=[r for r in rows if r['bucket']==b]
        group.sort(key=lambda r:(-r['lights_in_image'],-r['best_same_iou'],r['ann_id']))
        seen=set();chosen=[]
        for r in group:
            if r['image_id'] not in seen:
                seen.add(r['image_id']);chosen.append(r)
            if len(chosen)==3:break
        selected.extend(chosen)
    if not selected:return
    canvas=Image.new('RGB',(1000,330*((len(selected)+1)//2)),(25,25,25))
    for i,r in enumerate(selected):
        im=Image.open(args.image_root/r['file_name']).convert('RGB')
        x,y,w,h=r['gt_bbox'];cw=max(160,w*3,h*3*490/270);ch=cw*270/490
        crop=(int(x+w/2-cw/2),int(y+h/2-ch/2),int(x+w/2+cw/2),int(y+h/2+ch/2))
        panel=im.crop(crop);d=ImageDraw.Draw(panel)
        def box(b,color):
            a,c,bw,bh=b;d.rectangle([a-crop[0],c-crop[1],a+bw-crop[0],c+bh-crop[1]],outline=color,width=3)
        box(r['gt_bbox'],'#22ff55')
        if r['candidate']:box(r['candidate']['bbox'],'#ff9900')
        panel=panel.resize((490,270));ox=(i%2)*500;oy=(i//2)*330
        canvas.paste(panel,(ox,oy));d=ImageDraw.Draw(canvas)
        d.text((ox+4,oy+271),r['file_name'],fill='white')
        score=r['candidate']['score'] if r['candidate'] else 0.
        d.text((ox+4,oy+287),f"{r['bucket']} IoU={r['best_same_iou']:.2f} score={score:.3f}",fill='white')
        d.text((ox+4,oy+303),f"GT green / candidate orange; {r['size']}; lights={r['lights_in_image']}",fill='white')
    canvas.save(args.out/'V34_failure_examples.jpg',quality=92)
    dump(args.out/'V34_failure_examples.json',selected)


if __name__=='__main__':
    main()
