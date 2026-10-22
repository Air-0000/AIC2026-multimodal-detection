#!/usr/bin/env python3
"""Read-only paired V35 loading analysis using the existing COCO/threshold code."""
import argparse
from collections import Counter, defaultdict
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from analyze_light_val_thresholds import diagnose, dump, evaluate, metrics, nms, scan, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.bundle/'manifest.json').read_text())
    reference = json.loads((args.reference/'manifest.json').read_text())
    assert sha(args.annotations) == manifest['annotation_sha256'] == reference['annotation_sha256']
    assert manifest['inputs_sha256'] == reference['inputs_sha256']
    assert manifest['source_sha256'] == reference['source_sha256']
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.annotations))
    light = next(cid for cid,c in coco.cats.items() if c['name']=='light')
    predictions = {}
    for run in manifest['runs']:
        name = run['name']
        assert run['status']=='complete' and set(run['evaluated_image_ids'])==set(coco.imgs)
        f = args.bundle/(name+'_predictions.json')
        assert sha(f)==run['prediction_sha256']
        q = args.bundle/(name+'_query_selection.json')
        assert sha(q)==run['query_selection_sha256']
        predictions[name] = json.loads(f.read_text())
        assert len(predictions[name])==60000
        if name=='V35_NATIVE12':
            assert run['head_classes']==12 and all(r['category_id'] in coco.cats for r in predictions[name])
    predictions['V35_RAW80_FILTERED12'] = [r for r in predictions['V35_RAW80'] if r['category_id'] in coco.cats]
    results = {}; scans = []
    for name,raw in predictions.items():
        by_image = defaultdict(list)
        for r in raw: by_image[r['image_id']].append(r)
        modes = {}
        for mode in ('nms60','nms60_cap100'):
            post = [r for rows in by_image.values() for r in (nms(rows)[:100] if mode.endswith('cap100') else nms(rows))]
            ev = evaluate(coco,post)
            rows = scan(ev,light)
            for r in rows: r.update(model=name,mode=mode)
            scans.extend(rows)
            dg48 = diagnose(coco,ev,raw,light,.48)
            dg30 = diagnose(coco,ev,raw,light,.30)
            dump(args.out/(name+'_'+mode+'_gt048.json'),dg48)
            dump(args.out/(name+'_'+mode+'_gt030.json'),dg30)
            modes[mode] = dict(metrics=metrics(ev,light), diagnosis048=dict(Counter(r['bucket'] for r in dg48)),
                               diagnosis030=dict(Counter(r['bucket'] for r in dg30)),
                               recovered_gt=[b['ann_id'] for a,b in zip(dg48,dg30) if a['bucket']!='detected' and b['bucket']=='detected'])
        results[name] = modes
        print(name,json.dumps(modes,ensure_ascii=False),flush=True)
    qs = {}
    for name in ('V35_RAW80','V35_NATIVE12'):
        q = json.loads((args.bundle/(name+'_query_selection.json')).read_text())
        assert len(q)==200 and len({x['image_id'] for x in q})==200
        qs[name] = {r['image_id']:r['selected_anchor_logits'] for r in q}
    transitions = []
    for image_id in sorted(coco.imgs):
        aa,bb = qs['V35_RAW80'][image_id], qs['V35_NATIVE12'][image_id]
        assert len(aa)==len(bb)==300
        a,b = set(map(tuple,aa)),set(map(tuple,bb))
        assert len(a)==len(b)==300
        transitions.append(dict(image_id=image_id, shared_tokens=len(a&b), replaced_tokens=len(a-b),
                                ordered_query_positions_changed=sum(x!=y for x,y in zip(aa,bb))))
    old35 = next(r for r in reference['runs'] if r['name']=='V35')
    new80 = next(r for r in manifest['runs'] if r['name']=='V35_RAW80')
    selection = dict(images_with_changed_set=sum(r['replaced_tokens']>0 for r in transitions),
                     total_replaced_tokens=sum(r['replaced_tokens'] for r in transitions),
                     images_with_changed_order=sum(r['ordered_query_positions_changed']>0 for r in transitions),
                     max_replaced_per_image=max(r['replaced_tokens'] for r in transitions), per_image=transitions)
    dump(args.out/'query_selection_comparison.json', selection)
    summary = dict(models=results, old80_prediction_sha_reproduced=old35['prediction_sha256']==new80['prediction_sha256'],
                   manifest_sha256=sha(args.bundle/'manifest.json'), analysis_script_sha256=sha(__file__),
                   metric_helper_sha256=sha(Path(__file__).with_name('analyze_light_val_thresholds.py')),
                   scope='Paired checkpoint inference adaptation only; no training; same validation split; no claim of full production-pipeline equivalence',
                   query_selection={k:v for k,v in selection.items() if k!='per_image'})
    dump(args.out/'summary.json',summary)
    with (args.out/'threshold_scan.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(scans[0]));w.writeheader();w.writerows(scans)
    lines=['# V35 既有12类加载适配补评','',
           '同一冻结框架、同一权重、200张验证图、800个输入哈希、FP32、1440×2560；无训练。仅复用生产推理的类头切片，验证数据和共同后处理不等于完整测试推理生产管线。原始80类重跑预测SHA是否复现旧导出：'+str(summary['old80_prediction_sha_reproduced']), '',
           '| 加载/处理 | mAP | AP_small | AP75 | light AP |', '|---|---:|---:|---:|---:|']
    for name,r in results.items():
        mm=r['nms60']['metrics'];lines.append(f"| {name} | {mm['mAP']:.9f} | {mm['AP_small']:.9f} | {mm['AP75']:.9f} | {mm['light_AP']:.9f} |")
    lines+=['','主指标使用NMS0.60与COCO每类maxDets100；全图cap100结果另存，不与标准AP混淆。RAW80_FILTERED12仅在导出后删除超范围类别，是无法改变encoder候选的反事实对照。','',
            '| 版本 | light阈值 | TP | FP | FN | F1 |','|---|---:|---:|---:|---:|---:|']
    for name in results:
        for t in (.48,.30):
            r=next(r for r in scans if r['model']==name and r['mode']=='nms60' and r['scope']=='light' and r['iou']==.5 and r['threshold']==t)
            lines.append(f"| {name} | {t:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['f1']:.6f} |")
    lines+=['',f"查询选点：{selection['images_with_changed_set']}/200图候选集合变化，总计替换{selection['total_replaced_tokens']}/60000个token；{selection['images_with_changed_order']}/200图有顺序变化。锚点logit四元组连同尺度唯一标识特征token，不按最终框去猜选点。",'',
            '所有51档阈值仅验证集探索，不能作为独立测试泛化收益；没有调整生产阈值或提交包。不同模型版本对照不作结构因果归因。完整转移、加载适配键及哈希见原始bundle。']
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
