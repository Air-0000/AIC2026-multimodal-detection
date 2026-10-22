#!/usr/bin/env python3
"""Descriptive layerwise paired analysis of a post-selected translation-loss cohort."""
import argparse, json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stats(v):
    v=np.asarray(v,dtype=float)
    return dict(n=len(v),median=float(np.median(v)) if len(v) else None,p10=float(np.quantile(v,.1)) if len(v) else None,p90=float(np.quantile(v,.9)) if len(v) else None)
def feature_names():
    return ['query_input']+[f'd{j}.{s}' for j in range(1,7) for s in ['norm1','gateway','out']]

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--paired-gt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    m=json.loads((a.bundle/'manifest.json').read_text());r=m['runs'][0];assert r['status']=='complete'
    assert sha(a.paired_gt)==m['trace_cohort']['paired_gt_sha256']
    assert sha(a.bundle/'layer_gt_trace.json')==r['layer_gt_trace_sha256'] and sha(a.bundle/'layer_queries.npz')==r['layer_queries_sha256']
    assert len(r['exact_forward_matches'])==57 and all(v['exact_output_match'] for v in r['exact_forward_matches'])
    trace=json.loads((a.bundle/'layer_gt_trace.json').read_text());old={v['ann_id']:v for v in json.loads(a.paired_gt.read_text())};lost=[v for v in trace if v['lost_cohort']]
    stable=[v for v in trace if 'interior' in old[v['ann_id']]['groups'] and old[v['ann_id']]['arms']['BASELINE']['detected'] and old[v['ann_id']]['arms']['SHIFT_X1']['detected']]
    assert len(lost)==19 and len({v['image_id'] for v in lost})==19
    result=dict(scope='Post-selected 19 lost GT / 19 images; controls are stable interior GT from the same images. Not held out, not causal attribution or new mAP.',cohorts=dict(lost=len(lost),stable_same_image=len(stable)),manifest_sha256=sha(a.bundle/'manifest.json'),script_sha256=sha(__file__),groups={})
    names=feature_names()
    for cohort,rows in [('lost',lost),('stable',stable)]:
        result['groups'][cohort]={}
        for arm in ['SHIFT_X1','SHIFT_X1_LOCKED']:
            dense={name:dict(cosine=stats([v['dense_pairs'][arm][name]['cosine'] for v in rows]),relative_l2=stats([v['dense_pairs'][arm][name]['relative_l2'] for v in rows])) for name in rows[0]['dense_pairs'][arm]} if rows else {}
            features={name:dict(endpoint_cosine=stats([v['arms'][arm]['features'][name]['endpoint_pair']['cosine'] for v in rows]),same_token_cosine=stats([v['arms'][arm]['features'][name]['same_token_pair']['cosine'] for v in rows if 'same_token_pair' in v['arms'][arm]['features'][name]])) for name in names}
            first_score=Counter();largest_fixed=Counter();largest_cos=Counter();layer=[]
            for row in rows:
                b=row['arms']['BASELINE'];v=row['arms'][arm]
                gaps=[x['best_valid_score']-y['best_valid_score'] for x,y in zip(b['layers'],v['layers'])]
                first_score[next((str(j+1) for j,gap in enumerate(gaps) if gap>=.2),'none')]+=1
                cos=[v['features'][n]['endpoint_pair']['cosine'] for n in names]
                j=int(np.argmax([cos[i-1]-cos[i] for i in range(1,len(cos))]));largest_cos[names[j+1]]+=1
                fg=[b['features'][n]['endpoint_fixed_logit']-v['features'][n]['endpoint_fixed_logit'] for n in names]
                if all(x is not None for x in fg):
                    j=int(np.argmax(np.diff(fg)));largest_fixed[names[j+1]]+=1
            for j in range(6):
                layer.append(dict(layer=j+1,oracle_score_gap=stats([v['arms']['BASELINE']['layers'][j]['best_valid_score']-v['arms'][arm]['layers'][j]['best_valid_score'] for v in rows]),endpoint_raw_logit_gap=stats([v['arms']['BASELINE']['layers'][j]['endpoint_raw_logit']-v['arms'][arm]['layers'][j]['endpoint_raw_logit'] for v in rows]),endpoint_lqe_gap=stats([v['arms']['BASELINE']['layers'][j]['endpoint_lqe_delta']-v['arms'][arm]['layers'][j]['endpoint_lqe_delta'] for v in rows])))
            result['groups'][cohort][arm]=dict(dense=dense,features=features,layers=layer,first_native_oracle_score_gap_ge20=dict(first_score),largest_endpoint_cosine_drop=dict(largest_cos),largest_fixed_readout_logit_gap_jump=dict(largest_fixed))
    dump(a.out/'summary.json',result)
    dump(a.out/'lost_gt_details.json',lost)
    lines=['# 微小平移逐层定位','',result['scope'],'','57次前向与前轮logits、逆变换boxes、查询选点逐值一致；冻结V36历史权重，无训练、无跨图状态。','',f"丢检组19个目标；同图稳定内部目标{len(stable)}个。",'', '## 丢检组：目标ROI特征对齐后的中位余弦相似度','','| Stage | +1x | +1x locked |','|---|---:|---:|']
    for name,v in result['groups']['lost']['SHIFT_X1']['dense'].items():lines.append(f"| {name} | {v['cosine']['median']:.4f} | {result['groups']['lost']['SHIFT_X1_LOCKED']['dense'][name]['cosine']['median']:.4f} |")
    lines+=['','## 丢检组：各层同类最佳有效候选分数差中位数（原图减平移）','','| Decoder layer | +1x | +1x locked | Raw endpoint logit gap +1x | LQE gap +1x |','|---|---:|---:|---:|---:|']
    for j,v in enumerate(result['groups']['lost']['SHIFT_X1']['layers']):lines.append(f"| {j+1} | {v['oracle_score_gap']['median']:.4f} | {result['groups']['lost']['SHIFT_X1_LOCKED']['layers'][j]['oracle_score_gap']['median']:.4f} | {v['endpoint_raw_logit_gap']['median']:.4f} | {v['endpoint_lqe_gap']['median']:.4f} |")
    lines+=['','## 严重个例','','| GT | Arm | Layer | Best valid GT score | Endpoint raw logit | Endpoint LQE correction |','|---|---|---:|---:|---:|']
    for row in lost:
        if row['ann_id'] not in [1471,1296,9]:continue
        for arm in ['BASELINE','SHIFT_X1','SHIFT_X1_LOCKED']:
            for v in row['arms'][arm]['layers']:lines.append(f"| {row['ann_id']} | {arm} | {v['layer']} | {v['best_valid_score']:.5f} | {v['endpoint_raw_logit']:.5f} | {v['endpoint_lqe_delta']:.5f} |")
    lines+=['','## 解读边界','','Dense特征按GT内7×7采样并补偿平移，余弦受采样/插值影响；不同阶段通道空间不同，不能把绝对余弦直接解释为因果贡献。Native每层head/LQE各自训练，不同层置信度不是统一校准曲线；同一个最终分类head的中间特征读数只作辅助。endpoint路径随各臂最终最佳query选取，同token路径和locked臂另外保留以识别换位混杂。最大跳变及首次20个百分点差异均为描述性筛查，不是显著性或唯一根因。早层没有IoU合格框时oracle分数记0，会混入定位因素，必须结合每GT原始轨迹。']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(cohorts=result['cohorts'],lost_unlocked_first=result['groups']['lost']['SHIFT_X1']['first_native_oracle_score_gap_ge20'],lost_locked_first=result['groups']['lost']['SHIFT_X1_LOCKED']['first_native_oracle_score_gap_ge20'],largest_fixed=result['groups']['lost']['SHIFT_X1_LOCKED']['largest_fixed_readout_logit_gap_jump']),ensure_ascii=False))

if __name__=='__main__':main()
