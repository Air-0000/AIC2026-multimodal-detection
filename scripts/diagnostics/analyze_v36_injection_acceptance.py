#!/usr/bin/env python3
"""Oracle bucketing and paired transition analysis for V36 injection acceptance.

Decomposes errors per GT across the 4 runs:
- tight (IoU_same >= 0.75)
- loose (0.50 <= IoU_same < 0.75)
- class_confusion (IoU_same < 0.50 and IoU_any >= 0.50)
- coarse (0.10 <= IoU_same < 0.50)
- no_candidate (IoU_same < 0.10 and IoU_any < 0.50)

Evaluates hypothesis:
- small-tight gain >= +3.0 pp (P3P4 - Control)
- small no_candidate must not increase (<= 0)
- overall no_candidate must not increase (<= 0)
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import numpy as np

SMALL, MEDIUM = 32 ** 2, 96 ** 2  # 1024, 9216


def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def size_bucket(area):
    return 'small' if area < SMALL else ('medium' if area < MEDIUM else 'large')


def bucket_of(iou_same, iou_any):
    if iou_same >= 0.75:
        return 'tight'
    if iou_same >= 0.50:
        return 'loose'
    if iou_any >= 0.50:
        return 'class_confusion'
    if iou_same >= 0.10:
        return 'coarse'
    return 'no_candidate'


def evaluate_run_oracle(ann_data, predictions, min_score=0.0):
    images = {im['id']: im for im in ann_data['images']}
    cats = {c['id']: c['name'] for c in ann_data['categories']}

    by_img = defaultdict(list)
    for r in predictions:
        if r['score'] >= min_score:
            by_img[r['image_id']].append(r)
    for k in by_img:
        by_img[k].sort(key=lambda r: -r['score'])

    gt_rows = []
    for a in ann_data['annotations']:
        if a.get('iscrowd', 0):
            continue
        gt_rows.append({
            'id': a['id'],
            'image_id': a['image_id'],
            'category_id': a['category_id'],
            'category_name': cats.get(a['category_id'], str(a['category_id'])),
            'bbox': a['bbox'],
            'area': a['area'],
            'size': size_bucket(a['area'])
        })

    gt_by_img = defaultdict(list)
    for i, r in enumerate(gt_rows):
        gt_by_img[r['image_id']].append(i)

    per_gt_results = {}
    rank_hist = Counter()
    confused = Counter()
    per_cat = defaultdict(lambda: Counter())
    iou_same_all, iou_any_all = [], []

    for img_id, idxs in gt_by_img.items():
        gts = [gt_rows[i] for i in idxs]
        prs = by_img.get(img_id, [])
        gb = xywh_to_xyxy([g['bbox'] for g in gts])
        pb = xywh_to_xyxy([r['bbox'] for r in prs]) if prs else np.zeros((0, 4))
        pc = np.array([r['category_id'] for r in prs]) if prs else np.zeros((0,), int)
        ps = np.array([r['score'] for r in prs]) if prs else np.zeros((0,))
        M = iou_matrix(gb, pb)

        for j, i in enumerate(idxs):
            gt_id = gt_rows[i]['id']
            same = np.where(pc == gt_rows[i]['category_id'])[0] if len(pc) else np.zeros((0,), int)
            iou_same = float(M[j, same].max()) if len(same) else 0.0
            iou_any = float(M[j].max()) if M.shape[1] else 0.0
            b = bucket_of(iou_same, iou_any)

            per_gt_results[gt_id] = {
                'gt_id': gt_id,
                'image_id': img_id,
                'category_id': gt_rows[i]['category_id'],
                'category_name': gt_rows[i]['category_name'],
                'size': gt_rows[i]['size'],
                'area': gt_rows[i]['area'],
                'iou_same': iou_same,
                'iou_any': iou_any,
                'bucket': b
            }

            iou_same_all.append(iou_same)
            iou_any_all.append(iou_any)
            per_cat[gt_rows[i]['category_id']][b] += 1

            if b == 'class_confusion' and M.shape[1]:
                confused[(cats.get(gt_rows[i]['category_id'], gt_rows[i]['category_id']),
                          cats.get(int(pc[int(M[j].argmax())]), int(pc[int(M[j].argmax())])))] += 1

    total_gt = len(gt_rows)
    bucket_overall = dict(Counter(per_gt_results[g['id']]['bucket'] for g in gt_rows))
    bucket_by_size = {
        s: dict(Counter(per_gt_results[g['id']]['bucket'] for g in gt_rows if g['size'] == s))
        for s in ('small', 'medium', 'large')
    }
    gt_by_size = dict(Counter(g['size'] for g in gt_rows))

    summary = {
        'total_gt': total_gt,
        'prediction_count': sum(len(v) for v in by_img.values()),
        'bucket_overall': bucket_overall,
        'bucket_by_size': bucket_by_size,
        'gt_by_size': gt_by_size,
        'rates': {
            'overall_tight_rate': bucket_overall.get('tight', 0) / total_gt,
            'overall_no_candidate_rate': bucket_overall.get('no_candidate', 0) / total_gt,
            'small_tight_rate': bucket_by_size['small'].get('tight', 0) / max(1, gt_by_size['small']),
            'small_no_candidate_rate': bucket_by_size['small'].get('no_candidate', 0) / max(1, gt_by_size['small']),
            'medium_tight_rate': bucket_by_size['medium'].get('tight', 0) / max(1, gt_by_size['medium']),
            'large_tight_rate': bucket_by_size['large'].get('tight', 0) / max(1, gt_by_size['large']),
        },
        'iou_same_quantiles': {q: float(np.quantile(iou_same_all, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        'class_confusion_top_pairs': [{'gt': g, 'pred': p, 'count': n} for (g, p), n in confused.most_common(5)],
        'per_category': {cats.get(c, str(c)): dict(v) for c, v in sorted(per_cat.items())}
    }
    return summary, per_gt_results


def compute_paired_transitions(control_gt, p3p4_gt):
    buckets = ['tight', 'loose', 'class_confusion', 'coarse', 'no_candidate']
    matrix = {b_from: {b_to: 0 for b_to in buckets} for b_from in buckets}
    small_matrix = {b_from: {b_to: 0 for b_to in buckets} for b_from in buckets}

    transitions = []
    for gt_id, c_res in control_gt.items():
        p_res = p3p4_gt[gt_id]
        b_from = c_res['bucket']
        b_to = p_res['bucket']
        matrix[b_from][b_to] += 1
        if c_res['size'] == 'small':
            small_matrix[b_from][b_to] += 1
        if b_from != b_to:
            transitions.append({
                'gt_id': gt_id,
                'category': c_res['category_name'],
                'size': c_res['size'],
                'from_bucket': b_from,
                'to_bucket': b_to,
                'control_iou_same': c_res['iou_same'],
                'p3p4_iou_same': p_res['iou_same']
            })

    return {
        'transition_matrix_overall': matrix,
        'transition_matrix_small': small_matrix,
        'change_count': len(transitions),
        'detailed_transitions': transitions
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--ann-file', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()

    ann_data = json.loads(args.ann_file.read_text())
    runs = ['seed42_control', 'seed42_p3p4', 'seed43_control', 'seed43_p3p4']

    results = {}

    for cand_type in ['post_nms', 'pre_nms']:
        results[cand_type] = {'runs': {}, 'paired': {}}
        for run_name in runs:
            pred_file = args.audit_dir / f"{run_name}_predictions_{cand_type}.json"
            if not pred_file.is_file():
                raise FileNotFoundError(f"Missing {pred_file}")
            preds = json.loads(pred_file.read_text())
            summary, per_gt = evaluate_run_oracle(ann_data, preds, min_score=0.0)
            results[cand_type]['runs'][run_name] = {
                'summary': summary,
                'per_gt': per_gt
            }

        # Compute paired transitions for seed42 and seed43
        for seed in ['42', '43']:
            ctrl_key = f"seed{seed}_control"
            p3p4_key = f"seed{seed}_p3p4"
            ctrl_gt = results[cand_type]['runs'][ctrl_key]['per_gt']
            p3p4_gt = results[cand_type]['runs'][p3p4_key]['per_gt']
            paired = compute_paired_transitions(ctrl_gt, p3p4_gt)

            ctrl_sum = results[cand_type]['runs'][ctrl_key]['summary']
            p3p4_sum = results[cand_type]['runs'][p3p4_key]['summary']

            delta_small_tight_pp = (p3p4_sum['rates']['small_tight_rate'] - ctrl_sum['rates']['small_tight_rate']) * 100
            delta_small_no_cand_pp = (p3p4_sum['rates']['small_no_candidate_rate'] - ctrl_sum['rates']['small_no_candidate_rate']) * 100
            delta_overall_no_cand_pp = (p3p4_sum['rates']['overall_no_candidate_rate'] - ctrl_sum['rates']['overall_no_candidate_rate']) * 100

            results[cand_type]['paired'][f"seed{seed}"] = {
                'control_run': ctrl_key,
                'p3p4_run': p3p4_key,
                'delta_small_tight_pp': delta_small_tight_pp,
                'delta_small_no_candidate_pp': delta_small_no_cand_pp,
                'delta_overall_no_candidate_pp': delta_overall_no_cand_pp,
                'hypothesis_check': {
                    'small_tight_gain_ge_3pp': delta_small_tight_pp >= 3.0,
                    'small_no_candidate_not_increased': delta_small_no_cand_pp <= 0.0,
                    'overall_no_candidate_not_increased': delta_overall_no_cand_pp <= 0.0,
                    'all_satisfied': (delta_small_tight_pp >= 3.0 and delta_small_no_cand_pp <= 0.0 and delta_overall_no_cand_pp <= 0.0)
                },
                'transitions': paired
            }

        # Compute mean and sample std across n=2 seeds
        s42 = results[cand_type]['paired']['seed42']
        s43 = results[cand_type]['paired']['seed43']

        d_st = [s42['delta_small_tight_pp'], s43['delta_small_tight_pp']]
        d_snc = [s42['delta_small_no_candidate_pp'], s43['delta_small_no_candidate_pp']]
        d_onc = [s42['delta_overall_no_candidate_pp'], s43['delta_overall_no_candidate_pp']]

        mean_st = float(np.mean(d_st))
        std_st = float(np.std(d_st, ddof=1))

        mean_snc = float(np.mean(d_snc))
        std_snc = float(np.std(d_snc, ddof=1))

        mean_onc = float(np.mean(d_onc))
        std_onc = float(np.std(d_onc, ddof=1))

        results[cand_type]['paired']['summary_n2'] = {
            'mean_delta_small_tight_pp': mean_st,
            'std_delta_small_tight_pp': std_st,
            'mean_delta_small_no_candidate_pp': mean_snc,
            'std_delta_small_no_candidate_pp': std_snc,
            'mean_delta_overall_no_candidate_pp': mean_onc,
            'std_delta_overall_no_candidate_pp': std_onc,
            'hypothesis_check': {
                'mean_small_tight_ge_3pp': mean_st >= 3.0,
                'mean_small_no_candidate_not_increased': mean_snc <= 0.0,
                'mean_overall_no_candidate_not_increased': mean_onc <= 0.0,
                'all_satisfied': (mean_st >= 3.0 and mean_snc <= 0.0 and mean_onc <= 0.0)
            }
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save compact json summaries without massive per_gt raw dumps
    for cand_type in ['post_nms', 'pre_nms']:
        compact = {
            'cand_type': cand_type,
            'runs': {
                r: results[cand_type]['runs'][r]['summary']
                for r in runs
            },
            'paired': results[cand_type]['paired']
        }
        (args.out_dir / f"oracle_analysis_{cand_type}.json").write_text(
            json.dumps(compact, indent=2, ensure_ascii=False) + "\n"
        )

    print("Oracle analysis completed successfully!")


if __name__ == "__main__":
    main()
