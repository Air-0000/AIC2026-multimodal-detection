#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

aic_root = '/root/autodl-tmp/AIC'
for p in [aic_root, os.path.join(aic_root, 'vendor', 'DEIMv2'), os.path.join(aic_root, 'scripts', 'lib')]:
    if p not in sys.path:
        sys.path.insert(0, p)

import faster_coco_eval
faster_coco_eval.init_as_pycocotools()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from engine.core import YAMLConfig

from scripts.analysis.eval_orthogonal_6ch_soup import (
    compute_phi_vt_and_3d_edges,
    ir_method1_multiscale_mca,
    ir_method3_anisotropic_thermal_deblooming
)

def main():
    ann_p = os.path.join(aic_root, 'work_dir/deimv2_dataset/valid/annotations/instances_val.json')
    coco_gt = COCO(ann_p)
    cats = {c['id']: c['name'] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    stem2id = {os.path.splitext(img['file_name'])[0]: img['id'] for img in coco_gt.dataset['images']}
    stems = sorted(list(stem2id.keys()))

    vis_dir = os.path.join(aic_root, 'work_dir/deimv2_dataset/valid/images')
    ir_dir = '/dev/shm/aic_cache/train_aligned_ir'
    dep_dir = '/dev/shm/aic_cache/depth_p0'

    print("Pre-caching 200 validation images ...", flush=True)
    val_data = {}
    for s in stems:
        vis_p = os.path.join(vis_dir, f'{s}.jpg') if os.path.exists(os.path.join(vis_dir, f'{s}.jpg')) else os.path.join(vis_dir, f'{s}.png')
        ir_p = os.path.join(ir_dir, f'{s}.jpg') if os.path.exists(os.path.join(ir_dir, f'{s}.jpg')) else os.path.join(ir_dir, f'{s}.png')
        dep_p = os.path.join(dep_dir, f'{s}.png')
        vis_im = Image.open(vis_p).convert('RGB')
        W, H = vis_im.size
        rgb_t = TF.pil_to_tensor(vis_im).float() / 255.0
        vis_bgr = cv2.imread(vis_p)
        ir_im = Image.open(ir_p).convert('L')
        ir_deb = ir_method3_anisotropic_thermal_deblooming(np.array(ir_im), vis_bgr)
        mca_np = ir_method1_multiscale_mca(ir_deb)
        dep_im = Image.open(dep_p)
        dep_arr = np.array(dep_im, dtype=np.float32)
        scale = 19999.0 if getattr(dep_im, 'mode', '') in ('I;16', 'I') else (65535.0 if dep_arr.max() > 255.5 else 255.0)
        dep_t = torch.from_numpy(np.clip(dep_arr / scale, 0.0, 1.0)).unsqueeze(0)
        if dep_t.shape[1:] != (H, W):
            dep_t = TF.resize(dep_t, [H, W])
        phi_vt, edge_3d = compute_phi_vt_and_3d_edges(dep_arr)
        val_data[s] = {
            'id': stem2id[s],
            'W': W, 'H': H,
            'rgb_t': rgb_t,
            'ir_deb': ir_deb,
            'mca_np': mca_np,
            'phi_vt': phi_vt,
            'edge_3d': edge_3d,
            'dep_t': dep_t
        }

    soup_p = os.path.join(aic_root, 'vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_pure_lmm_soup/best_stg1.pth')
    lmm_cfg = os.path.join(aic_root, 'vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_lmm_rtx6000.yml')
    cfg = YAMLConfig(lmm_cfg)
    model = cfg.model.cuda().eval()
    ck = torch.load(soup_p, map_location='cpu')
    w = ck['ema']['module'] if 'ema' in ck and 'module' in ck['ema'] else ck.get('model', ck)
    model.load_state_dict(w, strict=False)
    post = cfg.postprocessor.cuda()

    MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1).cuda()
    STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1).cuda()

    def eval_dets(mode):
        dets = []
        with torch.no_grad():
            for s in stems:
                d = val_data[s]
                H, W = d['H'], d['W']
                ir_t = torch.from_numpy(d['ir_deb'].astype(np.float32) / 255.0).unsqueeze(0)
                if ir_t.shape[1:] != (H, W):
                    ir_t = TF.resize(ir_t, [H, W])
                if mode == 'baseline':
                    ch4_np = d['mca_np']
                else:
                    ch4_np = np.clip(d['mca_np'].astype(np.float32) * d['phi_vt'] + 0.30 * 255.0 * d['edge_3d'], 0, 255).astype(np.uint8)
                lbg_t = torch.from_numpy(ch4_np.astype(np.float32) / 255.0).unsqueeze(0)
                if lbg_t.shape[1:] != (H, W):
                    lbg_t = TF.resize(lbg_t, [H, W])
                t6 = torch.cat([d['rgb_t'], ir_t, lbg_t, d['dep_t']], dim=0)
                inp = ((TF.resize(t6, [640, 640]).cuda() - MEAN_6CH) / STD_6CH).unsqueeze(0)
                out = model(inp)
                res = post(out, torch.tensor([[float(W), float(H)]], device='cuda'))[0]
                for l, b, sc in zip(res['labels'].cpu().numpy(), res['boxes'].cpu().numpy(), res['scores'].cpu().numpy()):
                    dets.append({
                        'image_id': d['id'],
                        'category_id': int(l),
                        'bbox': [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                        'score': float(sc)
                    })
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        s = ev.eval['precision']
        per_cat = {}
        for idx, catId in enumerate(coco_gt.getCatIds()):
            p = s[:, :, idx, 0, 2]
            p = p[p > -1]
            per_cat[catId] = float(np.mean(p)) if len(p) > 0 else 0.0
        return ev.stats[0], ev.stats[1], ev.stats[2], per_cat

    print("\nEvaluating Baseline (V17 Soup) ...", flush=True)
    map_base, ap50_base, ap75_base, cat_base = eval_dets('baseline')

    print("\nEvaluating Config 3 (All-Inclusive Orthogonal Split) ...", flush=True)
    map_c3, ap50_c3, ap75_c3, cat_c3 = eval_dets('config3')

    print('\n' + '=' * 70)
    print('PER-CLASS AP: BASELINE (0.5842) vs CONFIG 3 (0.5851)')
    print('=' * 70)
    print(f'Overall mAP:  Baseline {map_base:.4f} -> Config 3 {map_c3:.4f} ({map_c3 - map_base:+.4f})')
    print(f'Overall AP50: Baseline {ap50_base:.4f} -> Config 3 {ap50_c3:.4f} ({ap50_c3 - ap50_base:+.4f})')
    print(f'Overall AP75: Baseline {ap75_base:.4f} -> Config 3 {ap75_c3:.4f} ({ap75_c3 - ap75_base:+.4f})')
    print('-' * 70)
    print(f"{'Category':<16} | {'Baseline':<10} | {'Config 3 (All-Inclusive)':<24} | {'Delta':<8}")
    print('-' * 70)
    for catId in coco_gt.getCatIds():
        name = cats[catId]
        b = cat_base[catId]
        c = cat_c3[catId]
        print(f"{name:<16} | {b:.4f}     | {c:.4f}                   | {c - b:+.4f}")
    print('=' * 70)

if __name__ == '__main__':
    main()
