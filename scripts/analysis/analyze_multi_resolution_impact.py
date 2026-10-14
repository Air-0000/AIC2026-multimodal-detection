#!/usr/bin/env python3
"""
Empirical Multi-Resolution Inference Impact Analysis:
Evaluate Sovereign V29 Cycle 5 Peak across 4 distinct inference resolutions on the 200 validation images!

Resolutions tested:
1. 720p (720 x 1280)    - Half scale / Low latency
2. 1080p (1088 x 1920)  - Native sensor resolution
3. 2.5K (1440 x 2560)   - Model training native resolution
4. 2.8K (1632 x 2880)   - Super-resolution / Ultra fine grid

Metrics analyzed:
- Overall mAP (0.50:0.95), AP50, AP75
- AP_small, AP_med, AP_large
- UAV (CID 10) AP, AP50, Average confidence
- Average inference latency per image
"""

import os
import sys
import time
import json
import contextlib
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

_root = Path("/home/constant/CodeBuddy/Dev/AIC")
if not _root.exists():
    _root = Path("/home/ailab/ssd/bigdata/hejianzhong/AIC_V29")

sys.path.insert(0, str(_root / 'vendor/DEIMv2'))
sys.path.insert(0, str(_root / 'scripts/lib'))
os.chdir(str(_root / 'vendor/DEIMv2'))

from engine.core import YAMLConfig
from engine.data.dataset.coco_eval import CocoEvaluator
from faster_coco_eval import COCO

RESOLUTIONS = [
    ("704p (704x1280)", 704, 1280),
    ("1080p (1088x1920)", 1088, 1920),
    ("2.5K (1440x2560)", 1440, 2560),
    ("2.8K (1632x2880)", 1632, 2880),
]

def run_eval_at_resolution(model, postprocessor, val_dataset, coco_gt, h, w, device="cuda:0"):
    model.eval()
    processed_top100 = {}
    uav_scores = []
    
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(val_dataset)):
            img, target = val_dataset[i]
            # img is [6, H_curr, W_curr]
            # Resize to target (h, w)
            img_resized = F.interpolate(img.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).to(device)
            orig_sz = target["orig_size"].unsqueeze(0).to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(img_resized)
            processed = postprocessor(outputs, orig_sz, img_resized)
            
            p = processed[0]
            img_id = int(target["image_id"].item())
            
            labels = p["labels"].cpu().numpy()
            boxes = p["boxes"].cpu().numpy()
            scores = p["scores"].cpu().numpy()
            
            # Record UAV scores
            for lab, sc in zip(labels, scores):
                if lab == 10 and sc >= 0.10:
                    uav_scores.append(float(sc))
            
            idx = np.argsort(-scores)[:100]
            if len(idx) > 0:
                processed_top100[img_id] = {
                    "labels": torch.from_numpy(labels[idx]).long(),
                    "boxes": torch.from_numpy(boxes[idx]).float(),
                    "scores": torch.from_numpy(scores[idx]).float(),
                }
            else:
                processed_top100[img_id] = {
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                }
                
    elapsed = time.time() - t0
    fps = len(val_dataset) / elapsed
    latency_ms = (elapsed / len(val_dataset)) * 1000
    
    evaluator = CocoEvaluator(coco_gt, ["bbox"])
    evaluator.update(processed_top100)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        evaluator.summarize()
        
    stats = evaluator.coco_eval["bbox"].stats
    prec = evaluator.coco_eval["bbox"].eval["precision"]
    
    # UAV AP
    p_all = prec[:, :, 10, 0, -1]
    p_all = p_all[p_all > -1]
    uav_ap = float(np.mean(p_all)) if p_all.size else 0.0
    p50 = prec[0, :, 10, 0, -1]
    p50 = p50[p50 > -1]
    uav_ap50 = float(np.mean(p50)) if p50.size else 0.0
    
    avg_uav_score = float(np.mean(uav_scores)) if uav_scores else 0.0
    max_uav_score = float(np.max(uav_scores)) if uav_scores else 0.0
    
    return {
        "mAP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_s": float(stats[3]),
        "AP_m": float(stats[4]),
        "AP_l": float(stats[5]),
        "UAV_AP": uav_ap,
        "UAV_AP50": uav_ap50,
        "avg_uav_score": avg_uav_score,
        "max_uav_score": max_uav_score,
        "uav_det_count": len(uav_scores),
        "latency_ms": latency_ms,
        "fps": fps
    }

def main():
    device = "cuda:0"
    ckpt_path = "outputs/sovereign_v29_cycle5_stage2_anneal_2560/best_stg2.pth"
    print("=" * 110)
    print(" Sovereign V29 Multi-Resolution Inference Impact Benchmark ".center(110, "="))
    print("=" * 110)
    
    cfg = YAMLConfig("configs/deimv2/sovereign_v29_highres2560_anneal.yml")
    val_dataset = cfg.val_dataloader.dataset
    coco_gt = val_dataset.coco
    postprocessor = cfg.postprocessor
    
    print(f"[*] Loading model from {ckpt_path}...")
    model = cfg.model.to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    st = ck["ema"]["module"] if ("ema" in ck and "module" in ck["ema"]) else (ck["model"] if "model" in ck else ck)
    model.load_state_dict(st, strict=False)
    
    results = {}
    for name, h, w in RESOLUTIONS:
        print(f"\n[>] Testing resolution: {name} (H={h}, W={w}) on 200 images...")
        res = run_eval_at_resolution(model, postprocessor, val_dataset, coco_gt, h, w, device=device)
        results[name] = res
        print(f"    Done: mAP={res['mAP']*100:.2f}%, AP_small={res['AP_s']*100:.2f}%, UAV_AP={res['UAV_AP']*100:.2f}%, Latency={res['latency_ms']:.1f}ms ({res['fps']:.1f} FPS)")
        
    print("\n" + "=" * 115)
    print(" MULTI-RESOLUTION INFERENCE RESULTS SUMMARY (200 IMAGES) ".center(115, "="))
    print("=" * 115)
    hdr = f"{'Resolution':<20} | {'mAP':<7} | {'AP50':<7} | {'AP_s':<7} | {'AP_m':<7} | {'AP_l':<7} | {'UAV AP':<7} | {'UAV AvgConf':<11} | {'Latency':<9}"
    print(hdr)
    print("-" * 115)
    for name in results:
        r = results[name]
        print(f"{name:<20} | {r['mAP']*100:>5.2f}% | {r['AP50']*100:>5.2f}% | {r['AP_s']*100:>5.2f}% | {r['AP_m']*100:>5.2f}% | {r['AP_l']*100:>5.2f}% | {r['UAV_AP']*100:>5.2f}% | {r['avg_uav_score']:>11.3f} | {r['latency_ms']:>6.1f} ms")
    print("=" * 115)

    out_file = _root / "outputs/multi_resolution_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Detailed report saved to {out_file}")

if __name__ == "__main__":
    main()
