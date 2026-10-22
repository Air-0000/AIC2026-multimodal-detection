#!/usr/bin/env python3
"""
Hyperparameter Sweep for Stratified Quota Routing on AIC 200-image Validation Split.
Explores (p3_quota_ratio, p4_quota_ratio) grid to find optimal mAP / AP50 / AP75 / AP_s balance.
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = Path("/home/constant/CodeBuddy/Dev/AIC")
sys.path.insert(0, str(ROOT / "vendor/DEIMv2"))
sys.path.insert(0, str(ROOT / "scripts/lib"))
os.chdir(str(ROOT / "vendor/DEIMv2"))

from engine.core import YAMLConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANN_PATH = ROOT / "work_dir/deimv2_dataset/valid/annotations/instances_val.json"
MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)


class ValMultimodalDataset(Dataset):
    def __init__(self, ann_file, target_size=(1440, 2560)):
        self.target_size = target_size
        with open(ann_file) as f:
            self.ann = json.load(f)
        self.images = self.ann["images"]
        print(f"[+] Loaded {len(self.images)} validation images from {ann_file}")

    def __len__(self):
        return len(self.images)

    def _read_depth(self, fn, stem):
        search_dirs = [
            ROOT / "dataset/train/depth_p0",
            ROOT / "dataset/train/depth",
        ]
        for sdir in search_dirs:
            if not sdir.exists(): continue
            for ext in ('.png', '.jpg'):
                p = sdir / f"{stem}{ext}"
                if p.exists():
                    im = Image.open(p)
                    arr = np.array(im).astype(np.float32)
                    if arr.ndim == 3: arr = arr.mean(axis=2)
                    scale = 19999.0 if getattr(im, 'mode', '') in ('I;16', 'I') else 255.0
                    arr = (arr / scale).clip(0.0, 1.0)
                    return torch.from_numpy(arr).unsqueeze(0)
        raise FileNotFoundError(f"Depth not found for {stem}")

    def __getitem__(self, idx):
        item = self.images[idx]
        image_id = item["id"]
        fn = item["file_name"]
        stem = os.path.splitext(fn)[0]
        orig_w = item["width"]
        orig_h = item["height"]

        # 1. Visible RGB
        vis_path = ROOT / "dataset/train/visible" / fn
        if not vis_path.exists():
            vis_path = ROOT / "dataset/train/visible" / f"{stem}.png"
        vis_pil = Image.open(vis_path).convert("RGB")
        import torchvision.transforms as T
        rgb_t = T.functional.pil_to_tensor(vis_pil).float() / 255.0

        # 2. Aligned IR
        ira_path = ROOT / "dataset/train/infrared_aligned" / fn
        if not ira_path.exists():
            ira_path = ROOT / "dataset/train/infrared_aligned" / f"{stem}.png"
        ira_pil = Image.open(ira_path).convert("L")
        ira_t = T.functional.pil_to_tensor(ira_pil).float() / 255.0

        # 3. Raw/LBG IR
        ir_path = ROOT / "dataset/train/infrared" / fn
        if not ir_path.exists():
            ir_path = ROOT / "dataset/train/infrared" / f"{stem}.png"
        ir_pil = Image.open(ir_path).convert("L")
        lbg_t = T.functional.pil_to_tensor(ir_pil).float() / 255.0

        # 4. Depth
        depth_t = self._read_depth(fn, stem)

        if ira_t.shape[1:] != (orig_h, orig_w):
            ira_t = T.functional.resize(ira_t, [orig_h, orig_w])
        if lbg_t.shape[1:] != (orig_h, orig_w):
            lbg_t = T.functional.resize(lbg_t, [orig_h, orig_w])
        if depth_t.shape[1:] != (orig_h, orig_w):
            depth_t = T.functional.resize(depth_t, [orig_h, orig_w])

        sample = torch.cat([rgb_t, ira_t, lbg_t, depth_t], dim=0)
        sample = F.interpolate(sample.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
        sample = (sample - MEAN_6CH) / STD_6CH

        return sample, orig_w, orig_h, image_id


def load_v34_model():
    cfg = YAMLConfig(str(ROOT / "vendor/DEIMv2/configs/deimv2/sovereign_v34_clean_from_scratch_2560.yml"), resume=None)
    cfg.yaml_cfg['DEIM']['use_dinov3_distill'] = False
    cfg.yaml_cfg['DEIM']['use_co_aux'] = False
    model = cfg.model.eval()
    ckpt = torch.load(ROOT / "vendor/DEIMv2/outputs/sovereign_v34_clean_from_scratch_2560/best_stg2.pth", map_location='cpu', weights_only=False)
    state = ckpt.get('ema', {}).get('module', ckpt.get('model', ckpt))
    clean_state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items() if not k.startswith(('dinov3_distiller.', 'co_aux_head.'))}
    model.load_state_dict(clean_state, strict=False)
    model.cuda()
    postprocessor = cfg.postprocessor.eval().cuda()
    return model, postprocessor, cfg


def evaluate_model(model, postprocessor, dataloader, coco_gt, desc="Evaluating"):
    results = []
    with torch.no_grad():
        for sample, orig_w, orig_h, image_id in tqdm(dataloader, desc=desc):
            sample = sample.cuda()
            orig_size = torch.tensor([[orig_w.item(), orig_h.item()]]).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(sample)
                processed = postprocessor(outputs, orig_size)
            
            labels = processed[0]["labels"].cpu().numpy()
            boxes = processed[0]["boxes"].cpu().numpy()
            scores = processed[0]["scores"].cpu().numpy()

            img_id = image_id.item()
            for l, b, s in zip(labels, boxes, scores):
                if s < 0.05: continue
                x1, y1, x2, y2 = b
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                results.append({
                    "image_id": img_id,
                    "category_id": int(l),
                    "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(w), 2), round(float(h), 2)],
                    "score": round(float(s), 4)
                })

    if not results:
        return [0.0] * 12

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def main():
    print("=" * 85)
    print(" Hyperparameter Sweep: Stratified Routing Quotas on Valid Split ".center(85, "="))
    print("=" * 85)

    dataset = ValMultimodalDataset(ANN_PATH, target_size=(1440, 2560))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    coco_gt = COCO(str(ANN_PATH))

    model, postprocessor, cfg = load_v34_model()

    # Enable smooth depth fallback
    for m in model.modules():
        if m.__class__.__name__ == "DepthOcclusionGating":
            m.disable_fallback = False

    decoder = model.decoder
    decoder.use_pure_topk = False
    decoder.use_anisotropic_routing = True

    # Sweep grid
    sweep_grid = [
        {"p3": 0.20, "p4": 0.20, "name": "P3=20%, P4=20%, Glob=60% (宽裕全局)"},
        {"p3": 0.25, "p4": 0.20, "name": "P3=25%, P4=20%, Glob=55%"},
        {"p3": 0.25, "p4": 0.25, "name": "P3=25%, P4=25%, Glob=50% (中度平衡)"},
        {"p3": 0.30, "p4": 0.20, "name": "P3=30%, P4=20%, Glob=50%"},
        {"p3": 0.30, "p4": 0.25, "name": "P3=30%, P4=25%, Glob=45%"},
        {"p3": 0.30, "p4": 0.30, "name": "P3=30%, P4=30%, Glob=40% (激进保额)"},
        {"p3": 0.35, "p4": 0.25, "name": "P3=35%, P4=25%, Glob=40% (DFINE原版配额)"},
    ]

    results_table = []
    
    for i, cfg_item in enumerate(sweep_grid):
        p3 = cfg_item["p3"]
        p4 = cfg_item["p4"]
        name = cfg_item["name"]
        print(f"\n>>> [{i+1}/{len(sweep_grid)}] Evaluating Quota: P3={p3*100:.0f}%, P4={p4*100:.0f}%, Global={(1-p3-p4)*100:.0f}%...")
        
        decoder.p3_quota_ratio = p3
        decoder.p4_quota_ratio = p4
        
        t0 = time.time()
        stats = evaluate_model(model, postprocessor, dataloader, coco_gt, desc=f"[{i+1}/{len(sweep_grid)}] {name}")
        elapsed = time.time() - t0
        
        record = {
            "name": name,
            "p3": p3,
            "p4": p4,
            "global": round(1.0 - p3 - p4, 2),
            "mAP": stats[0],
            "AP50": stats[1],
            "AP75": stats[2],
            "AP_s": stats[3],
            "AP_m": stats[4],
            "AP_l": stats[5],
            "time_sec": round(elapsed, 1)
        }
        results_table.append(record)
        print(f"    --> mAP: {stats[0]:.4f} | AP50: {stats[1]:.4f} | AP75: {stats[2]:.4f} | AP_s: {stats[3]:.4f} | AP_l: {stats[5]:.4f} ({elapsed:.1f}s)")

    # Sort by mAP descending
    sorted_by_map = sorted(results_table, key=lambda x: x["mAP"], reverse=True)

    print("\n" + "=" * 95)
    print(" Stratified Quota Routing 超参网格搜索排行榜 (按 mAP 降序排列) ".center(95, "="))
    print("=" * 95)
    headers = ["排名", "配额配置", "mAP(0.5:0.95)", "AP50", "AP75", "AP_s (小目标)", "AP_m (中目标)", "AP_l (大目标)"]
    print(f"{headers[0]:<4} | {headers[1]:<35} | {headers[2]:<13} | {headers[3]:<8} | {headers[4]:<8} | {headers[5]:<13} | {headers[6]:<13} | {headers[7]:<13}")
    print("-" * 95)

    for rank, r in enumerate(sorted_by_map, 1):
        print(f"{rank:<4} | {r['name']:<35} | {r['mAP']:.4f}        | {r['AP50']:.4f}   | {r['AP75']:.4f}   | {r['AP_s']:.4f}        | {r['AP_m']:.4f}        | {r['AP_l']:.4f}")
    print("=" * 95)

    best = sorted_by_map[0]
    print(f"\n[★ 冠军超参配置]: {best['name']}")
    print(f"    - mAP: {best['mAP']:.4f} (基准: 0.5492, 提升: +{(best['mAP'] - 0.5492)*100:+.2f}%)")
    print(f"    - AP50: {best['AP50']:.4f} (基准: 0.8193, 提升: +{(best['AP50'] - 0.8193)*100:+.2f}%)")
    print(f"    - AP_s: {best['AP_s']:.4f}")
    print(f"    - AP_l: {best['AP_l']:.4f}")

    # Save to JSON
    out_json = ROOT / "outputs/metrics/quota_routing_sweep_results.json"
    with open(out_json, "w") as f:
        json.dump({"best_config": best, "all_results": sorted_by_map}, f, indent=2)
    print(f"[+] Full sweep results saved to {out_json}")


if __name__ == "__main__":
    main()
