#!/usr/bin/env python3
"""
Zero-Shot / No-Training Validation Evaluation on AIC 200-image Validation Split.
Compares:
  1. Baseline: V34 Default (Original DOG without fallback + Default Top-K)
  2. Mod 1: V34 + Smooth Depth Fallback Mask (Lossless transmission in far/blind depth)
  3. Mod 1 + Mod 2: V34 + Smooth Depth Fallback + Anisotropic Stratified Routing (P3 30%, P4 30%, Global 40%)
"""

import os
import sys
import json
import time
import glob
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

        # Stack 6 channels
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
                # Convert xyxy to xywh
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
        print("[!] No detections found!")
        return [0.0] * 12

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def main():
    print("=" * 80)
    print(" Sovereign V34 Zero-Shot Validation Evaluation on AIC Valid Split ".center(80, "="))
    print("=" * 80)

    dataset = ValMultimodalDataset(ANN_PATH, target_size=(1440, 2560))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    coco_gt = COCO(str(ANN_PATH))

    model, postprocessor, cfg = load_v34_model()

    # Find DepthOcclusionGating module in model
    dog_module = None
    for m in model.modules():
        if m.__class__.__name__ == "DepthOcclusionGating":
            dog_module = m
            break

    decoder = model.decoder

    eval_matrix = {}

    # -------------------------------------------------------------
    # Run 1: Baseline (Original DOG, Default Pure Top-K)
    # -------------------------------------------------------------
    print("\n>>> [1/3] Running Baseline: Original DOG (no fallback) + Pure Top-K...")
    if dog_module is not None:
        dog_module.disable_fallback = True
    decoder.use_pure_topk = True
    decoder.use_anisotropic_routing = False
    stats_base = evaluate_model(model, postprocessor, dataloader, coco_gt, desc="[1/3] V34 Baseline")
    eval_matrix["Baseline (原版无防御)"] = stats_base

    # -------------------------------------------------------------
    # Run 2: Mod 1: Smooth Depth Fallback Mask Only
    # -------------------------------------------------------------
    print("\n>>> [2/3] Running Mod 1: With Smooth Depth Fallback Mask + Pure Top-K...")
    if dog_module is not None:
        dog_module.disable_fallback = False
    decoder.use_pure_topk = True
    decoder.use_anisotropic_routing = False
    stats_mod1 = evaluate_model(model, postprocessor, dataloader, coco_gt, desc="[2/3] V34 + Smooth Depth Fallback")
    eval_matrix["+ Smooth Depth Fallback"] = stats_mod1

    # -------------------------------------------------------------
    # Run 3: Mod 1 + Mod 2: Smooth Depth Fallback + Stratified Routing
    # -------------------------------------------------------------
    print("\n>>> [3/3] Running Mod 1 + Mod 2: Smooth Depth Fallback + Stratified Routing (P3 30%, P4 30%)...")
    if dog_module is not None:
        dog_module.disable_fallback = False
    decoder.use_pure_topk = False
    decoder.use_anisotropic_routing = True
    decoder.p3_quota_ratio = 0.30
    decoder.p4_quota_ratio = 0.30
    stats_mod2 = evaluate_model(model, postprocessor, dataloader, coco_gt, desc="[3/3] V34 + Fallback + Stratified Routing")
    eval_matrix["+ Fallback + Stratified Routing"] = stats_mod2

    # -------------------------------------------------------------
    # Print Beautiful Final Score Card
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" Sovereign V34 免训练即时测分结果对比表 (200 张带真值验证集) ".center(90, "="))
    print("=" * 90)
    headers = ["评估配置", "mAP(0.5:0.95)", "AP50", "AP75", "AP_s (小目标)", "AP_m (中目标)", "AP_l (大目标)"]
    print(f"{headers[0]:<30} | {headers[1]:<13} | {headers[2]:<8} | {headers[3]:<8} | {headers[4]:<13} | {headers[5]:<13} | {headers[6]:<13}")
    print("-" * 90)

    for name, stats in eval_matrix.items():
        map_all = f"{stats[0]:.4f}"
        ap50 = f"{stats[1]:.4f}"
        ap75 = f"{stats[2]:.4f}"
        ap_s = f"{stats[3]:.4f}"
        ap_m = f"{stats[4]:.4f}"
        ap_l = f"{stats[5]:.4f}"
        print(f"{name:<30} | {map_all:<13} | {ap50:<8} | {ap75:<8} | {ap_s:<13} | {ap_m:<13} | {ap_l:<13}")
    print("=" * 90)

    # Save results to json
    out_json = ROOT / "outputs/metrics/v34_zeroshot_val_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {k: {
        "mAP": stats[0], "AP50": stats[1], "AP75": stats[2],
        "AP_s": stats[3], "AP_m": stats[4], "AP_l": stats[5]
    } for k, stats in eval_matrix.items()}
    with open(out_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[+] Results saved to {out_json}")


if __name__ == "__main__":
    main()
