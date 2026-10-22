#!/usr/bin/env python3
"""
Evaluate Sovereign V37 Fast 8-Epoch Micro Fine-Tuned Checkpoint on 200-Image Validation Set.
Directly compares with baseline V34 to measure net mAP/AP50/AP75 gain.
"""

import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
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

    def __len__(self):
        return len(self.images)

    def _read_depth(self, fn, stem):
        for sdir in [ROOT / "dataset/train/depth_p0", ROOT / "dataset/train/depth"]:
            if not sdir.exists(): continue
            for ext in ('.png', '.jpg'):
                p = sdir / f"{stem}{ext}"
                if p.exists():
                    img = Image.open(p)
                    arr = np.array(img, dtype=np.float32)
                    if arr.ndim == 2:
                        return arr / 19999.0
                    return arr[:, :, 0] / 19999.0
        return np.full((1080, 1920), 0.5, dtype=np.float32)

    def __getitem__(self, idx):
        item = self.images[idx]
        fn = item["file_name"]
        stem = Path(fn).stem

        # RGB
        rgb_path = ROOT / "dataset/train/visible" / fn
        if not rgb_path.exists():
            rgb_path = ROOT / "dataset/train/visible" / f"{stem}.jpg"
        rgb_img = Image.open(rgb_path).convert("RGB")
        orig_w, orig_h = rgb_img.size

        # Aligned IR
        ir_a_path = ROOT / "dataset/train/infrared_aligned" / f"{stem}.jpg"
        if not ir_a_path.exists():
            ir_a_path = ROOT / "dataset/train/infrared" / f"{stem}.jpg"
        ir_a_img = Image.open(ir_a_path).convert("L")

        # LBG IR
        ir_lbg_path = ROOT / "work_dir/train_lbg_ir" / f"{stem}.png"
        if not ir_lbg_path.exists():
            ir_lbg_path = ROOT / "work_dir/train_lbg_ir" / f"{stem}.jpg"
        if ir_lbg_path.exists():
            ir_lbg_img = Image.open(ir_lbg_path).convert("L")
        else:
            ir_lbg_img = ir_a_img

        # Depth
        depth_arr = self._read_depth(fn, stem)
        depth_img = Image.fromarray((np.clip(depth_arr, 0, 1) * 65535.0).astype(np.uint16))

        # Resize to target
        th, tw = self.target_size
        rgb_resized = rgb_img.resize((tw, th), Image.BILINEAR)
        ir_a_resized = ir_a_img.resize((tw, th), Image.BILINEAR)
        ir_lbg_resized = ir_lbg_img.resize((tw, th), Image.BILINEAR)
        depth_resized = depth_img.resize((tw, th), Image.NEAREST)

        rgb_t = torch.from_numpy(np.array(rgb_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        ir_a_t = torch.from_numpy(np.array(ir_a_resized, dtype=np.float32) / 255.0).unsqueeze(0)
        ir_lbg_t = torch.from_numpy(np.array(ir_lbg_resized, dtype=np.float32) / 255.0).unsqueeze(0)
        depth_t = torch.from_numpy(np.array(depth_resized, dtype=np.float32) / 65535.0).unsqueeze(0)

        tensor_6ch = torch.cat([rgb_t, ir_a_t, ir_lbg_t, depth_t], dim=0)
        tensor_6ch = (tensor_6ch - MEAN_6CH) / STD_6CH
        return tensor_6ch, orig_w, orig_h, item["id"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/sovereign_v37_fast8ep_verify/best_stg2.pth")
    parser.add_argument("--cfg", type=str, default="configs/deimv2/sovereign_v37_fast8ep_verify.yml")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / "vendor/DEIMv2" / ckpt_path

    if not ckpt_path.exists():
        print(f"[-] Checkpoint {ckpt_path} not found!")
        return

    print(f"[+] Loading model config: {args.cfg}")
    cfg = YAMLConfig(str(ROOT / "vendor/DEIMv2" / args.cfg), resume=None)
    cfg.yaml_cfg['DEIM']['use_dinov3_distill'] = False
    cfg.yaml_cfg['DEIM']['use_co_aux'] = False
    model = cfg.model.eval()

    print(f"[+] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('ema', {}).get('module', ckpt.get('model', ckpt))
    clean_state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items() if not k.startswith(('dinov3_distiller.', 'co_aux_head.'))}
    model.load_state_dict(clean_state, strict=False)
    model.cuda()
    postprocessor = cfg.postprocessor.eval().cuda()

    val_dataset = ValMultimodalDataset(str(ANN_PATH))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    coco_gt = COCO(str(ANN_PATH))

    results = []
    with torch.no_grad():
        for sample, orig_w, orig_h, image_id in tqdm(val_loader, desc="Evaluating V37"):
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

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats.tolist()
    out_json = ROOT / "outputs/metrics/v37_fast8ep_val_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "ckpt": str(ckpt_path),
            "mAP": stats[0],
            "AP50": stats[1],
            "AP75": stats[2],
            "APs": stats[3],
            "APm": stats[4],
            "APl": stats[5],
            "all_stats": stats
        }, f, indent=2)

    print("\n" + "="*70)
    print("=== SOVEREIGN V37 FAST VERIFICATION FINAL SCORECARD ===")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"mAP (0.50:0.95): {stats[0]:.4f} (Baseline V34: 0.5492)")
    print(f"AP50:           {stats[1]:.4f} (Baseline V34: 0.8193)")
    print(f"AP75:           {stats[2]:.4f} (Baseline V34: 0.5986)")
    print(f"AP_small:       {stats[3]:.4f}")
    print(f"AP_medium:      {stats[4]:.4f}")
    print(f"AP_large:       {stats[5]:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
