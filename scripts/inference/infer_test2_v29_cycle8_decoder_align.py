#!/usr/bin/env python3
"""
Sovereign V29 Cycle 8 Decoder & Head Aligned Model Inference Engine
Direct comparison between Cycle 5 Peak (baseline) and Cycle 8 Aligned on weak UAVs
"""

import os
import sys
import glob
import time
import zipfile
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

_root = Path("/home/constant/CodeBuddy/Dev/AIC")
if not _root.exists():
    _root = Path("/home/ailab/ssd/bigdata/hejianzhong/AIC_V29")
if not _root.exists():
    _root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

sys.path.insert(0, str(_root / 'vendor/DEIMv2'))
sys.path.insert(0, str(_root / 'scripts/lib'))
os.chdir(str(_root / 'vendor/DEIMv2'))

from engine.core import YAMLConfig

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

RARE_CLASSES = {1, 7, 9, 10, 11}

MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)

class TestSetPhase2Dataset(Dataset):
    def __init__(self, target_size=(1440, 2560)):
        self.target_size = target_size
        vis_dirs = [
            os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/visible"),
            "/dev/shm/aic_cache/test_visible",
        ]
        vis_dir = None
        for vd in vis_dirs:
            if os.path.exists(vd):
                vis_dir = vd
                break
        assert vis_dir, "Visible test image directory not found!"
        vis_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
        self.stems = [os.path.splitext(os.path.basename(f))[0] for f in vis_files if not f.endswith(".txt")]
        assert len(self.stems) == 1000, f"Expected 1,000 test images, got {len(self.stems)} from {vis_dir}"
        print(f"[+] Loaded {len(self.stems)} Phase 2 test images from {vis_dir}")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        vis_path = os.path.join(_root, f"dataset/test/AIC2026_PHASE_2_1000/visible/{stem}.jpg")
        ir_path = os.path.join(_root, f"work_dir/test_aligned_ir/{stem}.png")
        lbg_path = os.path.join(_root, f"work_dir/test_lbg_ir/{stem}.png")
        dep_path = os.path.join(_root, f"dataset/test/AIC2026_PHASE_2_1000/depth/{stem}.png")

        vis_img = Image.open(vis_path).convert("RGB")
        w_orig, h_orig = vis_img.size

        ir_img = Image.open(ir_path).convert("L")
        lbg_img = Image.open(lbg_path).convert("L")
        dep_img = Image.open(dep_path)

        rgb = np.array(vis_img, dtype=np.float32) / 255.0
        ir = np.array(ir_img, dtype=np.float32) / 255.0
        lbg = np.array(lbg_img, dtype=np.float32) / 255.0

        if dep_img.mode in ("I;16", "I"):
            dep = np.array(dep_img, dtype=np.float32) / 19999.0
        else:
            dep = np.array(dep_img, dtype=np.float32) / 255.0
        dep = np.clip(dep, 0.0, 1.0)

        tensor_6ch = np.zeros((6, h_orig, w_orig), dtype=np.float32)
        tensor_6ch[0:3] = rgb.transpose(2, 0, 1)
        tensor_6ch[3] = ir
        tensor_6ch[4] = lbg
        tensor_6ch[5] = dep

        t = torch.from_numpy(tensor_6ch).unsqueeze(0)
        t = F.interpolate(t, size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        t = (t - MEAN_6CH) / STD_6CH
        return t, w_orig, h_orig, stem


def run_inference(model, postprocessor, dataloader, desc="Cycle 8 (2560)"):
    raw_detections = {}
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (batch_samples, batch_w, batch_h, batch_stems) in enumerate(dataloader):
            batch_samples = batch_samples.cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(batch_samples)

            orig_sizes = torch.tensor([[float(w), float(h)] for w, h in zip(batch_w, batch_h)], device="cuda")
            results = postprocessor(outputs, orig_sizes, batch_samples)

            for res, orig_w, orig_h, stem in zip(results, batch_w, batch_h, batch_stems):
                w_f, h_f = float(orig_w), float(orig_h)
                labels = res['labels'].cpu().numpy()
                boxes = res['boxes'].float().cpu().numpy()
                scores = res['scores'].float().cpu().numpy()

                preds = []
                for lab, box, sco in zip(labels, boxes, scores):
                    cid, sc = int(lab), float(sco)
                    x1, y1, x2, y2 = box
                    cx = float(((x1 + x2) / 2.0) / w_f)
                    cy = float(((y1 + y2) / 2.0) / h_f)
                    bw = float((x2 - x1) / w_f)
                    bh = float((y2 - y1) / h_f)
                    preds.append((cid, cx, cy, bw, bh, sc))

                preds.sort(key=lambda x: x[5], reverse=True)
                raw_detections[stem] = preds

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(dataloader):
                elapsed = time.time() - t0
                pct = (batch_idx + 1) / len(dataloader) * 100.0
                print(f"[{desc}] [{batch_idx+1:3d}/{len(dataloader)}] {pct:5.1f}% | Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - t0
    return raw_detections, elapsed


def package_submission(detections, zip_path, filter_fn=None, max_cap=60):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for stem, boxes in sorted(detections.items()):
            if filter_fn is not None:
                boxes = filter_fn(boxes)
            boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
            if max_cap is not None:
                boxes = boxes[:max_cap]
            lines = []
            for b in boxes:
                cid, cx, cy, bw, bh, sc = b
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {sc:.6f}\r\n")
            zf.writestr(f"{stem}.txt", "".join(lines).encode('utf-8'))
    print(f"[✔] Packaged: {zip_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deimv2/sovereign_v29_cycle8_decoder_head_align.yml")
    parser.add_argument("--checkpoint", default="outputs/sovereign_v29_cycle8_decoder_head_align/best_stg2.pth")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print("=" * 80)
    print(" Sovereign V29 Cycle 8 Decoder Aligned Native 2.5K Inference ".center(80, "="))
    print("=" * 80)

    cfg = YAMLConfig(args.config)
    model = cfg.model.to(args.device)
    postprocessor = cfg.postprocessor

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ckpt["ema"]["module"] if ("ema" in ckpt and "module" in ckpt["ema"]) else (ckpt["model"] if "model" in ckpt else ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[✔] Model weights loaded! (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
    model.eval()

    dataset = TestSetPhase2Dataset(target_size=(1440, 2560))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    raw_detections, elapsed = run_inference(model, postprocessor, dataloader)
    out_dir = _root / "work_dir/submissions/v29_cycle8_align"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_pt_path = out_dir / "raw_detections_v29_c8_align.pt"
    torch.save(raw_detections, raw_pt_path)
    print(f"[✔] Saved full raw detections to {raw_pt_path}")

    # Generate Cap60
    pkg_cap60 = str(out_dir / "submission_sovereign_v29_c8_align_cap60.zip")
    package_submission(raw_detections, pkg_cap60, filter_fn=lambda bxs: [
        b for b in bxs if (b[5] >= 0.45 if b[0] in RARE_CLASSES else b[5] >= 0.50)
    ], max_cap=60)

    print(f"[✔] Package created: {pkg_cap60}")

if __name__ == "__main__":
    main()
