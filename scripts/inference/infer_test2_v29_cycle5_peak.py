#!/usr/bin/env python3
"""
Sovereign V29 Cycle 5 Peak Model Native Inference & Packaging Engine
=====================================================================
Runs pure standalone native 6-channel inference on Phase 2 test set (1,000 images)
under native 2.5K ultra-high resolution (1440 x 2560) using Cycle 5 best checkpoint (mAP 0.5808).

Outputs compliant competition packages (CRLF, <=100 boxes/img, normalized cxcywh [0, 1]).
"""

import os
import sys
import glob
import time
import zipfile
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

_here = Path(__file__).resolve().parent
_root = _here.parent.parent
sys.path.insert(0, str(_root / 'vendor/DEIMv2'))
sys.path.insert(0, str(_root / 'scripts/lib'))
os.chdir(str(_root / 'vendor/DEIMv2'))

from engine.core import YAMLConfig

try:
    from destripe_1d import stripe_non_uniformity_correction_1d
    HAS_DESTRIPE = True
except ImportError:
    HAS_DESTRIPE = False

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

RARE_CLASSES = {1, 7, 9, 10, 11} # boat, ball, garbage can, uav, tricycle
COMMON_CLASSES = {0, 2, 3, 4, 5, 6, 8}

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

    def _read_depth(self, stem):
        search_dirs = [
            os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/depth_p0"),
            os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/depth"),
            "/dev/shm/aic_cache/test_depth_p0",
        ]
        for sdir in search_dirs:
            if not os.path.exists(sdir):
                continue
            for ext in ('.png', '.jpg'):
                p = os.path.join(sdir, f"{stem}{ext}")
                if os.path.exists(p):
                    im = Image.open(p)
                    arr = np.array(im).astype(np.float32)
                    if arr.ndim == 3:
                        arr = arr.mean(axis=2)
                    scale = 19999.0 if getattr(im, 'mode', '') in ('I;16', 'I') else 255.0
                    arr = (arr / scale).clip(0.0, 1.0)
                    return torch.from_numpy(arr).unsqueeze(0)
        raise FileNotFoundError(f"Depth not found for {stem}")

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # 1. Visible RGB
        vis_dirs = [
            os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/visible"),
            "/dev/shm/aic_cache/test_visible",
        ]
        vis_path = None
        for vd in vis_dirs:
            cands = glob.glob(os.path.join(vd, f"{stem}.*"))
            if cands:
                vis_path = cands[0]
                break
        vis_pil = Image.open(vis_path).convert("RGB")
        orig_w, orig_h = vis_pil.size
        rgb_t = T.functional.pil_to_tensor(vis_pil).float() / 255.0

        # 2. Aligned IR
        ir_dirs = [
            os.path.join(_root, "work_dir/test_aligned_ir"),
            os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/infrared_aligned"),
            os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/infrared"),
        ]
        ir_path = None
        for idr in ir_dirs:
            cands = glob.glob(os.path.join(idr, f"{stem}.*"))
            if cands:
                ir_path = cands[0]
                break
        ir_pil = Image.open(ir_path).convert("L")
        if HAS_DESTRIPE:
            try:
                ir_arr = np.array(ir_pil)
                ir_clean = stripe_non_uniformity_correction_1d(ir_arr, max_stripe_width=5, blend=0.85)
                ir_pil = Image.fromarray(ir_clean)
            except Exception:
                pass
        ir_t = T.functional.pil_to_tensor(ir_pil).float() / 255.0

        # 3. LBG IR
        lbg_dirs = [
            os.path.join(_root, "work_dir/test_lbg_ir"),
            os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/infrared"),
        ]
        lbg_path = None
        for ld in lbg_dirs:
            cands = glob.glob(os.path.join(ld, f"{stem}.*"))
            if cands:
                lbg_path = cands[0]
                break
        if lbg_path:
            lbg_pil = Image.open(lbg_path).convert("L")
            lbg_t = T.functional.pil_to_tensor(lbg_pil).float() / 255.0
        else:
            lbg_t = ir_t.clone()

        # 4. Depth P0 (16-bit millimeter)
        depth_t = self._read_depth(stem)

        if ir_t.shape[1:] != (orig_h, orig_w):
            ir_t = T.functional.resize(ir_t, [orig_h, orig_w])
        if lbg_t.shape[1:] != (orig_h, orig_w):
            lbg_t = T.functional.resize(lbg_t, [orig_h, orig_w])
        if depth_t.shape[1:] != (orig_h, orig_w):
            depth_t = T.functional.resize(depth_t, [orig_h, orig_w])

        # Stack into 6 channels: [RGB (3), IR (1), LBG (1), Depth (1)]
        sample = torch.cat([rgb_t, ir_t, lbg_t, depth_t], dim=0)

        # Bilinear resize to 2.5K native resolution (1440, 2560)
        sample = F.interpolate(sample.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
        sample = (sample - MEAN_6CH) / STD_6CH

        return sample, orig_w, orig_h, stem


def run_inference(model, postprocessor, dataloader, desc="Sovereign V29 C5 Peak Inference"):
    raw_detections = defaultdict(list)
    t0 = time.time()
    with torch.no_grad():
        for batch_samples, batch_w, batch_h, batch_stems in tqdm(dataloader, desc=desc):
            batch_samples = batch_samples.cuda()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(batch_samples)

            orig_sizes = torch.tensor([[float(w), float(h)] for w, h in zip(batch_w, batch_h)], device="cuda")
            results = postprocessor(outputs, orig_sizes, batch_samples)

            for res, orig_w, orig_h, stem in zip(results, batch_w, batch_h, batch_stems):
                w_f = float(orig_w)
                h_f = float(orig_h)

                labels = res['labels'].cpu().numpy()
                boxes = res['boxes'].float().cpu().numpy()
                scores = res['scores'].float().cpu().numpy()

                preds = []
                for lab, box, sco in zip(labels, boxes, scores):
                    cid = int(lab)
                    sc = float(sco)
                    x1, y1, x2, y2 = box

                    # Normalize xyxy to cxcywh in [0, 1]
                    cx = float(((x1 + x2) / 2.0) / w_f)
                    cy = float(((y1 + y2) / 2.0) / h_f)
                    bw = float((x2 - x1) / w_f)
                    bh = float((y2 - y1) / h_f)

                    # Clamp to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))

                    preds.append((cid, cx, cy, bw, bh, sc))

                # Sort by confidence descending
                preds.sort(key=lambda x: x[5], reverse=True)
                raw_detections[stem] = preds

    elapsed = time.time() - t0
    return raw_detections, elapsed


def package_submission(detections, zip_path, filter_fn=None, max_cap=100):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for stem, boxes in sorted(detections.items()):
            if filter_fn is not None:
                boxes = filter_fn(boxes)
            # Ensure confidence descending
            boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
            if max_cap is not None:
                boxes = boxes[:max_cap]
            lines = []
            for b in boxes:
                cid, cx, cy, bw, bh, sc = b
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {sc:.4f}")
            # Strictly CRLF line endings per official spec
            txt_content = "\r\n".join(lines) + ("\r\n" if lines else "")
            zf.writestr(f"{stem}.txt", txt_content)


def parse_submission_zip(zip_path):
    total_boxes = 0
    cls_counts = Counter()
    scores = []
    img_counts = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for fname in sorted(zf.namelist()):
            if not fname.endswith(".txt"):
                continue
            content = zf.read(fname).decode('utf-8', errors='ignore').strip()
            if not content:
                img_counts.append(0)
                continue
            lines = content.splitlines()
            img_counts.append(len(lines))
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cid = int(parts[0])
                    conf = float(parts[5])
                    cls_counts[cid] += 1
                    scores.append(conf)
                    total_boxes += 1

    return {
        "total_boxes": total_boxes,
        "scores": np.array(scores, dtype=np.float32) if scores else np.array([], dtype=np.float32),
        "cls_counts": cls_counts,
        "img_counts": np.array(img_counts, dtype=np.int32),
        "empty_images": int(sum(1 for c in img_counts if c == 0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Sovereign V29 Cycle 5 Native Inference & Packaging")
    parser.add_argument("--config", default="configs/deimv2/sovereign_v29_highres2560_anneal.yml")
    parser.add_argument("--checkpoint", default="outputs/sovereign_v29_cycle5_stage2_anneal_2560/best_stg2.pth")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default=None, help="Custom output directory")
    args = parser.parse_args()

    print("=" * 80)
    print(" Sovereign V29 Cycle 5 Peak (mAP 0.5808) Native 2.5K Inference Engine ".center(80, "="))
    print("=" * 80)

    # 1. Check Checkpoint
    if not os.path.exists(args.checkpoint):
        cands = [
            args.checkpoint,
            "outputs/champions/sovereign_v29_alltime_peak_c5_5756.pth",
            "outputs/champions/sovereign_v29_champion.pth",
        ]
        found = False
        for c in cands:
            if os.path.exists(c):
                args.checkpoint = c
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    print(f"[+] Using Configuration: {args.config}")
    print(f"[+] Using Checkpoint:    {args.checkpoint}")

    # Load Model & Postprocessor
    cfg = YAMLConfig(args.config)
    model = cfg.model.to(args.device)
    postprocessor = cfg.postprocessor

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if "ema" in ckpt and "module" in ckpt["ema"]:
        print("[✔] Detected EMA state in checkpoint, using high-precision EMA weights!")
        state = ckpt["ema"]["module"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
        
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[✔] Model weights loaded successfully! (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

    model.eval()

    # 2. Build Dataset & DataLoader at 2.5K Resolution (1440, 2560)
    dataset = TestSetPhase2Dataset(target_size=(1440, 2560))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Run Inference over 1,000 Test Images
    print("\n[*] Commencing Pure Native 6-Channel 2.5K Inference over 1,000 Test Images...")
    raw_detections, elapsed = run_inference(model, postprocessor, dataloader, desc="Sovereign V29 C5 (2560)")
    out_dir = Path(args.out_dir) if args.out_dir else (_root / "work_dir/submissions/v29_cycle5_peak")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw detections for instantaneous offline threshold sweep
    raw_pt_path = out_dir / "raw_detections.pt"
    torch.save(raw_detections, raw_pt_path)
    print(f"[✔] Saved full raw detections to {raw_pt_path}")

    # 4. Generate Submissions
    # A) Cap60 (F5 Standard: Rare >= 0.45, Common >= 0.50, max 60 boxes)
    pkg_cap60 = str(out_dir / "submission_sovereign_v29_c5_peak_cap60.zip")
    package_submission(raw_detections, pkg_cap60, filter_fn=lambda bxs: [
        b for b in bxs if (b[5] >= 0.45 if b[0] in RARE_CLASSES else b[5] >= 0.50)
    ], max_cap=60)

    # B) Native Standard Baseline (Rare >= 0.45, Common >= 0.50, max 100 boxes)
    pkg_native = str(out_dir / "submission_sovereign_v29_c5_peak_native_thr45_50.zip")
    package_submission(raw_detections, pkg_native, filter_fn=lambda bxs: [
        b for b in bxs if (b[5] >= 0.45 if b[0] in RARE_CLASSES else b[5] >= 0.50)
    ], max_cap=100)

    # C) Thr45 Uniform (All classes >= 0.45, max 100 boxes)
    pkg_thr45 = str(out_dir / "submission_sovereign_v29_c5_peak_thr45.zip")
    package_submission(raw_detections, pkg_thr45, filter_fn=lambda bxs: [b for b in bxs if b[5] >= 0.45], max_cap=100)

    # D) Thr35 High Recall (All classes >= 0.35, max 100 boxes)
    pkg_thr35 = str(out_dir / "submission_sovereign_v29_c5_peak_thr35.zip")
    package_submission(raw_detections, pkg_thr35, filter_fn=lambda bxs: [b for b in bxs if b[5] >= 0.35], max_cap=100)

    # E) Thr30 User Target (All classes >= 0.30, max 100 boxes)
    pkg_thr30 = str(out_dir / "submission_sovereign_v29_c5_peak_thr30.zip")
    package_submission(raw_detections, pkg_thr30, filter_fn=lambda bxs: [b for b in bxs if b[5] >= 0.30], max_cap=100)

    # F) Thr25 Ultra Recall (All classes >= 0.25, max 100 boxes)
    pkg_thr25 = str(out_dir / "submission_sovereign_v29_c5_peak_thr25.zip")
    package_submission(raw_detections, pkg_thr25, filter_fn=lambda bxs: [b for b in bxs if b[5] >= 0.25], max_cap=100)

    print(f"\n[✔] Successfully generated competition packages:")
    print(f"  • Cap60 (F5 Standard): {pkg_cap60}")
    print(f"  • Native (45/50 Cap100):{pkg_native}")
    print(f"  • Uniform Thr45:        {pkg_thr45}")
    print(f"  • High Recall Thr35:    {pkg_thr35}")
    print(f"  • Target Thr30:         {pkg_thr30}")
    print(f"  • Ultra Recall Thr25:   {pkg_thr25}")

    # 5. Parse & Print Macro Comparison
    p_cap60 = parse_submission_zip(pkg_cap60)
    p_native = parse_submission_zip(pkg_native)
    p_thr45 = parse_submission_zip(pkg_thr45)
    p_thr35 = parse_submission_zip(pkg_thr35)

    print("\n" + "=" * 95)
    print(" Macro Comparison: Sovereign V29 Cycle 5 Submission Variants ".center(95, "="))
    print("=" * 95)
    print(f"{'Package Variant':<36} | {'Total Boxes':<12} | {'Empty Imgs':<10} | {'Mean Conf':<10}")
    print("-" * 95)
    for name, p in [
        ("★ V29 Cycle 5 Cap60 (F5 Standard)", p_cap60),
        ("★ V29 Cycle 5 Native (45/50 Cap100)", p_native),
        ("★ V29 Cycle 5 Uniform Thr45", p_thr45),
        ("★ V29 Cycle 5 High-Recall Thr35", p_thr35),
    ]:
        mean_sc = float(np.mean(p["scores"])) if len(p["scores"]) > 0 else 0.0
        print(f"{name:<36} | {p['total_boxes']:<12} | {p['empty_images']:<10} | {mean_sc:.4f}")
    print("=" * 95)

    # Class Breakdown Table
    print("\n" + "=" * 95)
    print(" 12-Class Breakdown for V29 Cycle 5 Peak ".center(95, "="))
    print("=" * 95)
    print(f"{'CID':<3} | {'Class Name':<12} | {'Cap60':<12} | {'Native (45/50)':<16} | {'Thr45':<12} | {'Thr35':<12}")
    print("-" * 95)
    for cid in range(12):
        cname = CLASS_NAMES[cid]
        c1 = p_cap60["cls_counts"][cid]
        c2 = p_native["cls_counts"][cid]
        c3 = p_thr45["cls_counts"][cid]
        c4 = p_thr35["cls_counts"][cid]
        print(f"{cid:<3} | {cname:<12} | {c1:<12} | {c2:<16} | {c3:<12} | {c4:<12}")
    print("=" * 95)


if __name__ == "__main__":
    main()
