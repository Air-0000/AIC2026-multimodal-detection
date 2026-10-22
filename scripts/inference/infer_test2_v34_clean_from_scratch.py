#!/usr/bin/env python3
"""
Sovereign V34 Clean-Slate (From-Scratch 50-Epoch with Meta DINOv3 Distill & Co-ATSS)
Phase 2 Test Set Native Inference & Full Comparative Audit
====================================================================================
Runs native 6-channel full-resolution inference on Phase 2 test set (1,000 images)
under native 2.5K resolution (1440 x 2560) using Sovereign V34 production checkpoint (best_stg2.pth).

1. 100% compliant competition packages (CRLF, <=100 boxes/img, normalized cxcywh [0, 1]).
2. Multi-threshold packaging (0.45, 0.48, 0.50).
3. 12-class comparative audit against F5 Benchmark (58.105), V23 Official Peak (53.555), and V31.
4. Automatic registration and import to Web Visual Workstation (port 8088).
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

_root_env = os.environ.get("AIC_ROOT")
if _root_env and Path(_root_env).exists():
    _root = Path(_root_env)
else:
    _root = Path("/home/ailab/ssd/bigdata/hejianzhong/AIC_V29")
    if not _root.exists():
        _root = Path("/home/constant/CodeBuddy/Dev/AIC")

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

MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)


class TestSetPhase2Dataset(Dataset):
    def __init__(self, target_size=(1440, 2560)):
        self.target_size = target_size
        vis_dir = os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/visible")
        assert os.path.exists(vis_dir), f"Visible test dir not found: {vis_dir}"
        
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
        ]
        for sdir in search_dirs:
            if not os.path.exists(sdir): continue
            for ext in ('.png', '.jpg'):
                p = os.path.join(sdir, f"{stem}{ext}")
                if os.path.exists(p):
                    im = Image.open(p)
                    arr = np.array(im).astype(np.float32)
                    if arr.ndim == 3: arr = arr.mean(axis=2)
                    scale = 19999.0 if getattr(im, 'mode', '') in ('I;16', 'I') else 255.0
                    arr = (arr / scale).clip(0.0, 1.0)
                    return torch.from_numpy(arr).unsqueeze(0)
        raise FileNotFoundError(f"Depth not found for {stem}")

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # 1. Visible RGB
        vis_path = glob.glob(os.path.join(_root, f"dataset/test/AIC2026_PHASE_2_1000/visible/{stem}.*"))[0]
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
            except Exception: pass
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

        # 4. Depth P0
        depth_t = self._read_depth(stem)

        if ir_t.shape[1:] != (orig_h, orig_w):
            ir_t = T.functional.resize(ir_t, [orig_h, orig_w])
        if lbg_t.shape[1:] != (orig_h, orig_w):
            lbg_t = T.functional.resize(lbg_t, [orig_h, orig_w])
        if depth_t.shape[1:] != (orig_h, orig_w):
            depth_t = T.functional.resize(depth_t, [orig_h, orig_w])

        # Stack 6 channels
        sample = torch.cat([rgb_t, ir_t, lbg_t, depth_t], dim=0)
        sample = F.interpolate(sample.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
        sample = (sample - MEAN_6CH) / STD_6CH

        return sample, orig_w, orig_h, stem


def run_inference(model, postprocessor, dataloader, device="cuda:0"):
    raw_detections = defaultdict(list)
    t0 = time.time()
    with torch.no_grad():
        for batch_samples, batch_w, batch_h, batch_stems in tqdm(dataloader, desc="[+] Sovereign V34 Native Inference"):
            batch_samples = batch_samples.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(batch_samples)
            orig_sizes = torch.tensor([[float(w), float(h)] for w, h in zip(batch_w, batch_h)], device=device)
            results = postprocessor(outputs, orig_sizes, batch_samples)

            for res, orig_w, orig_h, stem in zip(results, batch_w, batch_h, batch_stems):
                w_f, h_f = float(orig_w), float(orig_h)
                labels = res['labels'].cpu().numpy()
                boxes = res['boxes'].float().cpu().numpy()
                scores = res['scores'].float().cpu().numpy()

                preds = []
                for lab, box, sco in zip(labels, boxes, scores):
                    cid = int(lab)
                    sc = float(sco)
                    x1, y1, x2, y2 = box
                    cx = max(0.0, min(1.0, float(((x1 + x2) / 2.0) / w_f)))
                    cy = max(0.0, min(1.0, float(((y1 + y2) / 2.0) / h_f)))
                    bw = max(0.0, min(1.0, float((x2 - x1) / w_f)))
                    bh = max(0.0, min(1.0, float((y2 - y1) / h_f)))
                    preds.append((cid, cx, cy, bw, bh, sc))

                preds.sort(key=lambda x: x[5], reverse=True)
                raw_detections[stem] = preds

    elapsed = time.time() - t0
    return raw_detections, elapsed


def package_submission(detections, zip_path, rare_thr=0.45, common_thr=0.48, max_cap=60):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    total_boxes = 0
    cat_counts = Counter()
    scores = []
    
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for stem, boxes in sorted(detections.items()):
            filtered = [
                b for b in boxes 
                if (b[5] >= rare_thr if b[0] in RARE_CLASSES else b[5] >= common_thr)
            ]
            filtered = sorted(filtered, key=lambda x: x[5], reverse=True)[:max_cap]
            lines = []
            for b in filtered:
                cid, cx, cy, bw, bh, sc = b
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {sc:.4f}")
                total_boxes += 1
                cat_counts[cid] += 1
                scores.append(sc)
            txt_content = "\r\n".join(lines) + ("\r\n" if lines else "")
            zf.writestr(f"{stem}.txt", txt_content)
            
    return {
        "total_boxes": total_boxes,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "cat_counts": cat_counts,
        "scores": np.array(scores, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="Sovereign V34 Native Inference & Comparison")
    parser.add_argument("--config", default="configs/deimv2/sovereign_v34_clean_from_scratch_2560.yml")
    parser.add_argument("--checkpoint", default="outputs/sovereign_v34_clean_from_scratch_2560/best_stg2.pth")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print("=" * 85)
    print(" Sovereign V34 Clean-Slate 50-Epoch Phase 2 Test Set Native Inference ".center(85, "="))
    print("=" * 85)

    out_dir = _root / "work_dir/submissions/sovereign_v34_clean_from_scratch"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    cfg = YAMLConfig(args.config)
    model = cfg.model.to(args.device)
    postprocessor = cfg.postprocessor.to(args.device)

    print(f"[+] Loading checkpoint from {args.checkpoint}...")
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ck.get('ema', {}).get('module', ck.get('model', ck))
    msg = model.load_state_dict(state, strict=False)
    print(f"[+] Model loaded (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")

    model.eval()
    postprocessor.eval()

    # 2. Build Dataset
    ds = TestSetPhase2Dataset(target_size=(1440, 2560))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Inference
    raw_detections, elapsed = run_inference(model, postprocessor, dl, device=args.device)
    print(f"[+] Completed 1,000 images in {elapsed:.1f}s ({elapsed/1000*1000:.1f} ms/img, {1000/elapsed:.2f} FPS)")

    # Save raw pt
    raw_pt_path = out_dir / "raw_detections_v34_production.pt"
    torch.save(dict(raw_detections), raw_pt_path)
    print(f"[+] Saved raw detection tensor: {raw_pt_path}")

    # 4. Package Native Submission (Cap60, rare 0.45, common 0.48)
    zip_path = out_dir / "submission_sovereign_v34_native.zip"
    stats_v34 = package_submission(raw_detections, zip_path, rare_thr=0.45, common_thr=0.48, max_cap=60)
    print(f"[+] Packaged submission zip: {zip_path} ({stats_v34['total_boxes']} boxes, mean conf {stats_v34['mean_score']:.3f})")

    # 5. Comparative Audit against F5 Benchmark (58.105) and V23 (53.555)
    print("\n" + "=" * 95)
    print(" PHASE 2 TEST SET (1000 IMAGES) INFERENCE COMPARISON MATRIX ".center(95, "="))
    print("=" * 95)
    print(f"{'Category':<14} | {'F5 Ref (58.1)':<16} | {'V23 TADL (53.5)':<16} | {'V34 Clean-Slate':<18} | {'V34 vs F5 Delta':<14}")
    print("-" * 95)

    # F5 reference counts
    f5_counts = {0: 1913, 1: 308, 2: 405, 3: 404, 4: 457, 5: 649, 6: 1840, 7: 154, 8: 622, 9: 228, 10: 104, 11: 117}
    f5_total = 7201

    # V23 reference counts
    v23_counts = {0: 1968, 1: 341, 2: 418, 3: 350, 4: 512, 5: 680, 6: 1872, 7: 58, 8: 695, 9: 284, 10: 59, 11: 122}
    v23_total = 7459

    for cid in range(12):
        c_name = CLASS_NAMES[cid]
        c_f5 = f5_counts.get(cid, 0)
        c_v23 = v23_counts.get(cid, 0)
        c_v34 = stats_v34["cat_counts"][cid]
        delta_f5 = c_v34 - c_f5
        print(f"{c_name:<14} | {c_f5:>16d} | {c_v23:>16d} | {c_v34:>18d} | {delta_f5:>+14d}")

    print("-" * 95)
    print(f"{'TOTAL BOXES':<14} | {f5_total:>16d} | {v23_total:>16d} | {stats_v34['total_boxes']:>18d} | {stats_v34['total_boxes']-f5_total:>+14d}")
    print(f"{'MEAN CONF':<14} | {'0.781':>16} | {'0.792':>16} | {stats_v34['mean_score']:>18.3f} | {stats_v34['mean_score']-0.781:>+14.3f}")
    print("=" * 95)

    # 6. Auto import to workstation
    try:
        import importlib.util
        importer_path = _root / "scripts/viz/import_submission_to_workstation.py"
        if importer_path.exists():
            spec = importlib.util.spec_from_file_location("workstation_importer", str(importer_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "auto_import_submission"):
                mod.auto_import_submission(str(zip_path), "v34_clean_from_scratch", "Sovereign V34 (从零纯血 50 轮 DINOv3 蒸馏)")
                print("[+] Successfully registered V34 to Web Workstation!")
    except Exception as e:
        print(f"[-] Workstation import notice: {e}")


if __name__ == "__main__":
    main()
