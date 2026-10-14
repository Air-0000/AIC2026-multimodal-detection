#!/usr/bin/env python3
"""
Sovereign V30 (CSSP + ABC) Phase-2 Test Set Inference & Packaging Engine
"""

import os
import sys
import glob
import time
import zipfile
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

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
RARE_CLASSES = {1, 7, 9, 10, 11}

MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)


class TestPhase2Dataset(Dataset):
    def __init__(self, target_size=(576, 1024)):
        self.target_size = target_size
        vis_dir = os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/visible")
        vis_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
        self.stems = [os.path.splitext(os.path.basename(f))[0] for f in vis_files if not f.endswith(".txt")]
        assert len(self.stems) == 1000, f"Expected 1,000 test images, got {len(self.stems)}"
        print(f"[+] Loaded {len(self.stems)} Phase-2 test images from {vis_dir}")

    def __len__(self):
        return len(self.stems)

    def _read_depth(self, stem):
        search_dirs = [
            os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/depth"),
            os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/depth_p0"),
        ]
        for sdir in search_dirs:
            if not os.path.exists(sdir): continue
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

    def _find_img(self, dir_path, stem):
        for ext in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
            p = os.path.join(dir_path, f"{stem}{ext}")
            if os.path.exists(p):
                return p
        return None

    def __getitem__(self, idx):
        stem = self.stems[idx]
        rgb_path = self._find_img(os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/visible"), stem)
        ir_path = self._find_img(os.path.join(_root, "work_dir/test_aligned_ir"), stem)
        if not ir_path:
            ir_path = self._find_img(os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/infrared"), stem)
        raw_ir_path = self._find_img(os.path.join(_root, "dataset/test/AIC2026_PHASE_2_1000/infrared"), stem)

        # 1. RGB
        rgb_img = Image.open(rgb_path).convert('RGB')
        orig_w, orig_h = rgb_img.size
        rgb_np = np.array(rgb_img).astype(np.float32) / 255.0
        t_rgb = torch.from_numpy(rgb_np).permute(2, 0, 1)

        # 2. Aligned IR
        ir_img = Image.open(ir_path).convert('L')
        ir_np = np.array(ir_img).astype(np.float32) / 255.0
        if HAS_DESTRIPE:
            ir_np = stripe_non_uniformity_correction_1d(ir_np)
        t_ir = torch.from_numpy(ir_np).unsqueeze(0)

        # 3. Raw / LBG IR
        raw_ir_img = Image.open(raw_ir_path).convert('L')
        raw_ir_np = np.array(raw_ir_img).astype(np.float32) / 255.0
        t_raw_ir = torch.from_numpy(raw_ir_np).unsqueeze(0)

        # 4. Depth
        t_depth = self._read_depth(stem)

        # Resize all to match RGB
        H_orig, W_orig = t_rgb.shape[1], t_rgb.shape[2]
        if t_ir.shape[1:] != (H_orig, W_orig):
            t_ir = F.interpolate(t_ir.unsqueeze(0), size=(H_orig, W_orig), mode='bilinear', align_corners=False).squeeze(0)
        if t_raw_ir.shape[1:] != (H_orig, W_orig):
            t_raw_ir = F.interpolate(t_raw_ir.unsqueeze(0), size=(H_orig, W_orig), mode='bilinear', align_corners=False).squeeze(0)
        if t_depth.shape[1:] != (H_orig, W_orig):
            t_depth = F.interpolate(t_depth.unsqueeze(0), size=(H_orig, W_orig), mode='nearest').squeeze(0)

        t_6ch = torch.cat([t_rgb, t_ir, t_raw_ir, t_depth], dim=0)

        # Interpolate to target inference size
        t_6ch_resized = F.interpolate(t_6ch.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        t_6ch_norm = (t_6ch_resized - MEAN_6CH) / STD_6CH

        return stem, t_6ch_norm, orig_w, orig_h


def package_submission_dir(txt_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for txt in sorted(glob.glob(os.path.join(txt_dir, "*.txt"))):
            zf.write(txt, arcname=os.path.basename(txt))
    print(f"[+] Packaged submission zip: {zip_path}")


def main():
    cfg_path = os.path.join(_root, "vendor/DEIMv2/configs/deimv2/sovereign_v30_cssp_fast8ep.yml")
    ckpt_path = os.path.join(_root, "vendor/DEIMv2/outputs/sovereign_v30_cssp_fast8ep/best_stg2.pth")
    out_dir = os.path.join(_root, "work_dir/submissions/sovereign_v30_cssp")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Sovereign V30 (CSSP + ABC) Test Set Native Inference")
    print(f"Config:     {cfg_path}")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 60)

    cfg = YAMLConfig(cfg_path)
    model = cfg.model.cuda()
    postprocessor = cfg.postprocessor.cuda()

    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck.get('ema', {}).get('module', ck.get('model', ck))
    msg = model.load_state_dict(state, strict=False)
    print(f"[+] Loaded weights (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")

    model.eval()
    postprocessor.eval()

    ds = TestPhase2Dataset(target_size=(576, 1024))
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    raw_detections = {}
    print("\n[*] Running batched GPU forward pass...")
    t0 = time.time()

    with torch.no_grad():
        for stems, imgs, orig_ws, orig_hs in tqdm(dl):
            imgs = imgs.cuda()
            orig_sizes = torch.stack([orig_hs, orig_ws], dim=1).cuda()
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(imgs)
                results = postprocessor(outputs, orig_sizes)

            for i, stem in enumerate(stems):
                res = results[i]
                labels = res['labels'].cpu().numpy()
                scores = res['scores'].cpu().numpy()
                boxes = res['boxes'].cpu().numpy() # [x1, y1, x2, y2] absolute
                
                ow = orig_ws[i].item()
                oh = orig_hs[i].item()
                
                img_dets = []
                for cid, sc, (x1, y1, x2, y2) in zip(labels, scores, boxes):
                    cx = ((x1 + x2) / 2.0) / ow
                    cy = ((y1 + y2) / 2.0) / oh
                    w = (x2 - x1) / ow
                    h = (y2 - y1) / oh
                    cx = max(0.0, min(1.0, float(cx)))
                    cy = max(0.0, min(1.0, float(cy)))
                    w = max(0.0, min(1.0, float(w)))
                    h = max(0.0, min(1.0, float(h)))
                    img_dets.append((int(cid), cx, cy, w, h, float(sc)))
                    
                raw_detections[stem] = img_dets

    dur = time.time() - t0
    print(f"[+] Batched inference complete: 1000 images in {dur:.1f}s ({dur/1000*1000:.1f} ms/img)")

    # Save raw detections tensor
    raw_pt_path = os.path.join(out_dir, "raw_detections_v30_cssp.pt")
    torch.save(raw_detections, raw_pt_path)
    print(f"[+] Saved raw detection tensor: {raw_pt_path}")

    # Build Native Submission Package (Rare >= 0.45, Common >= 0.50, Cap=60)
    sub_dir = os.path.join(out_dir, "submission_v30_native_txt")
    os.makedirs(sub_dir, exist_ok=True)
    
    total_boxes = 0
    uav_boxes = 0
    cat_counts = Counter()

    for stem, dets in sorted(raw_detections.items()):
        filtered = [
            d for d in dets 
            if (d[5] >= 0.45 if d[0] in RARE_CLASSES else d[5] >= 0.50)
        ]
        filtered.sort(key=lambda x: -x[5])
        filtered = filtered[:60]
        
        txt_path = os.path.join(sub_dir, f"{stem}.txt")
        lines = []
        for cid, cx, cy, w, h, sc in filtered:
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {sc:.4f}\r\n")
            total_boxes += 1
            cat_counts[cid] += 1
            if cid == 10:
                uav_boxes += 1

        with open(txt_path, 'wb') as f:
            f.write("".join(lines).encode('ascii'))

    zip_path = os.path.join(out_dir, "submission_sovereign_v30_cssp_native.zip")
    package_submission_dir(sub_dir, zip_path)

    print("\n=== Sovereign V30 Native Submission Audit ===")
    print(f"Total Kept Boxes: {total_boxes} (avg {total_boxes/1000:.2f} boxes/image)")
    print(f"Total UAV Drones: {uav_boxes}")
    print("Category Breakdown:")
    for cid in range(12):
        print(f"  {CLASS_NAMES[cid]:<12}: {cat_counts[cid]:4d}")

    # Auto-import into workstation
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("workstation_importer", os.path.join(_root, "scripts/viz/import_submission_to_workstation.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "auto_import_submission"):
            mod.auto_import_submission(zip_path, "v30_cssp", "Sovereign V30 (CSSP+ABC 显著性觉醒)")
            print("[+] Successfully registered V30 to Web Workstation!")
    except Exception as e:
        print(f"[-] Workstation import notice: {e}")

if __name__ == "__main__":
    main()
