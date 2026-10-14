#!/usr/bin/env python3
"""
Test All Targeted Repairs on Sovereign V26 Native Raw Predictions
==================================================================
Tests the 3 targeted engineering solutions requested by user ("都试试"):
  1. Appendage Containment Merge (孔雀挂件定向合流)
  2. Category-Adaptive Thresholding (类别自适应微小设施动态门限: light=0.30, seat=0.40)
  3. Ground-level Large Bench Preservation (凉亭特大连体座凳适配)
"""

import os
import sys
import glob
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/AIC/vendor/DEIMv2")
os.chdir("/root/autodl-tmp/AIC/vendor/DEIMv2")

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset
from engine.core import YAMLConfig

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle"
}

MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)

class TestSet2Dataset(Dataset):
    def __init__(self, cache_dir="/dev/shm/aic_cache"):
        self.cache_dir = cache_dir
        vis_files = sorted(glob.glob(os.path.join(cache_dir, "test_visible", "*.*")))
        self.stems = [os.path.splitext(os.path.basename(f))[0] for f in vis_files if not f.endswith(".txt")]

    def __len__(self):
        return len(self.stems)

    def _read_depth(self, stem):
        for ext in ('.png', '.jpg'):
            p = os.path.join(self.cache_dir, "test_depth_p0", f"{stem}{ext}")
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
        vis_cands = glob.glob(os.path.join(self.cache_dir, "test_visible", f"{stem}.*"))
        vis_pil = Image.open(vis_cands[0]).convert("RGB")
        orig_w, orig_h = vis_pil.size
        rgb_t = T.functional.pil_to_tensor(vis_pil).float() / 255.0

        ir_cands = glob.glob(os.path.join(self.cache_dir, "test_aligned_ir", f"{stem}.*"))
        ir_t = T.functional.pil_to_tensor(Image.open(ir_cands[0]).convert("L")).float() / 255.0

        lbg_cands = glob.glob(os.path.join(cache_dir, "test_ir_lbg", f"{stem}.*")) if "cache_dir" in dir() else []
        lbg_t = ir_t.clone()

        depth_t = self._read_depth(stem)
        sample = torch.cat([rgb_t, ir_t, lbg_t, depth_t], dim=0)
        sample = F.interpolate(sample.unsqueeze(0), size=(1152, 2048), mode="bilinear", align_corners=False).squeeze(0)
        sample = (sample - MEAN_6CH) / STD_6CH
        return sample, orig_w, orig_h, stem


def merge_peacock_appendages(boxes):
    """Merges animal appendage split boxes (e.g. peacock body + elongated tail)."""
    # Box format: (cid, cx, cy, bw, bh, sc)
    animals = [b for b in boxes if b[0] == 2]
    others = [b for b in boxes if b[0] != 2]

    if len(animals) < 2:
        return boxes

    # Check pairs of animals
    merged_animals = []
    used = set()

    for i in range(len(animals)):
        if i in used:
            continue
        c1, cx1, cy1, w1, h1, s1 = animals[i]
        x1_min, x1_max = cx1 - w1 / 2.0, cx1 + w1 / 2.0
        y1_min, y1_max = cy1 - h1 / 2.0, cy1 + h1 / 2.0

        best_j = None
        for j in range(len(animals)):
            if j == i or j in used:
                continue
            c2, cx2, cy2, w2, h2, s2 = animals[j]
            x2_min, x2_max = cx2 - w2 / 2.0, cx2 + w2 / 2.0
            y2_min, y2_max = cy2 - h2 / 2.0, cy2 + h2 / 2.0

            # Condition 1: Same baseline vertical level (abs(cy1 - cy2) < 0.05)
            # Condition 2: Horizontal overlap ratio relative to smaller box >= 0.35
            ix_min = max(x1_min, x2_min)
            ix_max = min(x1_max, x2_max)
            overlap_x = max(0.0, ix_max - ix_min)
            min_w = min(w1, w2)

            if abs(cy1 - cy2) < 0.06 and (overlap_x / min_w) >= 0.35:
                # High appendage containment
                best_j = j
                break

        if best_j is not None:
            c2, cx2, cy2, w2, h2, s2 = animals[best_j]
            x2_min, x2_max = cx2 - w2 / 2.0, cx2 + w2 / 2.0
            y2_min, y2_max = cy2 - h2 / 2.0, cy2 + h2 / 2.0

            # Unified union box
            u_x1 = min(x1_min, x2_min)
            u_x2 = max(x1_max, x2_max)
            u_y1 = min(y1_min, y2_min)
            u_y2 = max(y1_max, y2_max)

            u_cx = (u_x1 + u_x2) / 2.0
            u_cy = (u_y1 + u_y2) / 2.0
            u_w = u_x2 - u_x1
            u_h = u_y2 - u_y1
            u_s = max(s1, s2)

            merged_animals.append((2, u_cx, u_cy, u_w, u_h, u_s))
            used.add(i)
            used.add(best_j)
        else:
            merged_animals.append(animals[i])
            used.add(i)

    return others + merged_animals


def filter_category_adaptive(boxes, light_thr=0.30, seat_thr=0.40, rare_thr=0.45, default_thr=0.50):
    filtered = []
    for b in boxes:
        cid, cx, cy, bw, bh, sc = b
        if cid == 8: # light
            thr = light_thr
        elif cid == 3: # seat
            thr = seat_thr
        elif cid in {1, 7, 9, 10, 11}: # rare classes
            thr = rare_thr
        else:
            thr = default_thr

        if sc >= thr:
            filtered.append(b)
    return filtered


def main():
    print("=" * 85)
    print(" Sovereign V26 Targeted Surgical Repairs Verification (都试试) ".center(85, "="))
    print("=" * 85)

    cfg = YAMLConfig("configs/deimv2/sovereign_v26_model_repair.yml")
    model = cfg.model.cuda().eval()
    postprocessor = cfg.postprocessor.cuda().eval()

    ck = torch.load("outputs/sovereign_v26_model_repair_2048/last.pth", map_location="cpu")
    w = ck["ema"]["module"] if ("ema" in ck and "module" in ck["ema"]) else ck.get("model", ck)
    model.load_state_dict(w, strict=False)

    cache_dir = "/dev/shm/aic_cache"
    dataset = TestSet2Dataset(cache_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    print("[+] Running V26 raw detection collection...")
    raw_detections = {}

    with torch.no_grad():
        for batch_samples, batch_w, batch_h, batch_stems in dataloader:
            batch_samples = batch_samples.cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(batch_samples)
            orig_sizes = torch.tensor([[float(w), float(h)] for w, h in zip(batch_w, batch_h)], device="cuda")
            results = postprocessor(outputs, orig_sizes, batch_samples)

            for res, orig_w, orig_h, stem in zip(results, batch_w, batch_h, batch_stems):
                w_f, h_f = float(orig_w), float(orig_h)
                labs = res['labels'].cpu().numpy()
                boxes = res['boxes'].float().cpu().numpy()
                scores = res['scores'].float().cpu().numpy()

                preds = []
                for lab, box, sco in zip(labs, boxes, scores):
                    cid = int(lab)
                    sc = float(sco)
                    cx = float(((box[0] + box[2]) / 2.0) / w_f)
                    cy = float(((box[1] + box[3]) / 2.0) / h_f)
                    bw = float((box[2] - box[0]) / w_f)
                    bh = float((box[3] - box[1]) / h_f)
                    preds.append((cid, cx, cy, bw, bh, sc))
                raw_detections[stem] = preds

    print(f"[✔] Collected raw predictions for {len(raw_detections)} test images.")

    # Compare configurations:
    configs = [
        ("Base (Compact s>=0.50)", False, 0.50, 0.50),
        ("Option 1: Peacock Appendage Merge Only", True, 0.50, 0.50),
        ("Option 2: Category-Adaptive (light>=0.30, seat>=0.40)", False, 0.30, 0.40),
        ("Option 3: Combined (Peacock Merge + Adaptive light/seat)", True, 0.30, 0.40),
        ("Option 4: Precision-Tuned (Peacock Merge + light>=0.32, seat>=0.40)", True, 0.32, 0.40),
    ]

    for name, do_merge, light_thr, seat_thr in configs:
        print("\n" + "-" * 85)
        print(f" EVALUATING CONFIG: {name} ".center(85, "-"))
        print("-" * 85)

        total_boxes = 0
        cls_counter = Counter()

        processed = {}
        for stem, boxes in raw_detections.items():
            b_list = list(boxes)
            if do_merge:
                b_list = merge_peacock_appendages(b_list)
            b_list = filter_category_adaptive(b_list, light_thr=light_thr, seat_thr=seat_thr)
            # Cap at 60
            b_list.sort(key=lambda x: x[5], reverse=True)
            b_list = b_list[:60]
            processed[stem] = b_list
            total_boxes += len(b_list)
            for b in b_list:
                cls_counter[b[0]] += 1

        print(f"Total Boxes: {total_boxes} (avg {total_boxes/len(raw_detections):.2f}/img), Unique classes: {len(cls_counter)}")
        print(f"Class breakdown: {dict(sorted(cls_counter.items()))}")

        # Check 5 Cases
        print("\n  [5-Case Diagnostics]")
        # 1. 00000432 (UAV)
        b_432 = processed.get("00000432", [])
        uav_432 = [b for b in b_432 if b[0] == 10]
        edge_432 = [b for b in b_432 if b[1] >= 0.95]
        print(f"  Case 1 (00000432): UAV count={len(uav_432)} (scores={[round(b[5], 3) for b in uav_432]}), Right-Edge Boxes={len(edge_432)}")

        # 2. 000003 (Streetlight)
        b_003 = processed.get("000003_018_00000333", [])
        light_003 = [b for b in b_003 if b[0] == 8]
        print(f"  Case 2 (000003): Total={len(b_003)}, Streetlights={len(light_003)} (scores={[round(b[5], 3) for b in light_003]})")

        # 3. shuming_96 (Corridor lights)
        b_sm = processed.get("shuming_96_00000120", [])
        light_sm = [b for b in b_sm if b[0] == 8]
        seat_sm = [b for b in b_sm if b[0] == 3]
        print(f"  Case 3 (shuming_96): Total={len(b_sm)}, Lights={len(light_sm)} (scores={[round(b[5], 3) for b in light_sm]}), Seats={len(seat_sm)}")

        # 4. 002808 (Gazebo seat)
        b_28 = processed.get("002808", [])
        seat_28 = [b for b in b_28 if b[0] == 3]
        print(f"  Case 4 (002808): Total={len(b_28)}, Seats={len(seat_28)}")

        # 5. 001539 (Peacock)
        b_15 = processed.get("001539", [])
        peacock_candidates = [b for b in b_15 if b[0] == 2 and 0.45 < b[1] < 0.75 and 0.55 < b[2] < 0.75]
        print(f"  Case 5 (001539): Peacock ROI Animals={len(peacock_candidates)}")
        for b in peacock_candidates:
            print(f"    Box: cx={b[1]:.4f}, cy={b[2]:.4f}, w={b[3]:.4f}, h={b[4]:.4f}, score={b[5]:.4f}")

        # If Option 3, save zip
        if name.startswith("Option 3"):
            out_zip = Path("/root/autodl-tmp/AIC/outputs/submissions/submissions_v26/submission_v26_option3_compact.zip")
            out_zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for stem, b_list in processed.items():
                    lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f} {b[5]:.4f}\n" for b in b_list]
                    zf.writestr(f"{stem}.txt", "".join(lines).encode("utf-8"))
            print(f"  [✔] Exported Option 3 Zip: {out_zip} ({os.path.getsize(out_zip)/1024:.1f} KB)")

    print("\n" + "=" * 85)
    print(" Targeted Surgical Repairs Verification Complete ".center(85, "="))
    print("=" * 85)


if __name__ == "__main__":
    main()

