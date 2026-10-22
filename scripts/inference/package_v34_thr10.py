#!/usr/bin/env python3
"""
Package Sovereign V34 Submission at Threshold 0.10 (With NMS and Raw)
====================================================================
Generates:
1. submission_sovereign_v34_thr10.zip: Score >= 0.10 with Class-wise NMS (IoU=0.60)
2. submission_sovereign_v34_thr10_raw.zip: Score >= 0.10 Raw (no NMS)

Both strictly adhere to competition specifications:
- Exact 1000 files corresponding to test images
- Format: <class_id> <cx> <cy> <w> <h> <confidence>
- CRLF line endings
- Descending confidence order
- <= 100 boxes per image
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torchvision

ROOT_DIR = Path("/home/constant/CodeBuddy/Dev/AIC")
sys.path.insert(0, str(ROOT_DIR))

from scripts.viz.import_submission_to_workstation import auto_import_submission

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

def package_v34_thr10():
    raw_path = ROOT_DIR / "work_dir/submissions/sovereign_v34_clean_from_scratch/raw_detections_v34_production.pt"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found!")
        sys.exit(1)

    print(f"[Loading] Raw detections from {raw_path}...")
    raw_detections = torch.load(raw_path, map_location="cpu")
    print(f"[Loaded] {len(raw_detections)} images.")

    # Target 1: With NMS 0.60
    zip_nms = ROOT_DIR / "work_dir/submissions/sovereign_v34_clean_from_scratch/submission_sovereign_v34_thr10.zip"
    # Target 2: Raw >= 0.10
    zip_raw = ROOT_DIR / "work_dir/submissions/sovereign_v34_clean_from_scratch/submission_sovereign_v34_thr10_raw.zip"

    configs = [
        ("NMS 0.60 (Clean)", zip_nms, True),
        ("Raw (No NMS)", zip_raw, False),
    ]

    for label, zip_p, apply_nms in configs:
        print(f"\n=======================================================")
        print(f"Packaging {label} -> {zip_p.name}")
        print(f"=======================================================")
        total_boxes = 0
        scores = []
        cat_counts = Counter()

        with zipfile.ZipFile(zip_p, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for stem in sorted(raw_detections.keys()):
                dets = raw_detections[stem]
                filtered = [b for b in dets if b[5] >= 0.10 and 0 <= b[0] < 12]
                filtered = sorted(filtered, key=lambda x: x[5], reverse=True)

                if apply_nms and len(filtered) > 1:
                    labels_t = torch.tensor([b[0] for b in filtered], dtype=torch.long)
                    scores_t = torch.tensor([b[5] for b in filtered], dtype=torch.float32)
                    xyxy = []
                    for b in filtered:
                        cx, cy, bw, bh = b[1:5]
                        xyxy.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
                    boxes_t = torch.tensor(xyxy, dtype=torch.float32)
                    keep = torchvision.ops.batched_nms(boxes_t, scores_t, labels_t, iou_threshold=0.60)
                    filtered = [filtered[i] for i in keep.tolist()]

                filtered = filtered[:100]

                lines = []
                for b in filtered:
                    cid, cx, cy, bw, bh, sc = b
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {sc:.4f}")
                    total_boxes += 1
                    cat_counts[cid] += 1
                    scores.append(sc)
                txt_content = "\r\n".join(lines) + ("\r\n" if lines else "")
                zf.writestr(f"{stem}.txt", txt_content)

        mean_sc = float(np.mean(scores)) if scores else 0.0
        print(f"Package Summary: {zip_p.name}")
        print(f"  Total Boxes: {total_boxes}")
        print(f"  Mean Score:  {mean_sc:.4f}")
        print(f"  Score Min/Max: {min(scores):.4f} / {max(scores):.4f}")
        print(f"  Class Distribution:")
        for cid in range(12):
            print(f"    {cid:2d} ({CLASS_NAMES[cid]:12s}): {cat_counts[cid]:5d}")

    # Validate against competition rules
    print("\n[Validation] Validating packages with check_submission.py...")
    for label, zip_p, _ in configs:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_p, 'r') as zf:
                zf.extractall(tmpdir)
            cmd = f"python3 {ROOT_DIR}/scripts/eval/check_submission.py {tmpdir} {ROOT_DIR}/dataset/test/AIC2026_PHASE_2_1000/visible"
            ret = os.system(cmd)
            if ret != 0:
                print(f"ERROR: check_submission failed for {zip_p}!")
                sys.exit(1)
            print(f"PASS: {zip_p.name} validated successfully!")

    # Register into Web Workstation
    print("\n[Workstation] Registering packages into Master Unified Workstation...")
    auto_import_submission(
        str(zip_nms),
        model_id="v34_thr10",
        name="Sovereign V34 (τ=0.10 + NMS 0.60)",
        score=f"0.485 均置信",
        badge="23,558 框 / τ=0.10 极限召回",
        color="#14b8a6",
        desc="Sovereign V34 纯血从零训练在 τ=0.10 超低门槛截断并执行 Class-wise NMS (IoU=0.60) 的提交包，总计 23,558 框"
    )

    auto_import_submission(
        str(zip_raw),
        model_id="v34_thr10_raw",
        name="Sovereign V34 (τ=0.10 原生无 NMS)",
        score=f"0.432 均置信",
        badge="30,969 框 / τ=0.10 原生全集",
        color="#0d9488",
        desc="Sovereign V34 纯血从零训练在 τ=0.10 原生置信度截断 (无额外 NMS) 的全量提交包，总计 30,969 框"
    )

    print("\n[Done] All packages ready and registered!")

if __name__ == "__main__":
    package_v34_thr10()
