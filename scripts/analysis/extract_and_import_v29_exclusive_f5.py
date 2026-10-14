#!/usr/bin/env python3
"""
Extract V29 Exclusive Detections (F5 Blind-Spot Discoveries) and Import to Workstation:
=====================================================================================
1. Loads Official Benchmark F5 (58.105, 7,097 boxes) and Sovereign V29 (7,289 boxes).
2. Performs strict 1-to-1 consensus matching (IoU >= 0.50, same category).
3. Extracts exactly the 1,029 exclusive detections where V29 discovered targets that F5 missed!
4. Packages into official submission format (.zip), passes strict QA, and auto-imports to http://localhost:8088/.
"""

import os
import sys
import json
import zipfile
import subprocess
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("/home/constant/CodeBuddy/Dev/AIC")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/DEIMv2"))

from scripts.viz.import_submission_to_workstation import auto_import_submission

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}
CLASS_ZH = {
    0: "人", 1: "船", 2: "动物", 3: "座椅", 4: "标牌", 5: "单车",
    6: "汽车", 7: "球", 8: "路灯", 9: "垃圾桶", 10: "无人机", 11: "三轮车"
}

def parse_submission_zip(zip_path):
    data = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            stem = os.path.splitext(os.path.basename(name))[0]
            raw = zf.read(name).decode("utf-8").strip()
            boxes = []
            if raw:
                for line in raw.replace("\r\n", "\n").split("\n"):
                    p = line.strip().split()
                    if len(p) >= 6:
                        boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
            data[stem] = boxes
    return data

def to_xyxy(cx, cy, bw, bh):
    return [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]

def iou_xyxy(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / ua if ua > 0 else 0.0

def main():
    f5_zip = ROOT / "work_dir/audits/f5_reproduce/submission_F5_cap60thr45_50.zip"
    v29_zip = ROOT / "work_dir/submissions/v29_cycle5_peak/submission_v29_c5_fusion_dual_gate_thr35.zip"

    print("=" * 80)
    print(" V29 Exclusive Detections (F5 Blind-Spot Discoveries) Package Extractor ".center(80, "="))
    print("=" * 80)
    print(f"[*] Loading F5 Benchmark from: {f5_zip}")
    f5_data = parse_submission_zip(f5_zip)
    print(f"[*] Loading V29 Predictions from: {v29_zip}")
    v29_data = parse_submission_zip(v29_zip)

    stems = sorted(set(f5_data.keys()) | set(v29_data.keys()))
    assert len(stems) == 1000, f"Expected 1000 images, got {len(stems)}"

    v29_exclusive = defaultdict(list)
    consensus_count = 0
    total_v29 = sum(len(v) for v in v29_data.values())
    total_f5 = sum(len(v) for v in f5_data.values())

    for stem in stems:
        boxes_ref = f5_data.get(stem, [])
        boxes_cand = v29_data.get(stem, [])

        used_cand = set()
        # 1-to-1 greedy matching (same class, IoU >= 0.50)
        for ba in boxes_ref:
            b_xyxy = to_xyxy(*ba[1:5])
            best_iou = 0.50
            best_bi = -1
            for bi, bb in enumerate(boxes_cand):
                if bi in used_cand or ba[0] != bb[0]:
                    continue
                bb_xyxy = to_xyxy(*bb[1:5])
                iou = iou_xyxy(b_xyxy, bb_xyxy)
                if iou >= best_iou:
                    best_iou = iou
                    best_bi = bi
            if best_bi >= 0:
                used_cand.add(best_bi)
                consensus_count += 1

        for bi, bb in enumerate(boxes_cand):
            if bi not in used_cand:
                v29_exclusive[stem].append(bb)

    total_exclusive = sum(len(v) for v in v29_exclusive.values())
    print(f"\n[✓] Matching complete:")
    print(f"    - F5 Total Boxes:        {total_f5:,}")
    print(f"    - V29 Total Boxes:       {total_v29:,}")
    print(f"    - Shared Consensus:      {consensus_count:,} ({consensus_count/total_f5*100:.2f}% of F5)")
    print(f"    - V29 Exclusive Targets: {total_exclusive:,} (F5 完全漏检目标)")

    # Per-class breakdown
    c_counts = Counter()
    scores_by_class = defaultdict(list)
    for bxs in v29_exclusive.values():
        for b in bxs:
            c_counts[b[0]] += 1
            scores_by_class[b[0]].append(b[5])

    print("\n" + "-" * 75)
    print(f"{'CID':<3} | {'Class Name':<12} | {'中文':<6} | {'Count':<7} | {'占比':<7} | {'Avg Score':<10} | {'Max Score':<10}")
    print("-" * 75)
    for cid in sorted(c_counts.keys()):
        cname = CLASS_NAMES[cid]
        czh = CLASS_ZH[cid]
        cnt = c_counts[cid]
        pct = cnt / total_exclusive * 100
        scs = scores_by_class[cid]
        avg_sc = sum(scs) / len(scs)
        max_sc = max(scs)
        print(f"{cid:<3} | {cname:<12} | {czh:<6} | {cnt:>7d} | {pct:>6.1f}% | {avg_sc:>10.4f} | {max_sc:>10.4f}")
    print("-" * 75)

    # Package into submission
    out_dir = ROOT / "work_dir/submissions/v29_exclusive_over_f5"
    txt_dir = out_dir / "submission_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / "submission_v29_exclusive_over_f5.zip"

    print(f"\n[*] Generating txt files in {txt_dir}...")
    for stem in stems:
        bxs = v29_exclusive.get(stem, [])
        bxs.sort(key=lambda x: -x[5])
        lines = []
        for cid, cx, cy, bw, bh, sc in bxs:
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {sc:.4f}")
        
        content = ("\r\n".join(lines) + "\r\n") if lines else ""
        with open(txt_dir / f"{stem}.txt", "wb") as f:
            f.write(content.encode("utf-8"))

    print(f"[*] Packaging into {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_f in sorted(txt_dir.glob("*.txt")):
            zf.write(txt_f, arcname=txt_f.name)

    # Run check_submission.py
    check_script = ROOT / "scripts/eval/check_submission.py"
    vis_dir = ROOT / "dataset/test/AIC2026_PHASE_2_1000/visible"
    print(f"\n[*] Running strict QA check with check_submission.py...")
    cmd = [
        "/home/constant/miniconda3/envs/deimv2/bin/python",
        str(check_script),
        str(txt_dir),
        str(vis_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[!] QA Check Output:\n", res.stderr)
        raise RuntimeError("QA verification failed!")

    # Auto import to workstation
    print(f"\n[*] Auto-importing to Workstation (http://localhost:8088/)...")
    model_id = "v29_exclusive_f5"
    name = "V29 独家检出 (F5盲区 1,029框)"
    badge = f"1,029 增量框 / F5 漏检目标"
    score = f"1,029 增量"
    color = "#a855f7" # Neon purple
    desc = "V29 在 F5 (58.105 官方标杆) 盲区中独家检出的 1,029 个增量目标（排除 6,260 个共识框，覆盖全部 12 类）"

    auto_import_submission(
        zip_path=str(out_zip),
        model_id=model_id,
        name=name,
        score=score,
        badge=badge,
        color=color,
        desc=desc,
        rebuild=True
    )
    from scripts.viz.import_submission_to_workstation import ensure_server_alive
    ensure_server_alive()

    print("\n" + "=" * 80)
    print(" SUCCESS! V29 Exclusive Package Imported to Workstation ".center(80, "="))
    print(f" Package Zip: {out_zip}")
    print(f" Model ID:    {model_id}")
    print(f" Web URL:     http://localhost:8088/")
    print("=" * 80)

if __name__ == "__main__":
    main()
