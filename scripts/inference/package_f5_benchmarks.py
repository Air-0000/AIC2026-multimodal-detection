#!/usr/bin/env python3
"""
Package F5 Benchmark Submissions from Official Raw Predictions
"""

import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

RARE_CLASSES = {1, 7, 9, 10, 11} # boat, ball, garbage can, uav, tricycle

def package_filtered(input_zip, out_zip, thr_rare, thr_common, max_per_image=60):
    total_boxes = 0
    class_counts = Counter()
    with zipfile.ZipFile(input_zip, 'r') as zin, zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zout:
        members = sorted(zin.namelist())
        assert len(members) == 1000, f"Expected 1000 members, got {len(members)}"
        for name in members:
            if not name.endswith('.txt'):
                continue
            raw = zin.read(name).decode('utf-8').strip()
            lines = []
            if raw:
                cands = []
                for line in raw.replace('\r\n', '\n').split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    cid = int(parts[0])
                    sc = float(parts[5])
                    thr = thr_rare if cid in RARE_CLASSES else thr_common
                    if sc >= thr:
                        cands.append((cid, sc, line.strip()))
                rare = sorted([c for c in cands if c[0] in RARE_CLASSES], key=lambda x: -x[1])
                common = sorted([c for c in cands if c[0] not in RARE_CLASSES], key=lambda x: -x[1])
                preferred = (rare + common)[:max_per_image]
                preferred.sort(key=lambda x: -x[1])
                lines = [c[2] for c in preferred]
                for c in preferred:
                    class_counts[c[0]] += 1
                total_boxes += len(lines)
            content = ('\r\n'.join(lines) + '\r\n') if lines else ''
            zout.writestr(Path(name).name, content.encode('utf-8'))
    print(f"[+] Packaged {Path(out_zip).name}: {total_boxes:,} boxes ({total_boxes/1000:.2f}/img), Size: {os.path.getsize(out_zip)/1024:.1f} KB")
    return total_boxes, class_counts

def package_raw(input_zip, out_zip, top_k=100):
    total_boxes = 0
    with zipfile.ZipFile(input_zip, 'r') as zin, zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zout:
        members = sorted(zin.namelist())
        assert len(members) == 1000, f"Expected 1000 members, got {len(members)}"
        for name in members:
            if not name.endswith('.txt'):
                continue
            raw = zin.read(name).decode('utf-8').strip()
            lines = []
            if raw:
                cands = []
                for line in raw.replace('\r\n', '\n').split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split()
                    sc = float(parts[5])
                    cands.append((sc, line.strip()))
                cands.sort(key=lambda x: -x[0])
                lines = [c[1] for c in cands[:top_k]]
                total_boxes += len(lines)
            content = ('\r\n'.join(lines) + '\r\n') if lines else ''
            zout.writestr(Path(name).name, content.encode('utf-8'))
    print(f"[+] Packaged {Path(out_zip).name}: {total_boxes:,} boxes ({total_boxes/1000:.2f}/img), Size: {os.path.getsize(out_zip)/1024:.1f} KB")
    return total_boxes

def main():
    if len(sys.argv) < 3:
        print("Usage: package_f5_benchmarks.py <input_raw_zip> <output_dir>")
        sys.exit(1)
    input_zip = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" Packaging Official F5 Submissions ".center(70, "="))
    print("=" * 70)

    # 1. Official 58.105 Peak
    zip_45_50 = out_dir / "submission_F5_cap60thr45_50.zip"
    tot_45_50, dist = package_filtered(input_zip, zip_45_50, 0.45, 0.50, 60)

    # 2. Tier-2 58.055
    zip_50_55 = out_dir / "submission_F5_cap60thr50_55.zip"
    package_filtered(input_zip, zip_50_55, 0.50, 0.55, 60)

    # 3. Tier-3 57.707
    zip_35_40 = out_dir / "submission_F5_cap60thr35_40.zip"
    package_filtered(input_zip, zip_35_40, 0.35, 0.40, 60)

    # 4. Raw Top-100 (53.100)
    zip_raw = out_dir / "submission_F5_raw_top100.zip"
    package_raw(input_zip, zip_raw, 100)

    print("\n[+] 58.105 Peak Class Distribution:")
    from predict_official import CLASS_NAMES
    for cid in range(12):
        print(f"  {cid:2d} ({CLASS_NAMES[cid]:<12}): {dist[cid]:5d} ({dist[cid]/tot_45_50*100:5.2f}%)")

if __name__ == "__main__":
    main()
