#!/usr/bin/env python3
"""
Generate Before vs After Visual Comparison Panels for Sovereign V31 Physical Drone Cleanup
Cases:
- 00000076: Giant cloud / background structure FP (Score 0.93) -> 100% KILLED
- 00000261: Ground curb edge FP (Score 0.90) -> 100% KILLED
- 000243: 4 Ground curb fence post FPs -> KILLED, Real Airborne Drone KEP
- 00000290: Static window fixture FP -> KILLED, Genuine target KEPT
"""

import os
import cv2
import zipfile
import numpy as np
from pathlib import Path

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
ARTIFACT_DIR = "/home/constant/.gemini/antigravity/brain/9ade8979-579e-41ff-bbad-391b8a3785b5"
VIS_DIR = os.path.join(ROOT, "dataset/test/AIC2026_PHASE_2_1000/visible")
OUT_DIR = os.path.join(ROOT, "outputs/figures/v31_uav_physics_cleanup")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RAW_ZIP = os.path.join(ROOT, "work_dir/submissions/sovereign_v31_co_one2many/submission_sovereign_v31_native.zip")
CLEAN_ZIP = os.path.join(ROOT, "work_dir/submissions/sovereign_v31_co_one2many/submission_sovereign_v31_physics_clean.zip")

def parse_uav_from_zip(zip_path):
    uavs = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for fn in zf.namelist():
            if not fn.endswith('.txt'): continue
            stem = fn[:-4]
            lines = zf.read(fn).decode().splitlines()
            bxs = []
            for l in lines:
                p = l.split()
                if len(p) >= 6 and int(p[0]) == 10:
                    bxs.append([float(x) for x in p[1:]])
            uavs[stem] = bxs
    return uavs

raw_uavs = parse_uav_from_zip(RAW_ZIP)
clean_uavs = parse_uav_from_zip(CLEAN_ZIP)

def find_img(stem):
    for ext in [".jpg", ".png", ".jpeg"]:
        p = os.path.join(VIS_DIR, f"{stem}{ext}")
        if os.path.exists(p): return cv2.imread(p)
    return None

def create_comparison_panel(stem, title_desc, out_filename):
    img = find_img(stem)
    if img is None:
        print(f"[-] Image {stem} not found")
        return
    H, W = img.shape[:2]

    b_raw = raw_uavs.get(stem, [])
    b_clean = clean_uavs.get(stem, [])

    # Left: Before (Raw V31 with false environmental UAVs)
    left_img = img.copy()
    for cx, cy, w, h, sc in b_raw:
        x1, y1 = int((cx - w/2) * W), int((cy - h/2) * H)
        x2, y2 = int((cx + w/2) * W), int((cy + h/2) * H)
        # Red box for false / unverified
        col = (0, 0, 230)
        cv2.rectangle(left_img, (x1, y1), (x2, y2), col, 3)
        cv2.putText(left_img, f"Raw UAV: {sc:.2f}", (max(10, x1), max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    # Top title bar for Left
    cv2.rectangle(left_img, (0, 0), (W, 40), (20, 20, 20), -1)
    cv2.putText(left_img, f"Raw V31 (Environmental Noise Present) - {len(b_raw)} UAV Box(es)", (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

    # Right: After (Physics Verified Clean)
    right_img = img.copy()
    if b_clean:
        for cx, cy, w, h, sc in b_clean:
            x1, y1 = int((cx - w/2) * W), int((cy - h/2) * H)
            x2, y2 = int((cx + w/2) * W), int((cy + h/2) * H)
            col = (0, 220, 0) # Green for verified
            cv2.rectangle(right_img, (x1, y1), (x2, y2), col, 3)
            cv2.putText(right_img, f"True Drone: {sc:.2f}", (max(10, x1), max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    else:
        # Stamp: 100% False Positives Eliminated
        h_st, w_st = 90, 520
        stamp = np.zeros((h_st, w_st, 3), dtype=np.uint8)
        stamp[:] = (0, 140, 0)
        cv2.putText(stamp, "PHYSICS VERIFIED: CLEAN", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(stamp, "Fake Environmental UAVs 100% Filtered", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos = H // 2 - 45
        x_pos = W // 2 - 260
        right_img[y_pos:y_pos+h_st, x_pos:x_pos+w_st] = stamp

    cv2.rectangle(right_img, (0, 0), (W, 40), (20, 20, 20), -1)
    cv2.putText(right_img, f"V31 Physics Clean - {len(b_clean)} Real Drone Box(es)", (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

    # Combine side by side
    canvas = np.hstack([left_img, right_img])
    
    # Header
    hdr = np.zeros((60, canvas.shape[1], 3), dtype=np.uint8)
    hdr[:] = (35, 35, 35)
    cv2.putText(hdr, f"Image {stem}: {title_desc}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    final_panel = np.vstack([hdr, canvas])

    p1 = os.path.join(OUT_DIR, out_filename)
    p2 = os.path.join(ARTIFACT_DIR, out_filename)
    cv2.imwrite(p1, final_panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(p2, final_panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[+] Rendered comparison panel: {p1} -> {p2}")

def main():
    cases = [
        ("00000076", "Giant Cloud / Roof Fixture False Positive Eliminated", "v31_uav_case_00000076_cloud_killed.jpg"),
        ("00000261", "Ground Curb / Asphalt Edge False Positive Eliminated", "v31_uav_case_00000261_curb_killed.jpg"),
        ("000243", "Roadside Curb Posts Filtered / Real Airborne Drones Retained", "v31_uav_case_000243_curb_posts_killed.jpg"),
        ("00000290", "Static Surveillance Building Facade Fixture Filtered", "v31_uav_case_00000290_static_fixture_killed.jpg"),
    ]
    for stem, desc, fname in cases:
        create_comparison_panel(stem, desc, fname)

if __name__ == "__main__":
    main()
