#!/usr/bin/env python3
"""
Generate Side-by-Side Visual Comparison Panels for Physics-Based Verification:
Case 1: 00000466 - Ground curb artifact killed by Law 1 (Aerodynamic Altitude)
Case 2: 00000256 - Static roadside pole killed by Law 2 (Newtonian Kinematics)
Case 3: 00000290 - Building canopy shadow killed by Law 3 (Perspective Scale & Symmetry)
Case 4: 00000511 - Genuine airborne drone confirmed by all 4 Physical Laws
"""

import os
import cv2
import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
VIS_DIR = os.path.join(ROOT, "dataset/test/AIC2026_PHASE_2_1000/visible")
OUT_DIR = os.path.join(ROOT, "outputs/figures/physics_verification_cases")
os.makedirs(OUT_DIR, exist_ok=True)

def find_img(stem):
    for ext in [".jpg", ".png"]:
        p = os.path.join(VIS_DIR, f"{stem}{ext}")
        if os.path.exists(p): return cv2.imread(p)
    return None

def create_panel(stem, left_title, left_box, left_txt, left_color,
                 right_title, right_box, right_txt, right_color, out_name):
    img = find_img(stem)
    if img is None:
        print(f"[!] Warning: Image {stem} not found")
        return
        
    H, W = img.shape[:2]
    
    # Left: Before Physics (Old Rescued with errors)
    left_img = img.copy()
    if left_box:
        x1, y1, x2, y2 = left_box
        cv2.rectangle(left_img, (int(x1*W), int(y1*H)), (int(x2*W), int(y2*H)), left_color, 3)
        cv2.putText(left_img, left_txt, (int(x1*W)-20, max(30, int(y1*H)-15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)
                    
    # Right: After Physics
    right_img = img.copy()
    if right_box:
        x1, y1, x2, y2 = right_box
        cv2.rectangle(right_img, (int(x1*W), int(y1*H)), (int(x2*W), int(y2*H)), right_color, 3)
        cv2.putText(right_img, right_txt, (int(x1*W)-20, max(30, int(y1*H)-15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)
    else:
        # Stamp "ELIMINATED BY PHYSICS" on right
        h_stamp, w_stamp = 80, 500
        stamp = np.zeros((h_stamp, w_stamp, 3), dtype=np.uint8)
        stamp[:] = (0, 0, 180) # deep red
        cv2.putText(stamp, "REJECTED BY PHYSICS (0 FP)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        # Overlay stamp on right image
        right_img[H//2-40:H//2+40, W//2-250:W//2+250] = stamp

    # Headers
    h_bar = 70
    bar_left = np.zeros((h_bar, W, 3), dtype=np.uint8)
    bar_left[:] = (20, 20, 30)
    cv2.putText(bar_left, left_title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
    
    bar_right = np.zeros((h_bar, W, 3), dtype=np.uint8)
    bar_right[:] = (20, 30, 20)
    cv2.putText(bar_right, right_title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    comp_l = np.vstack([bar_left, left_img])
    comp_r = np.vstack([bar_right, right_img])
    
    # Resize for side-by-side
    target_w = 1280
    target_h = int(target_w * comp_l.shape[0] / comp_l.shape[1])
    comp_l_r = cv2.resize(comp_l, (target_w, target_h))
    comp_r_r = cv2.resize(comp_r, (target_w, target_h))
    
    final = np.hstack([comp_l_r, comp_r_r])
    out_p = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out_p, final)
    print(f"[✔] Generated {out_p}")

# 1. Case 00000466 (Ground curb artifact)
create_panel(
    stem="00000466",
    left_title="BEFORE PHYSICS: Wrongly Rescued Ground Curb Artifact (Score 0.479)",
    left_box=[0.666-0.049/2, 0.988-0.026/2, 0.666+0.049/2, 0.988+0.026/2],
    left_txt="FP: 0.479 (Ground reflection touching y=1.0)",
    left_color=(0, 0, 255),
    right_title="AFTER PHYSICS LAW 1 (Altitude Law): Ground Touching (y=0.988 > 0.85) -> KILLED!",
    right_box=None,
    right_txt="",
    right_color=(0, 255, 0),
    out_name="case_00000466_ground_curb_killed.jpg"
)

# 2. Case 00000256 (Static roadside lamp post)
create_panel(
    stem="00000256",
    left_title="BEFORE PHYSICS: Wrongly Rescued Fixed Street Light / Pole (Score 0.501)",
    left_box=[0.572-0.047/2, 0.768-0.052/2, 0.572+0.047/2, 0.768+0.052/2],
    left_txt="FP: 0.501 (Stationary pole across 00000256/261/272)",
    left_color=(0, 0, 255),
    right_title="AFTER PHYSICS LAW 2 (Kinematics Law): Static Rigidity -> KILLED!",
    right_box=None,
    right_txt="",
    right_color=(0, 255, 0),
    out_name="case_00000256_static_pole_killed.jpg"
)

# 3. Case 00000290 (Building canopy shadow)
create_panel(
    stem="00000290",
    left_title="BEFORE PHYSICS: Wrongly Rescued Giant Building Shadow (Score 0.465)",
    left_box=[0.325-0.102/2, 0.578-0.065/2, 0.325+0.102/2, 0.578+0.065/2],
    left_txt="FP: 0.465 (Excessive Area: 0.67% of image)",
    left_color=(0, 0, 255),
    right_title="AFTER PHYSICS LAW 3 (Scale & Symmetry Law): Area 0.67% > 0.35% -> KILLED!",
    right_box=None,
    right_txt="",
    right_color=(0, 255, 0),
    out_name="case_00000290_building_shadow_killed.jpg"
)

# 4. Case 00000511 (Genuine airborne drone confirmed by all 4 physics laws)
create_panel(
    stem="00000511",
    left_title="RAW CAP60: Second Drone Missed Due to Single Query Split (0.278 < 0.45)",
    left_box=[0.6541-0.0455/2, 0.5541-0.0462/2, 0.6541+0.0455/2, 0.5541+0.0462/2],
    left_txt="Missed Drone #2 (Raw 0.278)",
    left_color=(0, 165, 255),
    right_title="PHYSICS VERIFIED: Drone Confirmed (Airborne y=0.55, Delta_T=+74.1, Symmetric 0.98)",
    right_box=[0.6541-0.0455/2, 0.5541-0.0462/2, 0.6541+0.0455/2, 0.5541+0.0462/2],
    right_txt="CONFIRMED DRONE #2: 0.428 (Passed 4 Physics Laws)",
    right_color=(0, 255, 0),
    out_name="case_00000511_physics_confirmed_drone.jpg"
)

print("[✔] All 4 physics verification comparison panels generated successfully!")
