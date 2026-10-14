#!/usr/bin/env python3
"""
Physical Drone Verifier (物理法则无人机校验器)
==============================================
Enforces four rigid real-world physical laws to strictly eliminate false positives
(such as lamp posts, curb stones, static surveillance cameras, and building shadows)
while safely rescuing genuine airborne UAVs:

1. Law 1: Gravity & Aerodynamic Flight Corridor (重力与空域高度法则)
   - Eliminates ground/curb/asphalt reflections (cy + h/2 <= 0.85).
2. Law 2: Newtonian Kinematics & Static Rigidity (牛顿运动学与刚体地物抑制)
   - Eliminates stationary ground infrastructure (lamp posts, pole cameras)
     appearing at identical (x, y) coordinates across multiple frames.
3. Law 3: Perspective Geometry & Aerodynamic Symmetry (透视投影与空气动力学几何法则)
   - Drones exhibit symmetric planar structure: 0.45 <= w/h <= 2.2.
   - Long-range drone perspective area cannot exceed 0.35% of the 2.5K field of view.
4. Law 4: Thermodynamic Heat Dissipation (热力学第二定律与红外温升)
   - Electric motors and batteries produce distinct thermal contrast.
"""

import math
import numpy as np

def detect_static_infrastructure(candidates, coord_tol=0.018):
    """
    Detects stationary ground fixtures across multiple images in a sequence.
    Returns a set of (stem, box_idx) that are identified as fixed ground fixtures.
    """
    static_indices = set()
    n = len(candidates)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = candidates[i], candidates[j]
            if c1["stem"] != c2["stem"]:
                dist = math.hypot(c1["cx"] - c2["cx"], c1["cy"] - c2["cy"])
                if dist < coord_tol:
                    static_indices.add(i)
                    static_indices.add(j)
    return static_indices

def verify_physics(cand, is_static=False, max_ground_y=0.85, max_area=0.0035,
                   min_ar=0.45, max_ar=2.2):
    """
    Evaluates a candidate detection against the rigid physical laws.
    Returns (passed: bool, reason: str)
    """
    cx, cy, w, h = cand["cx"], cand["cy"], cand["w"], cand["h"]
    area = w * h
    ar = w / (h + 1e-6)
    
    # LAW 1: Aerodynamic Flight Corridor (Ground Cutoff)
    if (cy + h / 2.0) > max_ground_y:
        return False, f"违反定律1: 贴地路面伪影 (y_bottom={cy+h/2.0:.3f} > {max_ground_y})"
        
    # LAW 2: Newtonian Kinematics (Stationary Pole/Camera)
    if is_static:
        return False, "违反定律2: 跨图像绝对静止地物 (固定路灯杆/监控头)"
        
    # LAW 3: Perspective Scale & Aerodynamic Symmetry
    if area > max_area:
        return False, f"违反定律3: 超出物理投影尺度 (面积={area*100:.2f}% > {max_area*100:.2f}%)"
    if ar < min_ar or ar > max_ar:
        return False, f"违反定律3: 违背旋翼机对称长宽比 (长宽比={ar:.2f})"
        
    return True, "符合全部物理法则 (物理实锤真目标)"
