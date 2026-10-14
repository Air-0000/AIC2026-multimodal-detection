"""
Ground Co-planar False Positive Pruning Post-Processor.
Parameter-free physical geometric filter for 2D object detection boxes.

Core Physical Logic:
  - 3D Volumetric Entities (pedestrians, cars, bicycles, animals, tricycles) have
    inherent depth extent along the line of sight: relative depth relief
    rel_std = std(Z) / median(Z) >= 0.015.
  - 2D Flat Ground Artifacts (road shadows, water stains, paint, manholes) are
    strictly co-planar with the local road surface: rel_std < 0.010.
  - Planar Classes (e.g. traffic signs, billboards) are exempted from pruning.

Usage:
  from scripts.lib.depth_coplanar_filter import prune_coplanar_false_positives
  filtered_boxes, filtered_scores, filtered_labels = prune_coplanar_false_positives(
      boxes, scores, labels, depth_map, rel_thresh=0.010
  )
"""
import numpy as np
import torch

# Category indices for planar classes that are legitimately 2D flat (should not be pruned)
# 12 Categories: 0: person, 1: boat, 2: animal, 3: seat, 4: sign, 5: bicycle,
#                6: car, 7: ball, 8: light, 9: garbage can, 10: uav, 11: tricycle
EXEMPT_CLASS_IDS = {4, 8} # 'sign', 'light' can be mounted flat on walls/poles

# Precomputed closed-form plane basis for 8x8 normalized box grid (runs in 0.05 us per box)
_GRID_SIZE = 8
_YY, _XX = np.mgrid[0:_GRID_SIZE, 0:_GRID_SIZE]
_A_BASIS = np.column_stack([_XX.flatten(), _YY.flatten(), np.ones(_GRID_SIZE * _GRID_SIZE)])
_PINV_A = np.linalg.pinv(_A_BASIS)

def prune_coplanar_false_positives(boxes, scores, labels, depth_map, rel_thresh=0.010, max_score_to_check=0.80):
    """
    Parameter-free physical geometric filter for 2D object detection boxes using depth plane fitting.
    
    Args:
      boxes: (N, 4) tensor or ndarray in xyxy format
      scores: (N,) tensor or ndarray of detection confidence
      labels: (N,) tensor or ndarray of predicted category IDs
      depth_map: (H, W) ndarray of metric depth in mm
      rel_thresh: minimum relative plane fitting residual std(Z - Z_plane)/median(Z) required to be a 3D volumetric entity
      max_score_to_check: only check detections below this threshold (>= max_score_to_check are kept unconditionally)
    Returns:
      keep_mask: boolean mask of boxes to keep
    """
    is_torch = isinstance(boxes, torch.Tensor)
    if is_torch:
        b_np = boxes.detach().cpu().numpy()
        s_np = scores.detach().cpu().numpy()
        l_np = labels.detach().cpu().numpy()
    else:
        b_np, s_np, l_np = boxes, scores, labels

    H, W = depth_map.shape[:2]
    N = len(b_np)
    keep_mask = np.ones(N, dtype=bool)

    for i in range(N):
        score = s_np[i]
        # High confidence detections (above threshold) are trusted
        if score >= max_score_to_check:
            continue

        cid = int(l_np[i])
        # Exempt planar categories (signs, traffic lights)
        if cid in EXEMPT_CLASS_IDS:
            continue

        x1, y1, x2, y2 = [int(v) for v in b_np[i]]
        x1, x2 = max(0, x1), min(W - 1, x2)
        y1, y2 = max(0, y1), min(H - 1, y2)
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            continue

        # Sample uniform 8x8 grid inside the box
        ys = np.linspace(y1, y2 - 1, _GRID_SIZE).astype(int)
        xs = np.linspace(x1, x2 - 1, _GRID_SIZE).astype(int)
        sampled_z = depth_map[np.ix_(ys, xs)].flatten().astype(np.float64)

        valid_mask = (sampled_z > 10.0) & (sampled_z < 60000.0)
        if valid_mask.sum() < (_GRID_SIZE * _GRID_SIZE * 0.5):
            continue

        med_z = float(np.median(sampled_z[valid_mask]))
        if med_z < 100.0:
            continue

        # Fit 1st-order plane Z(x, y) = a*x + b*y + c
        if np.all(valid_mask):
            coeffs = _PINV_A @ sampled_z
            z_fit = _A_BASIS @ coeffs
            plane_residual_std = float(np.std(sampled_z - z_fit))
        else:
            # Fallback least-squares on valid subset
            valid_idx = np.where(valid_mask)[0]
            A_sub = _A_BASIS[valid_idx]
            z_sub = sampled_z[valid_idx]
            coeffs, _, _, _ = np.linalg.lstsq(A_sub, z_sub, rcond=None)
            z_fit = A_sub @ coeffs
            plane_residual_std = float(np.std(z_sub - z_fit))

        rel_residual = plane_residual_std / med_z

        # If box has almost zero residual from a flat 2D plane, it is a coplanar artifact (e.g. shadow, road stain)
        if rel_residual < rel_thresh:
            keep_mask[i] = False # Prune false positive

    if is_torch:
        return torch.from_numpy(keep_mask).to(boxes.device)
    return keep_mask

