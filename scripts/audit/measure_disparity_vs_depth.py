import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import cv2
import numpy as np
from scripts.lib.depth_utils import read_depth_mm

def main():
    val_ann_path = "work_dir/deimv2_dataset/valid/annotations/instances_val.json"
    if not os.path.exists(val_ann_path):
        print(f"Error: {val_ann_path} not found")
        return

    with open(val_ann_path, "r") as f:
        val_data = json.load(f)

    img_map = {im["id"]: im for im in val_data["images"]}
    ann_by_img = {}
    for ann in val_data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    print(f"Found {len(val_data['images'])} validation images, {len(val_data['annotations'])} annotations.")

    results = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for img_id, anns in ann_by_img.items():
        im_info = img_map[img_id]
        fname = im_info["file_name"]
        stem = os.path.splitext(fname)[0]

        vp = os.path.join("dataset/train/visible", fname)
        ip = os.path.join("dataset/train/infrared", fname)
        dp_cand = [os.path.join("dataset/train/depth", stem + ext) for ext in [".png", ".jpg"]]
        dp = next((p for p in dp_cand if os.path.exists(p)), None)

        if not (os.path.exists(vp) and os.path.exists(ip) and dp):
            continue

        vis = cv2.imread(vp)
        ir = cv2.imread(ip)
        depth_mm = read_depth_mm(dp)

        if vis is None or ir is None or depth_mm is None:
            continue

        H, W = vis.shape[:2]
        is_1080p = (W >= 1280)

        vg = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        ig = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        ig = clahe.apply(ig)

        # Gradient magnitude for cross-modal structural correlation
        gv = cv2.magnitude(cv2.Scharr(vg, cv2.CV_32F, 1, 0), cv2.Scharr(vg, cv2.CV_32F, 0, 1))
        gi = cv2.magnitude(cv2.Scharr(ig, cv2.CV_32F, 1, 0), cv2.Scharr(ig, cv2.CV_32F, 0, 1))

        # Normalize gradients
        gv = (gv - np.mean(gv)) / (np.std(gv) + 1e-5)
        gi = (gi - np.mean(gi)) / (np.std(gi) + 1e-5)

        for ann in anns:
            x, y, w, h = [int(v) for v in ann["bbox"]]
            if w < 28 or h < 28 or w * h > 150000:
                continue

            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)

            crop_d = depth_mm[y0:y1, x0:x1]
            valid_d = crop_d[crop_d > 100]
            if len(valid_d) < 30:
                continue

            med_z_m = float(np.median(valid_d)) / 1000.0  # in meters
            if med_z_m < 0.5 or med_z_m > 30.0:
                continue

            crop_v = gv[y0:y1, x0:x1].astype(np.float32)

            # Search window in IR: +/- 16 px
            pad = 16
            sy0, sy1 = max(0, y0 - pad), min(H, y1 + pad)
            sx0, sx1 = max(0, x0 - pad), min(W, x1 + pad)
            search_i = gi[sy0:sy1, sx0:sx1].astype(np.float32)

            if search_i.shape[0] < crop_v.shape[0] or search_i.shape[1] < crop_v.shape[1]:
                continue

            res = cv2.matchTemplate(search_i, crop_v, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Center expected at (x0 - sx0, y0 - sy0)
            expected_x = x0 - sx0
            expected_y = y0 - sy0

            dx = max_loc[0] - expected_x
            dy = max_loc[1] - expected_y

            # Only accept confident matches
            if max_val >= 0.35:
                results.append({
                    "image": fname,
                    "is_1080p": is_1080p,
                    "depth_m": med_z_m,
                    "inv_z": 1.0 / med_z_m,
                    "dx": float(dx),
                    "dy": float(dy),
                    "dist": float(np.hypot(dx, dy)),
                    "conf": float(max_val),
                    "bbox": [x, y, w, h]
                })

    print(f"\nSuccessfully measured disparity on {len(results)} GT boxes.")

    # Breakdown by resolution
    for res_name, flag in [("1080p (Co-axial Rig)", True), ("640x360 (Offset Rig)", False)]:
        subset = [r for r in results if r["is_1080p"] == flag]
        if not subset:
            continue
        print(f"\n=================== {res_name} (N={len(subset)}) ===================")
        dx_vals = [r["dx"] for r in subset]
        dy_vals = [r["dy"] for r in subset]
        dist_vals = [r["dist"] for r in subset]

        print(f"Overall dx: mean={np.mean(dx_vals):+.3f} px, std={np.std(dx_vals):.3f} px, median={np.median(dx_vals):+.1f} px")
        print(f"Overall dy: mean={np.mean(dy_vals):+.3f} px, std={np.std(dy_vals):.3f} px, median={np.median(dy_vals):+.1f} px")
        print(f"Overall dist: mean={np.mean(dist_vals):.3f} px, median={np.median(dist_vals):.1f} px")

        # Depth stratification
        near = [r for r in subset if r["depth_m"] < 6.0]
        mid = [r for r in subset if 6.0 <= r["depth_m"] < 12.0]
        far = [r for r in subset if r["depth_m"] >= 12.0]

        print(f"  Near (<6m, N={len(near)}): dx={np.mean([r['dx'] for r in near]):+.2f}, dy={np.mean([r['dy'] for r in near]):+.2f}, dist={np.mean([r['dist'] for r in near]):.2f} px") if near else None
        print(f"  Mid (6-12m, N={len(mid)}): dx={np.mean([r['dx'] for r in mid]):+.2f}, dy={np.mean([r['dy'] for r in mid]):+.2f}, dist={np.mean([r['dist'] for r in mid]):.2f} px") if mid else None
        print(f"  Far (>=12m, N={len(far)}): dx={np.mean([r['dx'] for r in far]):+.2f}, dy={np.mean([r['dy'] for r in far]):+.2f}, dist={np.mean([r['dist'] for r in far]):.2f} px") if far else None

        # Linear regression dx vs 1/Z: dx = offset_0 + f*B / Z
        if len(subset) >= 10:
            inv_z = np.array([r["inv_z"] for r in subset])
            dx_arr = np.array([r["dx"] for r in subset])
            dy_arr = np.array([r["dy"] for r in subset])

            # dx fit
            A = np.vstack([inv_z, np.ones(len(inv_z))]).T
            m_x, c_x = np.linalg.lstsq(A, dx_arr, rcond=None)[0]
            corr_x = np.corrcoef(inv_z, dx_arr)[0, 1]

            # dy fit
            m_y, c_y = np.linalg.lstsq(A, dy_arr, rcond=None)[0]
            corr_y = np.corrcoef(inv_z, dy_arr)[0, 1]

            print(f"  Fit dx(Z) = {c_x:+.3f} + ({m_x:+.3f}) / Z [Pearson r = {corr_x:+.3f}]")
            print(f"  Fit dy(Z) = {c_y:+.3f} + ({m_y:+.3f}) / Z [Pearson r = {corr_y:+.3f}]")

    # Save summary
    out_path = "outputs/metrics/disparity_vs_depth_analysis.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed analysis to {out_path}")

if __name__ == "__main__":
    main()
