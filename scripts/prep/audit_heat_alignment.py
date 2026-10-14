#!/usr/bin/env python3
"""全集对齐审计：逐张验证 deleak 热图 / raw IR 与 RGB 的配准关系。

对每个样本（train 1800 + valid 200）计算：
  1. shift_ir    = phaseCorrelate(灰度 ir_na, 灰度 rgb)   → 干净 raw IR 相对 RGB 的 (dx,dy)
                  （热图=IR−RGB灰度 的残差，继承 IR 的配准；IR 对齐则热图对齐）
  2. response    = 相位相关峰值置信度
  3. r_prov      = pearson(heat_na, clip(ir_na - rgb_gray)) → 热图溯源一致性
  4. size_ok     = heat/ir 与 RGB 是否同尺寸

输出 outputs/metrics/heat_alignment_audit.json + stdout 汇总。
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SPLITS = {
    "train": {
        "rgb": f"{ROOT}/work_dir/deimv2_dataset/train/images",
        "heat": f"{ROOT}/work_dir/deleak/images",
        "ir": f"{ROOT}/work_dir/dataset/train/infrared/images",
    },
    "valid": {
        "rgb": f"{ROOT}/work_dir/deimv2_dataset/valid/images",
        "heat": f"{ROOT}/work_dir/deleak/valid",
        "ir": f"{ROOT}/work_dir/dataset/train/infrared/valid",
    },
}


def find(d, stem):
    for e in (".jpg", ".png", ".jpeg"):
        p = os.path.join(d, stem + e)
        if os.path.exists(p):
            return p
    return None


def gray(path, size=None):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    if size is not None and im.shape[:2] != size[:2]:
        im = cv2.resize(im, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float64)


def main():
    out = {"splits": {}, "flagged": []}
    for split, dirs in SPLITS.items():
        rgb_files = {os.path.splitext(f)[0] for f in os.listdir(dirs["rgb"])}
        stems = sorted(rgb_files)
        rows, n_missing, n_resized, bad_prov = [], 0, 0, []
        for i, s in enumerate(stems):
            rp, hp, ip = find(dirs["rgb"], s), find(dirs["heat"], s), find(dirs["ir"], s)
            if not (rp and hp and ip):
                n_missing += 1
                continue
            rgb_c = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
            heat = cv2.imread(hp, cv2.IMREAD_GRAYSCALE)
            ir = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            size_ok = heat.shape == rgb_c.shape and ir.shape == rgb_c.shape
            if not size_ok:
                n_resized += 1
            g_rgb = gray(rp)
            g_heat = gray(hp, g_rgb.shape)
            g_ir = gray(ip, g_rgb.shape)
            (dx, dy), resp = cv2.phaseCorrelate(g_ir, g_rgb)
            resid = np.clip(g_ir - g_rgb, 0, 255).astype(np.float32).ravel()
            hm = g_heat.astype(np.float32).ravel()
            if resid.std() > 1e-3 and hm.std() > 1e-3:
                r_prov = float(np.corrcoef(resid, hm)[0, 1])
            else:
                r_prov = 0.0
            rows.append({"stem": s, "dx": round(dx, 2), "dy": round(dy, 2),
                         "resp": round(resp, 3), "r_prov": round(r_prov, 3),
                         "size_ok": bool(size_ok)})
            if resp >= 0.15 and (abs(dx) > 1.5 or abs(dy) > 1.5):
                bad_prov.append(rows[-1])
            if (i + 1) % 400 == 0:
                print(f"[{split}] {i+1}/{len(stems)}", flush=True)
        shifts = np.array([[r["dx"], r["dy"]] for r in rows])
        mags = np.hypot(shifts[:, 0], shifts[:, 1])
        provs = np.array([r["r_prov"] for r in rows])
        out["splits"][split] = {
            "n": len(rows), "n_missing": n_missing, "n_resized": n_resized,
            "shift_median": [float(np.median(shifts[:, 0])), float(np.median(shifts[:, 1]))],
            "shift_p95_abs": [float(np.percentile(np.abs(shifts[:, 0]), 95)),
                              float(np.percentile(np.abs(shifts[:, 1]), 95))],
            "mag_max": float(mags.max()), "mag_median": float(np.median(mags)),
            "n_shift_gt1p5": int((mags > 1.5).sum()),
            "n_shift_gt5": int((mags > 5).sum()),
            "prov_median": float(np.median(provs)),
            "prov_p10": float(np.percentile(provs, 10)),
            "n_prov_lt0p8": int((provs < 0.8).sum()),
            "flagged": bad_prov[:20],
        }
        print(f"[{split}] done: n={len(rows)} missing={n_missing}", flush=True)
    with open(f"{ROOT}/outputs/metrics/heat_alignment_audit.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for split, s in out["splits"].items():
        print(f"\n== {split} ==")
        for k, v in s.items():
            if k != "flagged":
                print(f"  {k}: {v}")
        if s["flagged"]:
            print("  flagged:", s["flagged"][:5])


if __name__ == "__main__":
    sys.exit(main())
