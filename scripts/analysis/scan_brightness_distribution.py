"""统计 RGB 图像亮度分布 —— 判断「测试集到底有多暗」，决定优化资源投向。

用法:
    python scripts/analysis/scan_brightness_distribution.py

输出:
    outputs/metrics/brightness_distribution.json  + 控制台表格
"""
import os
import sys
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"

TARGETS = {
    "test_phase2_visible": f"{ROOT}/dataset/test/AIC2026_PHASE_2_1000/visible",
    "train_visible_all":   f"{ROOT}/dataset/train/visible",
    "val200_visible":      f"{ROOT}/work_dir/deimv2_dataset/valid/images",
    "bench_night50":       f"{ROOT}/data_aux/aic_valid_rgb_night50",
    "bench_night25":       f"{ROOT}/data_aux/aic_valid_rgb_night25",
    "bench_night12":       f"{ROOT}/data_aux/aic_valid_rgb_night12",
    "bench_night05":       f"{ROOT}/data_aux/aic_valid_rgb_night05",
    "bench_night01":       f"{ROOT}/data_aux/aic_valid_rgb_night01",
}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def one(path):
    try:
        im = Image.open(path).convert("L")
        a = np.asarray(im, dtype=np.float32)
        if im.width * im.height > 400 * 400:            # 抽样加速
            a = a[::4, ::4]
        return float(a.mean() / 255.0), float(np.percentile(a, 90) / 255.0)
    except Exception:
        return None


def scan(d, limit=1200):
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
    if not files:
        return None
    real = [os.path.realpath(os.path.join(d, f)) for f in files[:limit]]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = [r for r in ex.map(one, real, chunksize=16) if r is not None]
    if not res:
        return None
    mean = np.array([r[0] for r in res])
    p90 = np.array([r[1] for r in res])
    return {
        "n": len(mean),
        "mean": round(float(mean.mean()), 4),
        "p10": round(float(np.percentile(mean, 10)), 4),
        "p25": round(float(np.percentile(mean, 25)), 4),
        "median": round(float(np.median(mean)), 4),
        "p75": round(float(np.percentile(mean, 75)), 4),
        "p90": round(float(np.percentile(mean, 90)), 4),
        "frac_below_0.30": round(float((mean < 0.30).mean()), 4),
        "frac_below_0.15": round(float((mean < 0.15).mean()), 4),
        "frac_below_0.05": round(float((mean < 0.05).mean()), 4),
        "p90_gray_mean": round(float(p90.mean()), 4),
        "hist": [int(x) for x in np.histogram(mean, bins=20, range=(0, 1))[0]],
    }


def main():
    out = {}
    for k, d in TARGETS.items():
        r = scan(d)
        out[k] = r
        if r is None:
            print(f"{k:24s}  <missing: {d}>")
        else:
            print(f"{k:24s} n={r['n']:5d} mean={r['mean']:.4f} med={r['median']:.4f} "
                  f"p10={r['p10']:.3f} p25={r['p25']:.3f} p75={r['p75']:.3f} p90={r['p90']:.3f} "
                  f"|<0.30={r['frac_below_0.30']*100:.1f}% <0.15={r['frac_below_0.15']*100:.1f}% "
                  f"<0.05={r['frac_below_0.05']*100:.1f}%")
    os.makedirs(f"{ROOT}/outputs", exist_ok=True)
    with open(f"{ROOT}/outputs/metrics/brightness_distribution.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n-> {ROOT}/outputs/metrics/brightness_distribution.json")


if __name__ == "__main__":
    main()
