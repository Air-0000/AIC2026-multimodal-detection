#!/usr/bin/env python3
"""
Server-Side High-Speed NightXX Synthetic Dataset Generator
===========================================================
Generates the complete 10-level realistic night degradation spectrum
(night50 -> night25 -> night12 -> night05 -> night01 -> night005 ->
 night002 -> night001 -> night0005 -> black)
directly from validation RGB images using physical DNDT priors:
  1. Low-frequency illumination field (street-lamp blobs + vignette)
  2. Poisson-Gaussian sensor noise (photon shot noise + read noise)
  3. Cool white-balance color shift
  4. Global log-attenuation
"""

import os
import sys
import zlib
import time
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image

DEGRADATION_SPECS = [
    # (tag, level, photon_max, read_sigma)
    ("night50", 0.50, 70.0, 0.012),
    ("night25", 0.25, 70.0, 0.012),
    ("night12", 0.12, 70.0, 0.012),
    ("night05", 0.05, 70.0, 0.012),
    ("night01", 0.01, 30.0, 0.020),
    ("night005", 0.005, 30.0, 0.020),
    ("night002", 0.002, 30.0, 0.020),
    ("night001", 0.001, 30.0, 0.020),
    ("night0005", 0.0005, 30.0, 0.020),
    ("black", 0.0, 0.0, 0.0),
]


def _lowfreq_field(h, w, rng):
    gy, gx = 4, 4
    coarse = rng.normal(0.0, 0.22, size=(gy, gx)).astype(np.float32)
    f = np.asarray(Image.fromarray(coarse, mode="F").resize((w, h), Image.BICUBIC), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    f = 1.0 + f - 0.45 * np.clip(r - 0.4, 0, None)
    for _ in range(int(rng.integers(1, 4))):
        by = rng.uniform(0.0, h)
        bx = rng.uniform(0.0, w)
        rad = rng.uniform(0.10, 0.28) * max(h, w)
        blob = np.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2.0 * rad ** 2)))
        f = f + rng.uniform(0.6, 2.2) * blob
    f = np.clip(f, 0.15, 3.5)
    return (f / max(f.mean(), 1e-6)).astype(np.float32)


def degrade_image(arr, level, seed, photon_max, read_sigma):
    """arr: float32 HxWx3 in [0, 1]. Returns degraded float32 HxWx3 in [0, 1]."""
    if level <= 0.0:
        return np.zeros_like(arr)
    rng = np.random.default_rng(seed)
    h, w = arr.shape[:2]
    x = arr * level
    x = x * _lowfreq_field(h, w, rng)[..., None]
    gains = rng.uniform(0.85, 1.15, size=3).astype(np.float32) * np.array([0.92, 0.98, 1.12], np.float32)
    x = x * gains[None, None, :]
    x = np.clip(x, 0.0, None)
    if photon_max > 0:
        x = rng.poisson(x * photon_max).astype(np.float32) / photon_max
    if read_sigma > 0:
        x = x + rng.normal(0.0, read_sigma, size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 1.0)


def _seed(level, name):
    return (zlib.crc32((name + "|%.3f" % level).encode()) & 0xFFFFFFFF)


def process_single_task(args):
    src_path, dst_path, level, photon_max, read_sigma = args
    if os.path.exists(dst_path):
        return 0

    im = Image.open(src_path).convert("RGB")
    fname = os.path.basename(src_path)

    if level <= 0.0:
        w, h = im.size
        black_im = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
        if dst_path.lower().endswith(".png"):
            black_im.save(dst_path)
        else:
            black_im.save(dst_path, quality=95)
        return 1

    arr = np.asarray(im, dtype=np.float32) / 255.0
    seed = _seed(level, fname)
    deg = degrade_image(arr, level, seed, photon_max, read_sigma)
    out_im = Image.fromarray((deg * 255.0 + 0.5).astype(np.uint8))

    if dst_path.lower().endswith(".png"):
        out_im.save(dst_path)
    else:
        out_im.save(dst_path, quality=95)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/images")
    parser.add_argument("--dst_root", type=str, default="/root/autodl-tmp/AIC/data_aux")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_root = Path(args.dst_root)

    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist!")
        sys.exit(1)

    image_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print(f"[+] Found {len(image_files)} source validation images in {src_dir}")
    print(f"[+] Generating {len(DEGRADATION_SPECS)} degradation levels...")

    t0 = time.time()
    total_generated = 0

    for tag, level, photon_max, read_sigma in DEGRADATION_SPECS:
        level_dir = dst_root / f"aic_valid_rgb_{tag}"
        level_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for img_p in image_files:
            dst_p = level_dir / img_p.name
            tasks.append((str(img_p), str(dst_p), level, photon_max, read_sigma))

        workers = min(args.workers, cpu_count())
        with Pool(workers) as pool:
            results = pool.map(process_single_task, tasks)

        gen_count = sum(results)
        total_generated += gen_count
        print(f"  - [{tag:<10}] level={level:<6.4f} -> {len(tasks)} images ready ({gen_count} new) in {level_dir}")

    elapsed = time.time() - t0
    print(f"\n[+] NightXX Dataset Generation Complete in {elapsed:.2f}s! Total files generated/verified: {len(image_files) * len(DEGRADATION_SPECS)}")


if __name__ == "__main__":
    main()
