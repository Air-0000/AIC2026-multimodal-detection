#!/usr/bin/env python3
"""把 de-leak 热图（heat 残差图）正确叠加到 RGB 上。

热图来源（2026-09-28 phaseCorrelate 实测修正）:
  * deimv2_heat 与 deleak 逐像素一致（4 个平移样本 shift≈0，response=1.000）
    —— 热图从未被 align_ir() 平移，两版等价，均与 RGB 出厂像素级对齐。
  * 被平移 46~86px 的是 raw IR 通道 deimv2_ir（脏）；干净版是 deimv2_ir_na。
  * 叠加热图不需要任何配准，直接叠即可。

叠加配方:
  1. heat 以灰度读入; 若与 RGB 尺寸不同, cv2.resize 到 RGB 尺寸 (INTER_LINEAR)
  2. 鲁棒归一化: 按 [p_lo, p_hi] 百分位拉伸到 0..255 (防极值把色标压扁)
  3. cv2.applyColorMap 上色 (默认 JET)
  4. --thresh 0~1: 只叠加热值高于该分位数的像素(热区掩膜), 背景保持原 RGB
  5. out = (1-alpha)*RGB + alpha*heat_color
"""
import argparse
import os

import cv2
import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"


def find_file(d, stem):
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def load_heat(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = im[:, :, 0]
    return im.astype(np.float32)


def overlay(rgb, heat, alpha=0.45, colormap=cv2.COLORMAP_JET,
            p_lo=2.0, p_hi=98.0, thresh=None):
    if heat.shape[:2] != rgb.shape[:2]:
        heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    lo, hi = np.percentile(heat, p_lo), np.percentile(heat, p_hi)
    if hi <= lo:
        hi = lo + 1.0
    hn = np.clip((heat - lo) / (hi - lo), 0, 1)
    heat_bgr = cv2.applyColorMap((hn * 255).astype(np.uint8), colormap)
    if thresh is not None:
        mask = (hn > thresh).astype(np.float32)[:, :, None]
        mask = cv2.GaussianBlur(mask, (5, 5), 0)[:, :, None]
    else:
        mask = 1.0
    out = (1 - alpha) * rgb + alpha * heat_bgr
    if thresh is not None:
        m3 = np.repeat((hn > thresh)[:, :, None], 3, axis=2)
        out = np.where(m3, out, rgb)
    return np.clip(out, 0, 255).astype(np.uint8), (hn * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_dir", default=os.path.join(ROOT, "work_dir/deimv2_dataset/train/images"))
    ap.add_argument("--heat_dir", default=os.path.join(ROOT, "work_dir/deleak/images"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", nargs="*", default=None,
                    help="指定样本 stem(不带扩展名); 缺省取目录交集前 --limit 个")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--colormap", default="jet",
                    choices=["jet", "inferno", "hot", "viridis"])
    ap.add_argument("--thresh", type=float, default=None,
                    help="0~1, 只叠加高于该热值分位的像素; 不设则全图混色")
    ap.add_argument("--triple", action="store_true", help="额外输出 RGB|heat|overlay 三联图")
    args = ap.parse_args()

    cmap = {"jet": cv2.COLORMAP_JET, "inferno": cv2.COLORMAP_INFERNO,
            "hot": cv2.COLORMAP_HOT, "viridis": cv2.COLORMAP_VIRIDIS}[args.colormap]
    os.makedirs(args.out, exist_ok=True)

    if args.stems:
        stems = args.stems
    else:
        rgb_files = {os.path.splitext(f)[0] for f in os.listdir(args.rgb_dir)}
        heat_files = {os.path.splitext(f)[0] for f in os.listdir(args.heat_dir)}
        stems = sorted(rgb_files & heat_files)[: args.limit]

    for stem in stems:
        rp = find_file(args.rgb_dir, stem)
        hp = find_file(args.heat_dir, stem)
        if rp is None or hp is None:
            print(f"[skip] {stem}: rgb={rp} heat={hp}")
            continue
        rgb = cv2.imread(rp, cv2.IMREAD_COLOR)
        heat = load_heat(hp)
        ov, hn = overlay(rgb, heat, alpha=args.alpha, colormap=cmap, thresh=args.thresh)
        cv2.imwrite(os.path.join(args.out, f"{stem}_overlay.jpg"), ov)
        if args.triple:
            heat_bgr = cv2.applyColorMap(hn, cv2.COLORMAP_JET)
            h, w = rgb.shape[:2]
            bar = np.full((h, 8, 3), 255, np.uint8)
            trip = np.concatenate([rgb, bar, heat_bgr, bar, ov], axis=1)
            cv2.imwrite(os.path.join(args.out, f"{stem}_triple.jpg"), trip)
        print(f"[ok] {stem}")


if __name__ == "__main__":
    main()
