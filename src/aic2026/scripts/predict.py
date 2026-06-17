#!/usr/bin/env python3
"""
AIC 2026 — 推理脚本 (Track 1: M2I2HA Baseline)

从训练好的 checkpoint 加载模型，对测试集进行推理，
生成竞赛提交格式的 TXT 文件。

用法:
  python predict.py --ckpt runs/aic2026_baseline/best.pt \\
      --data-dir /path/to/aic2026/data \\
      --split test \\
      --output submissions/m2i2ha_baseline

  # 如果你已有 Predictor checkpoint
  python predict.py --ckpt /path/to/checkpoint.pt --data-dir /data

  # 指定置信度阈值
  python predict.py --ckpt best.pt --data-dir /data --conf 0.25 --iou 0.7
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

# -------------- 官方 M2I2HA 工具函数 --------------
from machine_learning.utils.boxes import rescale_boxes, non_max_suppression
from machine_learning.utils.data_augment import pad_to_square
from machine_learning.evaluator import Predictor


def parse_args():
    parser = argparse.ArgumentParser(description="AIC 2026 M2I2HA Inference")
    parser.add_argument("--ckpt", required=True, type=str, help="Checkpoint path (.pt)")
    parser.add_argument("--data-dir", required=True, type=str, help="Competition data root directory")
    parser.add_argument("--split", default="test", type=str, choices=["test", "val"], help="Dataset split")
    parser.add_argument("--output", default="submissions/m2i2ha_baseline", type=str, help="Output directory")
    parser.add_argument("--conf", default=0.25, type=float, help="Confidence threshold")
    parser.add_argument("--iou", default=0.7, type=float, help="NMS IoU threshold")
    parser.add_argument("--device", default="auto", type=str, help="Device (auto/cuda/cpu:0)")
    parser.add_argument("--max-det", default=100, type=int, help="Max detections per image")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------- 设备 ----------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    # ---------- 数据路径 ----------
    data_dir = Path(args.data_dir)
    rgb_dir = data_dir / "visible" / args.split
    ir_dir = data_dir / "infrared" / args.split

    if not rgb_dir.exists():
        print(f"[ERROR] RGB directory not found: {rgb_dir}")
        sys.exit(1)
    if not ir_dir.exists():
        print(f"[ERROR] IR directory not found: {ir_dir}")
        sys.exit(1)

    # 收集图像文件（支持 .png .jpg .jpeg）
    img_exts = {".png", ".jpg", ".jpeg"}
    img_files = sorted([f for f in os.listdir(rgb_dir) if Path(f).suffix.lower() in img_exts])
    print(f"[INFO] Found {len(img_files)} images in {rgb_dir}")

    # ---------- 加载模型 ----------
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    algo_cfg = state["cfg"]
    name = algo_cfg["algorithm"]["name"]
    imgsz = algo_cfg["algorithm"]["imgsz"]

    # 构建 algorithm
    from machine_learning.algorithms import global_factory

    algo = global_factory.create_algorithm(
        algo=name, cfg=algo_cfg, name=name, device=device, amp=False, ema=False,
    )
    # 加载权重
    algo._init_on_predictor(args.ckpt)
    algo.net.eval()
    print(f"[INFO] Model loaded: {name}, imgsz={imgsz}")

    # ---------- 创建输出目录 ----------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = out_dir / "txt"
    txt_dir.mkdir(exist_ok=True)
    print(f"[INFO] Output: {out_dir}")

    # ---------- 预处理 ----------
    tfs = Compose([ToTensor(), Normalize(mean=[0, 0, 0], std=[1, 1, 1])])

    # ---------- 推理 ----------
    print(f"[INFO] Running inference on {args.split} split ({len(img_files)} images)...")

    with torch.no_grad():
        for idx, fname in enumerate(img_files):
            stem = Path(fname).stem
            rgb_path = rgb_dir / fname
            ir_path = ir_dir / fname

            # 检查 IR 是否存在（可能不同扩展名）
            if not ir_path.exists():
                # 尝试相同 basename 的不同扩展名
                for ext in img_exts:
                    alt_path = ir_dir / f"{stem}{ext}"
                    if alt_path.exists():
                        ir_path = alt_path
                        break

            if not ir_path.exists():
                print(f"[WARN] IR not found for {fname}, skipping")
                # 生成空 TXT
                open(txt_dir / f"{stem}.txt", "w").close()
                continue

            # 读图
            img0 = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if img0 is None:
                print(f"[WARN] Failed to read {rgb_path}, skipping")
                open(txt_dir / f"{stem}.txt", "w").close()
                continue
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

            ir0 = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
            if ir0 is None:
                print(f"[WARN] Failed to read {ir_path}, skipping")
                open(txt_dir / f"{stem}.txt", "w").close()
                continue
            ir0 = cv2.cvtColor(ir0, cv2.COLOR_BGR2RGB)

            h0, w0 = img0.shape[:2]

            # pad → resize
            padded_img = pad_to_square(img=img0, pad_values=(114, 114, 114))
            padded_ir = pad_to_square(img=ir0, pad_values=(0, 0, 0))
            img_t = tfs(padded_img)
            img_t = Resize((imgsz, imgsz))(img_t).unsqueeze(0).to(device)
            ir_t = tfs(padded_ir)
            ir_t = Resize((imgsz, imgsz))(ir_t).unsqueeze(0).to(device)

            # 前向
            preds = algo.net(img_t, ir_t)

            # 解码 + NMS
            detections = algo.decode_preds(preds, imgsz)
            dets = non_max_suppression(
                detections.permute(0, 2, 1),
                conf_thres=args.conf,
                iou_thres=args.iou,
                multi_label=True,
                max_det=args.max_det,
                agnostic=algo.single_cls,
            )[0]

            # 写 TXT
            txt_path = txt_dir / f"{stem}.txt"
            if len(dets) == 0:
                # 空文件
                txt_path.write_text("")
            else:
                bboxes, confs, cls_ids = dets.split((4, 1, 1), dim=1)
                bboxes_np = np.array(bboxes.cpu(), dtype=np.float32)
                # rescale from (imgsz, imgsz) to original (h0, w0)
                bboxes_np = rescale_boxes(bboxes_np, (imgsz, imgsz), (h0, w0))
                # xyxy → xywh (归一化)
                lines = []
                for i in range(len(bboxes_np)):
                    x1, y1, x2, y2 = bboxes_np[i]
                    cid = int(cls_ids[i].item())
                    conf = float(confs[i].item())

                    # 转 YOLO 格式: cx cy w h (归一化)
                    bw = (x2 - x1) / w0
                    bh = (y2 - y1) / h0
                    cx = (x1 + x2) / 2 / w0
                    cy = (y1 + y2) / 2 / h0

                    # 裁剪到 [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    bw = max(0, min(1, bw))
                    bh = max(0, min(1, bh))

                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}")

                txt_path.write_text("\n".join(lines))

            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(img_files)}] done")

    # ---------- 打包提交文件 ----------
    import shutil
    import zipfile

    # 创建压缩包
    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(txt_dir.iterdir()):
            if txt_file.suffix == ".txt":
                zf.write(txt_file, arcname=txt_file.name)

    print(f"\n[INFO] ✅ Done! {len(img_files)} images processed.")
    print(f"[INFO]    TXT 文件: {txt_dir}/")
    print(f"[INFO]    提交包:   {zip_path}")
    print(f"[INFO]    文件数:   {len(list(txt_dir.glob('*.txt')))}")

    # 清理图片数量检查
    total_txts = len(list(txt_dir.glob("*.txt")))
    if total_txts != len(img_files):
        print(f"[WARN] TXT 数量 ({total_txts}) ≠ 图像数量 ({len(img_files)}), 可能有缺失")


if __name__ == "__main__":
    main()
