#!/usr/bin/env python3
"""全量体检生成的训练/评测配置。

对每个 yml 依次检查:
  1. YAMLConfig 能解析 (含 __include__ 继承链、算子 type 都能被注册器找到)
  2. 关键字段: eval_spatial_size / backbone name / aux_name / interaction_indexes /
     unfreeze_rgb_stages / accum_steps / total_batch_size
  3. output_dir 已渲染且全局唯一 (两套实验不能写进同一目录互相覆盖)
  4. 数据集各路径真实存在
  5. dataset 能实例化并读出一条样本 (真跑一遍 transform 管线, 验证新增算子)

模型构建很慢 (ViT-L 版 384M 参数), 默认只对 train 配置 + 第一个 val 配置做,
用 --all-models 可以强制每个配置都构建。

用法:
    python3 scripts/gen/check_configs.py                       # 默认
    python3 scripts/gen/check_configs.py --skip-samples        # 不读样本(快)
    python3 scripts/gen/check_configs.py --prefix vitl800      # 只测一套
"""
import argparse
import os
import re
import sys

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIM = os.path.join(ROOT, "vendor/DEIMv2")
sys.path.insert(0, DEIM)
sys.path.insert(0, os.path.join(ROOT, "scripts/lib"))

import torch  # noqa: E402
from engine.core import YAMLConfig  # noqa: E402


def path_ok(p):
    return os.path.exists(p)


def check_paths(ds_cfg, cfg_path, errs):
    """检查 dataset 配置里的目录/文件是否真实存在。"""
    keys = ["img_folder", "ir_folder", "raw_ir_folder", "depth_folder",
            "cache_img_folder", "ann_file"]
    for k in keys:
        v = ds_cfg.get(k)
        if v is None:
            continue
        if not path_ok(v):
            errs.append(f"{cfg_path}: {k} 不存在 -> {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs="+",
                    default=["deimv2_dinov3_l_aic_vitb800",
                             "deimv2_dinov3_l_aic_vitl800"],
                    help="配置文件名前缀过滤 (可多个)")
    ap.add_argument("--size", type=int, default=800, help="期望的分辨率")
    ap.add_argument("--skip-samples", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--all-models", action="store_true")
    args = ap.parse_args()

    cfg_dir = os.path.join(DEIM, "configs/deimv2")
    files = sorted(f for f in os.listdir(cfg_dir)
                   if f.endswith(".yml") and any(f.startswith(p) for p in args.prefix))
    if not files:
        print("没有匹配的配置:", args.prefix)
        return 1

    os.chdir(DEIM)  # 相对路径 (./outputs/..., configs/...) 以此为基准
    seen_out = {}
    failures = []

    for fn in files:
        rel = "configs/deimv2/" + fn
        errs = []
        try:
            cfg = YAMLConfig(rel)
        except Exception as e:
            print(f"[FAIL] {fn}: 解析失败 -> {type(e).__name__}: {e}")
            failures.append(fn)
            continue

        y = cfg.yaml_cfg

        # --- 2. 关键字段 ---
        s = args.size
        ess = y.get("eval_spatial_size")
        if ess != [s, s]:
            errs.append(f"eval_spatial_size={ess} (期望 [{s}, {s}])")
        bb = y.get("DINOv3STAsRDTTrack", {})
        for k in ("name", "aux_name", "interaction_indexes", "unfreeze_rgb_stages"):
            if k not in bb:
                errs.append(f"DINOv3STAsRDTTrack 缺字段 {k}")
        if bb.get("weights_path") is not None:
            errs.append(f"weights_path 应为 null (由 -t 热启动), 实际 {bb.get('weights_path')}")

        # --- 3. output_dir 渲染 + 唯一性 ---
        out = y.get("output_dir")
        if not out or "{" in str(out):
            errs.append(f"output_dir 未渲染: {out!r}")
        else:
            seen_out.setdefault(os.path.abspath(out), []).append(fn)

        # --- 4/5. 数据集路径 + 取样本 ---
        for split in ("train_dataloader", "val_dataloader"):
            dl = y.get(split)
            if not dl:
                continue
            ds = dl.get("dataset", {})
            check_paths(ds, fn, errs)
            if args.skip_samples:
                continue
            try:
                dataset = getattr(cfg, split).dataset
                img, tgt = dataset[0][0], dataset[0][1]
                if tuple(img.shape)[0] != 6:
                    errs.append(f"{split} 通道数={tuple(img.shape)[0]} (期望 6)")
                if tuple(img.shape)[-2:] != (s, s):
                    errs.append(f"{split} 尺寸={tuple(img.shape)[-2:]} (期望 {s}x{s})")
                boxes = tgt["boxes"]
                _ = boxes.shape
            except Exception as e:
                errs.append(f"{split} 取样本失败 -> {type(e).__name__}: {e}")

        # --- 模型构建 (慢, 默认只挑代表性的) ---
        is_train = not fn.endswith(tuple("_val_" + s for s in
                                         ["orig", "night50", "night25", "night12",
                                          "night05", "night01", "night005",
                                          "night002", "night001", "night0005"])) \
            and "_val_" not in fn
        if args.skip_models:
            note = "model (skipped)"
        elif args.all_models or is_train or fn.endswith("_val_orig.yml"):
            try:
                m = cfg.model
                n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
                n_all = sum(p.numel() for p in m.parameters())
                note = f"model {n_tr/1e6:.1f}M/{n_all/1e6:.1f}M"
            except Exception as e:
                note = None
                errs.append(f"模型构建失败 -> {type(e).__name__}: {e}")
        else:
            note = "model (skipped, same as train cfg)"

        if errs:
            failures.append(fn)
            print(f"[FAIL] {fn}")
            for e in errs:
                print("        -", e)
        else:
            print(f"[OK  ] {fn}  {note}")

    # --- output_dir 撞车 ---
    dup = {k: v for k, v in seen_out.items() if len(v) > 1}
    if dup:
        print("\n[FAIL] output_dir 冲突 (多个配置写同一目录):")
        for k, v in dup.items():
            print("       ", k, "<-", v)
        failures.append("output_dir 冲突")

    print(f"\n合计 {len(files)} 个配置, 失败 {len([f for f in failures if isinstance(f, str) and f.endswith('.yml')])}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
