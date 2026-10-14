#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同 epoch 对比 vitl800 与基线(midboost_l4u3 / 可选 rdttrack_heat) 的逐 epoch mAP。
用法: python3 scripts/analysis/compare_vitl800_baseline_map.py
输出: work_dir/vitl800_vs_baseline_map.png + 控制台对比表
"""
import json
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
OUT_BASE = os.path.join(ROOT, "vendor/DEIMv2/outputs")

TARGET = "deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug"
BASE_L4U3 = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3"
BASE_HEAT = "deimv2_dinov3_l_aic_rdttrack_heat"  # 纯 ViT-S RDTTrack, 作为次级参考


def load_curve(name):
    """从 outputs/<name>/log.txt 抽取 {epoch: mAP} (取该 epoch 最后一条记录)。"""
    path = os.path.join(OUT_BASE, name, "log.txt")
    if not os.path.isfile(path):
        print(f"[skip] 找不到 {name}/log.txt")
        return {}
    curve = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            ep = d.get("epoch")
            m = d.get("test_coco_eval_bbox")
            if ep is None or not isinstance(m, list) or not m:
                continue
            curve[int(ep)] = float(m[0])
    return curve


def main():
    vit = load_curve(TARGET)
    l4 = load_curve(BASE_L4U3)
    heat = load_curve(BASE_HEAT)

    print(f"vitl800 已记录 epoch 数: {len(vit)}  (最后 epoch={max(vit) if vit else '-'})")
    print(f"l4u3 基线 已记录 epoch 数: {len(l4)}  (最佳 mAP={max(l4.values()):.4f} @ep{max(l4,key=l4.get)})")
    if heat:
        print(f"rdttrack_heat 已记录 epoch 数: {len(heat)}  (最佳 mAP={max(heat.values()):.4f} @ep{max(heat,key=heat.get)})")

    # 同 epoch 对比表: 以 vitl800 已跑到的 epoch 为范围, 取两边都有的
    common = sorted(set(vit) & set(l4))
    print("\n=== 同 epoch mAP 对比 (vitl800 vs midboost_l4u3) ===")
    print(f"{'epoch':>5} | {'vitl800':>8} | {'l4u3':>8} | {'delta':>7} | {'领先ep数':>7}")
    print("-" * 48)
    lead_sum = 0
    for ep in common:
        v, b = vit[ep], l4[ep]
        delta = v - b
        # 找 l4u3 中 mAP 等于 v 的最近 epoch, 估算 vitl800 领先多少 epoch
        lead = ""
        try:
            nearer = [e for e in l4 if l4[e] <= v]
            if nearer:
                lead_ep = max(nearer)
                lead = f"+{ep - lead_ep}"
        except Exception:
            pass
        print(f"{ep:>5} | {v:>8.4f} | {b:>8.4f} | {delta:>+7.4f} | {lead:>7}")
        lead_sum += delta

    if common:
        print("-" * 48)
        print(f"区间平均 delta (vitl800 - l4u3): {lead_sum/len(common):+.4f}")
        vbest = max(vit.values())
        print(f"vitl800 当前最佳 mAP={vbest:.4f} @ep{max(vit,key=vit.get)} | l4u3 最佳={max(l4.values()):.4f} @ep{max(l4,key=l4.get)}")

    # ---- 画图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        xs = sorted(vit)
        ax.plot(xs, [vit[e] for e in xs], "o-", color="#d62728", lw=2, label=f"vitl800 (ViT-L/800, ep{max(xs)})")
        if l4:
            xl = sorted(l4)
            ax.plot(xl, [l4[e] for e in xl], "s-", color="#1f77b4", lw=1.6, label=f"baseline l4u3 (ViT-S/640, best {max(l4.values()):.3f}@ep{max(l4,key=l4.get)})")
        if heat:
            xh = sorted(heat)
            ax.plot(xh, [heat[e] for e in xh], ":", color="#7f7f7f", lw=1.2, label="rdttrack_heat (ViT-S/640 frozen)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("mAP (test_coco_eval_bbox[0])")
        ax.set_title("vitl800 vs baseline: per-epoch mAP")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out_png = os.path.join(ROOT, "work_dir", "vitl800_vs_baseline_map.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        print(f"\n[OK] 曲线图已保存: {out_png}")
    except Exception as e:
        print(f"\n[warn] 画图失败 (matplotlib 问题): {e}")


if __name__ == "__main__":
    main()
