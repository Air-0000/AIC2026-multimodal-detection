#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练显存探测器 —— 在真实 GPU 上回答 "这套配置跑不跑得动"。

为什么需要它:
    DEIMv2 的 train.py 要跑到第一个 print_freq 才输出 `max mem`, 而在 16GB 卡上
    跑 640+ViT-S 基线就已经峰值 12.2GB。换更大 backbone / 更高分辨率时, 靠"跑起来
    看会不会 OOM"来试错代价太高(一次 OOM 要等几分钟加载 + 数据集预热)。
    本脚本只跑 2-3 个 iteration 就报出稳态峰值, 并自动二分搜索最大可行 micro batch。

用法 (必须在 vendor/DEIMv2 的 conda env `deimv2` 下运行):
    cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
    python /home/constant/CodeBuddy/Dev/AIC/scripts/train/probe_mem.py \
        -c configs/deimv2/<cfg>.yml                 # 测配置里写死的 micro
    ... --auto                                      # 自动二分搜最大 micro
    ... --micro 4 --micro 2                         # 手动测若干档
    ... --no-ema                                    # 不建 EMA (省一份模型副本)

输出解读:
    static   = 参数 + 梯度 + AdamW状态 + EMA 副本 + AMP fp16 cache (与 batch 无关)
    dynamic  = 激活 + 碎片 (随 micro 近似线性增长)
    peak     = 进程实测峰值 (torch.cuda.max_memory_allocated, 不含 CUDA context)
    headroom = GPU 总显存 - 当前已占用 - peak    ← 看这个决定安不安全
"""
import argparse
import os
import sys
import time

DT = []  # 每个 micro forward+backward 的实测耗时(秒), 用于推算训练总时长

DEIM = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2"


# ---------------------------------------------------------------------------
# 实测锚点: 来源于 logs/midboost_l4u3_train.log 的 `max mem: 12228`
#   = 640x640 / ViT-S 三流 / micro=5 / accum=2 / use_amp / use_ema
# 对应模型 107.4M 总参数 / 85.8M 可训练 (与 vitb800/vitl800 同一套静态公式)
# ---------------------------------------------------------------------------
_MB = 2**20
ANCHOR = dict(
    peak_mb=12228.0,
    total_params=107.4e6,
    trainable=85.8e6,
    micro=5,
    size=640,
    patch=16,
    units=3 * 1600 * 384 * 12,      # 三流 ViT-S: tokens x embed_dim x layers
    gpu_total_mb=16596140032 / _MB,  # 16.6GB (RTX 5060 Ti, 取自日志 OOM 行的 total)
)


def _train_size(y):
    """从 train_dataloader 的 transforms 里找 Resize 的目标边长。"""
    ops = y.get("train_dataloader", {}).get("dataset", {}).get("transforms", {})
    for op in ops.get("ops", []) or []:
        if op.get("type") == "Resize":
            s = op.get("size")
            return s[0] if isinstance(s, (list, tuple)) else s
    ess = y.get("eval_spatial_size")
    return ess[0] if ess else 640


def _backbone_units(model, size):
    """每图的主干激活规模: sum over streams( tokens x embed_dim x n_blocks )。

    RDTTrack 有三条 DINOv3 流(rgb / ir / depth)。若开了 use_modality_stem,
    辅助流退化成小 conv stem, 不计入(其激活量远小于 ViT)。
    """
    bb = model.backbone if hasattr(model, "backbone") else model
    if hasattr(bb, "module"):
        bb = bb.module
    units = 0.0
    for name in ("rgb_dinov3", "ir_dinov3", "depth_dinov3"):
        dv = getattr(bb, name, None)
        if dv is None:
            continue
        ps = getattr(dv, "patch_size", 16)
        tokens = (size // ps) ** 2
        units += tokens * dv.embed_dim * len(dv.blocks)
    return units


def _estimate(args, static, units, base_micro, tbs, total_gb):
    """用实测锚点做一阶外推: 动态显存 ∝ micro x 主干激活单位。"""
    a = ANCHOR
    a_static = (a["total_params"] * 4 + a["trainable"] * 4 + a["trainable"] * 8
                + a["total_params"] * 4 + a["total_params"] * 2) / _MB
    k = (a["peak_mb"] - a_static) / (a["micro"] * a["units"])

    print("\n[估算模式] 无 GPU, 用实测锚点外推 —— 数字是量级参考, 不是承诺值")
    print("  锚点: 640/ViT-S/micro=5 实测峰值 %.0f MB (静态 %.0f + 动态 %.0f)"
          % (a["peak_mb"], a_static, a["peak_mb"] - a_static))
    print("  系数: 动态 = %.4g MB per (micro x 主干单位)" % k)
    print("  模型: 静态 %.0f MB, 主干单位 %.2fM/图 (锚点 %.2fM/图)"
          % (static, units / 1e6, a["units"] / 1e6))

    gpu_mb = total_gb * 1024 if total_gb else a["gpu_total_mb"]
    print("\n  %-6s %10s %10s %12s" % ("micro", "估算峰值", "占卡", "余量"))
    print("  " + "-" * 44)
    safe = []
    for bs in range(base_micro, 0, -1):
        est = static + k * bs * units
        left = gpu_mb - est
        flag = ""
        if left < 0:
            flag = "  OOM"
        elif left < 1500:
            flag = "  偏紧"
        else:
            safe.append(bs)
        print("  %-6d %8.0f MB %9.1f%% %9.0f MB%s"
              % (bs, est, 100.0 * est / gpu_mb, left, flag))

    print("  " + "-" * 44)
    print("  (占卡/余量按 GPU %.2f GiB 计算; 所有 MB 均为 MiB)" % (gpu_mb / 1024))
    if safe:
        b = safe[0]
        print("\n[建议] micro=%d, accum_steps=%d 可保住等效 batch %d"
              % (b, max(1, tbs // b), tbs))
    else:
        print("\n[建议] 全部档位都不安全 -> 降分辨率 / use_modality_stem / 退小 backbone")
    print("\n  一阶近似的已知偏差(真实值可能上下浮动 ~20%):")
    print("   - HybridEncoder 多尺度特征图随分辨率增长, 未被单独建模")
    print("   - decoder/匈牙利匹配部分 ∝ micro 而非主干单位, micro 小时会略省")
    print("   - 8bit/碎片/CUDA context 未计入")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--micro", type=int, nargs="*", default=None,
                    help="要测的 micro batch 列表; 不给则用配置里的 tbs//accum")
    ap.add_argument("--auto", action="store_true",
                    help="自动从配置值往下二分, 找出最大可行 micro")
    ap.add_argument("--iters", type=int, default=3,
                    help="每个 micro 跑几个 iteration (第2步起才是稳态峰值)")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-micro", type=int, default=8)
    ap.add_argument("--max-scale", action="store_true",
                    help="多取几个 batch 并选尺寸最大的那个 —— 测最坏情况"
                         " (配置开了多尺度训练时, 峰值随尺寸平方变化)")
    ap.add_argument("--estimate", action="store_true",
                    help="无 GPU 时也可用: 基于实测锚点做解析外推(不实测)")
    args = ap.parse_args()

    # chdir 到 vendor/DEIMv2 之前先把 config 路径固化, 否则相对路径会失效
    cfg_path = os.path.abspath(args.config)

    os.chdir(DEIM)
    sys.path.insert(0, DEIM)
    sys.path.insert(0, "/home/constant/CodeBuddy/Dev/AIC/scripts/lib")

    import torch
    from engine.core import YAMLConfig

    cfg = YAMLConfig(cfg_path)
    y = cfg.yaml_cfg
    tbs = y["train_dataloader"]["total_batch_size"]
    accum = int(y.get("accum_steps", 1))
    base_micro = max(1, tbs // accum)
    MB = 2**20
    size = _train_size(y)

    gpu_ok = torch.cuda.is_available()
    total_gb = None
    print("=" * 78)
    if gpu_ok:
        pdev = torch.cuda.get_device_properties(0)
        total_gb = pdev.total_memory / 2**30
        print("GPU        : %s  %.2f GiB" % (pdev.name, total_gb))
    else:
        print("GPU        : 不可用 (无 /dev/nvidia* 设备节点)")
    print("config     : %s" % os.path.basename(args.config))
    print("train size : %d x %d" % (size, size))
    print("tbs/accum  : %d / %d   ->  配置 micro = %d" % (tbs, accum, base_micro))
    print("=" * 78)

    model = cfg.model
    criterion = cfg.criterion
    optimizer = cfg.optimizer
    ema = None
    if not args.no_ema and getattr(cfg, "ema", None) is not None:
        ema = cfg.ema
    scaler = getattr(cfg, "scaler", None)
    if args.no_amp:
        scaler = None

    tot = sum(t.numel() for t in model.parameters())
    tr = sum(t.numel() for t in model.parameters() if t.requires_grad)
    # fp32 参数 + fp32 梯度(可训练) + AdamW(exp_avg,exp_avg_sq) + EMA 副本 + AMP bf16 cache
    static = (tot * 4 + tr * 4 + tr * 8) / MB
    if ema is not None:
        static += tot * 4 / MB
    if not args.no_amp:
        static += tot * 2 / MB
    print("静态占用   : %.0f MB  (总参数 %.1fM / 可训练 %.1fM%s%s)"
          % (static, tot / 1e6, tr / 1e6,
             " + EMA" if ema is not None else "",
             " + AMP" if not args.no_amp else ""))
    print("             = 参数%.0f + 梯度%.0f + AdamW%.0f%s%s"
          % (tot * 4 / MB, tr * 4 / MB, tr * 8 / MB,
             " + EMA%.0f" % (tot * 4 / MB) if ema is not None else "",
             " + AMPcache%.0f" % (tot * 2 / MB) if not args.no_amp else ""))

    units = _backbone_units(model, size)
    print("主干激活单位: %.2fM / 图  (tokens x embed_dim x layers, 三流合计)" % (units / 1e6))

    if not gpu_ok:
        if not args.estimate:
            print("\n[FATAL] torch.cuda 不可用 —— 无法实测。")
            print("        请在能看到 GPU 的终端运行, 或加 --estimate 做解析外推。")
            print("        (本会话的 /dev 被 tmpfs 覆盖 + seccomp 拦截 mknod, 无 nvidia 设备节点;")
            print("         用 `ssh 127.0.0.1` 拿到非沙箱 shell 即可见到 GPU, 见 gpu-access-in-sandbox skill)")
            return 2
        return _estimate(args, static, units, base_micro, tbs, total_gb)

    dev = torch.device("cuda")
    model.to(dev)
    criterion.to(dev)
    if ema is not None:
        ema.to(dev)
    # 实测静态基线: 公式估算会高估(AdamW 状态首轮未分配 / AMP cache 未满),
    # 导致「动态」出现负值。这里用真实 alloc 做基准。
    base_alloc = torch.cuda.memory_allocated() / MB
    print("实测静态   : %.0f MB  (模型+EMA+criterion 已上卡, 梯度/AdamW 未分配)" % base_alloc)

    # 真实 batch: 直接从训练集取, 保证 targets/通道数/尺寸都对
    dl = cfg.train_dataloader
    print("取真实训练 batch ...")
    it = iter(dl)
    samples, targets = next(it)
    if args.max_scale:
        for _ in range(6):
            s_, t_ = next(it)
            if s_.shape[-1] > samples.shape[-1]:
                samples, targets = s_, t_
    print("batch 实取 : %s  (配置开了多尺度, 尺寸是随机抽的 —— 峰值随尺寸平方变化)"
          % (tuple(samples.shape),))

    metas = dict(epoch=0, step=0, global_step=0, epoch_step=1)

    def one_step(bs):
        """用 bs 张图跑一次 forward+backward, 返回峰值(MB)。

        严格对齐 engine/solver/det_engine.py 的训练路径: bf16 autocast 下前向、
        关掉 autocast 算 loss、GradScaler 缩放反传。criterion 必须传 metas
        (DEIMCriterion 的 change_matcher 依赖 epoch), 且 loss_dict 只算一次。
        """
        s = samples[:bs].to(dev, non_blocking=True)
        t = [{k: v.to(dev) for k, v in tg.items()} for tg in targets[:bs]]
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        DT.append(t0)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                cache_enabled=True, enabled=not args.no_amp):
                out = model(s, targets=t)
            with torch.autocast(device_type="cuda", enabled=False):
                loss_dict = criterion(out, t, **metas)
            loss = sum(loss_dict.values())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return None, str(e).split(".")[0]
            raise
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if ema is not None:
            ema.update(model)
        torch.cuda.synchronize()
        DT[-1] = time.perf_counter() - DT[-1]
        peak = torch.cuda.max_memory_allocated() / MB
        return peak, None

    results = {}
    RT = {}   # micro -> 单步耗时中位数(秒)
    cands = args.micro if args.micro else [base_micro]
    if args.auto:
        cands = list(range(min(args.max_micro, base_micro), 0, -1))

    print("-" * 78)
    for bs in cands:
        if bs > samples.shape[0]:
            print("micro=%-2d  跳过 (取到的 batch 只有 %d 张)" % (bs, samples.shape[0]))
            continue
        peaks = []
        oom = None
        for i in range(args.iters):
            pk, err = one_step(bs)
            if pk is None:
                oom = err
                break
            peaks.append(pk)
        if oom:
            print("micro=%-2d  OOM  (%s)" % (bs, oom))
            results[bs] = None
            if args.auto:
                break
            continue
        steady = max(peaks[1:]) if len(peaks) > 1 else peaks[0]
        results[bs] = steady
        _dts = DT[-len(peaks):] if peaks else []
        if _dts:
            RT[bs] = sorted(_dts)[len(_dts) // 2]
        print("micro=%-2d  peak %7.0f MB  (激活+梯度增量 %7.0f MB)  iters=%s"
              % (bs, steady, steady - base_alloc,
                 [round(x) for x in peaks]))

    if not any(v is not None for v in results.values()):
        print("\n[结论] 所有档位都 OOM —— 需要降分辨率 / use_modality_stem / 退小 backbone")
        return 1

    print("=" * 78)
    ok = [(b, v) for b, v in results.items() if v is not None]
    best_bs, best_peak = max(ok, key=lambda kv: kv[0])
    free = total_gb * 1024 - torch.cuda.memory_reserved() / MB
    print("最大可行 micro = %d   峰值 %.0f MB (%.2f GB)" % (best_bs, best_peak, best_peak / 1024))
    print("GPU 总 %.2f GB, 训练进程已 reserve %.2f GB"
          % (total_gb, torch.cuda.memory_reserved() / MB / 1024))
    print("剩余余量 ≈ %.2f GB" % (free / 1024))
    if best_bs >= 1:
        print("建议: total_batch_size=%d, accum_steps=%d  (等效 batch 不变, micro=%d)"
              % (tbs, max(1, tbs // best_bs), best_bs))

    dt = RT.get(best_bs)
    if dt:
        try:
            n_ds = len(dl.dataset)
        except Exception:
            n_ds = 0
        ep = int(y.get("epoches") or 0)
        if n_ds and ep:
            micro_per_epoch = (n_ds + best_bs - 1) // best_bs
            secs = micro_per_epoch * ep * dt
            print("-" * 78)
            print("训练时长推算 (micro=%d, 训练集 %d 张 -> 每 epoch %d 个 micro step, %d epochs)"
                  % (best_bs, n_ds, micro_per_epoch, ep))
            print("  单步 forward+backward+step : %.3f s  (中位数, 已 synchronize)" % dt)
            print("  每 epoch                   : %.1f min" % (micro_per_epoch * dt / 60))
            print("  全程 %d epochs             : %.1f 小时" % (ep, secs / 3600))
            print("  (不含每个 epoch 结束后的验证/评测时间; 数据加载与以增强停顿未计入)")
    if best_bs < base_micro:
        print("  ↑ 配置里写的 micro=%d 跑不动, 必须按上面改动" % base_micro)
    return 0


if __name__ == "__main__":
    sys.exit(main())
