#!/usr/bin/env python3
"""损失地形可视化：Filter-Normalized 2D 切片 + 训练轨迹投影 + 路径插值。

两种切片方向（`--dir-mode`）：
  random —— Li et al. 2018 标准做法。两个随机高斯方向，**逐 filter 归一化**
            （每个输出通道的随机方向缩放到与对应权重同范数，避免高维尺度被大层淹没）。
            它回答的是「θ* 局部有多陡」。⚠ 真实训练轨迹在这张图里几乎不可见
            （384M 维里任意 2 维只能捕获 ~1e-6 的位移），这是高维必然，不是 bug。
  pca    —— 以轨迹自身张成的平面：e1 = 走得最远的那个 ckpt 的位移方向，
            e2 = 对 e1 正交化后残差最大的那个位移方向。这样「路径点」真落在图上，
            回答的是「模型走到了哪、那块地方平不平」。

两轴单位：方向整体缩放到 边缘(α=±1) 位移 = rel·||θ||，于是坐标可直接读成
「相对训练实际走过的距离」。rel 默认 0.05。

工程约束（都在代码里，别删）：
  * BN 全部 eval —— 否则每次前向都改 running stats，地形自己会漂；
  * 每次前向重设随机种子 —— 否则 DN(去噪) 分支的随机 query 会让同一格两次不一样；
  * 参数在 CPU 上做叠加（单次 1.5GB H2D 传输），而不是每格在 GPU 上分配临时张量。

用法：
  干跑（只算几何，不需要 GPU / 数据集）:
    python scripts/analysis/loss_landscape.py --config <yml> --base <ckpt> \
        --traj <ckpt...> --dry-run --json outputs/landscape_geom.json
  完整:
    python scripts/analysis/loss_landscape.py --config <yml> --base <best.pth> \
        --traj a.pth b.pth c.pth --grid 9 --rel 0.05 --batch 4 --dir-mode both \
        --out-png outputs/loss_landscape.png --json outputs/loss_landscape.json
"""
import argparse
import gc
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "vendor", "DEIMv2"))
# engine/data/transforms 会 import nightify_aic，缺了直接 ModuleNotFoundError
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))

import torch  # noqa: E402


def resolve(p):
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    for c in (p, os.path.join(ROOT, p), os.path.join(ROOT, "vendor", "DEIMv2", p)):
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(p)


def flat_norm(d):
    s = 0.0
    for v in d.values():
        s += float(v.detach().float().pow(2).sum())
    return s ** 0.5


def flat_dot(a, b, keys):
    s = 0.0
    for k in keys:
        if k in b:
            s += float((a[k].float() * b[k].float()).sum())
    return s


def load_model_sd(path):
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = {k: v for k, v in d["model"].items()}
    le = d.get("last_epoch", None)
    del d
    gc.collect()
    return sd, le


def filter_normalized_dir(params, seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, p in params:
        w = p.detach().float().cpu()  # 拉到 cpu，避免与下方 randn(CPU) 做 norm 时设备不一致
        d = torch.randn(w.shape, generator=g, dtype=torch.float32)
        if w.dim() > 1:
            wf = w.view(w.shape[0], -1)
            df = d.view(d.shape[0], -1)
            scale = wf.norm(dim=1, keepdim=True) / (df.norm(dim=1, keepdim=True) + 1e-12)
            d = (df * scale).view(w.shape)
        else:
            d = d * (w.norm() / (d.norm() + 1e-12))
        out[n] = d
    return out


def scale_dir(d, keys, target):
    n = flat_norm(d)
    s = target / n if n > 0 else 1.0
    for k in keys:
        d[k] = d[k] * s
    return s


def orth(d2, d1, keys):
    c = flat_dot(d1, d2, keys) / flat_dot(d1, d1, keys)
    for k in keys:
        d2[k] = d2[k] - c * d1[k]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True, help="地形中心 θ*（建议 best checkpoint）")
    ap.add_argument("--traj", nargs="*", default=[], help="轨迹 checkpoint，按时间顺序")
    ap.add_argument("--interp", default=None, help="与 base 做直线插值的另一端")
    ap.add_argument("--dir-mode", choices=["random", "pca", "both"], default="both")
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--rel", type=float, default=0.05, help="边缘位移 = rel·||θ||")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--loader", choices=["val", "train"], default="val")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest-plot", action="store_true",
                    help="用合成数据只测 plot()，不建模型（验证画图管线）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out-png", default=None)
    a = ap.parse_args()

    if a.selftest_plot:
        import math
        g = [-1.0 + 2.0 * i / (a.grid - 1) for i in range(a.grid)]
        Z = [[(x * x + 0.4 * y * y + 0.1 * math.cos(3 * x)) for x in g] for y in g]
        traj = [dict(epoch=e, rel_move=r, loss=l,
                     **{m: dict(alpha=al, beta=be) for m in ("random", "pca")})
                for e, r, l, al, be in [(4, 0.0246, 3.0, 0.0, 0.0), (14, 0.0138, 1.2, 0.0, 0.0),
                                        (24, 0.0167, 2.4, 0.0, 0.0), (28, 0.0196, 3.4, 0.0, 0.0)]]
        res = dict(base="selftest.pth", rel=a.rel, center_loss={"random": 0.8, "pca": 0.8},
                   landscapes={"random": dict(grid=g, Z=Z), "pca": dict(grid=g, Z=Z)},
                   traj=traj, interp=dict(to="last.pth", t=[i / 10 for i in range(11)],
                                          loss=[0.8 + i * 0.05 for i in range(11)]))
        plot(res, resolve(a.out_png or "outputs/figures/_landscape_plot_selftest.png"))
        return

    from engine.core import YAMLConfig
    from engine.misc import dist_utils

    dist_utils.setup_distributed(0, "builtin", seed=0)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg_path, base_path = resolve(a.config), resolve(a.base)
    print("device = %s\nconfig = %s\nbase   = %s" % (device, cfg_path, base_path))

    print("\n[1/7] 构建模型/准则 ...")
    cfg = YAMLConfig(cfg_path)
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    model.eval()
    criterion.eval()

    print("\n[2/7] 载入地形中心 ...")
    sd, _ = load_model_sd(base_path)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        print("   [warn] missing=%d unexpected=%d" % (len(miss), len(unexp)))
    del sd
    gc.collect()

    params = [(n, p) for n, p in model.named_parameters()]
    keys = [n for n, _ in params]
    base = {n: p.detach().float().cpu().clone() for n, p in params}
    buf = {n: torch.empty_like(base[n]) for n in keys}
    th = flat_norm(base)
    print("   ||θ*|| = %.4f（仅参数，不含 buffer）" % th)

    # ---- 轨迹清单 ----
    traj_meta = []
    for p in a.traj:
        pp = resolve(p)
        if os.path.exists(pp):
            traj_meta.append(pp)
        else:
            print("   跳过（不存在）%s" % p)

    # ---- PCA 方向：先找走得最远的一条位移作为 e1 ----
    print("\n[3/7] 构造切片方向 ...")
    dirs = {}
    if a.dir_mode in ("random", "both"):
        d1 = filter_normalized_dir(params, a.seed)
        d2 = filter_normalized_dir(params, a.seed + 1)
        scale_dir(d1, keys, a.rel * th)
        scale_dir(d2, keys, a.rel * th)
        orth(d2, d1, keys)
        dirs["random"] = (d1, d2)

    if a.dir_mode in ("pca", "both"):
        far, far_n = None, -1.0
        for pp in traj_meta:
            sd, _ = load_model_sd(pp)
            dd = {k: sd[k].float() - base[k] for k in keys if k in sd}
            n = flat_norm(dd)
            if n > far_n:
                far, far_n = pp, n
            del sd, dd
            gc.collect()
        if far is None:
            print("   [warn] pca 模式需要 --traj，回退到 random")
            d1 = filter_normalized_dir(params, a.seed)
            d2 = filter_normalized_dir(params, a.seed + 1)
            scale_dir(d1, keys, a.rel * th)
            scale_dir(d2, keys, a.rel * th)
            orth(d2, d1, keys)
            dirs["pca"] = (d1, d2)
        else:
            sd, _ = load_model_sd(far)
            e1 = {k: sd[k].float() - base[k] for k in keys if k in sd}
            del sd
            gc.collect()
            # e2 = 对 e1 正交化后残差最大的那条位移
            best, best_r, best_p = None, -1.0, None
            e1sq = flat_dot(e1, e1, keys)
            for pp in traj_meta:
                sd, _ = load_model_sd(pp)
                dd = {k: sd[k].float() - base[k] for k in keys if k in sd}
                r2 = flat_norm(dd) ** 2 - flat_dot(dd, e1, keys) ** 2 / e1sq
                if r2 > best_r:
                    best_r, best_p = r2, pp
                del sd, dd
                gc.collect()
            sd, _ = load_model_sd(best_p)
            e2 = {k: sd[k].float() - base[k] for k in keys if k in sd}
            del sd
            gc.collect()
            orth(e2, e1, keys)
            scale_dir(e1, keys, a.rel * th)
            scale_dir(e2, keys, a.rel * th)
            dirs["pca"] = (e1, e2)
            print("   pca 基: e1 ← %s（走得最远）; e2 ← %s（正交后残差最大）"
                  % (os.path.basename(far), os.path.basename(best_p)))

    d1sq = {m: flat_dot(d[0], d[0], keys) for m, d in dirs.items()}
    d2sq = {m: flat_dot(d[1], d[1], keys) for m, d in dirs.items()}
    for m, (u, v) in dirs.items():
        print("   [%s] ||δ||=%.3f ||η||=%.3f  正交性 ⟨δ,η⟩/(||δ||||η||)=%+.4f"
              % (m, flat_norm(u), flat_norm(v),
                 flat_dot(u, v, keys) / (flat_norm(u) * flat_norm(v))))

    # ---- 轨迹投影 ----
    print("\n[4/7] 轨迹投影 ...")
    proj = {m: [] for m in dirs}
    for pp in traj_meta:
        sd, le = load_model_sd(pp)
        dd = {k: sd[k].float() - base[k] for k in keys if k in sd}
        nrm = flat_norm(dd)
        rec = dict(file=os.path.basename(pp), path=pp, epoch=le, move_norm=nrm, rel_move=nrm / th)
        for m, (u, v) in dirs.items():
            al = flat_dot(dd, u, keys) / d1sq[m]
            be = flat_dot(dd, v, keys) / d2sq[m]
            resid2 = max(nrm ** 2 - al ** 2 * d1sq[m] - be ** 2 * d2sq[m], 0.0)
            rec[m] = dict(alpha=al, beta=be, off_plane=resid2 ** 0.5,
                          off_plane_frac=(resid2 ** 0.5) / nrm if nrm > 0 else 0.0)
        for m in dirs:
            proj[m].append(rec)
        print("   ep%-3s ||Δθ||/||θ||=%.4f  " % (le, nrm / th) +
              "  ".join("[%s] a=%+.4f b=%+.4f 平面外 %.1f%%"
                        % (m, rec[m]["alpha"], rec[m]["beta"], 100 * rec[m]["off_plane_frac"])
                        for m in dirs))
        del sd, dd
        gc.collect()

    if a.dry_run:
        out = dict(base=os.path.basename(base_path), theta_norm=th, rel=a.rel,
                   modes={m: dict(dir_norm=[flat_norm(dirs[m][0]), flat_norm(dirs[m][1])]) for m in dirs},
                   traj=proj)
        if a.json:
            jp = resolve(a.json)
            os.makedirs(os.path.dirname(jp), exist_ok=True)
            json.dump(out, open(jp, "w"), indent=1)
            print("\n[干跑] 未做任何前向。写出 %s" % jp)
        return

    # ---- 固定 batch ----
    print("\n[5/7] 取固定 mini-batch (%s loader, bs=%d) ..." % (a.loader, a.batch))
    loader = cfg.val_dataloader if a.loader == "val" else cfg.train_dataloader
    it = iter(loader)
    samples, targets = next(it)
    del it
    samples = samples[:a.batch].to(device)
    targets = [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()}
               for t in targets[:a.batch]]
    print("   samples %s  n_gt=%d" % (tuple(samples.shape), sum(len(t["labels"]) for t in targets)))
    metas = dict(epoch=0, step=0, global_step=0, epoch_step=max(len(loader), 1))

    import torch.nn as nn
    n_bn = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            n_bn += 1
    print("   BN %d 个已切 eval（防地形自漂）" % n_bn)

    amp = (not a.no_amp) and device.type == "cuda"

    def eval_loss():
        torch.manual_seed(a.seed)
        model.train()
        criterion.train()
        with torch.no_grad():
            if amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(samples, targets=targets)
                with torch.autocast("cuda", enabled=False):
                    ld = criterion(out, targets, **metas)
            else:
                out = model(samples, targets=targets)
                ld = criterion(out, targets, **metas)
        v = float(sum(x.item() for x in ld.values()))
        del out, ld
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return v

    def set_params(al, be, d1, d2):
        for n, p in params:
            torch.add(base[n], d1[n], alpha=al, out=buf[n])
            buf[n].add_(d2[n], alpha=be)
            p.data.copy_(buf[n])

    grid = [-1.0 + 2.0 * i / (a.grid - 1) for i in range(a.grid)]
    results = {}
    for m, (u, v) in dirs.items():
        print("\n[6/7] 扫网格 [%s] %dx%d ..." % (m, a.grid, a.grid))
        Z = []
        for be in grid:
            row = []
            for al in grid:
                set_params(al, be, u, v)
                row.append(eval_loss())
            Z.append(row)
            print("   β=%+.2f  loss %.4f .. %.4f" % (be, min(row), max(row)))
            sys.stdout.flush()
        results[m] = dict(grid=grid, Z=Z)

    # ---- 轨迹点自身 loss + 插值势垒 ----
    print("\n[7/7] 路径点 loss / 插值势垒 ...")
    for rec in proj[list(dirs)[0]]:
        sd, _ = load_model_sd(rec["path"])
        for n, p in params:
            if n in sd:
                p.data.copy_(sd[n].float().to(device))
        rec["loss"] = eval_loss()
        print("   ep%-3s loss=%.4f" % (rec["epoch"], rec["loss"]))
        del sd
        gc.collect()

    interp = None
    if a.interp:
        ip = resolve(a.interp)
        sd2, _ = load_model_sd(ip)
        ts = [i / 10.0 for i in range(11)]
        interp = dict(to=os.path.basename(ip), t=ts, loss=[])
        for tv in ts:
            for n, p in params:
                if n in sd2:
                    torch.lerp(base[n], sd2[n].float(), tv, out=buf[n])
                    p.data.copy_(buf[n])
            interp["loss"].append(eval_loss())
            print("   t=%.1f loss=%.4f" % (tv, interp["loss"][-1]))
        del sd2
        gc.collect()

    result = dict(base=os.path.basename(base_path), theta_norm=th, rel=a.rel,
                  batch=a.batch, loader=a.loader, landscapes=results,
                  traj=proj, interp=interp,
                  center_loss={m: results[m]["Z"][a.grid // 2][a.grid // 2] for m in results})
    if a.json:
        jp = resolve(a.json)
        os.makedirs(os.path.dirname(jp), exist_ok=True)
        json.dump(result, open(jp, "w"), indent=1)
        print("写出 %s" % jp)
    if a.out_png:
        plot(result, resolve(a.out_png))


def plot(res, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    modes = list(res["landscapes"].keys())
    ncol = len(modes) + 1 + (1 if res.get("interp") else 0)
    fig, ax = plt.subplots(1, ncol, figsize=(6.4 * ncol, 5.9), squeeze=False)
    ax = ax[0]
    titles = {"random": "随机方向切片（Li et al.）\n回答：θ* 局部有多陡",
              "pca": "轨迹张成的平面\n回答：模型走到了哪、那块平不平"}
    for i, m in enumerate(modes):
        L = res["landscapes"][m]
        g, Z = L["grid"], L["Z"]
        cs = ax[i].contourf(g, g, Z, levels=18, cmap="viridis")
        ax[i].contour(g, g, Z, levels=18, colors="w", linewidths=0.35, alpha=0.55)
        fig.colorbar(cs, ax=ax[i], label="train loss")
        ax[i].plot(0, 0, "w*", ms=17)
        ax[i].annotate("θ*", (0, 0), textcoords="offset points", xytext=(7, -12), color="w", fontsize=10)
        xs = [t[m]["alpha"] for t in res["traj"]]
        ys = [t[m]["beta"] for t in res["traj"]]
        ax[i].plot(xs, ys, "-o", color="crimson", lw=1.8, ms=6)
        for x, y, t in zip(xs, ys, res["traj"]):
            ax[i].annotate("ep%s L=%.2f" % (t["epoch"], t.get("loss", float("nan"))),
                           (x, y), textcoords="offset points", xytext=(7, 5),
                           color="w", fontsize=7.5)
        ax[i].set_xlabel("α   (边缘 = %.3f·||θ||)" % res["rel"])
        ax[i].set_ylabel("β")
        ax[i].set_title(r"$loss\ landscape$ — " + titles.get(m, m), fontsize=10)
    # —— 盆地剖面：loss vs 离 θ* 的距离（最直接回答「是不是宽平盆地」）——
    j = len(modes)
    m0 = modes[0]
    pts = [(0.0, res["center_loss"][m0], "θ*")]
    for t in res["traj"]:
        if "loss" in t:
            pts.append((t["rel_move"], t["loss"], "ep%s" % t["epoch"]))
    ax[j].plot([p[0] for p in pts], [p[1] for p in pts], "-o", color="seagreen")
    for x, y, lab in pts:
        ax[j].annotate(lab, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax[j].set_xlabel("||θ − θ*|| / ||θ*||")
    ax[j].set_ylabel("train loss (fixed batch)")
    ax[j].set_title("盆地剖面：走多远，loss 变了多少", fontsize=10)
    ax[j].grid(alpha=0.3)

    if res.get("interp"):
        j = len(modes) + 1
        ax[j].plot(res["interp"]["t"], res["interp"]["loss"], "-o", color="darkorange")
        ax[j].set_xlabel("t   (0 = θ*, 1 = %s)" % res["interp"]["to"])
        ax[j].set_ylabel("train loss")
        ax[j].set_title("沿直线插值的势垒", fontsize=10)
        ax[j].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print("写出 %s" % png)


if __name__ == "__main__":
    main()
