#!/usr/bin/env python
"""为 Stage A (RGB-only) 派生专属的 10 档验证配置。

为什么 Stage A 不能直接用现成的 val 配置:
    val 配置里 ir_drop_prob/depth_drop_prob = 0.15, 而
        _zero_stream(training, prob):
            if prob >= 1.0: return True          # eval 也恒零
            return training and ...              # eval 时恒 False
    所以 eval 时 aux 是**真实特征**, 会流进 ir_norms / depth_norms (BatchNorm2d)。
    而 Stage A 训练了 40 个 epoch 的零输入, 这两个 BN 的 running_mean -> 0,
    running_var -> 0。eval 模式下 BN 用 running 统计:
        y = (x - 0) / sqrt(0 + eps) * w + b
    真实 IR 特征会被放大到天文数字, 再经 prompt 残差加到 RGB 特征上 -> 评测直接崩。

    Stage A 专属 val 把 drop 设成 1.0 -> eval 也恒零 -> 与训练时看到的完全一致,
    这才是"纯 RGB 检测器"的正确评测口径。

Stage B/C 不需要专属 val: 它们在训练时就用真实模态, eval 时 BN running 统计是
正常拟合的, 沿用原 val 配置 (drop=0.15 在 eval 下不生效) 即可。
"""
import glob
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
CFGDIR = os.path.join(ROOT, "vendor/DEIMv2/configs/deimv2")
PREFIX = "deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_"
OUT_PREFIX = "deimv2_dinov3_l_aic_vitl800_scratch_stageA_rgb_val_"
OUT_ROOT = "./outputs/vitl800_scratch_stageA_rgb_eval"

HEADER = (
    "# =========================================================================\n"
    "# Stage A (RGB-only) 专属验证配置 —— 由 scripts/gen/gen_stage_val_configs.py 派生\n"
    "# 唯一改动: ir/depth_drop_prob 0.15 -> 1.0 (eval 时同样恒零 aux 流)\n"
    "# 原因见脚本顶部注释: 否则真实 IR 特征会撞进 running_var≈0 的 BatchNorm2d 而崩掉\n"
    "# =========================================================================\n"
)


def main():
    srcs = sorted(glob.glob(os.path.join(CFGDIR, PREFIX + "*.yml")))
    if not srcs:
        print("no source val configs found")
        return
    for src in srcs:
        lv = os.path.basename(src)[len(PREFIX): -len(".yml")]
        with open(src) as f:
            text = f.read()
        out, touched = [], set()
        for ln in text.splitlines():
            s = ln.strip()
            indent = ln[: len(ln) - len(ln.lstrip())]
            if s.startswith("ir_drop_prob:"):
                out.append(f"{indent}ir_drop_prob: 1.0        # Stage A: eval 也恒零 aux")
                touched.add("ir")
                continue
            if s.startswith("depth_drop_prob:"):
                out.append(f"{indent}depth_drop_prob: 1.0     # Stage A: eval 也恒零 aux")
                touched.add("depth")
                continue
            if s.startswith("output_dir:") and "output_dir" not in touched:
                out.append(f"{indent}output_dir: {OUT_ROOT}/{lv}")
                touched.add("output_dir")
                continue
            out.append(ln)
        dst = os.path.join(CFGDIR, OUT_PREFIX + lv + ".yml")
        with open(dst, "w") as f:
            f.write(HEADER)
            f.write("\n".join(out) + "\n")
        ok = touched >= {"ir", "depth", "output_dir"}
        print(f"  {os.path.basename(dst):70s} {'OK' if ok else 'MISSING ' + str(touched)}")
    print(f"\n生成 {len(srcs)} 个 Stage A 专属 val 配置 -> {OUT_ROOT}/<level>")


if __name__ == "__main__":
    main()
