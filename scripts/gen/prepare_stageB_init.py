#!/usr/bin/env python
"""从 Stage A 的产物构造 Stage B 的起点: 保留 RGB 检测器, 重置融合通路。

Stage A 训练时 ir/depth_drop_prob=1.0, forward 里
    ir_feats = [torch.zeros_like(f) for f in ir_feats]
zeros_like 断开了 aux **主干** 的计算图, 但融合通路本身仍在被训练, 而且输入恒为零:

    ir_feats(全 0) -> ir_norms(BatchNorm2d) -> dt_ort -> prompt -> rgb_feats + prompt

结果:
  * ir_norms / depth_norms 的 running_mean -> 0, running_var -> 0
    (Stage B 一开始就喂真实特征, eval 模式下 BN 会除以 sqrt(eps) 放大到爆炸)
  * dt_ort / prompt 学到的是"零输入 -> 某个常量残差", 没有真实模态信息

所以对 Stage B 来说这批权重是废的, 留着只会拖慢收敛并让前几个 epoch 的 val 数字
失真。本脚本把它们换成**全新随机初始化**(按 Stage B 配置现场建一个随机模型取值,
保证与框架自身的初始化分布一致), 其余部分——RGB 主干、IR/Depth 主干、encoder、
decoder、sta、convs——全部沿用 Stage A 的成果。

重置范围 (前缀):
    backbone.ir_norms / backbone.depth_norms    aux 特征的 BatchNorm2d
    backbone.ir_aux_proj / backbone.depth_aux_proj  384 -> 1024 升维 1x1
    backbone.dt_ort                             IR<->Depth 正交去冗余
    backbone.prompt                             残差注入 RGB 的 PromptBlock

用法:
    python scripts/gen/prepare_stageB_init.py \
        [--src outputs/vitl800_scratch_stageA_rgb/last.pth] \
        [--dst ckpts/dinov3/stageB_init.pth]
    # --no-reset 则只做透传 (不重置, 用于对照实验)
"""
import argparse
import os
import sys

AIC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(AIC_ROOT, 'vendor', 'DEIMv2')
sys.path.insert(0, os.path.join(AIC_ROOT, 'scripts', 'lib'))
sys.path.insert(0, REPO)

import torch  # noqa: E402
from engine.core.yaml_config import YAMLConfig  # noqa: E402

CFG_B = os.path.join(REPO, 'configs', 'deimv2',
                     'deimv2_dinov3_l_aic_vitl800_scratch_stageB_mod.yml')

# 融合通路前缀 —— 这些在 Stage A 里是在零输入上训练出来的, 一律重置
FUSION_PREFIXES = (
    'backbone.ir_norms.',
    'backbone.depth_norms.',
    'backbone.ir_aux_proj.',
    'backbone.depth_aux_proj.',
    'backbone.dt_ort.',
    'backbone.prompt.',
)


def is_fusion(key):
    return any(key.startswith(p) for p in FUSION_PREFIXES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(
        REPO, 'outputs/vitl800_scratch_stageA_rgb/last.pth'))
    ap.add_argument('--dst', default=os.path.join(
        REPO, 'ckpts/dinov3/stageB_from_stageA.pth'))
    ap.add_argument('--no-reset', action='store_true',
                    help='透传, 不重置融合通路 (用于对照实验)')
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f'[ERROR] 找不到 Stage A 产物: {args.src}')
        return 1
    if not os.path.exists(CFG_B):
        print(f'[ERROR] 找不到 Stage B 配置: {CFG_B}')
        return 1

    print(f'src  : {args.src}')
    state = torch.load(args.src, map_location='cpu')
    print(f'   top-level keys = {list(state.keys())}')

    def model_sd_of(s):
        if 'model' in s:
            return s['model']
        if 'ema' in s:
            e = s['ema']
            return e.get('module', e)
        return s

    base = model_sd_of(state)
    print(f'   params = {sum(v.numel() for v in base.values())/1e6:.2f}M')

    if args.no_reset:
        out = dict(base)
        print('\n[--no-reset] 直接透传, 不改动任何权重')
    else:
        print('\n建随机模型以取得融合通路的初始分布 (weights_path=null -> 全随机)...')
        cfg = YAMLConfig(CFG_B)
        fresh = cfg.model.state_dict()
        print(f'   fresh model = {sum(v.numel() for v in fresh.values())/1e6:.2f}M')

        out = {}
        n_reset, n_reset_params, n_keep = 0, 0, 0
        for k, v in base.items():
            if is_fusion(k):
                if k in fresh and fresh[k].shape == v.shape:
                    out[k] = fresh[k].clone()
                    n_reset += 1
                    n_reset_params += v.numel()
                else:
                    print(f'   [warn] fresh 里没有 {k} (shape {tuple(v.shape)}), 保持原值')
                    out[k] = v
            else:
                out[k] = v
                n_keep += 1
        print(f'\n   重置 {n_reset} keys / {n_reset_params/1e6:.2f}M  (融合通路)')
        print(f'   保留 {n_keep} keys / '
              f'{sum(v.numel() for k, v in out.items() if not is_fusion(k))/1e6:.2f}M'
              f'  (RGB 检测器 + 三条主干)')

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    # `-t` 走 load_tuning_state, 读的是 state['model'] —— 必须有这层外壳
    torch.save({'model': out}, args.dst)
    size = os.path.getsize(args.dst) / 1e6
    print(f'\n写出: {args.dst}  ({size:.1f} MB)')
    print('VERDICT: OK — Stage B 起点就绪 (RGB 检测器来自 Stage A, 融合通路全新随机)'
          if not args.no_reset else 'VERDICT: pass-through')
    return 0


if __name__ == '__main__':
    sys.exit(main())
