"""校验三阶段训练配置 + Stage A 专属 val 配置的解析结果。

重点确认:
  1. Stage A 训练 *和* 评测都把 aux 恒零 (drop=1.0), 否则 eval 时真实 IR 特征
     会撞进 running_var≈0 的 BatchNorm2d。
  2. Stage B 的 unfreeze=0 (RGB 全冻结), Stage C 的 unfreeze=6 + llrd 2e-5。
  3. 三个阶段的 epoch / output_dir 各不相同, 不会互相覆盖 checkpoint。
"""
import os
import sys

AIC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(AIC_ROOT, 'vendor', 'DEIMv2')
sys.path.insert(0, os.path.join(AIC_ROOT, 'scripts', 'lib'))
sys.path.insert(0, REPO)

from engine.core.yaml_config import YAMLConfig  # noqa: E402

CFGS = [
    ('A train', 'deimv2_dinov3_l_aic_vitl800_scratch_stageA_rgb.yml'),
    ('A val  ', 'deimv2_dinov3_l_aic_vitl800_scratch_stageA_rgb_val_orig.yml'),
    ('A val n', 'deimv2_dinov3_l_aic_vitl800_scratch_stageA_rgb_val_night01.yml'),
    ('B train', 'deimv2_dinov3_l_aic_vitl800_scratch_stageB_mod.yml'),
    ('C train', 'deimv2_dinov3_l_aic_vitl800_scratch_stageC_fuse.yml'),
]


def get(d, k, default=None):
    """yaml_cfg 可能是 dict 也可能是 attribute-style 对象, 两种都支持。"""
    if isinstance(d, dict):
        return d.get(k, default)
    return getattr(d, k, default)


def load(fn):
    return YAMLConfig(os.path.join(REPO, 'configs', 'deimv2', fn)).yaml_cfg


def main():
    for tag, fn in CFGS:
        path = os.path.join(REPO, 'configs', 'deimv2', fn)
        if not os.path.exists(path):
            print(f'{tag}  MISSING {fn}')
            continue
        c = load(fn)
        b = get(c, 'DINOv3STAsRDTTrack')
        llrd = get(b, 'llrd_top_lr', None)
        llrd_s = 'n/a' if llrd is None else f'{llrd:g}'
        print(f'{tag}  {fn[24:]:52s} ir_drop={get(b, "ir_drop_prob"):<5} '
              f'depth_drop={get(b, "depth_drop_prob"):<5} '
              f'unfreeze={get(b, "unfreeze_rgb_stages")} '
              f'llrd_top={llrd_s:<8} ep={get(c, "epoches")} '
              f'out={os.path.basename(str(get(c, "output_dir")))}')

    # Stage A 的一致性: 训练与评测必须同为 1.0
    print()
    ta = get(load(CFGS[0][1]), 'DINOv3STAsRDTTrack')
    tv = get(load(CFGS[1][1]), 'DINOv3STAsRDTTrack')
    ta_ir, ta_dp = get(ta, 'ir_drop_prob'), get(ta, 'depth_drop_prob')
    tv_ir, tv_dp = get(tv, 'ir_drop_prob'), get(tv, 'depth_drop_prob')
    ok = (ta_ir >= 1.0 and ta_dp >= 1.0 and tv_ir >= 1.0 and tv_dp >= 1.0)
    print('Stage A train/eval 口径一致 (aux 恒零):', 'OK' if ok else 'MISMATCH -> 评测会崩')
    print('VERDICT:', 'OK' if ok else 'PROBLEM')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
