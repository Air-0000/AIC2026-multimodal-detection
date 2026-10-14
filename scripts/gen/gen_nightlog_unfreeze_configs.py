"""Derive deeper-unfreeze variants of the nightlog recipe.

nightlog (unfreeze_rgb_stages=2) gave the current best night curve (MEAN 0.4285).
The RGB-only baseline trains the WHOLE backbone and reaches 0.525 in-domain vs
our 0.475, and unfreeze 0->2 was the single biggest lever so far (+0.049), so the
depth is the natural next knob. DINOv3 ViT-S/16 has 12 blocks.

The catch is the LR: nightlog uses backbone lr 5e-5, tuned for 2 blocks. At 12
blocks that is very likely to wash out the DINOv3 prior -- the RGB baseline,
which unfreezes everything, uses 1.25e-5. So two controlled variants:

    E6  unfreeze_rgb_stages: 6   backbone lr 5e-5     <- clean depth-only delta
    E7  unfreeze_rgb_stages: 12  backbone lr 2.5e-5   <- full depth, gentler LR
"""
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
BASE = "deimv2_dinov3_l_aic_rdttrack_heat_night_log.yml"
OUT_BASE = "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log"

base = open(os.path.join(CFG, BASE)).read()
assert base.count("  unfreeze_rgb_stages: 2") == 1
assert base.count("      lr: 0.00005") == 2, base.count("      lr: 0.00005")

VARIANTS = [
    ("uf6", 6, 0.00005),
    ("uf12", 12, 0.000025),
]

for tag, n, lr in VARIANTS:
    txt = base.replace("./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log",
                       f"{OUT_BASE}_{tag}")
    txt = txt.replace("  unfreeze_rgb_stages: 2", f"  unfreeze_rgb_stages: {n}")
    # MUST be plain decimal: PyYAML parses "5e-05" as the STRING '5e-05'
    # (its float resolver needs a '.' in the mantissa), which then blows up in
    # LinearWarmup with "can't multiply sequence by non-int of type 'float'".
    txt = txt.replace("      lr: 0.00005", f"      lr: {lr:.8f}")
    dst = os.path.join(CFG, f"deimv2_dinov3_l_aic_rdttrack_heat_night_log_{tag}.yml")
    open(dst, "w").write(txt)

    c = yaml.safe_load(open(dst))
    bb = c["DINOv3STAsRDTTrack"]
    lrs = [p["lr"] for p in c["optimizer"]["params"] if "lr" in p]
    assert all(isinstance(v, float) for v in lrs), f"non-float LR parsed: {lrs}"
    print(f"{os.path.basename(dst):<58} unfreeze={bb['unfreeze_rgb_stages']:<3} "
          f"backbone_lr={lrs[0]}  out={c['output_dir']}")
print("OK")
