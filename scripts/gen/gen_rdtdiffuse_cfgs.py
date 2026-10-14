"""Generate all configs for the hybrid backbone (train / LLVIP eval / IR ablation)."""
import os

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
AIC = "/home/constant/CodeBuddy/Dev/AIC"

JOBS = [
    # (source, output filename, output_dir, extra ir/depth replacement?)
    ("deimv2_dinov3_l_aic_rdttrack_heat.yml",
     "deimv2_dinov3_l_aic_rdtdiffuse.yml",
     "deimv2_dinov3_l_aic_rdtdiffuse", None),

    ("deimv2_dinov3_l_aic_rdttrack_heat_llvip.yml",
     "deimv2_dinov3_l_aic_rdtdiffuse_llvip.yml",
     "deimv2_dinov3_l_aic_rdtdiffuse_llvip", None),

    # AIC-val IR ablation: same as training config but val IR -> all black
    ("deimv2_dinov3_l_aic_rdttrack_heat_val_noir.yml",
     "deimv2_dinov3_l_aic_rdtdiffuse_val_noir.yml",
     "deimv2_dinov3_l_aic_rdtdiffuse_val_noir", None),
]


def convert(text, out_dir):
    text = text.replace("DINOv3STAsRDTTrack", "DINOv3STAsRDTDiffuse")
    text = text.replace("  unfreeze_rgb_stages: 0\n",
                        "  unfreeze_rgb_stages: 0\n  inject: diffuse\n")
    # replace any rdttrack_heat* output_dir with the new one
    for old in ("deimv2_dinov3_l_aic_rdttrack_heat_llvip",
                "deimv2_dinov3_l_aic_rdttrack_heat_val_noir",
                "deimv2_dinov3_l_aic_rdttrack_heat"):
        text = text.replace("output_dir: ./outputs/" + old,
                            "output_dir: ./outputs/" + out_dir)
    return text


for src, dst, out_dir, _ in JOBS:
    text = open(os.path.join(CFG, src)).read()
    open(os.path.join(CFG, dst), "w").write(convert(text, out_dir))
    print("wrote", dst, "->", out_dir)
    for ln in convert(text, out_dir).splitlines():
        if ln.strip().startswith(("output_dir:", "backbone:", "inject:")):
            print("     ", ln.strip())
