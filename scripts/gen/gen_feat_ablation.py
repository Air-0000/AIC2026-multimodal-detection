"""Feature-space modality ablation configs.

Why this exists
---------------
Every IR/depth ablation run so far zeroed the *input pixels* (pointing the
dataset at an all-black IR/depth folder). That is not a valid ablation on these
backbones: the aux streams are conv+BatchNorm stacks, so a constant image exits
the stem as a large out-of-distribution activation rather than as "no signal".
Measured on the 30ep RDTDiffuse checkpoint with IR pixels blacked:

    ||ir_feat|| with real heat-IR = 4451   ->   8060 with black pixels

i.e. the "ablated" branch carries *more* energy than the real one, and the
reported IR contribution collapses to ~1%.

`_zero_stream()` in multimodal_backbones.py makes `*_drop_prob >= 1.0` also fire
in eval, which zeroes the *feature* -- exactly what the model sees whenever
modality dropout fires during training. That is the honest ablation switch.
"""
import os

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"

# src config -> output_dir to patch -> variants to emit
JOBS = [
    ("deimv2_dinov3_l_aic_rdttrack_heat.yml",
     "./outputs/deimv2_dinov3_l_aic_rdttrack_heat",
     ["ir", "depth", "ir+depth"]),
    ("deimv2_dinov3_l_aic_diffuse_4ch.yml",
     "./outputs/deimv2_dinov3_l_aic_diffuse_4ch",
     ["ir"]),
    ("deimv2_dinov3_l_aic_rdtdiffuse_30ep.yml",
     "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_30ep",
     ["ir", "depth", "ir+depth"]),
    ("deimv2_dinov3_l_aic_rdtdiffuse.yml",
     "./outputs/deimv2_dinov3_l_aic_rdtdiffuse",
     ["ir", "depth", "ir+depth"]),
    ("deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt.yml",
     "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt",
     ["ir", "depth", "ir+depth"]),
    ("deimv2_dinov3_l_aic_rdtdiffuse_30ep_both.yml",
     "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_30ep_both",
     ["ir", "depth", "ir+depth"]),
    # LLVIP (OOD): depth is a null folder there, so "ir+depth" == "no aux".
    ("deimv2_dinov3_l_aic_rdtdiffuse_llvip.yml",
     "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_llvip",
     ["ir", "ir+depth"]),
    ("deimv2_dinov3_l_aic_rdttrack_heat_llvip.yml",
     "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_llvip",
     ["ir", "ir+depth"]),
    ("deimv2_dinov3_l_aic_diffuse_4ch_llvip.yml",
     "./outputs/deimv2_dinov3_l_aic_diffuse_4ch_llvip",
     ["ir"]),
    # FLIR (OOD, test-only): same recipes, FLIR thermal driving set.
    ("deimv2_dinov3_l_aic_rdtdiffuse_flir.yml",
     "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_flir",
     ["ir", "depth", "ir+depth"]),
    ("deimv2_dinov3_l_aic_rdttrack_heat_flir.yml",
     "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_flir",
     ["ir", "depth", "ir+depth"]),
    ("deimv2_dinov3_l_aic_diffuse_4ch_flir.yml",
     "./outputs/deimv2_dinov3_l_aic_diffuse_4ch_flir",
     ["ir"]),
    # FLIR for the 30ep inject A/B (prompt / both) -- the headline experiment.
    ("deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt_flir.yml",
     "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt_flir",
     ["ir", "depth", "ir+depth"]),
    ("deimv2_dinov3_l_aic_rdtdiffuse_30ep_both_flir.yml",
     "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_30ep_both_flir",
     ["ir", "depth", "ir+depth"]),
]

SUFFIX = {"ir": "featnoir", "depth": "featnodepth", "ir+depth": "featnoaux"}

for src, out_dir, variants in JOBS:
    text = open(os.path.join(CFG, src)).read()
    for v in variants:
        t = text
        if "ir" in v:
            assert "  ir_drop_prob: 0.15" in t, src
            t = t.replace("  ir_drop_prob: 0.15", "  ir_drop_prob: 1.0")
        if "depth" in v:
            assert "  depth_drop_prob: 0.15" in t, src
            t = t.replace("  depth_drop_prob: 0.15", "  depth_drop_prob: 1.0")
        dst = src.replace(".yml", f"_val_{SUFFIX[v]}.yml")
        t = t.replace(f"output_dir: {out_dir}",
                      f"output_dir: {out_dir}__{SUFFIX[v]}")
        open(os.path.join(CFG, dst), "w").write(t)
        print("wrote", dst)
