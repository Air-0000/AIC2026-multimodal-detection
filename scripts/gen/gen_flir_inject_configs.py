"""Derive FLIR (OOD test) eval configs for the 30ep inject A/B variants (prompt / both).

The 30ep prompt/both training configs (deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt.yml /
_both.yml) already define the model (inject=prompt / both). We only need to swap
their AIC val_dataloader for the FLIR 5ch test block (rgb + heat residual + null
depth, instances_flir_val.json). The FLIR val block is taken verbatim from
deimv2_dinov3_l_aic_rdtdiffuse_flir.yml so it stays in sync.
"""
import os

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
FLIR_VAL = open(os.path.join(CFG, "deimv2_dinov3_l_aic_rdtdiffuse_flir.yml")).read() \
    .split("val_dataloader:")[1]

PAIRS = [
    ("deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt.yml",
     "deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt_flir.yml",
     "deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt"),
    ("deimv2_dinov3_l_aic_rdtdiffuse_30ep_both.yml",
     "deimv2_dinov3_l_aic_rdtdiffuse_30ep_both_flir.yml",
     "deimv2_dinov3_l_aic_rdtdiffuse_30ep_both"),
]

for src, dst, stem in PAIRS:
    t = open(os.path.join(CFG, src)).read()
    head = t.split("val_dataloader:")[0]
    head = head.replace(f"output_dir: ./outputs/{stem}",
                        f"output_dir: ./outputs/{stem}_flir")
    open(os.path.join(CFG, dst), "w").write(head + "val_dataloader:" + FLIR_VAL)
    print("wrote", dst)
