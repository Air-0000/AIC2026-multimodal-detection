"""Controlled A/B over the `inject` switch of DINOv3STAsRDTDiffuse.

`inject` selects how the aux (de-redundified IR-vs-depth) branch reaches the RGB
stream. The class builds all three paths, so nothing else moves:
    prompt  : rgb + PromptBlock(cat[rgb, dt])      <- RDTTrack, 68ep baseline=0.466
    diffuse : DiffWeightedFusion(rgb, dt)          <- REPLACES rgb
    both    : 0.5 * (prompt path + diffuse path)

Measured so far on AIC val (feature-space ablations, identical recipe):
    prompt  68ep(b8): AP 0.466, aux-all-off -0.9%
    diffuse 30ep(b4): AP 0.345, aux-all-off -12.2%
    diffuse 68ep(b4): AP 0.432, aux-all-off  -5.1%

That is a trade-off, not a win: diffuse buys aux reliance but gives up accuracy.
`both` was never trained -- it is the actual "combine them" candidate. This
script emits the 30ep prompt/both twins at the SAME batch (4) and schedule as the
existing 30ep diffuse run, so the three points are fully controlled.
"""
import os

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
SRC = "deimv2_dinov3_l_aic_rdtdiffuse_30ep.yml"
BASE_OUT = "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_30ep"

VARIANTS = ["prompt", "both"]

text = open(os.path.join(CFG, SRC)).read()
assert "  inject: diffuse" in text, SRC
assert f"output_dir: {BASE_OUT}" in text, SRC

for v in VARIANTS:
    t = text.replace("  inject: diffuse", f"  inject: {v}")
    t = t.replace(f"output_dir: {BASE_OUT}", f"output_dir: {BASE_OUT}_{v}")
    dst = SRC.replace(".yml", f"_{v}.yml")
    open(os.path.join(CFG, dst), "w").write(t)
    print("wrote", dst)
