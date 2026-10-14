"""Derive night-training variants that target the dark05/12/25/50 curve.

E1 (already training) : prompt   + unfreeze + degrade  brightness [0.08,0.60] linear
E2 (this script)      : prompt   + unfreeze + degrade  brightness [0.05,0.80] LOG-uniform
E3 (this script)      : diffuse  + unfreeze + degrade  brightness [0.05,0.80] LOG-uniform

Log-uniform light sampling gives the four (roughly x2-spaced) evaluation levels
equal mass, instead of over-sampling the bright end like the linear draw does.
E3 also re-tests the RDTDiffuse path now that OrthogonalProjectionFusion uses a
true (linear-in-magnitude) Gram-Schmidt projection.
"""
import os

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"

E1 = "deimv2_dinov3_l_aic_rdttrack_heat_night.yml"
E1_OUT = "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night"
DEG_LINE = ("        - {type: RandomRGBDegrade5ch, p: 0.5, brightness: [0.08, 0.60], "
            "contrast: [0.30, 0.90], noise_std: [0.02, 0.12]}")
DEG_LINE_LOG = ("        - {type: RandomRGBDegrade5ch, p: 0.5, brightness: [0.05, 0.80], "
                "contrast: [0.30, 0.90], noise_std: [0.02, 0.12], log_brightness: true}")

# ---------------------------------------------------------------- E2: prompt
base = open(os.path.join(CFG, E1)).read()
assert DEG_LINE in base
e2 = base.replace(E1_OUT, E1_OUT + "_log").replace(DEG_LINE, DEG_LINE_LOG)
out2 = os.path.join(CFG, "deimv2_dinov3_l_aic_rdttrack_heat_night_log.yml")
open(out2, "w").write(e2)
print("wrote", out2)

# ------------------------------------------------- E3: diffuse + unfreeze + degrade
UF = "deimv2_dinov3_l_aic_rdtdiffuse_unfreeze.yml"
UF_OUT = "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_unfreeze"
uf = open(os.path.join(CFG, UF)).read()
anchor = ("        - {type: SanitizeBoundingBoxes, min_size: 1}\n"
          "        - {type: Normalize,")
assert uf.count(anchor) == 1, f"anchor hits={uf.count(anchor)}"
e3 = uf.replace(UF_OUT, UF_OUT + "_night").replace(anchor, DEG_LINE_LOG + "\n" + anchor)
out3 = os.path.join(CFG, "deimv2_dinov3_l_aic_rdtdiffuse_unfreeze_night.yml")
open(out3, "w").write(e3)
print("wrote", out3)

# ------------------------------------------------------------------- verify
import yaml
for p in (out2, out3):
    c = yaml.safe_load(open(p))
    ops = c["train_dataloader"]["dataset"]["transforms"]["ops"]
    deg = [o for o in ops if o and o.get("type") == "RandomRGBDegrade5ch"]
    print(f"  {os.path.basename(p)}")
    print(f"    output_dir : {c['output_dir']}")
    print(f"    backbone   : {c['DEIM']['backbone']}")
    print(f"    degrade    : {deg[0] if deg else 'MISSING'}")
    assert deg, "degrade op missing"
    bb = c["DEIM"]["backbone"]
    assert c[bb]["unfreeze_rgb_stages"] == 2
print("OK")
