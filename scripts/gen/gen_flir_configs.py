"""Derive FLIR (OOD test) eval configs from the existing LLVIP ones.

The LLVIP configs already encode the right backbone / inject / n_channels for each
recipe; we only swap the dataset paths from LLVIP -> FLIR (test-only, never trained).
Token swaps (all path/name strings):
    LLVIP_heat      -> FLIR_heat
    LLVIP_depth_null-> FLIR_depth_null
    LLVIP/rgb       -> FLIR/rgb
    LLVIP/ir        -> FLIR/ir
    instances_llvip.json -> instances_flir_val.json
    output_dir ..._llvip -> ..._flir
The 4ch recipe uses raw IR (LLVIP/ir -> FLIR/ir) and the 5ch recipes use heat
(LLVIP_heat -> FLIR_heat); both fall out of the token swap automatically.
"""
import os

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"

SRC_DST = [
    ("deimv2_dinov3_l_aic_rdtdiffuse_llvip.yml",
     "deimv2_dinov3_l_aic_rdtdiffuse_flir.yml"),
    ("deimv2_dinov3_l_aic_rdttrack_heat_llvip.yml",
     "deimv2_dinov3_l_aic_rdttrack_heat_flir.yml"),
    ("deimv2_dinov3_l_aic_diffuse_4ch_llvip.yml",
     "deimv2_dinov3_l_aic_diffuse_4ch_flir.yml"),
    ("deimv2_dinov3_l_aic_rgb3ch_llvip.yml",
     "deimv2_dinov3_l_aic_rgb3ch_flir.yml"),
]

REPL = [
    ("LLVIP_heat", "FLIR_heat"),
    ("LLVIP_depth_null", "FLIR_depth_null"),
    ("LLVIP/rgb", "FLIR/rgb"),
    ("LLVIP/ir", "FLIR/ir"),
    ("instances_llvip.json", "instances_flir_val.json"),
    ("_llvip", "_flir"),
]


def main():
    for src, dst in SRC_DST:
        t = open(os.path.join(CFG, src)).read()
        for a, b in REPL:
            t = t.replace(a, b)
        # cosmetic: relabel comments that say LLVIP -> FLIR
        t = t.replace("on LLVIP", "on FLIR")
        t = t.replace("LLVIP as the", "FLIR as the")
        open(os.path.join(CFG, dst), "w").write(t)
        print("wrote", dst)


if __name__ == "__main__":
    main()
