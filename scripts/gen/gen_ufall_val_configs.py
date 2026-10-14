"""Per-level AIC val configs for E11 (ufall) that carry input_norm: instance.

Bug this fixes: run_e11_e12.sh evaluates E11 by reusing rnight's *_val_*.yml,
but rnight was trained WITHOUT input_norm while E11 was trained WITH it
(`input_norm: instance`). DINOv3STAsRDTTrack.forward applies
`_normalize_rgb` only when `input_norm == 'instance'`, so evaluating E11 under
rnight's config feeds the backbone un-normalised RGB -- a train/test mismatch
that would silently understate E11. (nli/E9 has its own val configs with the flag
set; rnight's do not.)

Generates the full 8-level set so E11 lands in the summary table directly:
    dark50 dark25 dark12 dark05   -> images in data_aux/aic_valid_rgb_darkXX
    night50 night25 night12 night05 -> images in data_aux/aic_valid_rgb_nightXX
Both come straight from the rnight configs, which already point at the right
img_folder, heat and depth paths.
"""
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
SRC = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight"
DST = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall"
LEVELS = ["dark50", "dark25", "dark12", "dark05",
          "night50", "night25", "night12", "night05"]

ANCHOR = "  unfreeze_rgb_stages: 2\n"
OUT_OLD = "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight__"

for lv in LEVELS:
    src = os.path.join(CFG, f"{SRC}_val_{lv}.yml")
    txt = open(src).read()
    assert txt.count(ANCHOR) == 1, f"{lv}: unfreeze anchor not unique"
    assert "input_norm" not in txt, f"{lv}: source already has input_norm"
    assert txt.count(OUT_OLD + lv) == 1, f"{lv}: output_dir anchor not unique"

    # unfreeze depth is irrelevant at inference; input_norm is NOT (it changes
    # the RGB stream), so it must match how E11 was trained.
    txt = txt.replace(ANCHOR, ANCHOR + "  input_norm: instance\n")
    txt = txt.replace(OUT_OLD + lv,
                      "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall__" + lv)

    dst = os.path.join(CFG, f"{DST}_val_{lv}.yml")
    open(dst, "w").write(txt)

    c = yaml.safe_load(open(dst))
    b = c["DINOv3STAsRDTTrack"]
    assert b["input_norm"] == "instance", b
    assert c["output_dir"].endswith("rnight_ufall__" + lv), c["output_dir"]
    print(f"wrote {os.path.basename(dst):<70} input_norm={b['input_norm']}")
print("OK")
