"""Aux-modality ablation configs: does the 5ch model actually USE IR/Depth?

Both readings of "multimodal loses to RGB-only" are testable:

  (a) the fusion never learned anything usable from aux, i.e. the model is
      effectively RGB-only with extra parameters; or
  (b) it did learn aux features, but they do not help at night.

Zeroing the aux channels at inference separates them. If replacing IR and
Depth with all-black images leaves AP essentially unchanged, the model was not
reading them at all -> (a). If AP collapses, the model depends on them and the
problem is that the dependency does not pay off -> (b).

Black mirrors already exist (make_black_aux.py): 200 symlinks each, one per AIC
val image, so they drop straight in as ir_folder / depth_folder.

Variants per (model, level):
    irblk    IR -> black, Depth kept real     (isolates the IR channel)
    depblk   Depth -> black, IR kept real     (isolates the Depth channel)
    bothblk  both -> black                    (aux fully removed; this is what
                                               BDD100K effectively sees)

Baselines (real aux) for these same levels already exist as
/tmp/night_robust/<tag>_<lv>.log and /tmp/night_real/<tag>_<lv>.log, so only the
ablated runs need GPU time.
"""
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
IR_BLACK = "/home/constant/CodeBuddy/Dev/AIC/data_aux/aic_valid_ir_black"
DEP_BLACK = "/home/constant/CodeBuddy/Dev/AIC/data_aux/aic_valid_depth_black"
IR_REAL = "/home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_heat/valid"
DEP_REAL = "/home/constant/CodeBuddy/Dev/AIC/dataset/train/depth"

LEVELS = ["dark50", "dark05", "night50", "night05"]

MODELS = [
    # (stem, output_dir prefix, tag)
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight",
     "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight__", "rnight"),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall",
     "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall__", "ufall"),
]

VARIANTS = {
    "irblk":  (IR_BLACK, DEP_REAL),
    "depblk": (IR_REAL, DEP_BLACK),
    "bothblk": (IR_BLACK, DEP_BLACK),
}

IR_LINE = f"    ir_folder: {IR_REAL}\n"
DEP_LINE = f"    depth_folder: {DEP_REAL}\n"

made = []
for stem, out_pref, tag in MODELS:
    for lv in LEVELS:
        src = os.path.join(CFG, f"{stem}_val_{lv}.yml")
        base = open(src).read()
        # the val configs keep a train_dataloader section too, and depth_folder
        # points at the same directory in both, so edits must be scoped to the
        # val_dataloader half.
        head, val = base.split("val_dataloader:", 1)
        assert val.count(IR_LINE) == 1, f"{stem}/{lv}: ir_folder line not unique in val"
        assert val.count(DEP_LINE) == 1, f"{stem}/{lv}: depth_folder line not unique in val"
        old_out = f"{out_pref}{lv}"
        assert head.count(old_out) == 1, f"{stem}/{lv}: output_dir not unique"

        for vname, (ir, dep) in VARIANTS.items():
            v = val.replace(IR_LINE, f"    ir_folder: {ir}\n")
            v = v.replace(DEP_LINE, f"    depth_folder: {dep}\n")
            t = (head + "val_dataloader:" + v).replace(
                old_out, f"{out_pref}{lv}_ab{vname}")
            dst = os.path.join(CFG, f"{stem}_val_{lv}_ab{vname}.yml")
            open(dst, "w").write(t)

            c = yaml.safe_load(open(dst))
            vd = c["val_dataloader"]["dataset"]
            assert vd["ir_folder"] == ir and vd["depth_folder"] == dep
            assert vd["n_channels"] == 5
            assert c["output_dir"].endswith(f"_ab{vname}")
            made.append(f"{tag}_{lv}_{vname}")

print(f"wrote {len(made)} ablation configs")
for tag in ["rnight", "ufall"]:
    for vname in VARIANTS:
        print(f"  {tag:<7} {vname:<8} {sum(1 for m in made if m.startswith(f'{tag}_') and m.endswith(vname))} levels")
print("OK")
