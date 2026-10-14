"""Derive AIC RGB-only training configs that add day->night simulation.

Evidence driving this (BDD100K night zero-shot, rgb3ch, AIC-trained):
    baseline                AP 0.147 / AP50 0.284   (person .40 car .54 bicycle .25)
    + test-time brightness rescale/gamma  -> 0.129-0.133 / 0.246-0.253   WORSE
    + per-image standardisation           -> 0.131 / 0.249              WORSE
So the night drop is NOT fixable by correcting the input photometrically -- the
information the *weights* extract is what fails. The lever is therefore training
with night-simulated input, i.e. make the model see low-light day frames.

E4  deimv2_dinov3_l_aic_nightsim        RGB + RandomRGBDegrade5ch(3ch)
E5  deimv2_dinov3_l_aic_nightsim_inorm  E4 + backbone input_norm: instance
    (expected weaker -- the test-time standardisation probe already hurt -- but
     it is a 1-line flag, so it is cheap to rule in/out)
"""
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
BASE = "deimv2_dinov3_l_aic.yml"
OUT_BASE = "./outputs/deimv2_dinov3_l_aic"
ANCHOR = ("        - {type: Normalize, mean: [0.485, 0.456, 0.406], "
          "std: [0.229, 0.224, 0.225]}")
DEG = ("        - {type: RandomRGBDegrade5ch, p: 0.5, brightness: [0.05, 0.80], "
       "contrast: [0.30, 0.90], noise_std: [0.02, 0.12], log_brightness: true}")

base = open(os.path.join(CFG, BASE)).read()
assert base.count(ANCHOR) == 1, f"anchor hits={base.count(ANCHOR)}"

VARIANTS = [
    ("nightsim", False),
    ("nightsim_inorm", True),
]

for tag, inorm in VARIANTS:
    txt = base.replace(ANCHOR, DEG + "\n" + ANCHOR)
    txt = txt.replace(f"output_dir: {OUT_BASE}\n", f"output_dir: {OUT_BASE}_{tag}\n")
    # shorter fine-tune from the converged AIC RGB checkpoint
    txt = txt.replace("epoches: 68 # Increase to search for the optimal ema",
                      f"epoches: 40\n# day->night fine-tune ({tag}); shorter than the 68ep scratch run")
    txt = txt.replace("flat_epoch: 34", "flat_epoch: 20")
    if inorm:
        txt = txt.replace("  conv_inplane: 32\n  hidden_dim: 224",
                          "  conv_inplane: 32\n  hidden_dim: 224\n  input_norm: instance")
    dst = os.path.join(CFG, f"deimv2_dinov3_l_aic_{tag}.yml")
    open(dst, "w").write(txt)

    c = yaml.safe_load(open(dst))
    ops = c["train_dataloader"]["dataset"]["transforms"]["ops"]
    types = [o.get("type") for o in ops]
    assert "RandomRGBDegrade5ch" in types, dst
    assert types[types.index("Normalize") - 1] == "RandomRGBDegrade5ch", types
    bb = c["DINOv3STAs"]
    print(f"{os.path.basename(dst):<42} degrade OK | input_norm={bb.get('input_norm','none')} "
          f"| epoches={c['epoches']} | out={c['output_dir']}")
print("OK")
