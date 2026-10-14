import os, re, glob
base = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
cfgs = sorted(glob.glob(os.path.join(base, "*rdttrack_heat_night_log_rnight_ufall*_val_*.yml")) +
              glob.glob(os.path.join(base, "*val_ufall_*.yml")))
for c in cfgs:
    t = open(c).read()
    nc = re.search(r"n_channels:\s*([0-9]+)", t)
    dt = re.search(r"dataset.*?type.*?'(CocoDetection\w+)'", t)
    ir = re.search(r"ir_folder:\s*(\S+)", t)
    dp = re.search(r"depth_folder:\s*(\S+)", t)
    print("%s | nch=%s | %s | ir=%s | depth=%s" % (
        os.path.basename(c), nc.group(1) if nc else "?",
        dt.group(1) if dt else "?", ir.group(1) if ir else "-", dp.group(1) if dp else "-"))
print("TOTAL", len(cfgs))
