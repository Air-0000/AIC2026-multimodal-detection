import os, re
import matplotlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_DIR, LOG_TEXT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/constant/CodeBuddy/Dev/AIC/logs"
PAT = re.compile(r"IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)")
DARK = {"day": 1.0, "night50": 0.50, "night25": 0.25, "night12": 0.12,
        "night05": 0.05, "night01": 0.01, "night005": 0.005, "night002": 0.002,
        "night001": 0.001, "night0005": 0.0005}
LEVELS = ["day", "night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]

# ufall 5ch baseline (strict smoke re-eval, from prior runs) -- fixed reference
UFALL = {"day": 0.483, "night50": 0.486, "night25": 0.475, "night12": 0.462,
         "night05": 0.426, "night01": 0.211, "night005": 0.117}


def read_map(fname):
    if not os.path.exists(fname):
        return None
    s = open(fname, errors="ignore").read()
    m = PAT.search(s)
    return float(m.group(1)) if m else None


def load(prefix):
    return {lv: read_map(os.path.join(BASE, f"{prefix}_{lv if lv!='day' else 'orig'}.log")) for lv in LEVELS}


base = load("deepir_eval")        # deepir 6ch base (no gate)
mid = load("midboost_eval")       # mid-boost fine-tune

xs = [DARK[l] for l in LEVELS]


def series(d, color, label, style="-", marker="o"):
    pts = [(DARK[l], d[l]) for l in LEVELS if d[l] is not None]
    if pts:
        plt.plot([p[0] for p in pts], [p[1] for p in pts], style, color=color,
                 marker=marker, linewidth=2, label=label)


plt.figure(figsize=(10.5, 6))
series(base, "gray", "deepir base (6ch)", "--", "s")
series(mid, "C3", "deepir mid-boost (mid oversampling FT)", "-", "^")
# ufall points (only where known)
up = [(DARK[l], UFALL[l]) for l in UFALL]
plt.plot([p[0] for p in up], [p[1] for p in up], "k", marker="o", linewidth=2,
         label="ufall 5ch baseline")
plt.xscale("log"); plt.gca().invert_xaxis()
plt.xlabel("RGB brightness multiplier (log); left=bright -> right=near-dark")
plt.ylabel("mAP @ IoU=0.50:0.95")
plt.title("Mid-boost fine-tune: deepir vs ufall (fill the mid gap?)")
plt.legend(fontsize=9); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "midboost_vs_ufall.png"), dpi=130); plt.close()
print("saved", os.path.join(FIG_DIR, "midboost_vs_ufall.png"))

print("\nlevel     ufall   base    midboost")
for lv in LEVELS:
    def f(v): return f"{v:.3f}" if v is not None else "  -  "
    u = UFALL.get(lv)
    print(f"{lv:9s} {f(u)}  {f(base[lv])}  {f(mid[lv])}")

# mid-gap check: levels where deepir base trailed ufall
print("\n--- mid-gap closure (levels where base < ufall) ---")
for lv in ["night50", "night25", "night12", "night05", "night01", "night005"]:
    u = UFALL.get(lv); b = base[lv]; m = mid[lv]
    if u and b and b < u:
        delta_base = b - u
        delta_mid = (m - u) if m is not None else None
        print(f"{lv:9s} ufall={u:.3f} base={b:.3f}({delta_base:+.3f}) "
              f"mid={(f'mid={m:.3f}({delta_mid:+.3f})' if m is not None else 'pending')}")
