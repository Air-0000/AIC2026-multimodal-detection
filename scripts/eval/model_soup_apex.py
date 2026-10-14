"""
Model Soup Parameter Fusion Script for Sovereign V22
Blends Cycle 1 Apex weights with Cycle 2 Apex weights in parameter space.
Formula: theta_soup = alpha * theta_cycle1 + (1 - alpha) * theta_cycle2
"""

import sys
import torch
from pathlib import Path


def soup_models(path_c1: str, path_c2: str, out_path: str, alpha: float = 0.5):
    print(f"Loading Cycle 1 checkpoint: {path_c1}")
    state_c1 = torch.load(path_c1, map_location="cpu", weights_only=False)

    print(f"Loading Cycle 2 checkpoint: {path_c2}")
    state_c2 = torch.load(path_c2, map_location="cpu", weights_only=False)

    # Extract pure state dicts
    def get_sd(state):
        if "ema" in state and "module" in state["ema"]:
            return state["ema"]["module"]
        elif "model" in state:
            return state["model"]
        return state

    sd1 = get_sd(state_c1)
    sd2 = get_sd(state_c2)

    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())
    common_keys = keys1.intersection(keys2)

    print(f"Cycle 1 keys: {len(keys1)}, Cycle 2 keys: {len(keys2)}, Common keys: {len(common_keys)}")
    assert len(common_keys) > 0, "No common keys found between checkpoints!"

    soup_sd = {}
    for k in sd1:
        if k in sd2:
            v1 = sd1[k]
            v2 = sd2[k]
            if v1.dtype.is_floating_point:
                soup_sd[k] = (alpha * v1.float() + (1.0 - alpha) * v2.float()).to(v1.dtype)
            else:
                # For non-floating point buffers (e.g., num_batches_tracked)
                soup_sd[k] = v2
        else:
            soup_sd[k] = sd1[k]

    for k in sd2:
        if k not in soup_sd:
            soup_sd[k] = sd2[k]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup_sd, out_path)
    print(f"Successfully saved Model Soup (alpha={alpha}) to: {out_path} ({len(soup_sd)} keys)")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python model_soup_apex.py <path_c1> <path_c2> <out_path> [alpha=0.5]")
        sys.exit(1)
    c1 = sys.argv[1]
    c2 = sys.argv[2]
    out = sys.argv[3]
    a = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    soup_models(c1, c2, out, a)
