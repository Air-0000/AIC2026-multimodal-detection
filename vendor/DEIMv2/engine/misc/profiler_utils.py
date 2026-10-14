"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:
    # NOTE: calflops monkey-patches F.conv2d / F.multi_head_attention_forward on
    # import and asserts a strict weight/input channel match in its flop hook,
    # which breaks multi-modal backbones (concatenated rgb/ir features, 4ch
    # inputs) and adds global overhead. FLOPs reporting is purely informational,
    # so we skip calflops entirely and just report parameter count.
    try:
        params = sum(p.numel() for p in cfg.model.parameters())
    except Exception:
        params = 0
    return params, {"FLOPs stats skipped: calflops incompatible with this model"}

    # NOTE: calflops monkey-patches F.conv2d on import and asserts a strict
    # weight.shape[1]*groups == input.shape[1] in its flop hook. Multi-modal
    # backbones (e.g. DINOv3STAsCSD) use non-standard conv layouts (concatenated
    # rgb/ir features, 4ch inputs) that trip that assertion. FLOPs reporting is
    # purely informational, so we degrade gracefully instead of aborting training.
    try:
        base_size = cfg.train_dataloader.collate_fn.base_size
        input_shape = (1, 3, base_size, base_size)

        model_for_info = copy.deepcopy(cfg.model).deploy()

        flops, macs, _ = calculate_flops(model=model_for_info,
                                            input_shape=input_shape,
                                            output_as_string=True,
                                            output_precision=4,
                                            print_detailed=False)
        params = sum(p.numel() for p in model_for_info.parameters())
        del model_for_info

        return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}
    except Exception as e:  # pragma: no cover - informational only
        print(f"[profiler_utils] FLOPs stats skipped (model incompatible with calflops): {e}")
        params = sum(p.numel() for p in cfg.model.parameters())
        return params, {"FLOPs stats skipped: calflops incompatible with this model"}
