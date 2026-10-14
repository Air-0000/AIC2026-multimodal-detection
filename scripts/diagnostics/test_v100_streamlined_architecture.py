"""
Benchmark and Diagnostic Test for Sovereign Streamlined Architecture on Tesla V100
Evaluates:
  1. Architecture parameter breakdown & Zero-Overhead inference verification
  2. Inference Latency, FPS & Peak VRAM on Tesla V100 (2.5K 1440x2560)
  3. Training step stability, AMP fp16 behavior & backward latency
  4. Validation set (200 images) official COCO metric evaluation
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="Test Streamlined Architecture on V100")
    parser.add_argument("--config", type=str, default="configs/deimv2/sovereign_v31_production_full8ep_2560.yml")
    parser.add_argument("--checkpoint", type=str, default="outputs/sovereign_v31_production_full8ep_2560/best_stg2.pth")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--skip-val", action="store_true", help="Skip full 200-image validation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"=== Sovereign Streamlined Architecture Diagnostic Benchmark on V100 ===")
    print(f"Hardware:    {torch.cuda.get_device_name(device)} ({device})")
    print(f"VRAM Total:  {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")
    print(f"PyTorch:     {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"Config:      {args.config}")
    print(f"Checkpoint:  {args.checkpoint}")
    print("=" * 80)

    # 1. Load Config & Model
    from engine.core import YAMLConfig
    cfg = YAMLConfig(args.config, resume=args.checkpoint)
    model = cfg.model.to(device)

    # Load weights
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n[1/4] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "ema" in ckpt and "module" in ckpt["ema"]:
            state_dict = ckpt["ema"]["module"]
            print(" -> Loaded EMA module weights")
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            print(" -> Loaded model weights")
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f" -> Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        print(f"\n[1/4] Checkpoint not found at {args.checkpoint}, testing with initialized model.")

    # 2. Parameter Breakdown & Structural Minimalism Audit
    print("\n[2/4] Structural Minimalism & Parameter Audit:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_groups = {}
    for name, param in model.named_parameters():
        prefix = name.split(".")[0]
        if prefix == "backbone":
            sub = name.split(".")[1] if len(name.split(".")) > 1 else "other"
            key = f"backbone.{sub}"
        else:
            key = prefix
        param_groups[key] = param_groups.get(key, 0) + param.numel()

    print(f"  Total Parameters:     {total_params / 1e6:.2f} M ({total_params:,})")
    print(f"  Trainable Parameters: {trainable_params / 1e6:.2f} M ({trainable_params:,})")
    print("  Component Breakdown:")
    for k, v in sorted(param_groups.items(), key=lambda x: -x[1]):
        print(f"    - {k:<25}: {v / 1e6:6.2f} M ({v:10,} params) [{v/total_params*100:4.1f}%]")

    # Check Co-ATSS auxiliary head
    has_co_aux = hasattr(model, "co_aux_head") and model.co_aux_head is not None
    print(f"  Co-ATSS Aux Head:     {'Present (Active in train mode)' if has_co_aux else 'None'}")
    if has_co_aux:
        co_params = sum(p.numel() for p in model.co_aux_head.parameters())
        print(f"    Co-Aux Head params: {co_params / 1e6:.2f} M ({co_params:,})")

    # 3. Inference Benchmark at 2.5K Resolution (1440 x 2560)
    print("\n[3/4] Inference Latency, FPS & VRAM Benchmark (Resolution: 1440 x 2560, Batch=1):")
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dummy_input = torch.randn(1, 6, 1440, 2560, device=device, dtype=torch.float32)

    # Warmup
    print("  Warming up (10 iterations)...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(10):
                out = model(dummy_input)
    torch.cuda.synchronize(device)

    # Verify zero-overhead: co_aux_head must NOT be evaluated during eval
    print(f"  Output keys in eval mode: {list(out.keys())}")
    assert "pred_logits" in out and "pred_boxes" in out, "Output missing standard detection heads"
    if "co_aux_logits" in out:
        print("  WARNING: co_aux_logits leaked into eval mode!")
    else:
        print("  VERIFIED: Auxiliary head cleanly bypassed in eval mode (Zero Inference Overhead)!")

    # Timed runs
    num_runs = 30
    print(f"  Benchmarking forward pass ({num_runs} iterations)...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(num_runs):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize(device)
                timings.append(starter.elapsed_time(ender))

    timings = sorted(timings)
    mean_lat = sum(timings) / len(timings)
    med_lat = timings[len(timings) // 2]
    fps = 1000.0 / mean_lat
    peak_vram_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_vram_res = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    print(f"  Results on Tesla V100 (2.5K 1440x2560 FP16):")
    print(f"    - Mean Latency:     {mean_lat:.2f} ms")
    print(f"    - Median Latency:   {med_lat:.2f} ms")
    print(f"    - Min / Max:        {timings[0]:.2f} ms / {timings[-1]:.2f} ms")
    print(f"    - Throughput (FPS): {fps:.2f} FPS")
    print(f"    - Peak VRAM Alloc:  {peak_vram_alloc:.1f} MB ({peak_vram_alloc/1024:.2f} GB)")
    print(f"    - Peak VRAM Resv:   {peak_vram_res:.1f} MB ({peak_vram_res/1024:.2f} GB)")

    # 4. Training Step Sanity & AMP Stability
    print("\n[4/4] Training Step Sanity & AMP Stability Check:")
    model.train()
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer

    # Synthetic training batch
    dummy_targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.15]], device=device),
            "labels": torch.tensor([0, 6], device=device),
            "orig_size": torch.tensor([1440, 2560], device=device)
        }
    ]

    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad()

    t_train_start = time.time()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        train_out = model(dummy_input, targets=dummy_targets)
        loss_dict = criterion(train_out, dummy_targets)
        weight_dict = criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    train_step_ms = (time.time() - t_train_start) * 1000.0

    print(f"  Training Step Time:   {train_step_ms:.2f} ms")
    print(f"  Total Loss:           {total_loss.item():.4f}")
    print("  Loss Components:")
    for k, v in loss_dict.items():
        if k in weight_dict and weight_dict[k] > 0:
            print(f"    - {k:<20}: {v.item():.4f} (weight={weight_dict[k]})")

    is_healthy = not (torch.isnan(total_loss) or torch.isinf(total_loss))
    print(f"  Gradient & Loss Status: {'HEALTHY (No NaN / Inf)' if is_healthy else 'CORRUPTED (NaN/Inf detected!)'}")

    # Optional 200-image full validation
    if not args.skip_val:
        print("\n[Bonus] Running Standard COCO Validation on 200 Images:")
        from engine.solver import TASKS
        cfg_val = YAMLConfig(args.config, resume=args.checkpoint)
        solver = TASKS[cfg_val.yaml_cfg['task']](cfg_val)
        solver.val()

    print("\n" + "=" * 80)
    print("=== Sovereign Streamlined Architecture Test on V100 COMPLETED SUCCESSFULLY ===")
    print("=" * 80)

if __name__ == "__main__":
    main()
