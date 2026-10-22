import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure path
ROOT = "/home/constant/CodeBuddy/Dev/AIC"
if not os.path.exists(ROOT):
    ROOT = "/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"

sys.path.insert(0, os.path.join(ROOT, "vendor/DEIMv2"))
sys.path.insert(0, os.path.join(ROOT, "scripts/lib"))

from engine.core import YAMLConfig

def check_tensor(name, t):
    if t is None:
        return
    if not isinstance(t, torch.Tensor):
        return
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        print(f"  [ALERT] {name}: nan={has_nan}, inf={has_inf}, shape={t.shape}, min={t.min().item() if not has_nan else 'nan'}, max={t.max().item() if not has_nan else 'nan'}")
        return True
    return False

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    cfg_path = os.path.join(ROOT, "vendor/DEIMv2/configs/deimv2/sovereign_v36_p2_direct_injection_2560.yml")
    cfg = YAMLConfig(cfg_path)
    
    # Build model, criterion, optimizer
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    
    # Load tuning weights if any
    tuning_path = os.path.join(ROOT, "vendor/DEIMv2/ckpts/dfine_x_obj2coco.pth")
    if os.path.exists(tuning_path):
        print(f"Loading tuning checkpoint: {tuning_path}")
        ckpt = torch.load(tuning_path, map_location='cpu')
        state_dict = ckpt.get('ema', ckpt.get('model', ckpt))
        if 'module' in state_dict:
            state_dict = state_dict['module']
        # Filter out shape mismatches
        model_dict = model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping mismatch key: {k}")
        msg = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded weights: missed={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    model.train()
    criterion.train()
    
    optimizer = cfg.optimizer
    lr_scheduler = cfg.lr_scheduler
    
    # Create fake batch matching training input
    B = 2
    samples = torch.randn(B, 6, 1440, 2560, device=device)
    targets = [
        {
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.1, 0.1]], device=device),
            'labels': torch.tensor([0, 6], device=device, dtype=torch.int64),
            'orig_size': torch.tensor([1440, 2560], device=device)
        },
        {
            'boxes': torch.tensor([[0.6, 0.6, 0.3, 0.3]], device=device),
            'labels': torch.tensor([1], device=device, dtype=torch.int64),
            'orig_size': torch.tensor([1440, 2560], device=device)
        }
    ]
    
    accum_steps = 2
    micro_bs = 1
    n_micro = 2
    
    print("\n--- Running Iteration 0 ---")
    optimizer.zero_grad()
    for a in range(n_micro):
        s_micro = samples[a*micro_bs:(a+1)*micro_bs]
        t_micro = targets[a*micro_bs:(a+1)*micro_bs]
        
        print(f"Iter 0 micro {a} forward...")
        outputs = model(s_micro, targets=t_micro)
        check_tensor(f"iter0 micro{a} pred_boxes", outputs.get('pred_boxes'))
        check_tensor(f"iter0 micro{a} pred_logits", outputs.get('pred_logits'))
        
        loss_dict = criterion(outputs, t_micro)
        loss = sum(loss_dict.values())
        print(f"Iter 0 micro {a} loss: {loss.item():.4f}")
        
        (loss / n_micro).backward()
    
    # Check gradients
    print("\n--- Checking Gradients after Iter 0 backward ---")
    nan_grads = 0
    huge_grads = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                print(f"  [GRAD ALERT] {name}: nan={torch.isnan(p.grad).any().item()}, inf={torch.isinf(p.grad).any().item()}")
                nan_grads += 1
            else:
                max_abs_grad = p.grad.abs().max().item()
                if max_abs_grad > 1e4:
                    print(f"  [HUGE GRAD] {name}: max_abs={max_abs_grad:.2e}")
                    huge_grads += 1
    print(f"Total parameters with NaN/Inf grad: {nan_grads}, Huge grad (>1e4): {huge_grads}")
    
    # Clip grads
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    print(f"Grad norm before clipping: {grad_norm.item():.4f}")
    
    # Optimizer step
    print("Executing optimizer.step()...")
    optimizer.step()
    optimizer.zero_grad()
    
    # Check parameters after optimizer.step()
    print("\n--- Checking Parameters after optimizer.step() ---")
    nan_params = 0
    for name, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            print(f"  [PARAM ALERT] {name}: nan={torch.isnan(p).any().item()}, inf={torch.isinf(p).any().item()}")
            nan_params += 1
    print(f"Total parameters with NaN/Inf after step: {nan_params}")
    
    # Check running mean / var of batchnorms
    print("\n--- Checking BN Buffers ---")
    nan_buffers = 0
    for name, b in model.named_buffers():
        if torch.isnan(b).any() or torch.isinf(b).any():
            print(f"  [BUFFER ALERT] {name}: nan={torch.isnan(b).any().item()}, inf={torch.isinf(b).any().item()}")
            nan_buffers += 1
    print(f"Total buffers with NaN/Inf: {nan_buffers}")
    
    # Now run Iteration 1 forward
    print("\n--- Running Iteration 1 Forward ---")
    s_micro = samples[0:1]
    t_micro = targets[0:1]
    
    # Hook intermediate layers to find exactly where NaN appears!
    hooks = []
    def make_hook(layer_name):
        def hook(module, inp, outp):
            if isinstance(outp, torch.Tensor):
                check_tensor(f"{layer_name} output", outp)
            elif isinstance(outp, (tuple, list)):
                for idx, item in enumerate(outp):
                    if isinstance(item, torch.Tensor):
                        check_tensor(f"{layer_name} output[{idx}]", item)
            elif isinstance(outp, dict):
                for k, v in outp.items():
                    if isinstance(v, torch.Tensor):
                        check_tensor(f"{layer_name} output['{k}']", v)
        return hook

    # Register hooks on major submodules
    for name, m in model.named_children():
        hooks.append(m.register_forward_hook(make_hook(f"model.{name}")))
    
    # Also hook inside backbone, encoder, decoder submodules
    if hasattr(model, 'backbone'):
        for name, m in model.backbone.named_children():
            hooks.append(m.register_forward_hook(make_hook(f"backbone.{name}")))
    if hasattr(model, 'encoder'):
        for name, m in model.encoder.named_children():
            hooks.append(m.register_forward_hook(make_hook(f"encoder.{name}")))
    if hasattr(model, 'decoder'):
        for name, m in model.decoder.named_children():
            hooks.append(m.register_forward_hook(make_hook(f"decoder.{name}")))
            
    print("Calling model forward on iter 1...")
    outputs1 = model(s_micro, targets=t_micro)
    check_tensor("iter1 pred_boxes", outputs1.get('pred_boxes'))
    check_tensor("iter1 pred_logits", outputs1.get('pred_logits'))
    print("Done probe.")

if __name__ == "__main__":
    main()
