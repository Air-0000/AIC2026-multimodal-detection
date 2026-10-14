"""Unit and Integration Test for HGNetv2ViTDualStream Multimodal Backbone.

Tests:
  1. Instantiation and weight loading for HGNetv2-B5 + DAv2 ViT-Small.
  2. Stage 1 Mode (Frozen ViT): parameter counts, freezing verification.
  3. Forward pass with multimodal tensor [2, 6, 640, 640] (RGB + IR + Depth).
  4. Multi-scale feature map dimensions matching D-FINE [P3: 512, P4: 1024, P5: 2048].
  5. Orthogonal regularization loss calculation and validity.
  6. Backward pass and gradient flow verification.
  7. Stage 2 Mode (LLRD Unfreezing of ViT tail 4 blocks):
     - parameter `llrd_lr` stamping verification
     - gradient flow into unfreezed blocks
  8. Downstream integration test: feeding backbone output into D-FINE HybridEncoder.
"""

import os
import sys
import torch
import torch.nn as nn

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts/lib"))
sys.path.insert(0, os.path.join(ROOT, "vendor/DEIMv2"))
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))

from engine.backbone.hgnetv2_vit_dualstream import HGNetv2ViTDualStream
from engine.deim.hybrid_encoder import HybridEncoder

def main():
    print("=" * 80)
    print("=== TEST 1: Instantiating HGNetv2ViTDualStream in Stage 1 (Frozen ViT) ===")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    backbone = HGNetv2ViTDualStream(
        name='B5',
        return_idx=[1, 2, 3],
        ir_in_chans=2,
        stem_base=64,
        vit_encoder='vits',
        vit_ckpt_path=os.path.join(ROOT, 'work_dir/da2_weights/depth_anything_v2_vits.pth'),
        vit_embed_dim=384,
        vit_input_size=518,
        unfreeze_vit_blocks=0,  # Stage 1: frozen
        pretrained=False,       # Unit test without downloading hgnet weights if offline
    ).to(device)

    # Verify ViT is frozen
    vit_trainable = [p for p in backbone.vit.parameters() if p.requires_grad]
    assert len(vit_trainable) == 0, f"Expected 0 trainable ViT params in Stage 1, got {len(vit_trainable)}"
    print("[PASS] Stage 1 verification: ViT is 100% frozen.")

    print("\n" + "=" * 80)
    print("=== TEST 2: Forward Pass with Multimodal Input [2, 6, 640, 640] ===")
    print("=" * 80)

    x = torch.randn(2, 6, 640, 640, device=device)
    outs = backbone(x)

    print(f"Number of output stages: {len(outs)}")
    for i, out in enumerate(outs):
        print(f"  Stage {i+1} output shape: {out.shape}, expected channels: {backbone._out_channels[i]}, stride: {backbone._out_strides[i]}")

    assert len(outs) == 3, f"Expected 3 stages, got {len(outs)}"
    assert outs[0].shape == (2, 512, 80, 80), f"P3 shape mismatch: {outs[0].shape}"
    assert outs[1].shape == (2, 1024, 40, 40), f"P4 shape mismatch: {outs[1].shape}"
    assert outs[2].shape == (2, 2048, 20, 20), f"P5 shape mismatch: {outs[2].shape}"
    print("[PASS] Feature shapes strictly match D-FINE HybridEncoder inputs!")

    print("\n" + "=" * 80)
    print("=== TEST 3: Orthogonal Disentanglement Loss & Backward Flow ===")
    print("=" * 80)

    orth_loss = backbone.get_orthogonal_loss()
    print(f"Computed Orthogonal Loss: {orth_loss.item():.6f}")
    assert orth_loss.item() > 0, "Orthogonal loss should be strictly positive"

    total_loss = sum(out.sum() for out in outs) + 0.1 * orth_loss
    total_loss.backward()

    # Check that cross-bridges and gating receive gradients
    assert backbone.c2v_bridges[0].scale.grad is not None, "C2V bridge scale gradient missing"
    assert backbone.v2c_bridges[0].gamma.grad is not None, "V2C bridge gamma gradient missing"
    assert backbone.fusion_gates[0].priv_scale.grad is not None, "Fusion gate priv_scale gradient missing"
    print("[PASS] Gradients flow correctly through cross-bridges and gating modules!")

    # Check that frozen ViT did NOT receive gradients
    for p in backbone.vit.parameters():
        assert p.grad is None, "Frozen ViT received gradient!"
    print("[PASS] Frozen ViT correctly received ZERO gradients in Stage 1!")

    print("\n" + "=" * 80)
    print("=== TEST 4: Stage 2 LLRD Unfreezing & Layer-wise Learning Rates ===")
    print("=" * 80)

    backbone.unfreeze_vit(n_unfreeze=4, top_lr=2.0e-5, decay=0.85)

    # Check blocks 8, 9, 10, 11
    blocks = backbone.vit.blocks
    for gi in range(8):
        for p in blocks[gi].parameters():
            assert not p.requires_grad, f"Block {gi} should be frozen!"
            assert not hasattr(p, 'llrd_lr'), f"Block {gi} should not have llrd_lr!"

    for gi in range(8, 12):
        for p in blocks[gi].parameters():
            assert p.requires_grad, f"Block {gi} should be trainable!"
            assert hasattr(p, 'llrd_lr'), f"Block {gi} must have llrd_lr attribute!"

    print(f"Block 8 llrd_lr:  {getattr(next(iter(blocks[8].parameters())), 'llrd_lr', None):.6e}")
    print(f"Block 9 llrd_lr:  {getattr(next(iter(blocks[9].parameters())), 'llrd_lr', None):.6e}")
    print(f"Block 10 llrd_lr: {getattr(next(iter(blocks[10].parameters())), 'llrd_lr', None):.6e}")
    print(f"Block 11 llrd_lr: {getattr(next(iter(blocks[11].parameters())), 'llrd_lr', None):.6e}")

    # Forward & Backward with Stage 2
    backbone.zero_grad()
    x2 = torch.randn(2, 6, 640, 640, device=device)
    outs2 = backbone(x2)
    loss2 = sum(out.sum() for out in outs2) + backbone.get_orthogonal_loss()
    loss2.backward()

    # Now block 11 must have gradients!
    b11_p = next(iter(blocks[11].parameters()))
    assert b11_p.grad is not None, "Block 11 should have received gradients in Stage 2!"
    # And block 0 must still have NO gradients
    b0_p = next(iter(blocks[0].parameters()))
    assert b0_p.grad is None, "Block 0 should still have NO gradients!"
    print("[PASS] Stage 2 LLRD gradient routing verified successfully!")

    print("\n" + "=" * 80)
    print("=== TEST 5: Downstream Native Integration with D-FINE HybridEncoder ===")
    print("=" * 80)

    encoder = HybridEncoder(
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=384,
        csp_type='csp',
    ).to(device)

    with torch.no_grad():
        encoder_outs = encoder(outs2)

    print(f"HybridEncoder output count: {len(encoder_outs)}")
    for i, eo in enumerate(encoder_outs):
        print(f"  Encoder output {i}: shape {eo.shape}")
        assert eo.shape == (2, 384, 80 // (2**i), 80 // (2**i))

    print("[PASS] Full pipeline (DualStream Backbone -> HybridEncoder) runs flawlessly!")
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == '__main__':
    main()
