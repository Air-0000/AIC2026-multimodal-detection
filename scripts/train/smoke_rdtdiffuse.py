"""CPU smoke test for DINOv3STAsRDTDiffuse: build + forward on all inject modes."""
import sys
import torch

sys.path.insert(0, "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2")
from engine.backbone.multimodal_backbones import DINOv3STAsRDTDiffuse  # noqa: E402

x = torch.randn(2, 5, 320, 320)

for inject in ("prompt", "diffuse", "both"):
    m = DINOv3STAsRDTDiffuse(
        name="dinov3_vits16",
        weights_path=None,          # random init: only checking shapes/graph
        interaction_indexes=[5, 8, 11],
        conv_inplane=32,
        hidden_dim=224,
        inject=inject,
    )
    m.eval()
    with torch.no_grad():
        outs = m(x)
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"inject={inject:8s} -> {[tuple(o.shape) for o in outs]} trainable={n_tr/1e6:.2f}M")

print("SMOKE OK")
