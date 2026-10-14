import torch
from engine.backbone.hgnetv2_mm import HGNetv2MultiModal

# ir_in_chans=1 (deleak only); pretrained=False to skip weight download in smoke
m = HGNetv2MultiModal(name='B5', ir_in_chans=1, fusion='cssa',
                      inject_stages=[2, 3], ir_drop_prob=0.0, pretrained=False)
m.eval()

x = torch.randn(2, 6, 640, 640)          # 6ch cache layout (RGB3+heat1+raw_ir1+depth1)
with torch.no_grad():
    outs = m(x)
print('[smoke] ir_in_chans=', m.ir_in_chans)
print('[smoke] out shapes:', [tuple(o.shape) for o in outs])

# gradient flow through IR stem when drop disabled
m.train()
x2 = torch.randn(2, 6, 640, 640, requires_grad=True)
outs2 = m(x2)
loss = sum(o.sum() for o in outs2)
loss.backward()
ir_grad = x2.grad[:, 3, 0, 0].abs().mean().item()   # heat channel gradient must be non-zero
rgb_grad = x2.grad[:, 0, 0, 0].abs().mean().item()
print(f'[smoke] heat(ch3) grad={ir_grad:.4f}  rgb(ch0) grad={rgb_grad:.4f}')
assert ir_grad > 0, 'IR stem gradient broken!'
print('[smoke] OK: IR(deleak) stream active, RGB stream active, ir_in_chans=1 builds+forwards.')
