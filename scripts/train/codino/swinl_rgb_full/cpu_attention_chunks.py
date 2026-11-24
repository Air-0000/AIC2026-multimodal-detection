"""Bound CPU reference MSDA temporaries by chunking independent query positions."""
import importlib
import torch


def enable_query_chunks(chunk=4096):
    module=importlib.import_module('mmcv.ops.multi_scale_deform_attn')
    original=module.multi_scale_deformable_attn_pytorch
    if hasattr(original,'_aic_original'):
        return
    def bounded(value,spatial_shapes,locations,weights):
        if locations.shape[1]<=chunk:
            return original(value,spatial_shapes,locations,weights)
        return torch.cat([original(value,spatial_shapes,locations[:,i:i+chunk],weights[:,i:i+chunk])
                          for i in range(0,locations.shape[1],chunk)],dim=1)
    # Verify the same pointwise interpolation and reduction before installing.
    g=torch.Generator().manual_seed(20260918)
    shapes=torch.tensor([[10,12],[5,6]])
    value=torch.rand(2,150,4,8,generator=g)
    locations=torch.rand(2,79,4,2,4,2,generator=g)
    weights=torch.rand(2,79,4,2,4,generator=g)
    expected=original(value,shapes,locations,weights)
    actual=torch.cat([original(value,shapes,locations[:,i:i+17],weights[:,i:i+17]) for i in range(0,79,17)],dim=1)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    bounded._aic_original=original
    module.multi_scale_deformable_attn_pytorch=bounded
    print(f'CPU_QUERY_CHUNKS={chunk} EXACT_REFERENCE_TEST_PASS',flush=True)
