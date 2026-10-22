#!/usr/bin/env python3
"""GPU health self-test before a long training run. Exit code 1 on any failure.

1. nvidia-smi: ECC mode and retired-page counters (informational unless --require-ecc).
2. Memory re-read: seeded data checksummed repeatedly as exact int64 sums. The sum is order-independent,
   so any mismatch means the stored bits changed.
3. Repeatability: the same matmul and convolution repeated under the training backend settings
   (TF32 off; cuDNN deterministic, disabled on Volta exactly as train.py does) must be bitwise identical.
4. Every result must be finite.

Reference (MODEL_RECORDS §四十): ailab40 V100 GPU1 with ECC off failed 84/660 re-reads and 60/60 convolutions;
an RTX 5060 Ti passed 0/300 and 0/60. Use CUDA_VISIBLE_DEVICES to pick the GPU.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch


def nvidia_smi(uuid):
    fields = ['name', 'ecc.mode.current', 'retired_pages.single_bit_ecc.count', 'retired_pages.double_bit.count',
              'retired_pages.pending', 'driver_version']
    try:
        out = subprocess.run(['nvidia-smi', '-i', uuid, f'--query-gpu={",".join(fields)}', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return dict(zip(fields, [v.strip() for v in out.split(',')])) if out else {}
    except Exception as e:  # informational only
        return dict(error=str(e))


def checksums(blocks):
    return [int(b.view(torch.int32).sum(dtype=torch.int64)) for b in blocks]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gib', type=int, help='memory to test in GiB (default: 80%% of free memory, at most 24)')
    p.add_argument('--passes', type=int, default=30)
    p.add_argument('--matmul-reps', type=int, default=200)
    p.add_argument('--conv-reps', type=int, default=40)
    p.add_argument('--require-ecc', action='store_true', help='fail when ECC is not enabled')
    p.add_argument('--json', type=Path)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('no CUDA device visible')
    dev = torch.device('cuda')
    props = torch.cuda.get_device_properties(dev)
    uuid = f'GPU-{props.uuid}' if hasattr(props, 'uuid') else '0'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.get_device_capability(dev) == (7, 0):
        torch.backends.cudnn.enabled = False
    free, _ = torch.cuda.mem_get_info(dev)
    gib = a.gib or max(1, min(24, int(free * 0.8 / 2**30)))
    report = dict(gpu=props.name, capability=list(torch.cuda.get_device_capability(dev)), torch=torch.__version__,
                  cudnn_enabled=torch.backends.cudnn.enabled, smi=nvidia_smi(uuid), gib=gib, passes=a.passes)
    g = torch.Generator(device=dev)
    g.manual_seed(1234)
    t0 = time.time()

    blocks = [torch.rand(2**28, device=dev, generator=g) for _ in range(gib)]
    ref = checksums(blocks)
    report['memory_mismatches'] = sum(sum(r != c for r, c in zip(ref, checksums(blocks))) for _ in range(a.passes))
    report['memory_reads'] = gib * a.passes
    del blocks
    torch.cuda.empty_cache()

    A = torch.randn(8192, 8192, device=dev, generator=g)
    B = torch.randn(8192, 8192, device=dev, generator=g)
    c0 = A @ B
    report['matmul_mismatches'] = sum(int(not torch.equal(A @ B, c0)) for _ in range(a.matmul_reps))
    report['matmul_finite'] = bool(torch.isfinite(c0).all())
    del A, B, c0

    x = torch.randn(8, 256, 180, 320, device=dev, generator=g)
    w = torch.randn(256, 256, 3, 3, device=dev, generator=g) * 0.02
    y0 = torch.nn.functional.conv2d(x, w, padding=1)
    report['conv_mismatches'] = sum(int(not torch.equal(torch.nn.functional.conv2d(x, w, padding=1), y0)) for _ in range(a.conv_reps))
    report['conv_finite'] = bool(torch.isfinite(y0).all())
    report['seconds'] = round(time.time() - t0, 1)

    ecc = report['smi'].get('ecc.mode.current', '')
    failures = [k for k in ('memory_mismatches', 'matmul_mismatches', 'conv_mismatches') if report[k]]
    failures += [k for k in ('matmul_finite', 'conv_finite') if not report[k]]
    if a.require_ecc and ecc != 'Enabled':
        failures.append(f'ecc={ecc or "unknown"}')
    report['failures'] = failures
    report['ok'] = not failures
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    print('GPU_SELFTEST_PASS' if report['ok'] else f'GPU_SELFTEST_FAIL {failures}')
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
