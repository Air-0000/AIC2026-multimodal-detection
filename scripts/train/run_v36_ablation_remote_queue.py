#!/usr/bin/env python3
"""One-shot remote training queue. Never preempts an existing GPU process."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--wait-pid', type=int, default=44551)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--free-memory-threshold', type=int, default=512,
                        help='maximum preexisting used MiB before the job can start')
    args = parser.parse_args()
    if str(ROOT).startswith('/home/constant/') or socket.gethostname() == 'constant-B850M-EAGLE-WIFI6E-ICE':
        parser.error('Remote-only queue: local training is prohibited')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    status_path = args.out / 'remote_queue_status.json'
    if status_path.exists():
        parser.error('Queue status already exists; refusing duplicate launch')
    def status(state, **extra):
        record = {'state': state, 'host': socket.gethostname(), 'queue_pid': os.getpid(),
                  'gpu': args.gpu, 'wait_pid': args.wait_pid, 'time_unix': time.time(), **extra}
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(status_path)
        print(json.dumps(record), flush=True)
    status('waiting_for_existing_training')
    stable = 0
    while True:
        predecessor_running = args.wait_pid > 0 and Path(f'/proc/{args.wait_pid}').exists()
        query = subprocess.run(['nvidia-smi', f'--id={args.gpu}',
                                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                               text=True, capture_output=True)
        if query.returncode:
            stable = 0
        else:
            used, utilization = [int(x.strip()) for x in query.stdout.strip().split(',')]
            stable = stable + 1 if not predecessor_running and used < args.free_memory_threshold and utilization < 10 else 0
        if stable >= 3:
            break
        time.sleep(args.poll_seconds)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'scripts/lib'), str(ROOT / 'vendor/DEIMv2')])
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['PYTHONUNBUFFERED'] = '1'
    tool = ROOT / 'scripts/eval/run_v36_mechanism_ablation.py'
    # Native-resolution FP32 regression on real images before launching the matrix.
    smoke_log = ROOT / 'logs/v36_remote_training_smoke.log'
    with smoke_log.open('x') as stream:
        smoke = subprocess.Popen([sys.executable, '-u', str(tool), 'step', '--out', str(args.out),
                                  '--variants', 'baseline', 'combined'],
                                 cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status('training_smoke_running', child_pid=smoke.pid, log=str(smoke_log))
        code = smoke.wait()
    if code:
        status('training_smoke_failed', exit_code=code, log=str(smoke_log))
        return code
    command = [sys.executable, '-u', str(tool), 'train', '--out', str(args.out),
               '--variants', 'baseline', 'routing', 'fallback', 'aux_only', 'combined']
    with (ROOT / 'logs/v36_remote_matrix.log').open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status('matrix_running', child_pid=child.pid, command=command)
        code = child.wait()
    status('complete' if code == 0 else 'matrix_failed', exit_code=code)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
