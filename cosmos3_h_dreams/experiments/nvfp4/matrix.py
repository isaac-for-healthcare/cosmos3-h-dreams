# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the published loopback protocols in bounded, fresh worker processes."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def protocol(name, commit):
    value = json.loads((HERE / f'{name}.json').read_text())
    value.update(commits={'latest': commit}, run_name=name,
                 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    return value


def worker_commands(plan, output, python=sys.executable):
    for index, variant in enumerate(plan['order']):
        case = plan['variants'][variant]
        yield [python, str(HERE / 'browser.py'), '--mode', 'latest',
               '--attention-variant', case['attention_variant'], '--variant', variant,
               '--display-fps', str(case['display_fps']), '--lead-ms', str(case['lead_ms']),
               '--seconds', str(plan['seconds_per_trial']),
               '--out', str(output / 'browser' / f'{index:02d}-{variant}')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['latency30', 'rate', 'attention45'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--nvfp4-weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--gpu-lock', type=Path, required=True,
                        help='Use the same lock as other GPU work on this host.')
    parser.add_argument('--worker-timeout', type=int, default=420)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.worker_timeout <= 0:
        parser.error('--worker-timeout must be positive')
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    plan = protocol(args.experiment, commit)
    plan['worker_timeout_s'] = args.worker_timeout
    plan.pop('outer_timeout_s', None)
    output = args.output.resolve()
    commands = list(worker_commands(plan, output))
    if args.dry_run:
        print(json.dumps(dict(protocol=plan, commands=commands), indent=2))
        return
    if subprocess.check_output(['git', '-C', str(REPO), 'diff', 'HEAD', '--']):
        raise RuntimeError('Commit tracked source changes before recording a comparison')
    checkpoint = args.checkpoint.resolve(strict=True)
    overlay = args.nvfp4_weights.resolve(strict=True)
    manifest = json.loads((overlay / 'manifest.json').read_text())
    with checkpoint.open('rb') as file:
        actual_hash = hashlib.file_digest(file, 'sha256').hexdigest()
    if actual_hash != plan['checkpoint_sha256'] or actual_hash != manifest['source_sha256']:
        raise ValueError('This protocol requires the original Dreams checkpoint and its matching overlay')
    args.gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.gpu_lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'logs').mkdir()
        (output / 'browser').mkdir()
        (output / 'protocol.json').write_text(json.dumps(plan, indent=2) + '\n')
        (output / 'source-sha256.json').write_text(json.dumps({
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.iterdir() if p.suffix in ('.py', '.json', '.html')
        }, indent=2) + '\n')
        env = os.environ.copy()
        env.pop('DREAMS_FP8_WEIGHTS', None)
        env.update(DREAMS_CHECKPOINT=str(checkpoint), DREAMS_NVFP4_WEIGHTS=str(overlay),
                   DREAMS_BENCH_PROTOCOL=str(output / 'protocol.json'),
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO / 'cosmos3_h_dreams'),
            str(REPO / 'flashdreams'), str(HERE), env.get('PYTHONPATH')]))
        outcome = dict(completed=False, source_commit=commit, trials=[])
        try:
            subprocess.run([sys.executable, str(HERE / 'selfcheck.py')], cwd=REPO, env=env, check=True, timeout=60)
            for variant, command in zip(plan['order'], commands, strict=True):
                attention = plan['variants'][variant]['attention_variant']
                cache = args.cache_root.resolve() / attention
                env.update(TORCHINDUCTOR_CACHE_DIR=str(cache / 'torch'), TRITON_CACHE_DIR=str(cache / 'triton'))
                tag = Path(command[-1]).name
                with (output / 'logs' / f'{tag}.log').open('x') as log:
                    # GNU timeout also terminates browser descendants on expiry.
                    subprocess.run(['timeout', '--kill-after=15s', str(args.worker_timeout), *command],
                                   cwd=REPO, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
                outcome['trials'].append(tag)
                print(tag, 'completed', flush=True)
            with (output / 'analysis.log').open('x') as log:
                subprocess.run([sys.executable, str(HERE / 'analyze.py'), str(output)],
                               cwd=REPO, env=env, check=True, stdout=log,
                               stderr=subprocess.STDOUT, timeout=60)
            outcome['completed'] = True
        finally:
            (output / 'outcome.json').write_text(json.dumps(outcome, indent=2) + '\n')


if __name__ == '__main__':
    main()
