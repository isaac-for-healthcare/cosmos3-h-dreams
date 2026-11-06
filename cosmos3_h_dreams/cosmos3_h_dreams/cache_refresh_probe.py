# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual CUDA probe for the eight-frame experimental cache-refresh option.

Run each mode in a fresh process on an otherwise idle GPU. Timing ends at
prepared host uint8 RGB, not at browser presentation. See LATENCY_TRIAL.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cosmos3_h_dreams.config import COSMOS3_H_DREAMS_PRESETS, Cosmos3HDreamsConfig
from cosmos3_h_dreams.headset_server import MjpegFrames
from cosmos3_h_dreams.session import Cosmos3HDreamsSession
from flashdreams.demo import CanonicalInputWindow
from flashdreams.infra.time import TimeWindow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-cache-rope", action="store_true")
    parser.add_argument("--defer-cache-refresh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--reference-quality", type=Path,
                        help="Optional same-mode JSON containing frame_sha256.")
    args = parser.parse_args()
    if args.frames < 320 or args.frames % 8:
        parser.error("--frames must be a multiple of eight and at least 320")
    if not torch.cuda.is_available():
        parser.error("This manual probe requires CUDA")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    trajectory_path = Path("assets/example_data/scenes/trajectory.npy")
    trajectory = np.load(trajectory_path, allow_pickle=False)
    indices = np.resize(np.r_[np.arange(len(trajectory)),
                             np.arange(len(trajectory) - 2, 0, -1)], args.frames)
    np.save(args.output / "actions.npy", trajectory[indices])
    np.save(args.output / "static-actions.npy", np.repeat(trajectory[:1], 64, axis=0))
    config = Cosmos3HDreamsConfig(**{
        **COSMOS3_H_DREAMS_PRESETS["wan-taehv-chunk2-2steps-fademem-compile"],
        "checkpoint": args.checkpoint, "seed": args.seed,
        "initial_frame": Path("assets/example_data/scenes/example_frame.png"),
        "actions": args.output / "actions.npy",
        "compile_cache_refresh": args.mode == "compiled",
        "reuse_cache_rope": args.reuse_cache_rope,
        "defer_cache_refresh": args.defer_cache_refresh, "profile": False,
    })
    session = Cosmos3HDreamsSession(config=config)
    converter = MjpegFrames(gpu_frame_conversion=True)
    report = dict(
        completed=False, mode=args.mode, config=dataclasses.asdict(config),
        command=sys.argv, source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(), driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True).strip().splitlines()[0],
        compiler_cache={key: os.environ.get(key) for key in
                        ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")},
        trajectory_sha256=hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
        warmup_chunks=8, timed_chunks=32, metric="generation_to_prepared_host_uint8_rgb",
        note="Readiness excludes deferred refresh. Complete throughput includes it and the final GPU drain. Neither is screen latency.",
    )

    def step():
        result = session.step(CanonicalInputWindow(window=TimeWindow(start_s=0, end_s=.8)))
        frames = converter.prepare_result_frames(result)
        return result, frames

    def hashes(frames):
        return [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]

    try:
        started = time.perf_counter()
        session.init()
        report["startup_s_including_hidden_compile"] = time.perf_counter() - started
        _, rollout = session._require_runtime()
        assert rollout.fademem.reuse_cache_rope == config.reuse_cache_rope
        report["reuse_cache_rope_active"] = rollout.fademem.reuse_cache_rope
        assert rollout.defer_cache_refresh == config.defer_cache_refresh
        report["defer_cache_refresh_active"] = rollout.defer_cache_refresh
        compiled_calls = 0
        compiled_layer = rollout._compiled_refresh_layer
        assert (compiled_layer is not None) == config.compile_cache_refresh

        def counted_layer(*a, **kw):
            nonlocal compiled_calls
            compiled_calls += 1
            return compiled_layer(*a, **kw)

        if compiled_layer is not None:
            rollout._compiled_refresh_layer = counted_layer
        print(json.dumps({"phase": "initialized", "mode": args.mode,
                          "startup_s": report["startup_s_including_hidden_compile"]}), flush=True)
        for _ in range(8):
            step()
            session.submit_pending_cache_update()
        rollout.wait_for_pending_cache_update(block=True)
        wall_ms = []
        complete_started = time.perf_counter()
        for _ in range(32):
            started = time.perf_counter()
            step()
            wall_ms.append((time.perf_counter() - started) * 1000)
            session.submit_pending_cache_update()
        rollout.wait_for_pending_cache_update(block=True)
        report["complete_ms_per_chunk"] = (time.perf_counter() - complete_started) * 1000 / 32
        report["step_wall_ms"] = wall_ms
        report["generation_ms"] = {f"p{p}": float(np.percentile(wall_ms, p))
                                   for p in (50, 90, 95)}
        session.reset()
        before_calls = compiled_calls
        frame_hashes = []
        for _ in range(args.frames // 8):
            result, frames = step()
            session.submit_pending_cache_update()
            assert len(frames) == 8 and torch.isfinite(result.video_chunk).all().item()
            frame_hashes.extend(hashes(frames))
        report.update(frame_sha256=frame_hashes, frames=len(frame_hashes), all_finite=True,
                      quality_compiled_layer_calls=compiled_calls - before_calls)
        expected_calls = ((args.frames // 8 - 6) * len(rollout._layer_list())
                          if config.compile_cache_refresh else 0)
        assert report["quality_compiled_layer_calls"] == expected_calls
        session.reset()
        _, frames = step()
        report["reset_exact"] = hashes(frames) == frame_hashes[:8]
        assert report["reset_exact"]
        session.switch_scene(dataclasses.replace(config, actions=args.output / "static-actions.npy"))
        for _ in range(8):
            result, _ = step()
            assert torch.isfinite(result.video_chunk).all().item()
        session.switch_scene(config)
        _, frames = step()
        report["scene_return_exact"] = hashes(frames) == frame_hashes[:8]
        assert report["scene_return_exact"]
        # Exercise a retained chunk without the pixel-preparation consumer.
        session.reset()
        session.step(CanonicalInputWindow(window=TimeWindow(start_s=0, end_s=.8)))
        session.reset()
        _, frames = step()
        report["pending_reset_exact"] = hashes(frames) == frame_hashes[:8]
        assert report["pending_reset_exact"]
        session.reset()
        raw = session.step(CanonicalInputWindow(window=TimeWindow(start_s=0, end_s=.8)), decode=False)
        report["raw_refresh_submitted"] = session._require_runtime()[1]._deferred_cache is None
        assert report["raw_refresh_submitted"]
        report["raw_latent_sha256"] = hashlib.sha256(
            raw.output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        # Leave a pixel step pending to verify close drains it below.
        session.step(CanonicalInputWindow(window=TimeWindow(start_s=0, end_s=.8)))
        if args.reference_quality:
            reference = json.loads(args.reference_quality.read_text())["frame_sha256"]
            report["reference_quality"] = str(args.reference_quality)
            report["same_mode_reference_exact"] = frame_hashes == reference[:args.frames]
            assert report["same_mode_reference_exact"], "Integrated path differs from same-mode reference"
        report["completed"] = True
    finally:
        session.close()
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["dynamo_counters"] = {k: dict(v) for k, v in torch._dynamo.utils.counters.items()}
        (args.output / "quality.json").write_text(json.dumps(report, default=str, indent=2))
    print(json.dumps({"completed": True, "mode": args.mode,
                      "generation_ms": report["generation_ms"]}), flush=True)


if __name__ == "__main__":
    main()
