# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two completed cache_refresh_probe runs with only FP8 changed."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(baseline: Path, candidate: Path) -> dict:
    reports = [json.loads((p / "quality.json").read_text()) for p in (baseline, candidate)]
    a, b = reports
    for report in reports:
        if not all(report.get(k) for k in ("completed", "all_finite", "reset_exact", "scene_return_exact", "pending_reset_exact")):
            raise ValueError("Both runs must complete all correctness checks")
        if report["config"]["profile"]:
            raise ValueError("Use unprofiled runs for the throughput comparison")
    if a.get("quantization") is not None or not b.get("quantization"):
        raise ValueError("Expected BF16 baseline and FP8 candidate")
    if b["quantization"]["source_sha256"] != a["checkpoint_sha256"]:
        raise ValueError("FP8 overlay and BF16 baseline have different source weights")
    if a["config"].keys() != b["config"].keys():
        raise ValueError("Unmatched configuration keys")
    # Output-local action filenames and overlay selection are the only config differences allowed.
    for key in a["config"]:
        if key not in {"actions", "fp8_weights"} and a["config"][key] != b["config"].get(key):
            raise ValueError(f"Unmatched configuration: {key}")
    for key in ("checkpoint_sha256", "trajectory_sha256", "initial_frame_sha256", "source_commit", "torch", "cuda", "cudnn", "gpu", "gpu_uuid", "driver", "warmup_chunks", "timed_chunks", "frames"):
        if a[key] != b[key]:
            raise ValueError(f"Unmatched benchmark field: {key}")
    result = {
        "baseline": str(baseline), "candidate": str(candidate),
        "complete_ms_per_chunk": [a["complete_ms_per_chunk"], b["complete_ms_per_chunk"]],
        "generation_time_reduction_pct": 100 * (1 - b["complete_ms_per_chunk"] / a["complete_ms_per_chunk"]),
        "generation_throughput_gain_pct": 100 * (a["complete_ms_per_chunk"] / b["complete_ms_per_chunk"] - 1),
        "frame_readiness_median_ms": [r["generation_ms"]["p50"] for r in reports],
        "allocated_bytes_after_timing": [r["allocated_bytes_after_timing"] for r in reports],
        "control_to_screen_latency": "Not measured by this probe",
        "quality_status": "Visual review required; numerical similarity is not quality acceptance",
    }
    for name in ("motion-first64", "held-controls"):
        x, y = [np.load(p / (name + ".npy"), allow_pickle=False) for p in (baseline, candidate)]
        if x.shape != y.shape or x.dtype != np.uint8 or y.dtype != np.uint8:
            raise ValueError(f"Unmatched video array: {name}")
        metrics = {}
        for count in (8, 64):
            if len(x) < count:
                raise ValueError("Incomplete saved quality clip")
            error = x[:count].astype(np.float64) - y[:count].astype(np.float64)
            mse = float(np.mean(error ** 2))
            metrics[str(count)] = {"mae_uint8": float(np.abs(error).mean()),
                                   "psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None}
        result[name] = metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
