# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from cosmos3_h_dreams.fp8_compare import compare

pytestmark = pytest.mark.ci_cpu


def test_time_reduction_is_distinct_from_throughput_gain(tmp_path):
    paths = [tmp_path / name for name in ("bf16", "fp8")]
    reports = []
    for index, path in enumerate(paths):
        path.mkdir()
        report = dict(completed=True, all_finite=True, reset_exact=True,
                      scene_return_exact=True, pending_reset_exact=True,
                      config={"profile": False, "actions": str(path / "actions.npy"),
                              "fp8_weights": None if index == 0 else "overlay"},
                      quantization=None if index == 0 else {"source_sha256": "checkpoint_sha256"},
                      complete_ms_per_chunk=100 - 20 * index,
                      generation_ms={"p50": 60 - 10 * index},
                      allocated_bytes_after_timing=1000 - 100 * index)
        for key in ("checkpoint_sha256", "trajectory_sha256", "initial_frame_sha256", "source_commit",
                    "torch", "cuda", "cudnn", "gpu", "gpu_uuid", "driver", "warmup_chunks", "timed_chunks", "frames"):
            report[key] = key
        (path / "quality.json").write_text(json.dumps(report))
        reports.append(report)
        for name in ("motion-first64", "held-controls"):
            np.save(path / (name + ".npy"), np.full((64, 2, 2, 3), index, dtype=np.uint8))
    result = compare(*paths)
    assert result["generation_time_reduction_pct"] == pytest.approx(20)
    assert result["generation_throughput_gain_pct"] == pytest.approx(25)
    assert result["frame_readiness_median_ms"] == [60, 50]
    assert result["motion-first64"]["8"]["mae_uint8"] == 1
    assert "Not measured" in result["control_to_screen_latency"]
    reports[1]["gpu_uuid"] = "different GPU"
    (paths[1] / "quality.json").write_text(json.dumps(reports[1]))
    with pytest.raises(ValueError, match="gpu_uuid"):
        compare(*paths)
    reports[1]["gpu_uuid"] = "gpu_uuid"
    reports[1]["config"]["profile"] = True
    (paths[1] / "quality.json").write_text(json.dumps(reports[1]))
    with pytest.raises(ValueError, match="unprofiled"):
        compare(*paths)
