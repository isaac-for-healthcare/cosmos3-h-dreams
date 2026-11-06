# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the real keyboard launcher wiring without loading a model or GPU."""

import sys
from types import SimpleNamespace

import pytest

from cosmos3_h_dreams import token_demo

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("enabled", [False, True])
def test_keyboard_presentation_flags_preserve_model_settings(monkeypatch, enabled):
    captured = {}

    def manager(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    def run(app, **kwargs):
        captured["network"] = kwargs
        captured["routes"] = {route.resource.canonical for route in app.router.routes()}

    argv = [
        "token_demo", "--checkpoint", "model.safetensors",
        "--initial-frame", "assets/example_data/scenes/example_frame.png",
        "--config", "wan-taehv-chunk2-2steps-fademem-compile",
    ]
    if enabled:
        argv += ["--display-fps", "30", "--deadline-pacing",
                 "--gpu-frame-conversion", "--dedicated-inference-thread",
                 "--generation-lead-ms", "180"]
        argv += ["--compile-cache-refresh"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(token_demo, "HeadsetServer", manager)
    monkeypatch.setattr(token_demo, "prepare_decoder", lambda *a, **kw: {"version": "test"})
    monkeypatch.setattr(token_demo, "create_app", lambda manager: token_demo.web.Application())
    monkeypatch.setattr(token_demo.web, "run_app", run)
    token_demo.main()
    config = captured["session"].config
    assert config.fps == 10 and config.chunk_size == 2
    assert config.compile_denoise and config.cuda_graph_denoise
    assert config.compile_cache_refresh is enabled
    assert captured["display_fps"] == (30 if enabled else 10)
    for name in ["deadline_pacing", "gpu_frame_conversion", "dedicated_inference_thread"]:
        assert captured[name] is enabled
    assert captured["generation_lead_ms"] == (180 if enabled else 0)
    assert captured["network"] == {"host": "127.0.0.1", "port": 8081}
    assert "/keyboard" in captured["routes"]


@pytest.mark.parametrize("fps", ["0", "-1"])
def test_invalid_presentation_rate_fails_before_model_setup(monkeypatch, fps):
    monkeypatch.setattr(sys, "argv", ["token_demo", "--display-fps", fps])
    with pytest.raises(SystemExit) as error:
        token_demo.main()
    assert error.value.code == 2


@pytest.mark.parametrize("args", [
    ["--generation-lead-ms", "-1"], ["--generation-lead-ms", "nan"],
    ["--generation-lead-ms", "inf"], ["--generation-lead-ms", "180"],
])
def test_invalid_generation_lead_fails_before_model_setup(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["token_demo", *args])
    with pytest.raises(SystemExit) as error:
        token_demo.main()
    assert error.value.code == 2
