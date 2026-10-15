# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU check for finite offline action replay."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cosmos3_h_dreams.config import Cosmos3HDreamsConfig
from cosmos3_h_dreams.session import Cosmos3HDreamsSession
from flashdreams.demo import CanonicalInputWindow
from flashdreams.infra.time import TimeWindow
from PIL import Image

from cosmos3_h_dreams import session as session_module

pytestmark = pytest.mark.ci_cpu


class _Rollout:
    accepts_partial_chunks = True
    last_timing: dict[str, float] = {}
    _compile_start_step = 6

    def __init__(self) -> None:
        self.rows: list[int] = []
        self.prewarm_steps = 0

    def prewarm_rollout(self, steps: int, *, profile: bool = False):
        assert profile
        self.prewarm_steps = steps
        return [{"cuda_graph_captured": 1.0, "cuda_graph_fill_captured": 1.0}]

    def step(self, actions: torch.Tensor, *, profile: bool = False):
        del profile
        self.rows.append(int(actions.shape[0]))
        return [torch.zeros((1, 1, 1, 1))]

    def wait_for_pending_cache_update(self, *, block: bool = False) -> float:
        del block
        return 0.0


class _Model:
    device = "cpu"

    def __init__(self, rollout: _Rollout) -> None:
        self.rollout = rollout

    def new_session(self, _scene):
        return self.rollout

    def release_session(self, _scene, rollout):
        assert rollout is self.rollout

    def decode_tensor(self, _latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 3, 2, 2))


def test_session_consumes_full_and_partial_action_chunks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = tmp_path / "frame.png"
    actions = tmp_path / "actions.npy"
    Image.new("RGB", (2, 2)).save(frame)
    np.save(actions, np.zeros((24, 20), dtype=np.float32))
    replay_actions = torch.zeros((24, 64))
    scene = SimpleNamespace(fps=15.0, replay_actions=replay_actions)
    rollout = _Rollout()
    monkeypatch.setattr(session_module, "prepare_scene", lambda *_args: scene)
    session = Cosmos3HDreamsSession(
        config=Cosmos3HDreamsConfig(
            checkpoint="/checkpoint",
            initial_frame=frame,
            actions=actions,
            domain_id=7,
            prompt="A robot folds a towel.",
            device="cpu",
        ),
        model_factory=lambda **_kwargs: _Model(rollout),
    )
    window = CanonicalInputWindow(window=TimeWindow(start_s=0.0, end_s=1.0))

    session.init()
    first = session.step(window)
    second = session.step(window)

    assert first.metadata["action_rows"] == 16
    assert second.metadata["action_rows"] == 8
    assert rollout.rows == [16, 8]
    assert session.next_step_requirements() is None


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_profile_prints_first_and_steady_summary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    device,
) -> None:
    frame = tmp_path / "frame.png"
    actions = tmp_path / "actions.npy"
    Image.new("RGB", (2, 2)).save(frame)
    np.save(actions, np.zeros((24, 20), dtype=np.float32))
    scene = SimpleNamespace(fps=15.0, replay_actions=torch.zeros((24, 64)))

    class _ProfileRollout(_Rollout):
        def step(self, actions: torch.Tensor, *, profile: bool = False):
            result = super().step(actions, profile=profile)
            self.last_timing = {"kv_cache_bytes": float(len(self.rows) * 2**20)}
            return result

    rollout = _ProfileRollout()
    cuda_calls = []
    for name, value in (
        ("synchronize", None),
        ("reset_peak_memory_stats", None),
        ("memory_allocated", 3 * 2**20),
        ("memory_reserved", 5 * 2**20),
        ("max_memory_allocated", 7 * 2**20),
        ("max_memory_reserved", 9 * 2**20),
    ):

        def cuda_stat(selected_device, name=name, value=value):
            cuda_calls.append((name, str(selected_device)))
            return value

        monkeypatch.setattr(torch.cuda, name, cuda_stat)
    monkeypatch.setattr(session_module, "prepare_scene", lambda *_args: scene)
    session = Cosmos3HDreamsSession(
        config=Cosmos3HDreamsConfig(
            checkpoint="/checkpoint",
            initial_frame=frame,
            actions=actions,
            domain_id=7,
            prompt="A robot folds a towel.",
            device=device,
            profile=True,
        ),
        model_factory=lambda **_kwargs: _Model(rollout),
    )
    window = CanonicalInputWindow(window=TimeWindow(start_s=0.0, end_s=1.0))

    session.init()
    session.step(window)
    session.step(window)
    session.close()

    output = capsys.readouterr().out
    assert "=== Cosmos3-H-Dreams profile" in output
    assert "first total:" in output
    assert "steady total_s:" in output
    assert "step 1: total" in output
    assert "KV cache 1.0 MiB" in output
    assert "KV cache: first 1.0 MiB | last 2.0 MiB | peak 2.0 MiB" in output
    if device == "cpu":
        assert not cuda_calls
        assert "VRAM allocated" not in output
    else:
        assert "VRAM allocated: peak 7.0 MiB | average 3.0 MiB" in output
        assert "VRAM reserved: peak 9.0 MiB | average 5.0 MiB" in output
        assert cuda_calls.count(("reset_peak_memory_stats", device)) == 1
        assert all(selected == device for _, selected in cuda_calls)
        assert session._vram_thread is None


def test_vram_average_weights_elapsed_time_and_restarts(tmp_path, monkeypatch) -> None:
    frame = tmp_path / "frame.png"
    frame.touch()
    session = Cosmos3HDreamsSession(
        config=Cosmos3HDreamsConfig(
            checkpoint="/checkpoint",
            initial_frame=frame,
            profile=True,
            device="cuda",
            prompt="A robot folds a towel.",
        )
    )
    calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", calls.append)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 700)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 900)
    monkeypatch.setattr(
        session_module.threading,
        "Thread",
        lambda *, target, daemon: SimpleNamespace(start=lambda: None, join=target),
    )
    for _ in range(2):
        timestamps = iter([10.0, 12.0, 15.0])
        allocated = iter([100, 300, 200])
        reserved = iter([200, 500, 400])
        stops = iter([False, True])
        monkeypatch.setattr(
            session_module.time, "perf_counter", lambda: next(timestamps)
        )
        monkeypatch.setattr(
            torch.cuda, "memory_allocated", lambda device: next(allocated)
        )
        monkeypatch.setattr(
            torch.cuda, "memory_reserved", lambda device: next(reserved)
        )
        monkeypatch.setattr(session._vram_stop, "wait", lambda interval: next(stops))
        session._start_vram_profile()
        session._start_vram_profile()  # Later chunks must preserve the run's peaks.
        session._stop_vram_profile()
        session._stop_vram_profile()
        assert session._vram_profile == {
            "allocated_average": 220.0,
            "reserved_average": 380.0,
            "allocated_peak": 700,
            "reserved_peak": 900,
        }
        assert session._vram_thread is None
    assert calls == [torch.device("cuda:2"), torch.device("cuda:2")]


@pytest.mark.parametrize("cache_mode", ["block", "fademem"])
@pytest.mark.parametrize("denoise_steps, capture_steps", [(1, 4), (2, 2), (4, 1)])
def test_session_forwards_optimization_flags(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    cache_mode: str,
    denoise_steps: int,
    capture_steps: int,
) -> None:
    frame = tmp_path / "frame.png"
    actions = tmp_path / "actions.npy"
    Image.new("RGB", (2, 2)).save(frame)
    np.save(actions, np.zeros((16, 28), dtype=np.float32))
    scene = SimpleNamespace(fps=15.0, replay_actions=torch.zeros((16, 64)))
    monkeypatch.setattr(session_module, "prepare_scene", lambda *_args: scene)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    model_kwargs = {}

    def model_factory(**kwargs):
        model_kwargs.update(kwargs)
        return _Model(_Rollout())

    session = Cosmos3HDreamsSession(
        config=Cosmos3HDreamsConfig(
            checkpoint="/checkpoint",
            initial_frame=frame,
            actions=actions,
            prompt="A robot folds a towel.",
            device="cpu",
            cache_mode=cache_mode,
            fademem_slots=9,
            fademem_beta=0.4,
            fademem_merge_selection="cosine",
            overlap_cache_update=True,
            compile_denoise=True,
            compile_denoise_mode="default",
            cuda_graph_denoise=True,
            cuda_graph_warmup_iters=3,
            denoise_steps=denoise_steps,
        ),
        model_factory=model_factory,
    )

    session.init()
    rollout = session._rollout
    assert rollout.prewarm_steps == rollout._compile_start_step + capture_steps
    assert session.model_status == "Model compiled"
    assert session._rows_consumed == 0
    assert session._step_index == 0
    assert rollout.rows == []
    session.reset()
    assert session._rollout is rollout
    session.init()
    assert session.model_status == "Model compiled"

    for name in (
        "cache_mode",
        "fademem_slots",
        "fademem_beta",
        "fademem_merge_selection",
        "overlap_cache_update",
        "compile_denoise",
        "compile_denoise_mode",
        "cuda_graph_denoise",
        "cuda_graph_warmup_iters",
    ):
        assert model_kwargs[name] == getattr(session.config, name)


@pytest.mark.parametrize("capture_fill", [False, True])
def test_warmup_retries_short_fill_before_reporting_compiled(
    tmp_path, monkeypatch, capture_fill
):
    frame = tmp_path / "frame.png"
    frame.touch()
    actions = tmp_path / "actions.npy"
    np.save(actions, np.zeros((16, 28), dtype=np.float32))
    rollout = _Rollout()
    rollout._compile_start_step = 1
    calls = []

    def prewarm(steps, *, profile):
        assert session.model_status == "Compiling model…"
        assert profile and steps == 3
        calls.append(steps)
        return [
            {
                "cuda_graph_captured": 1.0,
                "cuda_graph_fill_captured": float(capture_fill and len(calls) == 2),
            }
        ]

    rollout.prewarm_rollout = prewarm
    monkeypatch.setattr(
        session_module,
        "prepare_scene",
        lambda *_args: SimpleNamespace(replay_actions=torch.zeros((16, 64))),
    )
    session = Cosmos3HDreamsSession(
        config=Cosmos3HDreamsConfig(
            checkpoint="/checkpoint",
            initial_frame=frame,
            actions=actions,
            device="cpu",
            cache_mode="fademem",
            prompt="A surgical scene.",
            compile_denoise=True,
            cuda_graph_denoise=True,
            cuda_graph_warmup_iters=2,
            denoise_steps=2,
        ),
        model_factory=lambda **_kwargs: _Model(rollout),
    )
    if capture_fill:
        session.init()
        assert session.model_status == "Model compiled"
    else:
        with pytest.raises(RuntimeError, match="did not capture"):
            session.init()
        assert session.model_status != "Model compiled"
    assert len(calls) == 2


def test_cosmos_model_forwards_fademem_selection_to_rollout(monkeypatch):
    import inspect

    from cosmos3_h_dreams._runtime._hand_model import HandModel

    from cosmos3_h_dreams import model as model_module

    model = model_module.Cosmos3HDreamsModel.__new__(model_module.Cosmos3HDreamsModel)
    for name, parameter in inspect.signature(HandModel).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            setattr(model, name, parameter.default)
    model.net = object()
    model.grid_hw = (30, 52)
    model.latent_hw = (60, 104)
    model.tcf = 4
    model.t_list = [1.0, 0.5]
    model.rollout_seed = 0
    model.action_domain_id = 25
    model.fademem_merge_selection = "cosine"
    scene = SimpleNamespace(text_ids=None, init_latent=object(), fps=10.0)
    monkeypatch.setattr(
        model_module, "HandKVSession", lambda _net, **kwargs: SimpleNamespace(**kwargs)
    )
    rollout = model._new_session(scene)
    assert rollout.fademem_merge_selection == "cosine"
    assert "fademem_consolidation" not in vars(rollout)
    assert "fademem_position_mode" not in vars(rollout)


def test_scene_switch_warms_new_rollout_before_reporting_ready(tmp_path, monkeypatch):
    from dataclasses import replace

    frame = tmp_path / "frame.png"
    frame.touch()
    actions = tmp_path / "actions.npy"
    np.save(actions, np.zeros((16, 28), dtype=np.float32))
    scene = SimpleNamespace(fps=15, replay_actions=torch.empty((0, 64)))
    monkeypatch.setattr(session_module, "prepare_scene", lambda *_args: scene)
    rollouts = []

    class Model(_Model):
        action_domain_id = 32
        net = SimpleNamespace(config=SimpleNamespace(num_embodiment_domains=40))

        def new_session(self, _scene):
            self.rollout = _Rollout()
            rollouts.append(self.rollout)
            return self.rollout

    config = Cosmos3HDreamsConfig(
        checkpoint="model.pt",
        initial_frame=frame,
        prompt="A",
        device="cpu",
        actions=actions,
        cache_mode="block",
        compile_denoise=True,
        cuda_graph_denoise=True,
        compile_denoise_mode="default",
    )
    session = Cosmos3HDreamsSession(
        config=config, model_factory=lambda **_: Model(_Rollout())
    )
    session.init()
    session.switch_scene(replace(config, prompt="B"))
    assert len(rollouts) == 2
    assert rollouts[1].prewarm_steps == rollouts[0].prewarm_steps > 0
    assert session.model_status == "Model compiled"
    assert session._rows_consumed == 0
