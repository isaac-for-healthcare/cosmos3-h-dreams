# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU checks for Cosmos3 latent output and disconnected flow control."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from flashdreams.demo import CanonicalInputWindow
from flashdreams.infra.time import TimeWindow
from flashdreams.serving.token_stream import RawFloat16TokenCodecConfig, TokenFrameEmitter

from cosmos3_h_dreams.headset_server import HeadsetServer, LatestHeadsetFrames
from cosmos3_h_dreams.session import Cosmos3HDreamsSession
from cosmos3_h_dreams.keyboard import KeyboardActionSource

pytestmark = pytest.mark.ci_cpu


def test_eager_warmup_runs_before_ready_without_consuming_inputs():
    calls = []
    session = Cosmos3HDreamsSession(
        config=SimpleNamespace(compile_denoise=False), eager_warmup_steps=8,
    )
    def prewarm(steps, *, profile):
        assert session.model_status == "Warming GPU kernels…"
        calls.append((steps, profile))
    session._warmup_rollout(SimpleNamespace(prewarm_rollout=prewarm))
    assert calls == [(8, True)]
    assert session.model_status == "Model ready"
    assert session._rows_consumed == session._step_index == 0
    with pytest.raises(ValueError):
        Cosmos3HDreamsSession(config=SimpleNamespace(), eager_warmup_steps=-1)


def test_keyboard_release_stale_input_and_pose_layout():
    source = KeyboardActionSource(fps=10)
    source.set_keys(["arrowup", "x"])
    moved = source.next_actions(8)
    assert moved.shape == (8, 64)
    assert moved[-1, 15] < moved[0, 15] < 0
    assert moved[-1, 23] == -2
    assert moved[0, 3:9].tolist() == [1, 0, 0, 0, 1, 0]
    assert torch.count_nonzero(moved[:, 28:]) == 0
    source.set_keys([])
    stopped = source.next_actions(8)
    assert torch.all(stopped[:, 15] == moved[-1, 15])
    source.set_keys(["arrowup"])
    source._updated = 0
    assert torch.all(source.next_actions(8)[:, 15] == moved[-1, 15])
    with pytest.raises(ValueError):
        source.set_keys(["unknown"])


def test_keyboard_tap_survives_generation_and_is_consumed_once():
    source = KeyboardActionSource(fps=10)
    source.set_keys(["arrowup", "x"])
    source.set_keys([])
    actions = source.next_actions(8)
    assert torch.allclose(actions[:, 15], torch.full((8,), -0.01))
    assert actions[0, 23] == -2
    assert torch.all(actions[1:, 23] == 1)
    assert torch.equal(source.next_actions(8)[:, 15], actions[:, 15])
    source.set_keys(["arrowup"])
    source.set_keys([])
    source.reset_calibration()
    assert torch.count_nonzero(source.next_actions(8)[:, 15]) == 0


def test_raw_output_skips_decode_and_preserves_cropped_tchw():
    latent = torch.arange(48 * 2 * 4 * 6).reshape(48, 2, 4, 6).to(torch.bfloat16)
    session = Cosmos3HDreamsSession(config=SimpleNamespace(
        actions=Path("replay.npy"), chunk_size=2, profile=False, device="cpu",
    ))
    session._scene = SimpleNamespace(replay_actions=torch.zeros(8, 64))
    session._model = SimpleNamespace(_crop_decode_latent=lambda z: z[..., :3, :5])
    session._rollout = SimpleNamespace(step=lambda *a, **kw: [latent], last_timing={})
    result = session.step(
        CanonicalInputWindow(window=TimeWindow(start_s=0, end_s=0.8)), decode=False,
    )
    assert result.layout is None
    assert result.frame_count == 8
    assert result.output.dtype == torch.bfloat16
    assert result.output.shape == (2, 48, 3, 5)
    assert torch.equal(result.output, latent[..., :3, :5].permute(1, 0, 2, 3))
    assert result.metadata["streaming_mode"] == "raw_tokens"


def test_disconnect_unblocks_full_token_window():
    async def exercise():
        class Socket:
            async def send_bytes(self, data):
                pass

        emitter = TokenFrameEmitter(Socket(), codec=RawFloat16TokenCodecConfig().setup(), fps=10, flow_window_size=1)
        manager = HeadsetServer(session=SimpleNamespace(), frames=LatestHeadsetFrames(fps=10))
        manager.token_emitter = emitter
        latent = torch.zeros(1, 48, 2, 2)
        await emitter.emit_chunk(latent, chunk_index=0)
        manager.token_chunk = 1
        blocked = asyncio.create_task(manager._emit_token(emitter, latent))
        await asyncio.sleep(0)
        manager.token_emitter = None
        assert await asyncio.wait_for(blocked, 2) is False

    asyncio.run(exercise())
