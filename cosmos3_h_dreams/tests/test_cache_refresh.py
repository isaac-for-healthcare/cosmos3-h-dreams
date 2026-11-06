# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the experimental cache-refresh boundary and option guards."""

from types import SimpleNamespace

import pytest

from cosmos3_h_dreams.config import Cosmos3HDreamsConfig, COSMOS3_H_DREAMS_PRESETS
from cosmos3_h_dreams._runtime._hand_kv_session import HandKVSession, validate_cache_options

pytestmark = pytest.mark.ci_cpu


def test_cache_compile_is_off_in_every_preset(tmp_path):
    frame = tmp_path / "frame.png"
    frame.touch()
    for preset in COSMOS3_H_DREAMS_PRESETS.values():
        config = Cosmos3HDreamsConfig(**{
            **preset, "checkpoint": "unused", "initial_frame": frame,
        })
        assert not config.compile_cache_refresh
        assert not config.reuse_cache_rope


@pytest.mark.parametrize("overrides, message", [
    ({"cache_mode": "block"}, "requires cache_mode='fademem'"),
    ({"compile_denoise": False}, "requires compile_denoise=True"),
    ({"device": "cpu"}, "requires a CUDA device"),
])
def test_unsupported_cache_compile_fails_early(overrides, message):
    options = dict(cache_mode="fademem", compile_denoise=True,
                   compile_cache_refresh=True, device="cuda:0")
    with pytest.raises(ValueError, match=message):
        validate_cache_options(**{**options, **overrides})


@pytest.mark.parametrize("enabled", [False, True])
def test_refresh_compiles_only_full_memory_and_returns_to_eager_after_reset(enabled):
    calls = []

    class Layer:
        def gen_compute_with_kv(self, *args, **kwargs):
            calls.append(("eager", self, args, kwargs))
            return "hidden", "key", "value"

    layer = Layer()

    def compiled(self, *args, **kwargs):
        calls.append(("compiled", self, args, kwargs))
        return "hidden", "key", "value"

    memory = SimpleNamespace(count=1, capacity=3, merge_selection="age",
                             attention_positions=lambda *_: (None, 5))

    def commit(start, positions, entries, *, descriptors):
        assert entries == [("key", "value")] and descriptors is None
        calls.append(("commit", start, positions))
        memory.count = min(memory.capacity, memory.count + 2)

    rollout = object.__new__(HandKVSession)
    rollout.fademem = SimpleNamespace(
        memory=memory, text=[("tk", "tv")], history=lambda: [("hk", "hv")],
        commit=commit, reset=lambda: setattr(memory, "count", 1),
    )
    rollout.net = SimpleNamespace(_mrope=lambda _: ("cos", "sin"))
    rollout.chunk_size = 2
    rollout._positions = lambda *args: args
    rollout._clean_tokens = lambda *_: "input"
    rollout._causal_within = lambda _: "mask"
    rollout._layer_list = lambda: [layer]
    rollout._compiled_refresh_layer = compiled if enabled else None
    rollout.sdpa_backend = "default"
    rollout.native_gqa_attention = False
    rollout._pending_cache = []
    rollout.wait_for_pending_cache_update = lambda **_: calls.append(("wait",))

    rollout._refresh_chunk(1, None, None, 2)  # The filling chunk stays eager.
    rollout._refresh_chunk(3, None, None, 2)  # Only now is history saturated.
    rollout.reset_rollout()
    assert rollout._compiled_refresh_layer is (compiled if enabled else None)
    rollout._refresh_chunk(1, None, None, 2)
    assert [c[0] for c in calls] == [
        "eager", "commit", "compiled" if enabled else "eager", "commit",
        "wait", "eager", "commit",
    ]
    compute_calls = [c for c in calls if c[0] in {"eager", "compiled"}]
    assert all(c[1:] == compute_calls[0][1:] for c in compute_calls)
