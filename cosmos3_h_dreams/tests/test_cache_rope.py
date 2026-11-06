# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small CPU cache checks for reuse of temporal positional factors."""

from types import SimpleNamespace

import pytest
import torch

from cosmos3_h_dreams._runtime import _fademem
from cosmos3_h_dreams._runtime._hand_kv_session import validate_cache_options

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("mode", ["cat", "block"])
def test_positional_reuse_requires_fademem(mode):
    with pytest.raises(ValueError, match="reuse_cache_rope requires cache_mode='fademem'"):
        validate_cache_options(cache_mode=mode, reuse_cache_rope=True)


@pytest.mark.parametrize("selection", ["age", "cosine"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reuse_matches_fill_merge_and_reset(monkeypatch, selection, dtype):
    generator = torch.Generator().manual_seed(7)

    def random(*shape):
        return torch.randn(*shape, generator=generator).to(dtype)

    def positions(start, frames):
        index = torch.arange(frames * 3, dtype=torch.float32)
        return torch.stack((start + index / 3, index % 3, index % 2))

    source = [{"k": random(1, 2, 5, 16), "v": random(1, 2, 5, 16)} for _ in range(3)]
    spec = SimpleNamespace(llm=SimpleNamespace(
        head_dim=16, rope_theta=10000.0, mrope_section=(4, 2, 2),
    ))
    seed = random(1, 7)
    caches = [_fademem.CosmosFadeMem(
        source, n_text=2, capacity=4, beta=.3, merge_selection=selection,
        seed_latent=seed, chunk_frames=2, positions=positions, spec=spec,
        reuse_cache_rope=enabled,
    ) for enabled in (False, True)]
    calls = []
    original = _fademem.mrope_cos_sin

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(_fademem, "mrope_cos_sin", counted)
    for step in range(12):
        if step == 6:
            for cache in caches:
                cache.reset()
        incoming = [(random(1, 2, 6, 16), random(1, 2, 6, 16)) for _ in range(3)]
        descriptors = random(2, 7).float() if selection == "cosine" else None
        views = []
        for enabled, cache in zip((False, True), caches, strict=True):
            start = cache.memory.next_frame
            before = len(calls)
            current, history = cache.prepare(start, 2, fixed_shape=True)
            views.append((current, [(k.clone(), v.clone()) for k, v in history]))
            cache.commit(start, current, incoming, descriptors=descriptors)
            assert len(calls) - before == (2 if enabled else 6)
        assert torch.equal(views[0][0], views[1][0])
        for left, right in zip(views[0][1], views[1][1], strict=True):
            assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
        assert caches[0].memory.entries == caches[1].memory.entries
        assert torch.equal(caches[0].mask, caches[1].mask)
        assert torch.equal(caches[0].valid_length, caches[1].valid_length)
        for left, right in zip(caches[0].memory.layers(), caches[1].memory.layers(), strict=True):
            assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
