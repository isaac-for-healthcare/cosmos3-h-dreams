# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from flashdreams.serving.webrtc import bootstrap

pytestmark = pytest.mark.ci_cpu


def test_initialize_cuda_distributed_single_process_uses_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    set_device_calls: list[torch.device] = []
    logging_ranks: list[int | None] = []
    monkeypatch.setattr(bootstrap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bootstrap.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        bootstrap.torch.cuda,
        "set_device",
        lambda device: set_device_calls.append(device),
    )

    context = bootstrap.initialize_cuda_distributed(
        default_device="cuda:2",
        configure_logging_fn=lambda *, world_rank: logging_ranks.append(world_rank),
    )

    assert context.device == torch.device("cuda:2")
    assert context.world_rank == 0
    assert context.world_size == 1
    assert set_device_calls == [torch.device("cuda:2")]
    assert logging_ranks == [0]


def test_initialize_cuda_distributed_defaults_unspecified_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    set_device_calls: list[torch.device] = []
    monkeypatch.setattr(bootstrap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bootstrap.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        bootstrap.torch.cuda,
        "set_device",
        lambda device: set_device_calls.append(device),
    )

    context = bootstrap.initialize_cuda_distributed(
        default_device="cuda",
        configure_logging_fn=lambda *, world_rank: None,
    )

    assert context.device == torch.device("cuda:0")
    assert set_device_calls == [torch.device("cuda:0")]


def test_initialize_cuda_distributed_uses_local_rank_for_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")

    init_calls = 0

    def _fake_distributed_init() -> None:
        nonlocal init_calls
        init_calls += 1

    set_device_calls: list[torch.device] = []
    monkeypatch.setattr(bootstrap.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(bootstrap.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(bootstrap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bootstrap.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        bootstrap.torch.cuda,
        "set_device",
        lambda device: set_device_calls.append(device),
    )

    context = bootstrap.initialize_cuda_distributed(
        distributed_init_fn=_fake_distributed_init,
        configure_logging_fn=lambda *, world_rank: None,
    )

    assert init_calls == 1
    assert context.device == torch.device("cuda:1")
    assert context.world_rank == 3
    assert context.world_size == 8
    assert set_device_calls == [torch.device("cuda:1")]


def test_initialize_cuda_distributed_requires_rank_world_size_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(bootstrap.torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="both RANK and WORLD_SIZE"):
        bootstrap.initialize_cuda_distributed()


def test_initialize_cuda_distributed_rejects_cpu_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(bootstrap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bootstrap.torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="CUDA device is required"):
        bootstrap.initialize_cuda_distributed(default_device="cpu")
