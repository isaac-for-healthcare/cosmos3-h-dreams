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

"""Cosmos action/vision mRoPE adapter for clean FadeMem history."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from flashdreams.core.attention.fademem import FadeMemKVCache
from flashdreams.core.attention.kvcache import LayerKV
from torch import Tensor

from cosmos_h_dreams._runtime._rope import mrope_cos_sin

if TYPE_CHECKING:
    from cosmos_h_dreams._runtime._spec import Cosmos3HDreamsSpec


class CosmosFadeMem:
    """Own fixed text context, phase-free temporal memory, and attention views."""

    def __init__(
        self,
        cache: Sequence[dict[str, Tensor]],
        *,
        n_text: int,
        capacity: int,
        beta: float,
        merge_selection: str = "age",
        seed_latent: Tensor | None = None,
        chunk_frames: int,
        positions: Callable[[int, int], Tensor],
        spec: Cosmos3HDreamsSpec,
    ) -> None:
        """Seed memory from text plus one clean frame in the normal Cosmos cache."""
        self.chunk_frames = chunk_frames
        self.positions: Callable[[int, int], Tensor] = positions
        self.spec = spec
        self.text = [
            (layer["k"][:, :, :n_text].clone(), layer["v"][:, :, :n_text].clone())
            for layer in cache
        ]
        self.seed = [
            (
                self.rotate_temporal(
                    layer["k"][:, :, n_text:], positions(0, 1), inverse=True
                ),
                layer["v"][:, :, n_text:].clone(),
            )
            for layer in cache
        ]
        self.seed_descriptor = (
            seed_latent.detach().reshape(1, -1).float().clone()
            if merge_selection == "cosine" and seed_latent is not None
            else None
        )
        self.memory = FadeMemKVCache(
            self.seed,
            capacity=capacity,
            beta=beta,
            merge_selection=merge_selection,
            descriptors=self.seed_descriptor,
        )
        self.prepared = []
        for key, value in self.memory.layers(full_capacity=True):
            # Values have no positional phase: read the original fixed buffer.
            self.prepared.append((torch.zeros_like(key), value))
        self.mask = torch.zeros(
            n_text + capacity * self.memory.tokens_per_frame,
            dtype=torch.bool,
            device=self.seed[0][0].device,
        )
        self.valid_length = torch.zeros((), dtype=torch.int32, device=self.mask.device)

    def rotate_temporal(
        self, key: Tensor, positions: Tensor, *, inverse: bool = False
    ) -> Tensor:
        """Remove/reapply only temporal mRoPE, including fractional action times."""
        temporal = positions.clone()
        temporal[1:] = 0
        llm = self.spec.llm
        cos, sin = mrope_cos_sin(
            temporal, llm.head_dim, llm.rope_theta, llm.mrope_section
        )
        x = key.float()
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        sign = -1 if inverse else 1
        return (x * cos[None, None] + sign * rotated * sin[None, None]).to(key.dtype)

    @torch.no_grad()
    def prepare(
        self, start: int, frames: int, *, fixed_shape: bool = False
    ) -> tuple[Tensor, list[LayerKV]]:
        """Prepare one shared positional layout for history and current Q/K."""
        centers, current = self.memory.attention_positions(start, self.chunk_frames)
        positions = torch.cat([self.positions(center, 1) for center in centers], dim=1)
        valid = self.memory.count * self.memory.tokens_per_frame
        for (key, _), (prepared_k, _) in zip(
            self.memory.layers(), self.prepared, strict=True
        ):
            prepared_k[:, :, :valid].copy_(self.rotate_temporal(key, positions))
        self.mask.zero_()
        self.mask[: self.text[0][0].shape[2] + valid] = True
        self.valid_length.fill_(self.text[0][0].shape[2] + valid)
        history = self.prepared if fixed_shape else self.history()
        return self.positions(current, frames), history

    def history(self) -> list[LayerKV]:
        """Return the valid prefix of the prepared history for eager attention."""
        valid = self.memory.count * self.memory.tokens_per_frame
        return [(k[:, :, :valid], v[:, :, :valid]) for k, v in self.prepared]

    @torch.no_grad()
    def commit(
        self,
        start: int,
        positions: Tensor,
        layers: Sequence[LayerKV],
        *,
        descriptors: Tensor | None = None,
    ) -> None:
        """Commit only finalized clean keys, using their attention-time positions."""
        self.memory.append(
            start,
            [(self.rotate_temporal(k, positions, inverse=True), v) for k, v in layers],
            descriptors=descriptors,
        )

    def reset(self) -> None:
        """Restore the original seed while preserving allocated buffer addresses."""
        self.memory.reset(self.seed, descriptors=self.seed_descriptor)

    @property
    def storage_bytes(self) -> int:
        """Include persistent, prepared, seed, text, and mask storage."""
        # Prepared values alias memory storage, which is already counted.
        tensors = [key for key, _ in self.prepared]
        tensors += [
            t for pairs in (self.seed, self.text) for pair in pairs for t in pair
        ]
        tensors += [self.mask, self.valid_length]
        if self.seed_descriptor is not None:
            tensors.append(self.seed_descriptor)
        return self.memory.storage_bytes + sum(
            t.numel() * t.element_size() for t in tensors
        )
