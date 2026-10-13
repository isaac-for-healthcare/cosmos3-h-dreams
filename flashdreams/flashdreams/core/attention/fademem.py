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

"""Fixed-budget, distance-aware consolidation of frame-level KV memory."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from flashdreams.core.attention.kvcache import LayerKV


@dataclass(frozen=True)
class FadeMemEntry:
    """Temporal metadata shared by all layers of a memory entry."""

    center_frame: int
    """Span-weighted center, rounded to a latent-frame index as upstream does."""

    span: int
    """Number of original frames represented by this entry."""


class FadeMemKVCache:
    """Consolidate adjacent frame entries while preserving a fixed KV budget.

    Keys must have their temporal positional phase removed by the model adapter.
    Spatial token correspondence is retained: a merge averages corresponding
    tokens, not arbitrary tokens within a frame. Only committed clean frames
    belong here; provisional denoising KV stays outside this memory.

    Storage reserves one scratch entry beyond ``capacity`` for insertion. Reads
    expose at most ``capacity`` entries and all writes retain buffer addresses.
    """

    def __init__(
        self,
        prefilled: Sequence[LayerKV],
        *,
        capacity: int,
        beta: float = 0.3,
        merge_selection: str = "age",
        descriptors: torch.Tensor | None = None,
        seq_dim: int = 2,
        preserve_anchor: bool = True,
    ) -> None:
        """Allocate memory and seed frame zero.

        Args:
            prefilled: One frame of phase-aligned K/V per layer.
            capacity: Historical frame entries, including the frame-zero anchor.
            beta: Power-law distance exponent in ``(0, 1]``.
            merge_selection: Rank adjacent pairs by age gap or cosine similarity.
            descriptors: One flattened clean-frame descriptor, ``[1, D]``, for cosine.
            seq_dim: Token axis in each layer tensor.
            preserve_anchor: Keep frame zero's original KV when its pair merges.

        Raises:
            ValueError: Capacity, exponent, or tensor layouts are invalid.
        """
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 2:
            raise ValueError("FadeMem capacity must be an integer of at least two")
        if not math.isfinite(beta) or not 0 < beta <= 1:
            raise ValueError("FadeMem beta must be finite and in (0, 1]")
        if merge_selection not in {"age", "cosine"}:
            raise ValueError("FadeMem merge_selection must be age or cosine")
        if not prefilled:
            raise ValueError("FadeMem needs at least one layer")
        rank = prefilled[0][0].ndim
        if not -rank <= seq_dim < rank:
            raise ValueError("FadeMem seq_dim is outside the tensor rank")
        self.seq_dim = seq_dim % rank
        self.capacity = capacity
        self.beta = beta
        self.merge_selection = merge_selection
        self.preserve_anchor = preserve_anchor
        self.tokens_per_frame = prefilled[0][0].shape[self.seq_dim]
        if self.tokens_per_frame < 1:
            raise ValueError("FadeMem needs at least one token per frame")
        self._storage: list[LayerKV] = []
        for key, value in prefilled:
            if key.ndim != rank or value.ndim != rank:
                raise ValueError("FadeMem layers must share a tensor rank")
            if (
                key.shape[self.seq_dim] != self.tokens_per_frame
                or value.shape != key.shape
            ):
                raise ValueError("FadeMem seed K/V must have matching frame layouts")
            if (
                key.device != value.device
                or key.dtype != value.dtype
                or not key.is_floating_point()
            ):
                raise ValueError("FadeMem K/V must share a floating dtype and device")
            shape = list(key.shape)
            shape[self.seq_dim] = (capacity + 1) * self.tokens_per_frame
            self._storage.append((key.new_zeros(shape), value.new_zeros(shape)))
        self._descriptors: torch.Tensor | None = None
        if merge_selection == "cosine":
            if (
                descriptors is None
                or descriptors.ndim != 2
                or descriptors.shape[0] != 1
                or descriptors.shape[1] < 1
            ):
                raise ValueError(
                    "FadeMem cosine selection needs a [1, D] seed descriptor"
                )
            self._descriptors = torch.zeros(
                (capacity + 1, descriptors.shape[1]),
                dtype=torch.float32,
                device=prefilled[0][0].device,
            )
        self._entries: list[FadeMemEntry] = []
        self.next_frame = 0
        self.append(0, prefilled, descriptors=descriptors)

    @property
    def entries(self) -> tuple[FadeMemEntry, ...]:
        """Return the ordered temporal metadata."""
        return tuple(self._entries)

    @property
    def count(self) -> int:
        """Return the number of historical frame entries."""
        return len(self._entries)

    @property
    def storage_bytes(self) -> int:
        """Return allocated cache bytes, including descriptors and scratch space."""
        tensors = [t for pair in self._storage for t in pair]
        if self._descriptors is not None:
            tensors.append(self._descriptors)
        return sum(t.numel() * t.element_size() for t in tensors)

    def _slice(self, start: int, end: int) -> tuple[slice, ...]:
        index = [slice(None)] * self._storage[0][0].ndim
        index[self.seq_dim] = slice(
            start * self.tokens_per_frame, end * self.tokens_per_frame
        )
        return tuple(index)

    def layers(self, *, full_capacity: bool = False) -> list[LayerKV]:
        """Return history views, optionally including padding for fixed shapes.

        Full-capacity reads must mask entries beyond ``count``. The insertion
        scratch entry is excluded from both views.
        """
        valid = self._slice(0, self.capacity if full_capacity else self.count)
        return [(key[valid], value[valid]) for key, value in self._storage]

    @torch.no_grad()
    def append(
        self,
        start_frame: int,
        chunk: Sequence[LayerKV],
        *,
        descriptors: torch.Tensor | None = None,
    ) -> None:
        """Commit consecutive clean frames, consolidating after each insertion.

        Validate the entire chunk before changing memory. The newest entry is
        excluded from merge selection; the first minimum wins a tie. All entries
        inserted in one chunk measure age at its exclusive end boundary.

        Args:
            start_frame: Absolute first frame, equal to ``next_frame``.
            chunk: Concatenated phase-aligned frame KV for every layer.
            descriptors: Clean-frame descriptors, ``[N, D]``, for cosine selection.

        Raises:
            ValueError: Frames are non-consecutive or K/V layouts differ.
        """
        if (
            isinstance(start_frame, bool)
            or not isinstance(start_frame, int)
            or start_frame != self.next_frame
        ):
            raise ValueError(
                f"FadeMem expected frame {self.next_frame}, got {start_frame}"
            )
        if len(chunk) != len(self._storage):
            raise ValueError("FadeMem chunk must contain every layer")
        count = (
            chunk[0][0].shape[self.seq_dim] if chunk[0][0].ndim > self.seq_dim else 0
        )
        if count < 1 or count % self.tokens_per_frame:
            raise ValueError("FadeMem chunk must contain whole nonempty frames")
        for incoming, buffers in zip(chunk, self._storage, strict=True):
            for tensor, buffer in zip(incoming, buffers, strict=True):
                expected = list(buffer.shape)
                expected[self.seq_dim] = count
                if (
                    list(tensor.shape) != expected
                    or tensor.device != buffer.device
                    or tensor.dtype != buffer.dtype
                ):
                    raise ValueError(
                        "FadeMem chunk shape, dtype, or device differs from storage"
                    )
        frames = count // self.tokens_per_frame
        if self._descriptors is not None and (
            descriptors is None
            or descriptors.shape != (frames, self._descriptors.shape[1])
            or descriptors.device != self._descriptors.device
            or not descriptors.is_floating_point()
            or not bool(torch.isfinite(descriptors).all())
        ):
            raise ValueError(
                "FadeMem descriptors must be finite [N, D] on the cache device"
            )
        for offset in range(frames):
            destination = self._slice(self.count, self.count + 1)
            source = self._slice(offset, offset + 1)
            for (key, value), (key_buffer, value_buffer) in zip(
                chunk, self._storage, strict=True
            ):
                key_buffer[destination].copy_(key[source])
                value_buffer[destination].copy_(value[source])
            if self._descriptors is not None:
                self._descriptors[self.count].copy_(descriptors[offset])
            self._entries.append(FadeMemEntry(start_frame + offset, 1))
            if self.count > self.capacity:
                self._consolidate(start_frame + count // self.tokens_per_frame)
        self.next_frame += count // self.tokens_per_frame

    def _consolidate(self, current_frame: int) -> None:
        # ponytail: O(capacity) selection and compaction suit small KV budgets;
        # use indexed slots and a priority queue if large budgets become useful.
        if self._descriptors is not None:
            # ponytail: full-frame cosine can miss a small moving tool;
            # use spatial tile scores if whole-frame ranking proves inadequate.
            descriptors = self._descriptors[: self.count]
            similarities = torch.nn.functional.cosine_similarity(
                descriptors[:-2], descriptors[1:-1], dim=1
            )
            index = int(similarities.argmax().item())
        else:
            warped = [
                (current_frame - entry.center_frame) ** self.beta
                for entry in self._entries
            ]
            index = min(
                range(self.count - 2), key=lambda i: abs(warped[i] - warped[i + 1])
            )
        left, right = self._entries[index : index + 2]
        span = left.span + right.span
        anchor = self.preserve_anchor and left.center_frame == 0 and index == 0
        if anchor:
            center = 0
        else:
            center = round(
                left.span / span * left.center_frame
                + right.span / span * right.center_frame
            )
            weight = math.sqrt(left.span) / (
                math.sqrt(left.span) + math.sqrt(right.span)
            )
            left_slice, right_slice = (
                self._slice(index, index + 1),
                self._slice(index + 1, index + 2),
            )
            for pair in self._storage:
                for tensor in pair:
                    tensor[left_slice].copy_(
                        tensor[left_slice] * weight
                        + tensor[right_slice]
                        * (
                            math.sqrt(right.span)
                            / (math.sqrt(left.span) + math.sqrt(right.span))
                        )
                    )
            if self._descriptors is not None:
                self._descriptors[index].copy_(
                    self._descriptors[index] * weight
                    + self._descriptors[index + 1] * (1 - weight)
                )
        self._entries[index : index + 2] = [FadeMemEntry(center, span)]
        shift_start = index + 1
        for pair in self._storage:
            for tensor in pair:
                tensor[self._slice(shift_start, self.count)].copy_(
                    tensor[self._slice(shift_start + 1, self.count + 1)].clone()
                )
                tensor[self._slice(self.count, self.count + 1)].zero_()
        if self._descriptors is not None:
            self._descriptors[shift_start : self.count].copy_(
                self._descriptors[shift_start + 1 : self.count + 1].clone()
            )
            self._descriptors[self.count].zero_()

    def attention_positions(
        self, current_start: int, chunk_frames: int
    ) -> tuple[tuple[int, ...], int]:
        """Assign a bounded temporal layout and return the current chunk's start.

        Generalize the released 21-position/three-frame layout: reserve six
        chunk lengths before the current chunk, expanding that range if needed
        to fit the configured budget. During fill, retain absolute positions.
        """
        if (
            isinstance(current_start, bool)
            or not isinstance(current_start, int)
            or current_start != self.next_frame
            or isinstance(chunk_frames, bool)
            or not isinstance(chunk_frames, int)
            or chunk_frames < 1
        ):
            raise ValueError(
                "FadeMem attention must begin at the next frame with a positive chunk size"
            )
        # ponytail: this scales the released three-frame grid to other chunks;
        # its clamped far-history distances need model-specific quality evaluation.
        target = max(6 * chunk_frames, self.capacity + int(not self.preserve_anchor))
        if self.count < self.capacity or current_start < target:
            return tuple(entry.center_frame for entry in self._entries), current_start
        anchor_count = int(self.preserve_anchor)
        normals = self._entries[anchor_count:]
        bridge_count = min(chunk_frames, len(normals))
        far = normals[: len(normals) - bridge_count]
        positions = [0] if anchor_count else []
        previous_gap = target
        for i, entry in enumerate(far):
            lo = bridge_count + 1 + len(far) - i - 1
            hi = previous_gap - 1
            wanted = min(
                max(current_start - entry.center_frame, bridge_count + 1), target - 1
            )
            gap = min(
                range(lo, hi + 1), key=lambda g: abs(g**self.beta - wanted**self.beta)
            )
            positions.append(target - gap)
            previous_gap = gap
        positions.extend(range(target - bridge_count, target))
        return tuple(positions), target

    @torch.no_grad()
    def reset(
        self, prefilled: Sequence[LayerKV], *, descriptors: torch.Tensor | None = None
    ) -> None:
        """Restore frame zero without reallocating the buffers.

        Args:
            prefilled: One frame of original phase-aligned KV for every layer.
            descriptors: Original frame-zero descriptor for cosine selection.
        """
        if any(
            key.ndim <= self.seq_dim or key.shape[self.seq_dim] != self.tokens_per_frame
            for key, _ in prefilled
        ):
            raise ValueError("FadeMem reset requires exactly one seed frame")
        # Validate using a fresh seed write before discarding metadata. Buffers
        # outside the valid prefix are never exposed by layers().
        previous_entries, previous_next = self._entries, self.next_frame
        self._entries, self.next_frame = [], 0
        try:
            self.append(0, prefilled, descriptors=descriptors)
        except ValueError:
            self._entries, self.next_frame = previous_entries, previous_next
            raise


__all__ = ["FadeMemEntry", "FadeMemKVCache"]
