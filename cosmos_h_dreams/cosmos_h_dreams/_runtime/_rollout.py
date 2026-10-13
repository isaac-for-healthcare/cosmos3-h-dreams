# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic frame and denoising seed strides for cached generation."""

from __future__ import annotations

# Per-frame and per-step SDE noise seed offsets:
# step_seed = seed + frame_idx * _FRAME_SEED_STRIDE + step_idx * _STEP_SEED_STRIDE.
_FRAME_SEED_STRIDE = 1_000_003
_STEP_SEED_STRIDE = 9_176
