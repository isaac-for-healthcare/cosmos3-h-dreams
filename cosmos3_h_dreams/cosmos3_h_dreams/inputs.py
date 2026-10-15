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

"""Image and action preparation for Cosmos3-H-Dreams."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor

from cosmos3_h_dreams._runtime._hand_scenes import Scene
from cosmos3_h_dreams._runtime._tokenizer import tokenize_prompt

from .config import Cosmos3HDreamsConfig

ACTION_DIM = 28
"""Semantic action width of the Cosmos3-H-Dreams checkpoint."""

MODEL_ACTION_DIM = 64
"""Shared Cosmos3 action-projection width after zero padding."""

TEMPORAL_COMPRESSION = 4
"""Action rows consumed for each generated latent frame."""


class ConditioningModel(Protocol):
    """Model operations needed to prepare one offline scene."""

    device: str

    def encode(self, image: Tensor) -> Tensor: ...

    def patchify_encoded(
        self, latent: Tensor, *, content_height: int, content_width: int
    ) -> Tensor: ...


def load_actions(path: Path, normalizer: Path | None = None) -> Tensor:
    """Load, optionally normalize, and pad a 28D action trajectory.

    Args:
        path: ``.npy`` file shaped ``[N, 28]``.
        normalizer: Optional ``.npz`` containing 28D ``offset`` and ``scale``.

    Returns:
        Contiguous float32 actions shaped ``[N, 64]``.

    Raises:
        ValueError: The action or normalizer arrays violate the checkpoint contract.
    """
    actions = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(
            f"actions must have shape [N, {ACTION_DIM}], got {actions.shape}"
        )
    if actions.shape[0] == 0 or actions.shape[0] % TEMPORAL_COMPRESSION:
        raise ValueError(
            "action rows must be nonzero and divisible by "
            f"{TEMPORAL_COMPRESSION}, got {actions.shape[0]}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("actions contain non-finite values")

    return normalize_and_pad_actions(actions, normalizer)


def normalize_and_pad_actions(
    actions: np.ndarray,
    normalizer: Path | None = None,
) -> Tensor:
    """Normalize 28D actions and zero-pad them to the model width.

    Args:
        actions: Finite action array shaped ``[N, 28]``.
        normalizer: Optional ``.npz`` containing 28D ``offset`` and ``scale``.

    Returns:
        Contiguous float32 actions shaped ``[N, 64]``.

    Raises:
        ValueError: The action or normalizer arrays violate the checkpoint contract.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(
            f"actions must have shape [N, {ACTION_DIM}], got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("actions contain non-finite values")

    if normalizer is not None:
        with np.load(normalizer, allow_pickle=False) as archive:
            if "offset" not in archive or "scale" not in archive:
                raise ValueError("normalizer must contain offset and scale arrays")
            offset = np.asarray(archive["offset"], dtype=np.float32)
            scale = np.asarray(archive["scale"], dtype=np.float32)
        if offset.shape != (ACTION_DIM,) or scale.shape != (ACTION_DIM,):
            raise ValueError("normalizer offset and scale must both be 28D")
        if not np.isfinite(offset).all() or not np.isfinite(scale).all():
            raise ValueError("normalizer contains non-finite values")
        if np.any(scale <= 0):
            raise ValueError("normalizer scale must be positive")
        actions = (actions - offset) / scale

    padded = np.zeros((actions.shape[0], MODEL_ACTION_DIM), dtype=np.float32)
    padded[:, :ACTION_DIM] = actions
    return torch.from_numpy(padded)


def prepare_scene(config: Cosmos3HDreamsConfig, model: ConditioningModel) -> Scene:
    """Prepare an initial frame, prompt, and action trajectory for inference."""
    rgb = _load_initial_frame(config.initial_frame, config.width, config.height)
    latent = model.encode(torch.from_numpy(np.ascontiguousarray(rgb)))
    init_latent = model.patchify_encoded(
        latent,
        content_height=config.height,
        content_width=config.width,
    )
    text_ids = tokenize_prompt(
        config.prompt,
        height=config.height,
        width=config.width,
    ).to(model.device)
    return Scene(
        name=config.initial_frame.stem,
        init_latent=init_latent,
        init_rgb=rgb,
        text_ids=text_ids,
        fps=config.fps,
        content_height=config.height,
        content_width=config.width,
        replay_actions=(
            load_actions(config.actions, config.normalizer)
            if config.actions is not None
            else torch.empty((0, MODEL_ACTION_DIM), dtype=torch.float32)
        ),
    )


def _load_initial_frame(path: Path, width: int, height: int) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            image = image.convert("RGB").resize(
                (width, height),
                resample=Image.Resampling.BICUBIC,
            )
            return np.asarray(image, dtype=np.uint8).copy()
    except UnidentifiedImageError as error:
        raise ValueError(f"initial frame is not a decodable image: {path}") from error
