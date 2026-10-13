# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial-frame conditioning and replay data for a Cosmos3-H-Dreams session."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor


@dataclass(frozen=True, slots=True)
class Scene:
    """Initial frame, caption, and replay actions for one model session."""

    name: str
    init_latent: Tensor
    init_rgb: np.ndarray
    text_ids: Tensor
    fps: float
    content_height: int
    content_width: int
    replay_actions: Tensor
