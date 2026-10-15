# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model action width, VAE compression, and latent geometry."""

from __future__ import annotations

AGIBOT_DOMAIN_ID = 15
"""Domain ID selecting the AgiBot row of the model's domain-aware action embed."""


MAX_ACTION_DIM = 64
"""Width the model consumes; dimensions 29:64 are reserved and must stay zero."""

FPS = 15.0
"""Conditioning frame rate the AgiBot checkpoint was trained at."""

TEMPORAL_COMPRESSION_FACTOR = 4
"""VAE temporal compression: four actions drive one latent frame."""

SPATIAL_COMPRESSION_FACTOR = 16
"""VAE spatial compression from pixels to latents."""

CONTENT_HEIGHT = 360
CONTENT_WIDTH = 640
"""Corrected 16:9 AgiBot head-camera content size in output pixels."""

LATENT_HEIGHT = (
    CONTENT_HEIGHT + SPATIAL_COMPRESSION_FACTOR - 1
) // SPATIAL_COMPRESSION_FACTOR
LATENT_WIDTH = (
    CONTENT_WIDTH + SPATIAL_COMPRESSION_FACTOR - 1
) // SPATIAL_COMPRESSION_FACTOR
TOKEN_GRID = ((LATENT_HEIGHT + 1) // 2, (LATENT_WIDTH + 1) // 2)
"""Transformer grid after the 2x2 patchifier pads odd latent edges."""


# Exact post-rotations applied after AgiBot forward kinematics. The model's
# wrist deltas live in these side-specific gripper/OpenCV frames, not in
# participant XYZ.


def logical_latent_geometry(
    content_height: int,
    content_width: int,
    *,
    expected: tuple[int, int] | None = (LATENT_HEIGHT, LATENT_WIDTH),
) -> tuple[int, int]:
    """Map a scene's top-left content region to the checkpoint's logical latent.

    Wan's strided encoder rounds an incomplete final spatial cell up. The
    360-pixel corrected AgiBot height therefore becomes 23 latent rows; the
    transformer patchifier pads that odd edge to 24 before forming 2x2 tokens.

    Raises:
        ValueError: If the content size is non-positive or does not match
            ``expected`` after ceil division by the VAE compression factor.
    """
    content_height, content_width = int(content_height), int(content_width)
    factor = SPATIAL_COMPRESSION_FACTOR
    if content_height <= 0 or content_width <= 0:
        raise ValueError(
            f"AgiBot content size must be positive, got {content_height}x{content_width}"
        )
    logical = (
        (content_height + factor - 1) // factor,
        (content_width + factor - 1) // factor,
    )
    if expected is not None and logical != tuple(expected):
        raise ValueError(
            "AgiBot scene is incompatible with this checkpoint's logical model "
            f"geometry: content {content_height}x{content_width} gives latent "
            f"{logical[0]}x{logical[1]}, expected {expected[0]}x{expected[1]}"
        )
    return logical
