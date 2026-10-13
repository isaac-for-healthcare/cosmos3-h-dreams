# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-NumPy rotation and rigid-pose helpers shared by the hand demo."""

from __future__ import annotations

from typing import Any

import numpy as np

# Orthonormality is checked with a loose tolerance because these rotations
# arrive from float32 WebXR quaternions and accumulated basis products.


def quat_xyzw_to_matrix(quaternion: Any) -> np.ndarray:
    """Convert ``[...,4]`` ``(x, y, z, w)`` quaternions to ``[...,3,3]`` rotations.

    Args:
        quaternion: Quaternion values in ``(x, y, z, w)`` order. Normalized
            internally, so unnormalized input is accepted.

    Returns:
        The equivalent rotation matrices as ``float32``.

    Raises:
        ValueError: If any quaternion has effectively zero norm.
    """
    values = np.asarray(quaternion, dtype=np.float32)
    if values.shape[-1:] != (4,):
        raise ValueError(f"quaternion must end in 4 values, got {values.shape}")
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("quaternion has zero norm")
    x, y, z, w = np.moveaxis(values / norm, -1, 0)
    matrix = np.empty((*values.shape[:-1], 3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def matrix_to_rot6d(matrix: Any) -> np.ndarray:
    """Encode ``[...,3,3]`` rotations as the first two columns, ``[...,6]``.

    This is the rot6d convention the AgiBot action contract expects; the
    identity rotation maps to ``[1, 0, 0, 0, 1, 0]``.
    """
    values = np.asarray(matrix, dtype=np.float32)
    if values.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end in [3,3], got {values.shape}")
    return np.concatenate((values[..., :, 0], values[..., :, 1]), axis=-1).astype(
        np.float32
    )


def unit_vector(vector: Any, *, name: str = "vector") -> np.ndarray:
    """Normalize a 3D vector, rejecting degenerate input."""
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"cannot normalize a degenerate {name}")
    return (values / norm).astype(np.float32)
