# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""headset controller schema, validation, and calibration.

Pure NumPy and stdlib. This module defines the wire contract between headset
controller producers and the demo; the managed transports live in
:mod:`cosmos3_h_dreams._runtime.hand_client.headset_source`.

Messages are JSON objects of the form::

    {"type": "xr_frame", "t_client": <float>, "controllers": {"left": {...},
     "right": {...}}}

Each controller carries ``position[3]`` in meters, ``orientation[4]`` as an
``(x, y, z, w)`` quaternion, and either explicit ``trigger``/``grip`` scalars or
a WebXR ``buttons`` array. The headset pose is deliberately not part of this
schema: calibration uses gravity plus controller placement instead.

"""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np

from cosmos3_h_dreams._runtime._rot import quat_xyzw_to_matrix, unit_vector

CONTROLLER_CALIBRATION_EPOCH_KEY = "controller_calibration_epoch"
"""Frame field binding a model window to one participant calibration."""

WEBXR_TO_MODEL = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
"""OpenXR is +X right, +Y up, -Z forward; the model frame is +Y down, +Z forward."""

WORLD_DOWN = np.array([0.0, -1.0, 0.0], dtype=np.float32)
"""Gravity direction in OpenXR world coordinates."""

CONTROLLER_RAY_LOCAL = np.array([0.0, 0.0, -1.0], dtype=np.float32)
"""An OpenXR controller points along its own local -Z axis."""

SIDES = ("left", "right")

ControllerPayload = dict[str, Any]
"""One validated controller: pose, analog values, and named button state."""

ControllerFrame = dict[str, Any]
"""One observation: a payload per side plus ``client_capture_ns``."""

# WebXR ``xr-standard`` gamepad button indices.
_BUTTON_TRIGGER = 0
_BUTTON_GRIP = 1
_BUTTON_PRIMARY = 4  # right A / left X
_BUTTON_SECONDARY = 5  # right B / left Y


def as_mapping(value: object) -> dict[str, Any] | None:
    """Return ``value`` as a string-keyed mapping, or ``None`` if it is not one.

    Relay and HTTP payloads arrive as untrusted JSON, so every field access goes
    through this first.
    """
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _button_value(buttons: object, index: int) -> float:
    """Read an analog button, tolerating both scalar and object encodings."""
    if not isinstance(buttons, list) or index >= len(buttons):
        return 0.0
    button: Any = buttons[index]
    mapping = as_mapping(button)
    value = (
        mapping.get("v", mapping.get("value", 0.0)) if mapping is not None else button
    )
    return float(np.clip(float(value), 0.0, 1.0))


def _button_pressed(buttons: object, index: int) -> bool:
    """Read a digital button, falling back to an analog threshold."""
    if not isinstance(buttons, list) or index >= len(buttons):
        return False
    mapping = as_mapping(buttons[index])
    if mapping is not None:
        if "pressed" in mapping:
            return bool(mapping["pressed"])
        if "p" in mapping:
            return bool(mapping["p"])
    return _button_value(buttons, index) >= 0.5


def _thumbstick_axes(value: dict[str, Any]) -> np.ndarray | None:
    """Read the thumbstick pair, or ``None`` when the source omits it.

    The ``xr-standard`` mapping puts the thumbstick at ``axes[2:4]``; two-axis
    controllers report it directly. Absence stays absent rather than defaulting
    to zero, so a relay that never sends ``gamepad.axes`` remains detectable
    instead of silently pinning camera control at identity.

    An empty array counts as absent. WebXR reports ``gamepad.axes: []`` for a
    controller it has not finished enumerating, which is a normal transient
    rather than a malformed frame.
    """
    if "axes" not in value:
        return None
    axes = np.asarray(value["axes"], dtype=np.float32).reshape(-1)
    if axes.size == 0:
        return None
    if axes.size == 2:
        stick = axes
    elif axes.size >= 4:
        stick = axes[2:4]
    else:
        raise ValueError(
            f"Headset controller axes must hold 2 or 4+ values, got {axes.size}"
        )
    if not np.isfinite(stick).all():
        raise ValueError("Headset controller axes must be finite")
    return np.clip(stick, -1.0, 1.0)


def _controller_payload(raw: object, *, side: str) -> ControllerPayload | None:
    """Validate one controller into the demo's wire schema, or ``None``."""
    value = as_mapping(raw)
    if value is None:
        return None
    position = np.asarray(value.get("position"), dtype=np.float32).reshape(-1)
    orientation = np.asarray(value.get("orientation"), dtype=np.float32).reshape(-1)
    if position.shape != (3,) or orientation.shape != (4,):
        raise ValueError("Headset controller requires position[3] and orientation[4]")
    if not np.isfinite(position).all() or not np.isfinite(orientation).all():
        raise ValueError("Headset controller pose must be finite")
    norm = float(np.linalg.norm(orientation))
    if norm < 1e-8:
        raise ValueError("Headset controller quaternion has zero norm")
    buttons = value.get("buttons")
    payload: dict[str, Any] = {
        "position": position.tolist(),
        "orientation": (orientation / norm).tolist(),
        "trigger": float(
            np.clip(
                float(value.get("trigger", _button_value(buttons, _BUTTON_TRIGGER))),
                0.0,
                1.0,
            )
        ),
        "grip": float(
            np.clip(
                float(value.get("grip", _button_value(buttons, _BUTTON_GRIP))),
                0.0,
                1.0,
            )
        ),
        "tracked": bool(value.get("tracked", True)),
        "recenter": bool(
            value.get(
                "recenter",
                side == "right" and _button_pressed(buttons, _BUTTON_SECONDARY),
            )
        ),
        "toggle_live": bool(
            value.get(
                "toggle_live",
                side == "right" and _button_pressed(buttons, _BUTTON_PRIMARY),
            )
        ),
        "next_scene": bool(
            value.get(
                "next_scene",
                side == "left" and _button_pressed(buttons, _BUTTON_PRIMARY),
            )
        ),
        "previous_scene": bool(
            value.get(
                "previous_scene",
                side == "left" and _button_pressed(buttons, _BUTTON_SECONDARY),
            )
        ),
    }
    axes = _thumbstick_axes(value)
    if axes is not None:
        payload["axes"] = axes.tolist()
    if "battery" in value and value["battery"] is not None:
        battery = float(value["battery"])
        if not np.isfinite(battery) or not 0.0 <= battery <= 1.0:
            raise ValueError("Headset controller battery must be between 0 and 1")
        payload["battery"] = battery
    if "gripper_open" in value:
        gripper_open = value["gripper_open"]
        if isinstance(gripper_open, (bool, np.bool_)) or not isinstance(
            gripper_open, (int, float, np.integer, np.floating)
        ):
            raise ValueError("controller gripper_open must be a number in [0, 1]")
        gripper_open = float(gripper_open)
        if not np.isfinite(gripper_open) or not 0.0 <= gripper_open <= 1.0:
            raise ValueError("controller gripper_open must be a number in [0, 1]")
        payload["gripper_open"] = gripper_open
    return payload


def parse_xr_frame(
    raw: object, *, received_ns: int | None = None
) -> ControllerFrame | None:
    """Validate one relay ``xr_frame`` into the demo wire schema.

    Args:
        raw: Decoded JSON message from the relay.
        received_ns: Host receive time; defaults to the current perf counter.

    Returns:
        The validated frame, or ``None`` if the message is not an ``xr_frame``.

    Raises:
        ValueError: If the message claims to be an ``xr_frame`` but its
            controller payloads are malformed.
    """
    payload = as_mapping(raw)
    if payload is None or payload.get("type") != "xr_frame":
        return None
    controllers = as_mapping(payload.get("controllers"))
    if controllers is None:
        raise ValueError("xr_frame controllers must be an object")
    frame: ControllerFrame = {
        side: _controller_payload(controllers.get(side), side=side) for side in SIDES
    }
    frame["client_capture_ns"] = int(received_ns or time.perf_counter_ns())
    frame["relay_timestamp"] = payload.get("t_client")
    return frame


# Optional client timestamps carried only while a latency benchmark runs. The
# validator rebuilds frames from known keys, so anything not listed here is
# dropped on the way in.
TRACE_KEYS = ("client_upload_queued_ns", "client_upload_started_ns")

SOURCE_PROVENANCE_KEYS = (
    "source_frame_index",
    "source_capture_ns",
    "source_time_origin_ns",
)
"""Stable producer identity carried through controller-window HTTP uploads."""

SOURCE_FRAME_INDEX_IS_EXACT_KEY = "source_frame_index_is_exact"
"""Qualifier distinguishing real source indices from interpolated lineage."""


def validate_controller_frame(raw: object) -> ControllerFrame:
    """Validate a controller observation received over HTTP.

    Raises:
        ValueError: If the frame is not an object, lacks an integer
            ``client_capture_ns``, or carries a malformed participant basis or
            calibration epoch.
    """
    value = as_mapping(raw)
    if value is None:
        raise ValueError("controller frame must be an object")
    frame: ControllerFrame = {
        side: _controller_payload(value.get(side), side=side) for side in SIDES
    }
    timestamp = value.get("client_capture_ns")
    if not isinstance(timestamp, int):
        raise ValueError("controller frame requires integer client_capture_ns")
    frame["client_capture_ns"] = timestamp
    if "participant_basis" in value:
        frame["participant_basis"] = validate_participant_basis(
            value["participant_basis"]
        ).tolist()
    if CONTROLLER_CALIBRATION_EPOCH_KEY in value:
        epoch = value[CONTROLLER_CALIBRATION_EPOCH_KEY]
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ValueError("controller calibration epoch must be an integer")
        if epoch < 1:
            raise ValueError("controller calibration epoch must be positive")
        frame[CONTROLLER_CALIBRATION_EPOCH_KEY] = epoch
    provenance_present = [key for key in SOURCE_PROVENANCE_KEYS if key in value]
    if provenance_present and len(provenance_present) != len(SOURCE_PROVENANCE_KEYS):
        missing = sorted(set(SOURCE_PROVENANCE_KEYS).difference(provenance_present))
        raise ValueError(
            "controller frame source provenance is missing " + ", ".join(missing)
        )
    for key in provenance_present:
        stamp = value[key]
        if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0:
            raise ValueError(f"controller frame {key} must be a non-negative integer")
        frame[key] = stamp
    if SOURCE_FRAME_INDEX_IS_EXACT_KEY in value:
        if not provenance_present:
            raise ValueError(
                "controller frame source_frame_index_is_exact requires source "
                "provenance"
            )
        exact = value[SOURCE_FRAME_INDEX_IS_EXACT_KEY]
        if not isinstance(exact, bool):
            raise ValueError(
                "controller frame source_frame_index_is_exact must be a boolean"
            )
        frame[SOURCE_FRAME_INDEX_IS_EXACT_KEY] = exact
    for key in TRACE_KEYS:
        stamp = value.get(key)
        if isinstance(stamp, int):
            frame[key] = stamp
    return frame


def validate_participant_basis(value: Any) -> np.ndarray:
    """Return a participant basis as ``float32`` after checking it is a rotation."""
    basis = np.asarray(value, dtype=np.float32)
    if (
        basis.shape != (3, 3)
        or not np.isfinite(basis).all()
        or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(basis), 1.0, atol=1e-4)
    ):
        raise ValueError(
            "controller participant_basis must be right-handed orthonormal"
        )
    return basis


def controller_sides(frame: ControllerFrame | None) -> dict[str, ControllerPayload]:
    """Return the present controller payloads, keyed by side.

    Callers use this instead of subscripting a possibly-``None`` frame, which
    keeps the "both present" check and the access in one place.
    """
    if not frame:
        return {}
    present = {
        side: payload for side in SIDES if isinstance(payload := frame.get(side), dict)
    }
    return present if len(present) == len(SIDES) else {}


def has_thumbstick_axes(frame: ControllerFrame | None) -> bool:
    """Report whether either controller carried a WebXR thumbstick pair."""
    if not frame:
        return False
    return any(
        isinstance(payload := frame.get(side), dict) and "axes" in payload
        for side in SIDES
    )


def has_both_controllers(frame: ControllerFrame | None) -> bool:
    """Report whether both controllers are present in the frame."""
    return bool(controller_sides(frame))


def both_controllers_tracked(frame: ControllerFrame | None) -> bool:
    """Report whether both controllers are present and currently tracked."""
    sides = controller_sides(frame)
    return bool(sides) and all(
        bool(payload.get("tracked", True)) for payload in sides.values()
    )


def controller_separation_basis(frame: ControllerFrame) -> np.ndarray:
    """Derive a stable level frame from controller placement only.

    Controller pointing rays are deliberately excluded. When the controllers
    point toward one another those rays cancel and become numerically unstable,
    while the horizontal left-to-right hand axis remains well defined.
    """
    sides = controller_sides(frame)
    if not sides:
        raise ValueError("both headset controllers are required for calibration")

    hand_right = np.asarray(sides["right"]["position"], dtype=np.float32) - np.asarray(
        sides["left"]["position"], dtype=np.float32
    )
    hand_right = hand_right - WORLD_DOWN * float(np.dot(hand_right, WORLD_DOWN))
    hand_right = unit_vector(hand_right, name="controller separation")
    hand_forward = unit_vector(np.cross(hand_right, WORLD_DOWN), name="hand forward")
    return np.stack([hand_right, WORLD_DOWN, hand_forward], axis=0).astype(np.float32)


def controller_participant_basis(frame: ControllerFrame) -> np.ndarray:
    """Derive a level participant frame from the controllers at calibration.

    WebXR world yaw follows a possibly off-axis headset, so participant forward
    comes from the controllers' neutral pointing direction, cross-checked
    against the left-to-right hand axis. Gravity fixes model +Y downward.
    Headset pitch, roll, position, and viewing direction never participate.

    Returns:
        A ``[3,3]`` right-handed basis whose rows are ``right``, ``down``, and
        ``forward``.

    Raises:
        ValueError: If either controller is missing or the pose is degenerate.
    """
    sides = controller_sides(frame)
    if not sides:
        raise ValueError("both headset controllers are required for calibration")

    separation_basis = controller_separation_basis(frame)
    hand_right = separation_basis[0]
    hand_forward = separation_basis[2]

    pointing = np.mean(
        [
            quat_xyzw_to_matrix(np.asarray(payload["orientation"], dtype=np.float32))
            @ CONTROLLER_RAY_LOCAL
            for payload in sides.values()
        ],
        axis=0,
    )
    pointing = pointing - WORLD_DOWN * float(np.dot(pointing, WORLD_DOWN))

    forward = hand_forward
    if float(np.linalg.norm(pointing)) >= 1e-6:
        controller_forward = unit_vector(pointing, name="controller forward")
        # Some runtimes expose the physical ray with the opposite local-Z sign;
        # anatomical left/right resolves that ambiguity.
        if float(np.dot(controller_forward, hand_forward)) < 0.0:
            controller_forward = -controller_forward
        controller_right = unit_vector(
            np.cross(WORLD_DOWN, controller_forward), name="controller right"
        )
        if float(np.dot(controller_right, hand_right)) >= 0.5:
            forward = controller_forward

    right_axis = unit_vector(np.cross(WORLD_DOWN, forward), name="participant right")
    if float(np.dot(right_axis, hand_right)) < 0.0:
        right_axis = -right_axis
        forward = -forward
    forward = unit_vector(np.cross(right_axis, WORLD_DOWN), name="participant forward")
    return np.stack([right_axis, WORLD_DOWN, forward], axis=0).astype(np.float32)


def controller_wrist_offsets(
    frame: ControllerFrame,
    basis: np.ndarray,
    *,
    orientation_basis: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Calibrate each controller orientation so the pose at rest is identity.

    Returns:
        Per-side ``[3,3]`` offsets that right-multiply a mapped controller
        rotation to cancel the operator's grip orientation.
    """
    sides = controller_sides(frame)
    if not sides:
        raise ValueError("both headset controllers are required for wrist calibration")
    basis = np.asarray(basis, dtype=np.float32)
    orientation_basis = (
        basis
        if orientation_basis is None
        else np.asarray(orientation_basis, dtype=np.float32)
    )
    return {
        side: (
            orientation_basis
            @ quat_xyzw_to_matrix(np.asarray(payload["orientation"], dtype=np.float32))
        ).T.astype(np.float32)
        for side, payload in sides.items()
    }
