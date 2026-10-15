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

"""headset controller input for Cosmos3-H-Dreams."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cosmos3_h_dreams._runtime._rot import (
    matrix_to_rot6d,
    quat_xyzw_to_matrix,
)
from cosmos3_h_dreams._runtime.hand_client.headset_source import HeadsetSource
from cosmos3_h_dreams._runtime.headset_controller import validate_controller_frame

from .inputs import ACTION_DIM, normalize_and_pad_actions

_HEADSET_TO_CAMERA = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
"""Empirical headset play-space to surgical camera-frame rotation."""

_ARM_LAYOUT = {"left": (0, 9), "right": (14, 23)}
"""Pose and pince indices for the checkpoint's contiguous 14D arm blocks."""

_ENERGY_INDEX = {"left": 10, "right": 24}
_THUMBSTICK_BUTTON_INDEX = {"left": 11, "right": 25}
_THUMBSTICK_INDEX = {"left": 12, "right": 26}
_PRIMARY_BUTTON_FIELD = {"left": "next_scene", "right": "toggle_live"}
_TRANSLATION_SCALE = 10.0
"""Pose encoding scale from controller metres to model translation units."""

_CALIBRATION_HOLD_S = 0.3
_CALIBRATION_POSITION_TOLERANCE_M = 0.02
_CALIBRATION_ROTATION_TOLERANCE_RAD = np.deg2rad(8.0)
_CALIBRATION_TRIGGER_TOLERANCE = 0.1
_JAW_CLOSE_TRIGGER = 0.8
"""Trigger position at which a jaw closes."""

_JAW_OPEN_TRIGGER = 0.6
"""Trigger position below which a closed jaw opens again."""

_RESTART_HOLD_S = 1.0


class GenerationRestartRequested(RuntimeError):
    """Raised when the participant requests a fresh initial-frame rollout."""


def controller_frames_to_actions(
    frames: Sequence[object],
    *,
    normalizer: Path | None = None,
    reference_frame: object | None = None,
    pivot_offsets: dict[str, Sequence[float]] | None = None,
    jaw_open: float = 1.0,
    jaw_closed: float = -2.0,
    jaw_is_closed: dict[str, bool] | None = None,
) -> Tensor:
    """Convert an overlapping headset pose window into model actions.

    The supplied reference frame, or the first frame when omitted, is the
    fixed pose reference for every later row.

    Args:
        frames: Ordered controller frames. ``N + 1`` frames produce ``N`` rows.
        normalizer: Optional checkpoint action normalizer.
        reference_frame: Optional pose anchor retained across action chunks.
        pivot_offsets: Controller-local grip-to-pivot offsets in metres, by side.
            ``None`` uses the tracked grip positions directly.
        jaw_open: Model-space jaw target when the index trigger is released.
        jaw_closed: Model-space jaw target when the index trigger is clicked.
        jaw_is_closed: Mutable per-arm click state retained across chunks.

    Returns:
        Normalized, zero-padded actions shaped ``[N, 64]``.

    Raises:
        ValueError: Fewer than two frames are supplied or the reference frame
            does not contain both tracked controllers.
    """
    if len(frames) < 2:
        raise ValueError("Headset action conversion requires at least two frames")
    if not (
        np.isfinite(jaw_closed) and np.isfinite(jaw_open) and jaw_closed < jaw_open
    ):
        raise ValueError("jaw values must be finite with closed < open")
    validated = [validate_controller_frame(frame) for frame in frames]
    reference_frame = (
        validated[0]
        if reference_frame is None
        else validate_controller_frame(reference_frame)
    )
    actions = np.zeros((len(validated) - 1, ACTION_DIM), dtype=np.float32)
    if jaw_is_closed is None:
        jaw_is_closed = {"left": False, "right": False}
    pivot_offsets = pivot_offsets or {}

    for side, (pose_start, pince_index) in _ARM_LAYOUT.items():
        pivot_offset = np.asarray(
            pivot_offsets.get(side, (0.0, 0.0, 0.0)), dtype=np.float32
        )
        if pivot_offset.shape != (3,) or not np.isfinite(pivot_offset).all():
            raise ValueError(f"{side} headset pivot offset must be three finite values")
        reference = _controller_pose(reference_frame.get(side), pivot_offset)
        if reference is None:
            raise ValueError(
                f"Headset reference frame requires a tracked {side} controller"
            )
        reference_position, reference_rotation = reference
        last_position, last_rotation = reference
        closed = jaw_is_closed[side]

        for row, frame in enumerate(validated[1:]):
            controller = frame.get(side)
            pose = _controller_pose(controller, pivot_offset)
            if pose is not None:
                last_position, last_rotation = pose
            actions[row, pose_start : pose_start + 3] = _TRANSLATION_SCALE * (
                last_position - reference_position
            )
            actions[row, pose_start + 3 : pose_start + 9] = matrix_to_rot6d(
                reference_rotation.T @ last_rotation
            )
            if isinstance(controller, dict):
                tracked = bool(controller.get("tracked", True))
                if tracked:
                    trigger = float(controller["trigger"])
                    if trigger >= _JAW_CLOSE_TRIGGER:
                        closed = True
                    elif trigger <= _JAW_OPEN_TRIGGER:
                        closed = False
                actions[row, _ENERGY_INDEX[side]] = float(
                    tracked and float(controller["grip"]) >= 0.5
                )
                axes = np.asarray(
                    controller.get("axes", (0.0, 0.0)) if tracked else (0.0, 0.0),
                    dtype=np.float32,
                ).reshape(-1)
                actions[
                    row,
                    _THUMBSTICK_INDEX[side] : _THUMBSTICK_INDEX[side] + 2,
                ] = axes[:2] if axes.size >= 2 else 0.0
                actions[row, _THUMBSTICK_BUTTON_INDEX[side]] = float(
                    tracked and bool(controller.get(_PRIMARY_BUTTON_FIELD[side], False))
                )
            actions[row, pince_index] = jaw_closed if closed else jaw_open

        jaw_is_closed[side] = closed

    discrete_values = torch.from_numpy(actions[:, [9, 10, 23, 24]].copy())
    padded = normalize_and_pad_actions(actions, normalizer)
    padded[:, [9, 10, 23, 24]] = discrete_values
    return padded


def _controller_pose(
    controller: object, pivot_offset: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(controller, dict) or not bool(controller.get("tracked", True)):
        return None
    rotation = quat_xyzw_to_matrix(controller["orientation"])
    position = _HEADSET_TO_CAMERA @ (
        np.asarray(controller["position"], dtype=np.float32) + rotation @ pivot_offset
    )
    return position, _HEADSET_TO_CAMERA @ rotation @ _HEADSET_TO_CAMERA.T


class HeadsetActionSource:
    """Build action windows from recorded controller observations."""

    def __init__(
        self,
        *,
        fps: float,
        normalizer: Path | None,
        transport: str,
        relay_url: str | None,
        port: int,
        startup_timeout_s: float,
        manual_xr: bool,
        jaw_open: float = 1.0,
        jaw_closed: float = -2.0,
        pivot_offsets: dict[str, Sequence[float]] | None = None,
        print_actions: bool = False,
        max_buffered_frames: int | None = None,
        source_factory: Callable[..., Any] = HeadsetSource,
    ) -> None:
        self._source = source_factory(
            transport=transport,
            relay_url=relay_url,
            managed_port=port,
            rate_hz=fps,
            startup_timeout_s=startup_timeout_s,
            manual_xr=manual_xr,
        )
        self._normalizer = normalizer
        self._jaw_open = jaw_open
        self._jaw_closed = jaw_closed
        self._jaw_is_closed = {"left": False, "right": False}
        self._pivot_offsets = pivot_offsets
        self._print_actions = print_actions
        self.last_action_capture_ns: int | None = None
        self.last_action_first_capture_ns: int | None = None
        self._fps = fps
        self._rows_emitted = 0
        self._chunks_emitted = 0
        self._frames: deque[dict[str, Any]] = deque(maxlen=max_buffered_frames)
        self._ready = threading.Condition()
        self._closed = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._calibrated = False
        self._calibration_version = 0
        self._calibration_status = "Starting headset controller input"
        self._calibration_fraction = 0.0
        self._calibration_reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._action_reference_frame: dict[str, Any] | None = None
        self._calibration_started_ns = 0
        self._restart_started_ns: int | None = None
        self._restart_requested = False

    def start(self) -> None:
        """Start headset setup and calibration without blocking presentation."""
        self._thread = threading.Thread(
            target=self._sample,
            name="cosmos-h-headset-input",
            daemon=True,
        )
        self._thread.start()

    def calibration_state(self) -> tuple[bool, str, float]:
        """Return readiness plus participant-facing calibration progress."""
        with self._ready:
            if self._error is not None:
                raise RuntimeError("Headset controller sampling failed") from self._error
            return (
                self._calibrated,
                self._calibration_status,
                self._calibration_fraction,
            )

    def next_actions(self, rows: int) -> Tensor:
        """Return an action window from consecutive measured poses."""
        if rows < 1:
            raise ValueError("Headset action rows must be positive")
        needed = rows + 1
        with self._ready:
            calibration_version = self._calibration_version
            self._ready.wait_for(
                lambda: (
                    self._closed
                    or self._error is not None
                    or self._restart_requested
                    or not self._calibrated
                    or calibration_version != self._calibration_version
                    or len(self._frames) >= needed
                )
            )
            if self._error is not None:
                raise RuntimeError("Headset controller sampling failed") from self._error
            if self._closed:
                raise RuntimeError("Headset controller source is closed")
            if (
                self._restart_requested
                or not self._calibrated
                or calibration_version != self._calibration_version
            ):
                raise GenerationRestartRequested
            frames = tuple(list(self._frames)[:needed])
            for _ in range(rows):
                self._frames.popleft()
            reference_frame = self._action_reference_frame
            self.last_action_capture_ns = int(frames[-1]["client_capture_ns"])
            self.last_action_first_capture_ns = int(frames[1]["client_capture_ns"])
        actions = controller_frames_to_actions(
            frames,
            normalizer=self._normalizer,
            reference_frame=reference_frame,
            pivot_offsets=self._pivot_offsets,
            jaw_open=self._jaw_open,
            jaw_closed=self._jaw_closed,
            jaw_is_closed=self._jaw_is_closed,
        )
        if self._print_actions:
            anchor = reference_frame if reference_frame is not None else frames[0]
            for index, (frame, action) in enumerate(
                zip(frames[1:], actions.tolist(), strict=True)
            ):
                row = self._rows_emitted + index
                print(
                    "[headset-actions] "
                    + json.dumps(
                        {
                            "chunk": self._chunks_emitted,
                            "row": row,
                            "model_time_s": (row + 1) / self._fps,
                            "capture_ns": frame["client_capture_ns"],
                            "reference_controllers": {
                                side: anchor.get(side) for side in ("left", "right")
                            },
                            "controllers": {
                                side: frame.get(side) for side in ("left", "right")
                            },
                            "left_xyz": action[0:3],
                            "left_rot6d": action[3:9],
                            "right_xyz": action[14:17],
                            "right_rot6d": action[17:23],
                            "action": action,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
        self._rows_emitted += rows
        self._chunks_emitted += 1
        return actions

    def take_restart_request(self) -> bool:
        """Consume one controller restart request."""
        with self._ready:
            requested = self._restart_requested
            self._restart_requested = False
            return requested

    def request_restart(self) -> None:
        """Wake a blocked action window for a browser-requested restart."""
        with self._ready:
            self._restart_requested = True
            self._frames.clear()
            self._ready.notify_all()

    def set_normalizer(self, normalizer: Path | None) -> None:
        """Change scene action normalization and require a fresh neutral pose."""
        with self._ready:
            self._normalizer = normalizer
            self.reset_calibration()

    def reset_calibration(self) -> None:
        """Discard buffered motion and require a fresh stable neutral pose."""
        with self._ready:
            self._calibration_version += 1
            self._calibrated = False
            self._frames.clear()
            self._restart_requested = False
            self._restart_started_ns = None
            self._reset_calibration("Restarting - hold both controllers still")
            self._ready.notify_all()

    def close(self) -> None:
        """Stop controller sampling and release the managed headset transport."""
        with self._ready:
            if self._closed:
                return
            self._closed = True
            self._ready.notify_all()
        self._source.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _sample(self) -> None:
        try:
            self._source.start()
            with self._ready:
                self._calibration_status = "Track both controllers and release triggers"
                self._ready.notify_all()
            while not self._closed:
                observation = self._source.poll()
                if observation is None or observation.frame is None:
                    time.sleep(0.001)
                    continue
                with self._ready:
                    frame = validate_controller_frame(observation.frame)
                    if not self._calibrated:
                        self._update_calibration(frame)
                    elif self._update_restart(frame):
                        pass
                    else:
                        self._frames.append(frame)
                    self._ready.notify_all()
        except Exception as error:  # noqa: BLE001
            with self._ready:
                self._error = error
                self._ready.notify_all()

    def _update_calibration(self, frame: dict[str, Any]) -> None:
        """Capture a released, tracked neutral pose after a stable hold."""
        controllers = {side: frame.get(side) for side in ("left", "right")}
        if any(
            not isinstance(controller, dict)
            or not bool(controller.get("tracked", True))
            for controller in controllers.values()
        ):
            self._reset_calibration("Track both controllers and release triggers")
            return
        if any(
            float(controller.get("trigger", 0.0)) > _CALIBRATION_TRIGGER_TOLERANCE
            for controller in controllers.values()
            if isinstance(controller, dict)
        ):
            self._reset_calibration("Release both triggers")
            return

        current = {
            side: (
                np.asarray(controller["position"], dtype=np.float32),
                np.asarray(controller["orientation"], dtype=np.float32),
            )
            for side, controller in controllers.items()
            if isinstance(controller, dict)
        }
        capture_ns = frame.get("client_capture_ns")
        now_ns = (
            int(capture_ns)
            if isinstance(capture_ns, int) and not isinstance(capture_ns, bool)
            else time.perf_counter_ns()
        )
        if not self._calibration_reference:
            self._calibration_reference = current
            self._calibration_started_ns = now_ns
            self._calibration_status = "Hold both controllers still"
            return

        position_delta = max(
            float(
                np.linalg.norm(current[side][0] - self._calibration_reference[side][0])
            )
            for side in current
        )
        rotation_delta = max(
            2.0
            * float(
                np.arccos(
                    np.clip(
                        abs(
                            np.dot(
                                current[side][1],
                                self._calibration_reference[side][1],
                            )
                        ),
                        0.0,
                        1.0,
                    )
                )
            )
            for side in current
        )
        if (
            position_delta > _CALIBRATION_POSITION_TOLERANCE_M
            or rotation_delta > _CALIBRATION_ROTATION_TOLERANCE_RAD
        ):
            self._calibration_reference = current
            self._calibration_started_ns = now_ns
            self._calibration_fraction = 0.0
            self._calibration_status = "Hold both controllers still"
            return

        elapsed_s = max(0.0, (now_ns - self._calibration_started_ns) / 1e9)
        self._calibration_fraction = min(1.0, elapsed_s / _CALIBRATION_HOLD_S)
        self._calibration_status = "Hold both controllers still"
        if elapsed_s < _CALIBRATION_HOLD_S:
            return
        self._calibrated = True
        self._calibration_status = "Calibration complete - starting simulation"
        self._action_reference_frame = frame
        self._frames.clear()
        self._frames.append(frame)
        print("[headset] controller calibration complete", flush=True)

    def _update_restart(self, frame: dict[str, Any]) -> bool:
        """Latch a restart after both primary buttons are held for one second."""
        if self._restart_requested:
            return True
        controllers = {side: frame.get(side) for side in ("left", "right")}
        held = all(
            isinstance(controller, dict)
            and bool(controller.get("tracked", True))
            and bool(controller.get(_PRIMARY_BUTTON_FIELD[side], False))
            for side, controller in controllers.items()
        )
        if not held:
            self._restart_started_ns = None
            return False
        capture_ns = frame.get("client_capture_ns")
        now_ns = (
            int(capture_ns)
            if isinstance(capture_ns, int) and not isinstance(capture_ns, bool)
            else time.perf_counter_ns()
        )
        if self._restart_started_ns is None:
            self._restart_started_ns = now_ns
            return True
        if (now_ns - self._restart_started_ns) / 1e9 < _RESTART_HOLD_S:
            return True
        self._restart_requested = True
        self._frames.clear()
        self._calibration_status = "Restart requested"
        print("[headset] generation restart requested", flush=True)
        return True

    def _reset_calibration(self, status: str) -> None:
        self.last_action_capture_ns = None
        self.last_action_first_capture_ns = None
        self._calibration_reference = {}
        self._action_reference_frame = None
        self._jaw_is_closed = {"left": False, "right": False}
        self._calibration_started_ns = 0
        self._calibration_fraction = 0.0
        self._calibration_status = status
        self._rows_emitted = 0
        self._chunks_emitted = 0


__all__ = [
    "GenerationRestartRequested",
    "HeadsetActionSource",
    "controller_frames_to_actions",
]
