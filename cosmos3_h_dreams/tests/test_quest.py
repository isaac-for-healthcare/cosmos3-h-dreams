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

"""CPU checks for Cosmos3-H-Dreams Quest control."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cosmos3_h_dreams.app import Cosmos3HDreamsApplication
from cosmos3_h_dreams.config import Cosmos3HDreamsConfig
from cosmos3_h_dreams.quest import QuestActionSource, controller_frames_to_actions
from cosmos3_h_dreams.session import Cosmos3HDreamsSession
from flashdreams.demo import CanonicalInputWindow
from flashdreams.infra.time import TimeWindow
from PIL import Image

from cosmos3_h_dreams import session as session_module

pytestmark = pytest.mark.ci_cpu


def _controller(
    position: tuple[float, float, float],
    *,
    trigger: float = 0.0,
    grip: float = 0.0,
    axes: tuple[float, float] = (0.0, 0.0),
    primary: bool = False,
    tracked: bool = True,
    side: str,
) -> dict[str, object]:
    return {
        "position": position,
        "orientation": (0.0, 0.0, 0.0, 1.0),
        "trigger": trigger,
        "grip": grip,
        "tracked": tracked,
        "axes": axes,
        "next_scene" if side == "left" else "toggle_live": primary,
    }


def _frame(
    timestamp: int,
    *,
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    return {
        "left": left,
        "right": right,
        "client_capture_ns": timestamp,
    }


def test_quest_frames_map_to_the_28d_checkpoint_layout() -> None:
    reference = _frame(
        1,
        left=_controller((0.0, 0.0, 0.0), side="left"),
        right=_controller((0.0, 0.0, 0.0), side="right"),
    )
    current = _frame(
        2,
        left=_controller(
            (1.0, 2.0, 3.0),
            trigger=0.25,
            grip=0.75,
            axes=(0.2, -0.4),
            primary=True,
            side="left",
        ),
        right=_controller(
            (4.0, 5.0, 6.0),
            trigger=0.5,
            grip=0.25,
            axes=(-0.3, 0.6),
            side="right",
        ),
    )

    actual = controller_frames_to_actions((reference, current)).numpy()

    assert actual.shape == (1, 64)
    np.testing.assert_allclose(actual[0, 0:3], (10.0, -30.0, 20.0))
    np.testing.assert_allclose(actual[0, 3:9], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    assert actual[0, 9] == pytest.approx(1.0)
    np.testing.assert_allclose(actual[0, 10:14], (1.0, 1.0, 0.2, -0.4))
    np.testing.assert_allclose(actual[0, 14:17], (40.0, -60.0, 50.0))
    np.testing.assert_allclose(actual[0, 17:23], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    assert actual[0, 23] == pytest.approx(1.0)
    np.testing.assert_allclose(actual[0, 24:28], (0.0, 0.0, -0.3, 0.6))
    assert not np.count_nonzero(actual[0, 28:])


def test_quest_pivot_offset_removes_rotation_induced_translation() -> None:
    reference = _frame(
        0,
        left=_controller((0.0, 0.0, 0.0), side="left"),
        right=_controller((0.0, 0.0, 0.0), side="right"),
    )
    rotated = _controller((0.0, -0.07, -0.03), side="left")
    rotated["orientation"] = (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))
    current = _frame(
        1,
        left=rotated,
        right=_controller((0.0, 0.0, 0.0), side="right"),
    )

    uncompensated = controller_frames_to_actions((reference, current)).numpy()
    compensated = controller_frames_to_actions(
        (reference, current), pivot_offsets={"left": (0.0, -0.02, -0.05)}
    ).numpy()

    np.testing.assert_allclose(uncompensated[0, :3], (0.0, 0.3, -0.7))
    np.testing.assert_allclose(compensated[0, :3], 0.0, atol=1e-6)
    np.testing.assert_allclose(compensated[0, 3:9], uncompensated[0, 3:9])
    np.testing.assert_allclose(compensated[0, 14:17], 0.0)


def test_application_selects_quest_without_an_action_file(tmp_path) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32)).save(frame)
    application = Cosmos3HDreamsApplication()

    application.init(
        [
            "--checkpoint",
            "/checkpoint",
            "--initial-frame",
            str(frame),
            "--quest",
            "--device",
            "cpu",
        ]
    )

    session = application.create_session()
    assert isinstance(session, Cosmos3HDreamsSession)
    assert application.supports_session_reset
    assert session.config.actions is None
    assert session.config.quest_transport == "webxr"
    assert (session.config.quest_jaw_open, session.config.quest_jaw_closed) == (
        1.0,
        -2.0,
    )


def test_quest_pose_rows_are_relative_to_the_window_reference() -> None:
    frames = tuple(
        _frame(
            index,
            left=_controller((float(index), 0.0, 0.0), side="left"),
            right=_controller((0.0, 0.0, 0.0), side="right"),
        )
        for index in range(3)
    )

    actual = controller_frames_to_actions(frames).numpy()

    np.testing.assert_allclose(actual[:, 0], (10.0, 20.0))


def test_quest_jaw_click_hysteresis_and_tracking_loss() -> None:
    frames = tuple(
        _frame(
            index,
            left=_controller(
                (0.0, 0.0, 0.0),
                trigger=trigger,
                tracked=tracked,
                side="left",
            ),
            right=_controller((0.0, 0.0, 0.0), side="right"),
        )
        for index, (trigger, tracked) in enumerate(
            (
                (0.0, True),
                (0.4, True),
                (0.85, True),
                (0.7, True),
                (0.0, False),
                (0.55, True),
            )
        )
    )
    jaw_state = {"left": False, "right": False}

    actions = controller_frames_to_actions(
        frames, jaw_open=0.9, jaw_closed=0.1, jaw_is_closed=jaw_state
    )

    assert actions[:, 9].tolist() == pytest.approx([0.9, 0.1, 0.1, 0.1, 0.9])
    assert actions[:, 23].tolist() == pytest.approx([0.9] * 5)
    assert jaw_state == {"left": False, "right": False}


def test_quest_jaw_targets_bypass_pose_normalization(tmp_path) -> None:
    normalizer = tmp_path / "normalizer.npz"
    np.savez(
        normalizer,
        offset=np.full(28, 0.5, dtype=np.float32),
        scale=np.full(28, 2.0, dtype=np.float32),
    )
    reference = _frame(
        0,
        left=_controller((0.0, 0.0, 0.0), side="left"),
        right=_controller((0.0, 0.0, 0.0), side="right"),
    )
    clicked = _frame(
        1,
        left=_controller((0.0, 0.0, 0.0), trigger=1.0, grip=1.0, side="left"),
        right=_controller((0.0, 0.0, 0.0), side="right"),
    )
    switched = _frame(
        2,
        left=_controller((0.0, 0.0, 0.0), side="left"),
        right=_controller((0.0, 0.0, 0.0), grip=1.0, side="right"),
    )

    actions = controller_frames_to_actions(
        (reference, clicked, switched), normalizer=normalizer
    )

    assert actions[0, 9].item() == pytest.approx(-2.0)
    assert actions[0, 23].item() == pytest.approx(1.0)
    np.testing.assert_array_equal(actions[:, [10, 24]], ((1.0, 0.0), (0.0, 1.0)))


def test_quest_source_keeps_its_pose_anchor_across_chunks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    frames = [
        _frame(
            index,
            left=_controller(
                (float(index), 0.0, 0.0),
                trigger=(0.0, 0.9, 0.7, 0.7, 0.5)[index],
                side="left",
            ),
            right=_controller((0.0, 0.0, 0.0), side="right"),
        )
        for index in range(5)
    ]
    source = QuestActionSource(
        fps=10.0,
        normalizer=None,
        transport="webxr",
        relay_url=None,
        port=8444,
        startup_timeout_s=1.0,
        manual_xr=False,
        print_actions=True,
        source_factory=lambda **kwargs: _RawQuestSource(frames=[], **kwargs),
    )
    source._calibrated = True
    source._action_reference_frame = frames[0]
    source._frames.extend(frames)

    first = source.next_actions(2)
    second = source.next_actions(2)
    source.reset_calibration()
    source.close()

    np.testing.assert_allclose(first[:, 0], (10.0, 20.0))
    np.testing.assert_allclose(second[:, 0], (30.0, 40.0))
    np.testing.assert_allclose(first[:, 9], (-2.0, -2.0))
    np.testing.assert_allclose(second[:, 9], (-2.0, 1.0))
    assert source._action_reference_frame is None
    records = [
        json.loads(line.removeprefix("[quest-actions] "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[quest-actions] ")
    ]
    assert [record["row"] for record in records] == list(range(4))
    assert [record["chunk"] for record in records] == [0, 0, 1, 1]
    assert records[0]["reference_controllers"]["left"]["position"] == [0.0, 0.0, 0.0]
    assert records[2]["controllers"]["left"]["position"] == [3.0, 0.0, 0.0]
    assert [record["action"] for record in records] == [
        *first.tolist(),
        *second.tolist(),
    ]


def test_measured_windows_preserve_30hz_motion_with_10hz_model_time() -> None:
    source = QuestActionSource(
        fps=10,
        normalizer=None,
        transport="webxr",
        relay_url=None,
        port=8444,
        startup_timeout_s=1,
        manual_xr=False,
        max_buffered_frames=9,
        source_factory=lambda **kwargs: _RawQuestSource(frames=[], **kwargs),
    )

    def frame(index):
        return _frame(
            1_000_000_000 + index * 33_333_333,
            left=_controller(
                (0.01 * index, 0.001 * index * (8 - index), 0), side="left"
            ),
            right=_controller((0, 0, 0), side="right"),
        )

    source._calibrated = True
    source._action_reference_frame = frame(0)
    source._frames.extend(frame(index) for index in range(9))
    try:
        actions = source.next_actions(8)
        indices = np.arange(1, 9)
        np.testing.assert_allclose(actions[:, 0], indices * 0.1, atol=1e-6)
        np.testing.assert_allclose(
            actions[:, 2], indices * (8 - indices) * 0.01, atol=1e-6
        )
        assert (
            source.last_action_capture_ns - source.last_action_first_capture_ns
            == 7 * 33_333_333
        )
        assert source._fps == 10
        assert len(source._frames) == 1
        # A slower presenter retains the newest measured window instead of a backlog.
        source._frames.extend(frame(index) for index in range(9, 21))
        assert len(source._frames) == 9
        np.testing.assert_allclose(
            source.next_actions(8)[:, 0], np.arange(13, 21) * 0.1, atol=1e-6
        )
    finally:
        source.close()


class _Rollout:
    accepts_partial_chunks = False
    last_timing: dict[str, float] = {}

    def __init__(self) -> None:
        self.actions: list[torch.Tensor] = []

    def step(self, actions: torch.Tensor, *, profile: bool = False):
        del profile
        self.actions.append(actions)
        return [torch.zeros((1, 1, 1, 1))]

    def wait_for_pending_cache_update(self, *, block: bool = False) -> float:
        del block
        return 0.0


class _Model:
    device = "cpu"

    def __init__(self, rollout: _Rollout) -> None:
        self._rollout = rollout

    def new_session(self, _scene):
        return self._rollout

    def decode_tensor(self, _latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 3, 2, 2))


class _QuestSource:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        self.calibrated = True
        self.restart_requested = False
        self.calibration_resets = 0
        self.last_action_capture_ns = 1234

    def start(self) -> None:
        self.started = True

    def next_actions(self, rows: int) -> torch.Tensor:
        return torch.full((rows, 64), 0.5)

    def calibration_state(self) -> tuple[bool, str, float]:
        status = "Calibration complete" if self.calibrated else "Hold still"
        return self.calibrated, status, float(self.calibrated)

    def take_restart_request(self) -> bool:
        requested = self.restart_requested
        self.restart_requested = False
        return requested

    def reset_calibration(self) -> None:
        self.calibrated = False
        self.restart_requested = False
        self.calibration_resets += 1

    def close(self) -> None:
        self.closed = True


def test_session_uses_live_quest_actions(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (2, 2)).save(frame)
    scene = SimpleNamespace(fps=15.0, replay_actions=torch.empty((0, 64)))
    rollout = _Rollout()
    source: _QuestSource | None = None

    def create_source(**kwargs: object) -> _QuestSource:
        nonlocal source
        source = _QuestSource(**kwargs)
        return source

    monkeypatch.setattr(session_module, "prepare_scene", lambda *_args: scene)
    session = Cosmos3HDreamsSession(
        config=Cosmos3HDreamsConfig(
            checkpoint="/checkpoint",
            initial_frame=frame,
            prompt="A robot folds a towel.",
            device="cpu",
        ),
        model_factory=lambda **_kwargs: _Model(rollout),
        quest_source_factory=create_source,
    )
    window = CanonicalInputWindow(window=TimeWindow(start_s=0.0, end_s=1.0))

    session.init()
    requirements = session.next_step_requirements()
    assert requirements is not None
    assert requirements.input_frame_count == 16
    result = session.step(window)
    session.close()

    assert source is not None
    assert source.started and source.closed
    assert source.kwargs["transport"] == "webxr"
    assert result.metadata["action_rows"] == 16
    assert result.metadata["action_capture_ns"] == 1234
    assert len(rollout.actions) == 1
    assert tuple(rollout.actions[0].shape) == (16, 64)


def test_live_session_presents_calibration_before_generation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), color=(20, 40, 60)).save(frame)
    scene = SimpleNamespace(
        fps=10.0,
        replay_actions=torch.empty((0, 64)),
        init_rgb=np.full((32, 32, 3), (20, 40, 60), dtype=np.uint8),
    )
    rollout = _Rollout()
    source: _QuestSource | None = None

    def create_source(**kwargs: object) -> _QuestSource:
        nonlocal source
        source = _QuestSource(**kwargs)
        source.calibrated = False
        return source

    monkeypatch.setattr(session_module, "prepare_scene", lambda *_args: scene)
    session = Cosmos3HDreamsSession(
        config=Cosmos3HDreamsConfig(
            checkpoint="/checkpoint",
            initial_frame=frame,
            prompt="A robot folds a towel.",
            device="cpu",
            fps=10.0,
        ),
        model_factory=lambda **_kwargs: _Model(rollout),
        quest_source_factory=create_source,
    )
    window = CanonicalInputWindow(window=TimeWindow(start_s=0.0, end_s=0.1))

    session.init()
    setup_requirements = session.next_step_requirements()
    assert setup_requirements is not None
    assert setup_requirements.input_frame_count == 1
    setup = session.step(window)

    assert setup.metadata["action_rows"] == 0
    assert setup.metadata["calibration"] == "Hold still"
    assert tuple(setup.video_chunk.shape) == (1, 3, 32, 32)
    assert not rollout.actions

    assert source is not None
    source.calibrated = True
    live_requirements = session.next_step_requirements()
    assert live_requirements is not None
    assert live_requirements.input_frame_count == 16
    session.step(window)
    session.close()

    assert len(rollout.actions) == 1


def test_session_reset_rebuilds_rollout_from_initial_frame(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32)).save(frame)
    scene = SimpleNamespace(
        fps=10.0,
        replay_actions=torch.empty((0, 64)),
        init_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
    )
    rollouts: list[_Rollout] = []

    class _ResetModel(_Model):
        def __init__(self) -> None:
            pass

        def new_session(self, _scene):
            rollout = _Rollout()
            rollouts.append(rollout)
            return rollout

    source: _QuestSource | None = None

    def create_source(**kwargs: object) -> _QuestSource:
        nonlocal source
        source = _QuestSource(**kwargs)
        return source

    monkeypatch.setattr(session_module, "prepare_scene", lambda *_args: scene)
    session = Cosmos3HDreamsSession(
        config=Cosmos3HDreamsConfig(
            checkpoint="/checkpoint",
            initial_frame=frame,
            prompt="A robot folds a towel.",
            device="cpu",
            fps=10.0,
        ),
        model_factory=lambda **_kwargs: _ResetModel(),
        quest_source_factory=create_source,
    )
    window = CanonicalInputWindow(window=TimeWindow(start_s=0.0, end_s=1.0))

    session.init()
    session.step(window)
    assert source is not None
    source.restart_requested = True
    gesture_setup = session.step(window)

    assert len(rollouts) == 2
    assert source.calibration_resets == 1
    assert gesture_setup.metadata["action_rows"] == 0

    session.reset()

    assert len(rollouts) == 3
    assert source.calibration_resets == 2
    assert session.next_step_requirements().input_frame_count == 1
    setup = session.step(window)
    session.close()

    assert setup.step_index == 0
    assert setup.metadata["action_rows"] == 0


class _RawQuestSource:
    def __init__(self, *, frames: list[dict[str, object]], **_kwargs: object) -> None:
        self.frames = frames
        self.stopped = False

    def start(self) -> None:
        pass

    def poll(self) -> SimpleNamespace | None:
        if self.frames:
            return SimpleNamespace(frame=self.frames.pop(0))
        return None

    def stop(self) -> None:
        self.stopped = True


def test_quest_source_calibrates_before_emitting_actions() -> None:
    frames = [
        _frame(
            timestamp,
            left=_controller((0.0, 0.0, 0.0), side="left"),
            right=_controller((0.2, 0.0, 0.0), side="right"),
        )
        for timestamp in (0, 100_000_000, 200_000_000, 300_000_000, 400_000_000)
    ]
    raw_source: _RawQuestSource | None = None

    def create_source(**kwargs: object) -> _RawQuestSource:
        nonlocal raw_source
        raw_source = _RawQuestSource(frames=frames, **kwargs)
        return raw_source

    source = QuestActionSource(
        fps=10.0,
        normalizer=None,
        transport="webxr",
        relay_url=None,
        port=8444,
        startup_timeout_s=1.0,
        manual_xr=False,
        source_factory=create_source,
    )
    source.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        calibrated, _status, fraction = source.calibration_state()
        if calibrated:
            break
        time.sleep(0.001)
    else:
        pytest.fail("Quest source did not calibrate")

    actions = source.next_actions(1)
    source.close()

    assert fraction == pytest.approx(1.0)
    assert tuple(actions.shape) == (1, 64)
    assert raw_source is not None and raw_source.stopped


def test_quest_source_requests_restart_after_both_primary_buttons_are_held() -> None:
    frames = [
        _frame(
            timestamp,
            left=_controller(
                (0.0, 0.0, 0.0), primary=timestamp >= 400_000_000, side="left"
            ),
            right=_controller(
                (0.2, 0.0, 0.0), primary=timestamp >= 400_000_000, side="right"
            ),
        )
        for timestamp in range(0, 1_500_000_000, 100_000_000)
    ]

    source = QuestActionSource(
        fps=10.0,
        normalizer=None,
        transport="webxr",
        relay_url=None,
        port=8444,
        startup_timeout_s=1.0,
        manual_xr=False,
        source_factory=lambda **kwargs: _RawQuestSource(frames=frames, **kwargs),
    )
    source.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if source.take_restart_request():
            break
        time.sleep(0.001)
    else:
        pytest.fail("Quest source did not request a restart")

    source.reset_calibration()
    calibrated, status, fraction = source.calibration_state()
    source.close()

    assert not calibrated
    assert status == "Restarting - hold both controllers still"
    assert fraction == 0.0


@pytest.mark.parametrize("reset_before_read", [False, True])
def test_recalibration_wakes_a_blocked_action_window(reset_before_read):
    import threading

    from cosmos3_h_dreams.quest import GenerationRestartRequested

    source = QuestActionSource(
        fps=10,
        normalizer=None,
        transport="webxr",
        relay_url=None,
        port=8444,
        startup_timeout_s=1,
        manual_xr=False,
        source_factory=lambda **kwargs: _RawQuestSource(frames=[], **kwargs),
    )
    source._calibrated = True
    if reset_before_read:
        source.reset_calibration()
    waiting = threading.Event()
    original_wait = source._ready.wait_for

    def wait_for(predicate):
        waiting.set()
        return original_wait(predicate)

    source._ready.wait_for = wait_for
    errors = []

    def read_actions():
        try:
            source.next_actions(8)
        except RuntimeError as error:
            errors.append(error)

    reader = threading.Thread(target=read_actions)
    reader.start()
    try:
        assert waiting.wait(timeout=1)
        if not reset_before_read:
            source.reset_calibration()
        reader.join(timeout=1)
        assert not reader.is_alive(), (
            "Recalibration left the model waiting for old poses"
        )
        assert len(errors) == 1 and isinstance(errors[0], GenerationRestartRequested)
    finally:
        source.close()
        reader.join(timeout=1)
