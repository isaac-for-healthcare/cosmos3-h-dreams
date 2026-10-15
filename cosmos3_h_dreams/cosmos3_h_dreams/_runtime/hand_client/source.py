# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input sources and the lifecycle that keeps the UI responsive while they start.

A headset stack can take minutes to come up: a relay process starts, a headset is
woken over ADB, and a browser has to enter immersive WebXR. Starting that on the
UI thread would freeze the demo, so :class:`InputSourceManager` starts sources on
a worker thread and reports ``PREPARING`` until they are ready.

Two properties matter and are easy to get wrong:

* observations captured before a source stopped must never be attributed to the
  one that replaced it, which the generation counter enforces, and
* teardown must not race an in-flight ``poll``, which the call gate enforces.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, cast, runtime_checkable

# Cadence used while no source is ready, so the app still ticks and renders.
_IDLE_POLL_HZ = 30.0
# Ceiling once it is. High enough that presentation is smooth and the controller
# is sampled above any control rate we ask for, low enough not to spin.
_TARGET_FPS = 120.0


class SourceState(str, Enum):
    """Lifecycle state of the active input source."""

    IDLE = "idle"
    PREPARING = "preparing"
    READY = "ready"
    ERROR = "error"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class Observation:
    """One controller sample tagged with the source that produced it.

    Attributes:
        frame: Controller state in the headset wire schema, or ``None`` when the
            source has no sample yet.
        capture_ns: Host capture time.
        source: Name of the producing source.
        generation: Lifecycle generation, used to reject stale samples.
    """

    frame: dict | None
    capture_ns: int
    source: str
    generation: int = 0


@runtime_checkable
class InputSource(Protocol):
    """A source of controller observations.

    ``start`` may block for a long time; the manager always calls it on a worker
    thread. ``poll`` must be cheap and non-blocking beyond its own pacing.
    """

    name: str

    def start(self) -> None:
        """Bring the source up. May block."""

    def stop(self) -> None:
        """Release everything the source owns. Must be idempotent."""

    def poll(self) -> Observation | None:
        """Return the newest observation, or ``None`` if none is available."""

    def ready(self) -> bool:
        """Report whether the source is currently producing usable samples."""

    # A source may additionally implement ``begin_live``, ``end_live``,
    # ``configure_live_calibration``, ``prepare_live_window``, and the
    # transactional live-window hooks. The manager invokes those optional hooks
    # at the app/session boundary. Most sources need no hook; replay sources use
    # lifecycle hooks, while Isaac sources own the participant-pose and control
    # transforms applied to model-bound windows.


@dataclass(frozen=True, slots=True)
class SourceStatus:
    """Snapshot of the manager's lifecycle for display and gating."""

    name: str
    state: SourceState
    error: str | None = None
    detail: str = ""

    @property
    def is_ready(self) -> bool:
        return self.state is SourceState.READY


@dataclass(frozen=True, slots=True)
class LiveCalibrationState:
    """Model-facing pose contract installed by an input source.

    Attributes:
        pose_space: Identifier understood by the HandDreams server.
        epoch: Positive generation attached to every transformed live window.
        control_space: Optional absolute-command contract owned by the source.
    """

    pose_space: str
    epoch: int
    control_space: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pose_space, str) or not self.pose_space:
            raise ValueError("live calibration pose space must be a nonempty string")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int):
            raise ValueError("live calibration epoch must be an integer")
        if self.epoch < 1:
            raise ValueError("live calibration epoch must be positive")
        if self.control_space is not None and (
            not isinstance(self.control_space, str) or not self.control_space
        ):
            raise ValueError(
                "live calibration control space must be a nonempty string or None"
            )


@dataclass(frozen=True, slots=True)
class PreparedLiveWindow:
    """Model-facing frames plus an optional source-owned transaction receipt.

    A receipt is opaque to HandDreams. The sender commits it only after the
    server accepts the corresponding window, or aborts it when the upload is
    dropped or rejected. Sources without transactional shaping leave it
    ``None`` and retain their existing one-shot preparation behavior.

    Attributes:
        frames: Model-facing controller observations, oldest first.
        receipt: Opaque transaction object owned by the active input source.
    """

    frames: list[dict]
    receipt: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frames, list) or any(
            not isinstance(frame, dict) for frame in self.frames
        ):
            raise ValueError("prepared live window frames must be a list of objects")


@dataclass
class _CallGate:
    """Counts in-flight calls so teardown can wait for them to drain."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    drained: threading.Condition = field(init=False)
    active: int = 0

    def __post_init__(self) -> None:
        self.drained = threading.Condition(self.lock)

    def enter(self) -> None:
        with self.lock:
            self.active += 1

    def leave(self) -> None:
        with self.lock:
            self.active -= 1
            if self.active <= 0:
                self.drained.notify_all()

    def wait_until_idle(self, timeout: float = 5.0) -> None:
        with self.lock:
            while self.active > 0:
                if not self.drained.wait(timeout=timeout):
                    return


class InputSourceManager:
    """Starts one input source asynchronously and guards access to it.

    Args:
        factory: Builds the source. Called on the worker thread.
        name: Source name reported in status and stamped onto observations.
    """

    def __init__(self, factory: Callable[[], InputSource], *, name: str) -> None:
        self._factory = factory
        self._name = name
        self._condition = threading.Condition()
        self._state = SourceState.IDLE
        self._error: str | None = None
        self._source: InputSource | None = None
        self._starting: InputSource | None = None
        self._generation = 0
        self._gate = _CallGate()
        self._worker: threading.Thread | None = None

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def status(self) -> SourceStatus:
        """Return the current lifecycle snapshot."""
        with self._condition:
            state = self._state
            name = self._name
            error = self._error
            starting = self._starting

        detail = ""
        if state is SourceState.PREPARING:
            detail = f"starting {name}"
            provider = getattr(starting, "startup_detail", None)
            if callable(provider):
                try:
                    source_detail = str(provider()).strip()
                except Exception:  # noqa: BLE001 - status must remain renderable
                    source_detail = ""
                if source_detail:
                    detail = source_detail
        elif state is SourceState.READY:
            detail = f"{name} ready"
        return SourceStatus(name=name, state=state, error=error, detail=detail)

    def select(self, factory: Callable[[], InputSource], *, name: str) -> None:
        """Replace the source with a new one, stopping the current one first.

        The generation bump makes any in-flight start discard itself, so a rapid
        sequence of switches settles on the last request rather than racing.
        """
        with self._condition:
            source, self._source = self._source, None
            starting, self._starting = self._starting, None
            self._factory = factory
            self._name = name
            self._generation += 1
            self._state = SourceState.IDLE
            self._error = None
            self._condition.notify_all()
        self._gate.wait_until_idle()
        for candidate in (source, starting):
            if candidate is not None:
                candidate.stop()
        self.start()

    def restart(self) -> None:
        """Recreate the active source through its complete stop/start lifecycle.

        This is intentionally equivalent to selecting the current source again:
        managed hardware transports get their normal cleanup before a fresh
        instance is constructed, while the manager remains usable by the UI.
        """
        with self._condition:
            factory = self._factory
            name = self._name
        self.select(factory, name=name)

    def start(self) -> None:
        """Begin bringing the source up. Returns immediately."""
        with self._condition:
            if self._state in (SourceState.PREPARING, SourceState.READY):
                return
            self._state = SourceState.PREPARING
            self._error = None
            self._generation += 1
            generation = self._generation
            self._condition.notify_all()
            # The generation is captured here, not inside the worker: reading it
            # there could race a close() that already bumped it, and the worker
            # would then install a source into a closed manager.
            self._worker = threading.Thread(
                target=self._start_source,
                args=(generation,),
                name="input-source-start",
                daemon=True,
            )
            self._worker.start()

    def _start_source(self, generation: int) -> None:
        try:
            source = self._factory()
        except Exception as error:  # noqa: BLE001 - surfaced through status
            self._fail(generation, error)
            return
        # Published before start() so close() can reach a source that is still
        # coming up. Starting a headset stack takes minutes, and that is exactly
        # when an operator interrupts; without this the external relay would be
        # orphaned and keep holding its port. Check the generation first in case
        # close() won the race while the factory was still constructing it.
        with self._condition:
            superseded = (
                generation != self._generation or self._state is SourceState.CLOSED
            )
            if not superseded:
                self._starting = source
        if superseded:
            source.stop()
            return
        try:
            source.start()
        except Exception as error:  # noqa: BLE001 - surfaced through status
            with self._condition:
                self._starting = None
            # start() may have acquired part of its stack before failing.
            source.stop()
            self._fail(generation, error)
            return
        with self._condition:
            self._starting = None
            superseded = (
                generation != self._generation or self._state is SourceState.CLOSED
            )
            if not superseded:
                self._source = source
                self._state = SourceState.READY
                self._condition.notify_all()
        if superseded:
            # A newer request or a close superseded this one while it started.
            source.stop()

    def _fail(self, generation: int, error: Exception) -> None:
        with self._condition:
            if generation == self._generation:
                self._state = SourceState.ERROR
                self._error = str(error)
                self._condition.notify_all()

    def wait_for_ready(self, timeout: float | None = None) -> bool:
        """Block until the source is ready, or until it fails or times out."""
        with self._condition:
            return (
                self._condition.wait_for(
                    lambda: (
                        self._state
                        in (SourceState.READY, SourceState.ERROR, SourceState.CLOSED)
                    ),
                    timeout=timeout,
                )
                and self._state is SourceState.READY
            )

    def poll(self) -> Observation | None:
        """Return the newest observation from the active source, if any.

        Returns ``None`` while the source is starting, so callers keep ticking
        at their own cadence instead of blocking on hardware.
        """
        with self._condition:
            source = self._source
            generation = self._generation
            if source is None or self._state is not SourceState.READY:
                return None
        self._gate.enter()
        try:
            observation = source.poll()
        except Exception as error:  # noqa: BLE001 - a failing source must not kill the app
            with self._condition:
                if generation == self._generation:
                    self._state = SourceState.ERROR
                    self._error = str(error)
            return None
        finally:
            self._gate.leave()
        if observation is None:
            return None
        # Stamp the generation so a consumer can drop samples from a source that
        # has since been stopped.
        return Observation(
            frame=observation.frame,
            capture_ns=observation.capture_ns,
            source=observation.source,
            generation=generation,
        )

    def ready(self) -> bool:
        """Report whether the active source is producing usable samples."""
        with self._condition:
            source = self._source
            if source is None or self._state is not SourceState.READY:
                return False
        try:
            return bool(source.ready())
        except Exception:  # noqa: BLE001
            return False

    def begin_live(self) -> None:
        """Notify the active source that the server accepted a live session.

        Sources without a lifecycle hook are unchanged.  Replay sources use
        this boundary to start from frame zero only after setup is complete.
        """
        self._call_optional_hook("begin_live")

    def end_live(self) -> None:
        """Notify the active source that the live session has ended."""
        self._call_optional_hook("end_live")

    def configure_live_calibration(
        self,
        calibration: dict[str, Any] | None,
        *,
        control: dict[str, Any] | None = None,
    ) -> LiveCalibrationState | None:
        """Configure model-facing calibration and optional source-owned control.

        Args:
            calibration: Canonical wrist poses, neutral triggers, and participant
                basis captured by the HandDreams setup UX.
            control: Optional controller-shaping configuration owned by the
                Isaac Teleop sidecar.

        Returns:
            Installed pose/control-space contract and epoch, or ``None`` when
            the active source leaves calibration in HandDreams.
        """
        if calibration is None and control is not None:
            raise ValueError("controller control requires participant calibration")
        if control is None:
            result = self._call_optional_hook(
                "configure_live_calibration", calibration, require_ready=True
            )
        else:
            result = self._call_optional_hook(
                "configure_live_calibration",
                calibration,
                control=control,
                require_ready=True,
            )
        if result is not None and not isinstance(result, LiveCalibrationState):
            raise RuntimeError("input source returned an invalid calibration state")
        return result

    def prepare_live_window(self, frames: list[dict]) -> list[dict]:
        """Prepare one raw controller window for model upload.

        This compatibility path performs a one-shot transform. New senders use
        :meth:`prepare_live_window_transaction` after coalescing pending windows
        so source state advances only when the server accepts the upload.

        Args:
            frames: Raw canonical controller observations, oldest first.

        Returns:
            Model-facing observations with unchanged ordering and timestamps.
        """
        result = self._call_optional_hook(
            "prepare_live_window", frames, require_ready=True
        )
        if result is None:
            return frames
        if not isinstance(result, list) or len(result) != len(frames):
            raise RuntimeError("input source returned an invalid live window")
        return cast("list[dict]", result)

    def prepare_live_window_transaction(
        self, frames: list[dict], command_sequence: int
    ) -> PreparedLiveWindow:
        """Prepare the surviving pending window without committing source state.

        Args:
            frames: Raw canonical controller observations, oldest first.
            command_sequence: Stable sender sequence for this upload attempt.

        Returns:
            Model-facing frames and an opaque receipt to commit or abort.
        """
        if isinstance(command_sequence, bool) or not isinstance(command_sequence, int):
            raise ValueError("live-window command sequence must be an integer")
        if command_sequence < 0:
            raise ValueError("live-window command sequence must be non-negative")
        result = self._call_optional_hook(
            "prepare_live_window_transaction",
            frames,
            command_sequence=command_sequence,
            require_ready=True,
        )
        if result is None:
            return PreparedLiveWindow(frames=self.prepare_live_window(frames))
        if not isinstance(result, PreparedLiveWindow):
            raise RuntimeError("input source returned an invalid live transaction")
        if len(result.frames) != len(frames):
            raise RuntimeError("input source returned an invalid live window")
        return result

    def commit_live_window(self, prepared: PreparedLiveWindow) -> None:
        """Commit source state after the server accepts ``prepared``."""
        self._finish_live_window("commit_live_window", prepared)

    def abort_live_window(self, prepared: PreparedLiveWindow) -> None:
        """Discard source state after ``prepared`` is dropped or rejected."""
        self._finish_live_window("abort_live_window", prepared)

    def _finish_live_window(self, hook: str, prepared: PreparedLiveWindow) -> None:
        """Resolve one opaque source transaction without inspecting it."""
        if not isinstance(prepared, PreparedLiveWindow):
            raise TypeError("prepared live window has an incompatible type")
        if prepared.receipt is None:
            return
        result = self._call_optional_hook(
            hook,
            prepared.receipt,
            require_ready=True,
            require_hook=True,
        )
        if result is not None:
            raise RuntimeError("input source returned an invalid transaction result")

    def _call_optional_hook(
        self,
        name: str,
        *args: object,
        require_ready: bool = False,
        require_hook: bool = False,
        **kwargs: object,
    ) -> object | None:
        """Call one source hook under the normal in-flight-call gate."""
        with self._condition:
            source = self._source
            if source is None or self._state is not SourceState.READY:
                if name == "begin_live" or require_ready:
                    raise RuntimeError("input source is not ready")
                return None
        callback = getattr(source, name, None)
        if not callable(callback):
            if require_hook:
                raise RuntimeError(f"input source does not implement {name}")
            return None
        self._gate.enter()
        try:
            return callback(*args, **kwargs)
        finally:
            self._gate.leave()

    def close(self) -> None:
        """Stop the source and reject further polls.

        Stops a source that is still starting as well as a ready one, so an
        interrupt during a long startup still releases external processes.
        """
        with self._condition:
            source, self._source = self._source, None
            starting, self._starting = self._starting, None
            self._generation += 1
            self._state = SourceState.CLOSED
            self._condition.notify_all()
        self._gate.wait_until_idle()
        for candidate in (source, starting):
            if candidate is not None:
                candidate.stop()


def idle_poll_interval() -> float:
    """Seconds between ticks while no source is ready."""
    return 1.0 / _IDLE_POLL_HZ


def pace_tick(started: float, *, source_ready: bool) -> None:
    """Sleep whatever is left of this tick's budget.

    Two different budgets, because the two situations want opposite things.
    While a source is coming up the app only has to stay responsive, and
    spinning would burn a core for the minutes a headset stack can take. Once it
    is running the budget is one frame.

    Sleeping the idle cadence unconditionally is what held the window under 30
    fps however fast a frame actually rendered, and it throttled input with it:
    the tick rate is what drives ``poll``, so the controller was sampled at the
    tick rate rather than at the rate the demo asked for.

    Args:
        started: ``perf_counter`` reading from the top of the tick.
        source_ready: Whether the input source is producing samples.
    """
    budget = (1.0 / _TARGET_FPS) if source_ready else idle_poll_interval()
    time.sleep(max(0.0, budget - (time.perf_counter() - started)))
