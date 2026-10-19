# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""headset controller input over HTTPS WebXR or a WebSocket relay."""

from __future__ import annotations

import json
import ssl as ssl_module
import threading
import time
from contextlib import suppress

from cosmos3_h_dreams._runtime import headset_controller
from cosmos3_h_dreams._runtime.hand_client.source import Observation
from cosmos3_h_dreams._runtime.hand_client.webxr_server import (
    DEFAULT_PORT,
    WebXRServer,
)

_STALE_MS = 500.0
_RECONNECT_S = 1.0
_TRANSPORTS = ("webxr",)
"""Headset transports exposed through ``--headset-transport``."""

# A malformed producer emits bad frames at full rate, so reports are throttled.
_REJECT_LOG_INTERVAL_S = 5.0

_WEBXR_STAGE_HELP = {
    "": "the page has not loaded; check the headset is awake and on the demo URL",
    "loaded": "the page is loaded but not in VR; press Start Teleop in the headset",
    "ended": "the VR session ended; press Start Teleop in the headset again",
    "failed": "the page could not start a VR session",
    "session": "the page is in VR but no controller is tracked; pick the controllers up",
}
"""WebXR startup stages reported by the hosted page."""

_RELAY_STAGE_HELP = {
    **_WEBXR_STAGE_HELP,
    "": (
        "the relay has not published a fresh tracked controller pair; "
        "open hand-controller and click Arm"
    ),
}
"""Startup guidance while subscribing to an external controller relay."""


class FrameStore:
    """Newest validated browser/relay controller frame and startup status."""

    def __init__(self, *, stage_help: dict[str, str] | None = None) -> None:
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._received = 0
        self._source_time_origin_ns: int | None = None
        self._rejected = 0
        self._next_reject_log = 0.0
        self._stage = ""
        self._stage_detail = ""
        self._stage_help = dict(stage_help or _WEBXR_STAGE_HELP)

    @property
    def received(self) -> int:
        """Return the number of accepted controller frames."""
        with self._lock:
            return self._received

    @property
    def rejected(self) -> int:
        """Return the number of malformed messages or frames."""
        with self._lock:
            return self._rejected

    @property
    def stage(self) -> str:
        """Return the producer's last reported startup stage."""
        with self._lock:
            return self._stage

    def explain(self) -> str:
        """Describe what the selected controller producer is waiting for."""
        with self._lock:
            stage = self._stage
            detail = self._stage_detail
            stage_help = dict(self._stage_help)
        message = stage_help.get(stage, f"the controller source reported {stage!r}")
        return f"{message} ({detail})" if detail else message

    def snapshot(self) -> dict | None:
        """Return a shallow copy of the newest frame, when available."""
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def age_ms(self) -> float | None:
        """Return the newest frame's host-side age in milliseconds."""
        frame = self.snapshot()
        if frame is None:
            return None
        return (time.perf_counter_ns() - int(frame["client_capture_ns"])) / 1e6

    def ready(self, stale_ms: float = _STALE_MS) -> bool:
        """Report whether a fresh, tracked controller pair is available."""
        age = self.age_ms()
        return (
            age is not None
            and age <= stale_ms
            and headset_controller.both_controllers_tracked(self.snapshot())
        )

    def consume(self, message: str | bytes) -> None:
        """Take one message off the wire, tolerating malformed input.

        A bad frame must not drop the socket. Otherwise one unparseable message
        becomes a reconnect loop when the producer sends it again.
        """
        try:
            payload = json.loads(message)
        except (TypeError, ValueError) as error:
            self._note_rejected(error)
            return
        if isinstance(payload, dict) and payload.get("type") == "status":
            self._note_status(payload)
            return
        try:
            frame = headset_controller.parse_xr_frame(payload)
        except (TypeError, ValueError) as error:
            self._note_rejected(error)
            return
        if frame is not None:
            self._store(frame)

    def _note_status(self, payload: dict) -> None:
        """Record a producer startup status."""
        stage = str(payload.get("stage", ""))
        detail = str(payload.get("detail", ""))
        with self._lock:
            changed = stage != self._stage or detail != self._stage_detail
            self._stage = stage
            self._stage_detail = detail
        if changed:
            print(
                f"[headset] source reports {stage}{f': {detail}' if detail else ''}",
                flush=True,
            )

    def _store(self, frame: dict) -> None:
        """Assign source identity on the backend recording's host receive clock."""
        with self._lock:
            first = self._received == 0
            if not all(key in frame for key in headset_controller.SOURCE_PROVENANCE_KEYS):
                # Producers use different timestamp units and may reset their
                # clocks on reconnect. Host receive time matches the backend
                # GT recording; relay_timestamp remains a producer diagnostic.
                source_capture_ns = int(frame["client_capture_ns"])
                if self._source_time_origin_ns is None:
                    self._source_time_origin_ns = source_capture_ns
                frame["source_frame_index"] = self._received
                frame["source_capture_ns"] = source_capture_ns
                frame["source_time_origin_ns"] = self._source_time_origin_ns
            elif self._source_time_origin_ns is None:
                self._source_time_origin_ns = int(frame["source_time_origin_ns"])
            previous = self._latest
            if previous is not None:
                # Preserve pose continuity while exposing the tracking loss.
                for side in headset_controller.SIDES:
                    if frame.get(side) is None and previous.get(side) is not None:
                        held = dict(previous[side])
                        held["tracked"] = False
                        frame[side] = held
            self._latest = frame
            self._received += 1
            missing_axes = not headset_controller.has_thumbstick_axes(frame)
        if first:
            print("[headset] first controller frame received", flush=True)
            if missing_axes:
                print(
                    "[headset] no thumbstick axes in the frames; camera control "
                    "will hold the camera at identity",
                    flush=True,
                )

    def _note_rejected(self, error: Exception) -> None:
        """Count a rejected message, logging at most once per interval."""
        with self._lock:
            self._rejected += 1
            count = self._rejected
            now = time.monotonic()
            report = now >= self._next_reject_log
            if report:
                self._next_reject_log = now + _REJECT_LOG_INTERVAL_S
        if report:
            print(
                f"[headset] ignoring malformed frame ({error}); {count} rejected so far",
                flush=True,
            )


class RelayClient(threading.Thread):
    """Subscribes to a source someone else is hosting, for ``--headset-relay``."""

    def __init__(self, url: str, store: FrameStore) -> None:
        super().__init__(name="headset-relay", daemon=True)
        self.url = url
        self._store = store
        self._running = True
        self._connection = None

    def run(self) -> None:
        from websockets.sync.client import connect

        while self._running:
            try:
                with connect(self.url, open_timeout=5, close_timeout=1) as connection:
                    self._connection = connection
                    print(f"[headset] relay connected: {self.url}", flush=True)
                    for message in connection:
                        if not self._running:
                            return
                        self._store.consume(message)
            except Exception as error:  # noqa: BLE001 - reconnect on any fault
                if self._running:
                    print(f"[headset] relay unavailable ({error}); retrying", flush=True)
                    time.sleep(_RECONNECT_S)
            finally:
                self._connection = None

    def stop(self) -> None:
        self._running = False
        connection = self._connection
        if connection is not None:
            with suppress(Exception):
                connection.close()


class HeadsetSource:
    """Input source backed by a HTTPS WebXR page or WebSocket relay."""

    name = "headset"

    def __init__(
        self,
        *,
        transport: str = "webxr",
        relay_url: str | None = None,
        managed_port: int = DEFAULT_PORT,
        rate_hz: float = 36.0,
        startup_timeout_s: float = 300.0,
        manual_xr: bool = False,
        host: str = "127.0.0.1",
        ssl_context: ssl_module.SSLContext | None = None,
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError("headset source rate must be positive")
        if transport not in _TRANSPORTS:
            raise ValueError(f"unknown headset transport: {transport}")
        self.transport = transport
        self._store = FrameStore(
            stage_help=_RELAY_STAGE_HELP
            if relay_url is not None
            else _WEBXR_STAGE_HELP,
        )
        self._url = relay_url
        self._port = int(managed_port)
        self._host = str(host)
        self._ssl_context = ssl_context
        self._server: WebXRServer | None = None
        self._client: RelayClient | None = None
        self._interval_ns = int(1e9 / rate_hz)
        self._startup_timeout_s = float(startup_timeout_s)
        self._stopped = threading.Event()
        self._next_ns = 0
        self._last_capture_ns = 0

    def start(self) -> None:
        """Start the selected transport and await a fresh tracked pair."""
        if self._stopped.is_set():
            raise RuntimeError("the headset source was stopped before starting")
        self._require_websockets()
        if self._url is None:
            self._serve()
        else:
            print(
                "[headset] waiting for a fresh tracked controller pair from the "
                "relay; open hand-controller and click Arm",
                flush=True,
            )
            client = RelayClient(self._url, self._store)
            self._client = client
            client.start()
        self._await_controllers()

    @staticmethod
    def _require_websockets() -> None:
        """Fail before touching the headset when WebXR support is absent."""
        try:
            import websockets  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Headset WebXR input needs websockets. Install Cosmos3-H-Dreams with "
                "uv sync --package flashdreams-cosmos3-h-dreams."
            ) from error

    def _serve(self) -> None:
        """Host a secure WebXR page for the operator to open."""
        if self._ssl_context is None:
            raise ValueError(
                "Headset input requires an HTTPS server or --headset-relay-url; "
                "use python -m cosmos3_h_dreams.headset_server for browser setup."
            )
        server = WebXRServer(
            on_message=self._store.consume,
            host=self._host,
            port=self._port,
            ssl_context=self._ssl_context,
        )
        server.start()
        self._server = server
        print(f"[headset] open {server.url} in the headset browser", flush=True)

    def _await_controllers(self) -> None:
        """Wait for a compatible, fresh pair or raise a diagnostic timeout."""
        deadline = time.monotonic() + self._startup_timeout_s
        while time.monotonic() < deadline:
            if self._stopped.is_set():
                raise RuntimeError("the headset source was stopped while starting")
            if self._store.ready():
                self._next_ns = time.perf_counter_ns()
                return
            time.sleep(0.1)
        explanation = self._store.explain()
        self.stop()
        producer = "the controller relay" if self._url is not None else "Headset"
        raise TimeoutError(
            f"no fresh tracked controller pair from {producer} within "
            f"{self._startup_timeout_s:.0f}s: "
            f"{explanation}"
        )

    def stop(self) -> None:
        """Release the receiver and relay client. Idempotent."""
        self._stopped.set()
        client, self._client = self._client, None
        if client is not None:
            client.stop()
        server, self._server = self._server, None
        if server is not None:
            server.stop()

    def ready(self) -> bool:
        """Report whether fresh tracked controllers are available."""
        return self._store.ready()

    @property
    def url(self) -> str:
        """Return the producer endpoint for logging and diagnostics."""
        if self._url is not None:
            return self._url
        return self._server.url if self._server is not None else ""

    def poll(self) -> Observation | None:
        """Return the newest frame, rate-limited and de-duplicated."""
        now = time.perf_counter_ns()
        if now < self._next_ns:
            return None
        frame = self._store.snapshot()
        if frame is None:
            return None
        capture_ns = int(frame["client_capture_ns"])
        if capture_ns <= self._last_capture_ns:
            return None
        self._next_ns = max(now, self._next_ns + self._interval_ns)
        self._last_capture_ns = capture_ns
        return Observation(frame=frame, capture_ns=capture_ns, source=self.name)
