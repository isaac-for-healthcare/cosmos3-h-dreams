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

"""CPU contract check for LAN Headset samples reaching the 28D action encoder."""

from __future__ import annotations

import asyncio
import io
import sys
import threading
from importlib.machinery import ModuleSpec, NamespaceLoader, PathFinder
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aiohttp import MultipartReader
from aiohttp.test_utils import TestClient, TestServer
from cosmos3_h_dreams.headset import controller_frames_to_actions
from cosmos3_h_dreams.headset_webrtc import (
    LatestHeadsetFrames,
    HeadsetWebRTCServer,
    create_app,
)

from cosmos3_h_dreams import headset_webrtc as webrtc_module

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "headset_config, flags, input_fps, display_fps, transport, overlap, model_fps",
    [
        ("headset-base", [], 30, 20, "mjpeg", True, 10),
        (
            "headset-base",
            [
                "--headset-input-fps",
                "15",
                "--display-fps",
                "10",
                "--video-transport",
                "webrtc",
                "--no-overlap-playback",
                "--fps",
                "12",
            ],
            15,
            10,
            "webrtc",
            False,
            12,
        ),
        (None, [], 10, 30, "webrtc", True, 10),
    ],
)
def test_headset_preset_and_explicit_overrides(
    tmp_path,
    monkeypatch,
    headset_config,
    flags,
    input_fps,
    display_fps,
    transport,
    overlap,
    model_fps,
):
    frame = tmp_path / "frame.png"
    frame.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "headset_webrtc",
            "--config",
            "wan-taehv-chunk2-2steps-fademem-compile",
            "--cert",
            "cert.pem",
            "--key",
            "key.pem",
            "--checkpoint",
            "/checkpoint",
            "--initial-frame",
            str(frame),
            *(["--headset-config", headset_config] if headset_config is not None else []),
            *flags,
        ],
    )
    monkeypatch.setattr(
        webrtc_module.ssl,
        "SSLContext",
        lambda _protocol: SimpleNamespace(load_cert_chain=lambda *_args: None),
    )
    monkeypatch.setattr(webrtc_module, "create_app", lambda manager: manager)
    managers = []
    monkeypatch.setattr(
        webrtc_module.web,
        "run_app",
        lambda manager, **_kwargs: managers.append(manager),
    )

    webrtc_module.main()

    (manager,) = managers
    config = manager.session.config
    assert config.checkpoint == "/checkpoint"
    assert config.initial_frame == frame
    assert (config.domain_id, config.height, config.width) == (25, 480, 832)
    assert config.fps == model_fps
    assert (config.chunk_size, config.denoise_steps, config.decoder) == (2, 2, "taehv")
    assert config.cache_mode == "fademem"
    assert config.fademem_merge_selection == "cosine"
    assert config.compile_denoise and config.cuda_graph_denoise
    assert config.overlap_cache_update
    assert not config.profile
    assert manager.frames._period_ns == (
        0 if input_fps is None else round(1e9 / input_fps)
    )
    assert manager.display_fps == display_fps
    assert manager.video_transport == transport
    assert manager.overlap_playback is overlap
    assert HeadsetWebRTCServer(
        session=manager.session, frames=manager.frames
    ).overlap_playback


def _controller(y: float, trigger: float) -> dict[str, object]:
    return {
        "position": [0.0, y, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "buttons": [{"v": trigger}, {"v": 0.0}],
        "axes": [],
    }


@pytest.mark.parametrize(
    "sample_fps, interval_ns", [(10, 100_000_000), (30, 33_333_333), (None, 5_000_000)]
)
def test_browser_absolute_poses_drive_translation_and_jaw_without_rotation(
    monkeypatch: pytest.MonkeyPatch,
    sample_fps,
    interval_ns,
) -> None:
    clock_ns = 1_000_000_000
    monkeypatch.setattr(
        "cosmos3_h_dreams.headset_webrtc.time.perf_counter_ns", lambda: clock_ns
    )
    source = LatestHeadsetFrames(fps=sample_fps)

    def push(left_y: float, right_y: float, right_trigger: float) -> None:
        assert source.push(
            {
                "type": "xr_frame",
                "t_client": clock_ns / 1e6,
                "controllers": {
                    "left": _controller(left_y, 0.0),
                    "right": _controller(right_y, right_trigger),
                },
            }
        )

    push(0.0, 0.0, 0.0)
    anchor = source.poll()
    assert anchor is not None and anchor.frame is not None
    assert source.poll() is None

    clock_ns += interval_ns
    push(-0.10, -0.20, 0.75)
    next_sample = source.poll()
    assert next_sample is not None and next_sample.frame is not None
    action = controller_frames_to_actions((anchor.frame, next_sample.frame)).numpy()[0]

    assert action[2] == pytest.approx(-1.0)
    assert action[16] == pytest.approx(-2.0)
    assert action[23] == pytest.approx(1.0)
    assert tuple(action[3:9]) == pytest.approx((1, 0, 0, 0, 1, 0))
    assert tuple(action[17:23]) == pytest.approx((1, 0, 0, 0, 1, 0))


def test_display_fps_is_independent_of_action_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePeer:
        localDescription = SimpleNamespace(sdp="answer", type="answer")

        def __init__(self, configuration: webrtc_module.RTCConfiguration) -> None:
            assert configuration.iceServers == []

        def addTrack(self, _track: object) -> None:
            pass

        def on(self, _event: str):
            return lambda callback: callback

        async def setRemoteDescription(self, _offer: object) -> None:
            pass

        async def createAnswer(self) -> object:
            return self.localDescription

        async def setLocalDescription(self, _answer: object) -> None:
            pass

        async def close(self) -> None:
            pass

    monkeypatch.setattr(webrtc_module, "RTCPeerConnection", FakePeer)
    frames = LatestHeadsetFrames(fps=10)
    session = SimpleNamespace(config=SimpleNamespace(fps=10), actions_per_step=16)
    manager = HeadsetWebRTCServer(session=session, frames=frames, display_fps=30)

    async def check() -> None:
        try:
            await manager.answer("offer", "offer")
            headset_track = manager.track
            await manager.answer_viewer("offer", "offer")
            assert frames._period_ns == 100_000_000
            assert manager.track is headset_track
            assert manager.viewer_track is not headset_track
            assert manager.viewer_track is not None
            assert manager.track.fps == 30
        finally:
            await manager.close_peer()
            await manager.close_viewer_peer()

    asyncio.run(check())


def test_mjpeg_streams_generated_frames_to_multiple_viewers_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    requirements = iter([SimpleNamespace(input_frame_count=1), None])
    result = SimpleNamespace(metadata={"action_rows": 1})
    session = SimpleNamespace(
        config=SimpleNamespace(fps=10, profile=False),
        init=lambda: None,
        close=lambda: None,
        next_step_requirements=lambda: next(requirements),
        step=lambda _window: result,
    )
    frame = np.full((4, 6, 3), (220, 30, 50), dtype=np.uint8)
    monkeypatch.setattr(
        webrtc_module.BufferedVideoTrack,
        "prepare_result_frames",
        lambda _track, _result: (frame,),
    )

    async def check() -> None:
        manager = HeadsetWebRTCServer(
            session=session,
            frames=LatestHeadsetFrames(fps=10),
            video_transport="mjpeg",
            display_fps=30,
        )
        async with TestClient(TestServer(create_app(manager))) as client:
            for path in ("/headset", "/viewer"):
                page = await client.get(path)
                assert 'data-transport="mjpeg"' in await page.text()
            for path in ("/headset/offer", "/viewer/offer"):
                response = await client.post(
                    path, json={"sdp": "offer", "type": "offer"}
                )
                assert response.status == 409
            streams = [await client.get("/video") for _ in range(2)]
            readers = [MultipartReader.from_response(stream) for stream in streams]
            async with client.ws_connect("/ws"):
                assert manager._ready.is_set()
                for reader in readers:
                    part = await asyncio.wait_for(reader.next(), timeout=3)
                    assert part.headers["Content-Type"] == "image/jpeg"
                    jpeg = bytes(await part.read())
                    assert int(part.headers["Content-Length"]) == len(jpeg)
                    image = Image.open(io.BytesIO(jpeg))
                    assert image.size == (6, 4)
                    assert image.getpixel((0, 0)) == pytest.approx((220, 30, 50), abs=3)
                assert manager.peer is None
                assert manager.viewer_peer is None
                assert manager.track.qsize() == 0
            for _ in range(20):
                if manager.socket is None:
                    break
                await asyncio.sleep(0.01)
            assert manager.socket is None
            assert manager.track is None
            assert manager._jpeg is None
            assert not manager._ready.is_set()
            await manager.close()
            for stream in streams:
                assert await asyncio.wait_for(stream.content.read(), timeout=3) == b""

    asyncio.run(check())


def test_mjpeg_profile_measures_cycle_and_block_boundary(monkeypatch, capsys) -> None:
    now = 0.0

    async def inline(function, *args):
        return function(*args)

    async def sleep(seconds):
        nonlocal now
        now += seconds

    def step(_window):
        nonlocal now
        now += 0.2
        return SimpleNamespace(
            step_index=0,
            metadata={
                "action_rows": 2,
                "action_capture_ns": 100_000_000,
                "action_first_capture_ns": 66_666_667,
            },
            metrics={
                "input_s": 0.1,
                "model_wall_s": 0.08,
                "decode_s": 0.02,
                "total_s": 0.1,
            },
        )

    def prepare(_result):
        nonlocal now
        now += 0.03
        return (None, None)

    def encode(_frame):
        nonlocal now
        now += 0.01
        return b"jpeg"

    monkeypatch.setattr(webrtc_module.asyncio, "to_thread", inline)
    monkeypatch.setattr(webrtc_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(webrtc_module.time, "perf_counter", lambda: now)
    monkeypatch.setattr(webrtc_module, "encode_rgb_frame_to_jpeg", encode)
    requirements = iter([SimpleNamespace(input_frame_count=2)] * 2 + [None])
    manager = HeadsetWebRTCServer(
        session=SimpleNamespace(
            config=SimpleNamespace(fps=10, profile=True),
            next_step_requirements=lambda: next(requirements),
            step=step,
        ),
        frames=LatestHeadsetFrames(fps=10),
        display_fps=20,
        video_transport="mjpeg",
        overlap_playback=False,
    )

    async def check():
        monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: now)
        manager.socket = object()
        manager.track = SimpleNamespace(prepare_result_frames=prepare)
        manager._ready.set()
        await manager._generate()

    asyncio.run(check())
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    for line in lines:
        assert "input=100.0ms model=80.0ms decode=20.0ms" in line
        assert "prepare/dispatch=30.0ms generation=230.0ms queue_wait=0.0ms" in line
        assert "jpeg=20.0ms publish/pacing=80.0ms" in line
        assert "cycle=330.0ms" in line
        assert "input_span=33.3ms" in line
    assert "boundary_interval=0.0ms" in lines[0]
    assert "boundary_interval=280.0ms" in lines[1]
    assert "pose_age_first_frame=140.0ms" in lines[0]
    assert "pose_age_first_frame=470.0ms" in lines[1]


@pytest.mark.parametrize("overlap", [False, True])
def test_mjpeg_overlap_is_bounded_and_preserves_frame_order(monkeypatch, overlap):
    async def check():
        encoding = asyncio.Event()
        release = asyncio.Event()
        second_generated = asyncio.Event()
        steps = []
        encoded = []

        def step(_window):
            index = len(steps)
            steps.append(index)
            if index == 1:
                second_generated.set()
            return SimpleNamespace(step_index=index, metadata={"action_rows": 2})

        def encode(frame):
            encoded.append(frame)
            return str(frame).encode()

        async def to_thread(function, *args):
            if function is encode and args[0] == 0:
                encoding.set()
                await release.wait()
            return function(*args)

        monkeypatch.setattr(webrtc_module.asyncio, "to_thread", to_thread)
        monkeypatch.setattr(webrtc_module, "encode_rgb_frame_to_jpeg", encode)
        requirements = iter([SimpleNamespace(input_frame_count=2)] * 3 + [None])
        manager = HeadsetWebRTCServer(
            session=SimpleNamespace(
                config=SimpleNamespace(fps=10, profile=False),
                next_step_requirements=lambda: next(requirements),
                step=step,
            ),
            frames=LatestHeadsetFrames(fps=10),
            display_fps=1000,
            video_transport="mjpeg",
            overlap_playback=overlap,
        )
        manager.socket = object()
        manager.track = webrtc_module.BufferedVideoTrack(fps=1000, maxsize=1)
        manager.track.prepare_result_frames = lambda result: (
            2 * result.step_index,
            2 * result.step_index + 1,
        )
        manager._ready.set()
        producer = asyncio.create_task(manager._generate())
        try:
            await asyncio.wait_for(encoding.wait(), timeout=1)
            if overlap:
                await asyncio.wait_for(second_generated.wait(), timeout=1)
            await asyncio.sleep(0)
            assert steps == ([0, 1] if overlap else [0])
            release.set()
            await asyncio.wait_for(producer, timeout=1)
            assert steps == [0, 1, 2]
            assert encoded == list(range(6))
            assert manager._jpeg == b"5"
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            await manager.close_peer()

    asyncio.run(check())


@pytest.mark.parametrize(
    "transition", ["disconnect", "reconnect", "calibration", "shutdown"]
)
def test_mjpeg_overlap_discards_stale_playback(monkeypatch, transition):
    async def check():
        encoding = asyncio.Event()
        release_encode = asyncio.Event()
        next_started = asyncio.Event()
        release_step = asyncio.Event()
        playback_cancelled = asyncio.Event()
        steps = []

        def step(_window):
            index = len(steps)
            steps.append(index)
            return SimpleNamespace(
                step_index=index,
                metadata={
                    "action_rows": 0
                    if transition == "calibration" and index == 1
                    else 2
                },
            )

        def encode(frame):
            return str(frame).encode()

        async def to_thread(function, *args):
            if function is step and steps:
                next_started.set()
                await release_step.wait()
            if function is encode and args[0] == 0:
                encoding.set()
                try:
                    await release_encode.wait()
                except asyncio.CancelledError:
                    playback_cancelled.set()
                    raise
            return function(*args)

        monkeypatch.setattr(webrtc_module.asyncio, "to_thread", to_thread)
        monkeypatch.setattr(webrtc_module, "encode_rgb_frame_to_jpeg", encode)
        requirements = iter([SimpleNamespace(input_frame_count=2)] * 2 + [None])
        manager = HeadsetWebRTCServer(
            session=SimpleNamespace(
                config=SimpleNamespace(fps=10, profile=False),
                next_step_requirements=lambda: next(requirements),
                step=step,
            ),
            frames=LatestHeadsetFrames(fps=10),
            display_fps=1000,
            video_transport="mjpeg",
            overlap_playback=True,
        )
        manager.socket = object()
        track = webrtc_module.BufferedVideoTrack(fps=1000, maxsize=1)
        track.prepare_result_frames = lambda result: (
            2 * result.step_index,
            2 * result.step_index + 1,
        )
        manager.track = track
        manager._ready.set()
        producer = asyncio.create_task(manager._generate())
        try:
            await asyncio.wait_for(encoding.wait(), timeout=1)
            await asyncio.wait_for(next_started.wait(), timeout=1)
            if transition == "shutdown":
                producer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await producer
                assert playback_cancelled.is_set()
            else:
                if transition in {"disconnect", "reconnect"}:
                    await manager.close_peer()
                    if transition == "reconnect":
                        manager.track = webrtc_module.BufferedVideoTrack(
                            fps=1000, maxsize=1
                        )
                        manager._ready.set()
                    release_encode.set()
                release_step.set()
                if transition == "disconnect":
                    # Wake the producer to finish the finite stub session.
                    manager._closed = True
                    manager._ready.set()
                await asyncio.wait_for(producer, timeout=1)
                if transition == "calibration":
                    assert playback_cancelled.is_set()
                    assert manager._jpeg == b"3"
                else:
                    assert manager._jpeg is None
            assert manager._last_publish_s is None or transition == "calibration"
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            await manager.close_peer()

    asyncio.run(check())


def test_generated_frames_reach_headset_and_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def inline(function, *args):
        return function(*args)

    monkeypatch.setattr(webrtc_module.asyncio, "to_thread", inline)
    requirements = iter([SimpleNamespace(input_frame_count=1), None])
    result = SimpleNamespace(metadata={"action_rows": 1})
    session = SimpleNamespace(
        config=SimpleNamespace(fps=10),
        actions_per_step=2,
        next_step_requirements=lambda: next(requirements),
        step=lambda _window: result,
    )
    manager = HeadsetWebRTCServer(session=session, frames=LatestHeadsetFrames(fps=10))

    async def check() -> None:
        manager.socket = object()
        manager.track = webrtc_module.BufferedVideoTrack(fps=30, maxsize=4)
        manager.viewer_track = webrtc_module.BufferedVideoTrack(fps=30, maxsize=4)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        manager.track.prepare_result_frames = lambda _result: (frame,)
        manager._ready.set()
        try:
            await manager._generate()
            assert manager.track.qsize() == 1
            assert manager.viewer_track.qsize() == 1
        finally:
            await manager.close_peer()
            await manager.close_viewer_peer()

    asyncio.run(check())


@pytest.mark.parametrize("namespace_package", [False, True])
def test_headset_page_and_offer_boundary(monkeypatch, namespace_package) -> None:
    if namespace_package:
        package = sys.modules["cosmos3_h_dreams"]
        outer = str(Path(webrtc_module.__file__).resolve().parent.parent)
        monkeypatch.setattr(
            package,
            "__spec__",
            ModuleSpec(
                "cosmos3_h_dreams",
                NamespaceLoader("cosmos3_h_dreams", [outer], PathFinder.find_spec),
                is_package=True,
            ),
        )
        monkeypatch.setattr(package, "__path__", [outer])
        assert not files("cosmos3_h_dreams").joinpath("web/headset.html").is_file()

    class FakeManager:
        video_transport = "webrtc"
        runtime_ready = False
        socket = None
        source = None
        frames = LatestHeadsetFrames(fps=10.0)
        session = SimpleNamespace(model_status="Loading model…")

        async def start(self) -> None:
            pass

        async def close(self) -> None:
            pass

        async def close_peer(self) -> None:
            pass

    async def check() -> None:
        manager = FakeManager()
        async with TestClient(TestServer(create_app(manager))) as client:
            page = await client.get("/headset")
            assert page.status == 200
            assert "Enter VR" in await page.text()
            assert 'id="model-status"' in await page.text()
            viewer = await client.get("/viewer")
            assert viewer.status == 200
            assert "Live view of the headset session" in await viewer.text()
            viewer_script = await client.get("/viewer.js")
            assert viewer_script.status == 200
            assert "/viewer/offer" in await viewer_script.text()
            assert "/ws" not in await viewer_script.text()
            script = await client.get("/headset.js")
            assert script.status == 200
            assert "xr_frame" in await script.text()
            scene_script = await client.get("/scenes.js")
            assert scene_script.status == 200
            offer = await client.post(
                "/headset/offer", json={"sdp": "bad", "type": "offer"}
            )
            assert offer.status == 409
            viewer_offer = await client.post("/viewer/offer", json={"type": "answer"})
            assert viewer_offer.status == 400
            async with client.ws_connect("/ws") as ws:
                await ws.send_json(
                    {
                        "type": "xr_frame",
                        "controllers": {
                            "left": _controller(0.0, 0.0),
                            "right": _controller(0.0, 0.0),
                        },
                    }
                )
                sample = None
                for _ in range(20):
                    sample = manager.frames.poll()
                    if sample is not None:
                        break
                    await asyncio.sleep(0.01)
                assert sample is not None and sample.frame is not None

    asyncio.run(check())


@pytest.mark.parametrize("fail", [False, True])
def test_model_startup_status_is_served_while_loading(fail):
    entered = threading.Event()
    release = threading.Event()

    class Session:
        model_status = "Loading model…"

        def init(self):
            self.model_status = "Compiling model…"
            entered.set()
            assert release.wait(timeout=5)
            if fail:
                raise RuntimeError("compile failed")
            self.model_status = "Model compiled"

        def close(self):
            pass

    async def check():
        manager = HeadsetWebRTCServer(session=Session(), frames=LatestHeadsetFrames(fps=10))
        try:
            async with TestClient(TestServer(create_app(manager))) as client:
                assert await asyncio.to_thread(entered.wait, 2)
                response = await client.get("/healthz")
                assert response.headers["Cache-Control"] == "no-store"
                assert await response.json() == {
                    "ready": False,
                    "headset_connected": False,
                    "model_status": "Compiling model…",
                    "scenes": [],
                    "active_scene": None,
                    "scene_switching": False,
                }
                assert (await client.get("/headset")).status == 200
                assert (await client.get("/viewer")).status == 200
                release.set()
                expected_status = (
                    "Model startup failed — check server logs"
                    if fail
                    else "Model compiled"
                )
                for _ in range(100):
                    health = await (await client.get("/healthz")).json()
                    if (
                        health["model_status"] == expected_status
                        and health["ready"] is not fail
                    ):
                        break
                    await asyncio.sleep(0.01)
                assert health["ready"] is not fail
                assert health["model_status"] == expected_status
        finally:
            release.set()

    asyncio.run(check())


def test_scene_gallery_serves_only_configured_frames(tmp_path):
    from PIL import Image

    frame = tmp_path / "frame.png"
    Image.new("RGB", (2, 2)).save(frame)

    class Manager:
        video_transport = "webrtc"

        def __init__(self):
            self.scenes = {"frame &.png": SimpleNamespace(initial_frame=frame)}

        async def start(self):
            pass

        async def close(self):
            pass

    async def check():
        async with TestClient(TestServer(create_app(Manager()))) as client:
            for page_name in ("headset", "viewer"):
                page = await (await client.get(f"/{page_name}")).text()
                assert "Scene gallery" in page
                assert '<dialog id="scene-gallery"' in page
                assert '<select id="scene"' not in page
            preview = await client.get("/scene/frame", params={"name": "frame &.png"})
            assert preview.status == 200
            assert preview.content_type == "image/png"
            assert await preview.read() == frame.read_bytes()
            for name in ("", "missing", "../../pyproject.toml", str(frame)):
                response = await client.get("/scene/frame", params={"name": name})
                assert response.status == 404

    asyncio.run(check())


def test_generation_failure_is_logged_and_reported(caplog):
    def fail(_window):
        raise RuntimeError("denoising failed")

    async def check():
        session = SimpleNamespace(
            config=SimpleNamespace(fps=10),
            model_status="Model ready",
            next_step_requirements=lambda: SimpleNamespace(input_frame_count=1),
            step=fail,
        )
        manager = HeadsetWebRTCServer(session=session, frames=LatestHeadsetFrames(fps=10))
        manager.track = object()
        manager.socket = object()
        manager.runtime_ready = True
        manager._ready.set()
        await manager._generate()
        assert not manager.runtime_ready
        assert session.model_status == "Generation failed — check server logs"
        assert "denoising failed" in caplog.text

    asyncio.run(check())
