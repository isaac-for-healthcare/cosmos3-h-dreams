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

"""Headset WebXR input and HTTPS/MJPEG video for the Cosmos3-H-Dreams session."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import functools
import json
import logging
import math
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web
from flashdreams.demo import CanonicalInputWindow
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import StepResult
from flashdreams.serving.realtime.media import (
    encode_rgb_frame_to_jpeg,
    rgb_array_to_uint8_frames,
)

from cosmos3_h_dreams._runtime.hand_client.source import Observation
from cosmos3_h_dreams._runtime.headset_controller import parse_xr_frame

from .app import Cosmos3HDreamsApplication
from .config import COSMOS3_H_DREAMS_PRESETS, Cosmos3HDreamsConfig
from .headset import HeadsetActionSource
from .session import Cosmos3HDreamsSession
from .session_config import load_session_config

_WEB_DIR = Path(__file__).resolve().parent / "web"

# Empirical left grip-to-pivot offset from a Meta Quest controller X-rotation trace, in metres.
_LEFT_GRIP_PIVOT_OFFSET_M = (0.0, -0.02415, -0.04658)

HEADSET_PRESETS = {
    "headset-base": {
        "headset_input_fps": 30.0,
        "display_fps": 20,
    },
}


class LatestHeadsetFrames:
    """Sample the newest validated pose, optionally limiting acquisition rate."""

    def __init__(self, *, fps: float | None) -> None:
        self._period_ns = 0 if fps is None else round(1e9 / fps)
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._version = 0
        self._sampled_version = 0
        self._next_sample_ns = 0
        self._closed = False

    def start(self) -> None:
        """Keep the sampler ready for controller frames from the HTTPS server."""

    def stop(self) -> None:
        """Stop accepting controller frames."""
        with self._lock:
            self._closed = True
            self._latest = None

    def clear(self) -> None:
        """Discard poses across browser disconnects and reconnects."""
        with self._lock:
            self._latest = None
            self._sampled_version = self._version
            self._next_sample_ns = 0

    def push(self, payload: object) -> bool:
        """Validate an ``xr_frame`` and retain its absolute controller poses."""
        frame = parse_xr_frame(payload)
        if frame is None:
            return False
        for side in ("left", "right"):
            controller = frame.get(side)
            if controller is not None and any(
                not math.isfinite(float(controller[field]))
                for field in ("trigger", "grip")
            ):
                raise ValueError(f"{side} controller buttons must be finite")
        with self._lock:
            if self._closed:
                return False
            self._latest = frame
            self._version += 1
        return True

    def poll(self) -> Observation | None:
        now_ns = time.perf_counter_ns()
        with self._lock:
            if (
                self._closed
                or self._latest is None
                or self._version == self._sampled_version
                or now_ns < self._next_sample_ns
                or now_ns - int(self._latest["client_capture_ns"]) > 500_000_000
            ):
                return None
            frame = self._latest
            self._sampled_version = self._version
            self._next_sample_ns = now_ns + self._period_ns
        return Observation(frame=frame, capture_ns=now_ns, source="headset-mjpeg")


class MjpegFrames:
    """Convert model frames for Pillow JPEG encoding."""

    def __init__(self, *, gpu_frame_conversion: bool = False) -> None:
        self.gpu_frame_conversion = gpu_frame_conversion

    def prepare_result_frames(self, result: StepResult) -> tuple[Any, ...]:
        video = result.video_chunk
        if self.gpu_frame_conversion and video.is_cuda and video.is_floating_point():
            # Match the host float32 scaling before transferring one byte/channel.
            video = ((video.float() + 1.0) / 2.0 * 255.0).clamp(0, 255).byte()
        return tuple(
            rgb_array_to_uint8_frames(
                video,
                layout=result.layout,
                value_range="minus_one_one" if video.is_floating_point() else "uint8",
                sync_device=True,
            )
        )

class HeadsetServer:
    """Bind one headset browser to the existing model session and video track."""

    def __init__(
        self,
        *,
        session: Cosmos3HDreamsSession,
        frames: LatestHeadsetFrames,
        display_fps: int = 30,
        overlap_playback: bool = True,
        deadline_pacing: bool = False,
        gpu_frame_conversion: bool = False,
        dedicated_inference_thread: bool = False,
        generation_lead_ms: float = 0.0,
        scenes: dict[str, Cosmos3HDreamsConfig] | None = None,
        scene_labels: dict[str, str] | None = None,
    ) -> None:
        if not math.isfinite(generation_lead_ms) or generation_lead_ms < 0:
            raise ValueError("generation_lead_ms must be finite and nonnegative")
        if generation_lead_ms and not (overlap_playback and deadline_pacing):
            raise ValueError("generation_lead_ms requires overlapping deadline pacing")
        self.generation_lead_ms = generation_lead_ms
        self._next_block_due_s: float | None = None
        self._generation_wakeup = asyncio.Event()
        self.scenes = scenes or {}
        self.scene_labels = scene_labels or {}
        self.active_scene = next(iter(self.scenes), None)
        self._scene_request: tuple[str, asyncio.Future[None]] | None = None
        self._scene_changed = asyncio.Event()
        self.session = session
        self.frames = frames
        self.display_fps = display_fps
        self.overlap_playback = overlap_playback
        self.deadline_pacing = deadline_pacing
        self.gpu_frame_conversion = gpu_frame_conversion
        self.dedicated_inference_thread = dedicated_inference_thread
        self._inference_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="dreams-inference")
            if dedicated_inference_thread else None
        )
        self._last_publish_s: float | None = None
        self._jpeg: bytes | None = None
        self._jpeg_version = 0
        self._jpeg_changed = asyncio.Condition()
        self.source: HeadsetActionSource | None = None
        self.track: MjpegFrames | None = None
        self.socket: web.WebSocketResponse | None = None
        self._ready = asyncio.Event()
        self._producer: asyncio.Task[None] | None = None
        self._closed = False
        self.runtime_ready = False
        self.token_emitter = None
        self.token_descriptor = None
        self.token_seed_pending = False
        self.token_chunk = 0
        self.restart_requested = False

    async def start(self) -> None:
        """Load the model and begin the asynchronous generation loop."""
        try:
            await self._run_inference(self.session.init)
        except Exception:
            self.session.model_status = "Model startup failed — check server logs"
            logging.getLogger(__name__).exception(
                "Cosmos3-H-Dreams model startup failed"
            )
            return
        self.frames.clear()
        self.runtime_ready = True
        self._producer = asyncio.create_task(self._generate())

    async def _run_inference(self, function, *args, **kwargs):
        """Keep GPU initialization and execution on the same optional worker."""
        if self._inference_executor is None:
            return await asyncio.to_thread(function, *args, **kwargs)
        context = contextvars.copy_context()
        return await asyncio.get_running_loop().run_in_executor(
            self._inference_executor, functools.partial(context.run, function, *args, **kwargs)
        )

    async def close(self) -> None:
        """Stop controller sampling, video delivery, and the model session."""
        if self._closed:
            return
        self._closed = True
        self.runtime_ready = False
        self._ready.set()
        async with self._jpeg_changed:
            self._jpeg_changed.notify_all()
        if self.source is not None:
            self.source.close()
        if self._producer is not None:
            self._producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._producer
        await self.reset_video()
        if self.socket is not None:
            await self.socket.close()
        try:
            # Queue cleanup behind an in-flight step even if its asyncio waiter
            # was cancelled; cancelling that waiter cannot stop its GPU thread.
            await self._run_inference(self.session.close)
        finally:
            if self._inference_executor is not None:
                self._inference_executor.shutdown(wait=True)

    async def reset_video(self) -> None:
        """Clear the current MJPEG image and invalidate in-flight playback."""
        self._ready.clear()
        self._last_publish_s = None
        self._next_block_due_s = None
        self._generation_wakeup.set()
        async with self._jpeg_changed:
            self._jpeg = None
            self._jpeg_version += 1
            self._jpeg_changed.notify_all()
        self.track = None

    async def select_scene(self, name: str) -> None:
        """Request a scene change at the next generation boundary."""
        if not isinstance(name, str) or name not in self.scenes:
            raise web.HTTPBadRequest(reason="Unknown scene")
        if not self.runtime_ready or self._closed:
            raise web.HTTPConflict(reason="Model is not ready")
        if self._scene_request is not None:
            raise web.HTTPConflict(reason="A scene change is already pending")
        if self._producer is None or self._producer.done():
            raise web.HTTPConflict(reason="Generation is not running")
        if name == self.active_scene:
            return
        done = asyncio.get_running_loop().create_future()
        self._scene_request = (name, done)
        self._scene_changed.set()
        self._next_block_due_s = None
        self._generation_wakeup.set()
        if self.source is not None:
            self.source.request_restart()
        await asyncio.shield(done)

    async def _wait_for_generation_slot(self, track: MjpegFrames) -> None:
        """Capture controls closer to the next MJPEG block's display deadline."""
        if not self.generation_lead_ms or self.token_emitter is not None:
            return
        # Give the newly scheduled publisher a turn to estimate its next block.
        await asyncio.sleep(0)
        self._generation_wakeup.clear()
        if (self._next_block_due_s is None or self.track is not track
                or not self._ready.is_set() or self._scene_request is not None):
            return
        delay = self._next_block_due_s - self.generation_lead_ms / 1000 - time.perf_counter()
        if delay > 0:
            try:
                await asyncio.wait_for(self._generation_wakeup.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def _generate(self) -> None:
        playback: asyncio.Task[None] | None = None
        try:
            while not self._closed:
                if self._scene_request is not None:
                    name, done = self._scene_request
                    if playback is not None:
                        playback.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await playback
                        playback = None
                    try:
                        await self._run_inference(
                            self.session.switch_scene, self.scenes[name]
                        )
                    except Exception:
                        logging.getLogger(__name__).exception("Scene switch failed")
                        done.set_exception(
                            web.HTTPInternalServerError(
                                reason="Scene switch failed; check server logs"
                            )
                        )
                    else:
                        self.frames.clear()
                        self.active_scene = name
                        async with self._jpeg_changed:
                            self._jpeg = None
                            self._jpeg_version += 1
                            self._jpeg_changed.notify_all()
                        self._last_publish_s = None
                        self._next_block_due_s = None
                        done.set_result(None)
                    finally:
                        if not done.done():
                            done.set_exception(
                                web.HTTPServiceUnavailable(reason="Generation stopped")
                            )
                        self._scene_request = None
                        self._scene_changed.clear()
                if not self._ready.is_set():
                    waits = [
                        asyncio.create_task(self._ready.wait()),
                        asyncio.create_task(self._scene_changed.wait()),
                    ]
                    try:
                        await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for waiter in waits:
                            waiter.cancel()
                        await asyncio.gather(*waits, return_exceptions=True)
                    if self._scene_request is not None:
                        continue
                if self._closed:
                    return
                track = self.track
                if self.socket is None or track is None:
                    self._ready.clear()
                    continue
                if self.restart_requested:
                    self._next_block_due_s = None
                    await self._run_inference(self.session.reset)
                    self.restart_requested = False
                requirements = self.session.next_step_requirements()
                if requirements is None:
                    if playback is not None:
                        await playback
                        playback = None
                    if self.token_descriptor is None:
                        return
                    await self._run_inference(self.session.reset)
                    if self.token_emitter is not None:
                        await self.socket.close(code=1000, message=b"Replay complete; reconnect to restart")
                    else:
                        self._ready.clear()
                    continue
                emitter = self.token_emitter
                if emitter is not None and self.token_seed_pending:
                    model, _ = self.session._require_runtime()
                    seed = model._frame_zero_latent(self.session._require_scene())
                    seed = model._crop_decode_latent(seed).permute(1, 0, 2, 3).contiguous()
                    if not await self._emit_token(emitter, seed, is_keyframe=True):
                        continue
                    self.token_chunk += 1
                    self.token_seed_pending = False
                window = CanonicalInputWindow(
                    window=TimeWindow(
                        start_s=0.0,
                        end_s=requirements.input_frame_count / self.session.config.fps,
                    )
                )
                await self._wait_for_generation_slot(track)
                if (self.track is not track or not self._ready.is_set()
                        or self._scene_request is not None or self.restart_requested):
                    continue
                cycle_started = time.perf_counter()
                try:
                    if emitter is None:
                        result = await self._run_inference(self.session.step, window)
                    else:
                        result = await self._run_inference(self.session.step, window, decode=False)
                except RuntimeError:
                    if self._closed:
                        return
                    raise
                if self._scene_request is not None:
                    continue
                if result.metadata.get("action_rows") == 0 and playback is not None:
                    playback.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await playback
                    playback = None
                    self._last_publish_s = None
                    self._next_block_due_s = None
                if self.track is track and self._ready.is_set():
                    if emitter is not None and result.layout is None:
                        for latent in result.output.split(1):
                            if not await self._emit_token(emitter, latent):
                                break
                            self.token_chunk += 1
                        logging.getLogger(__name__).info(
                            "[stream] mode=raw_tokens step=%s wire=raw_f16 shape=%s server_decode=False",
                            result.step_index, tuple(result.output.shape),
                        )
                        continue
                    frames = await self._run_inference(
                        track.prepare_result_frames, result
                    )
                    if self._scene_request is not None:
                        continue
                    prepared = time.perf_counter()
                    if playback is not None:
                        await playback
                    if self.track is track and self._ready.is_set():
                        # Keep only the playing block and one prepared block.
                        publish = self._publish_mjpeg(
                            track, result, frames, cycle_started, prepared
                        )
                        if self.overlap_playback:
                            playback = asyncio.create_task(publish)
                        else:
                            await publish
                if result.metadata.get("action_rows") == 0:
                    await asyncio.sleep(0.25)
        except Exception:
            self.session.model_status = "Generation failed — check server logs"
            logging.getLogger(__name__).exception("Cosmos3-H-Dreams generation failed")
        finally:
            self.runtime_ready = False
            if self._scene_request is not None:
                _, done = self._scene_request
                done.set_exception(
                    web.HTTPServiceUnavailable(reason="Generation stopped")
                )
                self._scene_request = None
            if playback is not None:
                playback.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await playback

    async def _emit_token(self, emitter, latent, *, is_keyframe=False) -> bool:
        """Bound backpressure and release a disconnected token producer."""
        send = asyncio.create_task(emitter.emit_chunk(
            latent, chunk_index=self.token_chunk, is_keyframe=is_keyframe,
        ))
        deadline = asyncio.get_running_loop().time() + 300
        try:
            while emitter is self.token_emitter and not self._closed:
                done, _ = await asyncio.wait({send}, timeout=1)
                if done:
                    try:
                        await send
                    except ConnectionError:
                        return False
                    return True
                if asyncio.get_running_loop().time() > deadline:
                    await self.socket.close(code=1011, message=b"Token decode acknowledgment timed out")
                    return False
            return False
        finally:
            if not send.done():
                send.cancel()
            await asyncio.gather(send, return_exceptions=True)

    async def _publish_mjpeg(
        self,
        track: MjpegFrames,
        result: StepResult,
        frames: tuple[Any, ...],
        cycle_started: float,
        prepared: float,
    ) -> None:
        """Publish one block in order, pacing frames at the display rate."""
        playback_started = time.perf_counter()
        if self.generation_lead_ms:
            first_due = max(
                playback_started,
                self._last_publish_s + 1 / self.display_fps
                if self._last_publish_s is not None else playback_started,
            )
            # ponytail: the lead is calibrated for the deployment's measured
            # compute time; future adaptive scheduling must retain a margin.
            self._next_block_due_s = first_due + len(frames) / self.display_fps
        jpeg_s = 0.0
        boundary_s = None
        pose_age = "n/a"
        capture_span = "n/a"
        if result.metadata.get("action_first_capture_ns") is not None:
            capture_span = f"{(result.metadata['action_capture_ns'] - result.metadata['action_first_capture_ns']) / 1e6:.1f}ms"
        for index, frame in enumerate(frames):
            if (
                self.track is not track
                or not self._ready.is_set()
                or self._scene_request is not None
            ):
                return
            started = asyncio.get_running_loop().time()
            jpeg = await asyncio.to_thread(encode_rgb_frame_to_jpeg, frame)
            jpeg_s += asyncio.get_running_loop().time() - started
            if self.deadline_pacing and self._last_publish_s is not None:
                # Encode ahead of publication, including across chunk boundaries.
                # Anchor to the last actual publish so late frames never cause a burst.
                await asyncio.sleep(
                    max(0, self._last_publish_s + 1 / self.display_fps - time.perf_counter())
                )
            if (
                self.track is not track
                or not self._ready.is_set()
                or self._scene_request is not None
            ):
                return
            # ponytail: retain one JPEG; slow viewers drop frames. Use
            # per-viewer bounded queues if complete playback is needed.
            async with self._jpeg_changed:
                self._jpeg = jpeg
                self._jpeg_version += 1
                self._jpeg_changed.notify_all()
            published = time.perf_counter()
            if index == 0 and result.metadata.get("action_capture_ns") is not None:
                pose_age = f"{(published - result.metadata['action_capture_ns'] / 1e9) * 1e3:.1f}ms"
            if index == 0 and self._last_publish_s is not None:
                boundary_s = published - self._last_publish_s
            self._last_publish_s = published
            if not self.deadline_pacing:
                await asyncio.sleep(
                    max(
                        0,
                        1 / self.display_fps
                        - (asyncio.get_running_loop().time() - started),
                    )
                )
        if self.session.config.profile and result.metadata.get("action_rows"):
            finished = time.perf_counter()
            metrics = result.metrics
            print(
                f"[headset-mjpeg] step={result.step_index} frames={len(frames)} "
                f"input={metrics['input_s'] * 1e3:.1f}ms "
                f"model={metrics['model_wall_s'] * 1e3:.1f}ms "
                f"decode={metrics['decode_s'] * 1e3:.1f}ms "
                f"prepare/dispatch={(prepared - cycle_started - metrics['input_s'] - metrics['total_s']) * 1e3:.1f}ms "
                f"generation={(prepared - cycle_started) * 1e3:.1f}ms "
                f"queue_wait={(playback_started - prepared) * 1e3:.1f}ms "
                f"jpeg={jpeg_s * 1e3:.1f}ms "
                f"publish/pacing={(finished - playback_started - jpeg_s) * 1e3:.1f}ms "
                f"cycle={(finished - cycle_started) * 1e3:.1f}ms "
                f"boundary_interval={boundary_s * 1e3 if boundary_s is not None else 0:.1f}ms "
                f"overlap={self.overlap_playback} pose_age_first_frame={pose_age} "
                f"deadline_pacing={self.deadline_pacing} "
                f"input_span={capture_span}",
                flush=True,
            )


def create_app(manager: HeadsetServer) -> web.Application:
    """Serve the headset page, controller socket, and MJPEG video."""
    app = web.Application(client_max_size=64 * 1024)

    async def page(_: web.Request) -> web.Response:
        body = (_WEB_DIR / "headset.html").read_text()
        return web.Response(
            text=body,
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def script(_: web.Request) -> web.Response:
        body = (_WEB_DIR / "headset.js").read_text()
        return web.Response(
            text=body,
            content_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    async def scene_script(_: web.Request) -> web.Response:
        body = (_WEB_DIR / "scenes.js").read_text()
        return web.Response(
            text=body,
            content_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    async def viewer_page(_: web.Request) -> web.Response:
        body = (_WEB_DIR / "viewer.html").read_text()
        return web.Response(
            text=body,
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def viewer_script(_: web.Request) -> web.Response:
        body = (_WEB_DIR / "viewer.js").read_text()
        return web.Response(
            text=body,
            content_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    async def socket(request: web.Request) -> web.WebSocketResponse:
        if manager.socket is not None:
            raise web.HTTPConflict(reason="A Headset browser is already connected")
        ws = web.WebSocketResponse(max_msg_size=64 * 1024, heartbeat=5.0)
        mode = request.query.get("mode", "pixels")
        if mode not in {"pixels", "raw_tokens"}:
            raise web.HTTPBadRequest(reason="Unknown streaming mode")
        if mode == "raw_tokens" and manager.token_descriptor is None:
            raise web.HTTPServiceUnavailable(reason="Token decoder is not ready")
        await ws.prepare(request)
        manager.socket = ws
        manager.frames.clear()
        if manager.source is not None:
            manager.source.reset_calibration()
        manager.track = MjpegFrames(gpu_frame_conversion=manager.gpu_frame_conversion)
        manager.restart_requested = manager.token_descriptor is not None
        if mode == "raw_tokens":
            from flashdreams.serving.token_stream import TokenFrameEmitter, RawFloat16TokenCodecConfig

            manager.token_emitter = TokenFrameEmitter(
                ws, codec=RawFloat16TokenCodecConfig().setup(),
                fps=manager.display_fps, flow_window_size=2,
                extra_header={"vae_model": manager.token_descriptor},
            )
            manager.token_seed_pending = True
        logging.getLogger(__name__).info("[stream] connected mode=%s", mode)
        manager._ready.set()
        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(message.data)
                    if not isinstance(payload, dict):
                        raise TypeError("message must be an object")
                    if payload.get("type") == "token_frame_ack" and manager.token_emitter is not None:
                        chunk = payload.get("chunk_id")
                        if type(chunk) is not int or chunk < 0:
                            raise ValueError("Invalid chunk acknowledgment")
                        manager.token_emitter.handle_ack(chunk)
                    elif payload.get("type") == "keys" and hasattr(manager.source, "set_keys"):
                        manager.source.set_keys(payload.get("keys"))
                    elif payload.get("type") == "reset":
                        if manager.token_emitter is not None:
                            await ws.close(code=1012, message=b"Reconnect to reset token decoder state")
                            break
                        if manager.source is not None:
                            manager.source.request_restart()
                        manager._next_block_due_s = None
                        manager._generation_wakeup.set()
                    else:
                        manager.frames.push(payload)
                except (ValueError, TypeError, KeyError):
                    await ws.send_json(
                        {"type": "error", "message": "Invalid controller frame"}
                    )
        finally:
            manager.token_emitter = None
            manager.socket = None
            manager.frames.clear()
            if manager.source is not None:
                manager.source.reset_calibration()
            await manager.reset_video()
        return ws

    async def video(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            }
        )
        await response.prepare(request)
        version = -1
        # Complete the next part's headers eagerly: Chromium otherwise holds
        # the finished image until the next generated frame arrives. MIME
        # boundaries delimit bodies; per-part Content-Length is optional.
        part_header = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        prefix = part_header
        try:
            while not manager._closed:
                async with manager._jpeg_changed:
                    # Periodically check for disconnected idle subscribers.
                    try:
                        await asyncio.wait_for(
                            manager._jpeg_changed.wait_for(
                                lambda version=version: (
                                    manager._closed or manager._jpeg_version != version
                                )
                            ),
                            timeout=1,
                        )
                    except TimeoutError:
                        if request.transport is None or request.transport.is_closing():
                            break
                        continue
                    version, jpeg = manager._jpeg_version, manager._jpeg
                if manager._closed:
                    break
                if jpeg is not None:
                    part = (
                        prefix
                        + jpeg
                        + b"\r\n"
                        + part_header
                    )
                    await asyncio.wait_for(response.write(part), timeout=5)
                    prefix = b""
        except (ConnectionError, TimeoutError):
            pass
        return response

    async def select_scene(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise TypeError
            name = payload["name"]
        except (ValueError, KeyError, TypeError) as error:
            raise web.HTTPBadRequest(reason="Expected a scene name") from error
        if manager.token_emitter is not None:
            raise web.HTTPConflict(reason="Stop the token stream before switching scenes")
        await manager.select_scene(name)
        return web.json_response({"active_scene": manager.active_scene})

    async def scene_frame(request: web.Request) -> web.FileResponse:
        scene = manager.scenes.get(request.query.get("name", ""))
        if scene is None:
            raise web.HTTPNotFound(reason="Unknown scene")
        return web.FileResponse(scene.initial_frame)

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "ready": manager.runtime_ready,
                "headset_connected": manager.socket is not None,
                "model_status": manager.session.model_status,
                **({"controls": "keyboard" if hasattr(manager.source, "set_keys") else "replay" if manager.session.config.actions is not None else "headset"} if manager.token_descriptor is not None else {}),
                "scenes": [
                    {"name": name, "label": manager.scene_labels.get(name, name)}
                    for name in manager.scenes
                ],
                "active_scene": manager.active_scene,
                "scene_switching": manager._scene_request is not None,
            },
            headers={"Cache-Control": "no-store"},
        )

    app.router.add_get("/headset", page)
    app.router.add_get("/headset.js", script)
    app.router.add_get("/scenes.js", scene_script)
    app.router.add_get("/viewer", viewer_page)
    app.router.add_get("/viewer.js", viewer_script)
    app.router.add_get("/ws", socket)
    app.router.add_get("/video", video)
    app.router.add_get("/healthz", health)
    app.router.add_post("/scene", select_scene)
    app.router.add_get("/scene/frame", scene_frame)

    async def lifecycle(_app: web.Application):
        startup = asyncio.create_task(manager.start())
        try:
            yield
        finally:
            await startup
            await manager.close()

    app.cleanup_ctx.append(lifecycle)
    return app


def main(argv: list[str] | None = None) -> None:
    """Start the independent HTTPS headset flow with the shared model options.

    Args:
        argv: Command-line arguments, defaulting to the process arguments.
    """
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--headset-config", choices=HEADSET_PRESETS)
    selector.add_argument("--session-config", type=Path)
    selected, _ = selector.parse_known_args(argv)
    session_data = (
        load_session_config(selected.session_config)
        if selected.session_config
        else None
    )
    headset_defaults = dict((session_data or {}).get("headset", {}))
    headset_preset = selected.headset_config or headset_defaults.pop("preset", None)
    if headset_preset not in {None, *HEADSET_PRESETS}:
        selector.error(f"unknown headset preset: {headset_preset}")
    parser = argparse.ArgumentParser(
        description="Serve Cosmos3-H-Dreams to a headset browser over HTTPS/MJPEG.",
        epilog=(
            "Model options such as --checkpoint, --initial-frame, --domain-id, "
            "and --fps match flashdreams-run cosmos3-h-dreams."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--cert", required="cert" not in headset_defaults)
    parser.add_argument("--key", required="key" not in headset_defaults)
    parser.add_argument("--config", choices=COSMOS3_H_DREAMS_PRESETS)
    parser.add_argument("--headset-config", choices=HEADSET_PRESETS)
    parser.add_argument("--display-fps", type=int, default=30)
    parser.add_argument("--print-actions", action="store_true")
    parser.add_argument(
        "--generation-lead-ms", type=float, default=0.0,
        help="MJPEG generation lead before the next block is due; 0 disables deferral",
    )
    parser.add_argument(
        "--deadline-pacing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Encode ahead of each MJPEG publication deadline to reduce frame jitter",
    )
    parser.add_argument(
        "--gpu-frame-conversion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convert decoded floating-point pixels to uint8 on CUDA before host transfer",
    )
    parser.add_argument(
        "--dedicated-inference-thread",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep model initialization, inference and GPU frame conversion on one worker",
    )
    parser.add_argument(
        "--headset-input-fps",
        type=float,
        help="Acquire measured action windows at this rate, independently of model --fps",
    )
    parser.add_argument(
        "--overlap-playback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate the next MJPEG block while playing the current block",
    )
    headset_defaults.pop("preset", None)
    parser.set_defaults(**HEADSET_PRESETS.get(headset_preset, {}))
    parser.set_defaults(**headset_defaults)
    network, model_args = parser.parse_known_args(argv)
    if not 1 <= network.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if network.display_fps <= 0:
        parser.error("--display-fps must be positive")
    if network.headset_input_fps is not None and (
        not math.isfinite(network.headset_input_fps) or network.headset_input_fps <= 0
    ):
        parser.error("--headset-input-fps must be finite and positive")
    if not math.isfinite(network.generation_lead_ms) or network.generation_lead_ms < 0:
        parser.error("--generation-lead-ms must be finite and nonnegative")
    if network.generation_lead_ms and not (network.deadline_pacing and network.overlap_playback):
        parser.error("--generation-lead-ms requires --deadline-pacing and --overlap-playback")
    application = Cosmos3HDreamsApplication()
    model_defaults = (
        ["--config", network.config]
        if network.config is not None
        else []
        if session_data is not None
        else [
            "--domain-id",
            "25",
            "--fps",
            "10",
            "--height",
            "288",
            "--width",
            "512",
        ]
    )
    application.init(["--headset", *model_defaults, *model_args])
    session = application.create_session()
    config = session.config
    input_fps = (
        network.headset_input_fps if network.headset_input_fps is not None else config.fps
    )
    frames = LatestHeadsetFrames(fps=input_fps)
    manager: HeadsetServer

    def source_factory(**kwargs: Any) -> HeadsetActionSource:
        source = HeadsetActionSource(
            **kwargs,
            max_buffered_frames=config.chunk_size * 4 + 1,
            pivot_offsets={"left": _LEFT_GRIP_PIVOT_OFFSET_M},
            print_actions=network.print_actions,
            source_factory=lambda **_source_kwargs: frames,
        )
        manager.source = source
        return source

    if any(scene.actions is not None for scene in application.scenes.values()):
        parser.error("Headset server scenes must use live controller input, not actions")
    session = Cosmos3HDreamsSession(config=config, headset_source_factory=source_factory)
    manager = HeadsetServer(
        session=session,
        frames=frames,
        display_fps=network.display_fps,
        overlap_playback=network.overlap_playback,
        deadline_pacing=network.deadline_pacing,
        gpu_frame_conversion=network.gpu_frame_conversion,
        dedicated_inference_thread=network.dedicated_inference_thread,
        generation_lead_ms=network.generation_lead_ms,
        scenes=application.scenes,
        scene_labels=application.scene_labels,
    )
    if config.normalizer is None:
        print(
            "[headset-mjpeg] no checkpoint normalizer supplied; "
            "live action magnitudes are unverified"
        )
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(network.cert, network.key)
    print(
        f"[headset-mjpeg] action/model rate: {config.fps:g} Hz; "
        f"acquisition: {'browser rate' if input_fps is None else str(input_fps) + ' Hz'}; "
        f"actions: recorded window; "
        f"video presentation: {network.display_fps} fps; "
        f"overlap playback: {network.overlap_playback}; "
        f"deadline pacing: {network.deadline_pacing}; "
        f"GPU frame conversion: {network.gpu_frame_conversion}; "
        f"MJPEG generation lead: {network.generation_lead_ms:g} ms (0 disables)"
    )
    print(f"Open https://<workstation-LAN-IP>:{network.port}/headset in the headset browser")
    print(f"Open https://<workstation-LAN-IP>:{network.port}/viewer on a PC")
    web.run_app(
        create_app(manager), host=network.host, port=network.port, ssl_context=tls
    )


if __name__ == "__main__":
    main()
