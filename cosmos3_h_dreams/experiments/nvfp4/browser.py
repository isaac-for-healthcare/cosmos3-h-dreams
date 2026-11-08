# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matched eight-frame keyboard/MJPEG trials in a local headless browser."""
import argparse
import asyncio
import contextvars
import dataclasses
import hashlib
import json
import os
import platform
import threading
import time
from pathlib import Path

import torch
import attention_variants as variants
from aiohttp import web, WSMsgType
from playwright.async_api import async_playwright
import cosmos3_h_dreams.headset_server as hs
from keyboard_adapter import KeyboardActionSource
from cosmos3_h_dreams.config import Cosmos3HDreamsConfig, COSMOS3_H_DREAMS_PRESETS
from cosmos3_h_dreams.session import Cosmos3HDreamsSession
from contract import config, metadata, assert_runtime
from flashdreams.runtime import StepRequirements
from flashdreams.infra.results import StepResult
from common import stamp, read_stamp, proxy_connection


async def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    trace = {k: [] for k in ("receives", "captures", "steps", "outputs", "encodes", "publications", "marker_ms")}
    control = contextvars.ContextVar("control", default={})
    original_receive = web.WebSocketResponse.receive
    async def receive(ws, *args, **kwargs):
        msg = await original_receive(ws, *args, **kwargs)
        if msg.type == WSMsgType.TEXT:
            payload = json.loads(msg.data)
            if payload.get("type") == "keys":
                control.set(payload)
        return msg
    web.WebSocketResponse.receive = receive

    class Source(KeyboardActionSource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.fps *= a.display_fps / 30.0  # Keep held-key velocity fixed per wall-clock second.
            self._lock = threading.RLock()
            self.event_id = self.last_event_id = 0
            self.last_capture_s = 0.0
        def set_keys(self, keys):
            with self._lock:
                super().set_keys(keys)
                self.event_id = int(control.get().get("seq", 0))
                trace["receives"].append({"seq": self.event_id, "keys": keys, "time_s": time.perf_counter()})
        def next_actions(self, rows):
            with self._lock:
                actions = super().next_actions(rows)
                self.last_event_id = self.event_id
                self.last_capture_s = time.perf_counter()
                trace["captures"].append({"seq": self.last_event_id, "time_s": self.last_capture_s, "rows": rows})
                return actions
        def reset_calibration(self):
            with self._lock:
                super().reset_calibration()
                self.event_id = self.last_event_id = 0

    class Session(Cosmos3HDreamsSession):
        def step(self, *args, **kwargs):
            started = time.perf_counter()
            result = super().step(*args, **kwargs)
            seq = self._headset_source.last_event_id
            result = dataclasses.replace(result, metadata={**result.metadata, "bench_seq": seq, "bench_capture_s": self._headset_source.last_capture_s, "bench_step_start_s": started})
            trace["steps"].append({"step": result.step_index, "seq": seq, "start_s": started,
                                   "end_s": time.perf_counter(), "thread": threading.get_ident()})
            return result

    class Track(hs.MjpegFrames):
        def prepare_result_frames(self, result):
            frames = super().prepare_result_frames(result)
            started = time.perf_counter()
            marked = []
            for i, frame in enumerate(frames):
                fid = len(trace["outputs"]) + 1
                trace["outputs"].append({"id": fid, "seq": result.metadata["bench_seq"],
                    "step": result.step_index, "frame_index": i, "prepared_s": started,
                    "capture_s": result.metadata["bench_capture_s"], "step_start_s": result.metadata["bench_step_start_s"]})
                marked.append(stamp(frame, fid))
            trace["marker_ms"].append((time.perf_counter() - started) * 1000)
            return tuple(marked)
    hs.MjpegFrames = Track
    original_encode = hs.encode_rgb_frame_to_jpeg
    jpeg_ids = {}
    def encode(frame):
        started = time.perf_counter()
        data = original_encode(frame)
        finished = time.perf_counter()
        fid = read_stamp(frame)
        jpeg_ids[hashlib.sha256(data).digest()] = fid
        trace["encodes"].append({"id": fid, "start_s": started, "end_s": finished, "bytes": len(data)})
        return data
    hs.encode_rgb_frame_to_jpeg = encode

    cfg = dataclasses.replace(config(a.mode),display_fps=a.display_fps)
    def factory(**kwargs):
        source = Source(**kwargs)
        manager.source = source
        return source
    if a.fake:
        class FakeSession:
            config=cfg
            model_status='Fake harness validation'
            def init(self):
                self._headset_source=factory(fps=10)
                self._headset_source.start();self.index=0
            def next_step_requirements(self):
                return StepRequirements(step_index=self.index,input_frame_count=8,steady_output_frame_count=8)
            def step(self, *args, **kwargs):
                self._headset_source.next_actions(8);time.sleep(.025)
                seq=self._headset_source.last_event_id
                result=StepResult.from_video_chunk(step_index=self.index,
                    video_chunk=torch.zeros(8,3,480,832,dtype=torch.uint8),layout='tchw',
                    metadata={'action_rows':8,'bench_seq':seq,'bench_capture_s':self._headset_source.last_capture_s,'bench_step_start_s':self._headset_source.last_capture_s})
                self.index+=1
                return result
            def submit_pending_cache_update(self):pass
            def close(self):self._headset_source.close()
        session=FakeSession()
    else:
        session=Session(config=cfg,headset_source_factory=factory,model_factory=variants.model_factory(a.attention_variant))
    options=dict(deadline_pacing=True,gpu_frame_conversion=True,
        dedicated_inference_thread=True,generation_lead_ms=a.lead_ms) if a.mode=='latest' else {}
    manager=hs.HeadsetServer(session=session,frames=hs.LatestHeadsetFrames(fps=a.display_fps),display_fps=a.display_fps,**options)
    app = hs.create_app(manager)
    async def page(_):
        return web.FileResponse(Path(__file__).with_name("keyboard.html"))
    app.router.add_get("/benchmark", page)
    async def controls(request):
        ws=web.WebSocketResponse(max_msg_size=64*1024,heartbeat=5.0)
        await ws.prepare(request)
        manager.socket=ws;manager.frames.clear();manager.source.reset_calibration()
        manager.track=Track(**({'gpu_frame_conversion':True} if a.mode=='latest' else {}))
        manager._ready.set()
        try:
            async for message in ws:
                if message.type==WSMsgType.TEXT:
                    payload=json.loads(message.data)
                    assert payload['type']=='keys'
                    manager.source.set_keys(payload['keys'])
        finally:
            manager.socket=None;manager.frames.clear();manager.source.reset_calibration()
            await manager.reset_video()
        return ws
    app.router.add_get('/benchmark-control',controls)

    startup_started = time.perf_counter()
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    started = time.perf_counter()
    while not manager.runtime_ready:
        if "failed" in session.model_status.lower():
            raise RuntimeError(session.model_status)
        if time.perf_counter() - started > 180:
            raise TimeoutError("model initialization")
        await asyncio.sleep(.1)
    startup = time.perf_counter() - startup_started
    if not a.fake:assert_runtime(session,a.mode)
    delay = 0
    proxy = await asyncio.start_server(lambda r, w: proxy_connection(r, w, port, delay, 20 if delay else 0), "127.0.0.1", 0)
    proxy_port = proxy.sockets[0].getsockname()[1]
    async def observe_publication():
        version = manager._jpeg_version
        while not manager._closed:
            async with manager._jpeg_changed:
                await manager._jpeg_changed.wait_for(lambda: manager._closed or manager._jpeg_version != version)
                version = manager._jpeg_version
                if manager._jpeg is not None:
                    trace["publications"].append({"version": version, "id": jpeg_ids[hashlib.sha256(manager._jpeg).digest()], "time_s": time.perf_counter()})
    monitor = asyncio.create_task(observe_publication())
    meta=metadata(a.mode,cfg)
    meta.update(seconds=a.seconds,display_fps=a.display_fps,keyboard_velocity_scale=30/a.display_fps,startup_s=startup,fake=a.fake,variant=a.variant,
        attention_variant=a.attention_variant,attention_records=variants.RECORDS,
        browser_location="server loopback",presentation="headless Chromium rAF observation",
        generation_lead_ms=a.lead_ms if a.mode=='latest' else 0,
        instrumentation='Identical keyboard adapter, benchmark-only control route, frame stamps and timing hooks; native generation and MJPEG route unchanged.',
        endpoint='Keyboard state send to first decoded corresponding image observed at browser rAF; not physical scanout or semantic motion onset.')
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows"])
            meta["browser"] = browser.version
            context = await browser.new_context(viewport={"width": 960, "height": 720})
            page = await context.new_page()
            page.on("pageerror", lambda error: print("PAGE ERROR", error, flush=True))
            await page.goto(f"http://127.0.0.1:{proxy_port}/benchmark?seconds={a.seconds}")
            await page.wait_for_function("window.bench && window.bench.done", timeout=(a.seconds + 25) * 1000)
            client = await page.evaluate("window.bench")
            (a.out / "client.json").write_text(json.dumps(client))
            await page.screenshot(path=str(a.out / "browser.png"))
            await browser.close()
        assert len(client["observed"]) > (20 if a.fake else 100)
        assert not client["errors"], client["errors"]
        assert all(row["state"] == "visible" for row in client["visibility"])
        assert "failed" not in session.model_status.lower(), session.model_status
        meta["completed"] = True
    finally:
        await runner.cleanup()
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        proxy.close()
        await proxy.wait_closed()
        assert "failed" not in session.model_status.lower(), session.model_status
        (a.out / "server.json").write_text(json.dumps(trace))
        (a.out / "runtime.json").write_text(json.dumps(meta, default=str, indent=2))
    print(json.dumps({"completed": True, "case": a.out.name, "frames": len(client["observed"])}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attention-variant",choices=["default","refresh-split"],default="default")
    p.add_argument("--display-fps",type=float,choices=[30,45],default=30)
    p.add_argument("--mode",choices=["latest"],required=True)
    p.add_argument("--fake",action="store_true")
    p.add_argument("--seconds", type=int, default=75)
    p.add_argument("--lead-ms", type=float, default=170)
    p.add_argument("--variant", required=True)
    p.add_argument("--out", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
