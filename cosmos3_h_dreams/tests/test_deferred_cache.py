# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for retained cache work, owner ordering, cancellation and cleanup."""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from cosmos3_h_dreams._runtime._hand_kv_session import HandKVSession, validate_cache_options
from cosmos3_h_dreams.headset_server import HeadsetServer, LatestHeadsetFrames
from cosmos3_h_dreams.session import Cosmos3HDreamsSession

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("overrides", [
    {"cache_mode": "cat", "overlap_cache_update": False},
    {"cache_mode": "block"}, {"overlap_cache_update": False}, {"device": "cpu"},
])
def test_invalid_deferred_options(overrides):
    options = dict(cache_mode="fademem", overlap_cache_update=True, device="cuda:0",
                   defer_cache_refresh=True)
    with pytest.raises(ValueError, match="defer_cache_refresh"):
        validate_cache_options(**(options | overrides))


def test_retained_work_drains_once_on_owner_and_propagates_errors():
    rollout = HandKVSession.__new__(HandKVSession)
    rollout.defer_cache_refresh = True
    rollout._deferred_cache = rollout._deferred_cache_owner = None
    rollout._cache_stream = None
    rollout._pending_cache = []
    calls = []
    rollout._submit_refresh_now = lambda *args: calls.append(args) or 0.25
    tensor = torch.ones(2, requires_grad=True)
    rollout._submit_refresh(1, tensor, tensor, 2)
    assert not calls and not rollout._deferred_cache[1].requires_grad
    with pytest.raises(RuntimeError, match="Previous deferred"):
        rollout._submit_refresh(3, tensor, tensor, 2)
    with ThreadPoolExecutor(max_workers=1) as other_owner:
        with pytest.raises(RuntimeError, match="inference owner"):
            other_owner.submit(rollout.submit_pending_cache_update).result()
    assert rollout._deferred_cache is not None
    rollout.wait_for_pending_cache_update(block=True)
    rollout.wait_for_pending_cache_update()
    assert len(calls) == 1 and calls[0][0] == 1
    assert rollout._deferred_cache is None

    def fail(*args):
        raise ValueError("injected cache failure")

    rollout._submit_refresh_now = fail
    drained = []
    rollout._cache_stream = SimpleNamespace(synchronize=lambda: drained.append(True))
    rollout._submit_refresh(3, tensor, tensor, 2)
    with pytest.raises(ValueError, match="injected cache failure"):
        rollout.submit_pending_cache_update()
    assert drained == [True] and rollout._deferred_cache is None


def test_browser_deferred_refresh_requires_dedicated_owner():
    with pytest.raises(ValueError, match="dedicated_inference_thread"):
        HeadsetServer(session=SimpleNamespace(config=SimpleNamespace(defer_cache_refresh=True)),
                      frames=LatestHeadsetFrames(fps=30))


@pytest.mark.parametrize("decode", [False, True])
def test_raw_tokens_submit_before_returning_but_pixels_defer(decode):
    calls = []
    session = Cosmos3HDreamsSession.__new__(Cosmos3HDreamsSession)
    session.config = SimpleNamespace(profile=False, defer_cache_refresh=True)
    session._headset_source = None
    session._rows_consumed = session._step_index = 0
    session._start_vram_profile = lambda: None
    session.next_step_requirements = lambda: SimpleNamespace(input_frame_count=8)
    session._require_scene = lambda: SimpleNamespace(replay_actions=torch.zeros(8, 64))
    rollout = SimpleNamespace(last_timing={},
        step=lambda *a, **kw: [torch.zeros(1, 1, 1, 1)],
        submit_pending_cache_update=lambda: calls.append("refresh"))
    model = SimpleNamespace(_crop_decode_latent=lambda latent: latent,
                            decode_tensor=lambda latent: torch.zeros(1, 3, 2, 2))
    session._require_runtime = lambda: (model, rollout)
    result = session.step(SimpleNamespace(values={}), decode=decode)
    assert calls == ([] if decode else ["refresh"])
    assert result.metadata["streaming_mode"] == ("pixels" if decode else "raw_tokens")


def test_cancelled_cache_waiter_preserves_work_and_close_waits():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def refresh():
        calls.append(("refresh", threading.get_ident()))
        entered.set()
        assert release.wait(5)
        calls.append(("finished", threading.get_ident()))

    async def check():
        manager = HeadsetServer(
            session=SimpleNamespace(config=SimpleNamespace(defer_cache_refresh=True),
                submit_pending_cache_update=refresh,
                close=lambda: calls.append(("close", threading.get_ident()))),
            frames=LatestHeadsetFrames(fps=30), dedicated_inference_thread=True,
        )
        try:
            manager._queue_cache_refresh()
            assert await asyncio.to_thread(entered.wait, 2)
            pending = manager._cache_refresh_future
            waiter = asyncio.create_task(manager._run_inference(lambda: calls.append(("next", 0))))
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert not pending.cancelled() and not pending.done()
            closing = asyncio.create_task(manager.close())
            await asyncio.sleep(.01)
            assert not closing.done()
            release.set()
            await asyncio.wait_for(closing, 2)
            assert [name for name, _ in calls] == ["refresh", "finished", "close"]
            assert len({owner for _, owner in calls}) == 1
            assert manager._cache_refresh_future is None
        finally:
            release.set()
            await manager.close()

    asyncio.run(check())


@pytest.mark.parametrize("surface_error_on_close", [False, True])
def test_background_failure_surfaces_and_does_not_skip_cleanup(surface_error_on_close):
    calls = []

    def fail():
        calls.append(("refresh", threading.get_ident()))
        raise ValueError("injected cache failure")

    async def check():
        manager = HeadsetServer(
            session=SimpleNamespace(config=SimpleNamespace(defer_cache_refresh=True),
                submit_pending_cache_update=fail,
                close=lambda: calls.append(("close", threading.get_ident()))),
            frames=LatestHeadsetFrames(fps=30), dedicated_inference_thread=True,
        )
        try:
            manager._queue_cache_refresh()
            with pytest.raises(ValueError, match="injected cache failure"):
                if surface_error_on_close:
                    await manager.close()
                else:
                    await manager._run_inference(lambda: calls.append(("unexpected", 0)))
            await manager.close()
            assert [name for name, _ in calls] == ["refresh", "close"]
            assert len({owner for _, owner in calls}) == 1
        finally:
            await manager.close()

    asyncio.run(check())


def test_cancellation_does_not_discard_an_unobserved_completed_error(monkeypatch):
    async def check():
        closed = []
        manager = HeadsetServer(
            session=SimpleNamespace(config=SimpleNamespace(defer_cache_refresh=True),
                                    close=lambda: closed.append(True)),
            frames=LatestHeadsetFrames(fps=30), dedicated_inference_thread=True,
        )
        pending = asyncio.get_running_loop().create_future()
        pending.set_exception(ValueError("completed cache failure"))
        manager._cache_refresh_future = pending
        cancelled = asyncio.get_running_loop().create_future()
        cancelled.cancel()
        # Exercise the race where caller cancellation wins over a ready error.
        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "shield", lambda _: cancelled)
            with pytest.raises(asyncio.CancelledError):
                await manager._wait_for_cache_refresh()
        assert manager._cache_refresh_future is pending
        with pytest.raises(ValueError, match="completed cache failure"):
            await manager.close()
        assert closed == [True]

    asyncio.run(check())
