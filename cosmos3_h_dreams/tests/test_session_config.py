# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for session YAML, preset precedence, and scene transitions."""

import asyncio
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml
from cosmos3_h_dreams.app import Cosmos3HDreamsApplication
from cosmos3_h_dreams.config import Cosmos3HDreamsConfig
from cosmos3_h_dreams.headset_server import LatestHeadsetFrames, HeadsetServer
from cosmos3_h_dreams.session import Cosmos3HDreamsSession
from cosmos3_h_dreams.session_config import load_session_config

from cosmos3_h_dreams import headset_server
from cosmos3_h_dreams import session as session_module
from cosmos3_h_dreams import session_config as session_config_module

pytestmark = pytest.mark.ci_cpu


def _session_file(tmp_path):
    (tmp_path / "a.png").touch()
    (tmp_path / "b.png").touch()
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs/session.yaml"
    data = {
        "runtime": {"preset": "base", "checkpoint": "./model.pt", "device": "cpu"},
        "headset": {"preset": "headset-base", "cert": "cert.pem", "key": "key.pem"},
        "scenes": [
            {
                "name": "a",
                "label": "Scene A",
                "initial_frame": "a.png",
                "prompt": "First scene.",
            },
            {"name": "b", "initial_frame": "b.png"},
        ],
    }
    path.write_text(yaml.safe_dump(data))
    return path, data


def test_yaml_paths_and_scene_defaults(tmp_path, monkeypatch):
    path, _ = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(path.parent)
    application = Cosmos3HDreamsApplication()
    application.init(["--session-config", str(path), "--headset", "--fps", "12"])
    assert list(application.scenes) == ["a", "b"]
    assert application.scene_labels == {"a": "Scene A", "b": "b"}
    assert application.scenes["a"].initial_frame == tmp_path / "a.png"
    assert application.scenes["b"].initial_frame == tmp_path / "b.png"
    assert application.scenes["a"].checkpoint == str(tmp_path / "model.pt")
    assert application.scenes["a"].prompt == "First scene."
    assert application.scenes["b"].prompt != "First scene."
    assert all(config.fps == 12 for config in application.scenes.values())


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda data: data["runtime"].update(chonk_size=2), "unknown runtime"),
        (lambda data: data["headset"].update(preset=23), "preset must"),
        (lambda data: data["headset"].update(display_fps=True), "integer"),
        (lambda data: data["headset"].update(video_transport="webrtc"), "must be mjpeg"),
        (lambda data: data["scenes"].append(dict(data["scenes"][0])), "unique"),
        (lambda data: data["scenes"][0].update(height=288), "unknown scene"),
        (lambda data: data["scenes"][0].update(domain_id=True), "Invalid type"),
        (lambda data: data.update(scenes=[]), "non-empty list"),
    ],
)
def test_yaml_rejects_invalid_schema(tmp_path, change, match):
    path, data = _session_file(tmp_path)
    change(data)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=match):
        load_session_config(path)


def test_both_yaml_presets_and_cli_overrides(tmp_path, monkeypatch):
    path, _ = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "headset_server",
            "--session-config",
            str(path),
            "--config",
            "wan-taehv-chunk2-2steps-fademem-compile",
            "--headset-config",
            "headset-base",
            "--display-fps",
            "17",
            "--fps",
            "12",
        ],
    )
    certificates = []
    monkeypatch.setattr(
        headset_server.ssl,
        "SSLContext",
        lambda _: SimpleNamespace(
            load_cert_chain=lambda *args: certificates.append(args)
        ),
    )
    monkeypatch.setattr(headset_server, "create_app", lambda manager: manager)
    managers = []
    monkeypatch.setattr(
        headset_server.web, "run_app", lambda manager, **_: managers.append(manager)
    )
    headset_server.main()
    (manager,) = managers
    assert certificates == [(str(tmp_path / "cert.pem"), str(tmp_path / "key.pem"))]
    assert manager.display_fps == 17
    assert manager.video_transport == "mjpeg"
    assert manager.frames._period_ns == round(1e9 / 30)
    assert all(
        config.fps == 12 and config.chunk_size == 2
        for config in manager.scenes.values()
    )
    assert manager.session.config.decoder == "taehv"
    assert list(manager.scenes) == ["a", "b"]


def test_switch_keeps_model_changes_domain_and_recalibrates(tmp_path, monkeypatch):
    frame = tmp_path / "frame.png"
    frame.touch()
    config = Cosmos3HDreamsConfig(
        checkpoint="model.pt", initial_frame=frame, prompt="A", device="cpu"
    )
    calls = []
    rollout = SimpleNamespace(
        wait_for_pending_cache_update=lambda **_: calls.append("wait")
    )
    model = SimpleNamespace(
        net=SimpleNamespace(config=SimpleNamespace(num_embodiment_domains=40)),
        action_domain_id=config.domain_id,
        new_session=lambda scene: calls.append(scene) or object(),
    )
    session = Cosmos3HDreamsSession(config=config)
    session._model, session._rollout, session._scene = model, rollout, object()
    session._headset_source = SimpleNamespace(
        set_normalizer=lambda path: calls.append(("normalizer", path))
    )
    monkeypatch.setattr(
        session_module, "prepare_scene", lambda config, _: config.prompt
    )
    session._rows_consumed = 20
    new_config = replace(config, prompt="B", domain_id=25)
    session.switch_scene(new_config)
    assert session._model is model and model.action_domain_id == 25
    assert session.config is new_config and session._scene == "B"
    assert session._rows_consumed == 0
    assert calls == ["wait", "B", ("normalizer", None)]
    with pytest.raises(ValueError, match="shared runtime"):
        session.switch_scene(replace(new_config, height=288))
    with pytest.raises(ValueError, match="domain_id"):
        session.switch_scene(replace(new_config, domain_id=40))


def test_scene_request_wakes_idle_generation_and_handles_failure():
    async def check():
        configs = {"a": object(), "b": object()}
        calls = []

        def switch(config):
            calls.append(config)
            if config is configs["a"]:
                raise ValueError("bad scene")

        manager = HeadsetServer(
            session=SimpleNamespace(switch_scene=switch),
            frames=LatestHeadsetFrames(fps=10),
            scenes=configs,
        )
        manager.runtime_ready = True
        manager._producer = asyncio.create_task(manager._generate())
        manager.source = SimpleNamespace(request_restart=lambda: calls.append("wake"))
        await asyncio.sleep(0)
        try:
            await asyncio.wait_for(manager.select_scene("b"), timeout=2)
            assert calls == ["wake", configs["b"]]
            assert manager.active_scene == "b" and manager._scene_request is None
            with pytest.raises(headset_server.web.HTTPInternalServerError):
                await asyncio.wait_for(manager.select_scene("a"), timeout=2)
            assert manager.active_scene == "b"
            assert manager._scene_request is None
            with pytest.raises(headset_server.web.HTTPBadRequest):
                await manager.select_scene({"name": "a"})
        finally:
            manager._producer.cancel()
            await asyncio.gather(manager._producer, return_exceptions=True)

    asyncio.run(check())


def test_switch_discards_inflight_output_and_flushes_video():
    async def check():
        entered = threading.Event()
        release = threading.Event()
        prepared = []
        manager = None

        def step(_window):
            entered.set()
            assert release.wait(timeout=2)
            return SimpleNamespace(metadata={"action_rows": 1})

        def switch(_config):
            manager._ready.clear()

        manager = HeadsetServer(
            session=SimpleNamespace(
                config=SimpleNamespace(fps=10),
                next_step_requirements=lambda: SimpleNamespace(input_frame_count=1),
                step=step,
                switch_scene=switch,
            ),
            frames=LatestHeadsetFrames(fps=10),
            scenes={"a": object(), "b": object()},
        )
        manager.source = SimpleNamespace(request_restart=release.set)
        manager.track = SimpleNamespace(prepare_result_frames=prepared.append)
        manager.socket = object()
        manager._jpeg = b"old video"
        manager.runtime_ready = True
        manager._ready.set()
        manager._producer = asyncio.create_task(manager._generate())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await asyncio.wait_for(manager.select_scene("b"), timeout=2)
            assert manager.active_scene == "b"
            assert prepared == []
            assert manager._jpeg is None
        finally:
            release.set()
            manager._producer.cancel()
            await asyncio.gather(manager._producer, return_exceptions=True)

    asyncio.run(check())


def test_yaml_resolves_all_assets_from_repo_root(tmp_path, monkeypatch):
    path, data = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(path.parent)
    data["scenes"][0].update(normalizer="data/stats.npz", actions="data/actions.npy")
    data["scenes"][1]["initial_frame"] = str(tmp_path / "absolute.png")
    path.write_text(yaml.safe_dump(data))
    resolved = load_session_config(path)
    assert resolved["scenes"][0]["normalizer"] == tmp_path / "data/stats.npz"
    assert resolved["scenes"][0]["actions"] == tmp_path / "data/actions.npy"
    assert resolved["scenes"][1]["initial_frame"] == tmp_path / "absolute.png"
    assert resolved["runtime"]["checkpoint"] == str(tmp_path / "model.pt")
    assert resolved["headset"]["cert"] == str(tmp_path / "cert.pem")
    assert resolved["headset"]["key"] == str(tmp_path / "key.pem")


def test_initial_frame_wildcard_builds_named_scenes(tmp_path, monkeypatch):
    path, data = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(path.parent)
    (tmp_path / "other.jpg").touch()
    (tmp_path / "directory.png").mkdir()
    data["scenes"] = [
        {"initial_frame": "*.png", "prompt": "Shared prompt.", "domain_id": 25}
    ]
    path.write_text(yaml.safe_dump(data))
    application = Cosmos3HDreamsApplication()
    application.init(["--session-config", str(path), "--headset"])
    assert list(application.scenes) == ["a.png", "b.png"]
    assert application.scene_labels == {"a.png": "a.png", "b.png": "b.png"}
    assert application.scenes["a.png"].initial_frame == tmp_path / "a.png"
    assert application.scenes["b.png"].initial_frame == tmp_path / "b.png"
    assert all(
        config.prompt == "Shared prompt." and config.domain_id == 25
        for config in application.scenes.values()
    )


@pytest.mark.parametrize("pattern", ["missing/*.png", "configs/*.png"])
def test_initial_frame_wildcard_rejects_empty_matches(tmp_path, monkeypatch, pattern):
    path, data = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    data["scenes"] = [{"initial_frame": pattern}]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="matches no files"):
        load_session_config(path)


def test_initial_frame_wildcards_mix_with_explicit_scenes_and_reject_duplicates(
    tmp_path, monkeypatch
):
    path, data = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    data["scenes"] = [data["scenes"][0], {"initial_frame": str(tmp_path / "[ab].png")}]
    path.write_text(yaml.safe_dump(data))
    assert [scene["name"] for scene in load_session_config(path)["scenes"]] == [
        "a",
        "a.png",
        "b.png",
    ]
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/a.png").touch()
    data["scenes"] = [{"initial_frame": "**/*.png"}]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="unique"):
        load_session_config(path)
