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

"""CPU checks for Cosmos3-H-Dreams application configuration."""

import numpy as np
import pytest
from cosmos3_h_dreams.config import COSMOS3_H_DREAMS_PRESETS
from cosmos3_h_dreams.model import _resolution_geometry
from PIL import Image

from cosmos3_h_dreams import Cosmos3HDreamsApplication, Cosmos3HDreamsSession

pytestmark = pytest.mark.ci_cpu


def test_application_builds_offline_session(tmp_path) -> None:
    frame = tmp_path / "frame.png"
    actions = tmp_path / "actions.npy"
    Image.new("RGB", (32, 32)).save(frame)
    np.save(actions, np.zeros((16, 28), dtype=np.float32))
    application = Cosmos3HDreamsApplication()

    application.init(
        [
            "--checkpoint",
            "/checkpoint",
            "--initial-frame",
            str(frame),
            "--actions",
            str(actions),
            "--prompt",
            "A robot folds a towel.",
            "--device",
            "cpu",
            "--height",
            "288",
            "--width",
            "512",
        ]
    )
    session = application.create_session()

    assert isinstance(session, Cosmos3HDreamsSession)
    assert session.config.domain_id == 32
    assert session.config.device == "cpu"
    assert (session.config.height, session.config.width) == (288, 512)
    assert session.config.encoder == "wan"
    assert not session.config.compile_denoise
    assert application.input_schema.modalities == ()


def test_resolution_geometry_matches_vae_and_patch_compression() -> None:
    assert _resolution_geometry(288, 512, patch=2) == ((18, 32), (9, 16))


def test_application_accepts_denoise_optimization_flags(tmp_path) -> None:
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
            "--cache-mode",
            "block",
            "--overlap-cache-update",
            "--compile-denoise",
            "--compile-denoise-mode",
            "default",
            "--cuda-graph-denoise",
            "--cuda-graph-warmup-iters",
            "3",
        ]
    )

    config = application.create_session().config
    assert config.cache_mode == "block"
    assert config.overlap_cache_update
    assert config.compile_denoise
    assert config.compile_denoise_mode == "default"
    assert config.cuda_graph_denoise
    assert config.cuda_graph_warmup_iters == 3


@pytest.mark.parametrize(
    "flags, match",
    [
        (["--compile-denoise"], "requires cache_mode"),
        (["--overlap-cache-update"], "requires cache_mode"),
        (["--cuda-graph-denoise"], "requires compile_denoise"),
        (["--cuda-graph-warmup-iters", "-1"], "must be non-negative"),
    ],
)
def test_application_rejects_invalid_optimization_flags(tmp_path, flags, match) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32)).save(frame)

    with pytest.raises(ValueError, match=match):
        Cosmos3HDreamsApplication().init(
            [
                "--checkpoint",
                "/checkpoint",
                "--initial-frame",
                str(frame),
                "--quest",
                *flags,
            ]
        )


def test_named_config_resolves_before_explicit_overrides(tmp_path) -> None:
    frame = tmp_path / "frame.png"
    actions = tmp_path / "actions.npy"
    Image.new("RGB", (32, 32)).save(frame)
    np.save(actions, np.zeros((16, 28), dtype=np.float32))
    application = Cosmos3HDreamsApplication()

    application.init(
        [
            "--chunk-size",
            "2",
            "--config",
            "wan-wan-chunk4-block-compile",
            "--checkpoint",
            "/checkpoint",
            "--initial-frame",
            str(frame),
            "--actions",
            str(actions),
            "--denoise-steps",
            "2",
            "--decoder",
            "taehv",
            "--no-overlap-cache-update",
            "--no-profile",
        ]
    )

    config = application.create_session().config
    assert (config.domain_id, config.fps, config.height, config.width) == (
        25,
        10.0,
        480,
        832,
    )
    assert (config.chunk_size, config.denoise_steps) == (2, 2)
    assert (config.encoder, config.decoder) == ("wan", "taehv")
    assert config.cache_mode == "block"
    assert config.compile_denoise and config.cuda_graph_denoise
    assert not config.overlap_cache_update
    assert not config.profile


def test_without_config_keeps_required_inputs_explicit(tmp_path) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32)).save(frame)

    with pytest.raises(SystemExit):
        Cosmos3HDreamsApplication().init(["--initial-frame", str(frame), "--quest"])


def test_named_config_supplies_checkpoint_and_initial_frame(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32)).save(frame)
    monkeypatch.setitem(COSMOS3_H_DREAMS_PRESETS["base"], "checkpoint", "/checkpoint")
    monkeypatch.setitem(COSMOS3_H_DREAMS_PRESETS["base"], "initial_frame", frame)
    application = Cosmos3HDreamsApplication()

    application.init(["--config", "base", "--quest"])

    config = application.create_session().config
    assert config.checkpoint == "/checkpoint"
    assert config.initial_frame == frame


def test_named_config_rejects_misspelled_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(COSMOS3_H_DREAMS_PRESETS["base"], "chonk_size", 4)

    with pytest.raises(SystemExit) as error:
        Cosmos3HDreamsApplication().init(["--config", "base", "--quest"])
    assert error.value.code == 2


def test_explicit_boolean_flags_can_restore_eager_cache(tmp_path) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32)).save(frame)
    application = Cosmos3HDreamsApplication()

    application.init(
        [
            "--config",
            "wan-wan-chunk4-block-compile",
            "--checkpoint",
            "/checkpoint",
            "--initial-frame",
            str(frame),
            "--quest",
            "--cache-mode",
            "cat",
            "--no-compile-denoise",
            "--no-cuda-graph-denoise",
            "--no-overlap-cache-update",
        ]
    )

    config = application.create_session().config
    assert config.cache_mode == "cat"
    assert not config.compile_denoise
    assert not config.cuda_graph_denoise
    assert not config.overlap_cache_update


@pytest.mark.parametrize("selection", ["age", "cosine"])
def test_fademem_cli_controls(tmp_path, selection) -> None:
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
            "--cache-mode",
            "fademem",
            "--fademem-slots",
            "9",
            "--fademem-beta",
            "0.4",
            "--fademem-merge-selection",
            selection,
            "--compile-denoise",
            "--overlap-cache-update",
        ]
    )
    config = application.create_session().config
    assert config.cache_mode == "fademem"
    assert config.fademem_slots == 9 and config.fademem_beta == 0.4
    assert config.fademem_merge_selection == selection
    assert config.compile_denoise and config.overlap_cache_update


def test_builtin_presets_require_explicit_assets() -> None:
    assert COSMOS3_H_DREAMS_PRESETS
    assert all(
        "checkpoint" not in preset and "initial_frame" not in preset
        for preset in COSMOS3_H_DREAMS_PRESETS.values()
    )


def test_application_rejects_removed_keyboard_option(tmp_path, capsys) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32)).save(frame)
    with pytest.raises(SystemExit) as error:
        Cosmos3HDreamsApplication().init(
            [
                "--checkpoint",
                "/checkpoint",
                "--initial-frame",
                str(frame),
                "--quest",
                "--keyboard",
            ]
        )
    assert error.value.code == 2
    assert "unrecognized arguments: --keyboard" in capsys.readouterr().err
