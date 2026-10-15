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

"""Command-line application for replay and headset generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

from flashdreams.demo import IFlashDreamsApplication
from flashdreams.runtime import CanonicalInputSchema

from cosmos3_h_dreams._runtime._compile_modes import COMPILE_DENOISE_MODES

from .config import (
    COSMOS_H_DOMAIN_ID,
    COSMOS3_H_DREAMS_PRESETS,
    DEFAULT_PROMPT,
    Cosmos3HDreamsConfig,
)
from .session import Cosmos3HDreamsSession
from .session_config import SCENE_FIELDS, load_session_config


class Cosmos3HDreamsApplication(IFlashDreamsApplication):
    """Create action-replay or live input rollouts from an initial image."""

    def __init__(self) -> None:
        self._config: Cosmos3HDreamsConfig | None = None
        self.scenes: dict[str, Cosmos3HDreamsConfig] = {}
        self.scene_labels: dict[str, str] = {}

    @property
    def input_schema(self) -> CanonicalInputSchema:
        """Declare the empty host-input contract for replay and headset."""
        return CanonicalInputSchema(modalities=())

    @property
    def supports_session_reset(self) -> bool:
        """Restart rollouts from the configured initial frame in place."""
        return True

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse and validate model inputs without loading the checkpoint."""
        selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        selector.add_argument("--config", choices=COSMOS3_H_DREAMS_PRESETS)
        selector.add_argument("--session-config", type=Path)
        selected, _ = selector.parse_known_args(list(commandline_args))

        session_data = (
            load_session_config(selected.session_config)
            if selected.session_config is not None
            else None
        )
        preset_name = selected.config or (session_data or {}).get("runtime", {}).get(
            "preset"
        )
        parser = argparse.ArgumentParser(prog="flashdreams-run cosmos3-h-dreams")
        parser.add_argument("--session-config", type=Path)
        parser.add_argument("--config", choices=COSMOS3_H_DREAMS_PRESETS)
        parser.add_argument("--checkpoint")
        parser.add_argument("--initial-frame", type=Path)
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument("--actions", type=Path)
        input_group.add_argument("--headset", action="store_true")
        parser.add_argument("--domain-id", default=COSMOS_H_DOMAIN_ID, type=int)
        parser.add_argument("--prompt", default=DEFAULT_PROMPT)
        parser.add_argument("--normalizer", type=Path)
        parser.add_argument("--headset-transport", choices=("webxr",), default="webxr")
        parser.add_argument("--headset-relay-url")
        parser.add_argument("--headset-port", type=int, default=8444)
        parser.add_argument(
            "--headset-startup-timeout",
            dest="headset_startup_timeout_s",
            metavar="HEADSET_STARTUP_TIMEOUT",
            type=float,
            default=300.0,
        )
        parser.add_argument(
            "--headset-manual-xr", action=argparse.BooleanOptionalAction, default=False
        )
        parser.add_argument("--headset-jaw-open", type=float, default=1.0)
        parser.add_argument("--headset-jaw-closed", type=float, default=-2.0)
        parser.add_argument("--credential-path")
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument(
            "--fps",
            type=float,
            default=15.0,
            help="Action and generated-video frame rate used for temporal mRoPE.",
        )
        parser.add_argument("--height", type=int, default=368)
        parser.add_argument("--width", type=int, default=640)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--chunk-size", type=int, default=4)
        parser.add_argument("--denoise-steps", type=int, default=4)
        parser.add_argument("--encoder", choices=("wan",), default="wan")
        parser.add_argument(
            "--decoder", choices=("wan", "taehv", "taehv2super"), default="wan"
        )
        parser.add_argument("--window-frames", type=int, default=12)
        parser.add_argument(
            "--fademem-slots",
            type=int,
            default=12,
            help="Historical latent-frame entries, including frame zero",
        )
        parser.add_argument(
            "--fademem-merge-selection",
            choices=("age", "cosine"),
            default="age",
            help="Rank adjacent merge pairs by age gap or clean-latent cosine similarity",
        )
        parser.add_argument(
            "--fademem-beta",
            type=float,
            default=0.3,
            help="Distance exponent in (0, 1]",
        )
        parser.add_argument(
            "--cache-mode", choices=("cat", "block", "fademem"), default="cat"
        )
        parser.add_argument(
            "--overlap-cache-update",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--compile-denoise", action=argparse.BooleanOptionalAction, default=False
        )
        parser.add_argument(
            "--compile-denoise-mode",
            choices=COMPILE_DENOISE_MODES,
            default="max-autotune-no-cudagraphs",
        )
        parser.add_argument(
            "--cuda-graph-denoise", action=argparse.BooleanOptionalAction, default=False
        )
        parser.add_argument("--cuda-graph-warmup-iters", type=int, default=2)
        parser.add_argument(
            "--profile",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Report stage timings, KV-cache memory, and whole-run peak/average CUDA allocator VRAM.",
        )
        if preset_name is not None:
            if preset_name not in COSMOS3_H_DREAMS_PRESETS:
                parser.error(f"unknown model preset: {preset_name}")
            preset = COSMOS3_H_DREAMS_PRESETS[preset_name]
            allowed = {field.name for field in fields(Cosmos3HDreamsConfig)} - {
                "actions"
            }
            unknown = preset.keys() - allowed
            if unknown:
                parser.error(f"unknown config options: {', '.join(sorted(unknown))}")
            parser.set_defaults(**preset)
        if session_data is not None:
            parser.set_defaults(
                **{
                    key: value
                    for key, value in session_data["runtime"].items()
                    if key != "preset"
                }
            )
        scene_defaults = {key: parser.get_default(key) for key in SCENE_FIELDS}
        scenes = (
            session_data["scenes"]
            if session_data
            else [{"name": "default", "label": "Default"}]
        )
        self.scenes = {}
        self.scene_labels = {}
        for scene in scenes:
            parser.set_defaults(**scene_defaults)
            parser.set_defaults(
                **{key: value for key, value in scene.items() if key in SCENE_FIELDS}
            )
            args = parser.parse_args(list(commandline_args))
            if args.checkpoint is None or args.initial_frame is None:
                parser.error("--checkpoint and --initial-frame are required")
            config = Cosmos3HDreamsConfig(
                **{
                    field.name: getattr(args, field.name)
                    for field in fields(Cosmos3HDreamsConfig)
                    if hasattr(args, field.name)
                }
            )
            self.scenes[scene["name"]] = config
            self.scene_labels[scene["name"]] = scene["label"]
        self._config = next(iter(self.scenes.values()))

    def create_session(self) -> Cosmos3HDreamsSession:
        """Create one uninitialized generation session."""
        if self._config is None:
            raise RuntimeError(
                "Cosmos3HDreamsApplication.init() must run before create_session()."
            )
        return Cosmos3HDreamsSession(config=self._config)


def create_app() -> IFlashDreamsApplication:
    """Return a new Cosmos3-H-Dreams application."""
    return Cosmos3HDreamsApplication()
