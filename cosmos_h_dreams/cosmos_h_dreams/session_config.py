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

"""Session YAML with shared model options, Quest options, and named scenes."""

from __future__ import annotations

from dataclasses import fields
from glob import glob, has_magic
from math import isfinite
from pathlib import Path
from typing import Any, get_args, get_type_hints

import yaml

from .config import CosmosHDreamsConfig

REPO_ROOT: Path | None = None
"""Optional asset-root override; paths otherwise start at the launch directory."""

SCENE_FIELDS = {"initial_frame", "prompt", "domain_id", "normalizer", "actions"}


def load_session_config(path: Path) -> dict[str, Any]:
    """Load and validate a session file, resolving asset paths from the launch directory."""
    base = REPO_ROOT or Path.cwd()
    path = path.expanduser().resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid session YAML {path}: {error}") from error
    _mapping(data, {"runtime", "quest", "scenes"}, "session")
    runtime = data.setdefault("runtime", {})
    quest = data.setdefault("quest", {})
    model_fields = {field.name for field in fields(CosmosHDreamsConfig)}
    _mapping(runtime, (model_fields - SCENE_FIELDS) | {"preset"}, "runtime")
    _mapping(
        quest,
        {
            "preset",
            "host",
            "port",
            "cert",
            "key",
            "display_fps",
            "video_transport",
            "quest_input_fps",
            "overlap_playback",
            "print_actions",
        },
        "quest",
    )
    for section in (runtime, quest):
        if "preset" in section and not isinstance(section["preset"], str):
            raise ValueError("preset must be a string")
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scenes must be a non-empty list")
    expanded = []
    for scene in scenes:
        _mapping(scene, SCENE_FIELDS | {"name", "label"}, "scene")
        initial_frame = scene.get("initial_frame")
        if isinstance(initial_frame, str) and has_magic(initial_frame):
            matches = sorted(
                match
                for match in glob(str(_path(initial_frame, base)), recursive=True)
                if Path(match).is_file()
            )
            if not matches:
                raise ValueError(
                    f"initial_frame pattern matches no files: {initial_frame}"
                )
            expanded.extend(
                {
                    **scene,
                    "initial_frame": match,
                    "name": Path(match).name,
                    "label": Path(match).name,
                }
                for match in matches
            )
        else:
            expanded.append(scene)
    scenes = data["scenes"] = expanded
    names = set()
    hints = get_type_hints(CosmosHDreamsConfig)
    for scene in scenes:
        name = scene.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("scene names must be non-empty and unique")
        names.add(name)
        if "initial_frame" not in scene:
            raise ValueError(f"scene {name!r} requires initial_frame")
        if (
            not isinstance(scene.setdefault("label", name), str)
            or not scene["label"].strip()
        ):
            raise ValueError(f"scene {name!r} requires a non-empty label")
    for section in [runtime, *scenes]:
        for key, value in section.items():
            if key not in hints:
                continue
            expected = get_args(hints[key]) or (hints[key],)
            if value is None and type(None) in expected:
                continue
            if Path in expected:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{key} must be a path string")
                section[key] = _path(value, base)
            elif not any(
                type(value) is kind or (kind is float and type(value) is int)
                for kind in expected
            ):
                raise ValueError(f"Invalid type for {key}")
            elif float in expected and not isfinite(float(value)):
                raise ValueError(f"{key} must be finite")
    for key, value in quest.items():
        if key in {"cert", "key"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"quest.{key} must be a path string")
            quest[key] = str(_path(value, base))
        elif key in {"display_fps", "port"} and type(value) is not int:
            raise ValueError(f"quest.{key} must be an integer")
        elif key == "quest_input_fps" and type(value) not in (float, int):
            raise ValueError("quest.quest_input_fps must be a number")
        elif key in {"overlap_playback", "print_actions"} and type(value) is not bool:
            raise ValueError(f"quest.{key} must be a boolean")
        elif key in {"host", "video_transport"} and not isinstance(value, str):
            raise ValueError(f"quest.{key} must be a string")
    if "checkpoint" in runtime:
        checkpoint = runtime["checkpoint"]
        # Registry names and URIs retain their meaning; filesystem paths start at the repo root.
        if checkpoint.startswith((".", "/", "~")) or (
            "://" not in checkpoint
            and (
                "/" in checkpoint
                or Path(checkpoint).suffix in {".pt", ".pth", ".json", ".safetensors"}
            )
        ):
            runtime["checkpoint"] = str(_path(checkpoint, base))
    return data


def _mapping(value: Any, allowed: set[str], context: str) -> None:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a mapping with string keys")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {context} options: {', '.join(sorted(unknown))}")


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()
