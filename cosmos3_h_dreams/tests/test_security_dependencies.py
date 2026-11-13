# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check security floors in every locked platform and CUDA variant."""

import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "name,minimum",
    [
        ("pillow", "13.0.0"),
        ("aiohttp", "3.14.5"),
        ("urllib3", "2.8.0"),
        ("filelock", "4.2.10"),
        ("fsspec", "2026.9.0"),
        ("setuptools", "84.2.1"),
        ("idna", "3.20"),
        ("multidict", "6.9.1"),
        ("certifi", "2026.10.23"),
    ],
)
def test_locked_security_floor(name, minimum):
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    packages = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
    versions = [Version(p["version"]) for p in packages if p["name"] == name]
    assert versions, f"{name} is missing from uv.lock"
    assert all(version >= Version(minimum) for version in versions), (name, versions)
