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

"""CPU checks for application selection and headset argument forwarding."""

import pytest

from cosmos3_h_dreams import cli, headset_server

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("name", ["cosmos-h-dreams", "cosmos_h_dreams", "cosmos3-h-dreams", "unknown"])
def test_unknown_application_lists_supported_name(name, capsys, monkeypatch):
    def fail_if_started(*args):
        pytest.fail("Unknown application started the headset server")

    monkeypatch.setattr(headset_server, "main", fail_if_started)
    with pytest.raises(SystemExit) as error:
        cli.main([name, "--session-config", "does-not-exist.yaml"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert f"Unknown application {name!r}" in message
    assert "Available applications: headset" in message
    assert "Traceback" not in message


def test_valid_application_forwards_options_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(headset_server, "main", calls.append)
    options = ["--session-config", "session.yaml", "--checkpoint", "model.pt"]
    cli.main(["headset", *options])
    cli.main(["headset", "--help"])
    assert calls == [options, ["--help"]]


@pytest.mark.parametrize("args,code", [([], 2), (["--help"], 0)])
def test_launcher_usage_lists_available_application(args, code, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == code
    output = capsys.readouterr()
    assert "APPLICATION" in output.out + output.err
    if code == 0:
        assert "headset" in output.out
