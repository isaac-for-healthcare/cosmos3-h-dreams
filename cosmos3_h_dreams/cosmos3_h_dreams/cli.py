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

"""Product launcher for Cosmos3-H-Dreams headset sessions."""

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Validate the application name and forward its headset options.

    Args:
        argv: Command-line arguments, defaulting to the process arguments.
    """
    args = sys.argv[1:] if argv is None else argv
    applications = ("headset",)
    parser = argparse.ArgumentParser(
        description="Launch a headset session. Available applications: "
        + ", ".join(applications),
        usage="%(prog)s APPLICATION [HEADSET_OPTIONS ...]",
        allow_abbrev=False,
    )
    parser.add_argument("application", help="application to launch")
    selected = parser.parse_args(args[:1])
    if selected.application not in applications:
        parser.error(
            f"Unknown application {selected.application!r}.\n"
            f"Available applications: {', '.join(applications)}"
        )

    from .headset_server import main as run_headset

    run_headset(args[1:])


if __name__ == "__main__":
    main()
