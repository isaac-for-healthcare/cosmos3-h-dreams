# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Separate Cosmos3 keyboard demo with MJPEG and upstream token transport."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aiohttp import web

from .app import Cosmos3HDreamsApplication
from .headset_server import HeadsetServer, LatestHeadsetFrames, create_app
from .token_decoder import prepare_decoder
from .keyboard import KeyboardActionSource
from .session import Cosmos3HDreamsSession


def main() -> None:
    """Serve keyboard or recorded actions on a dedicated loopback port."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--decoder-dir", type=Path, default=Path("data_local/decoder"))
    parser.add_argument("--controls", choices=("keyboard", "replay"), default="keyboard")
    parser.add_argument("--eager-warmup-steps", type=int, default=0)
    args, model_args = parser.parse_known_args()
    if args.eager_warmup_steps < 0:
        parser.error("--eager-warmup-steps must be non-negative")
    application = Cosmos3HDreamsApplication()
    application.init((["--headset"] if args.controls == "keyboard" else []) + model_args)
    session = application.create_session()
    if session.config.decoder != "taehv":
        parser.error("Token streaming requires --decoder taehv")
    if args.controls == "keyboard":
        def source_factory(**kwargs):
            manager.source = KeyboardActionSource(**kwargs)
            return manager.source

        session = Cosmos3HDreamsSession(
            config=session.config, headset_source_factory=source_factory,
            eager_warmup_steps=args.eager_warmup_steps,
        )
    else:
        session = Cosmos3HDreamsSession(
            config=session.config, eager_warmup_steps=args.eager_warmup_steps,
        )
    manager = HeadsetServer(
        session=session, frames=LatestHeadsetFrames(fps=session.config.fps),
        display_fps=round(session.config.fps), scenes=application.scenes,
        scene_labels=application.scene_labels,
    )
    manager.token_descriptor = prepare_decoder(
        args.decoder_dir, height=session.config.height,
        width=session.config.width, device=session.config.device,
    )
    app = create_app(manager)
    web_dir = Path(__file__).parent / "web"

    async def page(request):
        return web.FileResponse(web_dir / "token_demo.html", headers={"Cache-Control": "no-store"})

    async def decoder(request):
        descriptor = manager.token_descriptor
        if descriptor is None or request.match_info["version"] != descriptor["version"]:
            raise web.HTTPNotFound()
        return web.FileResponse(args.decoder_dir / descriptor["version"] / "fp32.onnx")

    app.router.add_get("/", page)
    app.router.add_get("/keyboard", page)
    app.router.add_get("/decoder/{version}/fp32.onnx", decoder)
    app.router.add_static("/static/", web_dir)
    logging.basicConfig(level=logging.INFO)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
