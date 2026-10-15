# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serves the WebXR page and receives its controller frames, on one port.

The headset needs two things from us: a page to load and a socket to send to.
Serving both from a single port means one ``adb reverse`` rule and a same-origin
socket the page can derive from ``location``, so there is nothing to configure
in the page and nothing to keep in sync.

WebXR only starts an immersive session from a secure context. Plain HTTP is
treated as secure on ``localhost``, which is why the headset reaches this
through ``adb reverse`` rather than over the network. Pass ``ssl_context`` to
serve HTTPS instead and reach it over the LAN; the page follows
``location.protocol`` either way.

This module is transport only. It hands raw messages to a callback and knows
nothing about the schema, which lives in
:mod:`cosmos3_h_dreams._runtime.headset_controller`.
"""

from __future__ import annotations

import ssl as ssl_module
import threading
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any

DEFAULT_PORT = 8444
WS_PATH = "/ws"

_PAGE_DIR = Path(__file__).resolve().parent / "webxr"
# Served fresh every time: the headset browser holds onto assets aggressively, and
# an operator cannot easily clear its cache mid-demo.
_NO_STORE = "no-store, max-age=0, must-revalidate"
_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def page_asset(name: str) -> bytes:
    """Read one page asset from the package.

    Raises:
        FileNotFoundError: If the asset is missing, which means the package was
            built without its data files.
    """
    return (_PAGE_DIR / name).read_bytes()


class WebXRServer:
    """Hosts the WebXR page and the socket its frames arrive on.

    Args:
        on_message: Called with every WebSocket text message, on the connection
            thread. Must not raise; it is the caller's job to be tolerant.
        host: Interface to bind. Loopback by default, which is all ``adb
            reverse`` needs and keeps the socket off the network.
        port: TCP port, or ``0`` to let the OS choose and read :attr:`port`.
        ssl_context: Serve HTTPS/WSS instead of HTTP/WS, for reaching the page
            over the LAN instead of over a cable.
    """

    def __init__(
        self,
        *,
        on_message: Callable[[str | bytes], None],
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        ssl_context: ssl_module.SSLContext | None = None,
    ) -> None:
        self._on_message = on_message
        self._host = str(host)
        self._requested_port = int(port)
        self._ssl_context = ssl_context
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._port = 0
        self._connections = 0
        self._lock = threading.Lock()

    @property
    def port(self) -> int:
        """The bound port, which differs from the requested one only for ``0``."""
        return self._port

    @property
    def connections(self) -> int:
        """How many page connections have been accepted, for diagnostics."""
        with self._lock:
            return self._connections

    @property
    def url(self) -> str:
        """The address to open in the headset."""
        scheme = "https" if self._ssl_context is not None else "http"
        return f"{scheme}://localhost:{self._port}/"

    def start(self) -> None:
        """Bind and serve on a background thread."""
        from websockets.sync.server import serve

        self._server = serve(
            self._handle,
            host=self._host,
            port=self._requested_port,
            ssl=self._ssl_context,
            process_request=self._route,
            # The page sends continuously; a silent socket means it is gone, and
            # the keepalive surfaces that rather than leaving a dead connection.
            ping_interval=5,
            ping_timeout=10,
        )
        self._port = int(self._server.socket.getsockname()[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="webxr-server", daemon=True
        )
        self._thread.start()
        print(f"[headset] serving the WebXR page at {self.url}", flush=True)

    def stop(self) -> None:
        """Stop serving and drop open connections. Idempotent."""
        server, self._server = self._server, None
        if server is None:
            return
        server.shutdown()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)

    def _route(self, connection: Any, request: Any) -> Any:
        """Serve the page over the same port the socket upgrades on."""
        path = request.path.split("?", 1)[0]
        if path == WS_PATH:
            # None lets the handshake proceed, which is what makes one port
            # enough for both the page and its socket.
            return None
        route = _ROUTES.get(path)
        if route is None:
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        name, content_type = route
        try:
            body = page_asset(name).decode("utf-8")
        except OSError:
            return connection.respond(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"the {name} page asset is missing from the installed package\n",
            )
        response = connection.respond(HTTPStatus.OK, body)
        # Headers append rather than replace, and respond() has already set a
        # text/plain type. Leaving both would serve the page as text and never
        # run the script.
        del response.headers["Content-Type"]
        response.headers["Content-Type"] = content_type
        response.headers["Cache-Control"] = _NO_STORE
        return response

    def _handle(self, connection: Any) -> None:
        """Pump one page connection's messages into the callback."""
        with self._lock:
            self._connections += 1
        print("[headset] the WebXR page connected", flush=True)
        try:
            for message in connection:
                self._on_message(message)
        except Exception:  # noqa: BLE001 - a dropped page must not kill the server
            pass
        finally:
            print("[headset] the WebXR page disconnected", flush=True)
