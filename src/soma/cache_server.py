from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

logger = logging.getLogger(__name__)

StartResponse = Callable[[str, list[tuple[str, str]]], Any]


class _ThreadedWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class _QuietRequestHandler(WSGIRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


class PreviewCacheServer:
    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir.resolve()
        self._thread: threading.Thread | None = None
        self._httpd: WSGIServer | None = None
        self._address: str | None = None

    @property
    def address(self) -> str | None:
        return self._address

    def start(self) -> str:
        if self._address:
            return self._address

        app = self._build_app()
        httpd = make_server(
            "127.0.0.1",
            0,
            app,
            server_class=_ThreadedWSGIServer,
            handler_class=_QuietRequestHandler,
        )
        self._httpd = httpd
        host, port = httpd.server_address[:2]
        self._address = f"http://{host.decode() if isinstance(host, bytes) else host}:{port}"

        thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="soma-preview-cache")
        thread.start()
        self._thread = thread
        logger.info("preview cache server started at %s", self._address)
        return self._address

    def _build_app(self) -> Callable[[dict[str, str], StartResponse], Iterable[bytes]]:
        cache_dir = self._cache_dir

        def app(environ: dict[str, str], start_response: StartResponse) -> Iterable[bytes]:
            path = environ.get("PATH_INFO", "/") or "/"
            if not path.startswith("/.soma-cache/"):
                return _respond(start_response, "404 Not Found", b"not found")
            rel = path[len("/.soma-cache/") :]
            target = (cache_dir / rel).resolve()
            if cache_dir not in target.parents and target != cache_dir:
                return _respond(start_response, "403 Forbidden", b"forbidden")
            if not target.exists() or not target.is_file():
                return _respond(start_response, "404 Not Found", b"not found")
            data = target.read_bytes()
            headers = [
                ("Content-Type", "application/octet-stream"),
                ("Content-Length", str(len(data))),
                ("Cache-Control", "no-store"),
                ("Access-Control-Allow-Origin", "*"),
            ]
            start_response("200 OK", headers)
            return [data]

        return app


def _respond(start_response: StartResponse, status: str, body: bytes) -> Iterable[bytes]:
    headers = [
        ("Content-Type", "text/plain"),
        ("Content-Length", str(len(body))),
        ("Access-Control-Allow-Origin", "*"),
    ]
    start_response(status, headers)
    return [body]
