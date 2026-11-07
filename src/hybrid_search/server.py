"""A small JSON HTTP API on top of the standard library's ``http.server``.

Endpoints::

    GET  /health
    GET  /stats
    GET  /search?q=...&k=5&mode=hybrid&fusion=rrf&alpha=0.5&filter=tags=raft&docs=1
    POST /search   {"query": "...", "k": 5, "filters": ["year>=2022"], ...}

The engine is read-only at query time, so one instance is shared by all
request threads.
"""

from __future__ import annotations

import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__
from .engine import SearchEngine
from .filters import FilterError

MAX_BODY_BYTES = 64 * 1024
MAX_K = 100


class ApiError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "on"}


def search_params(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate request parameters into keyword arguments for ``SearchEngine.search``."""
    query = raw.get("query", raw.get("q"))
    if not isinstance(query, str) or not query.strip():
        raise ApiError("missing query: pass ?q=... or {\"query\": ...}")
    try:
        k = int(raw.get("k", 10))
        alpha = float(raw.get("alpha", 0.5))
        rrf_k = int(raw.get("rrf_k", 60))
    except (TypeError, ValueError) as e:
        raise ApiError(f"bad numeric parameter: {e}") from e
    if not 1 <= k <= MAX_K:
        raise ApiError(f"k must be between 1 and {MAX_K}")
    filters = raw.get("filters", raw.get("filter", []))
    if isinstance(filters, str):
        filters = [filters]
    if not isinstance(filters, list) or not all(isinstance(f, str) for f in filters):
        raise ApiError("filters must be a list of strings")
    return {
        "query": query,
        "k": k,
        "mode": raw.get("mode", "hybrid"),
        "fusion": raw.get("fusion", "rrf"),
        "alpha": alpha,
        "rrf_k": rrf_k,
        "filters": filters,
        "unique_docs": _parse_bool(raw.get("docs", raw.get("unique_docs", False))),
    }


def make_handler(engine: SearchEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"hybrid-search/{__version__}"
        quiet = False

        def log_message(self, format: str, *args: Any) -> None:
            if not self.quiet:
                super().log_message(format, *args)

        def _send(self, status: HTTPStatus, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _search(self, raw: dict[str, Any]) -> None:
            params = search_params(raw)
            query = params.pop("query")
            k = params.pop("k")
            t0 = time.perf_counter()
            try:
                results = engine.search(query, k, **params)
            except (FilterError, ValueError) as e:
                raise ApiError(str(e)) from e
            self._send(
                HTTPStatus.OK,
                {
                    "query": query,
                    "took_ms": round((time.perf_counter() - t0) * 1000, 3),
                    "results": [r.to_dict() for r in results],
                },
            )

        def _dispatch(self, method: str) -> None:
            url = urlparse(self.path)
            try:
                if method == "GET" and url.path == "/health":
                    self._send(HTTPStatus.OK, {"status": "ok", "version": __version__})
                elif method == "GET" and url.path == "/stats":
                    self._send(HTTPStatus.OK, {**engine.stats, "chunks": len(engine.chunks)})
                elif url.path == "/search" and method == "GET":
                    qs = parse_qs(url.query)
                    raw: dict[str, Any] = {k: v[-1] for k, v in qs.items()}
                    raw["filters"] = qs.get("filter", [])
                    self._search(raw)
                elif url.path == "/search" and method == "POST":
                    self._search(self._read_json())
                else:
                    raise ApiError(f"no route for {method} {url.path}", HTTPStatus.NOT_FOUND)
            except ApiError as e:
                self._send(e.status, {"error": str(e)})

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ApiError("request body too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as e:
                raise ApiError(f"invalid JSON body: {e.msg}") from e
            if not isinstance(body, dict):
                raise ApiError("JSON body must be an object")
            return body

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

    return Handler


def make_server(
    engine: SearchEngine, host: str = "127.0.0.1", port: int = 8765, quiet: bool = False
) -> ThreadingHTTPServer:
    handler = make_handler(engine)
    handler.quiet = quiet
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
