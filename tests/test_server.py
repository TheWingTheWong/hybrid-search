import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from hybrid_search.engine import SearchEngine
from hybrid_search.server import make_server


@pytest.fixture()
def base_url(tiny_engine: SearchEngine) -> Iterator[str]:
    server = make_server(tiny_engine, port=0, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


def get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(url: str, body: bytes) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_and_stats(base_url: str) -> None:
    assert get(f"{base_url}/health") == (200, {"status": "ok", "version": "0.1.0"})
    status, stats = get(f"{base_url}/stats")
    assert status == 200 and stats["chunks"] > 0


def test_get_search_with_filters(base_url: str) -> None:
    status, body = get(f"{base_url}/search?q=leader+election&k=2&filter=tags%3Draft&docs=1")
    assert status == 200
    assert [r["doc_id"] for r in body["results"]] == ["notes/raft"]
    assert body["results"][0]["bm25"]["rank"] == 1


def test_post_search(base_url: str) -> None:
    payload = {"query": "bloom filter", "k": 1, "mode": "bm25"}
    status, body = post(f"{base_url}/search", json.dumps(payload).encode())
    assert status == 200 and body["results"][0]["doc_id"] == "bloom"
    assert body["results"][0]["dense"] is None


@pytest.mark.parametrize(
    "path",
    ["/search", "/search?q=x&k=0", "/search?q=x&k=abc", "/search?q=x&mode=nope",
     "/search?q=x&filter=bad"],
)  # fmt: skip
def test_bad_requests(base_url: str, path: str) -> None:
    status, body = get(base_url + path)
    assert status == 400 and "error" in body


def test_bad_json_and_unknown_route(base_url: str) -> None:
    assert post(f"{base_url}/search", b"{nope")[0] == 400
    assert post(f"{base_url}/search", b"[1]")[0] == 400
    assert get(f"{base_url}/missing")[0] == 404
