"""A caller that leaves a stream stops the backend's reply.

T1's ``@app.middleware("http")`` stack hid a departing client from the
streaming endpoints: ``/inference/chat/stream`` then read LM Studio (or the
HyperNix runner) to the end of the reply, and the backend kept writing it
— llama.cpp's one slot busy with a reply nobody would read, every later
request queued behind it. Found benchmarking a small model through T1,
where one timed-out request held the runner for minutes.

This needs a real server: the test client never disconnects.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402
from hypernix.t1api.registry import ModelRegistry  # noqa: E402

from test_inference_endpoints import _model  # noqa: E402

PIECES = 400          # a 20 s reply at one piece per 50 ms


class FakeLMStudio(BaseHTTPRequestHandler):
    """Streams a long reply, one piece every 50 ms, and notes when its client goes."""

    gone_after: list[int] = []

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        body = json.dumps({"data": [{"id": "model-a", "state": "loaded"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for sent in range(PIECES):
            time.sleep(0.05)
            chunk = {"choices": [{"index": 0, "delta": {"content": "la "}}]}
            try:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                type(self).gone_after.append(sent)
                return
        self.wfile.write(b"data: [DONE]\n\n")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def servers(tmp_path):
    FakeLMStudio.gone_after = []
    lms = ThreadingHTTPServer(("127.0.0.1", 0), FakeLMStudio)
    threading.Thread(target=lms.serve_forever, daemon=True).start()
    km = Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)
    registry = ModelRegistry()
    registry.register(_model("model-a"))
    app = create_app(
        config=T1APIConfig(
            token_secret="test-secret-value-that-is-long-enough",
            db_path=str(tmp_path / "t1.sqlite3"),
            module_storage_dir=str(tmp_path / "modules"),
            hyperlink_files_dir=str(tmp_path / "files"),
            lmstudio_url=f"http://127.0.0.1:{lms.server_address[1]}",
            default_plan="free",
        ),
        keymaster=km,
        gatekeeper=Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False),
        registry=registry,
    )
    key = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    yield port, key
    server.should_exit = True
    thread.join(timeout=5)
    lms.shutdown()
    lms.server_close()


def test_leaving_a_stream_stops_the_backend(servers):
    port, key = servers
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/inference/chat/stream",
                 body=json.dumps({"model": "model-a", "messages": [{"role": "user", "content": "go on"}]}),
                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    resp = conn.getresponse()
    assert resp.status == 200
    lines = 0
    while lines < 10:                                   # a few pieces in, the caller gives up
        if resp.fp.readline().startswith(b"data:"):
            lines += 1
    conn.sock.shutdown(socket.SHUT_RDWR)
    conn.close()

    deadline = time.time() + 5
    while not FakeLMStudio.gone_after and time.time() < deadline:
        time.sleep(0.05)
    assert FakeLMStudio.gone_after, "LM Studio was still writing the reply 5 s after the caller left"
    assert FakeLMStudio.gone_after[0] < PIECES // 2
