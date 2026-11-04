"""0.72.8.post2: `hnx-t1 runner start` did not load a model.

The managed runner started llama-server with its output on a pipe that
nothing read until a load had failed. llama-server logs every tensor it
loads -- hundreds of kilobytes for an ordinary model -- and past the
pipe's 64 KB it blocks on its next write, mid-load. The runner then
waited out its timeout on a model that would never finish. A quiet
server (the native HyperNix one) loaded fine, which is why it hid.

These start a stand-in server that logs like a real load and check that
it now comes up, and that a failure still reports what the server said.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from hypernix.hyperlink import managed
from hypernix.quant import runtime_bridge

CHATTY = textwrap.dedent('''
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    for i in range({lines}):
        print(f"llama_model_loader: - tensor {{i:5d}}: blk.{{i % 40}}.ffn_down.weight q4_K", flush=True)
    if {fail}:
        print("error: cannot allocate 12 GiB", flush=True)
        sys.exit(1)
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
''')


def _gguf(path: Path) -> Path:
    """The smallest GGUF the runner reads: a header, a llama architecture."""
    import struct

    def string(text: str) -> bytes:
        data = text.encode()
        return struct.pack("<Q", len(data)) + data

    with path.open("wb") as fh:
        fh.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1))
        fh.write(string("general.architecture") + struct.pack("<I", 8) + string("llama"))
    return path


@pytest.fixture
def runner(tmp_path, monkeypatch):
    def use(lines: int, fail: bool = False):
        script = tmp_path / "server.py"
        script.write_text(CHATTY.format(lines=lines, fail=fail), encoding="utf-8")
        monkeypatch.setattr(runtime_bridge, "find_build", lambda need=None: SimpleNamespace(
            bin_dir=tmp_path, server=tmp_path / "llama-server", missing=lambda types: []))
        monkeypatch.setattr(runtime_bridge, "serve_argv",
                            lambda build, model, *, host, port, **k: [sys.executable, str(script), str(port)])
        return managed.ManagedRunner(port=_free_port())
    return use


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_server_that_logs_a_lot_while_loading_still_comes_up(runner, tmp_path):
    managed_runner = runner(lines=6000)          # ~400 KB: far past a pipe's 64 KB
    try:
        record = managed_runner.load(_gguf(tmp_path / "m.gguf"), model_id="m", timeout=30)
        assert record.model_id == "m"
    finally:
        managed_runner.unload()


def test_a_failed_load_still_says_what_the_server_said(runner, tmp_path):
    managed_runner = runner(lines=6000, fail=True)
    with pytest.raises(managed.ManagedError) as failed:
        managed_runner.load(_gguf(tmp_path / "m.gguf"), model_id="m", timeout=30)
    assert "cannot allocate 12 GiB" in str(failed.value)
