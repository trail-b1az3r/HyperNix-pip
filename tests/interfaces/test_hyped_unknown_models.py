"""hyped-pro and hyped run models nobody wrote a catalog entry for.

hyped-pro refused any name outside its catalog (HPC-CFG-001), its picker
listed nothing that was actually on disk, and a T1 server routing to a
model it had indexed got "this client has no local model for it".
hyped (basic) could not run a local model at all -- "this build cannot
run it directly" -- and posted to a /v1/chat/completions the T1 server
does not have. These run a real (tiny) model rather than mock one.
"""
from __future__ import annotations

import http.server
import json
import os
import threading
from pathlib import Path

import pytest

from hypernix.interfaces import hyped_pro_core as core


@pytest.fixture(autouse=True)
def models_dir(tmp_path, monkeypatch):
    folder = tmp_path / "models"
    folder.mkdir()
    monkeypatch.setenv("HYPERNIX_MODELS_DIR", str(folder))
    core._RESOLVED.clear()
    core._OVEN_CACHE.clear()
    yield folder
    core._RESOLVED.clear()
    core._OVEN_CACHE.clear()


def _brewed(folder: Path, *, config_json: bool = True) -> Path:
    """A tiny, real hyperNix0x-v2 model with a byte-level BPE tokenizer."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from hypernix.training.brewer import BrewerConfig, BrewerModel

    folder.mkdir(parents=True)
    cfg = BrewerConfig(vocab_size=300, n_layers=2, n_heads=4, n_kv_heads=2, d_model=32,
                       d_ff=40, max_seq_len=64, use_sliding_window=False, name=folder.name)
    torch.manual_seed(0)
    model = BrewerModel(cfg)
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    if config_json:
        cfg.save(folder / "config.json")
        save_file(state, str(folder / "model.safetensors"))
    else:
        save_file(state, str(folder / "model.safetensors"),
                  metadata={"config": json.dumps(cfg.to_dict())})
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["the quick brown fox " * 20], trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(folder / "tokenizer.json"))
    return folder


class TestResolvingANameOutsideTheCatalog:
    def test_a_folder_in_the_models_dir_by_name(self, models_dir):
        _brewed(models_dir / "HyperNix.3.1-mini", config_json=False)
        model = core.get_model("HyperNix.3.1-mini")
        assert model.kind == "local" and core.is_local_path(model)
        assert model.context_window == 2048     # no config.json to read it from

    def test_a_gguf_by_path(self, tmp_path):
        gguf = tmp_path / "anything" / "mine.Q4_K_M.gguf"
        gguf.parent.mkdir()
        gguf.write_bytes(b"GGUF")
        model = core.get_model(str(gguf))
        assert model.format == "gguf" and model.gguf_filename == gguf.name
        assert core.is_downloaded(model) == (True, gguf.resolve())

    def test_a_gguf_by_name_in_the_models_dir(self, models_dir):
        (models_dir / "someone_Model-GGUF").mkdir()
        (models_dir / "someone_Model-GGUF" / "model-q8_0.gguf").write_bytes(b"GGUF")
        assert core.get_model("model-q8_0").format == "gguf"

    def test_a_hugging_face_repo_id(self):
        model = core.get_model("someone/some-model")
        assert model.repo == "someone/some-model" and model.format == "safetensors"
        named = core.get_model("someone/Some-GGUF:some.Q4_K_M.gguf")
        assert named.format == "gguf" and named.gguf_filename == "some.Q4_K_M.gguf"

    def test_a_typo_is_refused_with_what_was_tried(self):
        with pytest.raises(core.HypedProError) as caught:
            core.get_model("not a model at all")
        message = caught.value.message
        assert caught.value.code == "HPC-CFG-001"
        assert "catalog" in message and "Hugging Face" in message

    def test_the_catalog_lists_what_is_on_disk(self, models_dir):
        _brewed(models_dir / "my-brew")
        (models_dir / "loose.gguf").write_bytes(b"GGUF")
        shorts = {m["short"] for m in core.catalog_json()["models"]}
        assert {"my-brew", "loose"} <= shorts
        assert "gpt-4o" in shorts                       # the catalog is still there

    def test_a_server_routed_id_maps_to_a_model_here_by_exact_name(self, models_dir):
        _brewed(models_dir / "hypernix.3-mini")
        assert core._resolve_local_model("hypernix.3-mini") is not None
        assert core._resolve_local_model("hypernix.3-min") is None    # no fuzzy matching

    def test_the_bridge_resolves(self, models_dir):
        from hypernix.interfaces.hyped_pro_bridge import dispatch

        _brewed(models_dir / "bridged")
        got = dispatch({"id": 1, "cmd": "resolve", "model": "bridged"})
        assert got["ok"] and got["data"]["short"] == "bridged"
        refused = dispatch({"id": 2, "cmd": "resolve", "model": "no such thing"})
        assert not refused["ok"] and refused["code"] == "HPC-CFG-001"


class TestRunningOne:
    def test_hyped_pro_runs_a_brewed_model_it_has_no_entry_for(self, models_dir):
        _brewed(models_dir / "unlisted", config_json=False)
        reply = core.send_local(core.get_model("unlisted"),
                                [{"role": "user", "content": "the quick"}], max_tokens=4)
        assert isinstance(reply, str)

    def test_hyped_runs_a_local_model_itself(self, models_dir):
        from hypernix.interfaces.dots import Config, Settings
        from hypernix.interfaces.hyped_basic import Session, discover_backend

        _brewed(models_dir / "local-one")
        config = Config(server="http://127.0.0.1:9", models_dir=str(models_dir))
        backend = discover_backend(config, timeout=0.2)
        assert backend.kind == "local"
        session = Session(config, backend=backend, out=open(os.devnull, "w", encoding="utf-8"))
        session.config.settings_for = lambda _text: Settings(max_tokens=4)
        reply = session.send("the quick")
        assert "could not answer" not in reply

    def test_slash_model_takes_a_path(self, models_dir, tmp_path):
        from hypernix.interfaces.dots import Config
        from hypernix.interfaces.hyped_basic import Backend, Session

        folder = _brewed(tmp_path / "elsewhere" / "picked")
        session = Session(Config(models_dir=str(models_dir)), backend=Backend("none"),
                          out=open(os.devnull, "w", encoding="utf-8"))
        session.cmd_model(str(folder))
        assert session.backend.kind == "local"
        assert Path(session.backend.model) == folder.resolve()


class _Fake(http.server.BaseHTTPRequestHandler):
    runner_port = 0
    seen: list[str] = []

    def log_message(self, *_):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self.seen.append(self.path)
        if self.path == "/runner/status":
            self._send({"loaded": True, "base_url": f"http://127.0.0.1:{self.runner_port}",
                        "model": {"model_id": "hypernix.3-mini"}})
        else:
            self._send({"status": "ok"})

    def do_POST(self):  # noqa: N802
        self.seen.append(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path == "/v1/chat/completions" and self.server.server_port == self.runner_port:
            self._send({"choices": [{"message": {"content": "from the runner"}}]})
        else:
            self.send_response(404)
            self.end_headers()


def test_hyped_talks_to_the_t1_servers_runner():
    """Not to /v1/chat/completions on the T1 server, which has none."""
    from hypernix.interfaces.dots import Config
    from hypernix.interfaces.hyped_basic import Backend, Session

    runner = http.server.HTTPServer(("127.0.0.1", 0), _Fake)
    _Fake.runner_port = runner.server_port
    t1 = http.server.HTTPServer(("127.0.0.1", 0), _Fake)
    for server in (runner, t1):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        session = Session(Config(), backend=Backend("t1", base_url=f"http://127.0.0.1:{t1.server_port}"),
                          out=open(os.devnull, "w", encoding="utf-8"))
        assert session.send("hello") == "from the runner"
    finally:
        runner.shutdown()
        t1.shutdown()
