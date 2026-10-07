"""HyperNix.3-mini as the runner's default, and native models served at all.

The runner served GGUFs through llama.cpp and nothing else, so a model
HyperNix trained itself (a ``hyperNix0x-v2`` folder with config.json,
safetensors and a tokenizer, like ``ray0rf1re/HyperNix.3-mini``) could
not answer HyperLink. These build a tiny model of the same architecture,
since the real weights are 195 MB on the Hub, and push it through every
step: the loader, the server, the runner's subprocess, the catalogue and
the downloader.
"""
from __future__ import annotations

import io
import json
import threading
import urllib.request
from pathlib import Path

import pytest
from conftest import clear_t1_config

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from hypernix.hyperlink import brewed  # noqa: E402
from hypernix.hyperlink.brewed_server import BrewedModel, build_server, render_chat  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    clear_t1_config(monkeypatch)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    """A 2-layer hyperNix0x-v2 checkpoint, saved the way the Hub has it."""
    from safetensors.torch import save_model

    from hypernix.training.brewer import BrewerConfig, BrewerModel

    folder = tmp_path_factory.mktemp("models") / "tiny-brew"
    folder.mkdir()
    config = BrewerConfig(vocab_size=512, n_layers=2, n_heads=4, n_kv_heads=2, d_model=64,
                          max_seq_len=64, use_sliding_window=False, name="tiny-brew")
    torch.manual_seed(0)
    save_model(BrewerModel(config), str(folder / "model.safetensors"))
    config.save(folder / "config.json")
    return folder


class TestRecognisingOne:
    def test_a_brewer_folder_is_one(self, tiny):
        assert brewed.is_brewed_dir(tiny)
        assert brewed.brewed_dirs(tiny.parent) == [tiny]

    def test_a_hugging_face_folder_is_not(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 64, "model_type": "llama"}), encoding="utf-8")
        (tmp_path / "model.safetensors").write_bytes(b"x")
        assert not brewed.is_brewed_dir(tmp_path)

    def test_a_config_without_weights_is_not(self, tiny, tmp_path):
        (tmp_path / "config.json").write_text((tiny / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
        assert not brewed.is_brewed_dir(tmp_path)

    def test_its_description(self, tiny):
        facts = brewed.describe(tiny)
        assert facts["model_id"] == "tiny-brew" and facts["architecture"] == "hyperNix0x-v2"
        assert facts["context_limit"] == 64 and facts["size_bytes"] > 0

    def test_the_default_is_described_honestly(self, tmp_path, tiny):
        folder = tmp_path / brewed.DEFAULT_MODEL_DIR
        folder.mkdir()
        for name in ("config.json", "model.safetensors"):
            (folder / name).write_bytes((tiny / name).read_bytes())
        assert "does not follow instructions or call tools" in brewed.describe(folder)["detail"]


class TestLoadingSafetensors:
    def test_the_adapter_reads_model_safetensors(self, tiny):
        from hypernix.models import brewer_adapter

        model, config = brewer_adapter.load(tiny)
        assert config.n_layers == 2
        logits = model(torch.tensor([[1, 2, 3]]))["logits"]
        assert logits.shape[-1] == 512

    def test_a_mismatched_file_is_refused(self, tiny, tmp_path):
        from hypernix.models import brewer_adapter

        config = json.loads((tiny / "config.json").read_text(encoding="utf-8"))
        config["n_layers"] = 3
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (tmp_path / "model.safetensors").write_bytes((tiny / "model.safetensors").read_bytes())
        with pytest.raises(ValueError, match="does not match"):
            brewer_adapter.load(tmp_path)


class TestTheChatAsATranscript:
    def test_system_then_turns_then_the_assistants_cue(self):
        prompt = render_chat([
            {"role": "system", "content": "Be kind."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": [{"type": "text", "text": "and you?"}]},
        ])
        assert prompt == "Be kind.\n\nUser: hi\nAssistant: hello\nUser: and you?\nAssistant:"


@pytest.fixture(scope="module")
def served(tiny):
    model = BrewedModel(tiny, device="cpu")
    server = build_server(model, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=60)


class TestTheServer:
    def test_health_and_models(self, served):
        with urllib.request.urlopen(f"{served}/health") as response:
            assert json.loads(response.read()) == {"status": "ok"}
        with urllib.request.urlopen(f"{served}/v1/models") as response:
            assert json.loads(response.read())["data"][0]["id"] == "tiny-brew"

    def test_a_chat_completion(self, served):
        with _post(f"{served}/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 6,
            "tools": [{"type": "function", "function": {"name": "x"}}],  # accepted, ignored
        }) as response:
            body = json.loads(response.read())
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)

    def test_a_streamed_chat_reads_through_the_bridge(self, served):
        """The same client HyperLink uses for llama-server and LM Studio."""
        from hypernix.bridge.lmstudio import LMStudioBridge

        bridge = LMStudioBridge(served)
        chunks = list(bridge.chat_stream([{"role": "user", "content": "hi"}], max_tokens=6))
        finishes = [c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")]
        assert finishes[-1] == "stop"

    def test_a_bad_body_is_a_400(self, served):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(f"{served}/v1/chat/completions", {"messages": []})
        assert exc.value.code == 400


class TestTheRunnerServesIt:
    def test_load_starts_brewed_server_and_it_answers(self, tiny, tmp_path):
        import socket

        from hypernix.bridge.lmstudio import LMStudioBridge
        from hypernix.hyperlink.managed import ManagedRunner

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        runner = ManagedRunner(port=port)
        try:
            record = runner.load(tiny, model_id="tiny-brew", backend="cpu", timeout=120)
            assert record.context_length == 64
            assert "native HyperNix model" in record.placement.reason
            reply = LMStudioBridge(runner.base_url).chat(
                [{"role": "user", "content": "hi"}], max_tokens=4)
            assert reply["choices"][0]["message"]["role"] == "assistant"
        finally:
            runner.unload()
        assert runner.current is None


class TestTheCatalogueListsIt:
    def test_beside_the_ggufs(self, tiny):
        from hypernix.hyperlink.catalogue import local_models

        models, report = local_models(tiny.parent)
        assert [m.model_id for m in models] == ["tiny-brew"]
        assert models[0].path == str(tiny) and report.count == 1


class FakeHub:
    """urlopen for the Hub, serving files from a dict."""

    def __init__(self, files):
        self.files = files
        self.asked = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.asked.append(url)
        name = url.rsplit("/", 1)[1]
        if name not in self.files:
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return Response(self.files[name])


class TestTheDownload:
    @pytest.fixture
    def hub(self, tiny, monkeypatch):
        files = {n: (tiny / n).read_bytes() for n in ("config.json", "model.safetensors")}
        files["tokenizer.json"] = b"{}"
        fake = FakeHub(files)
        monkeypatch.setattr(brewed.urllib.request, "urlopen", fake)
        return fake

    def test_it_fetches_safetensors_and_never_the_pickle(self, hub, tmp_path):
        folder = brewed.download(models_dir=tmp_path)
        assert folder == tmp_path / "HyperNix.3-mini" and brewed.is_brewed_dir(folder)
        assert not any(url.endswith("model.pt") for url in hub.asked)
        assert all(url.startswith("https://huggingface.co/ray0rf1re/HyperNix.3-mini/resolve/main/")
                   for url in hub.asked)
        assert not list(folder.glob("*.part"))

    def test_a_second_call_fetches_nothing(self, hub, tmp_path):
        brewed.download(models_dir=tmp_path)
        before = len(hub.asked)
        brewed.download(models_dir=tmp_path)
        # Only the optional files that 404ed are asked for again.
        assert all("special_tokens_map" in u or "tokenizer_config" in u for u in hub.asked[before:])

    @pytest.mark.parametrize("repo", ["../etc", "a/b/c", "", "a b/c"])
    def test_a_repo_id_that_is_not_one(self, repo, tmp_path):
        with pytest.raises(brewed.BrewedError):
            brewed.download(repo, tmp_path / "x")

    def test_only_https_endpoints(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_ENDPOINT", "http://mirror.local")
        with pytest.raises(brewed.BrewedError, match="https"):
            brewed.download(models_dir=tmp_path)


class TestTheDefault:
    def test_it_is_hypernix_3_mini(self, monkeypatch):
        monkeypatch.delenv("T1_DEFAULT_MODEL", raising=False)
        assert brewed.default_model_id() == "hypernix.3-mini"

    def test_it_can_be_changed_or_turned_off(self, monkeypatch):
        monkeypatch.setenv("T1_DEFAULT_MODEL", "qwen3-8b")
        assert brewed.default_model_id() == "qwen3-8b"
        monkeypatch.setenv("T1_DEFAULT_MODEL", "")
        assert brewed.default_model_id() == ""

    @pytest.mark.parametrize("name", ["hypernix.3-mini", "HyperNix.3-mini", "ray0rf1re/HyperNix.3-mini"])
    def test_every_spelling_names_it(self, name):
        assert brewed.matches_default(name)

    def test_loading_it_by_name_downloads_it_first(self, tiny, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from hypernix.t1api.config import T1APIConfig
        from hypernix.t1api.routers import runner as runner_router

        fetched = []

        def fake_download(**kwargs):
            folder = tmp_path / "HyperNix.3-mini"
            folder.mkdir(exist_ok=True)
            for name in ("config.json", "model.safetensors"):
                (folder / name).write_bytes((tiny / name).read_bytes())
            fetched.append(kwargs)
            return folder

        monkeypatch.setattr(brewed, "download", fake_download)
        config = T1APIConfig(token_secret="x" * 40, hf_download_dir=str(tmp_path / "empty"))
        path, facts = runner_router._resolve("ray0rf1re/HyperNix.3-mini", config, None)
        assert Path(path) == tmp_path / "HyperNix.3-mini" and fetched
        assert facts["architecture"] == "hyperNix0x-v2"


class TestServedThroughLlamaCpp:
    """With a llama.cpp build, a brewed model is converted once and served by it."""

    @pytest.fixture
    def with_tokenizer(self, tiny, tmp_path):
        import shutil

        tokenizers = pytest.importorskip("tokenizers")
        from tokenizers import models, pre_tokenizers, trainers

        folder = tmp_path / "tok-brew"
        shutil.copytree(tiny, folder)
        tok = tokenizers.Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.train_from_iterator(["hello there " * 50], trainers.BpeTrainer(
            vocab_size=300, special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tok.save(str(folder / "tokenizer.json"))
        return folder

    @pytest.fixture
    def a_build(self, monkeypatch):
        from hypernix.quant import runtime_bridge

        monkeypatch.setattr(runtime_bridge, "find_build", lambda *a, **k: object())

    @pytest.fixture
    def no_build(self, monkeypatch):
        from hypernix.quant import runtime_bridge

        def missing(*_a, **_k):
            raise runtime_bridge.BridgeError("no llama.cpp here")

        monkeypatch.setattr(runtime_bridge, "find_build", missing)

    def test_converted_once_and_cached(self, with_tokenizer, a_build, monkeypatch, tmp_path):
        from hypernix.hyperlink import managed
        from hypernix.models import brewer_gguf

        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path / "cfg"))
        calls = []
        real = brewer_gguf.export_gguf
        monkeypatch.setattr(brewer_gguf, "export_gguf", lambda *a, **k: calls.append(1) or real(*a, **k))
        first = managed.brewed_gguf_for(with_tokenizer)
        assert first is not None and first.is_file() and first.suffix == ".gguf"
        assert first.parent == tmp_path / "cfg" / "cache" / "brewed-gguf"
        assert managed.brewed_gguf_for(with_tokenizer) == first
        assert len(calls) == 1

    def test_new_weights_are_converted_again(self, with_tokenizer, a_build, monkeypatch, tmp_path):
        import os

        from hypernix.hyperlink import managed

        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path / "cfg"))
        first = managed.brewed_gguf_for(with_tokenizer)
        weights = with_tokenizer / "model.safetensors"
        stat = weights.stat()
        os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))
        second = managed.brewed_gguf_for(with_tokenizer)
        assert second != first and second.is_file() and not first.exists()

    def test_the_pytorch_server_without_llama_cpp(self, with_tokenizer, no_build):
        from hypernix.hyperlink import managed

        assert managed.brewed_gguf_for(with_tokenizer) is None

    def test_the_pytorch_server_without_a_tokenizer(self, tiny, a_build, monkeypatch, tmp_path):
        from hypernix.hyperlink import managed

        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path / "cfg"))
        assert managed.brewed_gguf_for(tiny) is None

    def test_torch_is_a_choice(self, with_tokenizer, a_build, monkeypatch):
        from hypernix.hyperlink import managed

        monkeypatch.setenv(managed.BREWED_BACKEND_ENV, "torch")
        assert managed.brewed_gguf_for(with_tokenizer) is None

    def test_llama_is_a_requirement_that_says_why(self, tiny, no_build, monkeypatch):
        from hypernix.hyperlink import managed

        monkeypatch.setenv(managed.BREWED_BACKEND_ENV, "llama")
        with pytest.raises(managed.ManagedError, match="no llama.cpp build"):
            managed.brewed_gguf_for(tiny)
