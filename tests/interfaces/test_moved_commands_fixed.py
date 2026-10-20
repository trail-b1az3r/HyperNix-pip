"""0.72.8a3: the commands moved to `hnx extra` were checked, and these were broken.

* `pipeline` answered every recording with "this is a simulated response";
* its engines cast token ids to float, called a generate() VITS has not
  got, and wrote WAVs at 22,050 Hz whatever the model spoke at;
* `websearch` returned Bing/DuckDuckGo tracking redirects with &amp; in them;
* `scavenger` needed `datasets` to size anything, so with it missing
  every dataset was 0 rows and the default --min-entries rejected all;
  --max-age crashed on datetime.UTC;
* `camo`'s reward regex never matched a number, and its loss had the
  wrong sign;
* `wiki` did not know any module in a package (c4, neuron, t1api...);
* `download` ended a gated repo in three tracebacks; `convert` crashed on
  pad_token_id -1.

Nothing here touches the network.
"""
from __future__ import annotations

import base64
import json
import wave
from types import SimpleNamespace

import pytest


class TestWebsearchLinks:
    def test_bing_redirects_are_decoded(self):
        from hypernix.interfaces.websearch import _real_url

        target = "https://huggingface.co/docs/hub/index"
        encoded = "a1" + base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
        link = f"https://www.bing.com/ck/a?!&amp;&amp;p=abc&amp;u={encoded}&amp;ntb=1"
        assert _real_url(link) == target

    def test_duckduckgo_redirects_are_decoded(self):
        from hypernix.interfaces.websearch import _real_url

        link = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa%3Fb%3D1&amp;rut=x"
        assert _real_url(link) == "https://example.org/a?b=1"

    def test_a_plain_link_is_left_alone(self):
        from hypernix.interfaces.websearch import _real_url

        assert _real_url("https://example.org/x?a=1&amp;b=2") == "https://example.org/x?a=1&b=2"


class TestScavenger:
    def test_an_unknown_size_is_not_a_rejection(self):
        from hypernix.data.scavenger import ScavengerCriteria

        ok, reason = ScavengerCriteria(min_entries=1000).matches({"id": "a/b", "num_rows": None})
        assert ok, reason
        ok, _ = ScavengerCriteria(min_entries=1000).matches({"id": "a/b", "num_rows": 10})
        assert not ok

    def test_max_age_does_not_crash(self):
        from datetime import UTC, datetime, timedelta

        from hypernix.data.scavenger import ScavengerCriteria

        old = {"id": "a/b", "num_rows": None, "last_modified": datetime.now(UTC) - timedelta(days=900)}
        ok, reason = ScavengerCriteria(min_entries=0, min_likes=0, max_age_days=365).matches(old)
        assert not ok and "age" in reason

    def test_the_viewer_sizes_a_dataset(self, monkeypatch):
        import huggingface_hub.utils as hub_utils

        from hypernix.data.scavenger import Scavenger

        class Reply:
            status_code = 200

            @staticmethod
            def json():
                return {"size": {"dataset": {"num_rows": 2141709, "num_bytes_parquet_files": 2 * 1024**3},
                                 "configs": [{"config": "default"}]}}

        monkeypatch.setattr(hub_utils, "get_session", lambda: SimpleNamespace(get=lambda *a, **k: Reply()))
        info = Scavenger()._probe_size({"id": "roneneldan/TinyStories", "size_gb": 0.0})
        assert info["num_rows"] == 2141709 and info["size_gb"] == pytest.approx(2.0)


class TestCamo:
    def test_the_score_is_read(self):
        torch = pytest.importorskip("torch")
        from hypernix.training.camouflage import _get_reward

        class Batch(dict):
            input_ids = torch.zeros(1, 3, dtype=torch.long)

            def to(self, device):
                return self

        tokenizer = SimpleNamespace(decode=lambda *a, **k: " 8/10, a clear answer")
        tokenizer.__call__ = None
        evaluator = SimpleNamespace(generate=lambda **k: torch.zeros(1, 5, dtype=torch.long))

        class Tok:
            def __call__(self, *a, **k):
                return Batch()

            def decode(self, *a, **k):
                return " 8/10, a clear answer"

        assert _get_reward(evaluator, Tok(), "p", "r", "s", "cpu") == 8.0

    def test_the_loss_has_the_right_sign(self):
        import inspect

        from hypernix.training import camouflage

        source = inspect.getsource(camouflage.run_rlhf)
        assert "rl_loss = loss * advantage" in source and "rl_loss = loss * -advantage" not in source

    def test_prompts_from_a_file(self, tmp_path):
        from hypernix.training.camouflage import read_prompts

        (tmp_path / "p.jsonl").write_text('{"prompt": "a"}\n\n{"instruction": "b"}\n', encoding="utf-8")
        assert read_prompts(str(tmp_path / "p.jsonl")) == ["a", "b"]


class TestPipeline:
    def test_it_is_not_a_simulation(self):
        import inspect

        from hypernix.interfaces import cli

        source = inspect.getsource(cli._run_pipeline)
        assert "class SimpleLLM" not in source and "ChatLLM(" in source

    def test_an_llm_that_cannot_answer_is_an_error(self):
        pytest.importorskip("torch")
        from hypernix.models.workshop import ASREngine, ASRToLLMToTTS, TTSEngine

        pipe = ASRToLLMToTTS(ASREngine(), object(), TTSEngine())
        with pytest.raises(TypeError, match="no generate"):
            pipe._llm_generate("x", 5, 0.7)

    def test_token_ids_stay_integers(self):
        torch = pytest.importorskip("torch")
        from hypernix.models.workshop import _to_model

        model = torch.nn.Linear(2, 2).to(torch.float16)
        out = _to_model({"input_ids": torch.tensor([[1, 2]]), "values": torch.zeros(2)}, model)
        assert out["input_ids"].dtype == torch.long and out["values"].dtype == torch.float16

    def test_a_wav_reads_without_torchaudio(self, tmp_path, monkeypatch):
        torch = pytest.importorskip("torch")
        import sys

        from hypernix.models.workshop import load_audio

        monkeypatch.setitem(sys.modules, "torchaudio", None)
        monkeypatch.setitem(sys.modules, "soundfile", None)
        path = tmp_path / "a.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes((torch.full((800, 2), 16384, dtype=torch.int16)).numpy().tobytes())
        audio = load_audio(path, 16000)
        assert audio.shape == (1, 1600) and float(audio.mean()) == pytest.approx(0.5, abs=1e-3)

    def test_the_cli_names_the_models_it_knows(self, monkeypatch, capsys):
        from hypernix.interfaces import cli

        assert cli._run_pipeline(["--llm", "no-such-model", "--audio", "x.wav"]) == 2
        assert "unknown model" in capsys.readouterr().err


class TestWiki:
    def test_packages_are_documented(self):
        from hypernix.interfaces import wiki_cli

        modules = wiki_cli._get_all_modules()
        assert "c4" in modules and "c4.reap" in modules and "t1api.anthropic" in modules
        doc = wiki_cli._get_module_doc("c4.reap")
        assert doc and any(f["name"] == "reap" for f in doc["functions"])


class TestDownload:
    def _chain(self, *errors):
        exc = None
        for error in errors:
            error.__cause__ = exc
            exc = error
        return exc

    def test_a_gated_repo_is_explained(self):
        from huggingface_hub.errors import GatedRepoError

        from hypernix.models.download import explain_hub_error

        exc = self._chain(GatedRepoError("Cannot access gated repo", response=SimpleNamespace(
            headers={}, status_code=401, request=None, url="u")), RuntimeError("all candidates failed"))
        message = explain_hub_error(exc, "ray0rf1re/hyper-nix.1")
        assert "gated" in message and "https://huggingface.co/ray0rf1re/hyper-nix.1" in message

    def test_a_missing_repo_is_not_called_gated(self):
        from hypernix.models.download import explain_hub_error

        exc = RuntimeError("Repository Not Found for url. If you are trying to access a private or gated repo...")
        assert "not found" in explain_hub_error(exc, "a/b")

    def test_repo_colon_quant(self):
        from hypernix.models.hflinks import parse_source, pick_gguf_files

        assert parse_source("unsloth/Qwen3-4B-GGUF:Q4_K_M").quant == "Q4_K_M"
        files = ["m-Q4_K_M.gguf", "m-Q4_K.gguf", "m-Q8_0-00001-of-00002.gguf", "m-Q8_0-00002-of-00002.gguf",
                 "mmproj-Q8_0.gguf"]
        assert pick_gguf_files(files, "Q4_K") == ["m-Q4_K.gguf"]
        assert pick_gguf_files(files, "q8_0") == ["m-Q8_0-00001-of-00002.gguf", "m-Q8_0-00002-of-00002.gguf"]
        assert pick_gguf_files(files, "Q2_K") == []


class TestConvert:
    def test_a_negative_pad_token_is_no_token(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("gguf")
        from safetensors.torch import save_file

        from hypernix.quant.convert import convert_to_gguf
        from hypernix.quant.gguf import GGUFFile

        folder = tmp_path / "m"
        folder.mkdir()
        cfg = {"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
               "intermediate_size": 16, "vocab_size": 4, "bos_token_id": 0, "eos_token_id": [1, 2],
               "pad_token_id": -1}
        (folder / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (folder / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {
            "a": 0, "b": 1, "c": 2, "d": 3}, "merges": []}, "added_tokens": []}), encoding="utf-8")
        save_file({"model.embed_tokens.weight": torch.zeros(4, 8),
                   "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8)}, str(folder / "model.safetensors"))
        out = convert_to_gguf(folder, tmp_path / "m.gguf")
        meta = GGUFFile.read(out).metadata
        assert "tokenizer.ggml.padding_token_id" not in meta
        assert meta.get("tokenizer.ggml.eos_token_id") == 1


class TestAuto:
    def test_quants_hyprslug_does_not_write_go_to_llama_quantize(self):
        pytest.importorskip("gguf")
        from hypernix.quant.auto import resolve

        assert resolve("q4_k_m") == ("hyprslug", "Q4_K_M")
        assert resolve("IQ4_XS") == ("llama-quantize", "IQ4_XS")
        assert resolve("q6h4")[0] == "hyprslug"
        with pytest.raises(Exception, match="Unknown target"):
            resolve("Q9_Z")

    def test_a_repo_that_has_the_quant_is_only_downloaded(self, tmp_path, monkeypatch):
        pytest.importorskip("gguf")
        from hypernix.quant import auto

        monkeypatch.setattr(auto, "_repo_files", lambda *a: ["m-Q4_K_M.gguf", "m-f16.gguf", "README.md"])
        fetched = []

        def fetch(repo, name, folder, revision, token):
            fetched.append(name)
            (folder / name).write_bytes(b"GGUF")
            return folder / name

        monkeypatch.setattr(auto, "_fetch_file", fetch)
        monkeypatch.setattr(auto, "_quantise", lambda *a: pytest.fail("quantised a file the repo already had"))
        result = auto.auto("owner/m-GGUF", ["Q4_K_M"], output=tmp_path, log=lambda m: None)
        assert fetched == ["m-Q4_K_M.gguf"] and result.outputs == [tmp_path / "m-Q4_K_M.gguf"]

    def test_a_missing_quant_is_made_from_the_highest_precision(self, tmp_path, monkeypatch):
        pytest.importorskip("gguf")
        from hypernix.quant import auto

        monkeypatch.setattr(auto, "_repo_files", lambda *a: ["m-Q8_0.gguf", "m-f16.gguf"])
        monkeypatch.setattr(auto, "_fetch_file", lambda repo, name, folder, rev, tok: folder / name)
        made = []
        monkeypatch.setattr(auto, "_quantise", lambda src, q, out, say: made.append((src.name, q)) or out)
        auto.auto("owner/m-GGUF", ["Q5_K_M"], output=tmp_path, log=lambda m: None)
        assert made == [("m-f16.gguf", "Q5_K_M")]

    def test_a_bad_quant_is_refused_before_downloading(self, tmp_path, monkeypatch):
        pytest.importorskip("gguf")
        from hypernix.quant import auto

        monkeypatch.setattr(auto, "_repo_files", lambda *a: ["model.safetensors"])
        monkeypatch.setattr(auto, "_fetch_file", lambda *a: pytest.fail("downloaded"))
        with pytest.raises(Exception, match="Unknown target"):
            auto.auto("owner/m", ["Q9_Z"], output=tmp_path, log=lambda m: None)

    def test_the_converter_archive_filter(self):
        from hypernix.quant.llamaconvert import _wanted

        assert _wanted("llama.cpp-b1/convert_hf_to_gguf.py") == "convert_hf_to_gguf.py"
        assert _wanted("llama.cpp-b1/gguf-py/gguf/__init__.py") == "gguf-py/gguf/__init__.py"
        assert _wanted("llama.cpp-b1/conversion/llama.py") == "conversion/llama.py"
        assert _wanted("llama.cpp-b1/ggml/src/ggml.c") is None
        assert _wanted("pkg-1/vendor/llama.cpp/conversion/base.py", "vendor/llama.cpp/") == "conversion/base.py"
        assert _wanted("pkg-1/src/thing.py", "vendor/llama.cpp/") is None


def test_c4_rebuilds_the_buffers_a_checkpoint_does_not_carry(tmp_path):
    """to_empty() left rotary inv_freq as uninitialised memory: NaN losses on some runs."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    from hypernix.c4.model import load
    from hypernix.c4.probe import build_module

    torch.manual_seed(0)
    reference = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                             num_hidden_layers=1, num_attention_heads=2))
    reference.save_pretrained(tmp_path)
    module = build_module(load(tmp_path, device="cpu"))
    for name, buf in module.named_buffers():
        assert not buf.is_meta, name
        assert torch.equal(buf, dict(reference.named_buffers())[name]), name
