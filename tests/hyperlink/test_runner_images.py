"""Images on the HyperNix runner, not only on LM Studio.

Two things made every image fail on the runner. llama-server takes images
only when started with the model's vision projector (``--mmproj``), and
the runner never passed one. And uploads are stored as WebP, which
llama.cpp's image loader cannot decode. Now the projector beside a model
is found and loaded, every image is re-encoded to PNG or JPEG on the way
out, and a model with no projector says so instead of failing the turn.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest

from hypernix.quant import runtime_bridge as bridge

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _webp_url(alpha: bool = False) -> str:
    image = Image.new("RGBA" if alpha else "RGB", (64, 48), (200, 30, 30, 128) if alpha else (200, 30, 30))
    out = io.BytesIO()
    image.save(out, "WEBP")
    return "data:image/webp;base64," + base64.b64encode(out.getvalue()).decode()


def _message(url: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                         {"type": "image_url", "image_url": {"url": url}}]}]


# ---------------------------------------------------------------------------
# Finding the projector
# ---------------------------------------------------------------------------


class TestFindingTheProjector:
    def test_beside_the_model(self, tmp_path):
        (tmp_path / "qwen2.5-vl-7b-q4_k_m.gguf").write_bytes(b"GGUF")
        (tmp_path / "mmproj-qwen2.5-vl-7b-f16.gguf").write_bytes(b"GGUF")
        assert bridge.find_mmproj(tmp_path / "qwen2.5-vl-7b-q4_k_m.gguf").name.startswith("mmproj")

    def test_beside_the_file_a_link_points_at(self, tmp_path):
        real = tmp_path / "lmstudio" / "pub" / "repo"
        real.mkdir(parents=True)
        (real / "model.gguf").write_bytes(b"GGUF")
        (real / "mmproj-model-f16.gguf").write_bytes(b"GGUF")
        models = tmp_path / "models"
        models.mkdir()
        (models / "model.gguf").symlink_to(real / "model.gguf")
        assert bridge.find_mmproj(models / "model.gguf") == real / "mmproj-model-f16.gguf"

    def test_the_closest_name_then_the_higher_precision(self, tmp_path):
        (tmp_path / "gemma-3-4b-q4.gguf").write_bytes(b"GGUF")
        (tmp_path / "mmproj-gemma-3-4b-f32.gguf").write_bytes(b"GGUF")
        (tmp_path / "mmproj-gemma-3-4b-f16.gguf").write_bytes(b"GGUF")
        (tmp_path / "mmproj-other-f16.gguf").write_bytes(b"GGUF")
        assert bridge.find_mmproj(tmp_path / "gemma-3-4b-q4.gguf").name == "mmproj-gemma-3-4b-f16.gguf"

    def test_a_text_model_has_none(self, tmp_path):
        (tmp_path / "llama.gguf").write_bytes(b"GGUF")
        assert bridge.find_mmproj(tmp_path / "llama.gguf") is None

    def test_serve_argv_passes_it(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_runner_build_choice import ALL, fake_build

        build = bridge.find_build(fake_build(tmp_path / "b", types=ALL))
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF")
        argv = bridge.serve_argv(build, model, mmproj=tmp_path / "mmproj-m.gguf")
        assert argv[-2:] == ["--mmproj", str(tmp_path / "mmproj-m.gguf")]


@pytest.mark.skipif(sys.platform == "win32", reason="the fake server is a shell script")
def test_the_runner_starts_llama_server_with_the_projector(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_runner_build_choice import ALL, _only_home_candidates, fake_build, gguf_with

    from hypernix.hyperlink.managed import ManagedError, ManagedRunner
    from hypernix.quant.gguf import GGMLType

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("HNX_LLAMA_BUILD", raising=False)
    monkeypatch.setattr(bridge, "candidate_build_dirs", _only_home_candidates)
    seen = tmp_path / "argv.txt"
    fake_build(home / "llama.cpp", types=ALL, server=f'echo "$@" > {seen}; exit 1')
    model = gguf_with(tmp_path / "vl.gguf", GGMLType.F32)
    (tmp_path / "mmproj-vl-f16.gguf").write_bytes(b"GGUF")
    with pytest.raises(ManagedError):
        ManagedRunner(port=18998).load(model, timeout=10)
    assert f"--mmproj {tmp_path / 'mmproj-vl-f16.gguf'}" in seen.read_text()


# ---------------------------------------------------------------------------
# What is sent
# ---------------------------------------------------------------------------


class TestWhatIsSent:
    def test_webp_is_re_encoded_for_llama_cpp(self):
        from hypernix.hyperlink.imagecodec import vision_messages

        out = vision_messages(_message(_webp_url()))
        url = out[0]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        assert Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).format == "JPEG"

    def test_transparency_is_kept_as_png(self):
        from hypernix.hyperlink.imagecodec import vision_messages

        url = vision_messages(_message(_webp_url(alpha=True)))[0]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_no_projector_means_a_note_not_an_error(self):
        from hypernix.hyperlink.imagecodec import vision_messages

        out = vision_messages(_message(_webp_url()), images_ok=False, who="llama-3-8b")
        assert isinstance(out[0]["content"], str)
        assert "what is this?" in out[0]["content"] and "cannot see images" in out[0]["content"]

    def test_an_api_caller_is_refused_instead(self):
        from hypernix.hyperlink.imagecodec import ImagesNotSupported, vision_messages

        with pytest.raises(ImagesNotSupported, match="mmproj"):
            vision_messages(_message(_webp_url()), images_ok=False, refuse=True)

    def test_text_messages_pass_untouched(self):
        from hypernix.hyperlink.imagecodec import vision_messages

        plain = [{"role": "user", "content": "hello"}]
        assert vision_messages(plain, images_ok=False) == plain


class TestHyperLinkChatOnTheRunner:
    class _Backend:
        def __init__(self, hypernix: bool):
            self.is_hypernix = hypernix
            self.model_id = "vl-model"

    class _Runner:
        def __init__(self, mmproj: str):
            from hypernix.hyperlink.managed import ManagedModel, Placement

            self.current = ManagedModel(model_id="vl-model", path="/m.gguf", port=1,
                                        placement=Placement(), started_at=0.0, mmproj=mmproj)

    def test_a_runner_model_with_a_projector_gets_the_image(self):
        from hypernix.t1api.routers.hyperlink import _vision

        out = _vision(_message(_webp_url()), self._Backend(True), self._Runner("/mmproj.gguf"))
        assert out[0]["content"][1]["type"] == "image_url"

    def test_without_one_the_turn_still_answers(self):
        from hypernix.t1api.routers.hyperlink import _vision

        out = _vision(_message(_webp_url()), self._Backend(True), self._Runner(""))
        assert "cannot see images" in out[0]["content"]

    def test_lm_studio_gets_images_as_before(self):
        from hypernix.t1api.routers.hyperlink import _vision

        out = _vision(_message(_webp_url()), self._Backend(False), None)
        assert out[0]["content"][1]["type"] == "image_url"


# ---------------------------------------------------------------------------
# /inference
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")


class TestInferenceTakesImages:
    def _setup(self, tmp_path, mmproj: str):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "t1api"))
        import test_inference_endpoints as T

        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.security.keymaster import Keymaster, KeyScope, KeyType

        class Loaded(T._Loaded):
            supports_images = bool(mmproj)

        runner = T._Runner("model-a")
        runner._model = Loaded("model-a")
        km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
        gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False)
        client = T._app_with(km, gk, tmp_path, runner=runner, lmstudio=False)
        key = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key
        return T, client, {"Authorization": f"Bearer {key}"}

    def test_a_list_content_with_an_image_reaches_the_runner_as_jpeg(self, tmp_path):
        T, client, auth = self._setup(tmp_path, "/mmproj.gguf")
        made, patch = T._clients()
        with patch:
            got = client.post("/inference/chat", json={"model": "model-a",
                                                       "messages": _message(_webp_url())},
                              headers=auth)
        assert got.status_code == 200, got.text
        sent = made[T.RUNNER_URL + "/v1"].chat.call_args.args[0] if (T.RUNNER_URL + "/v1") in made \
            else next(iter(made.values())).chat.call_args.args[0]
        assert sent[0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg")

    def test_a_runner_model_without_a_projector_is_a_clear_400(self, tmp_path):
        T, client, auth = self._setup(tmp_path, "")
        _made, patch = T._clients()
        with patch:
            got = client.post("/inference/chat", json={"model": "model-a",
                                                       "messages": _message(_webp_url())},
                              headers=auth)
        assert got.status_code == 400
        assert "mmproj" in got.json()["error"]["message"]
