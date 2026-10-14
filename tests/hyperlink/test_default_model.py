"""LFM2.5-2.6B is the runner's default (0.72.8), fetched the first time."""
from __future__ import annotations

import io

import pytest

from hypernix.hyperlink import defaultmodel


@pytest.mark.parametrize("name", ["lfm2.5-2.6b", "LFM2.5-2.7B", "LiquidAI/LFM2.5-2.6B-GGUF",
                                  "LFM2.5-2.6B-Q4_K_M.gguf", "lfm2.5"])
def test_every_spelling_names_it(name):
    assert defaultmodel.matches(name)


def test_other_models_do_not():
    assert not defaultmodel.matches("hypernix.3-mini") and not defaultmodel.matches("")


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_it_downloads_once_into_the_models_dir(tmp_path, monkeypatch):
    fetched = []

    def fake(request, timeout=None):
        fetched.append(request.full_url)
        return _Response(b"GGUF" + b"\0" * 64)

    monkeypatch.setattr(defaultmodel, "safe_urlopen", fake)
    path = defaultmodel.download(models_dir=tmp_path)
    assert path == tmp_path / "LFM2.5-2.6B" / "LFM2.5-2.6B-Q4_K_M.gguf" and path.read_bytes()[:4] == b"GGUF"
    assert fetched == ["https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/LFM2.5-2.6B-Q4_K_M.gguf"]
    defaultmodel.download(models_dir=tmp_path)
    assert len(fetched) == 1
    assert defaultmodel.describe(path)["model_id"] == "lfm2.5-2.6b"


def test_something_that_is_not_a_gguf_is_not_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(defaultmodel, "safe_urlopen", lambda r, timeout=None: _Response(b"<html>login</html>"))
    with pytest.raises(defaultmodel.DefaultModelError, match="did not download as a GGUF"):
        defaultmodel.download(models_dir=tmp_path)
    assert list((tmp_path / "LFM2.5-2.6B").iterdir()) == []


def test_the_package_default_and_the_short_names():
    import hypernix
    from hypernix.models.download import resolve_repo_id

    assert hypernix.DEFAULT_MODEL == "lfm2.5-2.6b"
    assert resolve_repo_id("lfm2.5-2.7b") == resolve_repo_id("lfm2.5-2.6b") == "LiquidAI/LFM2.5-2.6B"
