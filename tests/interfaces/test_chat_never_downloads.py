"""``hnx chat`` downloads nothing unless a Hugging Face model is named with ``-r``."""
from __future__ import annotations

import pytest

from hypernix.interfaces import cli


@pytest.fixture
def no_hub(monkeypatch):
    """Any attempt to download fails the test."""
    calls: list[dict] = []

    def refuse(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("hnx chat tried to download a model")

    import hypernix.models.neo_oven as neo_oven

    monkeypatch.setattr(neo_oven, "download_model", refuse)
    return calls


def test_no_model_is_an_error_not_a_download(no_hub, capsys):
    assert cli._run_chat(["--message", "hi"]) == 2
    err = capsys.readouterr().err
    assert "-M PATH" in err and "-r HF_ID" in err
    assert no_hub == []


def test_M_with_a_missing_path_does_not_download(no_hub, tmp_path, capsys):
    assert cli._run_chat(["-M", str(tmp_path / "nope"), "--message", "hi"]) == 2
    assert "does not exist" in capsys.readouterr().err
    assert no_hub == []


def test_M_with_an_incomplete_folder_does_not_download(no_hub, tmp_path, capsys):
    pytest.importorskip("torch")
    folder = tmp_path / "half"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    assert cli._run_chat(["-M", str(folder), "--message", "hi"]) == 1
    assert "Nothing was downloaded" in capsys.readouterr().err
    assert no_hub == []


def test_r_is_the_one_option_that_downloads(monkeypatch, capsys):
    pytest.importorskip("torch")
    import hypernix.models.neo_oven as neo_oven

    asked: list[str] = []

    def fake_download(*, repo_id, **kwargs):
        asked.append(repo_id)
        raise FileNotFoundError("stop here: the request reached the Hub code")

    monkeypatch.setattr(neo_oven, "download_model", fake_download)
    assert cli._run_chat(["-r", "some-org/some-model", "--message", "hi"]) == 1
    assert asked == ["some-org/some-model"]
    err = capsys.readouterr().err
    assert "could not get some-org/some-model" in err and "Nothing was downloaded" not in err


def test_short_and_long_spellings():
    source = cli._run_chat.__code__.co_consts
    text = " ".join(str(c) for c in source)
    for flag in ("-r", "--repo-id", "-M", "--model-dir"):
        assert flag in text
