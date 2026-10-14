"""``hnx download`` takes repo links, links to one file, -o, and no
--repo-id (0.72.8). Nothing here reaches the Hub: its calls are faked."""
from __future__ import annotations

from pathlib import Path

import pytest

from hypernix.interfaces import cli
from hypernix.models.hflinks import HFSource, LinkError, parse_source, pick_gguf


class TestParsing:
    @pytest.mark.parametrize("text, expected", [
        ("owner/repo", HFSource("owner/repo")),
        ("nano-mini", HFSource("nano-mini")),
        ("https://huggingface.co/owner/repo", HFSource("owner/repo")),
        ("https://hf.co/owner/repo/", HFSource("owner/repo")),
        ("https://huggingface.co/owner/repo/tree/dev", HFSource("owner/repo", revision="dev")),
        ("https://huggingface.co/owner/repo/blob/main/m-Q4_K_M.gguf",
         HFSource("owner/repo", "m-Q4_K_M.gguf", "main")),
        ("https://huggingface.co/owner/repo/resolve/v2/sub/m.gguf?download=true",
         HFSource("owner/repo", "sub/m.gguf", "v2")),
        ("hf://owner/repo/m.gguf", HFSource("owner/repo", "m.gguf")),
        ("owner/repo/m.gguf", HFSource("owner/repo", "m.gguf")),
    ])
    def test_forms(self, text, expected):
        assert parse_source(text) == expected

    @pytest.mark.parametrize("text", [
        "https://huggingface.co/datasets/owner/data", "https://huggingface.co/spaces/owner/app",
        "https://example.com/owner/repo", "ftp://huggingface.co/owner/repo",
        "https://huggingface.co/owner", "https://huggingface.co/owner/repo/resolve/main/../../x",
        "", "owner/re po"])
    def test_refused(self, text):
        with pytest.raises(LinkError):
            parse_source(text)

    def test_picking_a_gguf(self):
        assert pick_gguf(["README.md", "a.gguf"]) == "a.gguf"
        assert pick_gguf(["m-Q4_K_M.gguf", "m-Q8_0.gguf", "mmproj-Q4_K_M.gguf"]) == "m-Q4_K_M.gguf"
        assert pick_gguf(["m-Q5_K_M.gguf", "m-Q8_0.gguf"]) is None


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """A fake Hub with one GGUF repo and one safetensors repo."""
    import huggingface_hub

    repos = {"owner/gguf": ["README.md", "sub/m-Q4_K_M.gguf", "sub/m-Q8_0.gguf"],
             "owner/one": ["one.gguf"],
             "owner/model": ["config.json", "model.safetensors"]}
    calls: list[dict] = []

    class Api:
        def list_repo_files(self, repo_id, revision=None, token=None):
            return repos[repo_id]

    def fetch(*, repo_id, filename, revision=None, local_dir, token=None, cache_dir=None):
        calls.append({"repo_id": repo_id, "filename": filename, "revision": revision})
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    from hypernix.models import download

    def model(**kwargs):
        calls.append({"model": kwargs["repo_id"], "local_dir": kwargs["local_dir"]})
        return Path(kwargs["local_dir"] or tmp_path)

    monkeypatch.setattr(download, "download_model", model)
    monkeypatch.setenv("HYPERNIX_MODELS_DIR", str(tmp_path / "models"))
    return calls


class TestTheCommand:
    def test_a_link_to_a_file_with_minus_o(self, hub, tmp_path, capsys):
        out = tmp_path / "mine.gguf"
        link = "https://huggingface.co/owner/gguf/blob/main/sub/m-Q8_0.gguf"
        assert cli.main(["download", link, "-o", str(out)]) == 0
        assert out.read_bytes() == b"GGUF" and str(out) in capsys.readouterr().out
        assert hub[-1] == {"repo_id": "owner/gguf", "filename": "sub/m-Q8_0.gguf", "revision": "main"}
        assert not (tmp_path / "sub").exists()  # the mirrored folder is tidied away

    def test_a_gguf_repo_takes_its_q4_k_m(self, hub, tmp_path):
        assert cli.main(["download", "https://huggingface.co/owner/gguf", "-o", str(tmp_path / "d")]) == 0
        assert (tmp_path / "d" / "m-Q4_K_M.gguf").is_file()

    def test_a_repo_with_one_gguf_takes_it(self, hub, tmp_path):
        assert cli.main(["download", "owner/one", "-o", str(tmp_path / "d")]) == 0
        assert (tmp_path / "d" / "one.gguf").is_file()

    def test_minus_f_picks_one(self, hub, tmp_path):
        assert cli.main(["download", "owner/gguf", "-f", "sub/m-Q8_0.gguf", "-o", str(tmp_path)]) == 0
        assert (tmp_path / "m-Q8_0.gguf").is_file()

    def test_a_safetensors_repo_is_a_model_download_and_repo_id_is_optional(self, hub, tmp_path):
        assert cli.main(["download", "owner/model", "-o", str(tmp_path / "m")]) == 0
        assert hub[-1] == {"model": "owner/model", "local_dir": str(tmp_path / "m")}
        assert cli.main(["download", "--repo-id", "owner/model"]) == 0
        assert hub[-1]["model"] == "owner/model"

    def test_a_dataset_link_is_refused(self, hub, capsys):
        assert cli.main(["download", "https://huggingface.co/datasets/o/d"]) == 2
        assert "not a model" in capsys.readouterr().err
