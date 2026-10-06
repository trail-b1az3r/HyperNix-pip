"""``hypernix-sync``: ~/.hypernix/models mirrored into ~/.hypernix/t1api/models."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from shell_support import BASH

from hypernix.t1api import modelsync
from hypernix.t1api.modelsync import MANIFEST, serving_dir, sync
from hypernix.t1api.modelsync_cli import main as cli

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")


@pytest.fixture
def folders(tmp_path):
    source = tmp_path / "models"
    target = tmp_path / "t1api" / "models"
    source.mkdir()
    (source / "qwen3-8b.Q4_K_M.gguf").write_bytes(b"GGUF one")
    repo = source / "unsloth__gemma"
    (repo / "sub").mkdir(parents=True)
    (repo / "gemma.gguf").write_bytes(b"GGUF two")
    (repo / "sub" / "README.md").write_text("card", encoding="utf-8")
    brewed = source / "HyperNix.3-mini"
    brewed.mkdir()
    from hypernix.hyperlink.brewed import _BREWER_KEYS

    (brewed / "config.json").write_text(json.dumps({k: 1 for k in _BREWER_KEYS}), encoding="utf-8")
    (brewed / "model.safetensors").write_bytes(b"w")
    return source, target


def test_every_file_is_linked_in_real_folders(folders):
    source, target = folders
    result = sync(source, target)

    assert sorted(result.linked) == sorted([
        "qwen3-8b.Q4_K_M.gguf",
        "unsloth__gemma/gemma.gguf",
        "unsloth__gemma/sub/README.md",
        "HyperNix.3-mini/config.json",
        "HyperNix.3-mini/model.safetensors",
    ])
    # Folders are real, so rglob and the checkpoint check see inside them.
    assert (target / "unsloth__gemma").is_dir() and not (target / "unsloth__gemma").is_symlink()
    link = target / "unsloth__gemma" / "gemma.gguf"
    assert link.is_symlink() and link.read_bytes() == b"GGUF two"
    assert Path(os.readlink(link)) == source / "unsloth__gemma" / "gemma.gguf"
    assert sorted(p.name for p in target.rglob("*.gguf")) == ["gemma.gguf", "qwen3-8b.Q4_K_M.gguf"]


def test_the_catalogue_finds_the_synced_models(folders):
    from hypernix.hyperlink.brewed import brewed_dirs
    from hypernix.hyperlink.catalogue import _gguf_files

    source, target = folders
    sync(source, target)
    assert {p.name for p in _gguf_files(target)} == {"gemma.gguf", "qwen3-8b.Q4_K_M.gguf"}
    assert [p.name for p in brewed_dirs(source)] == ["HyperNix.3-mini"]
    assert [p.name for p in brewed_dirs(target)] == ["HyperNix.3-mini"]


def test_a_second_sync_changes_nothing(folders):
    source, target = folders
    sync(source, target)
    again = sync(source, target)
    assert not again.changed
    assert again.unchanged == 5


def test_new_files_are_linked_and_gone_ones_removed(folders):
    source, target = folders
    sync(source, target)
    (source / "qwen3-8b.Q4_K_M.gguf").unlink()
    (source / "new.gguf").write_bytes(b"GGUF three")
    for f in (source / "unsloth__gemma").rglob("*"):
        if f.is_file():
            f.unlink()
    (source / "unsloth__gemma" / "sub").rmdir()
    (source / "unsloth__gemma").rmdir()

    result = sync(source, target)
    assert result.linked == ["new.gguf"]
    assert sorted(result.removed) == [
        "qwen3-8b.Q4_K_M.gguf", "unsloth__gemma/gemma.gguf", "unsloth__gemma/sub/README.md",
    ]
    assert not (target / "unsloth__gemma").exists()
    assert sorted(result.folders_removed) == ["unsloth__gemma", "unsloth__gemma/sub"]


def test_your_own_files_and_links_are_never_touched(folders, tmp_path):
    source, target = folders
    target.mkdir(parents=True)
    (target / "qwen3-8b.Q4_K_M.gguf").write_bytes(b"mine")
    elsewhere = tmp_path / "elsewhere.gguf"
    elsewhere.write_bytes(b"theirs")
    (target / "HyperNix.3-mini").mkdir()
    (target / "HyperNix.3-mini" / "config.json").symlink_to(elsewhere)
    (target / "private").mkdir()   # an empty folder of yours

    result = sync(source, target)
    assert (target / "qwen3-8b.Q4_K_M.gguf").read_bytes() == b"mine"
    assert Path(os.readlink(target / "HyperNix.3-mini" / "config.json")) == elsewhere
    assert {c["path"] for c in result.conflicts} == {
        "qwen3-8b.Q4_K_M.gguf", "HyperNix.3-mini/config.json",
    }
    sync(source, target)
    assert (target / "private").is_dir()
    assert (target / "HyperNix.3-mini" / "config.json").exists()


def test_a_moved_source_is_relinked(folders, tmp_path):
    source, target = folders
    sync(source, target)
    link = target / "qwen3-8b.Q4_K_M.gguf"
    link.unlink()
    link.symlink_to(source / "old-name.gguf")      # ours, but stale
    result = sync(source, target)
    assert result.relinked == ["qwen3-8b.Q4_K_M.gguf"]
    assert link.read_bytes() == b"GGUF one"


def test_downloads_in_progress_and_hidden_files_are_skipped(folders):
    source, target = folders
    (source / "big.gguf.part").write_bytes(b"x")
    (source / "big.gguf.incomplete").write_bytes(b"x")
    (source / ".cache").mkdir()
    (source / ".cache" / "blob").write_bytes(b"x")
    result = sync(source, target)
    assert not any("big" in p or ".cache" in p for p in result.linked)


def test_dry_run_changes_nothing(folders):
    source, target = folders
    result = sync(source, target, dry_run=True)
    assert len(result.linked) == 5
    assert not target.exists()


def test_no_prune_keeps_links(folders):
    source, target = folders
    sync(source, target)
    (source / "qwen3-8b.Q4_K_M.gguf").unlink()
    result = sync(source, target, prune=False)
    assert result.removed == []
    assert (target / "qwen3-8b.Q4_K_M.gguf").is_symlink()


def test_nested_folders_are_refused(tmp_path):
    (tmp_path / "models").mkdir()
    with pytest.raises(ValueError):
        sync(tmp_path / "models", tmp_path / "models" / "t1")
    with pytest.raises(FileNotFoundError):
        sync(tmp_path / "missing", tmp_path / "t1")


def test_manifest_lists_only_folders_it_made(folders):
    source, target = folders
    sync(source, target)
    made = json.loads((target / MANIFEST).read_text(encoding="utf-8"))["folders"]
    assert sorted(made) == ["HyperNix.3-mini", "unsloth__gemma", "unsloth__gemma/sub"]


class _Config:
    def __init__(self, source, target, on=True):
        self.model_sync = on
        self.models_source = str(source)
        # With sync off the server reads the download folder, as before.
        self.hf_download_dir = str(source)
        self.models_dir = str(target)


def test_the_server_serves_the_mirror_when_sync_is_on(folders):
    source, target = folders
    got = serving_dir(_Config(source, target), force=True)
    assert got == target
    assert (target / "qwen3-8b.Q4_K_M.gguf").is_symlink()


def test_the_server_reads_the_shared_folder_when_sync_is_off(folders):
    source, target = folders
    assert serving_dir(_Config(source, target, on=False)) == source
    assert not target.exists()


def test_the_server_throttles_its_syncs(folders, monkeypatch):
    source, target = folders
    calls = []
    real = modelsync.sync
    monkeypatch.setattr(modelsync, "sync", lambda *a, **k: calls.append(1) or real(*a, **k))
    config = _Config(source, target)
    serving_dir(config, force=True)
    serving_dir(config)
    serving_dir(config)
    assert len(calls) == 1


def test_config_reads_the_settings(monkeypatch, tmp_path):
    from hypernix.t1api.config import T1APIConfig

    monkeypatch.setenv("T1_MODEL_SYNC", "1")
    monkeypatch.setenv("T1_MODELS_DIR", str(tmp_path / "m"))
    config = T1APIConfig.from_env(load_dotenv=False)
    assert config.model_sync is True
    assert config.models_dir == str(tmp_path / "m")
    monkeypatch.delenv("T1_MODEL_SYNC")
    assert T1APIConfig.from_env(load_dotenv=False).model_sync is False


def test_cli_syncs_and_reports(folders, capsys):
    source, target = folders
    assert cli(["--source", str(source), "--target", str(target)]) == 0
    out = capsys.readouterr().out
    assert "5 linked" in out
    assert cli(["--source", str(source), "--target", str(target), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["unchanged"] == 5 and data["changed"] is False


def test_cli_exit_codes(folders, tmp_path, capsys):
    source, target = folders
    assert cli(["--source", str(tmp_path / "nope"), "--target", str(target)]) == 1
    target.mkdir(parents=True)
    (target / "qwen3-8b.Q4_K_M.gguf").write_bytes(b"mine")
    assert cli(["--source", str(source), "--target", str(target)]) == 2


def test_cli_reads_the_servers_env_file(folders, tmp_path, monkeypatch, capsys):
    source, target = folders
    config = tmp_path / "cfg"
    config.mkdir()
    (config / ".env").write_text(f"T1_MODELS_SOURCE={source}\nT1_MODELS_DIR={target}\n", encoding="utf-8")
    for name in ("T1_MODELS_SOURCE", "T1_MODELS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("T1_CONFIG_DIR", str(config))
    assert cli([]) == 0
    assert (target / "qwen3-8b.Q4_K_M.gguf").is_symlink()


def test_hypernix_t1_sync_runs_it(folders, tmp_path):
    source, target = folders
    config = tmp_path / "cfg"
    config.mkdir()
    (config / ".env").write_text(f"T1_MODELS_SOURCE={source}\nT1_MODELS_DIR={target}\n", encoding="utf-8")
    result = subprocess.run(
        [BASH, str(REPO_ROOT / "bin" / "hypernix-t1"), "sync"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "HOME": str(tmp_path), "T1_CONFIG_DIR": str(config),
             "PYTHONPATH": str(REPO_ROOT / "src"), "NO_COLOR": "1",
             "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5 linked" in result.stdout
    assert (target / "HyperNix.3-mini" / "config.json").is_symlink()


def test_console_scripts_are_declared():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'hypernix-sync = "hypernix.t1api.modelsync_cli:cli_main"' in text
    assert 't1-sync = "hypernix.t1api.modelsync_cli:cli_main"' in text


def test_a_server_with_sync_on_serves_from_the_mirror(folders, monkeypatch, tmp_path):
    """Startup syncs, and the model list's local models are the links."""
    from conftest import clear_t1_config
    from fastapi.testclient import TestClient

    from hypernix.t1api.app import create_app

    source, target = folders
    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1.sqlite3"))
    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    monkeypatch.setenv("T1_LMSTUDIO_ENABLED", "0")
    monkeypatch.setenv("T1_MODEL_SYNC", "1")
    monkeypatch.setenv("T1_MODELS_SOURCE", str(source))
    monkeypatch.setenv("T1_MODELS_DIR", str(target))

    app = create_app()
    assert (target / "qwen3-8b.Q4_K_M.gguf").is_symlink()

    (source / "added-later.gguf").write_bytes(b"GGUF four")
    monkeypatch.setattr(modelsync, "_MIN_INTERVAL", 0.0)
    client = TestClient(app, client=("127.0.0.1", 5000))
    response = client.get("/hyperlink/models")
    assert response.status_code == 200, response.text
    local = [m for m in response.json()["models"] if m["source"] == "local"]
    paths = {Path(m["path"]).name: m["path"] for m in local}
    assert "added-later.gguf" in paths
    assert all(p.startswith(str(target)) for p in paths.values())


def test_status_names_the_mirror(tmp_path):
    config = tmp_path / "cfg"
    config.mkdir()
    (config / ".env").write_text(f"T1_MODEL_SYNC=1\nT1_MODELS_DIR={tmp_path / 'mirror'}\n", encoding="utf-8")
    result = subprocess.run(
        [BASH, str(REPO_ROOT / "bin" / "hypernix-t1"), "status"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "HOME": str(tmp_path), "T1_CONFIG_DIR": str(config),
             "PYTHONPATH": str(REPO_ROOT / "src"), "NO_COLOR": "1"},
    )
    assert f"models    {tmp_path / 'mirror'}" in result.stdout + result.stderr
    assert "T1_MODEL_SYNC" in result.stdout + result.stderr


def _installer_style(tmp_path):
    """What install-t1.sh writes: the download folder IS the T1 folder.

    The first version of the sync mirrored T1_HF_DOWNLOAD_DIR, so on every
    installer-made server it mirrored the T1 folder into itself -- a
    refusal, and nothing synced (reported from a real server's status).
    """
    home = tmp_path / "home"
    shared = home / ".hypernix" / "models"
    shared.mkdir(parents=True)
    (shared / "gemma.gguf").write_bytes(b"GGUF")
    config = home / ".hypernix" / "t1api"
    config.mkdir(parents=True)
    (config / ".env").write_text(
        f"T1_HF_DOWNLOAD_DIR={config}/models\n"
        f"T1_MODEL_SYNC=1\nT1_MODELS_DIR={config}/models\n"
    , encoding="utf-8")
    return home, shared, config


def test_an_installer_config_syncs_from_the_shared_folder(tmp_path, monkeypatch, capsys):
    home, shared, config = _installer_style(tmp_path)
    for name in ("T1_MODELS_SOURCE", "T1_MODELS_DIR", "T1_HF_DOWNLOAD_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("T1_CONFIG_DIR", str(config))
    from hypernix.t1api import modelsync_cli

    monkeypatch.setattr(modelsync_cli, "default_source", lambda: shared)
    assert cli([]) == 0
    assert (config / "models" / "gemma.gguf").is_symlink()


def test_the_server_with_an_installer_config_syncs_too(tmp_path, monkeypatch):
    from hypernix.t1api.config import T1APIConfig

    home, shared, config = _installer_style(tmp_path)
    monkeypatch.setenv("T1_MODEL_SYNC", "1")
    monkeypatch.setenv("T1_HF_DOWNLOAD_DIR", str(config / "models"))
    monkeypatch.setenv("T1_MODELS_DIR", str(config / "models"))
    monkeypatch.delenv("T1_MODELS_SOURCE", raising=False)
    monkeypatch.setattr(modelsync, "default_source", lambda: shared)
    cfg = T1APIConfig.from_env(load_dotenv=False)
    assert modelsync.source_for(cfg) == shared
    assert serving_dir(cfg, force=True) == config / "models"
    assert (config / "models" / "gemma.gguf").is_symlink()


def test_status_shows_the_shared_folder_as_the_source(tmp_path):
    home, shared, config = _installer_style(tmp_path)
    result = subprocess.run(
        [BASH, str(REPO_ROOT / "bin" / "hypernix-t1"), "status"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "HOME": str(home), "T1_CONFIG_DIR": str(config),
             "PYTHONPATH": str(REPO_ROOT / "src"), "NO_COLOR": "1"},
    )
    out = result.stdout + result.stderr
    assert f"mirrored from {shared} " in out
    assert "nothing is synced" not in out


def test_status_warns_when_the_folders_are_one(tmp_path):
    config = tmp_path / "cfg"
    config.mkdir()
    (config / ".env").write_text(
        f"T1_MODEL_SYNC=1\nT1_MODELS_DIR={config}/models\nT1_MODELS_SOURCE={config}/models\n"
    , encoding="utf-8")
    result = subprocess.run(
        [BASH, str(REPO_ROOT / "bin" / "hypernix-t1"), "status"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "HOME": str(tmp_path), "T1_CONFIG_DIR": str(config),
             "PYTHONPATH": str(REPO_ROOT / "src"), "NO_COLOR": "1"},
    )
    assert "nothing is synced" in result.stdout + result.stderr


def test_status_counts_the_links(tmp_path):
    home, shared, config = _installer_style(tmp_path)
    sync(shared, config / "models")
    (config / "models" / "downloaded.gguf").write_bytes(b"GGUF")   # a real file
    result = subprocess.run(
        [BASH, str(REPO_ROOT / "bin" / "hypernix-t1"), "status"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "HOME": str(home), "T1_CONFIG_DIR": str(config),
             "PYTHONPATH": str(REPO_ROOT / "src"), "NO_COLOR": "1"},
    )
    assert "1 file(s) linked, 1 of its own" in result.stdout + result.stderr
