"""Tests for the auto-dep installer and Windows path handling.

These don't actually shell out to pip — every pip_install call is stubbed.
The goal is to pin down the contract: torch is protected, HYPERNIX_AUTO_INSTALL=0
disables everything, and the Windows code paths in quantize/fetcher do what we
expect without needing a real Windows machine.
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest


def test_disabled_when_env_set_to_zero(monkeypatch):
    from hypernix import deps
    monkeypatch.setenv("HYPERNIX_AUTO_INSTALL", "0")
    assert deps.disabled() is True


@pytest.mark.parametrize("val", ["false", "no", "off", "FALSE"])
def test_disabled_recognizes_common_falsey(monkeypatch, val):
    from hypernix import deps
    monkeypatch.setenv("HYPERNIX_AUTO_INSTALL", val)
    assert deps.disabled() is True


def test_enabled_by_default(monkeypatch):
    from hypernix import deps
    monkeypatch.delenv("HYPERNIX_AUTO_INSTALL", raising=False)
    assert deps.disabled() is False


def test_pip_install_honors_disabled(monkeypatch, capsys):
    from hypernix import deps
    monkeypatch.setenv("HYPERNIX_AUTO_INSTALL", "0")
    ok = deps.pip_install(["numpy"])
    assert ok is False
    err = capsys.readouterr().err
    assert "skipping pip install" in err


def test_pip_install_never_touches_torch(monkeypatch, capsys):
    from hypernix import deps
    monkeypatch.setenv("HYPERNIX_AUTO_INSTALL", "1")
    calls: list[list[str]] = []

    def fake_invoke(args):
        calls.append(args)
        return 0

    monkeypatch.setattr(deps, "_pip_invoke", fake_invoke)
    ok = deps.pip_install(["torch>=2.8", "numpy>=1.26"])
    assert ok is True
    # Exactly one install call, and torch must not appear in its args.
    assert len(calls) == 1
    flat = " ".join(calls[0])
    assert "torch" not in flat
    assert "numpy>=1.26" in flat
    err = capsys.readouterr().err
    assert "protected package" in err


def test_pip_install_retries_user_on_failure(monkeypatch):
    from hypernix import deps
    monkeypatch.setenv("HYPERNIX_AUTO_INSTALL", "1")
    # Pretend we're NOT in a venv so --user retry fires.
    monkeypatch.setattr(sys, "base_prefix", sys.prefix, raising=False)
    # prefix == base_prefix now, so in_venv=False, retry is attempted.
    calls: list[list[str]] = []

    def fake_invoke(args):
        calls.append(args)
        return 1 if "--user" not in args else 0

    monkeypatch.setattr(deps, "_pip_invoke", fake_invoke)
    ok = deps.pip_install(["numpy"])
    assert ok is True
    assert len(calls) == 2
    assert "--user" in calls[1]


def test_spec_name_parses_requirements():
    from hypernix.deps import _spec_name
    assert _spec_name("tokenizers>=0.20") == "tokenizers"
    assert _spec_name("hugging-face-hub==0.24.0") == "hugging-face-hub"
    assert _spec_name("  transformers  ") == "transformers"
    assert _spec_name("my_pkg") == "my-pkg"


def test_quantize_candidate_names_include_exe_on_windows(monkeypatch):
    from hypernix import quantize
    monkeypatch.setattr(sys, "platform", "win32")
    names = quantize._candidate_binary_names()
    assert "llama-quantize.exe" in names
    assert "quantize.exe" in names


def test_quantize_candidate_names_plain_on_posix(monkeypatch):
    from hypernix import quantize
    monkeypatch.setattr(sys, "platform", "linux")
    names = quantize._candidate_binary_names()
    assert not any(n.endswith(".exe") for n in names)


def test_quantize_system_search_paths_windows(monkeypatch):
    from hypernix import quantize
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\test\AppData\Local")
    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
    paths = [str(p) for p in quantize._system_search_paths()]
    # Must not include any POSIX system dir.
    assert not any(p.startswith("/usr") or p.startswith("/opt") for p in paths)
    # Must include at least one llama.cpp-ish Windows dir.
    assert any("llama.cpp" in p for p in paths)


def test_quantize_system_search_paths_posix(monkeypatch):
    from hypernix import quantize
    monkeypatch.setattr(sys, "platform", "linux")
    paths = [str(p).replace("\\", "/") for p in quantize._system_search_paths()]
    assert any(p == "/usr/local/bin" for p in paths)
    assert any(p == "/usr/bin" for p in paths)


def test_fetcher_cached_binary_finds_exe_on_windows(tmp_path, monkeypatch):
    from hypernix import fetcher
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(fetcher, "cache_dir", lambda: tmp_path)
    (tmp_path / "llama-quantize.exe").write_bytes(b"MZ\x00\x00")
    found = fetcher.cached_binary()
    assert found is not None
    assert found.name == "llama-quantize.exe"


def test_fetcher_cached_binary_misses_on_empty_dir(tmp_path, monkeypatch):
    from hypernix import fetcher
    monkeypatch.setattr(fetcher, "cache_dir", lambda: tmp_path)
    assert fetcher.cached_binary() is None


def test_fetcher_pick_asset_for_windows(monkeypatch):
    from hypernix import fetcher
    assets = [
        {"name": "llama-b9000-bin-ubuntu-x64.zip", "browser_download_url": "u"},
        {"name": "llama-b9000-bin-macos-arm64.zip", "browser_download_url": "u"},
        {"name": "llama-b9000-bin-win-x64.zip", "browser_download_url": "u"},
        {"name": "llama-b9000-bin-win-x64-cuda.zip", "browser_download_url": "u"},
    ]
    with mock.patch.object(fetcher, "_detect_asset_tokens",
                           return_value=("win", ["x64", "x86_64", "amd64"])):
        chosen = fetcher._pick_asset(assets)
    assert chosen is not None
    assert chosen["name"] == "llama-b9000-bin-win-x64.zip"


def test_doctor_run_cross_platform_smoke(capsys):
    """doctor.run() must not crash on any supported platform."""
    from hypernix import doctor
    # Don't --fix (that would pip install for real); just the read-only report.
    rc = doctor.run(fix=False)
    captured = capsys.readouterr().out
    assert "Python" in captured
    assert "torch" in captured
    # rc may be 0 or 1 depending on whether llama-quantize is resolvable in
    # the test env; we only care that it returned an int without crashing.
    assert rc in (0, 1)
