"""Tests for hypernix.hyped_pro's Python interpreter resolution
(resolve_python_for_subprocess) -- the fix for the bridge/GUI subprocess
ending up on a different Python than the one hypernix is actually
installed into on machines with multiple interpreters (pyenv/uv/conda).
"""
from __future__ import annotations

import sys

from hypernix import hyped_pro


def test_env_override_always_wins(monkeypatch):
    monkeypatch.setenv("HYPED_PRO_PYTHON", "/some/explicit/python")
    assert hyped_pro.resolve_python_for_subprocess() == "/some/explicit/python"


def test_this_interpreter_is_used_when_it_has_hypernix(monkeypatch):
    # The test runner's own interpreter has hypernix importable (it's
    # running these tests), so this is the common, real-world case.
    monkeypatch.delenv("HYPED_PRO_PYTHON", raising=False)
    assert hyped_pro.resolve_python_for_subprocess() == sys.executable


def test_falls_back_to_preferred_python_when_this_one_lacks_hypernix(monkeypatch):
    monkeypatch.delenv("HYPED_PRO_PYTHON", raising=False)
    monkeypatch.setattr(hyped_pro.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(hyped_pro.shutil, "which", lambda name: "/usr/bin/python3.12" if name == "python3.12" else None)
    monkeypatch.setattr(hyped_pro, "_has_hypernix", lambda path: path == "/usr/bin/python3.12")
    assert hyped_pro.resolve_python_for_subprocess() == "/usr/bin/python3.12"


def test_python312_is_tried_before_python314(monkeypatch):
    monkeypatch.delenv("HYPED_PRO_PYTHON", raising=False)
    monkeypatch.setattr(hyped_pro.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(hyped_pro.shutil, "which", lambda name: f"/usr/bin/{name}")
    # Both python3.12 and python3.14 "have" hypernix -- 3.12 must win since
    # it's tried first (documented as the best-tested version for older
    # GPU wheel compatibility).
    monkeypatch.setattr(hyped_pro, "_has_hypernix", lambda path: True)
    assert hyped_pro.resolve_python_for_subprocess() == "/usr/bin/python3.12"


def test_falls_back_to_path_scan_when_neither_preferred_version_available(monkeypatch, tmp_path):
    monkeypatch.delenv("HYPED_PRO_PYTHON", raising=False)
    monkeypatch.setattr(hyped_pro.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(hyped_pro.shutil, "which", lambda name: None)  # neither 3.12 nor 3.14 on PATH

    fake_python = tmp_path / "python3.11"
    fake_python.write_text("#!/bin/sh\n")
    fake_python.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(hyped_pro, "_has_hypernix", lambda path: path == str(fake_python))

    assert hyped_pro.resolve_python_for_subprocess() == str(fake_python)


def test_last_resort_is_bare_python3_when_nothing_found(monkeypatch, tmp_path):
    monkeypatch.delenv("HYPED_PRO_PYTHON", raising=False)
    monkeypatch.setattr(hyped_pro.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(hyped_pro.shutil, "which", lambda name: None)
    monkeypatch.setenv("PATH", str(tmp_path))  # empty dir, nothing to find
    monkeypatch.setattr(hyped_pro, "_has_hypernix", lambda path: False)
    assert hyped_pro.resolve_python_for_subprocess() == "python3"


def test_has_hypernix_true_for_this_interpreter():
    assert hyped_pro._has_hypernix(sys.executable) is True


def test_has_hypernix_false_for_nonexistent_interpreter():
    assert hyped_pro._has_hypernix("/no/such/python/binary/anywhere") is False


def test_cli_main_passes_resolved_python_to_node_subprocess(monkeypatch):
    # The actual bug this whole thing fixed: the launcher's debug message
    # used to *claim* HYPED_PRO_PYTHON would default to sys.executable, but
    # never actually set it in the subprocess env. Assert it really is set now.
    monkeypatch.setenv("HYPED_PRO_PYTHON", "/explicit/test/python")
    monkeypatch.setattr(hyped_pro.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(hyped_pro.Path, "exists", lambda self: True)

    captured_env = {}

    def fake_call(cmd, env=None):
        captured_env.update(env or {})
        return 0

    monkeypatch.setattr(hyped_pro.subprocess, "call", fake_call)
    hyped_pro.cli_main([])
    assert captured_env.get("HYPED_PRO_PYTHON") == "/explicit/test/python"
