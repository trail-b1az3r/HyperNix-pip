"""hypernix.system.torch_shim: PyTorch on Python 3.15+ (0.72.8).

pip used to refuse ``pip install hypernix`` on a Python stable PyTorch
had no wheel for. From 3.15 torch is not a pip dependency; the shim
installs stable torch when PyTorch has a wheel for the interpreter, and
the nightly build until it does.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from hypernix.system import torch_shim as ts

ROOT = Path(__file__).resolve().parents[2]


def _pypi(*names):
    return {"urls": [{"filename": n} for n in names]}


class TestWhatStableSupports:
    LINUX = ("linux", "x86_64")

    def test_a_matching_wheel_is_support(self):
        listing = _pypi("torch-2.15.0-cp315-cp315-manylinux_2_28_x86_64.whl")
        assert ts.stable_supports("cp315", words=self.LINUX, fetch=lambda url: listing) is True

    def test_another_python_or_platform_is_not(self):
        listing = _pypi("torch-2.15.0-cp314-cp314-manylinux_2_28_x86_64.whl",
                        "torch-2.15.0-cp315-cp315-macosx_14_0_arm64.whl",
                        "torch-2.15.0.tar.gz")
        assert ts.stable_supports("cp315", words=self.LINUX, fetch=lambda url: listing) is False

    def test_pypi_unreachable_is_unknown(self):
        def down(url):
            raise OSError("offline")

        assert ts.stable_supports("cp315", words=self.LINUX, fetch=down) is None

    @pytest.mark.parametrize("system,machine,words", [
        ("Linux", "x86_64", ("linux", "x86_64")),
        ("Linux", "aarch64", ("linux", "aarch64")),
        ("Darwin", "arm64", ("macosx", "arm64")),
        ("Windows", "AMD64", ("win", "amd64")),
    ])
    def test_platform_words(self, system, machine, words):
        assert ts._platform_words(system, machine) == words


class TestThePlan:
    def test_an_installed_torch_is_never_replaced(self):
        assert ts.plan(version=(3, 15), installed=True, supports=False).action == "none"

    def test_before_3_15_pip_does_it(self):
        assert ts.plan(version=(3, 14), installed=False, supports=False).action == "none"

    def test_stable_when_pytorch_has_the_wheel(self):
        chosen = ts.plan(version=(3, 15), installed=False, supports=True, python="py")
        assert chosen.action == "stable" and chosen.argv == ("py", "-m", "pip", "install", "torch")

    def test_nightly_until_it_does(self, monkeypatch):
        monkeypatch.delenv("HYPERNIX_TORCH_NIGHTLY_INDEX", raising=False)
        chosen = ts.plan(version=(3, 15), installed=False, supports=False, python="py")
        assert chosen.action == "nightly"
        assert chosen.argv == ("py", "-m", "pip", "install", "--pre", "torch", "--index-url", ts.NIGHTLY_CPU)

    def test_nightly_when_pypi_cannot_be_asked_and_the_index_can_be_chosen(self, monkeypatch):
        monkeypatch.setenv("HYPERNIX_TORCH_NIGHTLY_INDEX", "https://download.pytorch.org/whl/nightly/cu130")
        chosen = ts.plan(version=(3, 16), installed=False, supports=None, python="py")
        assert chosen.action == "nightly" and chosen.argv[-1].endswith("/cu130")
        assert "could not be asked" in chosen.reason

    def test_the_tag(self):
        assert ts.wheel_tag((3, 15)) == "cp315"


class TestRunningIt:
    def test_ensure_runs_the_plan(self, monkeypatch, capsys):
        ran = []
        monkeypatch.setattr(ts, "plan", lambda: ts.Plan("nightly", "why", ("py", "-m", "pip", "install", "torch")))
        monkeypatch.setattr(ts.subprocess, "run", lambda argv, check: ran.append(argv) or type("R", (), {"returncode": 0})())
        assert ts.ensure() == 0 and ran == [["py", "-m", "pip", "install", "torch"]]
        assert "has no torch" in capsys.readouterr().err

    def test_a_failed_install_says_where_to_get_torch(self, monkeypatch, capsys):
        monkeypatch.setattr(ts, "plan", lambda: ts.Plan("stable", "why", ("py",)))
        monkeypatch.setattr(ts.subprocess, "run", lambda argv, check: type("R", (), {"returncode": 1})())
        assert ts.ensure() == 1 and "pytorch.org" in capsys.readouterr().err

    def test_auto_can_be_turned_off(self, monkeypatch):
        monkeypatch.setenv("HYPERNIX_NO_TORCH_SHIM", "1")
        monkeypatch.setattr(ts, "ensure", lambda **k: pytest.fail("ran"))
        ts.auto()

    def test_auto_does_nothing_where_not_needed(self, monkeypatch):
        monkeypatch.delenv("HYPERNIX_NO_TORCH_SHIM", raising=False)
        monkeypatch.setattr(ts, "needed", lambda: False)
        monkeypatch.setattr(ts, "ensure", lambda **k: pytest.fail("ran"))
        ts.auto()

    def test_check_installs_nothing(self, monkeypatch, capsys):
        monkeypatch.setattr(ts, "plan", lambda: ts.Plan("stable", "it fits", ("py", "-m", "pip", "install", "torch")))
        monkeypatch.setattr(ts.subprocess, "run", lambda *a, **k: pytest.fail("installed"))
        assert ts.main(["--check"]) == 0
        assert "stable: it fits" in capsys.readouterr().out


class TestThePackage:
    """The markers that let pip install hypernix on 3.15 at all."""

    def test_torch_and_sentencepiece_are_not_pip_dependencies_from_3_15(self):
        deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
        for name in ("torch", "sentencepiece"):
            spec = next(d for d in deps if d.startswith(name))
            assert "python_version < '3.15'" in spec, spec

    def test_setup_cfg_agrees(self):
        text = (ROOT / "setup.cfg").read_text()
        assert "torch>=1.13,<3; python_version < '3.15'" in text
        assert "sentencepiece>=0.2.1; python_version < '3.15'" in text

    def test_the_installer_runs_the_shim(self):
        assert "-m hypernix.system.torch_shim" in (ROOT / "install-t1.sh").read_text()
