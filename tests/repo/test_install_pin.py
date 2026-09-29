"""Regression: install_requires must accept torch 1.13.

hypernix 0.46.0 added ``torch_compat`` shims that make the code run
on torch 1.13.x, but the install pin kept ``torch>=2.7`` so plain
``pip install hypernix`` still failed on old Intel Macs.  0.47.1
relaxed the pin to ``torch>=1.13,<3``.  This test guards against
the pin silently creeping back up.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _read(name: str) -> str:
    return (_ROOT / name).read_text(encoding="utf-8")


def test_pyproject_torch_pin_admits_1_13() -> None:
    text = _read("pyproject.toml")
    # Match the quoted torch spec in the dependencies array.
    m = re.search(r'"torch>=([0-9.]+),<([0-9.]+)"', text)
    assert m is not None, "torch pin missing from pyproject.toml"
    floor = tuple(int(p) for p in m.group(1).split("."))
    ceiling = tuple(int(p) for p in m.group(2).split("."))
    assert floor <= (1, 13), f"torch floor {floor} > 1.13 — regressed"
    assert ceiling >= (3,), f"torch ceiling {ceiling} dropped below 3"


def test_setup_cfg_torch_pin_admits_1_13() -> None:
    text = _read("setup.cfg")
    m = re.search(r"torch>=([0-9.]+),<([0-9.]+)", text)
    assert m is not None, "torch pin missing from setup.cfg"
    floor = tuple(int(p) for p in m.group(1).split("."))
    assert floor <= (1, 13), f"setup.cfg torch floor {floor} > 1.13"


def test_legacy_install_script_no_longer_uses_no_deps() -> None:
    """The ``--no-deps`` workaround was needed before the main pin
    was relaxed.  Now that it isn't, the installer should invoke
    pip normally — otherwise the script's behaviour silently drifts
    away from what the main install does."""
    text = _read("scripts/install_macos_legacy.sh")
    assert "hypernix[legacy-torch]" in text
    # The line that actually installs hypernix must not carry
    # --no-deps.  We allow the string to appear in a comment /
    # documentation.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if "pip install" in stripped and "hypernix" in stripped:
            assert "--no-deps" not in stripped, \
                f"stale --no-deps hack in install line: {stripped!r}"
