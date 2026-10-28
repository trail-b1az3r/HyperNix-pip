"""system.torch_shim — PyTorch on a Python that stable PyTorch may not support yet (0.72.8).

A new CPython arrives months before every stable PyTorch publishes wheels
for it, and ``pip install hypernix`` used to fail outright there: pip
found no torch wheel for the interpreter and gave up on the whole
install. So from Python 3.15 torch is not a hard requirement of the
package (see the markers in pyproject.toml), and this module installs it
instead:

* stable PyTorch from PyPI when its newest release has a wheel for this
  interpreter and platform;
* otherwise the nightly build from download.pytorch.org (CPU, or the
  CUDA index ``HYPERNIX_TORCH_NIGHTLY_INDEX`` names).

It runs from ``install-t1.sh``, ``hnx doctor --fix`` and the first ``hnx``
command on such a Python that finds torch missing, and by hand::

    python -m hypernix.system.torch_shim            # install what fits
    python -m hypernix.system.torch_shim --check    # say what it would do

``HYPERNIX_NO_TORCH_SHIM=1`` turns the automatic run off, and
``HYPERNIX_AUTO_INSTALL=0`` (no pip from hypernix at all) every run. A torch that
is already installed is never replaced: whichever build someone chose
(a CUDA index, a ROCm wheel, their own) stays theirs, as
:mod:`hypernix.system.deps` already promises.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec

from . import deps

__all__ = ["FIRST_SHIMMED", "Plan", "needed", "wheel_tag", "stable_supports", "plan", "ensure", "main"]

#: The first Python on which torch is installed by this shim, not by pip's
#: dependency resolution.
FIRST_SHIMMED = (3, 15)

PYPI_TORCH = "https://pypi.org/pypi/torch/json"
NIGHTLY_CPU = "https://download.pytorch.org/whl/nightly/cpu"


def wheel_tag(version: tuple[int, int] | None = None) -> str:
    """``cp315`` for Python 3.15."""
    major, minor = version or sys.version_info[:2]
    return f"cp{major}{minor}"


def _platform_words(system: str | None = None, machine: str | None = None) -> tuple[str, ...]:
    """Substrings a wheel's platform tag has for this machine."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arch = {"amd64": "x86_64", "arm64": "arm64" if system == "darwin" else "aarch64"}.get(machine, machine)
    if system == "linux":
        return ("linux", arch)
    if system == "darwin":
        return ("macosx", "arm64" if arch in ("arm64", "aarch64") else "x86_64")
    if system == "windows":
        return ("win", "amd64" if arch == "x86_64" else arch)
    return (system, arch)


def stable_supports(tag: str | None = None, *, words: tuple[str, ...] | None = None,
                    timeout: float = 15.0, fetch=None) -> bool | None:
    """Whether stable PyTorch's newest release has a wheel for *tag* on this
    platform. ``None`` when PyPI cannot be asked."""
    tag = tag or wheel_tag()
    words = words or _platform_words()
    try:
        if fetch is not None:
            data = fetch(PYPI_TORCH)
        else:
            from hypernix.security.safeurl import urlopen as safe_urlopen

            with safe_urlopen(PYPI_TORCH, timeout=timeout) as response:
                data = json.load(response)
    except (OSError, ValueError):
        return None
    for entry in data.get("urls", []):
        name = str(entry.get("filename", ""))
        if not name.endswith(".whl"):
            continue
        parts = name[:-4].split("-")
        if len(parts) < 5 or parts[2] != tag:
            continue
        if all(word in parts[4] for word in words):
            return True
    return False


def needed(version: tuple[int, int] | None = None) -> bool:
    """True on a shimmed Python with no torch installed."""
    return (version or sys.version_info[:2]) >= FIRST_SHIMMED and find_spec("torch") is None


@dataclass(frozen=True)
class Plan:
    action: str          # "none" | "stable" | "nightly"
    reason: str
    argv: tuple[str, ...] = ()


def plan(*, version: tuple[int, int] | None = None, installed: bool | None = None,
         supports: bool | None | str = "ask", python: str | None = None) -> Plan:
    """What to do on this interpreter. *supports* is the answer of
    :func:`stable_supports` (``"ask"`` asks PyPI)."""
    version = version or sys.version_info[:2]
    python = python or sys.executable
    has_torch = find_spec("torch") is not None if installed is None else installed
    if has_torch:
        return Plan("none", "torch is installed; it is left as it is")
    if version < FIRST_SHIMMED:
        return Plan("none", f"Python {version[0]}.{version[1]} gets torch from pip's normal "
                            "dependency resolution")
    tag = wheel_tag(version)
    if supports == "ask":
        supports = stable_supports(tag)
    pip = (python, "-m", "pip", "install")
    if supports:
        return Plan("stable", f"stable PyTorch has {tag} wheels for this platform", (*pip, "torch"))
    index = os.environ.get("HYPERNIX_TORCH_NIGHTLY_INDEX", "").strip() or NIGHTLY_CPU
    why = ("PyPI could not be asked" if supports is None
           else f"stable PyTorch has no {tag} wheel for this platform yet")
    return Plan("nightly", f"{why}; installing the nightly build from {index}",
                (*pip, "--pre", "torch", "--index-url", index))


def ensure(*, dry_run: bool = False, out=None) -> int:
    """Carry out :func:`plan`. 0 when torch is (or would be) there."""
    out = out or sys.stderr
    chosen = plan()
    if chosen.action == "none":
        return 0
    print(f"hypernix: Python {sys.version_info[0]}.{sys.version_info[1]} has no torch -- "
          f"{chosen.reason}.", file=out)
    print("hypernix: running " + " ".join(chosen.argv), file=out)
    if dry_run:
        return 0
    if deps.disabled():
        print("hypernix: HYPERNIX_AUTO_INSTALL=0, so torch is not installed.", file=out)
        return 0
    result = subprocess.run(list(chosen.argv), check=False)  # noqa: S603 - argv built above
    if result.returncode != 0:
        print("hypernix: installing torch failed. Install it yourself from "
              "https://pytorch.org/get-started/locally/ (the nightly build supports new "
              "Pythons first).", file=out)
    return result.returncode


def auto() -> None:
    """The ``hnx`` hook: install torch once, quietly skipped when not needed
    or turned off with ``HYPERNIX_NO_TORCH_SHIM=1`` / ``HYPERNIX_AUTO_INSTALL=0``."""
    if os.environ.get("HYPERNIX_NO_TORCH_SHIM") or deps.disabled() or not needed():
        return
    ensure()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m hypernix.system.torch_shim",
        description="Install PyTorch on a Python too new for hypernix's normal torch requirement: "
                    "stable when it has a wheel for this Python, nightly otherwise.")
    parser.add_argument("--check", action="store_true", help="say what would be done; install nothing")
    args = parser.parse_args(argv)
    if args.check:
        chosen = plan()
        print(f"{chosen.action}: {chosen.reason}")
        if chosen.argv:
            print("  " + " ".join(chosen.argv))
        return 0
    return ensure()


if __name__ == "__main__":
    raise SystemExit(main())
