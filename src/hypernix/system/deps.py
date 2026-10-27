"""Runtime auto-install / auto-upgrade for hypernix's optional deps.

When hypernix hits a package that's missing or too old at runtime (e.g.
the local ``tokenizers`` crate is older than the schema in the repo's
``tokenizer.json``), we shell out to ``pip`` to try to fix it instead of
crashing. This keeps the Windows / WSL / fresh-3.13-env experience
smooth: no manual ``pip install -U ...`` steps.

We deliberately **never** touch ``torch``. Users pick a torch flavour
(cu118 / cu124 / cpu / ROCm) by installing from a specific index URL, and
silently swapping that for a different wheel would quietly break GPU
support.

Set ``HYPERNIX_AUTO_INSTALL=0`` to disable all pip invocations from
hypernix, making the package strictly non-network at runtime.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Never auto-install or auto-upgrade these — user-managed.
PROTECTED: frozenset[str] = frozenset({"torch", "torchvision", "torchaudio"})

_SPEC_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


def disabled() -> bool:
    """Return True when env says not to touch pip (HYPERNIX_AUTO_INSTALL=0)."""
    v = os.environ.get("HYPERNIX_AUTO_INSTALL", "1").strip().lower()
    return v in {"0", "false", "no", "off"}


def _spec_name(spec: str) -> str:
    m = _SPEC_RE.match(spec)
    return (m.group(1) if m else spec).lower().replace("_", "-")


def current_version(pkg: str) -> str | None:
    """Return the installed version of ``pkg`` (PyPI name), or ``None``."""
    try:
        return _pkg_version(pkg)
    except PackageNotFoundError:
        return None


def _filter_specs(specs: list[str]) -> list[str]:
    out: list[str] = []
    for s in specs:
        name = _spec_name(s)
        if name in PROTECTED:
            print(
                f"[hypernix] refusing to auto-install protected package {name!r} "
                "(install it manually to pick the right CUDA / CPU wheel)",
                file=sys.stderr,
            )
            continue
        out.append(s)
    return out


def _pip_invoke(args: list[str]) -> int:
    """Run ``python -m pip <args>`` and return the exit code. Never raises."""
    cmd = [sys.executable, "-m", "pip", *args]
    print(f"[hypernix] {' '.join(cmd)}", file=sys.stderr)
    try:
        return subprocess.call(cmd)
    except OSError as exc:
        print(f"[hypernix] pip invocation failed: {exc}", file=sys.stderr)
        return 1


def pip_install(specs: list[str], *, upgrade: bool = True, quiet: bool = False) -> bool:
    """Install / upgrade ``specs`` via pip. Tries a normal install, then ``--user``.

    The ``--user`` retry handles PEP 668 "externally-managed" environments
    (system Python on modern Debian/Arch) and read-only site-packages.

    Returns True on success, False otherwise. Honors ``HYPERNIX_AUTO_INSTALL=0``.
    """
    if disabled():
        print(
            "[hypernix] HYPERNIX_AUTO_INSTALL=0, skipping pip install for: "
            + ", ".join(specs),
            file=sys.stderr,
        )
        return False
    specs = _filter_specs(specs)
    if not specs:
        return False

    base: list[str] = ["install"]
    if upgrade:
        base.append("--upgrade")
    if quiet:
        base.append("--quiet")
    # Disable pip's own interactive prompts. Helpful on Windows consoles
    # where pip occasionally asks for confirmation on --user installs.
    base.append("--disable-pip-version-check")

    if _pip_invoke([*base, *specs]) == 0:
        return True
    # Retry --user for PEP 668 / perms. If already in a venv, pip refuses
    # --user and we'd loop forever, so gate on sys.prefix vs. base_prefix.
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_venv:
        if _pip_invoke([*base, "--user", *specs]) == 0:
            return True
    return False


def reload_modules(names: list[str]) -> None:
    """Best-effort ``importlib.reload`` for already-imported modules."""
    for name in names:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        try:
            importlib.reload(mod)
        except Exception as exc:  # noqa: BLE001 — reload can throw anything
            print(f"[hypernix] warning: could not reload {name}: {exc}", file=sys.stderr)


def ensure(
    specs: list[str],
    *,
    reimport: list[str] | None = None,
    upgrade: bool = True,
    quiet: bool = False,
) -> bool:
    """Ensure ``specs`` are installed (and at the pinned versions). Reload on success.

    Example::

        ensure(["tokenizers>=0.20", "transformers>=5.10"],
               reimport=["tokenizers", "transformers"])
    """
    ok = pip_install(specs, upgrade=upgrade, quiet=quiet)
    if ok and reimport:
        reload_modules(reimport)
    return ok
