"""hyped_pro: Python entry point and launcher for hyped-plus, the Node.js TUI.

Since 0.72.5.post16 this is ``hyped-plus`` only. ``hyped-pro`` is the
OpenTUI client in :mod:`hypernix.interfaces.hyped_pro_otui`; the module
keeps its name because the bridge, the GUI and the tests import it.

Launches ``src/hypernix/hyped_pro.js`` via Node.js if present, or falls back to
the Python TUI engine in ``hypernix.hyped``.

NOTE: hyped_pro.js is a *compiled build artifact*, generated from
hyped_pro.ts via `npm run build` (tsc). Edit hyped_pro.ts, not hyped_pro.js —
changes made directly to the .js will be overwritten on the next build.

Interpreter selection: hyped_pro.js has no Python runtime of its own — it
shells out to a Python bridge (hyped_pro_bridge.py) and the GUI
(hyped_pro_gui.py) for everything real. Which `python3` those subprocesses
get matters: on a machine with more than one Python (pyenv/uv/conda
alongside system python3), the one that happens to be on $PATH as `python3`
may not be the one with `hypernix` installed. Two things fix that:

  1. Whatever interpreter is running *this file right now* is, by
     definition, one that has hypernix installed — so its ``sys.executable``
     is passed to the Node process as HYPED_PRO_PYTHON, guaranteeing the
     bridge/GUI use the exact same interpreter as this launcher, not a
     separately-guessed one. This is the primary fix and always correct.
  2. If HYPED_PRO_PYTHON is already set by the person (e.g. exported in
     their shell), that explicit choice always wins over both this file's
     own interpreter and the probing below.

The preference-order probing (python3.12, then python3.14, then any other
python3.x with hypernix importable) only matters for the one case neither
of the above covers: this file itself being run by a `python3` that does
*not* have hypernix installed (e.g. invoked via `python3 -m hypernix.hyped_pro`
where `python3` resolves to a bare system interpreter). In that case
sys.executable would be exactly the wrong thing to propagate, so
resolve_python_for_subprocess() is used instead of sys.executable directly.

Debug logging: set HYPED_PRO_DEBUG=1 (or pass --debug) to print launcher
diagnostics (interpreter resolution, node/js path resolution) to stderr
before handing off to Node.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Preferred interpreters, in order, before falling back to a PATH scan.
# python3.12 first: the best-tested version for CUDA/cuDNN wheel
# compatibility on older (Pascal-era) GPUs at the time this was written.
_PREFERRED_PYTHONS = ("python3.12", "python3.14")


def _debug_enabled(argv: list[str]) -> bool:
    return bool(os.environ.get("HYPED_PRO_DEBUG")) or "--debug" in argv


def _debug(msg: str) -> None:
    print(f"[hyped-pro launcher] DEBUG: {msg}", file=sys.stderr)


def _has_hypernix(python_bin: str) -> bool:
    try:
        return subprocess.run(
            [python_bin, "-c", "import hypernix"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_python_for_subprocess(debug: bool = False) -> str:
    """Pick the Python interpreter for the bridge/GUI subprocesses to use.

    Order: HYPED_PRO_PYTHON env override > this process's own interpreter
    (if it has hypernix — the common case, since this code is itself
    running under it) > python3.12 > python3.14 > any other python3.x on
    $PATH with hypernix importable > bare "python3" as a last resort
    (will surface a clear HPC-DEPS-001 later if that's wrong, rather than
    silently guessing further).
    """
    env_override = os.environ.get("HYPED_PRO_PYTHON")
    if env_override:
        if debug:
            _debug(f"HYPED_PRO_PYTHON already set: {env_override}")
        return env_override

    if importlib.util.find_spec("hypernix") is not None:
        if debug:
            _debug(f"this interpreter already has hypernix: {sys.executable}")
        return sys.executable

    for candidate in _PREFERRED_PYTHONS:
        found = shutil.which(candidate)
        if found and _has_hypernix(found):
            if debug:
                _debug(f"resolved preferred interpreter: {candidate} -> {found}")
            return found
        elif debug:
            _debug(f"{candidate}: {'not on PATH' if not found else 'found but hypernix not importable'}")

    seen = set(_PREFERRED_PYTHONS)
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in sorted(names):
            if name in seen or not (name == "python3" or (name.startswith("python3.") and name[8:].isdigit())):
                continue
            seen.add(name)
            full = os.path.join(directory, name)
            if os.access(full, os.X_OK) and _has_hypernix(full):
                if debug:
                    _debug(f"resolved via PATH scan: {name} -> {full}")
                return full

    if debug:
        _debug("no interpreter with hypernix found anywhere — falling back to bare 'python3'")
    return "python3"


def cli_main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    debug = _debug_enabled(argv)
    # --debug is consumed here (it's a launcher flag, not a hyped_pro.ts flag)
    # so it doesn't get forwarded and rejected as an unknown argument.
    forward_argv = [a for a in argv if a != "--debug"]

    # The Node build artifact lives at the package root next to its
    # package.json/tsconfig.json, not in this category subpackage.
    js_path = Path(__file__).resolve().parent.parent / "hyped_pro.js"
    node_bin = shutil.which("node")
    resolved_python = resolve_python_for_subprocess(debug=debug)

    if debug:
        _debug(f"node binary: {node_bin or 'NOT FOUND on PATH'}")
        _debug(f"hyped_pro.js: {js_path} (exists={js_path.exists()})")
        _debug(f"HYPED_PRO_PYTHON for bridge/GUI subprocesses: {resolved_python}")

    if node_bin and js_path.exists():
        cmd = [node_bin, str(js_path)] + forward_argv
        env = os.environ.copy()
        env["HYPED_PRO_PYTHON"] = resolved_python
        try:
            return subprocess.call(cmd, env=env)
        except Exception as exc:  # noqa: BLE001
            print(f"[hyped-pro launcher] WARNING HPL-001: node launcher failed ({exc}); "
                  f"falling back to the Python TUI (hypernix.hyped).", file=sys.stderr)
    elif debug:
        reason = "node not found on PATH" if not node_bin else f"{js_path} missing (run `npm run build` in src/hypernix)"
        _debug(f"skipping Node TUI: {reason}")

    from hypernix.interfaces.hyped import cli_main as python_hyped_main
    return python_hyped_main(forward_argv)


if __name__ == "__main__":
    sys.exit(cli_main())
