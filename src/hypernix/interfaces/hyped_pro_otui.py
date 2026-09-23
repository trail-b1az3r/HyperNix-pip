"""hyped-pro: launcher for the OpenTUI terminal client.

hyped-pro is a TypeScript program built on OpenTUI (https://opentui.com),
the library opencode's terminal UI is made with. OpenTUI's renderer is
native code that runs under Bun, so this launcher's job is to find Bun,
make sure the app's one dependency is installed, and hand over.

The previous hyped-pro, the readline TUI, is ``hyped-plus`` now and is
untouched: :mod:`hypernix.interfaces.hyped_pro` still launches it. Both
talk to the same Python bridge, :mod:`hypernix.interfaces.hyped_pro_bridge`,
so models, keys and T1 settings are shared between them.

Where the app lives: the TypeScript sources ship inside the wheel at
``hypernix/interfaces/hyped_pro_app``. Its ``node_modules`` do not — they
hold a platform-specific native library — so the first run installs them
with ``bun install``. When the installed package directory is not writable
(a system site-packages), the app is copied to
``~/.hypernix/hyped-pro/<version>`` and installed there instead.

Environment:
  HYPED_PRO_BUN      path to the bun binary, if it is not on PATH
  HYPED_PRO_PYTHON   interpreter for the bridge (see hyped_pro.py)
  HYPED_PRO_DEBUG=1  print what the launcher decided, to stderr
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: OpenTUI's own floor: it runs on Bun 1.3.0 or later.
MIN_BUN = (1, 3, 0)

#: What the Bun installer calls the binary on this platform.
BUN_EXECUTABLE = "bun.exe" if sys.platform == "win32" else "bun"

#: The app's directory inside the installed package.
APP_DIR = Path(__file__).resolve().parent / "hyped_pro_app"

#: What must exist after ``bun install`` for the app to start.
DEPENDENCY_MARKER = Path("node_modules") / "@opentui" / "core" / "package.json"

#: Files that make up the app. Everything else in the directory (tests,
#: node_modules from a developer checkout) stays behind when it is copied.
APP_FILES = ("package.json", "bun.lock", "tsconfig.json")

INSTALL_BUN_HINT = (
    "hyped-pro runs on Bun (OpenTUI's renderer needs it).\n"
    "  Install it:   curl -fsSL https://bun.sh/install | bash\n"
    "  or point HYPED_PRO_BUN at an existing bun binary.\n"
    "The previous hyped-pro needs only Node.js and is still here: run `hyped-plus`."
)


class LaunchError(RuntimeError):
    """Why hyped-pro could not start, with the exit code to use."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _debug(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[hyped-pro launcher] {message}", file=sys.stderr)


def parse_version(text: str) -> tuple[int, int, int] | None:
    """``"1.3.11"`` (or ``"bun 1.3.11-canary"``) as a comparable tuple."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(part) for part in match.groups()) if match else None  # type: ignore[return-value]


def bun_candidates(env: dict[str, str] | None = None) -> list[str]:
    """Where bun might be, most explicit first."""
    env = os.environ if env is None else env
    found: list[str] = []
    if env.get("HYPED_PRO_BUN"):
        found.append(env["HYPED_PRO_BUN"])
    on_path = shutil.which("bun", path=env.get("PATH"))
    if on_path:
        found.append(on_path)
    # The official installer puts bun here and only adds it to PATH for
    # shells started after the install. On Windows it is bun.exe.
    home = env.get("BUN_INSTALL") or str(Path(env.get("HOME") or env.get("USERPROFILE") or Path.home()) / ".bun")
    found.append(str(Path(home) / "bin" / BUN_EXECUTABLE))
    seen: set[str] = set()
    return [c for c in found if not (c in seen or seen.add(c))]


def bun_version(binary: str) -> tuple[int, int, int] | None:
    try:
        out = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_version(out.stdout) if out.returncode == 0 else None


def find_bun(env: dict[str, str] | None = None, *, debug: bool = False) -> str:
    """The first bun that exists and is new enough, or :class:`LaunchError`."""
    too_old: list[str] = []
    for candidate in bun_candidates(env):
        if not (os.path.isfile(candidate) and os.access(candidate, os.X_OK)):
            _debug(debug, f"no bun at {candidate}")
            continue
        version = bun_version(candidate)
        _debug(debug, f"bun at {candidate}: {version}")
        if version is None:
            continue
        if version >= MIN_BUN:
            return candidate
        too_old.append(f"{candidate} is {'.'.join(map(str, version))}")
    if too_old:
        need = ".".join(map(str, MIN_BUN))
        raise LaunchError(
            f"hyped-pro needs Bun {need} or later; {', '.join(too_old)}.\n"
            "  Upgrade it:   bun upgrade\n"
            "The previous hyped-pro is still here: run `hyped-plus`.",
            exit_code=127,
        )
    raise LaunchError(INSTALL_BUN_HINT, exit_code=127)


def app_version(app_dir: Path = APP_DIR) -> str:
    try:
        return str(json.loads((app_dir / "package.json").read_text())["version"])
    except (OSError, ValueError, KeyError):
        return "unknown"


_PEP440 = re.compile(
    r"^v?(\d+\.\d+\.\d+)"
    r"(?:[._-]?(postr|post|rc|pre|alpha|beta|a|b|dev)[._-]?(\d*))?"
    r"(?:-(\d+))?$"
)


def semver_of(version: str) -> str:
    """The app's spelling of a package version.

    ``package.json`` wants semver, which has no ``.post`` or ``.rc``:
    ``0.72.5.post17`` is ``0.72.5-post17`` and ``0.72.6.rc1`` (or
    ``0.72.6rc1``) is ``0.72.6-rc1``. A release's ``-N`` rebuild suffix
    is PEP 440's ``.postN``, so it becomes ``-postN`` too. Anything else
    is returned as it is rather than guessed at.
    """
    match = _PEP440.match(version.strip())
    if match is None:
        return version
    base, tag, number, rebuild = match.groups()
    if rebuild:
        return f"{base}-post{rebuild}"
    if tag is None:
        return base
    return f"{base}-{'post' if tag == 'postr' else tag}{number}"


def sync_app_version(version: str, app_dir: Path = APP_DIR) -> str:
    """Write *version* into the app's ``package.json`` and ``app.ts``.

    The release workflow calls this beside the other version strings it
    bumps; without it a release left the app announcing the last one.
    Returns what was written.
    """
    wanted = semver_of(version)
    package = app_dir / "package.json"
    package.write_text(
        re.sub(r'^(\s*"version":\s*)"[^"]*"', rf'\g<1>"{wanted}"',
               package.read_text(encoding="utf-8"), count=1, flags=re.M),
        encoding="utf-8",
    )
    app_ts = app_dir / "src" / "app.ts"
    app_ts.write_text(
        re.sub(r'^export const VERSION = "[^"]*"', f'export const VERSION = "{wanted}"',
               app_ts.read_text(encoding="utf-8"), count=1, flags=re.M),
        encoding="utf-8",
    )
    return wanted


def runtime_root(env: dict[str, str] | None = None, *, name: str = "hyped-pro") -> Path:
    env = os.environ if env is None else env
    base = env.get("HYPERNIX_HOME") or str(Path(env.get("HOME") or Path.home()) / ".hypernix")
    return Path(base) / name


def _is_ready(directory: Path) -> bool:
    return (directory / DEPENDENCY_MARKER).is_file()


def _copy_app(source: Path, target: Path) -> None:
    """Copy the app's sources, and nothing else, to ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    for name in APP_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    shutil.copytree(source / "src", target / "src", dirs_exist_ok=True)


def prepare_app(
    source: Path = APP_DIR,
    *,
    bun: str,
    env: dict[str, str] | None = None,
    debug: bool = False,
    name: str = "hyped-pro",
    fallback: str = "hyped-plus",
) -> Path:
    """Return a directory the app can run from, installing it if needed.

    The packaged directory is used when it is ready or can be written to.
    Otherwise the sources go to a per-version directory in the user's
    home, so an upgrade never runs new code against old dependencies.

    *name* and *fallback* let another OpenTUI app (tvtop-max) use the
    same installer: its own runtime directory, and the non-OpenTUI
    program to point at when the install fails.
    """
    if not (source / "src" / "index.ts").is_file():
        raise LaunchError(f"{name}'s app is missing from {source} — reinstall hypernix.")
    if _is_ready(source):
        _debug(debug, f"running from {source}")
        return source

    if os.access(source, os.W_OK):
        target = source
    else:
        target = runtime_root(env, name=name) / app_version(source)
        # Always refresh the sources: they are small, and a reinstall of
        # the same version may have changed them.
        _copy_app(source, target)
        if _is_ready(target):
            _debug(debug, f"running from {target}")
            return target

    print(f"{name}: installing OpenTUI into {target} (first run only) ...", file=sys.stderr)
    # --production: the dev dependencies are the type checker and its
    # types, which running the app does not need.
    result = subprocess.run([bun, "install", "--production"], cwd=str(target), check=False)
    if result.returncode != 0 or not _is_ready(target):
        raise LaunchError(
            f"`bun install` failed in {target} (exit {result.returncode}).\n"
            f"  {name} needs network access once, to fetch @opentui/core.\n"
            f"  `{fallback}` works without it.",
            exit_code=result.returncode or 1,
        )
    return target


def command_for(bun: str, app: Path, argv: list[str]) -> list[str]:
    # An absolute entry point: bun resolves the app's imports from the file,
    # so the working directory stays wherever the person ran hyped-pro —
    # which /noodle uses as the project root.
    return [bun, "run", str(app / "src" / "index.ts"), *argv]


def cli_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    debug = bool(os.environ.get("HYPED_PRO_DEBUG")) or "--debug" in argv
    forward = [a for a in argv if a != "--debug"]

    # Imported here: hyped_pro pulls in nothing heavy, but keeping the
    # launcher's import cheap keeps `hyped-pro --help` instant.
    from hypernix.interfaces.hyped_pro import resolve_python_for_subprocess

    try:
        bun = find_bun(debug=debug)
        app = prepare_app(bun=bun, debug=debug)
    except LaunchError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    env = os.environ.copy()
    env["HYPED_PRO_PYTHON"] = resolve_python_for_subprocess(debug=debug)
    command = command_for(bun, app, forward)
    _debug(debug, f"exec {command}")
    try:
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(cli_main())
