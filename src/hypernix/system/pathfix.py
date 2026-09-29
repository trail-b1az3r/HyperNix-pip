"""pathfix — put HyperNix's console scripts on ``PATH``, automatically.

The problem
-----------
``pip install hypernix`` writes ~20 console scripts (``hypernix``, ``hnx``,
``hyped``, ``waiter``, ``eth``, ``ups``, ``hnx-map``, …) into the
interpreter's *scripts directory*. On a lot of machines that directory is
not on ``PATH``:

* ``pip install --user`` on Linux/macOS puts them in ``~/.local/bin``,
  which many distributions only add to ``PATH`` from ``~/.profile`` — a
  file that a non-login shell never reads.
* Debian/Ubuntu's ``~/.profile`` adds ``~/.local/bin`` *only if it already
  exists at login*, so the very first ``pip install --user`` of anything
  lands in a directory that stays off ``PATH`` until the next login.
* Homebrew and pyenv Pythons put scripts somewhere else again.
* On Windows, ``%APPDATA%\\Python\\PythonXY\\Scripts`` is off ``PATH``
  unless the installer's "Add Python to PATH" box was ticked.

pip itself prints a warning about this and moves on, so the common
experience is ``pip install hypernix`` followed by ``hypernix: command not
found``, which looks like a broken package rather than a ``PATH`` gap.

What this module does
---------------------
:func:`ensure_on_path` appends an idempotent, clearly-marked, reversible
block to the right shell startup file for the current shell::

    # >>> hypernix path setup >>>
    # Added by hypernix <version>. Remove with: hypernix path --undo
    export PATH="/home/user/.local/bin:$PATH"
    # <<< hypernix path setup <<<

:func:`remove_from_path` takes it back out. Both are exact-block
operations — nothing else in the file is parsed, rewritten, or reordered.

Automatic behaviour and its limits
----------------------------------
:func:`maybe_autoconfigure` is called once from the CLI entry point. It is
deliberately conservative and does nothing at all when:

* the scripts directory is already on ``PATH`` (the common case);
* we're inside a virtualenv or conda env — that environment's scripts
  directory is on ``PATH`` only while it's activated, and baking it into
  ``~/.bashrc`` would leak one environment into every shell the person
  ever opens. This is the case where "helpfully" editing a profile does
  real damage, so it's refused rather than guessed at;
* ``HYPERNIX_NO_PATH_SETUP`` is set, or a CI environment is detected;
* it has already run once (a stamp file under ``~/.hypernix``), so a
  person who deleted the block doesn't get it silently written back on
  the next command.

It always prints what it changed. A ``PATH`` edit that happens invisibly
is worse than no ``PATH`` edit: the point is that the next shell works
*and* that the person knows why.

None of this affects the running process — a shell startup file is read
by future shells. :func:`session_hint` returns the one-liner for using
the scripts right now, and callers that just need the scripts importable
should use ``python -m hypernix`` instead.
"""
from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BLOCK_END",
    "BLOCK_START",
    "PathSetupResult",
    "candidate_scripts_dirs",
    "detect_shell",
    "ensure_on_path",
    "in_isolated_env",
    "is_on_path",
    "maybe_autoconfigure",
    "path_entries",
    "profile_for_shell",
    "remove_from_path",
    "scripts_dir",
    "session_hint",
    "snippet_for_shell",
]

BLOCK_START = "# >>> hypernix path setup >>>"
BLOCK_END = "# <<< hypernix path setup <<<"

# Set any of these to skip automatic configuration entirely.
OPT_OUT_ENV_VARS = ("HYPERNIX_NO_PATH_SETUP", "HYPERNIX_NO_AUTO_PATH")

# Presence of any of these means "a machine, not a person" — never edit a
# profile there. GITHUB_ACTIONS/GITLAB_CI/etc. all set CI, but not all of
# them only set CI, so the common specific ones are listed too.
CI_ENV_VARS = ("CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "GITLAB_CI",
               "BUILDKITE", "JENKINS_URL", "TF_BUILD")

_STAMP_NAME = ".path-setup-attempted"


@dataclass(frozen=True)
class PathSetupResult:
    """Outcome of an :func:`ensure_on_path` / :func:`maybe_autoconfigure` call.

    ``changed`` is the only thing a caller needs for control flow; ``status``
    distinguishes the reasons for the sake of an honest message.
    """

    status: str
    message: str
    scripts_dir: Path | None = None
    profile: Path | None = None
    shell: str | None = None
    changed: bool = False

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.changed


# ---------------------------------------------------------------------------
# Where the scripts are, and whether PATH can see them
# ---------------------------------------------------------------------------


def path_entries(path_value: str | None = None) -> list[Path]:
    """Every directory on ``PATH``, as paths, empty entries dropped."""
    raw = os.environ.get("PATH", "") if path_value is None else path_value
    return [Path(p) for p in raw.split(os.pathsep) if p]


def _normalize(p: Path) -> str:
    """Compare paths the way the OS does: resolve symlinks, and on Windows
    (and macOS) ignore case. ``~/.local/bin`` and ``/home/u/.local/bin/``
    must not read as two different directories."""
    try:
        resolved = p.expanduser().resolve()
    except OSError:  # pragma: no cover - unreadable path component
        resolved = p.expanduser()
    return os.path.normcase(str(resolved))


def is_on_path(directory: Path, path_value: str | None = None) -> bool:
    """True if *directory* is on ``PATH``, compared by resolved location."""
    target = _normalize(directory)
    return any(_normalize(entry) == target for entry in path_entries(path_value))


def candidate_scripts_dirs() -> list[Path]:
    """Every directory HyperNix's console scripts could plausibly be in,
    most-authoritative first.

    An installed script that is actually findable is the strongest possible
    evidence, so ``shutil.which`` comes first — it reflects where *this*
    install really put things, including editable and ``pipx`` installs
    that no sysconfig scheme would name.
    """
    found: list[Path] = []

    def add(p: Path | str | None) -> None:
        if not p:
            return
        path = Path(p)
        if path.is_dir() and not any(_normalize(path) == _normalize(e) for e in found):
            found.append(path)

    for script in ("hypernix", "hnx", "waiter", "hyped"):
        located = shutil.which(script)
        if located:
            add(Path(located).parent)

    add(sysconfig.get_path("scripts"))

    # The --user scheme. get_preferred_scheme exists on 3.10+; the explicit
    # names are the fallback for anything older or unusual.
    user_schemes = []
    try:
        user_schemes.append(sysconfig.get_preferred_scheme("user"))
    except (AttributeError, KeyError, LookupError):  # pragma: no cover
        pass
    user_schemes += [f"{os.name}_user", "posix_user", "nt_user"]
    for scheme in user_schemes:
        if scheme in sysconfig.get_scheme_names():
            try:
                add(sysconfig.get_path("scripts", scheme))
            except KeyError:  # pragma: no cover - malformed scheme
                pass

    # Where a `pip install --user` on a POSIX system lands by default, even
    # when the sysconfig scheme above disagrees (Debian patches this).
    if os.name != "nt":
        add(Path.home() / ".local" / "bin")

    return found


def scripts_dir() -> Path:
    """The directory HyperNix's console scripts are installed in.

    Falls back to ``sysconfig``'s answer when nothing is on ``PATH`` yet —
    which is precisely the situation this module exists to fix, so the
    fallback is the normal path, not an error case.
    """
    for candidate in candidate_scripts_dirs():
        if (candidate / "hypernix").exists() or (candidate / "hypernix.exe").exists():
            return candidate
    candidates = candidate_scripts_dirs()
    if candidates:
        return candidates[0]
    return Path(sysconfig.get_path("scripts"))  # pragma: no cover - no scheme


def in_isolated_env() -> bool:
    """True inside a virtualenv, venv, or conda environment.

    ``sys.prefix != sys.base_prefix`` catches venv and virtualenv; conda
    sets ``CONDA_PREFIX`` and can share the prefixes, so it's checked too.
    """
    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return True
    if hasattr(sys, "real_prefix"):  # pragma: no cover - virtualenv <20
        return True
    return bool(os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV"))


# ---------------------------------------------------------------------------
# Shell detection and the snippet each one needs
# ---------------------------------------------------------------------------


def detect_shell(os_name: str | None = None) -> str:
    """One of ``bash``, ``zsh``, ``fish``, ``powershell``, ``sh``.

    ``$SHELL`` is the person's *login* shell, which is what startup files
    are keyed to — deliberately not the shell that happens to have spawned
    this process, which for a script invocation is often ``sh``.

    *os_name* exists so a test can ask "what would this return on a POSIX
    box" without monkeypatching :data:`os.name` itself. That patch is not a
    harmless one: ``pathlib.Path()`` picks ``PosixPath`` vs ``WindowsPath``
    from ``os.name`` at construction time, so faking it on Windows makes
    every subsequent ``Path(...)`` raise ``NotImplementedError`` — including
    the one pytest uses to format a failure report, which turns a single
    failing assert into an INTERNALERROR that aborts the whole session.
    """
    override = os.environ.get("HYPERNIX_PATH_SHELL")
    if override:
        return override.strip().lower()
    if (os_name if os_name is not None else os.name) == "nt":
        return "powershell"
    shell = os.environ.get("SHELL", "")
    name = Path(shell).name.lower() if shell else ""
    if name.startswith("bash"):
        return "bash"
    if name.startswith("zsh"):
        return "zsh"
    if name.startswith("fish"):
        return "fish"
    if name:
        return "sh"
    return "sh"


def profile_for_shell(shell: str | None = None) -> Path | None:
    """The startup file to append to for *shell*, or ``None`` if unknown.

    Only files a *non-login interactive* shell actually reads are chosen,
    because that is the shell a person opens a terminal into. Writing to
    ``~/.bash_profile`` on Linux is the classic version of this mistake:
    the block is correct, and never runs.
    """
    shell = shell or detect_shell()
    home = Path.home()

    if shell == "bash":
        # macOS Terminal.app starts login shells, which read .bash_profile
        # and skip .bashrc; Linux terminals do the opposite.
        if sys.platform == "darwin":
            for name in (".bash_profile", ".bash_login", ".profile"):
                if (home / name).exists():
                    return home / name
            return home / ".bash_profile"
        return home / ".bashrc"

    if shell == "zsh":
        zdotdir = os.environ.get("ZDOTDIR")
        return (Path(zdotdir) if zdotdir else home) / ".zshrc"

    if shell == "fish":
        # fish sources every *.fish in conf.d, so this gets its own file
        # instead of being appended to config.fish.
        return home / ".config" / "fish" / "conf.d" / "hypernix.fish"

    if shell == "powershell":
        # Path.home() already resolves USERPROFILE on Windows and is what
        # every other branch uses, so going through it keeps all five
        # branches honouring the same notion of "home" — including for a
        # test that redirects it.
        return home / "Documents" / "PowerShell" / "Microsoft.PowerShell_profile.ps1"

    if shell == "sh":
        return home / ".profile"

    return None


def snippet_for_shell(directory: Path, shell: str | None = None) -> str:
    """The single line that puts *directory* on ``PATH`` for *shell*."""
    shell = shell or detect_shell()
    # A POSIX shell reads `\` as an escape, so `str(directory)` on Windows
    # produces a line that is not merely ugly but wrong — and these
    # snippets are written on Windows, for Git Bash and WSL. `as_posix()`
    # is identical to `str()` everywhere else, and turns `C:\Scripts` into
    # `C:/Scripts`, which is what those shells want anyway. PowerShell is
    # the one that gets the native separator, so it keeps `str()`.
    text = str(directory) if shell == "powershell" else directory.as_posix()

    if shell == "fish":
        # fish_add_path is idempotent and handles the universal/global
        # distinction itself, so re-sourcing never duplicates the entry.
        return f'fish_add_path "{text}"'

    if shell == "powershell":
        return f'$env:PATH = "{text};$env:PATH"'

    # POSIX shells. The guard means a re-sourced profile (or a second
    # terminal inheriting an already-fixed PATH) doesn't stack duplicates.
    return (
        f'case ":$PATH:" in\n'
        f'  *":{text}:"*) ;;\n'
        f'  *) export PATH="{text}:$PATH" ;;\n'
        f'esac'
    )


def _block(directory: Path, shell: str) -> str:
    from hypernix import __version__

    comment = "#"
    return (
        f"{BLOCK_START}\n"
        f"{comment} Added by hypernix {__version__}. Remove with: hypernix path --undo\n"
        f"{snippet_for_shell(directory, shell)}\n"
        f"{BLOCK_END}\n"
    )


def _strip_block(text: str) -> tuple[str, bool]:
    """Remove the marked block from *text*. Returns (new_text, removed)."""
    start = text.find(BLOCK_START)
    if start == -1:
        return text, False
    end = text.find(BLOCK_END, start)
    if end == -1:
        # Truncated block (a half-finished edit). Cut to end of file rather
        # than leaving a dangling start marker that would break re-adding.
        return text[:start].rstrip("\n") + "\n", True
    end += len(BLOCK_END)
    while end < len(text) and text[end] == "\n":
        end += 1
    cleaned = text[:start] + text[end:]
    return cleaned, True


def session_hint(directory: Path, shell: str | None = None) -> str:
    """The command that fixes ``PATH`` for the *current* shell session."""
    shell = shell or detect_shell()
    if shell == "powershell":
        return f'$env:PATH = "{directory};$env:PATH"'
    # Same reason as snippet_for_shell: forward slashes for a POSIX shell,
    # whatever host wrote the line.
    text = directory.as_posix()
    if shell == "fish":
        return f'set -gx PATH "{text}" $PATH'
    return f'export PATH="{text}:$PATH"'


# ---------------------------------------------------------------------------
# Apply / undo
# ---------------------------------------------------------------------------


def ensure_on_path(
    *,
    apply: bool = True,
    directory: Path | None = None,
    shell: str | None = None,
    profile: Path | None = None,
    force: bool = False,
) -> PathSetupResult:
    """Make sure the scripts directory is on ``PATH`` for future shells.

    With ``apply=False`` nothing is written and the result describes what
    *would* happen — that's what ``hypernix path --check`` reports and what
    ``hypernix doctor`` uses.

    ``force`` writes the block even when the directory already looks like
    it's on ``PATH``, for the case where ``PATH`` was set by something
    transient (a one-off ``export``) that won't survive the next login.
    """
    directory = Path(directory) if directory else scripts_dir()
    shell = shell or detect_shell()
    target = Path(profile) if profile else profile_for_shell(shell)

    if not force and is_on_path(directory):
        return PathSetupResult(
            "already-on-path",
            f"{directory} is already on PATH — nothing to do.",
            scripts_dir=directory, shell=shell, profile=target,  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
        )

    if target is None:
        return PathSetupResult(
            "no-profile",
            f"Don't know which startup file {shell!r} reads. Add this yourself:\n"
            f"  {snippet_for_shell(directory, shell)}",
            scripts_dir=directory, shell=shell,  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
        )

    existing = ""
    if target.exists():
        try:
            existing = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            # Same dead end as an unwritable profile: the only recourse is
            # doing it by hand, so say how.
            return PathSetupResult(
                "unreadable",
                f"Could not read {target}: {exc}\nAdd this yourself:\n"
                f"  {snippet_for_shell(directory, shell)}",
                scripts_dir=directory, shell=shell, profile=target,  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
            )

    if BLOCK_START in existing:
        # Already configured. If it points somewhere else (a different
        # Python), replace it rather than adding a second block.
        #
        # Compared against the snippet rather than str(directory): the
        # line in the file is spelled the way that shell needs it, which
        # on Windows is not how this platform writes a path. Comparing
        # the raw string there finds no match, and the block is rewritten
        # on every single run instead of being recognised as already ours.
        if snippet_for_shell(directory, shell) in existing:
            return PathSetupResult(
                "already-configured",
                f"{target} already has the hypernix PATH block for {directory}.\n"
                f"Open a new shell, or run: {session_hint(directory, shell)}",
                scripts_dir=directory, shell=shell, profile=target,  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
            )
        existing, _ = _strip_block(existing)

    if not apply:
        return PathSetupResult(
            "would-write",
            f"Would add {directory} to PATH via {target}.",
            scripts_dir=directory, shell=shell, profile=target,  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
        )

    body = existing
    if body and not body.endswith("\n"):
        body += "\n"
    body += "\n" + _block(directory, shell)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    except OSError as exc:
        return PathSetupResult(
            "unwritable",
            f"Could not write {target}: {exc}\nAdd this yourself:\n"
            f"  {snippet_for_shell(directory, shell)}",
            scripts_dir=directory, shell=shell, profile=target,  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
        )

    return PathSetupResult(
        "written",
        f"Added {directory} to PATH in {target}.\n"
        f"New shells pick it up automatically. For this one:\n"
        f"  {session_hint(directory, shell)}",
        scripts_dir=directory, shell=shell, profile=target, changed=True,  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
    )


def remove_from_path(
    *, shell: str | None = None, profile: Path | None = None
) -> PathSetupResult:
    """Take the marked block back out of the shell startup file."""
    shell = shell or detect_shell()
    target = Path(profile) if profile else profile_for_shell(shell)
    if target is None or not target.exists():
        return PathSetupResult(
            "not-configured",
            f"No hypernix PATH block to remove ({target or 'no known profile'}).",
            shell=shell, profile=target,  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
        )

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return PathSetupResult(
            "unreadable", f"Could not read {target}: {exc}", shell=shell, profile=target  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
        )

    cleaned, removed = _strip_block(text)
    if not removed:
        return PathSetupResult(
            "not-configured", f"No hypernix PATH block found in {target}.",
            shell=shell, profile=target,  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
        )

    try:
        target.write_text(cleaned, encoding="utf-8")
    except OSError as exc:
        return PathSetupResult(
            "unwritable", f"Could not write {target}: {exc}", shell=shell, profile=target  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
        )

    return PathSetupResult(
        "removed",
        f"Removed the hypernix PATH block from {target}.\n"
        "Existing shells keep the old PATH until they're restarted.",
        shell=shell, profile=target, changed=True,  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
    )


# ---------------------------------------------------------------------------
# The automatic path
# ---------------------------------------------------------------------------


def _stamp_file() -> Path:
    return Path.home() / ".hypernix" / _STAMP_NAME


def _autoconfig_blocked() -> str | None:
    """Why automatic setup must not run, or ``None`` if it may."""
    for var in OPT_OUT_ENV_VARS:
        if os.environ.get(var):
            return f"{var} is set"
    for var in CI_ENV_VARS:
        if os.environ.get(var):
            return f"{var} is set (CI)"
    if in_isolated_env():
        return "running inside a virtualenv/conda env"
    if _stamp_file().exists():
        return "already attempted once"
    return None


def maybe_autoconfigure(*, echo=print) -> PathSetupResult:
    """Called once from the CLI entry point. Quiet unless it changes something.

    Returns a result in every case so callers (and tests) can see the
    reason, but only prints when there is genuinely something to say.
    """
    directory = scripts_dir()

    if is_on_path(directory):
        return PathSetupResult(
            "already-on-path", "scripts directory is already on PATH",
            scripts_dir=directory,
        )

    blocked = _autoconfig_blocked()
    if blocked:
        return PathSetupResult(
            "skipped", f"automatic PATH setup skipped: {blocked}", scripts_dir=directory
        )

    # Stamp before acting, not after: if the write below fails or the
    # process is killed mid-way, the next run must not retry forever.
    try:
        stamp = _stamp_file()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text("hypernix pathfix ran once; delete this file to let it try again\n",
                         encoding="utf-8")
    except OSError:
        pass  # A read-only home is a reason to skip the stamp, not the fix.

    result = ensure_on_path(apply=True, directory=directory)
    if result.status in ("written", "no-profile", "unwritable"):
        echo("")
        echo(f"[hypernix] {result.message}")
        echo(f"[hypernix] Don't want this? Set {OPT_OUT_ENV_VARS[0]}=1, "
             f"or run: hypernix path --undo")
        echo("")
    return result


# ---------------------------------------------------------------------------
# CLI: `hypernix path`
# ---------------------------------------------------------------------------

_USAGE = """\
usage: hypernix path [--check | --apply | --undo | --print] [--shell SHELL]

Put HyperNix's console scripts (hypernix, hnx, hyped, waiter, eth, ups,
hnx-map, ...) on your PATH.

  --check     Report what would happen; write nothing. (default)
  --apply     Write the PATH block into your shell's startup file.
  --undo      Remove a previously written block.
  --print     Print the snippet only, for piping into a file yourself.
  --shell     Override shell detection (bash, zsh, fish, powershell, sh).
  --profile   Override which startup file is written.

The scripts directory is detected from this interpreter. Inside a
virtualenv, --apply is refused unless --force is given: that directory is
only meant to be on PATH while the environment is activated.
"""


def cli_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="hypernix path", add_help=False)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--undo", action="store_true")
    parser.add_argument("--print", dest="print_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shell")
    parser.add_argument("--profile")
    parser.add_argument("-h", "--help", action="store_true")

    try:
        ns = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    except SystemExit:
        print(_USAGE, file=sys.stderr)
        return 2

    if ns.help:
        print(_USAGE)
        return 0

    shell = ns.shell or detect_shell()
    directory = scripts_dir()
    profile = Path(ns.profile) if ns.profile else None

    if ns.print_only:
        print(snippet_for_shell(directory, shell))
        return 0

    if ns.undo:
        result = remove_from_path(shell=shell, profile=profile)  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
        print(result.message)
        return 0 if result.status in ("removed", "not-configured") else 1

    if ns.apply:
        if in_isolated_env() and not ns.force and profile is None:
            print(
                "Refusing to write a PATH block from inside a virtualenv/conda env.\n"
                f"  scripts directory: {directory}\n"
                "That directory belongs to this environment and is on PATH only while\n"
                "it is activated; adding it to your shell profile would leak it into\n"
                "every shell. Use --force if you really mean to.",
                file=sys.stderr,
            )
            return 1
        result = ensure_on_path(apply=True, directory=directory, shell=shell,  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
                                profile=profile, force=ns.force)
        print(result.message)
        return 0 if result.status in ("written", "already-on-path", "already-configured") else 1

    # Default: --check.
    result = ensure_on_path(apply=False, directory=directory, shell=shell, profile=profile)  # nosec B604 - `shell` here names a shell (bash, zsh, fish), it is not shell=True
    print(f"scripts directory : {directory}")
    print(f"shell             : {shell}")
    print(f"startup file      : {result.profile or '(unknown)'}")
    print(f"on PATH           : {'yes' if is_on_path(directory) else 'no'}")
    if in_isolated_env():
        print("environment       : virtualenv/conda (automatic setup is skipped here)")
    print()
    print(result.message)
    if result.status == "would-write":
        print("\nRun `hypernix path --apply` to write it.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
