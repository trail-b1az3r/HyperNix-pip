"""``hnx-t1 upgrade -S env`` — bring a T1 server's .env up to this version (0.72.8a4).

    hnx-t1 upgrade -S env           # the configuration
    hnx-t1 upgrade -S venv          # the Python the server runs on (what plain `upgrade` does)
    hnx-t1 upgrade -S all           # both: the venv first, so the env is upgraded by the new code

Upgrading the package gives a server new settings, but its .env was
written by an older installer and does not mention them, so nobody
finds out they exist. This adds every setting this version has that the
file lacks, **commented out at its default** with a line on what it
does: nothing changes until somebody sets one, and a default that moves
in a later release is not frozen at today's value by being written down.

It also:

* adds a token secret when the file has none (without one the server
  makes a new secret each start, and every token dies with it);
* sets the file to 0600 when it was readable by others;
* reports ``T1_*`` names nothing in HyperNix reads -- usually a typo,
  shown with the setting it was probably meant to be. They are left in.

The file it replaces is kept as ``.env.bak-<time>``.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import os
import re
import secrets
import shutil
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Setting", "Report", "backup_name", "known_settings", "read_by_hypernix", "upgrade_env", "cli_main"]

_NAME = re.compile(r"T1_[A-Z0-9_]+")
#: Never written as a commented example: a secret's line invites pasting one in.
_SECRETS = {"T1_TOKEN_SECRET", "T1_DEPLOY_SECRET", "T1_LMSTUDIO_API_KEY", "T1_HF_TOKEN", "T1_T2_ADMIN_PASSWORD"}


@dataclass(frozen=True)
class Setting:
    name: str
    default: str
    help: str


@dataclass
class Report:
    added: list[str] = field(default_factory=list)
    unknown: dict[str, str] = field(default_factory=dict)
    secret_added: bool = False
    mode_fixed: bool = False
    backup: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.added or self.secret_added or self.mode_fixed)


def _shown(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        return ""
    return str(value)


def _comments_above(lines: list[str], lineno: int) -> str:
    """The comment block just above line *lineno* (1-based), as one sentence."""
    out: list[str] = []
    i = lineno - 2
    while i >= 0 and lines[i].strip().startswith("#"):
        text = lines[i].strip().lstrip("#:").strip()
        if text.startswith("---"):
            break
        out.insert(0, text)
        i -= 1
    sentence = " ".join(t for t in out if t)
    first = re.split(r"(?<=[.!?])\s", sentence, maxsplit=1)[0] if sentence else ""
    return first[:160]


def known_settings() -> dict[str, Setting]:
    """Every setting :class:`~hypernix.t1api.config.T1APIConfig` reads, at its default."""
    from . import config as cfg_mod

    source = Path(cfg_mod.__file__).read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "T1APIConfig")
    field_env: dict[str, tuple[str, int]] = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            names = [c.value for c in ast.walk(node.value)
                     if isinstance(c, ast.Constant) and isinstance(c.value, str) and _NAME.fullmatch(c.value)]
            if names:
                field_env[node.target.id] = (names[0], node.lineno)

    # Defaults: the config built with no T1_* in the environment.
    saved = {k: v for k, v in os.environ.items() if k.startswith("T1_")}
    for k in saved:
        del os.environ[k]
    try:
        defaults = cfg_mod.T1APIConfig()
    except Exception:  # noqa: BLE001 - defaults are a nicety; names are what matter
        defaults = None
    finally:
        os.environ.update(saved)

    out: dict[str, Setting] = {}
    for attr, (name, lineno) in field_env.items():
        value = getattr(defaults, attr, None) if defaults is not None else None
        out[name] = Setting(name, _shown(value), _comments_above(lines, lineno))
    return out


def read_by_hypernix() -> set[str]:
    """Every ``T1_*`` name anything in the package reads, config or not
    (``T1_HOST`` and ``T1_PORT`` are read by the manager, for one)."""
    root = Path(__file__).resolve().parents[1]
    names: set[str] = set()
    for path in root.rglob("*.py"):
        try:
            names |= set(_NAME.findall(path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    manager = root.parents[1] / "bin" / "hypernix-t1"
    if manager.is_file():
        names |= set(_NAME.findall(manager.read_text(encoding="utf-8", errors="replace")))
    return names | {"T1_HOST", "T1_PORT", "T1_CONFIG_DIR", "T1_T2_ADMIN_PASSWORD"}


def _present(text: str) -> tuple[set[str], set[str]]:
    """``(set names, names that appear only commented out)`` in a .env."""
    active, commented = set(), set()
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"^(#\s*)?(T1_[A-Z0-9_]+)\s*=", stripped)
        if m:
            (commented if m.group(1) else active).add(m.group(2))
    return active, commented - active


def backup_name(path: Path) -> Path:
    """``.env.bak-<time>``, with a counter when that name is taken: two
    backups in one second must not be one."""
    stem = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    candidate, n = stem, 1
    while candidate.exists():
        candidate = stem.with_name(f"{stem.name}-{n}")
        n += 1
    return candidate


def upgrade_env(path: Path, *, version: str = "", dry_run: bool = False) -> tuple[Report, str]:
    """Upgrade the .env at *path*. Returns the report and the new text."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; `hnx-t1 setup` writes one")
    text = path.read_text(encoding="utf-8")
    report = Report()
    active, commented = _present(text)
    settings = known_settings()

    readers = read_by_hypernix()
    for name in sorted(active - readers):
        near = difflib.get_close_matches(name, sorted(readers), n=1, cutoff=0.75)
        report.unknown[name] = near[0] if near else ""

    additions: list[str] = []
    if "T1_TOKEN_SECRET" not in active:
        additions += ["# Signs tokens. Without it the server made a new one each start.",
                      f"T1_TOKEN_SECRET={secrets.token_hex(32)}"]
        report.secret_added = True
    for name in sorted(settings):
        if name in active or name in commented or name in _SECRETS:
            continue
        s = settings[name]
        if s.help:
            additions.append(f"# {s.help}")
        additions.append(f"# {name}={s.default}")
        report.added.append(name)

    new_text = text
    if additions:
        stamp = time.strftime("%Y-%m-%d", time.gmtime())
        header = [
            "",
            f"# --- Added by hnx-t1 upgrade -S env (hypernix {version or '?'}, {stamp}) ---------",
            "# Settings this version has that this file did not mention. Each is",
            "# commented out at its default, so nothing changes until you set one.",
        ]
        new_text = text.rstrip("\n") + "\n" + "\n".join(header + additions) + "\n"

    mode = stat.S_IMODE(path.stat().st_mode)
    report.mode_fixed = bool(mode & 0o077)
    if dry_run or not report.changed:
        return report, new_text

    backup = backup_name(path)
    shutil.copy2(path, backup)
    backup.chmod(0o600)
    report.backup = str(backup)
    tmp = path.with_name(".env.upgrade-tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    os.replace(tmp, path)
    path.chmod(0o600)
    return report, new_text


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hnx-t1 upgrade -S env",
                                     description="Bring a T1 server's .env up to this version.")
    parser.add_argument("--env-file", default="", help="default: $T1_CONFIG_DIR/.env or ~/.hypernix/t1api/.env")
    parser.add_argument("--dry-run", action="store_true", help="print what would change; write nothing")
    args = parser.parse_args(argv)
    from .. import __version__

    path = Path(args.env_file or Path(os.environ.get("T1_CONFIG_DIR") or Path.home() / ".hypernix" / "t1api") / ".env")
    try:
        report, new_text = upgrade_env(path, version=__version__, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        print(f"hnx-t1 upgrade -S env: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(new_text, end="")
    word = "would add" if args.dry_run else "added"
    if report.added:
        print(f"  ✓ {word} {len(report.added)} setting(s), commented out at their defaults: "
              + ", ".join(report.added))
    if report.secret_added:
        print(f"  ✓ {word} a token secret (there was none)")
    if report.mode_fixed:
        print("  ✓ " + ("would set" if args.dry_run else "set") + f" {path.name} to 0600 (others could read it)")
    for name, near in report.unknown.items():
        hint = f" — did you mean {near}?" if near else ""
        print(f"  ! {name} is not a setting this version reads{hint}", file=sys.stderr)
    if not report.changed:
        print(f"  ✓ {path} is up to date with hypernix {__version__}")
    elif report.backup:
        print(f"  ✓ the one before is {Path(report.backup).name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
