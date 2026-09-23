"""``hypernix-t1 override lms move-dir <folder>`` — point LM Studio elsewhere.

Changes where LM Studio (and its ``lms`` CLI, which reads the same
settings) keeps and looks for models. The default target is
``~/.hypernix/models``, which is the point: one folder that both LM
Studio and HyperNix read, instead of two copies of every GGUF.

This edits a file another application owns, so it is careful about it
in four ways:

* **Found, not assumed.** LM Studio has moved its home between releases
  and platforms. The settings file is searched for across the known
  homes, and ``LMSTUDIO_HOME`` wins — the same rule the rest of HyperNix
  uses to find LM Studio.
* **Backed up first**, next to the original, and ``override lms revert``
  puts the newest backup back.
* **Written atomically**, temp file then rename, with every other key
  kept exactly. A half-written settings file is how an app comes up with
  its preferences reset.
* **Not while LM Studio is running.** It can write its settings on exit,
  which would quietly undo this. ``--force`` goes ahead anyway.

Moving the model files themselves is opt-in (``--move-files``), shown as
a plan before anything moves, and never overwrites a file already at the
destination. Repointing is instant and reversible; moving forty
gigabytes is neither.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SETTING_KEY",
    "LMSOverrideError",
    "lmstudio_homes",
    "find_settings",
    "current_models_dir",
    "set_models_dir",
    "revert",
    "plan_move",
    "move_models",
    "lmstudio_running",
    "default_target",
]

#: The key LM Studio stores its models folder under.
SETTING_KEY = "downloadsFolder"
_BACKUP_SUFFIX = ".hnx-backup-"


class LMSOverrideError(RuntimeError):
    """Something the user can act on; the message says what."""


def default_target() -> Path:
    return Path.home() / ".hypernix" / "models"


def lmstudio_homes() -> list[Path]:
    """Where LM Studio might live, most likely first."""
    override = os.environ.get("LMSTUDIO_HOME")
    if override:
        return [Path(override).expanduser()]
    home = Path.home()
    homes = []
    # `lms` writes a pointer to the active home; if present it is the
    # most authoritative answer there is.
    pointer = home / ".lmstudio-home-pointer"
    if pointer.is_file():
        try:
            pointed = Path(pointer.read_text(encoding="utf-8").strip()).expanduser()
            if str(pointed):
                homes.append(pointed)
        except OSError:
            pass
    homes += [
        home / ".lmstudio",
        home / ".cache" / "lm-studio",
        home / "Library" / "Application Support" / "LM Studio",
        home / "AppData" / "Roaming" / "LM Studio",
    ]
    seen: set[Path] = set()
    return [h for h in homes if not (h in seen or seen.add(h))]


def _settings_candidates(home: Path) -> list[Path]:
    return [home / "settings.json", home / ".internal" / "app-settings.json"]


def find_settings() -> Path | None:
    """The settings file LM Studio is using, or None."""
    for home in lmstudio_homes():
        for candidate in _settings_candidates(home):
            if candidate.is_file():
                return candidate
    return None


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError as exc:
        # Not "fixed" by overwriting: an unreadable file is either
        # mid-write by LM Studio or damaged, and in both cases replacing
        # it with our version throws away settings we cannot see.
        raise LMSOverrideError(
            f"{path} is not valid JSON ({exc}). Not touching it — if LM "
            f"Studio is running, quit it and try again."
        ) from None
    if not isinstance(data, dict):
        raise LMSOverrideError(f"{path} does not hold a settings object")
    return data


def current_models_dir() -> tuple[Path | None, Path | None]:
    """``(settings file, models folder it names)``. Either may be None."""
    settings = find_settings()
    if settings is None:
        return None, None
    value = _read(settings).get(SETTING_KEY)
    if value:
        return settings, Path(value).expanduser()
    # No explicit folder: LM Studio uses <home>/models.
    home = settings.parent if settings.name == "settings.json" else settings.parent.parent
    return settings, home / "models"


def lmstudio_running() -> list[str]:
    """Names of running LM Studio processes."""
    try:
        import psutil
    except ImportError:
        return []
    found = []
    for proc in psutil.process_iter(["name"]):
        name = (proc.info.get("name") or "").lower()
        if name in ("lm studio", "lm-studio", "lmstudio", "lm studio.exe") or \
                name.startswith("lm studio"):
            found.append(proc.info["name"])
    return found


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class Change:
    settings: Path
    before: str | None
    after: str
    backup: Path | None
    created: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"settings": str(self.settings), "before": self.before,
                "after": self.after, "backup": str(self.backup) if self.backup else None,
                "created": self.created}


def set_models_dir(folder: Path | str | None = None, *, force: bool = False,
                   create: bool = True) -> Change:
    """Point LM Studio's models folder at *folder*. Backs up first."""
    target = Path(folder).expanduser() if folder else default_target()
    target = target.resolve() if target.exists() else target.absolute()
    if target.exists() and not target.is_dir():
        raise LMSOverrideError(f"{target} exists and is not a folder")

    running = lmstudio_running()
    if running and not force:
        raise LMSOverrideError(
            "LM Studio is running, and it can write its settings when it "
            "quits — which would undo this. Quit it first, or pass --force."
        )

    if create:
        target.mkdir(parents=True, exist_ok=True)

    settings = find_settings()
    created = False
    if settings is None:
        homes = [h for h in lmstudio_homes() if h.is_dir()]
        if not homes:
            raise LMSOverrideError(
                "LM Studio does not look installed here: no home in "
                + ", ".join(str(h) for h in lmstudio_homes())
                + ". Set LMSTUDIO_HOME if it lives somewhere else."
            )
        settings = homes[0] / "settings.json"
        data: dict[str, Any] = {}
        created = True
    else:
        data = _read(settings)

    before = data.get(SETTING_KEY)
    backup = None
    if not created:
        backup = settings.with_name(settings.name + _BACKUP_SUFFIX + time.strftime("%Y%m%d-%H%M%S"))
        shutil.copy2(settings, backup)

    data[SETTING_KEY] = str(target)
    _atomic_write(settings, data)
    return Change(settings, before, str(target), backup, created)


def backups(settings: Path) -> list[Path]:
    return sorted(settings.parent.glob(settings.name + _BACKUP_SUFFIX + "*"))


def revert(*, force: bool = False) -> Path:
    """Put the newest backup back. Returns the backup used."""
    settings = find_settings()
    if settings is None:
        raise LMSOverrideError("no LM Studio settings file to revert")
    found = backups(settings)
    if not found:
        raise LMSOverrideError(f"no backups of {settings} — nothing to revert to")
    if lmstudio_running() and not force:
        raise LMSOverrideError("LM Studio is running; quit it first, or pass --force")
    newest = found[-1]
    _read(newest)                     # refuse to restore a broken backup
    shutil.copy2(newest, settings)
    return newest


# ---------------------------------------------------------------------------
# Moving the files themselves
# ---------------------------------------------------------------------------


@dataclass
class MovePlan:
    source: Path
    target: Path
    move: list[tuple[Path, Path]] = field(default_factory=list)
    skip: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def bytes(self) -> int:
        return sum(src.stat().st_size for src, _ in self.move if src.exists())

    def summary(self) -> str:
        gib = self.bytes / 1024 ** 3
        return (f"{len(self.move)} file(s), {gib:.1f} GiB to move; "
                f"{len(self.skip)} skipped")


def plan_move(source: Path, target: Path) -> MovePlan:
    """What ``--move-files`` would move. Moves nothing."""
    source, target = Path(source).expanduser(), Path(target).expanduser()
    plan = MovePlan(source, target)
    if not source.is_dir():
        return plan
    if source.resolve() == target.resolve():
        return plan
    if target.resolve().is_relative_to(source.resolve()):
        # Moving a folder into itself recurses forever on some platforms
        # and silently nests on others.
        raise LMSOverrideError(f"{target} is inside {source}; pick a folder outside it")
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        destination = target / path.relative_to(source)
        if destination.exists():
            plan.skip.append((path, "already at the destination"))
        else:
            plan.move.append((path, destination))
    return plan


def move_models(plan: MovePlan) -> list[Path]:
    """Carry out *plan*. Returns the files moved."""
    moved = []
    for src, dst in plan.move:
        if dst.exists():          # appeared since planning; still not overwritten
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append(dst)
    return moved


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="hypernix-t1 override lms",
        description="Point LM Studio's models folder somewhere else.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    mv = sub.add_parser("move-dir", help="point LM Studio at a models folder")
    mv.add_argument("folder", nargs="?", default="",
                    help=f"default: {default_target()}")
    mv.add_argument("--move-files", action="store_true",
                    help="also move the models already in the old folder")
    mv.add_argument("--yes", action="store_true", help="skip the move confirmation")
    mv.add_argument("--force", action="store_true",
                    help="write even though LM Studio is running")

    sub.add_parser("show", help="where LM Studio's models are now")
    rv = sub.add_parser("revert", help="restore the newest backup")
    rv.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "show":
            settings, folder = current_models_dir()
            print(f"settings : {settings or 'not found'}")
            print(f"models   : {folder or 'unknown'}")
            return 0
        if args.command == "revert":
            used = revert(force=args.force)
            print(f"restored {used.name}")
            return 0

        _, old = current_models_dir()
        change = set_models_dir(args.folder or None, force=args.force)
        print(f"LM Studio models folder: {change.before or old or '(default)'}")
        print(f"                     -> {change.after}")
        print(f"settings: {change.settings}"
              + (" (created)" if change.created else ""))
        if change.backup:
            print(f"backup  : {change.backup}  (undo: hypernix-t1 override lms revert)")

        if args.move_files and old is not None:
            plan = plan_move(old, Path(change.after))
            print(f"\n{plan.summary()}")
            if plan.move and not args.yes:
                print("Re-run with --yes to move them. Nothing has moved.")
                return 0
            moved = move_models(plan)
            print(f"moved {len(moved)} file(s)")
        elif old is not None and Path(old) != Path(change.after) and Path(old).is_dir():
            print(f"\nThe models already in {old} stay there. "
                  f"Add --move-files to bring them across.")
        return 0
    except LMSOverrideError as exc:
        print(f"hypernix-t1 override lms: {exc}", file=sys.stderr)
        return 1


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    cli_main()
