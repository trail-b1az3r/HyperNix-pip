"""t1api.modelsync — mirror ``~/.hypernix/models`` into the T1 server's own folder.

``~/.hypernix/models`` is shared: HyperLink downloads land there, LM
Studio can be pointed at it (``hypernix-t1 override lms move-dir``), and
people drop GGUFs in by hand. The T1 server gets its own folder,
``~/.hypernix/t1api/models``, holding a symlink for every model file in
the shared one. With ``T1_MODEL_SYNC=1`` the server serves from that
folder and keeps it in step; ``hypernix-sync`` (also ``t1-sync`` and
``hypernix-t1 sync``) does the same by hand.

How the mirror is built
-----------------------
Folders are real folders and files are symlinks, one per file. Not one
symlink per top-level folder: ``Path.rglob`` does not descend into a
symlinked directory, so a linked Hugging Face repo would look empty to
the catalogue, and a native checkpoint's ``config.json`` would not be
found beside its weights.

"Fully" means both directions. A new file is linked, a moved one is
relinked, and a link whose file is gone is removed, along with a folder
this sync created that has nothing left in it. What it never touches:

* a real file or folder somebody put in the T1 folder themselves, which
  is reported as a conflict rather than replaced;
* a symlink pointing anywhere but the shared folder, likewise;
* hidden entries, and downloads still in progress (``.part``,
  ``.incomplete``, ``.tmp``, ``.lock``, ``.aria2``).

Only links that point into the source folder are ever removed, so
pointing the source somewhere else cannot delete what the old one
linked from other places. The folders it created are listed in
``.hypernix-sync.json`` so an empty folder of yours is never pruned.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MANIFEST",
    "SyncResult",
    "default_source",
    "default_target",
    "sync",
    "serving_dir",
]

#: The sync's own record of the folders it made, kept in the target.
MANIFEST = ".hypernix-sync.json"

#: A download still being written. Linking one would offer a model that
#: fails to load, and pruning would then fight the downloader.
_IN_PROGRESS = (".part", ".partial", ".incomplete", ".tmp", ".lock", ".download", ".aria2")


def default_source() -> Path:
    """The shared models folder: where HyperLink downloads and people drop files."""
    from ..hyperlink.catalogue import DEFAULT_LOCAL_DIR

    return DEFAULT_LOCAL_DIR


def default_target(config_dir: str | Path | None = None) -> Path:
    """``<T1 config dir>/models``, which is ``~/.hypernix/t1api/models``."""
    base = config_dir or os.environ.get("T1_CONFIG_DIR") or (Path.home() / ".hypernix" / "t1api")
    return Path(base) / "models"


@dataclass
class SyncResult:
    source: str
    target: str
    linked: list[str] = field(default_factory=list)
    relinked: list[str] = field(default_factory=list)
    unchanged: int = 0
    removed: list[str] = field(default_factory=list)
    folders_removed: list[str] = field(default_factory=list)
    conflicts: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    dry_run: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.linked or self.relinked or self.removed or self.folders_removed)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["changed"] = self.changed
        return data

    def summary(self) -> str:
        verb = "would link" if self.dry_run else "linked"
        parts = [
            f"{len(self.linked)} {verb}",
            f"{len(self.relinked)} relinked",
            f"{len(self.removed)} removed",
            f"{self.unchanged} unchanged",
        ]
        if self.conflicts:
            parts.append(f"{len(self.conflicts)} left alone")
        if self.errors:
            parts.append(f"{len(self.errors)} failed")
        return ", ".join(parts)


def _skipped(name: str) -> bool:
    lower = name.lower()
    return name.startswith(".") or lower.endswith(_IN_PROGRESS) or ".part." in lower


def _abs(path: Path) -> Path:
    """Absolute and normalised, without resolving symlinks in *path* itself."""
    return Path(os.path.normpath(os.path.abspath(path)))


def _points_into(link: Path, root: Path) -> Path | None:
    """Where *link* points, when that is inside *root*; else ``None``."""
    try:
        dest = Path(os.readlink(link))
    except OSError:
        return None
    if not dest.is_absolute():
        dest = link.parent / dest
    dest = _abs(dest)
    try:
        dest.relative_to(root)
    except ValueError:
        return None
    return dest


def _load_manifest(target: Path) -> set[str]:
    try:
        data = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
        return {str(p) for p in data.get("folders", [])}
    except (OSError, ValueError, AttributeError):
        return set()


def _save_manifest(target: Path, folders: set[str]) -> None:
    path = target / MANIFEST
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"folders": sorted(folders)}, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _wanted(source: Path) -> tuple[set[str], set[str]]:
    """``(folders, files)`` in *source*, as paths relative to it.

    Walks symlinked folders inside the source too (the LM Studio override
    leaves one), guarding against a loop by real path.
    """
    folders: set[str] = set()
    files: set[str] = set()
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(source, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not _skipped(d))
        rel_dir = os.path.relpath(dirpath, source)
        if rel_dir != ".":
            folders.add(rel_dir)
        for name in filenames:
            if _skipped(name):
                continue
            files.add(name if rel_dir == "." else os.path.join(rel_dir, name))
    return folders, files


def sync(
    source: str | Path | None = None,
    target: str | Path | None = None,
    *,
    prune: bool = True,
    dry_run: bool = False,
) -> SyncResult:
    """Make *target* a symlink mirror of every model file in *source*."""
    src = _abs(Path(source) if source else default_source())
    dst = _abs(Path(target) if target else default_target())
    result = SyncResult(source=str(src), target=str(dst), dry_run=dry_run)

    if not src.is_dir():
        raise FileNotFoundError(f"{src} does not exist, so there is nothing to sync from.")
    if dst == src or src in dst.parents or dst in src.parents:
        raise ValueError(
            f"The T1 models folder ({dst}) and the shared one ({src}) must not "
            f"contain each other."
        )

    folders, files = _wanted(src)
    made = _load_manifest(dst)
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    # Folders first, shallowest first, so a file always has somewhere to go.
    blocked: set[str] = set()
    for rel in sorted(folders, key=lambda p: (p.count(os.sep), p)):
        if any(rel == b or rel.startswith(b + os.sep) for b in blocked):
            continue
        here = dst / rel
        if here.is_symlink() or (here.exists() and not here.is_dir()):
            result.conflicts.append({"path": rel, "reason": "a file or link is where this folder goes"})
            blocked.add(rel)
            continue
        if not here.exists():
            if not dry_run:
                try:
                    here.mkdir()
                except OSError as exc:
                    result.errors.append({"path": rel, "error": str(exc)})
                    blocked.add(rel)
                    continue
            made.add(rel)

    for rel in sorted(files):
        if any(rel.startswith(b + os.sep) for b in blocked):
            continue
        link = dst / rel
        want = src / rel
        if link.is_symlink():
            points = _points_into(link, src)
            if points == want:
                result.unchanged += 1
                continue
            if points is None:
                result.conflicts.append({"path": rel, "reason": f"a link to {os.readlink(link)} is already here"})
                continue
            # One of ours, pointing at an old location.
            if not dry_run:
                try:
                    link.unlink()
                    link.symlink_to(want)
                except OSError as exc:
                    result.errors.append({"path": rel, "error": str(exc)})
                    continue
            result.relinked.append(rel)
            continue
        if link.exists():
            result.conflicts.append({"path": rel, "reason": "a real file is here; it is yours, so it stays"})
            continue
        if not dry_run:
            try:
                link.symlink_to(want)
            except OSError as exc:
                result.errors.append({"path": rel, "error": str(exc)})
                continue
        result.linked.append(rel)

    if prune and dst.is_dir():
        for dirpath, dirnames, filenames in os.walk(dst, topdown=False):
            base = Path(dirpath)
            for name in list(filenames) + list(dirnames):
                entry = base / name
                if not entry.is_symlink():
                    continue
                points = _points_into(entry, src)
                if points is None:
                    continue  # not ours
                rel = os.path.relpath(entry, dst)
                if rel in files and points == src / rel and points.exists():
                    continue
                if not dry_run:
                    try:
                        entry.unlink()
                    except OSError as exc:
                        result.errors.append({"path": rel, "error": str(exc)})
                        continue
                result.removed.append(rel)
            if base == dst:
                continue
            rel_dir = os.path.relpath(base, dst)
            if rel_dir in made and rel_dir not in folders:
                try:
                    empty = not any(base.iterdir())
                except OSError:
                    empty = False
                if empty:
                    if not dry_run:
                        try:
                            base.rmdir()
                        except OSError:
                            continue
                    made.discard(rel_dir)
                    result.folders_removed.append(rel_dir)

    if not dry_run and dst.is_dir():
        try:
            _save_manifest(dst, {m for m in made if (dst / m).is_dir()})
        except OSError as exc:
            result.errors.append({"path": MANIFEST, "error": str(exc)})
    result.removed.sort()
    return result


# ---------------------------------------------------------------------------
# The server's side
# ---------------------------------------------------------------------------

#: A listing syncs first; this stops a burst of listings doing it each time.
_MIN_INTERVAL = 5.0
_lock = threading.Lock()
_last: dict[tuple[str, str], float] = {}


def source_for(config: Any) -> Path:
    return Path(getattr(config, "hf_download_dir", "") or default_source())


def target_for(config: Any) -> Path:
    return Path(getattr(config, "models_dir", "") or default_target())


def serving_dir(config: Any, *, force: bool = False) -> Path | None:
    """The folder the server reads models from, synced first when it syncs.

    ``None`` (the catalogue's default, ``~/.hypernix/models``) unless
    ``T1_MODEL_SYNC`` is on or a download folder is configured. A failed
    sync is logged and the mirror served as it is: a model list must not
    disappear because one link could not be made.
    """
    if not getattr(config, "model_sync", False):
        configured = getattr(config, "hf_download_dir", "") or None
        return Path(configured) if configured else None
    src, dst = source_for(config), target_for(config)
    key = (str(src), str(dst))
    with _lock:
        now = time.monotonic()
        if force or now - _last.get(key, float("-inf")) >= _MIN_INTERVAL:
            _last[key] = now
            try:
                done = sync(src, dst)
                if done.changed or done.errors:
                    logger.info("t1api.modelsync: %s (%s -> %s)", done.summary(), src, dst)
            except (OSError, ValueError) as exc:
                logger.warning("t1api.modelsync: not synced: %s", exc)
    return dst
