"""waiter.kits — installing user-made mods (``waiter serv -k``, 0.72.6).

A **kit** is a folder, or a ``.zip`` of one, with a ``kit.json`` at its
top::

    {
      "name": "fancy-status",          # a-z, 0-9, - and _; the folder name
      "version": "1.2.0",
      "kind": "waiter",                # waiter | t1api-client | hyped-pro | other
      "description": "A status line with colours",
      "commands": {                    # waiter kits only, optional
        "status": "fancy_status:main"  # module:function, called with argv
      }
    }

Kits install to ``~/.hypernix/waiter/kits/<name>/``. ``waiter kits``
lists and removes them, and ``waiter kits run <kit> <command> [args]``
runs a waiter kit's command. Kits of the other kinds are installed for
the program they are for to find (:func:`installed`, filtered by kind).

**A kit is code, and it runs as you.** Installing one is the same trust
decision as ``pip install``: waiter checks that the archive is well-formed
and cannot write outside its folder, and it does nothing else to vet the
contents. Install kits you would run yourself.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "KitError",
    "Kit",
    "KINDS",
    "kits_dir",
    "install",
    "installed",
    "remove",
    "run_command",
]

KINDS = ("waiter", "t1api-client", "hyped-pro", "other")
MAX_KIT_BYTES = 50 * 1024 * 1024
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_ENTRY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$")


class KitError(ValueError):
    """A kit that cannot be installed or run, with the reason."""


@dataclass
class Kit:
    name: str
    version: str
    kind: str
    description: str = ""
    commands: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "kind": self.kind,
                "description": self.description, "commands": dict(self.commands),
                "path": str(self.path) if self.path else ""}


def kits_dir() -> Path:
    from .local_config import _DEFAULT_CONFIG_DIR

    return _DEFAULT_CONFIG_DIR / "kits"


def _read_manifest(folder: Path) -> Kit:
    manifest = folder / "kit.json"
    if not manifest.is_file():
        raise KitError(f"{folder} has no kit.json at its top")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise KitError(f"kit.json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise KitError("kit.json must be an object")
    name = str(data.get("name", ""))
    if not _NAME.match(name):
        raise KitError(f"kit name {name!r} must be lowercase letters, digits, - and _")
    kind = str(data.get("kind", "other"))
    if kind not in KINDS:
        raise KitError(f"kit kind {kind!r} is not one of {', '.join(KINDS)}")
    commands = data.get("commands") or {}
    if not isinstance(commands, dict):
        raise KitError("kit.json 'commands' must be an object")
    for command, entry in commands.items():
        if not _NAME.match(str(command)) or not _ENTRY.match(str(entry)):
            raise KitError(f"command {command!r}: {entry!r} is not 'module:function'")
    if commands and kind != "waiter":
        raise KitError("only a waiter kit has commands")
    return Kit(name=name, version=str(data.get("version", "0")), kind=kind,
               description=str(data.get("description", ""))[:200],
               commands={str(k): str(v) for k, v in commands.items()}, path=folder)


def _safe_extract(archive: Path, into: Path) -> Path:
    """Unzip, refusing anything that would land outside *into*."""
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise KitError(f"{archive} is not a zip file") from exc
    with zf:
        total = 0
        root = into.resolve()
        for info in zf.infolist():
            total += info.file_size
            if total > MAX_KIT_BYTES:
                raise KitError(f"the kit unpacks to more than {MAX_KIT_BYTES // (1024 * 1024)} MB")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise KitError(f"{info.filename!r} is a symbolic link; kits may not contain links")
            target = (into / info.filename).resolve()
            if target != root and root not in target.parents:
                raise KitError(f"{info.filename!r} would be written outside the kit's folder")
        zf.extractall(into)
    # A kit zipped as its folder has one directory at the top.
    if not (into / "kit.json").is_file():
        children = [c for c in into.iterdir() if not c.name.startswith(".")]
        if len(children) == 1 and children[0].is_dir():
            return children[0]
    return into


def install(source: str | Path, *, destination: Path | None = None) -> Kit:
    """Install a kit from a folder or a ``.zip``. Replaces the same name."""
    source = Path(source).expanduser()
    target_root = destination or kits_dir()
    with tempfile.TemporaryDirectory(prefix="waiter-kit-") as scratch:
        if source.is_dir():
            staged = Path(scratch) / "kit"
            shutil.copytree(source, staged, symlinks=False,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
        elif source.is_file():
            if source.stat().st_size > MAX_KIT_BYTES:
                raise KitError("the kit archive is larger than 50 MB")
            staged = _safe_extract(source, Path(scratch))
        else:
            raise KitError(f"{source} is not a folder or a file")
        kit = _read_manifest(staged)
        target_root.mkdir(parents=True, exist_ok=True)
        final = target_root / kit.name
        if final.exists():
            shutil.rmtree(final)
        shutil.copytree(staged, final, symlinks=False)
    kit.path = final
    return kit


def installed(*, kind: str | None = None, root: Path | None = None) -> list[Kit]:
    base = root or kits_dir()
    if not base.is_dir():
        return []
    kits = []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        try:
            kit = _read_manifest(folder)
        except KitError:
            continue
        if kind is None or kit.kind == kind:
            kits.append(kit)
    return kits


def remove(name: str, *, root: Path | None = None) -> bool:
    if not _NAME.match(name):
        raise KitError(f"{name!r} is not a kit name")
    folder = (root or kits_dir()) / name
    if not folder.is_dir():
        return False
    shutil.rmtree(folder)
    return True


def run_command(name: str, command: str, argv: list[str], *, root: Path | None = None) -> int:
    """Run a waiter kit's command: import ``module`` from the kit, call
    ``function(argv)``, return its exit code."""
    matches = [k for k in installed(root=root) if k.name == name]
    if not matches:
        raise KitError(f"no kit called {name!r} is installed (waiter kits lists them)")
    kit = matches[0]
    entry = kit.commands.get(command)
    if entry is None:
        known = ", ".join(sorted(kit.commands)) or "none"
        raise KitError(f"kit {name!r} has no command {command!r} (it has: {known})")
    module_name, function_name = entry.split(":", 1)
    module_path = kit.path / (module_name.replace(".", "/") + ".py")
    if not module_path.is_file():
        raise KitError(f"{module_path} is missing from the kit")
    spec = importlib.util.spec_from_file_location(f"waiter_kit_{name.replace('-', '_')}_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise KitError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(kit.path))
    try:
        spec.loader.exec_module(module)
        function = getattr(module, function_name, None)
        if not callable(function):
            raise KitError(f"{entry} is not a function in the kit")
        result = function(list(argv))
    finally:
        try:
            sys.path.remove(str(kit.path))
        except ValueError:
            pass
    return int(result or 0)
