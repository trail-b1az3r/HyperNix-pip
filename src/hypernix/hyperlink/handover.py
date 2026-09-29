"""Move the model LM Studio is serving onto the HyperNix runner.

A model loaded in LM Studio is served by LM Studio's own llama.cpp, with
LM Studio's settings, and HyperLink can only ask it for replies. Moving
it onto this server's runner means the server owns it: the phone can
see where its layers went, stop and switch it, and hyperchat can queue
prompts on it. Doing that by hand meant walking to the PC, ejecting the
model in LM Studio, and loading the same file here.

Three steps, in an order chosen so the VRAM is never asked for twice:

1. **Find the file.** LM Studio's API names a model but never says where
   it lives. ``lms ls --json`` does, when the ``lms`` command is on this
   machine. Otherwise LM Studio's models folder is searched for a GGUF
   whose folder or name matches the id.
2. **Unload it from LM Studio**, through its REST API, else ``lms
   unload``. Before the runner loads, because two copies of one model
   on one GPU is how the second load fails out of memory.
3. **Load it here.** If that fails, the model is loaded back into LM
   Studio, so a failed move costs a reload, not the model.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "HandoverError",
    "LMStudioFile",
    "find_lmstudio_file",
    "lmstudio_models_dirs",
    "load_in_lmstudio",
    "unload_from_lmstudio",
]


class HandoverError(RuntimeError):
    """The move could not be made; the message says which step and why."""

    def __init__(self, message: str, *, step: str, remedy: str = "") -> None:
        super().__init__(message)
        self.step = step
        self.remedy = remedy


@dataclass
class LMStudioFile:
    model_id: str
    path: Path
    #: How it was found: "lms" or "scan".
    found_by: str
    candidates: list[str] = field(default_factory=list)


def lmstudio_models_dirs() -> list[Path]:
    """Where LM Studio keeps its models, the configured folder first."""
    from ..t1api import lmsoverride

    dirs: list[Path] = []
    try:
        _settings, folder = lmsoverride.current_models_dir()
        if folder is not None:
            dirs.append(folder)
    except Exception:  # noqa: BLE001 - an unreadable settings file just means "look elsewhere"
        logger.debug("handover: LM Studio settings unreadable", exc_info=True)
    for home in lmsoverride.lmstudio_homes():
        dirs.append(home / "models")
    seen: set[Path] = set()
    return [d for d in dirs if d.is_dir() and not (d in seen or seen.add(d))]


def _lms() -> str | None:
    found = shutil.which("lms")
    if found:
        return found
    for home in (Path.home() / ".lmstudio", Path.home() / ".cache" / "lm-studio"):
        candidate = home / "bin" / "lms"
        if candidate.is_file():
            return str(candidate)
    return None


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _from_lms(model_id: str, dirs: list[Path]) -> LMStudioFile | None:
    lms = _lms()
    if lms is None:
        return None
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv
            [lms, "ls", "--json"], capture_output=True, text=True, timeout=30,
        )
        entries = json.loads(done.stdout or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not isinstance(entries, list):
        return None
    wanted = _normalise(model_id)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        keys = [str(entry.get(k) or "") for k in ("modelKey", "indexedModelIdentifier", "path")]
        if not any(_normalise(k) == wanted or (k and _normalise(k).endswith(wanted)) for k in keys):
            continue
        relative = str(entry.get("path") or "")
        path = Path(relative)
        if not path.is_absolute():
            for base in dirs:
                if (base / relative).exists():
                    path = base / relative
                    break
        if path.is_dir():
            from ..system.linkwalk import walk_files

            ggufs = sorted(walk_files(path, ".gguf"), key=lambda p: p.stat().st_size, reverse=True)
            path = ggufs[0] if ggufs else path
        if path.is_file():
            return LMStudioFile(model_id, path, "lms")
    return None


def _from_scan(model_id: str, dirs: list[Path]) -> LMStudioFile | None:
    """A GGUF whose folder or file name matches the id.

    ``google/gemma-3-4b`` is usually ``lmstudio-community/gemma-3-4b-it-GGUF/
    gemma-3-4b-it-Q4_K_M.gguf`` on disk, so the id's last part is matched
    against the repo folder and the file name, loosely. More than one
    match is a guess, and a guess is refused with the candidates listed.
    """
    tail = _normalise(model_id.rsplit("/", 1)[-1])
    if not tail:
        return None
    from ..system.linkwalk import walk_files

    matches: list[Path] = []
    for base in dirs:
        for path in walk_files(base, ".gguf"):
            name = path.name.lower()
            if "mmproj" in name or name.endswith(".part"):
                continue
            if tail in _normalise(path.parent.name) or _normalise(path.stem).startswith(tail):
                matches.append(path)
    if len(matches) == 1:
        return LMStudioFile(model_id, matches[0], "scan")
    if matches:
        raise HandoverError(
            f"{model_id} matches {len(matches)} files in LM Studio's models folder, so which "
            f"one LM Studio loaded cannot be told from here.",
            step="find",
            remedy="Install LM Studio's `lms` command (it says exactly which file is loaded), "
                   "or load the file you mean with `hypernix-t1 runner load`.",
        )
    return None


def find_lmstudio_file(model_id: str, dirs: list[Path] | None = None) -> LMStudioFile:
    """The GGUF on this machine that LM Studio serves as *model_id*."""
    folders = lmstudio_models_dirs() if dirs is None else dirs
    found = _from_lms(model_id, folders) or _from_scan(model_id, folders)
    if found is None:
        where = ", ".join(str(d) for d in folders) or "no LM Studio models folder was found"
        raise HandoverError(
            f"Could not find the file for {model_id} ({where}).",
            step="find",
            remedy="The move needs LM Studio on this machine. If its models are elsewhere, "
                   "point it at ~/.hypernix/models with `hypernix-t1 override lms move-dir`.",
        )
    return found


def unload_from_lmstudio(bridge: Any, model_id: str) -> str:
    """Eject *model_id* from LM Studio. Returns how it was done."""
    try:
        bridge._request("POST", "/api/v1/models/unload", body={"instance_id": model_id},
                        timeout=60)
        return "api"
    except Exception as exc:  # noqa: BLE001 - older LM Studio: fall back to lms
        logger.debug("handover: REST unload failed (%s); trying lms", exc)
    lms = _lms()
    if lms is None:
        raise HandoverError(
            f"LM Studio would not unload {model_id} over its API, and the `lms` command "
            f"is not on this machine.",
            step="unload",
            remedy="Eject the model in LM Studio, then load it with `hypernix-t1 runner load`.",
        )
    done = subprocess.run(  # noqa: S603 - fixed argv, model id as one argument
        [lms, "unload", model_id], capture_output=True, text=True, timeout=120,
    )
    if done.returncode != 0:
        raise HandoverError(
            f"`lms unload {model_id}` failed: {(done.stderr or done.stdout).strip()[:300]}",
            step="unload",
        )
    return "lms"


def load_in_lmstudio(bridge: Any, model_id: str) -> bool:
    """Put *model_id* back into LM Studio after a failed move. Best effort."""
    try:
        bridge._request("POST", "/api/v1/models/load", body={"model": model_id}, timeout=300)
        return True
    except Exception:  # noqa: BLE001
        logger.debug("handover: REST load failed; trying lms", exc_info=True)
    lms = _lms()
    if lms is None:
        return False
    try:
        done = subprocess.run(  # noqa: S603
            [lms, "load", model_id, "-y"], capture_output=True, text=True, timeout=600,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0
