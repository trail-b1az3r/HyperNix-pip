"""hypernix.hyperlink.catalogue — every model this server can run, in one list.

The bug this exists for
-----------------------
HyperLink's model picker had exactly one source: ``GET
/bridge/lmstudio/models``. So the app showed nothing at all unless LM
Studio was running, and it showed nothing *and said nothing* when it was
not — ``refreshModels`` swallowed the failure into an empty list on the
reasoning that a server with no LM Studio configured is a normal state,
which is true and is not the same as a server with no models.

Meanwhile the server had two other sets of models it never offered:

* the **registry**, which is what ``hypernix-t1 index`` writes and what
  every routed request is already checked against, and
* the GGUFs actually sitting in **``~/.hypernix/models``**, which the
  downloads page listed and the picker did not.

So a person could download a model through HyperLink, watch it appear in
"downloaded", and then not be able to pick it.

What this returns
-----------------
One list, three sources, deduplicated. Plus — and this is the part that
fixes the silent-empty-list behaviour — a per-source report saying which
sources answered, which did not, and why. An empty catalogue with
``lmstudio: unreachable`` on it is a screen somebody can act on; an empty
catalogue is not.

Nothing here decides whether a model is any *good*. It reports what the
files and the registry say, because guessing on the server and guessing
again in the app is two places to be wrong.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CatalogueModel",
    "SourceReport",
    "Catalogue",
    "collect",
    "local_models",
    "DEFAULT_LOCAL_DIR",
]

#: Where a model downloaded through HyperLink lands, and where somebody
#: dropping a .gguf in by hand expects it to be found.
DEFAULT_LOCAL_DIR = Path.home() / ".hypernix" / "models"

#: Sources, in the order a tie is broken. Registry first because an
#: operator edited it on purpose: its context limit is the one the server
#: enforces, so showing a different number from the file would be showing
#: a number that is not in effect.
SOURCE_ORDER = ("registry", "lmstudio", "local")


@dataclass
class CatalogueModel:
    """One model somebody could pick, whatever it came from."""

    model_id: str
    name: str
    #: "registry" | "lmstudio" | "local"
    source: str
    #: Where the file is, when this server has it. Empty for a model that
    #: only exists inside LM Studio — the bridge does not say.
    path: str = ""
    size_bytes: int = 0
    architecture: str = ""
    #: Billions. 0.0 when unknown rather than guessed from the filename.
    parameters_b: float = 0.0
    context_limit: int = 0
    quant: str = ""
    bits_per_weight: float = 0.0
    #: Loaded *right now*. Only LM Studio and the managed runner know
    #: this; a file on disk is never "loaded".
    loaded: bool = False
    #: False when something is known to stop this running here — an
    #: unreadable file, a sub-bit tier with no runtime. `detail` says
    #: what.
    runnable: bool = True
    detail: str = ""
    #: Which other sources also have this model, once deduplicated. What
    #: lets the app say "on disk and in LM Studio" rather than picking
    #: one and hiding the other.
    also_in: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "source": self.source,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "architecture": self.architecture,
            "parameters_b": round(self.parameters_b, 3),
            "context_limit": self.context_limit,
            "quant": self.quant,
            "bits_per_weight": round(self.bits_per_weight, 4),
            "loaded": self.loaded,
            "runnable": self.runnable,
            "detail": self.detail,
            "also_in": list(self.also_in),
        }


@dataclass
class SourceReport:
    """Whether one source answered, and what it said if not.

    The whole reason the catalogue is not just a list. "No models" and
    "LM Studio is not running" produce the same empty list and need
    completely different things from the person reading the screen.
    """

    name: str
    available: bool
    count: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "count": self.count,
            "detail": self.detail,
        }


@dataclass
class Catalogue:
    models: list[CatalogueModel] = field(default_factory=list)
    sources: list[SourceReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [m.to_dict() for m in self.models],
            "count": len(self.models),
            "sources": [s.to_dict() for s in self.sources],
        }


# ---------------------------------------------------------------------------
# Local files
# ---------------------------------------------------------------------------


def _gguf_files(root: Path) -> list[Path]:
    """Every .gguf under *root*, skipping the ones that are not models.

    A ``.part`` is an interrupted download. A dflash1 draft and a
    ``hnxq.`` bundle extract are outputs *of* a model rather than models
    somebody means to chat with, and listing them makes the picker a list
    of build artifacts — but only the suffix is used, because a file
    genuinely named that way is the user's business.
    """
    if not root.exists():
        return []
    found: list[Path] = []
    for path in sorted(root.rglob("*.gguf")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".part") or ".part." in name:
            continue
        if name.endswith(".draft.gguf"):
            continue
        found.append(path)
    return found


def local_models(root: Path | str | None = None) -> tuple[list[CatalogueModel], SourceReport]:
    """Every GGUF on this server's disk, read rather than guessed.

    Each file is opened and its own metadata used for the architecture,
    context limit and parameter count, because the alternative is parsing
    a filename and a filename is where those numbers go wrong. A file
    that cannot be read is still listed — with ``runnable=False`` and the
    reason — since "this model is broken" is the answer somebody is
    looking for when a model does not appear.
    """
    from ..t1api.modelindex import inspect, model_id_for

    directory = Path(root) if root is not None else DEFAULT_LOCAL_DIR
    if not directory.exists():
        return [], SourceReport(
            "local", available=False,
            detail=f"{directory} does not exist yet. Models downloaded through "
                   f"HyperLink, or dropped in by hand, are found here.",
        )

    models: list[CatalogueModel] = []
    for path in _gguf_files(directory):
        try:
            read = inspect(path)
        except Exception as exc:  # noqa: BLE001 - one bad file is not a dead list
            logger.debug("catalogue: %s could not be inspected: %s", path, exc)
            models.append(CatalogueModel(
                model_id=model_id_for(path), name=path.stem, source="local",
                path=str(path), size_bytes=_size(path),
                runnable=False, detail=f"could not be read: {exc}",
            ))
            continue
        models.append(CatalogueModel(
            model_id=read.model_id,
            name=read.display_name or path.stem,
            source="local",
            path=str(path),
            size_bytes=read.file_bytes,
            architecture=read.architecture,
            parameters_b=read.parameters_b,
            context_limit=read.context_limit,
            quant=read.tier,
            bits_per_weight=read.bits_per_weight,
            runnable=read.readable,
            detail=read.error or (
                # A sub-bit tier is a real file that a stock llama.cpp
                # cannot execute. Saying so here is the difference
                # between "why is this greyed out" and a failed load.
                "needs the HyperNix runtime (sub-bit tier)"
                if read.is_extension else ""
            ),
        ))
    return models, SourceReport(
        "local", available=True, count=len(models), detail=str(directory)
    )


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def registry_models(registry: Any) -> tuple[list[CatalogueModel], SourceReport]:
    """What ``hypernix-t1 index`` wrote, and what routing already enforces."""
    if registry is None:
        return [], SourceReport(
            "registry", available=False,
            detail="this server has no model registry configured.",
        )
    try:
        entries = list(registry.list())
    except Exception as exc:  # noqa: BLE001
        logger.debug("catalogue: registry unreadable: %s", exc)
        return [], SourceReport("registry", available=False, detail=str(exc))

    models = []
    for entry in entries:
        # An example entry is the installer's placeholder, marked "edit
        # before serving traffic". Offering it as something to chat with
        # is offering a model that is not there.
        if getattr(entry, "is_example_entry", False):
            continue
        routable = bool(getattr(entry, "is_routable", True))
        models.append(CatalogueModel(
            model_id=str(entry.model_id),
            name=str(entry.display_name or entry.model_id),
            source="registry",
            architecture=str(entry.architecture or ""),
            parameters_b=float(entry.total_parameters or 0.0),
            context_limit=int(entry.context_limit or 0),
            runnable=routable,
            detail=str(entry.notes or "") or (
                "" if routable else f"registry status: {entry.status}"
            ),
        ))
    return models, SourceReport("registry", available=True, count=len(models))


# ---------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------


def bridge_models(bridge: Any) -> tuple[list[CatalogueModel], SourceReport]:
    """Whatever LM Studio has, when LM Studio is there.

    Unreachable is reported, never raised. A server with no LM Studio is
    an ordinary server — it is only a problem when it is also the only
    source, which is the situation this whole module removes.
    """
    if bridge is None:
        return [], SourceReport(
            "lmstudio", available=False,
            detail="the LM Studio bridge is off (set T1_LMSTUDIO_ENABLED=1).",
        )
    try:
        found = bridge.list_models()
    except Exception as exc:  # noqa: BLE001 - the bridge raises its own type
        return [], SourceReport(
            "lmstudio", available=False,
            detail=f"not answering at {getattr(bridge, 'base_url', 'its address')}: {exc}",
        )

    models = []
    for entry in found:
        data = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
        model_id = str(data.get("model_id") or data.get("id") or "")
        models.append(CatalogueModel(
            model_id=model_id,
            name=model_id,
            source="lmstudio",
            architecture=str(data.get("architecture") or ""),
            loaded=bool(data.get("loaded")),
            # The loaded context when it is loaded, the maximum when it
            # is not. Reporting 0 for an unloaded model would make the
            # picker say a model has no context.
            context_limit=int(
                data.get("context_length") or data.get("max_context_length") or 0
            ),
            quant=str(data.get("quantization") or ""),
        ))
    return models, SourceReport("lmstudio", available=True, count=len(models))


# ---------------------------------------------------------------------------
# Putting them together
# ---------------------------------------------------------------------------


def _key(model: CatalogueModel) -> str:
    """What counts as "the same model" across two sources.

    The id, lowercased, with separators dropped. LM Studio reports
    ``TheBloke/Qwen2-7B-GGUF/qwen2-7b.Q4_K_M.gguf`` where the indexer
    makes ``qwen2-7b-q4-k-m``; the tail of the first, normalised, is the
    second. Matching on the tail is why the basename is used rather than
    the whole path.
    """
    tail = model.model_id.rsplit("/", 1)[-1]
    for suffix in (".gguf",):
        if tail.lower().endswith(suffix):
            tail = tail[: -len(suffix)]
    return "".join(ch for ch in tail.lower() if ch.isalnum())


def merge(groups: list[list[CatalogueModel]]) -> list[CatalogueModel]:
    """One entry per model, richest source winning, the rest recorded.

    "Richest" is :data:`SOURCE_ORDER`, and the loser is not discarded: it
    lands in ``also_in``, so the app can say a model is on disk *and*
    loaded in LM Studio instead of showing it once and leaving somebody
    wondering which one they picked.
    """
    by_key: dict[str, CatalogueModel] = {}
    rank = {name: index for index, name in enumerate(SOURCE_ORDER)}

    for group in groups:
        for model in group:
            if not model.model_id:
                continue
            key = _key(model)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = model
                continue
            winner, loser = (
                (existing, model)
                if rank.get(existing.source, 99) <= rank.get(model.source, 99)
                else (model, existing)
            )
            # Whichever wins, keep the facts only the other one had. A
            # registry entry knows no path and no size; a file on disk
            # knows both and does not know whether it is loaded.
            winner.path = winner.path or loser.path
            winner.size_bytes = winner.size_bytes or loser.size_bytes
            winner.architecture = winner.architecture or loser.architecture
            winner.parameters_b = winner.parameters_b or loser.parameters_b
            winner.context_limit = winner.context_limit or loser.context_limit
            winner.quant = winner.quant or loser.quant
            winner.bits_per_weight = winner.bits_per_weight or loser.bits_per_weight
            winner.loaded = winner.loaded or loser.loaded
            if not winner.runnable or not loser.runnable:
                winner.runnable = False
                winner.detail = winner.detail or loser.detail
            for name in (loser.source, *loser.also_in):
                if name != winner.source and name not in winner.also_in:
                    winner.also_in.append(name)
            by_key[key] = winner

    return sorted(
        by_key.values(),
        # Loaded first -- it is the one that answers without a wait --
        # then alphabetically, which is the only stable order when a
        # server has forty of them.
        key=lambda m: (not m.loaded, m.name.lower(), m.model_id),
    )


def collect(
    *,
    registry: Any = None,
    bridge: Any = None,
    local_dir: Path | str | None = None,
) -> Catalogue:
    """Every model this server can offer, from every source it has.

    Each source is optional and each failure is contained: a broken
    registry does not hide the files on disk, and LM Studio being off has
    never been a reason to show an empty picker.
    """
    groups: list[list[CatalogueModel]] = []
    reports: list[SourceReport] = []

    for models, report in (
        registry_models(registry),
        bridge_models(bridge),
        local_models(local_dir),
    ):
        groups.append(models)
        reports.append(report)

    return Catalogue(models=merge(groups), sources=reports)
