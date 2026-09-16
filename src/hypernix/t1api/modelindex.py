"""``hypernix-t1 index`` — turn a folder of GGUFs into a model registry.

The registry is the only place the T1 API looks up what a model can do:
its context limit, its token limits, whether it is routable at all. Every
other route calls :meth:`ModelRegistry.require` rather than trusting a
client-supplied ``model_id``, which is the right design and also means a
server with an empty registry serves nothing.

Until now the only ways to fill it were the installer's one-entry
template — placeholders, marked *"Edit before serving traffic"* — and
writing the JSON by hand. Both ask an operator to transcribe numbers
that are already in the files: the architecture, the context length, the
parameter count. Transcription is where those numbers go wrong, and a
context limit that is wrong in the registry is not caught anywhere; it
is simply the number the server enforces.

So this reads them. Every field it can derive comes from the GGUF's own
metadata and tensor table; every field it cannot -- pricing, plan,
availability -- is a policy decision that stays with the operator and
takes a documented default it can be told to change.

What it will not do
-------------------
**It never overwrites an entry you have edited.** A model already in the
registry is left exactly as it is unless ``--refresh`` is passed, and
even then only the derived fields move: pricing, plan, priority, notes
and status are yours. An indexer that reset a hand-tuned entry on every
run would be worse than no indexer, because the loss is silent.

**It does not invent a parameter count.** ``total_parameters`` is the
summed element count of the tensor table, which is the real number, not
a guess from the filename. A file called ``7B`` whose tensors say 6.74
billion is reported as 6.74.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import ModelAvailabilityFlag, ModelEntry, ModelPricing, ModelStatus

logger = logging.getLogger(__name__)

__all__ = [
    "IndexError_",
    "IndexedModel",
    "index_directory",
    "build_entry",
    "model_id_for",
    "write_registry",
    "DEFAULT_MODELS_DIR",
]

#: Where models live unless told otherwise. Relative on purpose: this is
#: the folder next to a checkout or an install, which is where someone
#: dropping a .gguf in expects it to be found.
DEFAULT_MODELS_DIR = Path("./hypernix/models")

#: Tasks a text-generation GGUF can serve. Anything narrower would need
#: to be read from the model card, which a GGUF does not carry.
_TASKS = ["chat", "completion"]

#: Architectures hnxrun can execute. Everything else is still indexed --
#: llama.cpp may well run it -- but ``local_available`` says which.
_HNXRUN_ARCHITECTURES = ("llama", "mistral", "qwen2", "hypernix")


class IndexError_(RuntimeError):
    """Indexing could not proceed. Named to avoid shadowing the builtin."""


@dataclass
class IndexedModel:
    """One GGUF, as the file describes itself."""

    path: Path
    model_id: str
    display_name: str
    architecture: str
    parameters_b: float
    context_limit: int
    tier: str
    bits_per_weight: float
    is_extension: bool
    file_bytes: int
    vocab_size: int = 0
    #: Parameters touched per token. Equal to `parameters_b` for a dense
    #: model; smaller for a mixture-of-experts, where most of the weights
    #: sit idle on any given token. 0.0 when it could not be worked out.
    active_parameters_b: float = 0.0
    #: Experts in total, and experts used per token. Both 0 for a dense
    #: model.
    expert_count: int = 0
    expert_used_count: int = 0
    error: str = ""
    #: Fields that could not be read, so a report can say so rather than
    #: presenting a default as though it were measured.
    assumed: list[str] = field(default_factory=list)

    @property
    def readable(self) -> bool:
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "model_id": self.model_id,
            "display_name": self.display_name,
            "architecture": self.architecture,
            "parameters_b": self.parameters_b,
            "context_limit": self.context_limit,
            "tier": self.tier,
            "bits_per_weight": self.bits_per_weight,
            "needs_hnxrun": self.is_extension,
            "file_bytes": self.file_bytes,
            "active_parameters_b": self.active_parameters_b,
            "expert_count": self.expert_count,
            "expert_used_count": self.expert_used_count,
            "assumed": list(self.assumed),
            "error": self.error,
        }


def model_id_for(path: str | Path) -> str:
    """A stable slug for *path*.

    The registry's own docstring is explicit that a ``model_id`` is a
    slug and never a parameter-count string, so this is derived from the
    filename with the quantisation suffix left on: two quantisations of
    one model are two entries with different limits and different
    quality, and collapsing them to one id would make the registry unable
    to express which is being served.
    """
    stem = Path(path).stem
    slug = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower()
    return re.sub(r"-{2,}", "-", slug) or "model"


def _read_gguf(path: Path) -> tuple[dict, list, str]:
    from ..quant.gguf import GGUFError, GGUFFile

    try:
        model = GGUFFile.read(path)
    except (GGUFError, OSError, ValueError) as exc:
        return {}, [], str(exc)
    return model.metadata, list(model.tensors), ""


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _elements(tensor) -> int:
    total = 1
    for dim in tensor.shape:
        total *= int(dim)
    return total


def inspect(path: str | Path) -> IndexedModel:
    """Read one GGUF and report what it says about itself."""
    model_path = Path(path)
    size = model_path.stat().st_size if model_path.exists() else 0
    metadata, tensors, error = _read_gguf(model_path)
    if error:
        return IndexedModel(
            path=model_path, model_id=model_id_for(model_path),
            display_name=model_path.stem, architecture="unknown",
            parameters_b=0.0, context_limit=0, tier="", bits_per_weight=0.0,
            is_extension=False, file_bytes=size, error=error,
        )

    assumed: list[str] = []
    architecture = str(metadata.get("general.architecture", "")) or "unknown"
    if architecture == "unknown":
        assumed.append("architecture")

    context = 0
    for key in (f"{architecture}.context_length", "llama.context_length"):
        if key in metadata:
            try:
                context = int(metadata[key])
                break
            except (TypeError, ValueError):
                continue
    if not context:
        # 8192 is the installer template's number, so a registry written
        # here and one written there agree about the unknown case.
        context = 8192
        assumed.append("context_limit")

    parameters = sum(_elements(t) for t in tensors)
    vocab = 0
    for tensor in tensors:
        if tensor.name == "token_embd.weight" and tensor.shape:
            vocab = int(tensor.shape[0])
            break

    tier, bpw, extension = "", 0.0, False
    try:
        from ..quant.hyprslug_headers import read_header

        header = read_header(model_path)
        tier = header.tier or ""
        bpw = float(header.bits_per_weight or 0.0)
        extension = bool(header.is_extension)
    except Exception as exc:  # noqa: BLE001 - a header is a nicety here
        logger.debug("modelindex: no header for %s: %s", model_path, exc)

    # Mixture-of-experts. A model with 235B total and 22B active does 22B
    # of work per token and needs 235B of memory, and the two numbers are
    # used for different things -- speed from the first, placement from
    # the second. Pricing by total is wrong by an order of magnitude.
    experts = _as_int(metadata.get(f"{architecture}.expert_count"))
    used = _as_int(metadata.get(f"{architecture}.expert_used_count"))
    active = float(parameters)
    if experts > 1 and 0 < used <= experts:
        # The expert tensors, identified by name. Summing them rather
        # than assuming a fraction, because the split between shared and
        # routed weights differs by architecture and a guess would be
        # wrong exactly on the models this matters for.
        expert_params = sum(
            _elements(t) for t in tensors if "_exps" in t.name or "experts" in t.name
        )
        if expert_params:
            active = parameters - expert_params + expert_params * (used / experts)

    display = metadata.get("general.name") or model_path.stem
    return IndexedModel(
        path=model_path,
        model_id=model_id_for(model_path),
        display_name=str(display),
        architecture=architecture,
        parameters_b=round(parameters / 1e9, 4),
        context_limit=context,
        tier=tier,
        bits_per_weight=bpw,
        is_extension=extension,
        file_bytes=size,
        vocab_size=vocab,
        active_parameters_b=round(active / 1e9, 4),
        expert_count=experts,
        expert_used_count=used,
        assumed=assumed,
    )


def build_entry(
    found: IndexedModel,
    *,
    plan: str = "free",
    input_price: float = 0.0,
    output_price: float = 0.0,
    currency: str = "USD",
    availability: str = "public",
    routing_priority: int = 10,
) -> ModelEntry:
    """A registry entry for *found*, with policy from the arguments.

    The split is deliberate: everything measurable comes off the file,
    and everything that is a decision -- what it costs, who may reach it,
    where it sits in the cascade -- is passed in. A default price of zero
    is accounting, not a claim that the model is free to run.
    """
    runnable_here = (
        found.architecture in _HNXRUN_ARCHITECTURES or not found.is_extension
    )
    notes = []
    if found.tier:
        notes.append(f"{found.tier} ({found.bits_per_weight:.4g} bpw)")
    if found.is_extension:
        notes.append("HyperNix extension type: needs hnxrun, not llama.cpp")
    if found.assumed:
        notes.append("assumed: " + ", ".join(found.assumed))
    notes.append(f"indexed from {found.path.name}")

    return ModelEntry(
        model_id=found.model_id,
        display_name=found.display_name,
        version="1.0",
        total_parameters=found.parameters_b,
        active_parameters=None,
        architecture=found.architecture,
        supported_tasks=list(_TASKS),
        availability=ModelAvailabilityFlag(availability),
        minimum_plan=plan,
        free_tier_available=(input_price == 0.0 and output_price == 0.0),
        api_available=True,
        local_available=runnable_here,
        remote_available=False,
        context_limit=found.context_limit,
        input_token_limit=found.context_limit,
        # Half the context, so a reply cannot be budgeted longer than the
        # window that has to hold the prompt as well.
        output_token_limit=max(256, found.context_limit // 2),
        tool_call_limit=8,
        pricing=ModelPricing(
            input_price_per_1k=input_price,
            output_price_per_1k=output_price,
            currency=currency,
        ),
        routing_priority=routing_priority,
        fallback_model=None,
        license="unspecified",
        status=ModelStatus.AVAILABLE,
        notes="; ".join(notes),
    )


def index_directory(
    directory: str | Path = DEFAULT_MODELS_DIR,
) -> list[IndexedModel]:
    """Every ``.gguf`` under *directory*, inspected.

    An unreadable file is reported in the list rather than raised: the
    point of walking a folder of models is to find out which one is the
    problem, and one bad file must not end the walk.
    """
    root = Path(directory)
    if not root.exists():
        raise IndexError_(
            f"No such directory: {root}. Create it and put .gguf files in it, "
            f"or pass --dir."
        )
    if not root.is_dir():
        raise IndexError_(f"Not a directory: {root}")
    return [inspect(p) for p in sorted(root.rglob("*.gguf"))]


def write_registry(
    entries: list[ModelEntry],
    path: str | Path,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    """Merge *entries* into the registry at *path*.

    Returns what changed, so the caller can report it rather than
    claiming success over a file it did not really alter.
    """
    target = Path(path)
    existing: dict[str, dict] = {}
    if target.exists():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise IndexError_(
                f"{target} is not valid JSON ({exc}). Move it aside and re-run "
                f"rather than have this overwrite something hand-written."
            ) from exc
        if not isinstance(loaded, list):
            raise IndexError_(f"{target} is not a JSON list of model entries.")
        for item in loaded:
            if isinstance(item, dict) and "model_id" in item:
                existing[item["model_id"]] = item

    added, updated, kept = [], [], []
    merged = dict(existing)
    for entry in entries:
        current = existing.get(entry.model_id)
        if current is None:
            merged[entry.model_id] = entry.to_dict()
            added.append(entry.model_id)
            continue
        if not refresh:
            kept.append(entry.model_id)
            continue
        fresh = entry.to_dict()
        # Only the measured fields move. Everything an operator may have
        # tuned stays theirs -- that is the whole contract of --refresh.
        for policy in (
            "minimum_plan", "pricing", "routing_priority", "availability",
            "status", "fallback_model", "license", "free_tier_available",
            "notes", "version", "is_example_entry",
        ):
            if policy in current:
                fresh[policy] = current[policy]
        if fresh != current:
            merged[entry.model_id] = fresh
            updated.append(entry.model_id)
        else:
            kept.append(entry.model_id)

    ordered = sorted(merged.values(), key=lambda d: d.get("model_id", ""))
    payload = json.dumps(ordered, indent=2) + "\n"

    # Only touch the file if the bytes actually differ. Re-indexing an
    # unchanged folder is the common case -- it is what running this
    # command twice does -- and rewriting identical content moves the
    # mtime, which is what a file watcher or a config-reload hook is
    # looking at. "Nothing changed" should look like nothing changed.
    written = True
    if target.exists() and target.read_text(encoding="utf-8") == payload:
        written = False
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")

    return {
        "path": str(target),
        "added": added,
        "updated": updated,
        "unchanged": kept,
        "total": len(ordered),
        "written": written,
    }
