"""t1api.registry — the Model Registry service.

**Hard requirement (see the T1 API spec): the T1 API must only expose and
operate on models explicitly registered here.** Nothing in this module
discovers models from HyperNix-pip's local checkpoint cache, HuggingFace
Hub, or any client-supplied path. A model must have an explicit
:class:`ModelEntry` before it can be selected, routed to, assigned to a
server/key/plan, or included in usage accounting — everything else in
``t1api`` (routers, the router/cascade engine landing in Beta 2, usage
metering) is required to call :meth:`ModelRegistry.require` rather than
trust a client-supplied model_id directly.

Seed data
---------
``t1api/data/model_registry.example.json`` ships nine example entries taken
directly from the T1 API spec (HyperNix 1, Ryiver 1, nanoNix, ...). The spec
is explicit that these are placeholders ("just examples and are not
currently real"), so every seed entry is loaded with
``status=ModelStatus.EXAMPLE`` and is excluded from
:meth:`ModelRegistry.list` / :meth:`ModelRegistry.require` unless the
registry is constructed with ``include_examples=True`` (or the
``T1_ENABLE_EXAMPLE_MODELS=1`` environment variable is set — see
``t1api.config``). This keeps the hard requirement honest: until Rayla
swaps in real entries, the registry is intentionally empty from the point
of view of routing and API responses.

Adding real models is a data change, not a code change: write a JSON file
in the same shape and point ``ModelRegistry.load(path=...)`` at it, or call
``registry.register(entry)`` from an admin-only code path.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .errors import T1APIError, T1ErrorCode

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
_EXAMPLE_REGISTRY_PATH = _DATA_DIR / "model_registry.example.json"


#: Filenames a registry is looked for under, in preference order.
#: ``.jsonl`` is accepted because an indexer that appends one entry per
#: line is the natural way to write a registry incrementally, and a
#: format the writer can produce but the reader cannot load is not a
#: format.
REGISTRY_NAMES = ("models.json", "models.jsonl")


def registry_locations(config_dir: str | Path | None = None) -> list[Path]:
    """Where a registry is looked for, in order, most specific first.

    Discovery exists because the alternative was an environment variable
    and nothing else. ``hypernix-t1 index`` would write ``models.json``,
    the server would not be told about it, and ``waiter models`` would
    show the *example* entries -- which are documented as not real --
    while the operator's own models sat in a file two directories away.
    Nothing anywhere said the two were unconnected.
    """
    roots: list[Path] = []
    if config_dir:
        roots.append(Path(config_dir))
    env_dir = os.environ.get("T1_CONFIG_DIR")
    if env_dir:
        roots.append(Path(env_dir))
    roots.append(Path.home() / ".hypernix" / "t1api")
    roots.append(Path("./hypernix/models"))
    roots.append(Path.cwd())

    seen: set[Path] = set()
    found: list[Path] = []
    for root in roots:
        for name in REGISTRY_NAMES:
            candidate = root / name
            if candidate not in seen:
                seen.add(candidate)
                found.append(candidate)
    return found


def discover(config_dir: str | Path | None = None) -> Path | None:
    """The first registry file that exists, or None."""
    for candidate in registry_locations(config_dir):
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # noqa: PERF203 - an unreadable root is not fatal
            continue
    return None


def read_entries(path: str | Path) -> tuple[list[ModelEntry], list[str]]:
    """Entries from *path*, plus a list of what could not be read.

    Never raises. A registry file is written by a separate process --
    ``hypernix-t1 index``, or a person with an editor -- so the server
    can and does open it mid-write. Before this, three ordinary states
    each ended startup with a traceback:

        half-written JSON   json.JSONDecodeError
        a dict, not a list  TypeError: string indices must be integers
        one entry missing   KeyError: 'display_name'
        a required key

    The last is the worst of the three: one malformed entry discarded
    every good one alongside it. Bad entries are now skipped and named,
    and the models that *are* readable load.
    """
    src = Path(path)
    problems: list[str] = []
    try:
        text = src.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [f"{src}: {exc}"]

    raw: list[Any]
    if src.suffix == ".jsonl":
        raw = []
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                raw.append(json.loads(stripped))
            except ValueError as exc:
                # A truncated final line is what a half-flushed append
                # looks like; the lines before it are still good.
                problems.append(f"{src}:{number}: {exc}")
    else:
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            return [], [f"{src}: {exc}"]
        if isinstance(parsed, dict):
            # A single entry, or {"models": [...]}, are both things people
            # write. Accepting them beats a TypeError about string indices.
            inner = parsed.get("models")
            raw = list(inner) if isinstance(inner, list) else [parsed]
        elif isinstance(parsed, list):
            raw = parsed
        else:
            return [], [f"{src}: expected a list of model entries, got {type(parsed).__name__}"]

    entries: list[ModelEntry] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"{src}[{index}]: not an object")
            continue
        if item.get("_comment") and "model_id" not in item:
            continue  # the installer template's explanatory stub
        try:
            entries.append(ModelEntry.from_dict(item))
        except (KeyError, TypeError, ValueError) as exc:
            name = item.get("model_id", f"entry {index}")
            problems.append(f"{src}: {name}: {exc}")
    return entries, problems


class ModelStatus(StrEnum):
    """Lifecycle status of a registry entry."""

    EXAMPLE = "example"  # placeholder / seed data — never routable by default
    AVAILABLE = "available"
    BETA = "beta"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"


class ModelAvailabilityFlag(StrEnum):
    """Coarse availability classification, independent of per-plan gating."""

    PUBLIC = "public"
    RESTRICTED = "restricted"
    INTERNAL = "internal"
    OFFLINE = "offline"


@dataclass
class ModelPricing:
    """Pricing metadata. All fields optional — a free/local-only model can
    leave everything at 0."""

    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0
    currency: str = "USD"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ModelPricing:
        d = d or {}
        return cls(
            input_price_per_1k=float(d.get("input_price_per_1k", 0.0)),
            output_price_per_1k=float(d.get("output_price_per_1k", 0.0)),
            currency=d.get("currency", "USD"),
        )


@dataclass
class ModelEntry:
    """One explicit, admin-controlled model registry entry.

    Field list matches the T1 API spec's "MODEL REGISTRY" section minimum
    set. ``model_id`` is a stable slug (never a parameter-count string —
    see the spec's ``nanonix-mini-lite`` vs ``85b-25.25b`` example).
    """

    model_id: str
    display_name: str
    version: str
    total_parameters: float  # in billions, e.g. 875.1 == 875.1B
    active_parameters: float | None  # None for dense models w/ no MoE split
    architecture: str
    supported_tasks: list[str]
    availability: ModelAvailabilityFlag
    minimum_plan: str
    free_tier_available: bool
    api_available: bool
    local_available: bool
    remote_available: bool
    context_limit: int
    input_token_limit: int
    output_token_limit: int
    tool_call_limit: int | None
    pricing: ModelPricing
    routing_priority: int  # lower = preferred; used by the Beta 2 router
    fallback_model: str | None  # model_id, or None
    license: str
    status: ModelStatus = ModelStatus.AVAILABLE
    is_example_entry: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        """Normalize the enum fields whichever way the entry was built.

        ``from_dict`` coerces these, but direct construction —
        ``ModelEntry(availability="public", ...)`` — did not, and because
        both are ``StrEnum`` the plain string compares equal to the member
        everywhere *except* where ``.value`` is read. That turned a
        perfectly reasonable construction into an ``AttributeError`` at
        serialization time, several layers from the call that caused it.
        Coercing here makes the invariant hold from the moment the entry
        exists, so there is only one shape of ModelEntry in the system.
        """
        if not isinstance(self.availability, ModelAvailabilityFlag):
            self.availability = ModelAvailabilityFlag(self.availability)
        if not isinstance(self.status, ModelStatus):
            self.status = ModelStatus(self.status)
        if isinstance(self.pricing, dict):
            self.pricing = ModelPricing.from_dict(self.pricing)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["availability"] = self.availability.value
        d["status"] = self.status.value
        d["pricing"] = self.pricing.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelEntry:
        return cls(
            model_id=d["model_id"],
            display_name=d["display_name"],
            version=d.get("version", "1.0"),
            total_parameters=float(d["total_parameters"]),
            active_parameters=(
                float(d["active_parameters"]) if d.get("active_parameters") is not None else None
            ),
            architecture=d.get("architecture", "unknown"),
            supported_tasks=list(d.get("supported_tasks", ["chat"])),
            availability=ModelAvailabilityFlag(d.get("availability", "public")),
            minimum_plan=d.get("minimum_plan", "free"),
            free_tier_available=bool(d.get("free_tier_available", False)),
            api_available=bool(d.get("api_available", True)),
            local_available=bool(d.get("local_available", False)),
            remote_available=bool(d.get("remote_available", True)),
            context_limit=int(d.get("context_limit", 8192)),
            input_token_limit=int(d.get("input_token_limit", d.get("context_limit", 8192))),
            output_token_limit=int(d.get("output_token_limit", 2048)),
            tool_call_limit=d.get("tool_call_limit"),
            pricing=ModelPricing.from_dict(d.get("pricing")),
            routing_priority=int(d.get("routing_priority", 100)),
            fallback_model=d.get("fallback_model"),
            license=d.get("license", "unspecified"),
            status=ModelStatus(d.get("status", "available")),
            is_example_entry=bool(d.get("is_example_entry", False)),
            notes=d.get("notes", ""),
        )

    @property
    def is_routable(self) -> bool:
        """True if this entry may ever be selected (manually or by the
        router) — excludes example/disabled/deprecated entries."""
        return self.status in (ModelStatus.AVAILABLE, ModelStatus.BETA)


class ModelRegistry:
    """Thread-safe, in-memory model registry backed by a JSON file.

    This is the *only* place model_id → capability/limit lookups happen.
    Routers, the usage meter, and (in Beta 2) the routing/cascade engine
    must call :meth:`require` rather than trust a caller-supplied model_id.
    """

    def __init__(
        self,
        entries: dict[str, ModelEntry] | None = None,
        *,
        include_examples: bool = False,
    ) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, ModelEntry] = dict(entries or {})
        self._include_examples = include_examples

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        include_examples: bool = False,
    ) -> ModelRegistry:
        """Load a registry, from *path* or from the conventional places.

        With no ``path`` this now **looks for one** before falling back
        to the shipped examples. That fallback is why an operator could
        run ``hypernix-t1 index``, write a perfectly good ``models.json``,
        and still see only entries the file itself documents as not real:
        the server was never told where the registry was, and silently
        served the seed instead of saying so.

        Reading is tolerant. The file is written by another process, so
        the server opens it mid-write in the ordinary course of things;
        a half-flushed file used to end startup with a JSONDecodeError.
        See :func:`read_entries`.
        """
        src = Path(path) if path is not None else discover()
        if src is None:
            src = _EXAMPLE_REGISTRY_PATH
            logger.info(
                "t1api.registry: no registry found in %s; using the example seed. "
                "Run `hypernix-t1 index` to build one from your models.",
                ", ".join(str(p) for p in registry_locations()[:4]),
            )
        if not src.exists():
            logger.warning("t1api.registry: %s not found, starting with an empty registry", src)
            return cls(include_examples=include_examples)

        found, problems = read_entries(src)
        for problem in problems:
            logger.warning("t1api.registry: %s", problem)
        entries = {entry.model_id: entry for entry in found}
        if problems and not entries:
            # Nothing usable. Say it at a level someone will see, because
            # the symptom downstream is an empty model list with no cause.
            logger.error(
                "t1api.registry: %s produced no usable entries (%d problem(s))",
                src, len(problems),
            )
        logger.info("t1api.registry: loaded %d entries from %s", len(entries), src)
        return cls(entries, include_examples=include_examples)

    def to_json_file(self, path: str | Path) -> None:
        with self._lock:
            payload = [e.to_dict() for e in self._entries.values()]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Admin-only mutation (callers must enforce the admin scope check —
    # the registry itself does not know about T1 keys/scopes)
    # ------------------------------------------------------------------

    def register(self, entry: ModelEntry) -> None:
        """Add or replace a registry entry. Admin-only at the API layer."""
        with self._lock:
            self._entries[entry.model_id] = entry
        logger.info("t1api.registry: registered model_id=%s status=%s", entry.model_id, entry.status)

    def deregister(self, model_id: str) -> None:
        with self._lock:
            self._entries.pop(model_id, None)
        logger.info("t1api.registry: deregistered model_id=%s", model_id)

    # ------------------------------------------------------------------
    # Read paths — used by every router and (later) the routing engine
    # ------------------------------------------------------------------

    def _visible(self, entry: ModelEntry) -> bool:
        if entry.is_example_entry and not self._include_examples:
            return False
        return True

    def get(self, model_id: str) -> ModelEntry | None:
        """Return the entry for *model_id*, or None. Does NOT raise —
        use :meth:`require` when an unregistered model should be a hard
        error (e.g. in an API handler)."""
        with self._lock:
            entry = self._entries.get(model_id)
        if entry is None or not self._visible(entry):
            return None
        return entry

    def require(self, model_id: str) -> ModelEntry:
        """Return the entry for *model_id* or raise ``MODEL_NOT_SUPPORTED``.

        This is the enforcement point referenced throughout the spec:
        "Unknown/unregistered models MUST return a stable error such as
        MODEL_NOT_SUPPORTED."
        """
        entry = self.get(model_id)
        if entry is None:
            raise T1APIError(
                T1ErrorCode.MODEL_NOT_SUPPORTED,
                f"Model '{model_id}' is not registered in the T1 model registry.",
                details={"model_id": model_id},
                http_status=404,
            )
        return entry

    def list(
        self,
        *,
        status: ModelStatus | None = None,
        plan: str | None = None,
        routable_only: bool = False,
    ) -> list[ModelEntry]:
        with self._lock:
            results = [e for e in self._entries.values() if self._visible(e)]
        if status is not None:
            results = [e for e in results if e.status == status]
        if routable_only:
            results = [e for e in results if e.is_routable]
        if plan is not None:
            results = [e for e in results if e.minimum_plan == plan or e.free_tier_available]
        results.sort(key=lambda e: (e.routing_priority, e.model_id))
        return results

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values() if self._visible(e))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ModelRegistry(entries={len(self)}, include_examples={self._include_examples})"


__all__ = [
    "ModelStatus",
    "ModelAvailabilityFlag",
    "ModelPricing",
    "ModelEntry",
    "ModelRegistry",
]
