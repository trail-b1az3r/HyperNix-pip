"""hypernix.optimizer_framework_v2 — one interface over every HyperNix optimizer.

Before this, using an optimizer meant knowing which module it lived in and
how that generation spelled its arguments (``peak_lr`` for V1, a
``ScheduleConfig`` for V4, ``lr`` for the rest). :class:`OptimizerFrameworkV2`
is the one place that knows:

``OptimizerFrameworkV2.names()``
    every optimizer by its key.
``OptimizerFrameworkV2.create(name, params, lr=..., **options)``
    construct one, with ``lr`` and ``weight_decay`` meaning the same thing
    whichever generation it is. Options a generation does not have are
    refused by name, not dropped.
``OptimizerFrameworkV2.auto_select(...)``
    recommend one for a model, a device and a goal, with a machine-readable
    reason and a sentence, and a note for every candidate passed over.
``OptimizerFrameworkV2.diagnostics(optimizer)``
    what an optimizer is doing: its variant, its state in measured bytes,
    its kernel, its last stability score.
``OptimizerFrameworkV2.table()``
    the comparison matrix. Its speed and memory columns are *targets* in
    words until a benchmark record is passed in; then they are numbers.

Nothing here replaces an optimizer's own class or changes its update rule:
the existing classes keep working exactly as they did, and this module
imports each one only when it is asked for.

Selection is by what is known, never by version number. In particular
V7Mini is picked automatically only for a parameter count that was
*verified* (counted from the parameters, or stated by the caller as
verified) and that is strictly between 39,000,000 and 750,000,000. An
unknown count, a guessed one, exactly 39M and exactly 750M all fall
outside it.
"""
from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

__all__ = [
    "OptimizerFrameworkV2",
    "OptimizerSpec",
    "Selection",
    "OPTIMIZERS",
    "OBJECTIVES",
    "FEATURES",
]

MINI_MIN_PARAMS = 39_000_000
MINI_MAX_PARAMS = 750_000_000

#: What a caller can say they are optimising for.
OBJECTIVES = ("balanced", "throughput", "memory", "stability", "quantization", "compatibility",
              "legacy")

#: Features a caller can require. An optimizer that lacks one is not chosen.
FEATURES = ("qat", "mtp", "ema", "quantization_sensitivity", "mixed_precision_per_layer",
            "model_preparation", "factored_state", "quantized_state", "cuda_graph", "compile",
            "grad_scaler", "grad_accumulation", "trust_ratio", "adaptive_pressure",
            "stability_score", "pressure_diffusion", "directional_trust_region",
            "gradient_norm_monitoring", "nonfinite_skip")


@dataclass(frozen=True)
class OptimizerSpec:
    """One optimizer: where it lives, what it offers, and what it is for.

    The four ratings are the design's intent in words, not measurements.
    """

    key: str
    label: str
    module: str
    class_name: str
    family: str                       # "torch", "legacy", "v5", "v6", "v7"
    speed: str
    memory: str
    stability: str
    adaptivity: str
    best_use: str
    features: frozenset[str] = frozenset()
    requires_cuda: bool = False
    legacy: bool = False
    #: How this generation spells the learning rate.
    lr_argument: str = "lr"
    defaults: tuple[tuple[str, Any], ...] = ()
    aliases: tuple[str, ...] = ()
    note: str = ""

    def load(self) -> type:
        return getattr(importlib.import_module(self.module), self.class_name)


_V7_COMMON = frozenset({"factored_state", "quantized_state", "grad_scaler", "grad_accumulation",
                        "trust_ratio", "adaptive_pressure", "stability_score",
                        "pressure_diffusion", "ema", "nonfinite_skip",
                        "gradient_norm_monitoring"})
_V7M_EXTRA = frozenset({"qat", "mtp", "quantization_sensitivity", "mixed_precision_per_layer",
                        "model_preparation", "directional_trust_region"})

_SPECS = [
    OptimizerSpec("adamw", "AdamW", "torch.optim", "AdamW", "torch",
                  "Baseline", "High", "High", "High", "General compatibility",
                  frozenset({"grad_scaler"}), aliases=("adam",)),
    OptimizerSpec("v1", "Pressure Cooker V1", "hypernix.optimizers.pressure_cooker",
                  "PressureCooker", "legacy", "Legacy", "High", "Legacy", "Low", "Legacy",
                  legacy=True, lr_argument="peak_lr", aliases=("pressure-cooker",)),
    OptimizerSpec("v2", "Pressure Cooker V2 (device tiers)",
                  "hypernix.optimizers.pressure_cooker", "ElectricCooker", "legacy",
                  "Legacy", "Medium", "Legacy", "Medium", "Legacy", legacy=True,
                  lr_argument="peak_lr",
                  note="The device-tier generation; on CUDA its InductionCooker is used."),
    OptimizerSpec("v3", "Pressure Cooker V3", "hypernix.optimizers.pressure_cooker_v3",
                  "PressureCookerV3", "legacy", "Legacy", "Medium", "Improved", "Medium",
                  "Legacy", legacy=True),
    OptimizerSpec("v4", "Pressure Cooker V4", "hypernix.optimizers.pressure_cooker_v4",
                  "PressureCookerV4", "legacy", "Legacy", "Medium", "Improved", "Medium",
                  "Legacy", frozenset({"ema"}), legacy=True),
    OptimizerSpec("v5", "Pressure Cooker V5", "hypernix.optimizers.pressure_cooker_v5",
                  "PressureCookerV5", "v5", "Low/Medium", "Very Low", "High", "Specialized",
                  "Memory-constrained training",
                  frozenset({"quantized_state", "factored_state", "qat", "mtp", "ema",
                             "trust_ratio"})),
    OptimizerSpec("v5plus", "Pressure Cooker V5+", "hypernix.optimizers.pressure_cooker_v5",
                  "PressureCookerV5Plus", "v5", "Medium", "Very Low", "High", "High",
                  "Quantization-aware training",
                  frozenset({"quantized_state", "factored_state", "qat", "mtp", "ema",
                             "trust_ratio", "model_preparation", "quantization_sensitivity",
                             "mixed_precision_per_layer", "gradient_norm_monitoring",
                             "directional_trust_region"}),
                  aliases=("v5+",),
                  note="Constructing it turns 6-bit QAT on unless told otherwise."),
    OptimizerSpec("v5s", "Pressure Cooker V5S", "hypernix.optimizers.pressure_cooker_v5s",
                  "PressureCookerV5S", "v5", "Medium/Low", "Extremely Low", "Very High",
                  "Specialized", "Stability + memory",
                  frozenset({"quantized_state", "factored_state", "pressure_diffusion",
                             "stability_score", "trust_ratio", "ema"})),
    OptimizerSpec("v6", "Pressure Cooker V6", "hypernix.optimizers.pressure_cooker_v6",
                  "PressureCookerV6", "v6", "Very High", "Medium", "Medium", "Low",
                  "Maximum throughput",
                  frozenset({"grad_scaler", "grad_accumulation", "trust_ratio",
                             "nonfinite_skip"})),
    OptimizerSpec("v6v", "Pressure Cooker V6V", "hypernix.optimizers.pressure_cooker_v6v",
                  "PressureCookerV6V", "v6", "Extremely High*", "Medium", "Medium", "Low",
                  "CUDA graph workloads",
                  frozenset({"grad_scaler", "grad_accumulation", "trust_ratio", "cuda_graph",
                             "compile", "nonfinite_skip"}),
                  requires_cuda=True, note="* Hardware and workload dependent."),
    OptimizerSpec("v7", "Pressure Cooker V7 Balanced",
                  "hypernix.optimizers.pressure_cooker_v7", "PressureCookerV7", "v7",
                  "High target", "Low target", "High target", "Medium/High target",
                  "General default and balanced workloads", _V7_COMMON,
                  aliases=("v7-balanced", "v7balanced", "balanced")),
    OptimizerSpec("v7v", "Pressure Cooker V7V", "hypernix.optimizers.pressure_cooker_v7v",
                  "PressureCookerV7V", "v7", "High target", "Low target", "High target",
                  "Medium/High target", "V7 with torch.compile and CUDA graphs where available",
                  _V7_COMMON | {"cuda_graph", "compile"},
                  note="Runs anywhere; the accelerated tiers need CUDA."),
    OptimizerSpec("v7mini", "Pressure Cooker V7Mini",
                  "hypernix.optimizers.pressure_cooker_v7", "PressureCookerV7Mini", "v7",
                  "Tuned for small-model throughput", "Low overhead target",
                  "Benchmark required", "Architecture-dependent",
                  "Models strictly above 39M and below 750M parameters", _V7_COMMON,
                  aliases=("v7-mini", "mini")),
    OptimizerSpec("v7q", "Pressure Cooker V7Q", "hypernix.optimizers.pressure_cooker_v7",
                  "PressureCookerV7Q", "v7", "Fastest variant target", "Low/medium target",
                  "Benchmark required", "Lower adaptivity is acceptable if documented",
                  "Maximum throughput where measured faster",
                  frozenset({"grad_scaler", "grad_accumulation", "trust_ratio"}),
                  aliases=("v7-q", "q"),
                  note="No stability tracking, pressure, clipping or non-finite check by "
                       "default."),
    OptimizerSpec("v7s", "Pressure Cooker V7S", "hypernix.optimizers.pressure_cooker_v7",
                  "PressureCookerV7S", "v7", "Benchmark required", "Very low state target",
                  "High target", "Specialized",
                  "Modernized V5S-style stability and memory efficiency", _V7_COMMON,
                  aliases=("v7-s", "s")),
    OptimizerSpec("v7m", "Pressure Cooker V7M", "hypernix.optimizers.pressure_cooker_v7",
                  "PressureCookerV7M", "v7", "Benchmark required",
                  "Lower than equivalent V5/V5+ target", "High target", "High target",
                  "Modernized V5/V5+ features, QAT/MTP and sensitivity-aware training",
                  _V7_COMMON | _V7M_EXTRA, aliases=("v7-m", "m"),
                  note="QAT, MTP and model preparation are explicit calls in "
                       "hypernix.optimizers.v7m_features; none is on by default."),
]

#: Every optimizer, by key.
OPTIMIZERS: dict[str, OptimizerSpec] = {spec.key: spec for spec in _SPECS}
_ALIASES: dict[str, str] = {alias: spec.key for spec in _SPECS for alias in spec.aliases}


@dataclass
class Selection:
    """What :meth:`OptimizerFrameworkV2.auto_select` chose, and why.

    ``reason_code`` is stable and meant for programs; ``reason`` is the
    sentence. ``considered`` says, for each candidate passed over, why.
    """

    key: str
    label: str
    reason_code: str
    reason: str
    objective: str = "balanced"
    explicit: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    considered: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    parameter_count: int | None = None
    parameter_count_verified: bool = False
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(name: str) -> OptimizerSpec:
    key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    key = _ALIASES.get(key, key).replace("-", "")
    key = {"v5+": "v5plus"}.get(key, key)
    if key not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}. Known: {', '.join(OPTIMIZERS)}")
    return OPTIMIZERS[key]


def _count(params: Iterable[Any] | None) -> tuple[int | None, str, torch.dtype | None]:
    """Trainable parameter count, device and dtype, counted from the tensors."""
    if params is None:
        return None, "cpu", None
    total, device, dtype = 0, "cpu", None
    seen = False
    for item in params:
        tensors = item["params"] if isinstance(item, dict) else [item]
        for p in tensors:
            if not getattr(p, "requires_grad", True):
                continue
            total += p.numel()
            if not seen:
                device, dtype, seen = str(p.device), p.dtype, True
    return (total if seen else None), device, dtype


class OptimizerFrameworkV2:
    """One interface over AdamW, Pressure Cooker V1-V6V and the V7 family."""

    # -- the catalogue -------------------------------------------------------

    @staticmethod
    def names(*, include_legacy: bool = True) -> list[str]:
        return [k for k, s in OPTIMIZERS.items() if include_legacy or not s.legacy]

    @staticmethod
    def spec(name: str) -> OptimizerSpec:
        return _resolve(name)

    @staticmethod
    def table(measured: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """The comparison matrix, one row per optimizer.

        The ratings are the design's targets in words. Pass *measured*
        (``{key: {"ms_per_step": ..., "state_bytes_per_parameter": ...,
        "final_loss": ...}}``, the shape ``hnx optimizer benchmark`` writes)
        and each row gains those numbers under ``measured``; a row without a
        measurement says so rather than showing a target as if it were one.
        """
        rows = []
        for key, spec in OPTIMIZERS.items():
            row = {"key": key, "optimizer": spec.label, "speed": spec.speed,
                   "memory": spec.memory, "stability": spec.stability,
                   "adaptivity": spec.adaptivity, "best_use": spec.best_use,
                   "ratings_are": "targets", "note": spec.note,
                   "measured": dict((measured or {}).get(key) or {})}
            rows.append(row)
        return rows

    # -- construction ----------------------------------------------------------

    @staticmethod
    def create(name: str, params: Iterable[Any], *, lr: float = 3e-4,
               weight_decay: float | None = None, **options: Any) -> torch.optim.Optimizer:
        """Construct optimizer *name*.

        ``lr`` and ``weight_decay`` mean the same thing for every
        generation. *options* go to the optimizer's own constructor; one it
        does not accept raises ``TypeError`` naming it and the optimizer,
        rather than being ignored.
        """
        spec = _resolve(name)
        materialized = list(params)
        _count_now, device, _dtype = _count(materialized)
        if spec.requires_cuda and not device.startswith("cuda"):
            raise ValueError(f"{spec.label} needs CUDA parameters; these are on {device}. "
                             "Move the model to a CUDA device, or use "
                             + ("v7v, which runs anywhere" if spec.key == "v6v" else "v7") + ".")
        cls = spec.load()
        if spec.key == "v2" and device.startswith("cuda"):
            cls = importlib.import_module(spec.module).InductionCooker
        kwargs: dict[str, Any] = dict(spec.defaults)
        kwargs[spec.lr_argument] = lr
        if weight_decay is not None:
            kwargs["weight_decay"] = weight_decay
        kwargs.update(options)
        try:
            return cls(materialized, **kwargs)
        except TypeError as exc:
            raise TypeError(f"{spec.label} ({spec.key}) does not accept these options: {exc}") \
                from exc

    @staticmethod
    def build(selection: Selection, params: Iterable[Any], *, lr: float = 3e-4,
              **options: Any) -> torch.optim.Optimizer:
        """Construct what :meth:`auto_select` chose, with the options it chose."""
        merged = {**selection.options, **options}
        optimizer = OptimizerFrameworkV2.create(selection.key, params, lr=lr, **merged)
        optimizer._framework_selection = selection       # for diagnostics and run metadata
        return optimizer

    # -- selection -------------------------------------------------------------

    @staticmethod
    def auto_select(
        params: Iterable[Any] | None = None,
        *,
        objective: str = "balanced",
        requested: str | None = None,
        parameter_count: int | None = None,
        parameter_count_verified: bool = False,
        device: str | torch.device | None = None,
        required_features: Iterable[str] = (),
        free_memory_bytes: int | None = None,
        fixed_shapes: bool = False,
        compile_available: bool | None = None,
        measured: dict[str, dict[str, Any]] | None = None,
    ) -> Selection:
        """Recommend an optimizer.

        Args:
            params: The parameters to be trained. When given, the count,
                device and dtype are read from them, and the count is
                verified.
            objective: One of :data:`OBJECTIVES`.
            requested: An optimizer the user asked for. Honoured when it is
                compatible; refused with the reason when it is not. Never
                silently replaced.
            parameter_count: The trainable parameter count, when *params*
                is not given.
            parameter_count_verified: Whether that count was verified
                against the instantiated model. Only a verified count can
                select V7Mini.
            device: Where training runs, when *params* is not given.
            required_features: Names from :data:`FEATURES` the optimizer
                must have.
            free_memory_bytes: Free memory on the training device, when
                known. Fp32 momentum that would take more than a quarter of
                it counts as memory pressure.
            fixed_shapes: Whether every step has the same tensor shapes
                (a CUDA graph needs that).
            compile_available: Whether ``torch.compile`` works here
                (default: whether torch has it).
            measured: Benchmark results for this hardware, in the shape
                :meth:`table` takes. V7Q is recommended for throughput only
                where these show it faster than V6, or where there are none
                to say otherwise (and then the selection says it is
                unmeasured).

        The same inputs always give the same answer.
        """
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}, "
                             f"not {objective!r}")
        required = [str(f) for f in required_features]
        unknown = [f for f in required if f not in FEATURES]
        if unknown:
            raise ValueError(f"unknown feature(s) {unknown}. Known: {', '.join(FEATURES)}")

        counted, counted_device, _dtype = _count(params)
        if counted is not None:
            count, verified, where = counted, True, counted_device
        else:
            count, verified = parameter_count, bool(parameter_count_verified and
                                                    parameter_count is not None)
            where = str(device) if device is not None else "cpu"
        on_cuda = where.startswith("cuda")
        if compile_available is None:
            compile_available = hasattr(torch, "compile")
        considered: list[dict[str, str]] = []
        notes: list[str] = []

        def supports(key: str) -> str:
            """'' when *key* can do what is required here, else why not."""
            spec = OPTIMIZERS[key]
            missing = [f for f in required if f not in spec.features]
            if missing:
                return f"lacks {', '.join(missing)}"
            if spec.requires_cuda and not on_cuda:
                return "needs CUDA"
            return ""

        def choose(key: str, code: str, reason: str, **options: Any) -> Selection:
            spec = OPTIMIZERS[key]
            return Selection(key=key, label=spec.label, reason_code=code, reason=reason,
                             objective=objective, explicit=False, options=options,
                             considered=considered, notes=notes, parameter_count=count,
                             parameter_count_verified=verified, device=where)

        def passed(key: str, why: str) -> None:
            considered.append({"key": key, "why_not": why})

        # 1. An explicit choice is honoured, or refused. Never substituted.
        if requested:
            spec = _resolve(requested)
            problem = supports(spec.key)
            if problem:
                raise ValueError(f"{spec.label} was asked for but cannot be used here: "
                                 f"{problem}. Choose another optimizer or drop the "
                                 "requirement; nothing was substituted.")
            selection = choose(spec.key, "explicit", f"{spec.label} was asked for.")
            selection.explicit = True
            if spec.key == "v7mini" and not (verified and count is not None
                                             and MINI_MIN_PARAMS < count < MINI_MAX_PARAMS):
                shown = f"{count:,}" if count is not None else "an unknown number of"
                notes.append(f"V7Mini is designed for more than {MINI_MIN_PARAMS:,} and fewer "
                             f"than {MINI_MAX_PARAMS:,} verified trainable parameters; this "
                             f"model has {shown}, where its advantage is not established.")
            return selection

        memory_pressure = bool(free_memory_bytes and count
                               and count * 4 > free_memory_bytes // 4)
        if memory_pressure:
            notes.append("fp32 momentum would take more than a quarter of the free memory")

        # Features only the V5/V5+ lineage has decide first: nothing else can do them.
        lineage = [f for f in required if f in _V7M_EXTRA]
        if lineage or objective == "quantization":
            if not supports("v7m"):
                return choose("v7m", "v5_lineage_features",
                              "V7M has the V5/V5+ features asked for"
                              + (f" ({', '.join(lineage)})" if lineage else "")
                              + "; each is an explicit call and none is on by default.")
            passed("v7m", supports("v7m"))
            if not supports("v5plus"):
                return choose("v5plus", "v5_lineage_fallback",
                              "V5+ has features V7M cannot provide here.")
            passed("v5plus", supports("v5plus"))

        if objective == "legacy":
            return choose("v4", "legacy", "Legacy compatibility was asked for; V4 is the "
                          "newest of the legacy generations.")
        if objective == "compatibility":
            if not supports("adamw"):
                return choose("adamw", "compatibility",
                              "AdamW is the most widely understood and compatible choice.")
            passed("adamw", supports("adamw"))

        # 2. V7Mini, for a verified count strictly inside its band.
        if objective == "balanced":
            if count is None:
                passed("v7mini", "the parameter count is unknown")
            elif not verified:
                passed("v7mini", "the parameter count is not verified")
            elif not MINI_MIN_PARAMS < count < MINI_MAX_PARAMS:
                passed("v7mini", f"{count:,} parameters is outside the band "
                                 f"({MINI_MIN_PARAMS:,} < n < {MINI_MAX_PARAMS:,}, strict)")
            elif supports("v7mini"):
                passed("v7mini", supports("v7mini"))
            elif memory_pressure:
                passed("v7mini", "memory is short; a smaller-state variant fits better")
            else:
                return choose("v7mini", "mini_band",
                              f"{count:,} verified trainable parameters is inside V7Mini's "
                              f"band ({MINI_MIN_PARAMS:,} < n < {MINI_MAX_PARAMS:,}).")

        # 3. Throughput.
        if objective == "throughput":
            if fixed_shapes and on_cuda and compile_available and not supports("v6v"):
                notes.append("fixed shapes on CUDA: V6V's CUDA graph is also a candidate; "
                             "benchmark both")
            v7q, v6 = (measured or {}).get("v7q", {}), (measured or {}).get("v6", {})
            if not supports("v7q"):
                if v7q.get("ms_per_step") and v6.get("ms_per_step"):
                    if v7q["ms_per_step"] < v6["ms_per_step"]:
                        return choose("v7q", "throughput_measured",
                                      f"V7Q measured {v7q['ms_per_step']:.2f} ms/step against "
                                      f"V6's {v6['ms_per_step']:.2f} on this hardware. It has "
                                      "no stability tracking and less adaptivity than "
                                      "Balanced.")
                    passed("v7q", f"V6 measured faster here ({v6['ms_per_step']:.2f} against "
                                  f"{v7q['ms_per_step']:.2f} ms/step)")
                else:
                    return choose("v7q", "throughput_unmeasured",
                                  "V7Q is the throughput-first variant. No benchmark for this "
                                  "hardware was given, so that it is the fastest here is "
                                  "unmeasured: run `hnx optimizer benchmark`. It has no "
                                  "stability tracking and less adaptivity than Balanced.")
            else:
                passed("v7q", supports("v7q"))
            if not supports("v6"):
                return choose("v6", "throughput_v6", "V6 is the established throughput choice.")
            passed("v6", supports("v6"))

        # 4. Stability and very low state.
        if objective in ("stability", "memory") or memory_pressure:
            if not supports("v7s") and objective == "stability":
                return choose("v7s", "stability",
                              "V7S keeps V5S's stability mechanisms and one byte of momentum "
                              "per parameter, on the V7 engine.")
            if not supports("v7"):
                return choose("v7", "memory",
                              "V7's memory preset: int8 momentum and factored statistics, "
                              "about one byte of state per parameter."
                              + (" Chosen because memory is short." if memory_pressure
                                 and objective != "memory" else ""),
                              preset="memory")
            passed("v7", supports("v7"))
            if not supports("v5s"):
                return choose("v5s", "memory_fallback",
                              "V5S, the established smallest-state optimizer.")
            passed("v5s", supports("v5s"))

        # 6. The default.
        if not supports("v7"):
            return choose("v7", "balanced",
                          "V7 Balanced: the general default. Nothing known about this run "
                          "calls for a specialised optimizer.")
        passed("v7", supports("v7"))

        # 7. Established fallbacks, when a requirement rules V7 out.
        for key, code, reason in (
                ("v6v", "fallback_v6v", "V6V supports what was required and V7 does not."),
                ("v6", "fallback_v6", "V6 supports what was required and V7 does not."),
                ("v5plus", "fallback_v5plus", "V5+ supports what was required."),
                ("adamw", "fallback_adamw", "AdamW, for broad compatibility.")):
            if not supports(key):
                return choose(key, code, reason)
            passed(key, supports(key))
        raise ValueError(f"no optimizer here supports all of: {', '.join(required)} on {where}")

    # -- diagnostics -----------------------------------------------------------

    @staticmethod
    def diagnostics(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """What *optimizer* is and is doing, in one dictionary.

        State bytes are counted from the state tensors, for any optimizer.
        """
        described = optimizer.describe() if hasattr(optimizer, "describe") else {}
        state = sum(v.numel() * v.element_size() for s in optimizer.state.values()
                    for v in s.values() if isinstance(v, torch.Tensor))
        if hasattr(optimizer, "state_bytes"):
            state = int(optimizer.state_bytes())
        params = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        cls = type(optimizer)
        key = next((k for k, s in OPTIMIZERS.items()
                    if s.class_name == cls.__name__ and s.module == cls.__module__), "")
        selection = getattr(optimizer, "_framework_selection", None)
        out = {"key": key, "class": cls.__name__, "parameters": params, "state_bytes": state,
               "state_bytes_per_parameter": round(state / max(1, params), 4),
               "selection": selection.to_dict() if selection is not None else None}
        out.update({k: v for k, v in described.items() if k not in out})
        return out
