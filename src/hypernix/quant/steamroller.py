"""hypernix.quant.steamroller — the descending quantiser.

llama.cpp will happily quantise FP16 straight to a 1-bit IQ type. The
result is usually rubbish, and the reason is not the final tier: it is
that a single pass has to choose every group scale from the full-precision
distribution at once, and at one or two bits there are not enough levels
left for that choice to be recoverable.

Steamroller does it in stages. Roll the weights down to an intermediate
tier, let the scales settle there, then roll again from the flattened
distribution. Hence the name, and hence the shape of the ladder:

    FP32 / BF16 / FP16 / FP8 / Q8_0
                  |
                  v
                Q3_K_L          <- the staging tier, always
                  |
        +---------+---------+-----------+
        v         v         v           v
      IQ1_M   IQ0.9_L   IQ0.75_M   IQ0.5_XXXL

Every descent passes through Q3_K_L. That is the whole trick and it is
not free — two passes cost roughly twice one pass — so :func:`plan` will
tell you when the staging step is pointless (a Q8_0 target does not need
it) and skip it.

The sub-1-bit tiers, honestly
-----------------------------
``IQ0.9_L``, ``IQ0.75_M`` and ``IQ0.5_XXXL`` are **HyperNix extension
types, not upstream llama.cpp quant types**. Stock ``llama-quantize`` has
never heard of them, and a GGUF written with one will not load in an
unpatched llama.cpp. They are produced by writing a Q3_K_L intermediate
and then applying HyperNix's own sub-bit packing, and they are marked in
the file's metadata so a loader can refuse them cleanly instead of
reading garbage.

They are also, at these bitrates, lossy in a way that is qualitatively
different from Q4_K_M. Below about 1.5 bits per weight a model stops
being a slightly worse version of itself and starts being a different,
much worse model. :attr:`Tier.honest_warning` says so on every tier where
it is true, and :func:`plan` surfaces it. Shipping a 0.5-bit model
without that warning would be the actual failure here.

Nothing in this module invents accuracy. It stages, it packs, it reports.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Tier",
    "TIERS",
    "SOURCE_FORMATS",
    "STAGING_TIER",
    "SteamrollerError",
    "SteamrollPlan",
    "PlanStep",
    "plan",
    "Steamroller",
    "get_tier",
]

#: What steamroller will accept as input. Anything wider than the staging
#: tier, which is every float format and the widest GGUF tier.
SOURCE_FORMATS: tuple[str, ...] = ("FP32", "BF16", "FP16", "FP8", "Q8_0")

#: Every descent lands here first. See the module docstring.
STAGING_TIER = "Q3_K_L"


class SteamrollerError(RuntimeError):
    """A quantisation run failed, with a code a caller can branch on.

    ``code`` is one of: ``missing_binary``, ``missing_source``,
    ``bad_source_format``, ``unknown_tier``, ``quantize_failed``,
    ``pack_failed``, ``verify_failed``.
    """

    def __init__(self, message: str, *, code: str = "error", hint: str = ""):
        super().__init__(message)
        self.code = code
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "hint": self.hint}


@dataclass(frozen=True)
class Tier:
    """One rung of the ladder."""

    name: str
    bits_per_weight: float
    #: True for the tiers stock llama.cpp can write. False for the
    #: HyperNix extension types — see the module docstring.
    upstream: bool
    summary: str
    #: The llama-quantize enum name, when there is one.
    llama_type: str = ""
    #: Sub-bit packing applied after the staging pass. Empty for upstream tiers.
    packing: str = ""
    #: Set when this tier is lossy enough that a caller should be told.
    honest_warning: str = ""
    #: Below this bitrate an importance matrix stops being optional.
    needs_imatrix: bool = False

    @property
    def is_extension(self) -> bool:
        return not self.upstream

    def estimate_bytes(self, parameters: int) -> int:
        return int(parameters * self.bits_per_weight / 8)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bits_per_weight": self.bits_per_weight,
            "upstream": self.upstream,
            "llama_type": self.llama_type,
            "packing": self.packing,
            "needs_imatrix": self.needs_imatrix,
            "summary": self.summary,
            "honest_warning": self.honest_warning,
        }


_SUB_BIT_WARNING = (
    "Below ~1.5 bits per weight a model does not degrade gracefully — it stops being a "
    "worse version of itself and becomes a different, much worse model. Evaluate on your "
    "own task before shipping this."
)

_EXTENSION_WARNING = (
    "This is a HyperNix extension type, not an upstream llama.cpp quant type. Stock "
    "llama.cpp will refuse the resulting GGUF; it needs a HyperNix-aware loader."
)

_QUARTER_WARNING = (
    "At a quarter of a bit roughly two signs in five are wrong and there is no "
    "magnitude at all. Treat this as a measurement of how far the packing goes, "
    "not as a model to deploy."
)

TIERS: dict[str, Tier] = {
    t.name: t
    for t in (
        Tier("Q8_0", 8.5, True, "8-bit GGUF. Effectively lossless.", llama_type="Q8_0"),
        Tier("Q3_K_L", 3.44, True,
             "3-bit k-quant, large. The staging tier every descent passes through.",
             llama_type="Q3_K_L"),
        Tier("IQ1_M", 1.75, True,
             "1-bit IQ, medium. The narrowest tier stock llama.cpp can write.",
             llama_type="IQ1_M", needs_imatrix=True,
             honest_warning=_SUB_BIT_WARNING),
        Tier("IQ0.9_L", 0.90, False,
             "Sub-1-bit, large groups. HyperNix extension type.",
             packing="hypernix-sub1-l", needs_imatrix=True,
             honest_warning=_SUB_BIT_WARNING + " " + _EXTENSION_WARNING),
        Tier("IQ0.75_M", 0.75, False,
             "Sub-1-bit, medium groups. HyperNix extension type.",
             packing="hypernix-sub1-m", needs_imatrix=True,
             honest_warning=_SUB_BIT_WARNING + " " + _EXTENSION_WARNING),
        Tier("IQ0.5_XXXL", 0.50, False,
             "Half a bit per weight, very large groups. The bottom of the ladder "
             "and an experiment, not a deployment target.",
             packing="hypernix-sub1-xxxl", needs_imatrix=True,
             honest_warning=_SUB_BIT_WARNING + " " + _EXTENSION_WARNING),
        Tier("IQ0.25_UXL", 0.25, False,
             "A quarter of a bit. Three signs kept of every sixteen, the other "
             "thirteen repeating the third — so about 59% of signs survive, "
             "against the 50% a coin gets.",
             packing="hypernix-sub1-uxl", needs_imatrix=True,
             honest_warning=_QUARTER_WARNING + " " + _EXTENSION_WARNING),
        # The fixed-codebook family. Named for the width of one weight,
        # the way Q4_0 is, so the rate is the name plus the FP16 block
        # scale -- stated here rather than left to a file size.
        Tier("INT1", 1.0625, False,
             "One bit per weight and a block scale: every sign kept, no "
             "magnitude. A binary net, and the k == g end of the sub-bit family.",
             packing="hypernix-int1",
             honest_warning=_SUB_BIT_WARNING + " " + _EXTENSION_WARNING),
        Tier("HNX_1375BIT", 1.375, False,
             "Every sign kept and sixteen magnitudes per block: 5 bits of "
             "sub-block scale over 16 weights each, on top of the FP16 block "
             "scale. The step above INT1, which reconstructs all 256 weights "
             "at a single magnitude.",
             packing="hypernix-1375",
             honest_warning=_SUB_BIT_WARNING + " " + _EXTENSION_WARNING),
        Tier("FP2", 2.0625, False,
             "Two-bit float: sign and exponent, levels ±1 and ±2, no zero. "
             "Block scale searched, because the obvious fit made it lose to INT1.",
             packing="hypernix-fp2",
             honest_warning=_EXTENSION_WARNING),
        Tier("INT4", 4.0625, False,
             "Signed 4-bit over a fixed codebook with a searched block scale. "
             "No k-quant machinery — Q4_K_M is better at this rate and any "
             "llama.cpp can read it; this is for when the codebook has to be "
             "exactly the integers.",
             packing="hypernix-int4",
             honest_warning=_EXTENSION_WARNING),
    )
}

_TIER_ALIASES = {
    "q8_0": "Q8_0", "q8": "Q8_0",
    "q3_k_l": "Q3_K_L", "q3l": "Q3_K_L", "q3kl": "Q3_K_L",
    "iq1_m": "IQ1_M", "iq1m": "IQ1_M",
    "iq0.9_l": "IQ0.9_L", "iq0.9l": "IQ0.9_L", "iq09l": "IQ0.9_L",
    "iq0.75_m": "IQ0.75_M", "iq0.75m": "IQ0.75_M", "iq075m": "IQ0.75_M",
    "iq0.5_xxxl": "IQ0.5_XXXL", "iq0.5xxxl": "IQ0.5_XXXL", "iq05xxxl": "IQ0.5_XXXL",
    # The user-facing spelling of this one is IQ-0.25UXL as often as
    # IQ0.25_UXL, so both resolve. get_tier() also strips underscores,
    # which covers the rest of the permutations.
    "iq0.25_uxl": "IQ0.25_UXL", "iq0.25uxl": "IQ0.25_UXL",
    "iq025uxl": "IQ0.25_UXL", "iq.0.25uxl": "IQ0.25_UXL",
    "int1": "INT1", "int4": "INT4", "fp2": "FP2",
    "i1": "INT1", "i4": "INT4", "f2": "FP2",
    # The rate has a decimal point in it and people write it every way
    # there is. get_tier() lower-cases and strips dashes and underscores,
    # so these cover the rest.
    "hnx_1375bit": "HNX_1375BIT", "hnx1375bit": "HNX_1375BIT",
    "hnx_1375": "HNX_1375BIT", "hnx1375": "HNX_1375BIT",
    "hnx_1.375bit": "HNX_1375BIT", "hnx1.375bit": "HNX_1375BIT",
    "1375bit": "HNX_1375BIT", "1.375": "HNX_1375BIT",
}


def get_tier(name: str) -> Tier:
    key = (name or "").strip().lower().replace("-", "_")
    canonical = _TIER_ALIASES.get(key) or _TIER_ALIASES.get(key.replace("_", ""))
    if canonical is None and name in TIERS:
        canonical = name
    if canonical is None:
        raise SteamrollerError(
            f"Unknown target tier {name!r}. Available: {', '.join(TIERS)}",
            code="unknown_tier",
        )
    return TIERS[canonical]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """One pass of the roll."""

    index: int
    kind: str                 # "quantize" | "pack"
    source: str
    target: str
    llama_type: str = ""
    packing: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "source": self.source,
            "target": self.target,
            "llama_type": self.llama_type,
            "packing": self.packing,
            "reason": self.reason,
        }


@dataclass
class SteamrollPlan:
    """What :class:`Steamroller` is going to do, before it does it."""

    source_format: str
    target: Tier
    steps: list[PlanStep]
    warnings: list[str] = field(default_factory=list)
    parameters: int = 0

    @property
    def staged(self) -> bool:
        """True when the plan goes via :data:`STAGING_TIER`."""
        return any(s.target == STAGING_TIER for s in self.steps)

    @property
    def estimated_bytes(self) -> int:
        return self.target.estimate_bytes(self.parameters) if self.parameters else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "target": self.target.to_dict(),
            "staged": self.staged,
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "warnings": list(self.warnings),
            "parameters": self.parameters,
            "estimated_bytes": self.estimated_bytes,
        }

    def describe(self) -> str:
        arrow = " -> ".join([self.source_format] + [s.target for s in self.steps])
        return f"{arrow}  ({len(self.steps)} pass{'es' if len(self.steps) != 1 else ''})"


def plan(
    source_format: str,
    target: str,
    *,
    parameters: int = 0,
    have_imatrix: bool = False,
    force_staging: bool | None = None,
) -> SteamrollPlan:
    """Work out the passes needed to get from *source_format* to *target*.

    ``force_staging`` overrides the staging decision in both directions:
    ``True`` stages even when it is pointless, ``False`` skips it even
    when it helps. Left as ``None`` the rule is the one in the module
    docstring — stage whenever the target is narrower than the staging
    tier, because that is exactly when a single pass has too few levels
    left to choose good scales.
    """
    source = (source_format or "").strip().upper().replace("-", "_")
    if source not in SOURCE_FORMATS:
        raise SteamrollerError(
            f"{source_format!r} is not a supported source. Steamroller rolls downhill from "
            f"{', '.join(SOURCE_FORMATS)}.",
            code="bad_source_format",
            hint="Convert to FP16 GGUF first with `hypernix convert`.",
        )
    tier = get_tier(target)
    staging = TIERS[STAGING_TIER]

    warnings: list[str] = []
    steps: list[PlanStep] = []

    should_stage = (
        tier.bits_per_weight < staging.bits_per_weight
        if force_staging is None
        else force_staging
    )
    if force_staging is True and tier.bits_per_weight >= staging.bits_per_weight:
        warnings.append(
            f"Staging through {STAGING_TIER} was forced, but {tier.name} is wider than it. "
            "The extra pass costs time and buys nothing."
        )
    if force_staging is False and tier.bits_per_weight < staging.bits_per_weight:
        if tier.is_extension:
            # The extension tiers are *packed from* a Q3_K_L intermediate;
            # there is no single-pass path to produce one. Saying "staging
            # disabled" and then staging anyway would be a lie in the
            # report, so say what actually happens.
            warnings.append(
                f"Staging cannot be skipped for {tier.name}: it is packed from a "
                f"{STAGING_TIER} intermediate, so that pass is required rather than an "
                "optimisation. force_staging=False was ignored."
            )
        else:
            warnings.append(
                f"Staging was disabled for a {tier.bits_per_weight}-bit target. A single pass "
                "has to pick every group scale from the full-precision distribution at once; "
                "expect noticeably worse output than the staged path."
            )

    index = 1
    if should_stage and tier.name != STAGING_TIER:
        steps.append(
            PlanStep(
                index=index, kind="quantize", source=source, target=STAGING_TIER,
                llama_type=staging.llama_type,
                reason="Staging pass: flatten the distribution before the final roll.",
            )
        )
        index += 1
        current = STAGING_TIER
    else:
        current = source

    if tier.is_extension:
        # The extension tiers are produced by packing a Q3_K_L
        # intermediate, so they always need something to pack *from*.
        if current != STAGING_TIER:
            steps.append(
                PlanStep(
                    index=index, kind="quantize", source=current, target=STAGING_TIER,
                    llama_type=staging.llama_type,
                    reason=f"{tier.name} is packed from a {STAGING_TIER} intermediate.",
                )
            )
            index += 1
            current = STAGING_TIER
        steps.append(
            PlanStep(
                index=index, kind="pack", source=current, target=tier.name,
                packing=tier.packing,
                reason="HyperNix sub-bit packing (extension type).",
            )
        )
    else:
        steps.append(
            PlanStep(
                index=index, kind="quantize", source=current, target=tier.name,
                llama_type=tier.llama_type,
                reason="Final roll." if len(steps) else "Single pass; target is wide enough.",
            )
        )

    if tier.honest_warning:
        warnings.append(tier.honest_warning)
    if tier.needs_imatrix and not have_imatrix:
        warnings.append(
            f"{tier.name} needs an importance matrix to be worth running. Without one the "
            "group scales are chosen blind and the output degrades far more than the bitrate "
            "alone suggests. Generate one with `llama-imatrix` first."
        )
    if parameters:
        warnings.append(
            f"Estimated weights: {tier.estimate_bytes(parameters) / 1e9:.2f} GB "
            f"for {parameters / 1e9:.1f}B parameters (weights only — no KV cache)."
        )

    return SteamrollPlan(
        source_format=source, target=tier, steps=steps,
        warnings=warnings, parameters=parameters,
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class Steamroller:
    """Runs a :class:`SteamrollPlan`.

    The upstream passes shell out to ``llama-quantize``; the extension
    passes call :meth:`pack_sub_bit`, which is HyperNix's own and is
    deliberately a separate, overridable method — the packing is the part
    most likely to be replaced, and a subclass should be able to do that
    without reimplementing the staging logic.
    """

    def __init__(
        self,
        *,
        quantize_binary: str | Path | None = None,
        workdir: str | Path | None = None,
        keep_intermediates: bool = False,
        hnx_only: bool = False,
        progress: Any = None,
    ) -> None:
        self.quantize_binary = str(quantize_binary) if quantize_binary else None
        self.workdir = Path(workdir) if workdir else None
        self.keep_intermediates = keep_intermediates
        #: Route every step through hyprslug instead of llama-quantize.
        #:
        #: The sub-bit tiers were always going to need this — they are
        #: HyperNix types llama-quantize has never heard of — but with it
        #: on, no llama.cpp binary is looked for, downloaded or built at
        #: any point, so a machine that has never had llama.cpp can still
        #: produce an IQ0.x model.
        self.hnx_only = bool(hnx_only)
        #: Called as ``progress(event: dict)`` after each step. Used by the
        #: live-stream TUI; ``None`` disables it.
        self.progress = progress

    # -- binary discovery ---------------------------------------------

    def resolve_binary(self) -> str:
        """Find ``llama-quantize``, or say what to do about it."""
        if self.quantize_binary:
            if not Path(self.quantize_binary).exists():
                raise SteamrollerError(
                    f"llama-quantize not found at {self.quantize_binary}",
                    code="missing_binary",
                )
            return self.quantize_binary
        for name in ("llama-quantize", "quantize"):
            found = shutil.which(name)
            if found:
                return found
        # hypernix.quant.quantize already knows how to fetch one; reuse it
        # rather than growing a second downloader.
        try:
            from .quantize import find_quantize_binary

            found = find_quantize_binary(auto_fetch=True)
            if found:
                return str(found)
        except Exception:  # noqa: BLE001 - fall through to the clear error
            logger.debug("steamroller: quantize.find_quantize_binary failed", exc_info=True)
        raise SteamrollerError(
            "No llama-quantize binary found.",
            code="missing_binary",
            hint="Install llama.cpp, set $LLAMA_QUANTIZE, or run `hypernix quantize --auto-fetch`.",
        )

    # -- the run ------------------------------------------------------

    def run(
        self,
        source: str | Path,
        target: str,
        output: str | Path,
        *,
        source_format: str = "FP16",
        imatrix: str | Path | None = None,
        parameters: int = 0,
        force_staging: bool | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Roll *source* down to *target*, writing *output*.

        Returns a result dict with the plan, the steps actually run, and
        timings. ``dry_run=True`` plans and reports without touching a
        binary — which is also what the GUI's "preview" button calls.
        """
        source_path = Path(source)
        output_path = Path(output)
        if not dry_run and not source_path.exists():
            raise SteamrollerError(f"No such source model: {source_path}", code="missing_source")

        the_plan = plan(
            source_format, target,
            parameters=parameters,
            have_imatrix=imatrix is not None,
            force_staging=force_staging,
        )
        self._emit({"event": "plan", "plan": the_plan.to_dict()})
        if dry_run:
            return {"dry_run": True, "plan": the_plan.to_dict(), "steps": [], "output": str(output_path)}

        workdir = self.workdir or output_path.parent
        workdir.mkdir(parents=True, exist_ok=True)
        # Not looked up in hnx mode. resolve_binary() will download a
        # llama.cpp build if it cannot find one, and "no llama.cpp, ever"
        # has to mean the lookup does not happen — not that it happens and
        # the result goes unused.
        binary = None if self.hnx_only else self.resolve_binary()

        current = source_path
        intermediates: list[Path] = []
        results: list[dict[str, Any]] = []

        for step in the_plan.steps:
            last = step is the_plan.steps[-1]
            destination = output_path if last else workdir / f"{output_path.stem}.{step.target}.gguf"
            started = time.monotonic()
            self._emit({"event": "step_start", "step": step.to_dict(), "output": str(destination)})

            if step.kind == "quantize" and not self.hnx_only:
                self._run_llama_quantize(binary, current, destination, step.llama_type, imatrix)
            elif step.kind == "quantize" and last:
                # The target itself is an upstream quant type, and
                # hyprslug can write those now. Skipping this step because
                # "hnx mode does not run llama-quantize" would produce no
                # output at all for `steamroller model.gguf Q4_K_M -hnx`.
                self.quantize_hnx(current, destination, step.llama_type, imatrix=imatrix)
            elif step.kind == "quantize":
                # A staging pass, and hnx mode has none to run: the sub-bit
                # packer reads the source directly, and passing it through
                # Q3_K_L first would only throw away precision it is about
                # to use.
                self._emit({
                    "event": "step_skipped",
                    "step": step.to_dict(),
                    "reason": "hnx mode packs from the source; no staging needed",
                })
                continue
            else:
                self.pack_sub_bit(current, destination, packing=step.packing, imatrix=imatrix)

            elapsed = time.monotonic() - started
            size = destination.stat().st_size if destination.exists() else 0
            results.append(
                {
                    **step.to_dict(),
                    "output": str(destination),
                    "bytes": size,
                    "seconds": round(elapsed, 2),
                }
            )
            self._emit({"event": "step_done", "step": results[-1]})
            if not last:
                intermediates.append(destination)
            current = destination

        if not self.keep_intermediates:
            for path in intermediates:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    # A leftover intermediate is wasted disk, not a failed
                    # run; the output is already written.
                    logger.warning("steamroller: could not remove intermediate %s", path)

        result = {
            "dry_run": False,
            "plan": the_plan.to_dict(),
            "steps": results,
            "output": str(output_path),
            "bytes": output_path.stat().st_size if output_path.exists() else 0,
            "warnings": the_plan.warnings,
        }
        self._emit({"event": "done", "result": result})
        return result

    def _run_llama_quantize(
        self,
        binary: str,
        source: Path,
        destination: Path,
        llama_type: str,
        imatrix: str | Path | None,
    ) -> None:
        cmd = [binary]
        if imatrix:
            cmd += ["--imatrix", str(imatrix)]
        cmd += [str(source), str(destination), llama_type]
        logger.info("steamroller: %s", " ".join(cmd))
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, no shell
                cmd, capture_output=True, text=True, check=False
            )
        except OSError as exc:
            raise SteamrollerError(
                f"Could not run {binary}: {exc}", code="quantize_failed"
            ) from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            raise SteamrollerError(
                f"llama-quantize failed producing {llama_type}:\n" + "\n".join(tail),
                code="quantize_failed",
            )

    def quantize_hnx(
        self,
        source: Path,
        destination: Path,
        llama_type: str,
        *,
        imatrix: str | Path | None = None,
    ) -> None:
        """Write an upstream quant type with hyprslug, no binary involved.

        The ``-hnx`` promise is "no llama.cpp is looked for, downloaded or
        built at any point" — not "only the HyperNix tiers work". A
        machine that cannot build llama.cpp is exactly the machine that
        needs a Q4_K_M.
        """
        from .hyprslug import HyprslugError, all_targets, quantize_gguf, resolve_recipe

        if resolve_recipe(llama_type) is None:
            raise SteamrollerError(
                f"-hnx cannot write {llama_type}: hyprslug has no encoder for it.",
                code="pack_failed",
                hint=f"Without -hnx this tier uses llama-quantize. hyprslug writes: "
                     f"{', '.join(all_targets())}",
            )
        try:
            report = quantize_gguf(
                source, destination, llama_type, imatrix=imatrix, progress=self.progress
            )
        except HyprslugError as exc:
            raise SteamrollerError(str(exc), code="pack_failed") from exc
        self._emit({"event": "quantized", "report": report.to_dict()})

    def pack_sub_bit(
        self,
        source: Path,
        destination: Path,
        *,
        packing: str,
        imatrix: str | Path | None = None,
    ) -> None:
        """Actually quantise *source* to the sub-bit tier *packing* names.

        This used to copy the staged file and write a sidecar JSON naming
        a tier. So a "0.5-bit model" was byte-identical to the 3-bit model
        it came from, the same size, and no more quantised — the tier was
        a label on an unchanged file. It now runs the packing.

        Overridable still: this is the part of steamroller most likely to
        be replaced, and a subclass should be able to swap the packing
        without touching the staging logic above.
        """
        if not source.exists():
            raise SteamrollerError(f"Staged model missing: {source}", code="pack_failed")
        tier = next((t for t in TIERS.values() if t.packing == packing), None)
        if tier is None:
            raise SteamrollerError(f"Unknown packing {packing!r}", code="pack_failed")

        from .hyprslug import HyprslugError, quantize_gguf

        try:
            report = quantize_gguf(
                source,
                destination,
                tier.name,
                imatrix=imatrix,
                progress=self.progress,
            )
        except HyprslugError as exc:
            raise SteamrollerError(str(exc), code="pack_failed") from exc

        self._emit({"event": "packed", "report": report.to_dict()})

        # The sidecar stays, but it is now a description of a file that
        # really is what it says rather than the only thing that made the
        # claim. The same facts are in the GGUF metadata, which a copy
        # cannot lose.
        header = {
            "hypernix.sub_bit": True,
            "hypernix.packing": packing,
            "hypernix.tier": tier.name,
            "hypernix.bits_per_weight": tier.bits_per_weight,
            "hypernix.imatrix": bool(imatrix),
            "hypernix.warning": tier.honest_warning,
            "hypernix.created": time.time(),
            "hypernix.report": report.to_dict(),
        }
        try:
            destination.with_suffix(destination.suffix + ".hypernix.json").write_text(
                json.dumps(header, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            # The model is written and valid; a missing sidecar is a
            # description, not the artefact.
            logger.warning("steamroller: could not write sidecar for %s: %s", destination, exc)

    def _emit(self, event: dict[str, Any]) -> None:
        if self.progress is None:
            return
        try:
            self.progress(event)
        except Exception:  # noqa: BLE001 - a broken listener must not fail a 40-minute run
            logger.debug("steamroller: progress callback raised", exc_info=True)
