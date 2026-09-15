"""hypernix.quant.multiquant — several quantisations, one GGUF.

Shipping a model means shipping a table: Q8_0 for the people with the
VRAM, Q4_K_M for most of them, IQ0.5_XXXL for the person trying it on a
phone. Five files, five downloads, and a README explaining which to take.

A bundle is those five in one file. One variant — the default — keeps the
ordinary tensor names, so a llama.cpp that has never heard of this loads
the file and runs that variant with nothing unusual happening. The others
sit under ``hnxq.<variant>.`` prefixes with their own metadata, invisible
to a loader that does not look for them and one metadata read away for
one that does.

Where the speed comes from
--------------------------
Two places, and only one of them is about arithmetic.

The first is the page cache. Five separate files that share a 600 MB
embedding table cost 3 GB of reads to sample all five; a bundle stores
that table once and every variant that did not requantise it points at
the same tensor. :func:`bundle` does this automatically — a tensor a
variant would have copied verbatim, byte for byte identical to the
default's, is not written twice.

The second is switching. Changing quantisation on a served model means
loading a different file, which means a cold mmap and a cold cache.
Inside one bundle the tensors are already in the same file, usually
already resident, and a runtime that supports it can move between tiers
without leaving the page cache — which is what makes "drop to Q4 under
load, go back to Q8 when it clears" a thing anyone would actually do.

What it is not
--------------
It is not a mixed-precision model. Each variant is a complete, coherent
quantisation of every tensor; nothing here runs half of one tier and half
of another. That is what a :class:`hypernix.quant.hyprslug.Recipe` is
for, inside a single variant.

It also does not make the *file* smaller than the sum of its parts by
much. Shared tensors are stored once, and that is the whole saving. A
bundle of Q8_0 and Q4_K_M is roughly the size of a Q8_0 plus a Q4_K_M.
The point is the second sentence of this docstring, not compression.

Compatibility
-------------
Every extra tensor is namespaced under ``hnxq.``, and every metadata key
this writes sits outside the ``general.``/``<arch>.`` namespaces upstream
uses. A stock llama.cpp finds every tensor it expects under the names it
expects, reads past the rest, and runs the default variant exactly as it
would run a plain file of that tier. :func:`extract` writes any variant
out as its own ordinary GGUF for a runtime that wants one file per tier —
the bundle still carries all of them.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gguf import GGUFError, GGUFFile, GGUFTensor, GGUFWriter
from .hyprslug import (
    HyprslugError,
    TargetSpec,
    TensorPlan,
    encode_tensor,
    load_imatrix,
    plan_tensors,
    target_spec,
    write_provenance,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MultiQuantError",
    "PREFIX",
    "VERSION",
    "VariantInfo",
    "BundleReport",
    "bundle",
    "is_bundle",
    "read_bundle_info",
    "variants",
    "extract",
    "strip",
    "canonical_tier",
    "slug_for",
]

#: Every tensor a non-default variant owns starts with this.
PREFIX = "hnxq."

#: The metadata generation. Bumped only when the layout changes in a way
#: a reader has to know about.
VERSION = 1

KEY_PRESENT = "hnxq.present"
KEY_VERSION = "hnxq.version"
KEY_VARIANTS = "hnxq.variants"
KEY_DEFAULT = "hnxq.default"
KEY_COUNT = "hnxq.variant_count"

#: Per-variant key suffixes, under ``hnxq.<slug>.``.
SUFFIX_TIER = "tier"
SUFFIX_BPW = "bits_per_weight"
SUFFIX_BYTES = "bytes"
SUFFIX_TENSORS = "tensor_count"
SUFFIX_SHARED = "shared_tensors"

_SLUG_BAD = re.compile(r"[^a-z0-9]+")


class MultiQuantError(Exception):
    """A bundle could not be built, read or taken apart."""


def canonical_tier(tier: str) -> str:
    """*tier* as hyprslug spells it, or unchanged when it is not a target.

    `build` resolves its targets through hyprslug, so a bundle built from
    `q4_m` stores `Q4_K_M` — and `extract … q4_m` then could not find it,
    because only the canonical spelling had been through the resolver.
    One name in, one name out, wherever a tier is named.

    Unresolvable names pass through rather than raising: the caller is
    about to look the slug up in a real file and say what is actually in
    there, which is a better error than this function's list of every
    target hyprslug can write.
    """
    try:
        return target_spec(tier).name
    except HyprslugError:
        return tier


def slug_for(tier: str) -> str:
    """A tensor-name-safe spelling of *tier*.

    ``IQ0.5_XXXL`` becomes ``iq0_5_xxxl``. The dot is the reason this
    exists: GGUF tensor names are dot-separated by convention and
    everything that parses one — including this module's own
    :func:`extract` — splits on it, so a variant whose slug carried a dot
    would be read as two path segments and not found.
    """
    return _SLUG_BAD.sub("_", tier.strip().lower()).strip("_")


@dataclass
class VariantInfo:
    """One quantisation inside a bundle."""

    tier: str
    slug: str
    default: bool = False
    bits_per_weight: float = 0.0
    tensor_count: int = 0
    shared_count: int = 0
    nbytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "slug": self.slug,
            "default": self.default,
            "bits_per_weight": round(self.bits_per_weight, 3),
            "tensor_count": self.tensor_count,
            "shared_tensors": self.shared_count,
            "bytes": self.nbytes,
        }


@dataclass
class BundleReport:
    """What :func:`bundle` produced, and what sharing saved."""

    variants: list[VariantInfo] = field(default_factory=list)
    source_bytes: int = 0
    output_bytes: int = 0
    shared_bytes: int = 0
    seconds: float = 0.0

    @property
    def default(self) -> str:
        return next((v.tier for v in self.variants if v.default), "")

    @property
    def separate_bytes(self) -> int:
        """What the same variants would cost as separate files.

        Approximate by exactly the shared tensors: each would appear once
        per file instead of once in total. Metadata and padding differ by
        kilobytes, which is not worth pretending to model.
        """
        return self.output_bytes + self.shared_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "variants": [v.to_dict() for v in self.variants],
            "default": self.default,
            "source_bytes": self.source_bytes,
            "output_bytes": self.output_bytes,
            "shared_bytes": self.shared_bytes,
            "separate_bytes": self.separate_bytes,
            "seconds": round(self.seconds, 2),
        }

    def describe(self) -> str:
        lines = [
            f"bundle: {len(self.variants)} variant(s), default {self.default}",
            f"  {self.source_bytes / 1e6:.1f} MB source -> "
            f"{self.output_bytes / 1e6:.1f} MB bundle",
        ]
        for variant in self.variants:
            mark = "*" if variant.default else " "
            lines.append(
                f"  {mark} {variant.tier:14} {variant.nbytes / 1e6:8.1f} MB  "
                f"{variant.tensor_count:4d} tensors"
                + (f"  (+{variant.shared_count} shared)" if variant.shared_count else "")
            )
        if self.shared_bytes:
            saved = self.shared_bytes / 1e6
            lines.append("")
            lines.append(
                f"  {saved:.1f} MB saved by storing shared tensors once "
                f"({self.separate_bytes / 1e6:.1f} MB as separate files)."
            )
        lines.append("")
        lines.append(
            "  * is the variant a stock llama.cpp runs. The others are under"
        )
        lines.append(
            f"    {PREFIX}<slug>. and are invisible to a loader that does not"
        )
        lines.append(
            "    look for them."
        )
        return "\n".join(lines)


def is_bundle(model: GGUFFile) -> bool:
    """Whether *model* carries more than one quantisation."""
    return bool(model.metadata.get(KEY_PRESENT))


def _variant_key(slug: str, suffix: str) -> str:
    return f"{PREFIX}{slug}.{suffix}"


def variants(path: str | Path) -> list[VariantInfo]:
    """Every variant in the bundle at *path*, default first.

    An empty list for an ordinary GGUF, which is not an error: "how many
    quantisations does this file carry" has the answer "one, the usual
    way" for almost every GGUF in existence.
    """
    try:
        model = GGUFFile.read(Path(path))
    except GGUFError as exc:
        raise MultiQuantError(f"{path}: {exc}") from exc
    return _variants_of(model)


def _variants_of(model: GGUFFile) -> list[VariantInfo]:
    if not is_bundle(model):
        return []
    default = str(model.metadata.get(KEY_DEFAULT) or "")
    names = model.metadata.get(KEY_VARIANTS) or []
    found: list[VariantInfo] = []
    for tier in names:
        tier = str(tier)
        slug = slug_for(tier)
        found.append(VariantInfo(
            tier=tier,
            slug=slug,
            default=(tier == default),
            bits_per_weight=float(
                model.metadata.get(_variant_key(slug, SUFFIX_BPW)) or 0.0
            ),
            tensor_count=int(
                model.metadata.get(_variant_key(slug, SUFFIX_TENSORS)) or 0
            ),
            shared_count=int(
                model.metadata.get(_variant_key(slug, SUFFIX_SHARED)) or 0
            ),
            nbytes=int(model.metadata.get(_variant_key(slug, SUFFIX_BYTES)) or 0),
        ))
    found.sort(key=lambda v: (not v.default, -v.bits_per_weight))
    return found


def read_bundle_info(path: str | Path) -> dict[str, Any]:
    """The bundle's metadata as a plain dict, or ``{}`` if it is not one."""
    try:
        model = GGUFFile.read(Path(path))
    except GGUFError as exc:
        raise MultiQuantError(f"{path}: {exc}") from exc
    if not is_bundle(model):
        return {}
    return {
        "version": int(model.metadata.get(KEY_VERSION) or 0),
        "default": str(model.metadata.get(KEY_DEFAULT) or ""),
        "variants": [v.to_dict() for v in _variants_of(model)],
    }


def bundle(
    source: str | Path,
    destination: str | Path,
    targets: Sequence[str],
    *,
    default: str | None = None,
    imatrix: str | Path | dict[str, list[float]] | None = None,
    quantize_embeddings: bool | None = None,
    quantize_output: bool | None = None,
    share: bool = True,
    progress: Callable[[dict], None] | None = None,
) -> BundleReport:
    """Quantise *source* to every name in *targets*, all into *destination*.

    *default* names the variant that keeps the ordinary tensor names —
    the one a stock llama.cpp will run. It defaults to the first of
    *targets*, which is the one most people mean: the list is usually
    written widest-first and the widest is the one to serve when nothing
    says otherwise.

    *share* stores a tensor once when a variant would have written bytes
    identical to the default's. That is exactly the tensors both copied
    verbatim — the norms, the biases, and whichever of the embedding
    table and output head the tier left alone — and it is where the size
    saving comes from. Turning it off makes every variant's tensor set
    complete and independent, which costs the saving and buys a file you
    can cut apart with a hex editor.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.exists():
        raise MultiQuantError(f"No such model: {source_path}")

    wanted = [str(t) for t in targets if str(t).strip()]
    if not wanted:
        raise MultiQuantError(
            "A bundle needs at least one target. With one it is an ordinary "
            "quantisation and `hyprslug SOURCE TIER` is the simpler command."
        )

    try:
        specs = [target_spec(t) for t in wanted]
    except HyprslugError as exc:
        raise MultiQuantError(str(exc)) from exc

    seen: dict[str, str] = {}
    for spec in specs:
        slug = slug_for(spec.name)
        if slug in seen:
            raise MultiQuantError(
                f"{seen[slug]!r} and {spec.name!r} are the same quantisation, "
                f"so the bundle would carry it twice under one name."
            )
        seen[slug] = spec.name

    if default is None:
        default_spec = specs[0]
    else:
        try:
            default_spec = target_spec(default)
        except HyprslugError as exc:
            raise MultiQuantError(str(exc)) from exc
        if default_spec.name not in {s.name for s in specs}:
            raise MultiQuantError(
                f"default {default!r} resolves to {default_spec.name}, which is "
                f"not among the targets ({', '.join(s.name for s in specs)}). "
                "The default has to be one of the variants in the file."
            )
    # Default first: it owns the un-prefixed names, and writing it first
    # keeps a stock loader's tensors at the front of the table where it
    # will find them without walking past everything else.
    specs.sort(key=lambda s: s.name != default_spec.name)

    weights_by_tensor: dict[str, list[float]] = {}
    if isinstance(imatrix, dict):
        weights_by_tensor = imatrix
    elif imatrix is not None:
        try:
            weights_by_tensor = load_imatrix(imatrix)
        except HyprslugError as exc:
            raise MultiQuantError(str(exc)) from exc

    started = time.time()
    try:
        model = GGUFFile.read(source_path)
    except GGUFError as exc:
        raise MultiQuantError(f"{source_path}: {exc}") from exc

    if any(tensor.name.startswith(PREFIX) for tensor in model.tensors):
        raise MultiQuantError(
            f"{source_path} is already a bundle. Bundling a bundle would nest "
            "the prefixes and no reader unwraps that. Use `hyprslug --extract` "
            "to pull a variant out first."
        )

    report = BundleReport(source_bytes=source_path.stat().st_size)
    writer = GGUFWriter(destination_path, alignment=model.alignment)
    writer.copy_metadata_from(model)
    writer.set_metadata(KEY_PRESENT, True)
    writer.set_metadata(KEY_VERSION, VERSION)
    writer.set_metadata(KEY_DEFAULT, default_spec.name)
    writer.set_metadata(KEY_VARIANTS, [s.name for s in specs])
    writer.set_metadata(KEY_COUNT, len(specs))

    # name in the file -> (source tensor, plan, spec). One flat table:
    # every tensor of every variant, which is what the writer streams.
    sources: dict[str, tuple[TensorPlan, TargetSpec]] = {}
    # The default's plan, so a later variant can ask "would this be the
    # same bytes?" without re-deriving it.
    default_plans: dict[str, TensorPlan] = {}

    for index, spec in enumerate(specs):
        slug = slug_for(spec.name)
        is_default = spec.name == default_spec.name
        plans = plan_tensors(
            model, spec,
            quantize_embeddings=quantize_embeddings,
            quantize_output=quantize_output,
        )
        info = VariantInfo(
            tier=spec.name,
            slug=slug,
            default=is_default,
            bits_per_weight=spec.bits_per_weight,
        )
        shared_names: list[str] = []
        for plan in plans:
            if is_default:
                name = plan.name
                default_plans[plan.name] = plan
            else:
                # Identical bytes to the default's copy of the same
                # tensor: both copied verbatim from the same source. Not
                # written again; the variant's reader is told to look at
                # the un-prefixed name instead.
                twin = default_plans.get(plan.name)
                if (
                    share
                    and twin is not None
                    and plan.copied
                    and twin.copied
                    and int(twin.ggml_type) == int(plan.ggml_type)
                ):
                    shared_names.append(plan.name)
                    info.shared_count += 1
                    report.shared_bytes += plan.source.nbytes
                    continue
                name = f"{PREFIX}{slug}.{plan.name}"
            declared = writer.add_tensor(name, plan.source.shape, plan.ggml_type)
            sources[name] = (plan, spec)
            info.tensor_count += 1
            info.nbytes += declared.nbytes
        if not is_default:
            writer.set_metadata(_variant_key(slug, SUFFIX_SHARED), len(shared_names))
            if shared_names:
                writer.set_metadata(
                    f"{PREFIX}{slug}.shared", sorted(shared_names)
                )
        writer.set_metadata(_variant_key(slug, SUFFIX_TIER), spec.name)
        writer.set_metadata(
            _variant_key(slug, SUFFIX_BPW), float(spec.bits_per_weight)
        )
        writer.set_metadata(_variant_key(slug, SUFFIX_TENSORS), info.tensor_count)
        writer.set_metadata(_variant_key(slug, SUFFIX_BYTES), info.nbytes)
        write_provenance(
            writer, spec,
            imatrix=bool(weights_by_tensor),
            prefix="" if is_default else f"{PREFIX}{slug}.",
        )
        report.variants.append(info)
        if progress is not None:
            try:
                progress({
                    "event": "variant",
                    "tier": spec.name,
                    "index": index + 1,
                    "total": len(specs),
                    "default": is_default,
                })
            except Exception:  # noqa: BLE001 - a listener must not fail the run
                logger.debug("multiquant: progress callback raised", exc_info=True)

    done = 0
    total = len(sources)

    def _data_for(declared: GGUFTensor) -> bytes:
        nonlocal done
        plan, spec = sources[declared.name]
        raw = model.tensor_bytes(plan.source)
        done += 1
        if progress is not None:
            try:
                progress({
                    "event": "tensor",
                    "name": declared.name,
                    "index": done,
                    "total": total,
                    "tier": spec.name,
                    "quantized": not plan.copied,
                })
            except Exception:  # noqa: BLE001
                logger.debug("multiquant: progress callback raised", exc_info=True)
        payload, _saturated = encode_tensor(
            raw, int(plan.source.ggml_type), plan, spec,
            importance=weights_by_tensor.get(plan.name),
        )
        return payload

    try:
        writer.write(_data_for)
    except (GGUFError, OSError, HyprslugError) as exc:
        raise MultiQuantError(f"Could not write {destination_path}: {exc}") from exc

    report.output_bytes = destination_path.stat().st_size
    report.seconds = time.time() - started
    if progress is not None:
        try:
            progress({"event": "done", **report.to_dict()})
        except Exception:  # noqa: BLE001
            logger.debug("multiquant: progress callback raised", exc_info=True)
    return report


def extract(
    source: str | Path,
    destination: str | Path,
    tier: str,
    *,
    progress: Callable[[dict], None] | None = None,
) -> VariantInfo:
    """Write one variant of *source* out as an ordinary GGUF.

    The result has no ``hnxq.`` anything: plain tensor names, plain
    metadata, a file indistinguishable from one quantised to that tier
    directly. For a runtime that wants one file per tier, or for
    uploading a single variant somewhere.
    """
    source_path = Path(source)
    out_path = Path(destination)
    try:
        model = GGUFFile.read(source_path)
    except GGUFError as exc:
        raise MultiQuantError(f"{source_path}: {exc}") from exc
    if not is_bundle(model):
        raise MultiQuantError(
            f"{source_path} carries one quantisation, not several, so there is "
            "nothing to extract from it. Copy the file."
        )

    found = _variants_of(model)
    wanted = slug_for(canonical_tier(tier))
    chosen = next((v for v in found if v.slug == wanted), None)
    if chosen is None:
        raise MultiQuantError(
            f"{source_path} has no {tier!r} variant. It carries: "
            f"{', '.join(v.tier for v in found)}"
        )

    prefix = f"{PREFIX}{chosen.slug}."
    shared = {
        str(name)
        for name in (model.metadata.get(f"{PREFIX}{chosen.slug}.shared") or [])
    }

    writer = GGUFWriter(out_path, alignment=model.alignment)
    # Everything except the bundle's own bookkeeping and every other
    # variant's. What comes out must not claim to be a bundle.
    for key, value in model.metadata.items():
        if key.startswith(PREFIX):
            continue
        writer.set_metadata(key, value, type_hint=model.metadata_types.get(key))
    for key, value in model.metadata.items():
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            if suffix.startswith("hypernix."):
                writer.set_metadata(suffix, value, type_hint=model.metadata_types.get(key))
    writer.set_metadata("hypernix.tier", chosen.tier)
    writer.set_metadata(
        "general.file_type_description",
        str(model.metadata.get(f"{prefix}hypernix.description") or chosen.tier),
    )

    sources: dict[str, GGUFTensor] = {}
    if chosen.default:
        # The default's tensors are the un-prefixed ones, minus every
        # other variant's.
        for tensor in model.tensors:
            if tensor.name.startswith(PREFIX):
                continue
            writer.add_tensor(tensor.name, tensor.shape, tensor.ggml_type)
            sources[tensor.name] = tensor
    else:
        for tensor in model.tensors:
            if not tensor.name.startswith(prefix):
                continue
            plain = tensor.name[len(prefix):]
            writer.add_tensor(plain, tensor.shape, tensor.ggml_type)
            sources[plain] = tensor
        # The tensors this variant shares with the default. They live
        # under the plain names and are as much this variant's as the
        # default's -- a file without them would be missing its norms.
        for name in sorted(shared):
            tensor = model.get(name)
            if tensor is None:
                raise MultiQuantError(
                    f"{source_path} says {chosen.tier} shares {name!r} with the "
                    "default, and that tensor is not in the file. The bundle is "
                    "damaged; re-run the bundling."
                )
            writer.add_tensor(name, tensor.shape, tensor.ggml_type)
            sources[name] = tensor

    if not sources:
        raise MultiQuantError(
            f"{source_path} declares a {chosen.tier} variant and carries no "
            f"tensors under {prefix!r}. The bundle is damaged."
        )

    done = 0

    def _data_for(declared: GGUFTensor) -> bytes:
        nonlocal done
        done += 1
        if progress is not None:
            try:
                progress({
                    "event": "tensor",
                    "name": declared.name,
                    "index": done,
                    "total": len(sources),
                })
            except Exception:  # noqa: BLE001
                logger.debug("multiquant: progress callback raised", exc_info=True)
        return model.tensor_bytes(sources[declared.name])

    try:
        writer.write(_data_for)
    except (GGUFError, OSError) as exc:
        raise MultiQuantError(f"Could not write {out_path}: {exc}") from exc

    return VariantInfo(
        tier=chosen.tier,
        slug=chosen.slug,
        default=chosen.default,
        bits_per_weight=chosen.bits_per_weight,
        tensor_count=len(sources),
        shared_count=chosen.shared_count,
        nbytes=out_path.stat().st_size,
    )


def strip(source: str | Path, destination: str | Path) -> int:
    """Write *source* without its extra variants. Returns how many went.

    The default variant survives, as an ordinary GGUF. Equivalent to
    ``extract(source, destination, <default tier>)`` and spelled
    separately because "make this a normal file again" is the thing
    people actually want, and they should not have to look up which tier
    the default was to ask for it.
    """
    try:
        model = GGUFFile.read(Path(source))
    except GGUFError as exc:
        raise MultiQuantError(f"{source}: {exc}") from exc
    if not is_bundle(model):
        raise MultiQuantError(f"{source} is not a bundle; there is nothing to strip.")
    found = _variants_of(model)
    default = next((v for v in found if v.default), None)
    if default is None:
        raise MultiQuantError(
            f"{source} declares variants but names no default, so there is no "
            "way to know which one to keep."
        )
    extract(source, destination, default.tier)
    return len(found) - 1
