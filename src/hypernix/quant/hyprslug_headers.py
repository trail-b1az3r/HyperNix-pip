"""hypernix.quant.hyprslug_headers — self-describing headers for any GGUF.

    hypernix hyprslug-headers install
    hypernix hyprslug-headers status
    hypernix hyprslug-headers stamp   model.gguf
    hypernix hyprslug-headers wrap    model.gguf --to Q4_K_M
    hypernix hyprslug-headers serve   model.gguf

What this cannot do, said first
-------------------------------
It cannot make a stock ``llama.cpp`` read an ``IQ0.5_XXXL`` tensor.

That is worth being blunt about, because the shape of the request
("headers that let any GGUF load") sounds like it should be a header
problem and it is not. When LM Studio says::

    llama_model_loader: failed to load model from
    .../Qwen3.8-2B-IQ0.9_L.gguf

it has read the tensor table, found GGML type 200, and stopped — because
its bundled ``llama.cpp`` has no *dequantisation kernel* for 200. The
type id is how it noticed; the missing kernel is why it stopped. Rewrite
the header to claim type 12 and the loader will happily read a 0.9-bit
tensor as ``Q4_K`` and generate confident noise. That is strictly worse
than the error: the error is honest.

So a header cannot add a kernel. What it can do is make the file
*explain itself* to anything willing to listen, and that is the part
that was genuinely missing.

Three mechanisms, and which one you want
----------------------------------------
**Stamp** (:func:`stamp`) writes a versioned ``hyprslug.header.*`` block
into the GGUF's own metadata: family, packing, block geometry, bits per
weight, the levels or the group/kept pair, and a named fallback. A
loader that has never heard of type 203 can read that block and know it
is looking at 256-weight blocks of 8 bytes each, three signs kept of
every sixteen — enough to write a decoder without this package. It is a
description, not a kernel, and it makes the file self-contained the way
a GGUF is supposed to be.

**Wrap** (:func:`wrap`) is the one that makes LM Studio open the file.
It decodes the extension tensors and re-encodes them into a type stock
``llama.cpp`` already has, keeping the stamp so the copy remembers where
it came from. The result loads anywhere and is *not* a 0.9-bit model any
more — it is a Q4_K_M model that used to be one, and it is bigger. That
trade is stated at the top of every wrap, because a compatibility export
silently presented as the original is exactly the class of lie this
package exists to stop telling.

**Serve** (:func:`serve_command`) is the one that keeps the tier. The
model stays at 0.9 bits inside :mod:`hypernix.models.hnxrun` and is
reached over an OpenAI-compatible endpoint, which LM Studio, Bionic, and
anything else that speaks ``/v1/chat/completions`` can talk to. Nothing
is converted and nothing is decoded twice.

``install`` sets up the third, finds the models the first two apply to,
and prints which is which.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .._compat import flatten

logger = logging.getLogger(__name__)

__all__ = [
    "HEADER_VERSION",
    "HeaderError",
    "Header",
    "header_for_type",
    "read_header",
    "stamp",
    "wrap",
    "scan",
    "install",
    "install_model",
    "status",
    "uninstall",
    "runtime_dir",
    "lmstudio_roots",
    "serve_command",
    "FALLBACKS",
]

#: Bumped when the meaning of a field changes, never when one is added.
#: A reader that knows version 1 must keep working against a file that
#: gained a key it does not recognise, or every future addition is a
#: breaking change.
HEADER_VERSION = 1

#: Metadata key prefix. Namespaced so it cannot collide with the
#: ``general.*`` and ``<arch>.*`` keys llama.cpp owns.
PREFIX = "hyprslug.header"


class HeaderError(RuntimeError):
    """A header could not be read, written, or acted on."""


#: For each extension type, the stock type a :func:`wrap` should target
#: by default: the narrowest upstream format that does not throw away
#: more than the extension type already has.
#:
#: Every one of these is *wider* than what it replaces, and that is not
#: avoidable — there is no 0.5-bit type in llama.cpp to land on. The
#: point of the mapping is that it does not pretend otherwise, and that
#: it does not default to Q8_0 and quadruple a file for nothing.
FALLBACKS: dict[str, str] = {
    "IQ0.25_UXL": "Q2_K",
    "IQ0.5_XXXL": "Q2_K",
    "IQ0.75_M": "Q2_K",
    "IQ0.9_L": "Q2_K",
    "INT1": "Q2_K",
    "FP2": "Q3_K",
    "INT4": "Q4_K",
}


@dataclass
class Header:
    """Everything a loader needs to decode a type it has never seen.

    Deliberately arithmetic rather than symbolic. ``packing`` names the
    codec for anything that has this package installed; the rest —
    ``block_elements``, ``block_bytes``, ``group``, ``kept``, ``levels``
    — is enough for a reader that does not, which is the whole point of
    writing it into the file.
    """

    version: int = HEADER_VERSION
    #: ``sign-and-scale``, ``fixed-codebook``, or ``upstream``.
    family: str = "upstream"
    #: The HyperNix tier name, e.g. ``IQ0.5_XXXL``.
    tier: str = ""
    #: The codec name inside this package.
    packing: str = ""
    #: The GGML type id the tensors actually carry.
    ggml_type: int = 0
    block_elements: int = 0
    block_bytes: int = 0
    bits_per_weight: float = 0.0
    #: Sign-and-scale only: signs stored per group, and the group size.
    kept: int = 0
    group: int = 0
    #: Fixed-codebook only: the levels a code indexes, in code order.
    levels: list[float] = field(default_factory=list)
    code_bits: int = 0
    #: Where the FP16 block scale sits, so a reader does not have to
    #: guess whether it leads or trails.
    scale_offset: int = 0
    scale_dtype: str = "f16"
    #: The stock type :func:`wrap` would target.
    fallback: str = ""
    #: The runtime that can execute this file as it stands.
    runtime: str = "hypernix.models.hnxrun"
    notes: str = ""

    @property
    def is_extension(self) -> bool:
        return self.family in ("sign-and-scale", "fixed-codebook")

    def to_metadata(self) -> dict[str, Any]:
        """The GGUF metadata keys this header writes."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, list):
                if not value:
                    continue
                value = json.dumps(value)
            elif value in ("", 0) and key not in ("version", "ggml_type"):
                continue
            out[f"{PREFIX}.{key}"] = value
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        if not self.is_extension:
            return "no HyperNix extension types in this file"
        if self.family == "sign-and-scale":
            shape = f"{self.kept} of every {self.group} signs kept"
        else:
            shape = f"{self.code_bits}-bit codes over {len(self.levels)} levels"
        return (
            f"{self.tier}: {shape}, {self.block_elements} weights per "
            f"{self.block_bytes}-byte block ({self.bits_per_weight:.3f} bpw), "
            f"GGML type {self.ggml_type}"
        )


def header_for_type(ggml_type: int) -> Header:
    """The header describing *ggml_type*, built from the codec itself.

    Read out of the packing tables rather than written down twice: a
    header that disagreed with the packer would be worse than no header,
    because it would be believed.
    """
    from .hyprslug import TIER_TYPES
    from .lowbit import BLOCK_SIZE as LOW_BLOCK
    from .lowbit import CODECS
    from .subbit import BLOCK_SIZE as SUB_BLOCK
    from .subbit import PACKINGS

    kind = int(ggml_type)
    tier = next((n for n, (t, _p) in TIER_TYPES.items() if t == kind), "")
    if not tier:
        return Header(family="upstream", ggml_type=kind)

    packing = TIER_TYPES[tier][1]
    fallback = FALLBACKS.get(tier, "Q4_K")
    if packing in PACKINGS:
        spec = PACKINGS[packing]
        return Header(
            family="sign-and-scale",
            tier=tier,
            packing=packing,
            ggml_type=kind,
            block_elements=SUB_BLOCK,
            block_bytes=spec.block_bytes,
            bits_per_weight=spec.bits_per_weight,
            kept=spec.kept,
            group=spec.group,
            fallback=fallback,
            notes=(
                "Signs only; magnitude is the block scale. Positions past "
                "`kept` in each group repeat the last stored sign."
            ),
        )
    codec = CODECS[packing]
    return Header(
        family="fixed-codebook",
        tier=tier,
        packing=packing,
        ggml_type=kind,
        block_elements=LOW_BLOCK,
        block_bytes=codec.block_bytes,
        bits_per_weight=codec.bits_per_weight,
        levels=list(codec.levels),
        code_bits=codec.code_bits,
        fallback=fallback,
        notes=(
            "Codes index `levels`; multiply by the block scale. Codes are "
            "packed least-significant-bit first."
        ),
    )


def _extension_type(path: Path) -> int:
    """The extension GGML type this file's tensors carry, or 0."""
    from .gguf import GGUFError, GGUFFile

    try:
        model = GGUFFile.read(Path(path))
    except (GGUFError, OSError) as exc:
        raise HeaderError(f"{path}: {exc}") from exc
    for tensor in model.tensors:
        if int(tensor.ggml_type) >= 200:
            return int(tensor.ggml_type)
    return 0


def resolve_model_path(path: str | Path) -> Path:
    """A ``.gguf`` file, from either a file or the folder holding one.

    LM Studio's layout is ``<root>/<publisher>/<name>/<name>.gguf``, and
    that is the layout ``install-model`` writes -- so the directory is
    what tab-completion stops at and what someone naturally pastes. It
    was refused, which meant the commands here would not accept the thing
    the commands here had just created.

    A directory with exactly one GGUF resolves to it. More than one is
    ambiguous and is refused *with the list*, because picking for the
    caller would be picking which model they meant.
    """
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise HeaderError(f"No such model: {candidate}")
    if not candidate.is_dir():
        raise HeaderError(f"Not a model file or directory: {candidate}")

    found = sorted(candidate.glob("*.gguf"))
    if len(found) == 1:
        return found[0]
    if not found:
        nested = sorted(candidate.rglob("*.gguf"))
        if len(nested) == 1:
            return nested[0]
        if not nested:
            raise HeaderError(
                f"{candidate} is a directory with no .gguf in it."
            )
        found = nested

    listing = "\n".join(f"    {p.name}" for p in found[:10])
    more = "" if len(found) <= 10 else f"\n    ... and {len(found) - 10} more"
    raise HeaderError(
        f"{candidate} holds {len(found)} .gguf files, so it is not clear "
        f"which one is meant. Name one:\n{listing}{more}"
    )


def read_header(path: str | Path) -> Header:
    """The header stamped into *path*, or one derived from its tensors.

    Falling back to derivation matters: every GGUF this package has ever
    written carries the extension type ids, and only the ones written
    after this module exists carry the stamp. A reader that needed the
    stamp would be useless on the files that already exist.
    """
    from .gguf import GGUFError, GGUFFile

    model_path = resolve_model_path(path)
    try:
        model = GGUFFile.read(model_path)
    except (GGUFError, OSError) as exc:
        raise HeaderError(f"{model_path}: {exc}") from exc

    stamped = {
        key[len(PREFIX) + 1:]: value
        for key, value in model.metadata.items()
        if key.startswith(PREFIX + ".")
    }
    derived = header_for_type(
        next((int(t.ggml_type) for t in model.tensors if int(t.ggml_type) >= 200), 0)
    )
    if not stamped:
        return derived

    fields = {f.name for f in Header.__dataclass_fields__.values()}
    for key, value in stamped.items():
        # Unknown keys are skipped rather than refused: a file stamped by
        # a newer HyperNix must stay readable here, or every added field
        # is a breaking change.
        if key not in fields:
            continue
        if key == "levels" and isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        setattr(derived, key, value)
    return derived


def stamp(path: str | Path, destination: str | Path | None = None) -> Header:
    """Write the ``hyprslug.header.*`` block into *path*.

    Rewrites the file, because GGUF metadata sits before the tensor data
    and growing it moves every offset. Written to a temporary file and
    moved into place, so an interrupted stamp leaves the original rather
    than half of one.
    """
    from .gguf import GGUFError, GGUFFile, GGUFWriter

    source = Path(path)
    target = Path(destination) if destination else source
    header = read_header(source)
    if not header.is_extension:
        raise HeaderError(
            f"{source} carries no HyperNix extension tensors, so there is "
            f"nothing to describe. Stamping it would add keys that say "
            f"'this is an ordinary GGUF', which it already does by being one."
        )

    try:
        model = GGUFFile.read(source)
    except (GGUFError, OSError) as exc:
        raise HeaderError(f"{source}: {exc}") from exc

    scratch = target.with_suffix(target.suffix + ".stamping")
    writer = GGUFWriter(scratch)
    # copy_metadata_from, not a set_metadata loop over the keys: the
    # loop re-infers a GGUF type for every value, and Python cannot tell
    # a UINT32 from an INT32 by looking at the number. That turned
    # general.alignment into an INT32 and the reference reader rejected
    # the result with "Bad type for general.alignment field" -- a file
    # that this package could still read and nothing else could.
    writer.copy_metadata_from(model)
    for key in list(writer.metadata):
        if key.startswith(PREFIX + "."):
            writer.metadata.pop(key, None)
            writer.metadata_types.pop(key, None)
    for key, value in header.to_metadata().items():
        writer.set_metadata(key, value)
    for tensor in model.tensors:
        writer.add_tensor(tensor.name, tensor.shape, int(tensor.ggml_type))

    payload = {t.name: model.tensor_bytes(t) for t in model.tensors}
    try:
        writer.write(lambda tensor: payload[tensor.name])
        scratch.replace(target)
    except (GGUFError, OSError) as exc:
        scratch.unlink(missing_ok=True)
        raise HeaderError(f"Could not stamp {target}: {exc}") from exc
    return header


def wrap(
    path: str | Path,
    destination: str | Path,
    *,
    to: str = "",
    progress=None,
) -> dict[str, Any]:
    """Re-encode *path* into a type stock ``llama.cpp`` can read.

    This is the command that makes the file in the screenshot open. It
    is also the command that stops it being a 0.9-bit model: the tensors
    are decoded and re-encoded, the result is several times larger, and
    the quality is whatever survived *both* quantisations. That is the
    honest trade and it is in the returned report rather than only in the
    documentation.

    The stamp is carried across, so the copy records the tier it came
    from and nobody has to reconstruct that from a filename.
    """
    from .hyprslug import HyprslugError, quantize_gguf

    source = Path(path)
    target = Path(destination)
    header = read_header(source)
    if not header.is_extension:
        raise HeaderError(
            f"{source} is already a type stock llama.cpp reads. Wrapping it "
            f"would re-quantise it for nothing."
        )

    fallback = to or header.fallback or "Q4_K_M"
    try:
        report = quantize_gguf(
            source, target, fallback,
            quantize_embeddings=True, quantize_output=True,
            progress=progress,
        )
    except HyprslugError as exc:
        raise HeaderError(f"Could not wrap {source}: {exc}") from exc

    # Verify rather than assume. The first version of this shipped a
    # wrap that reported success on a file still carrying type 200,
    # because hyprslug declined every extension tensor as unreadable and
    # copied it verbatim. The whole promise of this command is one
    # sentence -- "stock llama.cpp can open the result" -- so it is
    # checked, not asserted.
    leftover = _extension_type(target)
    if leftover:
        target.unlink(missing_ok=True)
        raise HeaderError(
            f"Wrapping {source} left GGML type {leftover} in the output, so the "
            f"result would still be refused by stock llama.cpp. The output has "
            f"been removed rather than left to be discovered by a loader."
        )

    before = source.stat().st_size
    after = target.stat().st_size
    return {
        "source": str(source),
        "output": str(target),
        "from_tier": header.tier,
        "to_type": fallback,
        "source_bytes": before,
        "output_bytes": after,
        "growth": round(after / before, 2) if before else 0.0,
        "loads_in_stock_llama_cpp": True,
        "honest_warning": (
            f"This is a {fallback} copy of a {header.tier} model, not a "
            f"{header.tier} model that llama.cpp can now read. It is "
            f"{round(after / before, 1) if before else '?'}x larger and it "
            f"carries the error of both quantisations. To keep the tier, "
            f"serve the original through hnxrun instead."
        ),
        "report": report.to_dict(),
    }


def install_model(
    path: str | Path,
    *,
    to: str = "",
    publisher: str = "HyperNix",
    name: str = "",
    root: str | Path | None = None,
    progress=None,
) -> dict[str, Any]:
    """Put a loadable copy of *path* where LM Studio and Bionic look.

    Both read the same tree — ``<root>/<publisher>/<repo>/<file>.gguf`` —
    and both list whatever is in it that their llama.cpp can open. Which
    is the constraint: a sub-bit model cannot go there as it stands, so
    what is installed is a :func:`wrap` of it.

    That trade is in the returned report and in what the command prints,
    because "installed into LM Studio" is exactly the phrase under which
    someone would otherwise assume the 0.9-bit file itself now works
    there. It does not. The installed copy is a stock quantisation of it,
    several times larger, and the original stays where it was.

    An upstream-typed GGUF is *copied* rather than re-quantised: it
    already opens, and re-encoding it would lose a generation of quality
    to no purpose.
    """
    import shutil

    source = Path(path)
    if not source.is_file():
        raise HeaderError(f"No such model: {source}")

    roots = [Path(root)] if root else lmstudio_roots()
    if not roots:
        default = Path.home() / ".lmstudio" / "models"
        raise HeaderError(
            f"No LM Studio model directory found. Pass --root, set "
            f"LMSTUDIO_HOME, or create {default}."
        )
    target_root = roots[0]

    header = read_header(source)
    label = name or source.stem
    directory = target_root / publisher / label
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{label}.gguf"

    if header.is_extension:
        fallback = to or header.fallback or "Q4_K_M"
        report = wrap(source, destination, to=fallback, progress=progress)
        report.update(
            installed_to=str(destination),
            publisher=publisher,
            model_name=label,
            converted=True,
            lmstudio_root=str(target_root),
        )
        return report

    shutil.copy2(source, destination)
    return {
        "source": str(source),
        "output": str(destination),
        "installed_to": str(destination),
        "publisher": publisher,
        "model_name": label,
        "converted": False,
        "from_tier": "upstream",
        "to_type": "unchanged",
        "source_bytes": source.stat().st_size,
        "output_bytes": destination.stat().st_size,
        "growth": 1.0,
        "lmstudio_root": str(target_root),
        "honest_warning": (
            "Copied unchanged: this file is already a type llama.cpp reads, "
            "so re-quantising it would lose a generation of quality for "
            "nothing."
        ),
    }


def runtime_dir() -> Path:
    """Where the installed runtime keeps its configuration."""
    root = os.environ.get("HYPRSLUG_HEADERS_HOME")
    if root:
        return Path(root)
    return Path.home() / ".hypernix" / "hyprslug-headers"


def lmstudio_roots() -> list[Path]:
    """Model directories LM Studio (and Bionic) are known to use.

    Both read the same tree; Bionic is a front end over LM Studio's
    model store, which is why one scan covers them. The environment
    variable wins so a non-standard install is not a dead end.
    """
    override = os.environ.get("LMSTUDIO_HOME")
    if override:
        return [Path(override) / "models"]
    home = Path.home()
    candidates = [
        home / ".lmstudio" / "models",
        home / ".cache" / "lm-studio" / "models",
        home / "Library" / "Application Support" / "LM Studio" / "models",
        home / "AppData" / "Roaming" / "LM Studio" / "models",
    ]
    return [c for c in candidates if c.is_dir()]


#: Tier names as they appear in filenames, longest first so IQ0.5_XXXL
#: is recognised before a shorter name that is a prefix of it.
_TIER_NAMES_IN_FILENAMES = (
    "IQ0.25_UXL", "IQ0.5_XXXL", "IQ0.75_M", "IQ0.9_L",
    "INT1", "INT4", "FP2",
)


def tier_in_name(path: str | Path) -> str:
    """The tier a filename *claims*, or "" if it names none.

    Names are not evidence -- the tensors are -- but a name that
    contradicts them is worth saying out loud. A file called
    ``…-IQ0.9_L.gguf`` whose tensors are type 202 is an IQ0.5_XXXL model,
    and someone reading a report that lists the real tier beside the
    filename has to notice the disagreement themselves.
    """
    stem = Path(path).name.replace("-", "_").replace(" ", "_").upper()
    for tier in _TIER_NAMES_IN_FILENAMES:
        if tier.replace(".", "").replace("_", "") in stem.replace(".", "").replace("_", ""):
            return tier
    return ""


def scan(root: str | Path) -> list[dict[str, Any]]:
    """Every GGUF under *root*, classified by whether llama.cpp can read it.

    The classification is the useful part. A directory of models is
    otherwise indistinguishable from a directory of models one of which
    will fail at load time with a message about a loader, and finding
    out which by loading each one is the slow way.
    """
    found: list[dict[str, Any]] = []
    for candidate in sorted(Path(root).rglob("*.gguf")):
        entry: dict[str, Any] = {"path": str(candidate), "bytes": 0}
        try:
            entry["bytes"] = candidate.stat().st_size
            header = read_header(candidate)
        except (HeaderError, OSError) as exc:
            entry.update(readable=False, error=str(exc))
            found.append(entry)
            continue
        claimed = tier_in_name(candidate)
        entry.update(
            tier=header.tier or "upstream",
            family=header.family,
            extension=header.is_extension,
            stock_llama_cpp=not header.is_extension,
            bits_per_weight=header.bits_per_weight,
            fallback=header.fallback,
            named_tier=claimed,
            # The tensors are the truth; the filename is a label someone
            # typed. Reporting the mismatch beats reporting either alone.
            misnamed=bool(claimed and header.tier and claimed != header.tier),
        )
        found.append(entry)
    return found


def serve_command(model: str | Path, *, host: str = "127.0.0.1",
                  port: int = 1234, cache_bytes: int = 0) -> list[str]:
    """The argv that serves *model* over an OpenAI-compatible endpoint."""
    argv = [
        "hypernix", "hyprslug-headers", "serve", str(model),
        "--host", host, "--port", str(port),
    ]
    if cache_bytes:
        argv += ["--cache-bytes", str(cache_bytes)]
    return argv


def install(*, host: str = "127.0.0.1", port: int = 1234,
            scan_lmstudio: bool = True) -> dict[str, Any]:
    """Set up the runtime and report what it found.

    Writes a config the ``serve`` command reads, then looks in LM
    Studio's model tree for files that will fail to load there, so the
    answer to "why did this model not open" arrives before the question.
    """
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)

    from .hyprslug import TIER_TYPES

    config = {
        "header_version": HEADER_VERSION,
        "runtime": "hypernix.models.hnxrun",
        "host": host,
        "port": port,
        "types": {
            tier: header_for_type(type_id).to_dict()
            for tier, (type_id, _packing) in TIER_TYPES.items()
        },
        "fallbacks": dict(FALLBACKS),
    }
    (directory / "runtime.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    roots = lmstudio_roots() if scan_lmstudio else []
    models: list[dict[str, Any]] = flatten(scan(root) for root in roots)
    needs_runtime = [m for m in models if m.get("extension")]

    return {
        "installed_to": str(directory),
        "config": str(directory / "runtime.json"),
        "host": host,
        "port": port,
        "lmstudio_roots": [str(r) for r in roots],
        "models_seen": len(models),
        "models_needing_the_runtime": needs_runtime,
        "hnx_available": shutil.which("hypernix") is not None,
    }


def status() -> dict[str, Any]:
    """What is installed, if anything, and what it can decode."""
    directory = runtime_dir()
    config_path = directory / "runtime.json"
    if not config_path.is_file():
        return {"installed": False, "runtime_dir": str(directory)}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "installed": False,
            "runtime_dir": str(directory),
            "error": f"{config_path} is unreadable: {exc}",
        }
    return {
        "installed": True,
        "runtime_dir": str(directory),
        "header_version": config.get("header_version"),
        "host": config.get("host"),
        "port": config.get("port"),
        "types": sorted(config.get("types", {})),
        "lmstudio_roots": [str(r) for r in lmstudio_roots()],
    }


def uninstall() -> dict[str, Any]:
    """Remove the runtime config. Models are never touched."""
    directory = runtime_dir()
    removed = []
    for name in ("runtime.json",):
        target = directory / name
        if target.is_file():
            target.unlink()
            removed.append(str(target))
    if directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
    return {"removed": removed, "runtime_dir": str(directory)}
