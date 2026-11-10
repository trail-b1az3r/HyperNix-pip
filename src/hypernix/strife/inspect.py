"""strife.inspect — what a GGUF is, and whether Strife can run it.

Nothing here allocates more than the header and tensor table. A model is
checked before any weight is touched:

* the header parses, with every length bounded by the file's real size
  (:mod:`hypernix.quant.gguf`);
* every tensor's offset is aligned, its byte range is inside the file,
  its shape arithmetic does not overflow, and no two tensors overlap;
* its architecture is one Strife implements, with every tensor that
  architecture needs, in shapes that agree with the metadata;
* every tensor type has a verified decoder (:mod:`.quants`) -- an unknown
  type is refused by name, never read as some other type.

:func:`inspect_model` returns a :class:`ModelReport` either way, and the
report says what is wrong. :func:`require_runnable` raises a coded
:class:`~.errors.StrifeError` for the first blocking problem.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..quant.gguf import GGUFError, GGUFFile, type_block_size
from .archs import ARCHITECTURES, PlanError, build_plan
from .errors import StrifeError
from .quants import quant_name, supported_type

__all__ = ["ARCHITECTURES", "ModelReport", "inspect_model", "require_runnable",
           "file_sha256", "read_gguf", "shapes_of"]

#: The largest element count accepted for one tensor (2^40): past any
#: real tensor, short of overflow in any arithmetic done on it.
_MAX_ELEMENTS = 1 << 40
_MAX_DIMS = 4


@dataclass
class ModelReport:
    path: str
    file_bytes: int = 0
    gguf_version: int = 0
    architecture: str = ""
    name: str = ""
    n_layers: int = 0
    n_embd: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    n_ff: int = 0
    n_vocab: int = 0
    context_length: int = 0
    rope_freq_base: float = 10000.0
    rope_scale: float = 1.0
    rms_eps: float = 1e-5
    tokenizer: str = ""
    bos_id: int | None = None
    eos_id: int | None = None
    chat_template: bool = False
    tensor_count: int = 0
    weight_bytes: int = 0
    types: dict[str, int] = field(default_factory=dict)
    tied_output: bool = False
    #: The architecture's own facts (layer mix, experts, rope dims, ...).
    details: dict[str, Any] = field(default_factory=dict)
    problems: list[dict[str, Any]] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        return not self.problems

    def problem(self, code: str, message: str, remedy: str = "", **details: Any) -> None:
        self.problems.append({"code": code, "message": message, "remedy": remedy,
                              "details": details})

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["runnable"] = self.runnable
        return out


def file_sha256(path: str | Path) -> str:
    """SHA-256 of the whole file, for benchmark records."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block(ggml_type: int) -> int:
    try:
        return type_block_size(ggml_type)
    except GGUFError:
        return 1


def _check_tensors(gguf: GGUFFile, report: ModelReport, file_size: int) -> None:
    spans = []
    seen: set[str] = set()
    for t in gguf.tensors:
        if t.name in seen:
            report.problem("strife.model.duplicate_tensor", f"tensor {t.name!r} appears twice",
                           tensor=t.name)
        seen.add(t.name)
        if not 1 <= len(t.shape) <= _MAX_DIMS or any(int(d) <= 0 for d in t.shape):
            report.problem("strife.model.bad_shape", f"tensor {t.name!r} has shape {t.shape}",
                           tensor=t.name)
            continue
        elements = 1
        for d in t.shape:
            elements *= int(d)
        if elements > _MAX_ELEMENTS:
            report.problem("strife.model.bad_shape",
                           f"tensor {t.name!r} claims {elements} elements", tensor=t.name)
            continue
        name = quant_name(t.ggml_type)
        report.types[name] = report.types.get(name, 0) + 1
        if not supported_type(t.ggml_type):
            report.problem("strife.quant.unsupported",
                           f"tensor {t.name!r} is {name} (type {t.ggml_type}), which Strife "
                           f"has no verified kernel for",
                           "Re-quantise to a supported type (`hnx strife inspect --types` lists "
                           "them), or run this model with the llama.cpp runner.",
                           tensor=t.name, ggml_type=t.ggml_type)
            continue
        if int(t.shape[0]) % _block(t.ggml_type):
            report.problem("strife.quant.bad_row",
                           f"tensor {t.name!r} has rows of {t.shape[0]}, not a whole number of "
                           f"{name} blocks", tensor=t.name)
            continue
        if t.offset % gguf.alignment:
            report.problem("strife.model.misaligned",
                           f"tensor {t.name!r} starts at {t.offset}, not a multiple of the "
                           f"file's alignment {gguf.alignment}", tensor=t.name)
        start = gguf.data_start + t.offset
        if t.nbytes <= 0 or start + t.nbytes > file_size:
            report.problem("strife.model.truncated",
                           f"tensor {t.name!r} runs past the end of the file "
                           f"({start + t.nbytes} > {file_size} bytes)",
                           "The file is incomplete or damaged; download it again.",
                           tensor=t.name)
            continue
        report.weight_bytes += t.nbytes
        spans.append((t.offset, t.offset + t.nbytes, t.name))
    spans.sort()
    for (_a0, a1, an), (b0, _b1, bn) in zip(spans, spans[1:], strict=False):
        if b0 < a1:
            report.problem("strife.model.overlap", f"tensors {an!r} and {bn!r} overlap",
                           tensor=bn)


def shapes_of(gguf: GGUFFile) -> dict[str, tuple[int, ...]]:
    return {t.name: tuple(int(d) for d in t.shape) for t in gguf.tensors}


def _check_architecture(gguf: GGUFFile, report: ModelReport) -> None:
    md = gguf.metadata
    report.architecture = str(md.get("general.architecture") or "")
    report.name = str(md.get("general.name") or "")
    try:
        plan = build_plan(md, shapes_of(gguf))
    except PlanError as exc:
        details = {"architecture": report.architecture}
        if exc.tensor:
            details["tensor"] = exc.tensor
        report.problem(exc.code, str(exc), exc.remedy, **details)
        return
    report.n_layers, report.n_embd, report.n_vocab = plan.n_layers, plan.n_embd, plan.n_vocab
    report.context_length, report.rms_eps, report.tied_output = plan.context_length, plan.eps, plan.tied
    attn = next((lay.attn for lay in plan.layers if lay.attn is not None), None)
    if attn is not None:
        report.n_heads, report.n_kv_heads, report.head_dim = attn.n_heads, attn.n_kv_heads, attn.head_dim
        report.rope_freq_base, report.rope_scale = attn.rope_base, attn.rope_scale
    report.n_ff = next((lay.n_ff for lay in plan.layers if lay.n_ff), 0) or next(
        (lay.moe.n_ff for lay in plan.layers if lay.moe), 0)
    report.details = dict(plan.summary)


def _check_tokenizer(gguf: GGUFFile, report: ModelReport) -> None:
    md = gguf.metadata
    model = str(md.get("tokenizer.ggml.model") or "")
    report.tokenizer = model
    if model not in ("gpt2", "llama"):
        report.problem("strife.tokenizer.unsupported",
                       f"tokenizer {model or '(none)'!r} is not one Strife reads",
                       "Strife reads byte-level BPE (gpt2) and SentencePiece (llama) "
                       "tokenizers stored in the GGUF.", tokenizer=model)
        return
    tokens = md.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or not tokens:
        report.problem("strife.tokenizer.missing", "the GGUF has no tokenizer.ggml.tokens")
        return
    if model == "gpt2" and not isinstance(md.get("tokenizer.ggml.merges"), list):
        report.problem("strife.tokenizer.missing", "a gpt2 tokenizer without merges")
    if model == "gpt2":
        from .tokenizer import _PRE

        pre = str(md.get("tokenizer.ggml.pre") or "default")
        if pre not in _PRE:
            report.problem("strife.tokenizer.pre_unsupported",
                           f"the pre-tokenizer {pre!r} is not one Strife knows, so text would be "
                           "split differently from how the model was trained",
                           "Strife knows: " + ", ".join(sorted(_PRE))
                           + ". Use the llama.cpp runner for this model.", pre=pre)
    if report.n_vocab and len(tokens) > report.n_vocab:
        report.problem("strife.tokenizer.mismatch",
                       f"{len(tokens)} tokens for an embedding of {report.n_vocab} rows")
    report.bos_id = md.get("tokenizer.ggml.bos_token_id")
    report.eos_id = md.get("tokenizer.ggml.eos_token_id")
    report.chat_template = bool(md.get("tokenizer.chat_template"))
    for value, label in ((report.bos_id, "BOS"), (report.eos_id, "EOS")):
        if value is not None and not 0 <= int(value) < len(tokens):
            report.problem("strife.tokenizer.bad_special",
                           f"the {label} token id {value} is outside the vocabulary")


def read_gguf(path: str | Path) -> tuple[GGUFFile | None, ModelReport]:
    """Parse and check *path*; the parsed file is ``None`` when it would not parse."""
    path = Path(path)
    report = ModelReport(path=str(path))
    try:
        size = os.stat(path).st_size
    except OSError as exc:
        report.problem("strife.model.unreadable", f"{path}: {exc.strerror or exc}")
        return None, report
    report.file_bytes = size
    try:
        gguf = GGUFFile.read(path)
    except (GGUFError, UnicodeDecodeError, ValueError, OverflowError, MemoryError) as exc:
        report.problem("strife.model.not_gguf", f"{path} is not a GGUF Strife can parse: {exc}",
                       "Check that the file is a complete .gguf download.")
        return None, report
    report.gguf_version = gguf.version
    report.tensor_count = len(gguf.tensors)
    if gguf.alignment <= 0 or gguf.alignment & (gguf.alignment - 1) or gguf.alignment > 1 << 16:
        report.problem("strife.model.bad_alignment",
                       f"alignment {gguf.alignment} is not a small power of two")
        return gguf, report
    _check_tensors(gguf, report, size)
    _check_architecture(gguf, report)
    _check_tokenizer(gguf, report)
    return gguf, report


def inspect_model(path: str | Path) -> ModelReport:
    """Read and check a GGUF without loading its weights."""
    return read_gguf(path)[1]


def require_runnable(report: ModelReport) -> None:
    if report.problems:
        first = report.problems[0]
        more = (f" (and {len(report.problems) - 1} more)" if len(report.problems) > 1 else "")
        raise StrifeError(first["code"], first["message"] + more, remedy=first["remedy"],
                          details={"problems": report.problems})
