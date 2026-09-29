"""hyperlink.hfmerge — turn "a link someone pasted" into a runnable model.

The situation this exists for: somebody on a phone finds a GGUF they
want. What they actually have is one or two URLs from a browser, and
neither is sufficient on its own.

* A **model page** — ``https://huggingface.co/bartowski/Qwen3-8B-GGUF``
  — says which repository, and (via the HF API) which files exist, how
  big they are, and whether there is a vision projector alongside. It
  does not say *which quant* they want; that repo has fourteen.
* A **direct file link** — what people call a "dflash link", the
  ``.../resolve/main/Qwen3-8B-Q4_K_M.gguf?download=true`` URL behind the
  download arrow — says exactly which bytes. It does not say anything
  about the rest of the repo, and for a vision model or a split GGUF
  those bytes alone will not load.

:func:`merge` takes either or both and produces one
:class:`ResolvedModel`: a concrete download plan, with every file that
must be fetched for the thing to actually run, in the order to fetch
them. That is the "merge the two so it runs properly" step, and it is
three separate pieces of knowledge:

1. **Split GGUFs.** ``model-00001-of-00003.gguf`` is one third of a
   model. Downloading the part someone happened to click gives a file
   llama.cpp will refuse. The resolver recognises the
   ``-NNNNN-of-NNNNN`` suffix and pulls the whole set.
2. **Vision projectors.** A VLM's ``mmproj-*.gguf`` lives beside the
   weights and is a separate file. Without it the model loads and then
   cannot see images — which is a much more confusing failure than not
   loading at all, and it is the failure mode for the app's "send a
   photo" button.
3. **Repository conflicts.** When the page and the direct link disagree
   about the repository, that is a real mistake worth surfacing rather
   than resolving silently: two tabs open, wrong one copied.
   :func:`merge` raises unless ``prefer`` says which side wins.

Everything here is standard library. ``huggingface_hub`` is a hypernix
dependency and is the right tool for *downloading*, but resolution has
to work on the API server, inside the iOS app's request path, and in
tests with no network, so it is plain HTTP against the public API with
an offline path that still parses URLs correctly.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from hypernix.security.safeurl import urlopen as safe_urlopen

__all__ = [
    "GGUFFile",
    "HFRef",
    "ResolvedModel",
    "HFResolveError",
    "parse_link",
    "merge",
    "resolve",
    "split_siblings",
    "guess_quantization",
]

HF_HOSTS = ("huggingface.co", "hf.co", "www.huggingface.co", "hf-mirror.com")
DEFAULT_HOST = "huggingface.co"
DEFAULT_REVISION = "main"

#: ``Meta-Llama-3-70B-Q5_K_M-00002-of-00005.gguf`` → part 2 of 5.
_SPLIT_RE = re.compile(r"^(?P<stem>.+?)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)

#: The quant tags llama.cpp emits. Ordered longest-first at match time so
#: ``Q4_K_M`` is not shadowed by ``Q4_K``.
_QUANT_TAGS = (
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS",
    "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL",
    "Q2_K_S", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K",
    "Q4_K_S", "Q4_K_M", "Q4_K", "Q4_0", "Q4_1",
    "Q5_K_S", "Q5_K_M", "Q5_K", "Q5_0", "Q5_1",
    "Q6_K", "Q8_0", "BF16", "F16", "FP16", "F32", "FP32",
)


class HFResolveError(ValueError):
    """A link could not be turned into a download plan.

    ``code`` is stable for callers that branch: ``not_a_hf_link``,
    ``no_repo``, ``repo_conflict``, ``file_not_found``, ``no_gguf``,
    ``api_error``, ``offline``.
    """

    def __init__(self, message: str, *, code: str = "invalid_link", hint: str = ""):
        super().__init__(message)
        self.code = code
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "hint": self.hint}


@dataclass(frozen=True)
class HFRef:
    """What a single pasted link told us. Fields are empty when unknown."""

    repo_id: str = ""
    revision: str = DEFAULT_REVISION
    filename: str = ""
    host: str = DEFAULT_HOST
    kind: str = "unknown"          # page | tree | blob | resolve | shorthand
    raw: str = ""

    @property
    def has_repo(self) -> bool:
        return bool(self.repo_id)

    @property
    def has_file(self) -> bool:
        return bool(self.filename)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "filename": self.filename,
            "host": self.host,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class GGUFFile:
    """One file in the download plan."""

    filename: str
    url: str
    size_bytes: int = 0
    role: str = "weights"          # weights | weights-part | mmproj | config | tokenizer
    part_index: int = 0
    part_total: int = 0
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "url": self.url,
            "size_bytes": self.size_bytes,
            "role": self.role,
            "part_index": self.part_index,
            "part_total": self.part_total,
            "sha256": self.sha256,
        }


@dataclass
class ResolvedModel:
    """Everything needed to fetch and run one GGUF model."""

    repo_id: str
    revision: str
    files: list[GGUFFile]
    host: str = DEFAULT_HOST
    quantization: str = ""
    gated: bool = False
    private: bool = False
    license: str = ""
    total_bytes: int = 0
    warnings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    metadata_from_api: bool = False

    @property
    def primary(self) -> GGUFFile | None:
        """The file a loader is pointed at: part 1, or the single weights file."""
        for f in self.files:
            if f.role == "weights":
                return f
        for f in self.files:
            if f.role == "weights-part" and f.part_index == 1:
                return f
        return self.files[0] if self.files else None

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def is_split(self) -> bool:
        return any(f.role == "weights-part" for f in self.files)

    @property
    def has_vision(self) -> bool:
        return any(f.role == "mmproj" for f in self.files)

    def to_dict(self) -> dict[str, Any]:
        primary = self.primary
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "host": self.host,
            "quantization": self.quantization,
            "gated": self.gated,
            "private": self.private,
            "license": self.license,
            "total_bytes": self.total_bytes,
            "total_size_human": human_bytes(self.total_bytes),
            "file_count": self.file_count,
            "is_split": self.is_split,
            "has_vision": self.has_vision,
            "primary_file": primary.filename if primary else "",
            "files": [f.to_dict() for f in self.files],
            "warnings": list(self.warnings),
            "sources": list(self.sources),
            "metadata_from_api": self.metadata_from_api,
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_link(link: str) -> HFRef:
    """Parse any of the shapes a Hugging Face link comes in.

    Accepted, all of them things people actually paste::

        https://huggingface.co/owner/repo
        https://huggingface.co/owner/repo/tree/main
        https://huggingface.co/owner/repo/blob/main/model-Q4_K_M.gguf
        https://huggingface.co/owner/repo/resolve/main/model.gguf?download=true
        https://hf.co/owner/repo
        hf://owner/repo/model.gguf
        owner/repo
        owner/repo:Q4_K_M

    The ``:QUANT`` shorthand is not a Hugging Face convention — it is an
    Ollama one — but it is what people type, and turning it into "pick
    the Q4_K_M file from that repo" costs one line and saves a support
    question.
    """
    text = (link or "").strip()
    if not text:
        raise HFResolveError("Empty link", code="not_a_hf_link")

    # hf://owner/repo[/path...]
    if text.startswith("hf://"):
        return _parse_path(text[5:], host=DEFAULT_HOST, raw=text, scheme_kind="shorthand")

    if "://" in text:
        parsed = urllib.parse.urlparse(text)
        host = (parsed.hostname or "").lower()
        if host not in HF_HOSTS:
            raise HFResolveError(
                f"{host or text!r} is not a Hugging Face address. "
                "Paste a huggingface.co model page or file link.",
                code="not_a_hf_link",
                hint="Expected a link starting https://huggingface.co/",
            )
        return _parse_path(parsed.path.lstrip("/"), host=host, raw=text, scheme_kind="")

    # Bare shorthand: owner/repo, optionally owner/repo:QUANT
    quant = ""
    if ":" in text:
        text, _, quant = text.partition(":")
    ref = _parse_path(text.strip("/"), host=DEFAULT_HOST, raw=link, scheme_kind="shorthand")
    if quant:
        # Recorded as a filename *hint*, not a filename: the real name
        # is discovered from the repo listing during resolution.
        return HFRef(
            repo_id=ref.repo_id,
            revision=ref.revision,
            filename=f"*{quant.strip()}*",
            host=ref.host,
            kind="shorthand",
            raw=link,
        )
    return ref


def _parse_path(path: str, *, host: str, raw: str, scheme_kind: str) -> HFRef:
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise HFResolveError(
            f"{raw!r} does not name a repository. A Hugging Face repo is owner/name, "
            "e.g. bartowski/Qwen3-8B-GGUF.",
            code="no_repo",
        )
    # "models/" prefixes appear on API URLs people sometimes paste.
    if parts[0] in ("models", "api") and len(parts) >= 3:
        parts = parts[1:]
        if parts and parts[0] == "models" and len(parts) >= 3:
            parts = parts[1:]
    owner, name = parts[0], parts[1]
    repo_id = f"{owner}/{name}"
    rest = parts[2:]

    if not rest:
        return HFRef(repo_id=repo_id, host=host, kind=scheme_kind or "page", raw=raw)

    marker = rest[0]
    if marker in ("blob", "resolve", "tree", "raw"):
        revision = urllib.parse.unquote(rest[1]) if len(rest) > 1 else DEFAULT_REVISION
        filename = "/".join(rest[2:]) if len(rest) > 2 else ""
        kind = "tree" if marker == "tree" else ("resolve" if marker == "resolve" else "blob")
        return HFRef(
            repo_id=repo_id,
            revision=revision,
            filename=urllib.parse.unquote(filename),
            host=host,
            kind=kind if filename or marker == "tree" else "page",
            raw=raw,
        )
    # hf://owner/repo/file.gguf — no marker, the rest is the path.
    return HFRef(
        repo_id=repo_id,
        filename=urllib.parse.unquote("/".join(rest)),
        host=host,
        kind=scheme_kind or "blob",
        raw=raw,
    )


def guess_quantization(filename: str) -> str:
    """Pull ``Q4_K_M`` out of a GGUF filename, or return ``""``.

    Matched longest-first and bounded by a non-alphanumeric on each side,
    so ``...-Q4_K_M.gguf`` gives ``Q4_K_M`` rather than ``Q4_K``, and a
    repo called ``Q8-Research`` does not read as a quant.
    """
    upper = filename.upper()
    for tag in sorted(_QUANT_TAGS, key=len, reverse=True):
        for match in re.finditer(re.escape(tag), upper):
            before = upper[match.start() - 1] if match.start() else "-"
            after_pos = match.end()
            after = upper[after_pos] if after_pos < len(upper) else "."
            if not before.isalnum() and not after.isalnum():
                return tag
    return ""


def split_siblings(filename: str) -> tuple[str, int, int]:
    """``(stem, index, total)`` for a split GGUF, or ``(filename, 0, 0)``."""
    match = _SPLIT_RE.match(filename)
    if not match:
        return filename, 0, 0
    return match["stem"], int(match["index"]), int(match["total"])


def _download_url(host: str, repo_id: str, revision: str, filename: str) -> str:
    quoted = urllib.parse.quote(filename)
    return f"https://{host}/{repo_id}/resolve/{urllib.parse.quote(revision, safe='')}/{quoted}?download=true"


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"


# ---------------------------------------------------------------------------
# The HF API
# ---------------------------------------------------------------------------


def fetch_repo_info(
    repo_id: str,
    *,
    revision: str = DEFAULT_REVISION,
    host: str = DEFAULT_HOST,
    token: str = "",
    timeout: float = 20.0,
) -> dict[str, Any]:
    """``GET /api/models/{repo}/revision/{rev}``. Raises on failure."""
    url = f"https://{host}/api/models/{repo_id}/revision/{urllib.parse.quote(revision, safe='')}"
    headers = {"Accept": "application/json", "User-Agent": "hypernix-hyperlink/1.0.26.8.0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with safe_urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise HFResolveError(
                f"{repo_id} is gated or private. Accept its licence on the model page "
                "and supply a Hugging Face token.",
                code="api_error",
                hint="Set HF_TOKEN, or pass --hf-token.",
            ) from exc
        if exc.code == 404:
            raise HFResolveError(
                f"No repository {repo_id!r} at revision {revision!r} on {host}.",
                code="no_repo",
            ) from exc
        raise HFResolveError(
            f"Hugging Face returned HTTP {exc.code} for {repo_id}", code="api_error"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HFResolveError(
            f"Could not reach {host} to look up {repo_id}: {exc}", code="offline"
        ) from exc
    except json.JSONDecodeError as exc:
        raise HFResolveError(
            f"Hugging Face returned non-JSON metadata for {repo_id}", code="api_error"
        ) from exc


# ---------------------------------------------------------------------------
# Merge + resolve
# ---------------------------------------------------------------------------


def merge(
    page_link: str = "",
    file_link: str = "",
    *,
    prefer: str = "strict",
) -> HFRef:
    """Combine a page link and a direct file link into one reference.

    ``prefer`` decides what happens when the two disagree about the
    repository:

    * ``"strict"`` (default) — raise. Two links naming two repos is a
      mistake, and quietly downloading one of them is how someone ends
      up with a model they did not choose.
    * ``"file"`` — the direct link wins. It names the actual bytes.
    * ``"page"`` — the page wins, and the filename is carried over to be
      looked for in that repo. Useful for a mirror: page on one host,
      filename discovered from another.

    Revisions are merged the same way, except that a link left at the
    default ``main`` never overrides an explicit one — pasting a page
    URL should not silently move a pinned revision back to ``main``.
    """
    if not page_link and not file_link:
        raise HFResolveError("Supply a model page link, a file link, or both", code="no_repo")

    page = parse_link(page_link) if page_link else None
    direct = parse_link(file_link) if file_link else None

    if page is None:
        if direct is None:
            raise AssertionError('direct is not None')
        return direct
    if direct is None:
        return page

    if page.repo_id != direct.repo_id:
        if prefer == "file":
            winner, loser = direct, page
        elif prefer == "page":
            winner = HFRef(
                repo_id=page.repo_id,
                revision=page.revision,
                filename=direct.filename,
                host=page.host,
                kind="merged",
                raw=f"{page_link} + {file_link}",
            )
            return winner
        else:
            raise HFResolveError(
                f"Those two links are different repositories: the page says "
                f"{page.repo_id!r} and the file link says {direct.repo_id!r}. "
                "Check you copied both from the same model, or choose which one wins.",
                code="repo_conflict",
                hint="Pass prefer='file' to trust the download link, or prefer='page'.",
            )
        return HFRef(
            repo_id=winner.repo_id,
            revision=winner.revision,
            filename=winner.filename or loser.filename,
            host=winner.host,
            kind="merged",
            raw=f"{page_link} + {file_link}",
        )

    # Same repo: take the filename from whichever link has one, and the
    # more specific revision.
    revision = direct.revision if direct.revision != DEFAULT_REVISION else page.revision
    return HFRef(
        repo_id=page.repo_id,
        revision=revision,
        filename=direct.filename or page.filename,
        host=direct.host or page.host,
        kind="merged",
        raw=f"{page_link} + {file_link}".strip(" +"),
    )


def resolve(
    page_link: str = "",
    file_link: str = "",
    *,
    prefer: str = "strict",
    token: str = "",
    include_vision: bool = True,
    offline: bool = False,
    timeout: float = 20.0,
) -> ResolvedModel:
    """Merge the links, ask Hugging Face what is in the repo, plan the download.

    With ``offline=True`` (or when the API is unreachable and the links
    were specific enough) this still returns a usable plan built purely
    from the URLs — one file, no size, no sibling detection — and says
    so via ``metadata_from_api=False`` and a warning. A phone on a bad
    connection should still be able to start a download it has the exact
    URL for.
    """
    ref = merge(page_link, file_link, prefer=prefer)
    sources = [link for link in (page_link, file_link) if link]

    if offline:
        return _offline_plan(ref, sources, "Resolved from the link alone (offline mode).")

    try:
        info = fetch_repo_info(
            ref.repo_id, revision=ref.revision, host=ref.host, token=token, timeout=timeout
        )
    except HFResolveError as exc:
        if exc.code == "offline" and ref.has_file and not _is_glob(ref.filename):
            return _offline_plan(
                ref, sources, f"Could not reach {ref.host} ({exc}); using the link as given."
            )
        raise

    siblings = [
        str(s.get("rfilename") or "")
        for s in (info.get("siblings") or [])
        if isinstance(s, dict) and s.get("rfilename")
    ]
    sizes = _sizes_from_info(info)
    gguf_files = [s for s in siblings if s.lower().endswith(".gguf")]
    if not gguf_files:
        raise HFResolveError(
            f"{ref.repo_id} has no .gguf files — it is probably a safetensors repo. "
            "Look for a matching '-GGUF' repository, or quantise it with `hypernix quantize`.",
            code="no_gguf",
        )

    chosen = _choose_file(ref, gguf_files)
    warnings: list[str] = []
    files = _plan_files(ref, chosen, gguf_files, sizes, include_vision, warnings)

    revision = str(info.get("sha") or ref.revision or DEFAULT_REVISION)
    # Keep the human-readable revision for URLs (a branch name still
    # resolves, and a commit sha in every URL is unreadable in logs);
    # the sha is reported separately by callers that pin.
    model = ResolvedModel(
        repo_id=ref.repo_id,
        revision=ref.revision or DEFAULT_REVISION,
        files=files,
        host=ref.host,
        quantization=guess_quantization(chosen),
        gated=bool(info.get("gated")),
        private=bool(info.get("private")),
        license=str((info.get("cardData") or {}).get("license") or ""),
        total_bytes=sum(f.size_bytes for f in files),
        warnings=warnings,
        sources=sources,
        metadata_from_api=True,
    )
    if model.gated:
        model.warnings.append(
            f"{ref.repo_id} is gated: accept its licence on the model page and use an HF token."
        )
    if revision and revision != model.revision:
        model.warnings.append(f"Revision {model.revision!r} currently points at commit {revision[:12]}.")
    return model


def _is_glob(name: str) -> bool:
    return "*" in name or "?" in name


def _choose_file(ref: HFRef, gguf_files: list[str]) -> str:
    """Decide which GGUF the reference means."""
    if ref.has_file and not _is_glob(ref.filename):
        if ref.filename in gguf_files:
            return ref.filename
        # A file link into a repo whose listing does not contain it:
        # usually a revision mismatch, occasionally a renamed file.
        raise HFResolveError(
            f"{ref.repo_id} has no file {ref.filename!r} at revision {ref.revision!r}. "
            f"Available: {', '.join(sorted(gguf_files)[:6])}"
            + (" …" if len(gguf_files) > 6 else ""),
            code="file_not_found",
        )
    if ref.has_file:  # a "*Q4_K_M*" hint from owner/repo:QUANT
        pattern = ref.filename.strip("*").upper()
        matches = [f for f in gguf_files if pattern in f.upper()]
        if not matches:
            raise HFResolveError(
                f"No {pattern} quantisation in {ref.repo_id}. "
                f"Available: {', '.join(sorted({guess_quantization(f) for f in gguf_files} - {''}))}",
                code="file_not_found",
            )
        return _pick_default(matches)
    return _pick_default(gguf_files)


#: Preference order when nobody said which quant. Q4_K_M first because
#: it is the one that fits on the most hardware while still being worth
#: running; then upward, then downward. This is a default, not a claim
#: that it is best for any given machine.
_QUANT_PREFERENCE = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0", "Q3_K_M", "IQ4_XS", "Q2_K")


def _pick_default(candidates: list[str]) -> str:
    """Choose one file from several, preferring part 1 of any split set."""
    # Never default to a middle part of a split model.
    non_middle = [c for c in candidates if split_siblings(c)[1] in (0, 1)]
    pool = non_middle or candidates
    # Skip vision projectors — they are companions, never the model.
    weights = [c for c in pool if not _is_mmproj(c)] or pool
    for quant in _QUANT_PREFERENCE:
        for name in sorted(weights):
            if guess_quantization(name) == quant:
                return name
    return sorted(weights)[0]


def _is_mmproj(filename: str) -> bool:
    base = filename.rsplit("/", 1)[-1].lower()
    return base.startswith("mmproj") or "mmproj" in base


def _plan_files(
    ref: HFRef,
    chosen: str,
    all_gguf: list[str],
    sizes: dict[str, int],
    include_vision: bool,
    warnings: list[str],
) -> list[GGUFFile]:
    files: list[GGUFFile] = []
    stem, index, total = split_siblings(chosen)

    if total:
        # Pull the whole split set, in order, whichever part was clicked.
        parts: list[tuple[int, str]] = []
        for name in all_gguf:
            p_stem, p_index, p_total = split_siblings(name)
            if p_stem == stem and p_total == total:
                parts.append((p_index, name))
        parts.sort()
        found = {i for i, _ in parts}
        missing = sorted(set(range(1, total + 1)) - found)
        if missing:
            warnings.append(
                f"Split model is incomplete on the hub: part(s) "
                f"{', '.join(f'{m:05d}' for m in missing)} of {total:05d} are missing."
            )
        for part_index, name in parts:
            files.append(
                GGUFFile(
                    filename=name,
                    url=_download_url(ref.host, ref.repo_id, ref.revision, name),
                    size_bytes=sizes.get(name, 0),
                    role="weights-part",
                    part_index=part_index,
                    part_total=total,
                )
            )
        if index:
            warnings.append(
                f"{chosen} is part {index} of {total}; all {len(parts)} parts are included "
                "because llama.cpp needs the full set to load."
            )
    else:
        files.append(
            GGUFFile(
                filename=chosen,
                url=_download_url(ref.host, ref.repo_id, ref.revision, chosen),
                size_bytes=sizes.get(chosen, 0),
                role="weights",
            )
        )

    if include_vision:
        projectors = [f for f in all_gguf if _is_mmproj(f)]
        if projectors:
            projector = _match_projector(chosen, projectors)
            files.append(
                GGUFFile(
                    filename=projector,
                    url=_download_url(ref.host, ref.repo_id, ref.revision, projector),
                    size_bytes=sizes.get(projector, 0),
                    role="mmproj",
                )
            )
            warnings.append(
                f"Vision projector {projector} included — without it the model loads "
                "but cannot read images."
            )
    return files


def _match_projector(chosen: str, projectors: list[str]) -> str:
    """Pick the projector that goes with this quant, if there are several.

    Repos that ship both ``mmproj-F16.gguf`` and ``mmproj-Q8_0.gguf``
    are common. Matching the weights' own quant is a reasonable default;
    F16 is the fallback because a projector is small and the
    higher-precision one is rarely the wrong choice.
    """
    quant = guess_quantization(chosen)
    if quant:
        for name in projectors:
            if guess_quantization(name) == quant:
                return name
    for name in projectors:
        if guess_quantization(name) in ("F16", "FP16", "BF16"):
            return name
    return sorted(projectors)[0]


def _sizes_from_info(info: dict[str, Any]) -> dict[str, int]:
    """File sizes, when the API was asked for them.

    ``/api/models/{repo}`` only carries sizes when queried with
    ``?blobs=true``, which needs auth for some repos. A plan with zero
    sizes is still a valid plan — it just cannot show a progress total —
    so a missing size is recorded as 0 rather than treated as an error.
    """
    sizes: dict[str, int] = {}
    for sibling in info.get("siblings") or []:
        if not isinstance(sibling, dict):
            continue
        name = str(sibling.get("rfilename") or "")
        size = sibling.get("size")
        if name and isinstance(size, int):
            sizes[name] = size
        lfs = sibling.get("lfs")
        if name and isinstance(lfs, dict) and isinstance(lfs.get("size"), int):
            sizes[name] = int(lfs["size"])
    return sizes


def _offline_plan(ref: HFRef, sources: list[str], note: str) -> ResolvedModel:
    if not ref.has_file or _is_glob(ref.filename):
        raise HFResolveError(
            "Offline resolution needs a direct file link — a model page alone does not "
            "say which quantisation to download.",
            code="offline",
            hint="Paste the download-arrow link from the Files tab as well.",
        )
    stem, index, total = split_siblings(ref.filename)
    files: list[GGUFFile] = []
    if total:
        for part in range(1, total + 1):
            name = f"{stem}-{part:05d}-of-{total:05d}.gguf"
            files.append(
                GGUFFile(
                    filename=name,
                    url=_download_url(ref.host, ref.repo_id, ref.revision, name),
                    role="weights-part",
                    part_index=part,
                    part_total=total,
                )
            )
    else:
        files.append(
            GGUFFile(
                filename=ref.filename,
                url=_download_url(ref.host, ref.repo_id, ref.revision, ref.filename),
                role="weights",
            )
        )
    warnings = [note]
    if total:
        warnings.append(
            f"Split model: all {total} part names derived from the link. "
            "Sizes are unknown until the download starts."
        )
    return ResolvedModel(
        repo_id=ref.repo_id,
        revision=ref.revision,
        files=files,
        host=ref.host,
        quantization=guess_quantization(ref.filename),
        warnings=warnings,
        sources=sources,
        metadata_from_api=False,
    )
