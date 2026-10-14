"""What ``hnx download`` was given: a repo, a link to one, or a link to a file.

All of these work (0.72.8)::

    hnx download ray0rf1re/hyper-nix.1
    hnx download https://huggingface.co/owner/repo
    hnx download https://huggingface.co/owner/repo/tree/dev
    hnx download https://huggingface.co/owner/repo/blob/main/model-Q4_K_M.gguf
    hnx download https://huggingface.co/owner/repo/resolve/main/sub/model.gguf?download=true
    hnx download hf://owner/repo/model.gguf
    hnx download owner/repo/model.gguf
    hnx download nano-mini                          (a short name)

Only model repositories: a dataset or a Space is refused by name rather
than being fetched as if it were a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

__all__ = ["HFSource", "LinkError", "parse_source", "pick_gguf"]

_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class LinkError(ValueError):
    """Not something ``download`` can fetch."""


@dataclass(frozen=True)
class HFSource:
    repo_id: str
    filename: str = ""
    revision: str | None = None

    @property
    def is_file(self) -> bool:
        return bool(self.filename)


def _repo(owner: str, name: str) -> str:
    if not (_PART.match(owner) and _PART.match(name)):
        raise LinkError(f"{owner}/{name} is not a Hugging Face repository name")
    return f"{owner}/{name}"


def _from_parts(parts: list[str], *, link: str) -> HFSource:
    if parts and parts[0] in ("datasets", "spaces"):
        raise LinkError(f"{link} is a {parts[0][:-1]}, not a model; download fetches models")
    if parts and parts[0] == "models":
        parts = parts[1:]
    if len(parts) < 2:
        raise LinkError(f"{link} does not name a repository (owner/name)")
    repo = _repo(parts[0], parts[1])
    rest = parts[2:]
    if not rest:
        return HFSource(repo)
    if rest[0] == "tree" and len(rest) >= 2:
        return HFSource(repo, revision=rest[1])
    if rest[0] in ("blob", "resolve", "raw") and len(rest) >= 3:
        return HFSource(repo, filename="/".join(rest[2:]), revision=rest[1])
    # owner/repo/path/to/file.gguf, as hf:// and bare forms write it.
    filename = "/".join(rest)
    if ".." in rest:
        raise LinkError(f"{link}: a file path cannot go up a directory")
    return HFSource(repo, filename=filename)


def parse_source(text: str) -> HFSource:
    """What *text* names. Raises :class:`LinkError` for anything else."""
    text = (text or "").strip()
    if not text:
        raise LinkError("give a repository, a link to one, or a link to a .gguf file")
    if text.startswith("hf://"):
        return _from_parts([unquote(p) for p in text[5:].split("/") if p], link=text)
    if "://" in text:
        url = urlsplit(text)
        if url.scheme not in ("http", "https") or url.hostname not in _HOSTS:
            raise LinkError(f"{text} is not a Hugging Face link (https://huggingface.co/owner/repo)")
        parts = [unquote(p) for p in url.path.split("/") if p]
        source = _from_parts(parts, link=text)
        if ".." in source.filename.split("/"):
            raise LinkError(f"{text}: a file path cannot go up a directory")
        return source
    parts = [p for p in text.split("/") if p]
    if len(parts) == 1:
        return HFSource(parts[0])  # a short name, resolved by download_model
    return _from_parts(parts, link=text)


def pick_gguf(files: list[str], prefer: str = "Q4_K_M") -> str | None:
    """The one GGUF to take from a repo that has several, or None to ask.

    One file is that file. Several: the *prefer* quant when exactly one
    file has it, else nobody can say and the caller lists them.
    """
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if len(ggufs) == 1:
        return ggufs[0]
    wanted = [f for f in ggufs if prefer.lower() in f.lower() and "mmproj" not in f.lower()]
    return wanted[0] if len(wanted) == 1 else None
