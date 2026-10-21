"""llama.cpp's own ``convert_hf_to_gguf.py``, for the models llama.cpp knows.

HyperNix's converter (:mod:`hypernix.quant.convert`) is architecture
agnostic: it writes ``general.architecture=hypernix`` and maps tensor
names by pattern. That is right for a HyperNix checkpoint and wrong for a
Llama, Qwen or LFM: llama.cpp needs its own architecture name, its own
metadata keys and, for Llama-style attention, Q and K permuted for its
rotation. llama.cpp's converter does all of that for every architecture
it supports, so ``hnx auto`` uses it for those and keeps HyperNix's for
the rest.

The converter is fetched from the official ``ggml-org/llama.cpp``
release tag (the same repository ``fetch-llama-quantize`` takes its
binary from); where GitHub cannot be reached, from the
``llama-cpp-python`` source release on PyPI, which vendors the same
llama.cpp tree and whose download is checked against PyPI's SHA-256.
Only the Python files and ``gguf-py`` are kept, cached per version under
``~/.cache/hypernix/llama-convert/``. Its ``gguf-py`` is the one beside
it, so the converter and the GGUF writer are always the same version.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

from .fetcher import LLAMA_CPP_REPO, _http_get, _latest_release, cache_dir

__all__ = ["LlamaConvertError", "convert_hf", "converter_dir", "fetch_converter", "fetch_from_pypi",
           "find_converter"]

SCRIPT = "convert_hf_to_gguf.py"
#: Pin the tag (e.g. b7000) instead of the latest release.
TAG_ENV = "HYPERNIX_LLAMA_CONVERT_TAG"


class LlamaConvertError(RuntimeError):
    """The converter could not be fetched, or could not convert this model."""

    def __init__(self, message: str, *, unsupported: bool = False) -> None:
        super().__init__(message)
        self.unsupported = unsupported


def converter_dir(tag: str) -> Path:
    return cache_dir().parent / "llama-convert" / tag


def _wanted(name: str, prefix: str = "") -> str | None:
    """The path to keep for a tarball member, or None.

    Drops the archive's top folder, and then *prefix* (for the PyPI
    sdist, ``vendor/llama.cpp/``).
    """
    parts = name.split("/", 1)
    if len(parts) != 2 or not parts[1]:
        return None
    rest = parts[1]
    if prefix:
        if not rest.startswith(prefix):
            return None
        rest = rest[len(prefix):]
        if not rest:
            return None
    top = rest.split("/", 1)[0]
    if "/" not in rest and rest.endswith(".py"):
        return rest
    if top in ("gguf-py", "conversion") or "convert" in top:
        return rest
    return None


def fetch_converter(tag: str | None = None, *, log: Callable[[str], None] | None = None) -> Path:
    """The folder holding ``convert_hf_to_gguf.py`` for *tag* (default: latest)."""
    say = log or (lambda msg: print(f"[hypernix] {msg}", file=sys.stderr))
    tag = tag or os.environ.get(TAG_ENV) or ""
    if not tag:
        cached = sorted((cache_dir().parent / "llama-convert").glob("*/" + SCRIPT))
        try:
            tag = str(_latest_release().get("tag_name") or "")
        except Exception as exc:  # noqa: BLE001 - offline: use what is cached
            if cached:
                return cached[-1].parent
            raise LlamaConvertError(f"could not ask GitHub for the latest llama.cpp release: {exc}") from exc
    folder = converter_dir(tag)
    if (folder / SCRIPT).is_file():
        return folder
    url = f"https://codeload.github.com/{LLAMA_CPP_REPO}/tar.gz/refs/tags/{tag}"
    say(f"fetching llama.cpp {tag}'s converter")
    try:
        data = _http_get(url, accept="application/gzip")
    except Exception as exc:  # noqa: BLE001
        raise LlamaConvertError(f"could not download llama.cpp {tag}: {exc}") from exc
    _unpack(data, folder)
    if not (folder / SCRIPT).is_file():
        raise LlamaConvertError(f"llama.cpp {tag} has no {SCRIPT}")
    return folder


def _unpack(data: bytes, folder: Path, prefix: str = "") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as fh:
        fh.write(data)
        fh.flush()
        with tarfile.open(fh.name, "r:gz") as tar:
            for member in tar.getmembers():
                keep = _wanted(member.name, prefix)
                if keep is None or not (member.isfile() or member.isdir()):
                    continue
                if ".." in Path(keep).parts:
                    continue
                member.name = keep
                tar.extract(member, folder, filter="data")


#: Pin the llama-cpp-python release whose vendored converter is used.
PYPI_VERSION_ENV = "HYPERNIX_LLAMA_CONVERT_PYPI"


def fetch_from_pypi(version: str | None = None, *, log: Callable[[str], None] | None = None) -> Path:
    """The converter vendored in the ``llama-cpp-python`` sdist on PyPI."""
    import hashlib
    import json

    say = log or (lambda msg: print(f"[hypernix] {msg}", file=sys.stderr))
    version = version or os.environ.get(PYPI_VERSION_ENV) or ""
    index = "https://pypi.org/pypi/llama-cpp-python" + (f"/{version}" if version else "") + "/json"
    try:
        meta = json.loads(_http_get(index))
    except Exception as exc:  # noqa: BLE001
        raise LlamaConvertError(f"could not ask PyPI for llama-cpp-python: {exc}") from exc
    version = meta["info"]["version"]
    folder = cache_dir().parent / "llama-convert" / f"pypi-{version}"
    if (folder / SCRIPT).is_file():
        return folder
    sdist = next((f for f in meta.get("urls", []) if f.get("packagetype") == "sdist"), None)
    if sdist is None:
        raise LlamaConvertError(f"llama-cpp-python {version} has no source release on PyPI")
    say(f"fetching llama.cpp's converter from llama-cpp-python {version} "
        f"({sdist.get('size', 0) / 1e6:.0f} MB)")
    data = _http_get(sdist["url"], accept="application/octet-stream")
    digest = hashlib.sha256(data).hexdigest()
    if digest != sdist.get("digests", {}).get("sha256"):
        raise LlamaConvertError(f"llama-cpp-python {version}: the download does not match PyPI's SHA-256")
    _unpack(data, folder, prefix="vendor/llama.cpp/")
    if not (folder / SCRIPT).is_file():
        raise LlamaConvertError(f"llama-cpp-python {version} has no vendored {SCRIPT}")
    return folder


def find_converter(tag: str | None = None, *, log: Callable[[str], None] | None = None) -> Path:
    """GitHub's release first; PyPI's vendored copy when GitHub cannot be reached."""
    try:
        return fetch_converter(tag, log=log)
    except LlamaConvertError as github:
        if tag:
            raise
        try:
            return fetch_from_pypi(log=log)
        except LlamaConvertError as pypi:
            raise LlamaConvertError(f"{github}; and from PyPI: {pypi}") from pypi


def convert_hf(model_dir: str | Path, output: str | Path, *, outtype: str = "f16",
               tag: str | None = None, log: Callable[[str], None] | None = None) -> Path:
    """Convert a Hugging Face folder to a GGUF llama.cpp can run."""
    folder = find_converter(tag, log=log)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "NO_LOCAL_GGUF"}
    done = subprocess.run(
        [sys.executable, str(folder / SCRIPT), str(model_dir), "--outfile", str(output),
         "--outtype", outtype],
        capture_output=True, text=True, env=env, check=False,
    )
    if done.returncode != 0 or not output.is_file():
        tail = "\n".join((done.stderr or done.stdout or "").strip().splitlines()[-6:])
        unsupported = "not supported" in tail.lower() or "unrecognized" in tail.lower()
        raise LlamaConvertError(f"llama.cpp's converter failed:\n{tail}", unsupported=unsupported)
    return output
