"""LFM2.5-2.6B, the runner's default model (0.72.8).

``hypernix-t1 runner start`` with no model named starts it, and the server
downloads it the first time: ``LFM2.5-2.6B-Q4_K_M.gguf`` from
``LiquidAI/LFM2.5-2.6B-GGUF``, about 1.7 GB. It is an instruction-tuned
model that follows a conversation and calls tools, which HyperNix.3-mini
(the default before, a 48.7M base model) could not; that one is still
fetched by name.

``lfm2.5-2.7b`` is accepted for it as well: Liquid AI publishes no 2.7B,
and the 2.6B is the model that name is reaching for.

``T1_DEFAULT_MODEL`` names a different default, and ``T1_DEFAULT_MODEL=``
(empty) turns the default off.
"""
from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from hypernix.security.safeurl import urlopen as safe_urlopen

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GGUF_FILE",
    "DEFAULT_GGUF_REPO",
    "DEFAULT_MODEL_ID",
    "DefaultModelError",
    "describe",
    "download",
    "matches",
]

DEFAULT_MODEL_ID = "lfm2.5-2.6b"
DEFAULT_GGUF_REPO = "LiquidAI/LFM2.5-2.6B-GGUF"
DEFAULT_GGUF_FILE = "LFM2.5-2.6B-Q4_K_M.gguf"
#: Its size on the Hub, said before a 1.7 GB download starts.
DEFAULT_GGUF_BYTES = 1_674_455_040
_NAMES = {
    DEFAULT_MODEL_ID, "lfm2.5-2.7b", "lfm2.5", "lfm-2.5-2.6b", "lfm2.5-2.6b-q4_k_m",
    "liquidai/lfm2.5-2.6b", DEFAULT_GGUF_REPO.lower(), DEFAULT_GGUF_FILE.lower(),
    DEFAULT_GGUF_FILE.lower().removesuffix(".gguf"),
}


class DefaultModelError(RuntimeError):
    """The default model could not be fetched."""


def matches(model_id: str) -> bool:
    """Does *model_id* name the default, however it is spelt?"""
    return (model_id or "").strip().lower() in _NAMES


def describe(path: str | Path) -> dict:
    file = Path(path)
    return {
        "model_id": DEFAULT_MODEL_ID, "path": str(file), "format": "gguf",
        "source": DEFAULT_GGUF_REPO, "quantization": "Q4_K_M",
        "size_bytes": file.stat().st_size if file.is_file() else DEFAULT_GGUF_BYTES,
        "detail": "the default model: Liquid AI's LFM2.5 2.6B, instruction-tuned",
    }


def download(models_dir: str | Path | None = None, *, timeout: float = 120.0) -> Path:
    """The default GGUF, fetched into the models directory if not already there.

    Written to ``.part`` and renamed when complete, so an interrupted
    download never leaves a file that looks whole.
    """
    if models_dir:
        base = Path(models_dir)
    else:
        from .catalogue import DEFAULT_LOCAL_DIR

        base = Path(DEFAULT_LOCAL_DIR)
    folder = base / "LFM2.5-2.6B"
    target = folder / DEFAULT_GGUF_FILE
    if target.is_file() and target.stat().st_size > 0:
        return target
    folder.mkdir(parents=True, exist_ok=True)
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    request = urllib.request.Request(  # noqa: S310 - https, through safe_urlopen
        f"{endpoint}/{DEFAULT_GGUF_REPO}/resolve/main/{DEFAULT_GGUF_FILE}")
    request.add_header("User-Agent", "hypernix-runner")
    token = os.environ.get("HF_TOKEN") or os.environ.get("T1_HF_TOKEN") or ""
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    partial = target.with_name(target.name + ".part")
    logger.info("defaultmodel: fetching %s/%s (%.1f GB)", DEFAULT_GGUF_REPO, DEFAULT_GGUF_FILE,
                DEFAULT_GGUF_BYTES / 1e9)
    try:
        with safe_urlopen(request, timeout=timeout) as response, partial.open("wb") as out:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise DefaultModelError(f"could not fetch {DEFAULT_GGUF_REPO}/{DEFAULT_GGUF_FILE}: "
                                f"HTTP {exc.code}") from exc
    except (OSError, urllib.error.URLError) as exc:
        partial.unlink(missing_ok=True)
        raise DefaultModelError(
            f"could not fetch {DEFAULT_GGUF_REPO}/{DEFAULT_GGUF_FILE}: {getattr(exc, 'reason', exc)}. "
            f"`hnx download {DEFAULT_GGUF_REPO} -f {DEFAULT_GGUF_FILE} -o {folder}` fetches it by hand."
        ) from exc
    with partial.open("rb") as check:
        if check.read(4) != b"GGUF":
            partial.unlink(missing_ok=True)
            raise DefaultModelError(f"{DEFAULT_GGUF_FILE} did not download as a GGUF")
    partial.replace(target)
    return target
