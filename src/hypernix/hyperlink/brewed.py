"""Native HyperNix models, and HyperNix.3-mini as the runner's default.

The runner served GGUF files through llama.cpp and nothing else. A model
HyperNix trained itself, a ``hyperNix0x-v2`` checkpoint written by
:mod:`hypernix.training.brewer` (a folder with ``config.json``, the
weights and a tokenizer), could be loaded in Python and chatted with
through ``NeoOven``, but not served to HyperLink.

This finds such folders, fetches the default one, and describes them to
the catalogue. :mod:`hypernix.hyperlink.brewed_server` serves one over
the same OpenAI-compatible API llama-server speaks, so nothing
downstream has to know which kind of model is answering.

The default model
-----------------
``ray0rf1re/HyperNix.3-mini``: 48.7M parameters, trained from scratch on
one GTX 1080, 512 tokens of context. ``hypernix-t1 runner start`` with no
model named starts it, and downloads it the first time (about 195 MB).
``T1_DEFAULT_MODEL`` names a different default, and ``T1_DEFAULT_MODEL=``
(empty) turns the default off, which brings back "start the only model
on this machine".

It is a **base model**. It completes text and has not been taught to
follow instructions or to call tools, and at this size its text is
often not coherent. It is the default because it is HyperNix's own
model and small enough to run anywhere, not because it is a good
assistant, and the catalogue says so beside it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REPO",
    "DOWNLOAD_FILES",
    "BrewedError",
    "brewed_dirs",
    "default_model_id",
    "describe",
    "download",
    "is_brewed_dir",
    "matches_default",
]

DEFAULT_MODEL_REPO = "ray0rf1re/HyperNix.3-mini"
DEFAULT_MODEL_ID = "hypernix.3-mini"
#: The folder it is saved to under the models directory.
DEFAULT_MODEL_DIR = "HyperNix.3-mini"

#: What is fetched. Never ``model.pt``: it is the same weights as a
#: pickle, and a pickle is a file ``torch.load`` has to be told not to
#: execute.
DOWNLOAD_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
)

#: Keys that make a config.json a brewer one rather than a Hugging Face one.
_BREWER_KEYS = {"d_model", "n_layers", "n_heads", "n_kv_heads", "vocab_size"}
_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_WEIGHTS = ("model.safetensors", "model.pt", "pytorch_model.bin", "weights.pt")


class BrewedError(RuntimeError):
    """A native model could not be found, fetched or read."""


def default_model_id() -> str:
    """The runner's default model id, or ``""`` when the default is off."""
    if "T1_DEFAULT_MODEL" in os.environ:
        return os.environ["T1_DEFAULT_MODEL"].strip()
    return DEFAULT_MODEL_ID


def matches_default(model_id: str) -> bool:
    """Does *model_id* name the built-in default, however it is spelt?"""
    wanted = (model_id or "").strip().lower()
    return bool(wanted) and wanted in {
        DEFAULT_MODEL_ID, DEFAULT_MODEL_REPO.lower(), DEFAULT_MODEL_DIR.lower(),
    }


def _config(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_brewed_dir(path: str | Path) -> bool:
    """A folder holding a ``hyperNix0x-v2`` checkpoint with weights."""
    folder = Path(path)
    if not folder.is_dir():
        return False
    config = _config(folder)
    if config is None or not _BREWER_KEYS <= set(config):
        return False
    return any((folder / name).is_file() for name in _WEIGHTS)


def brewed_dirs(root: str | Path) -> list[Path]:
    """Native checkpoints in *root* and its immediate subfolders."""
    base = Path(root)
    if not base.is_dir():
        return []
    found = [base] if is_brewed_dir(base) else []
    try:
        children = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return found
    found += [p for p in children if is_brewed_dir(p)]
    return found


def describe(path: str | Path) -> dict[str, Any]:
    """What the catalogue shows for one native model."""
    folder = Path(path)
    config = _config(folder) or {}
    weights = next((folder / n for n in _WEIGHTS if (folder / n).is_file()), None)
    size = weights.stat().st_size if weights is not None else 0
    name = str(config.get("name") or folder.name)
    try:
        from ..training.brewer import BrewerConfig

        params = BrewerConfig.from_dict(config).approx_params()
    except Exception:  # noqa: BLE001 - a count is a nicety, not a requirement
        params = 0
    detail = "native HyperNix model (hyperNix0x-v2), served without llama.cpp"
    if matches_default(name) or folder.name == DEFAULT_MODEL_DIR:
        detail = ("the default model: a small base model that completes text. "
                  "It does not follow instructions or call tools")
    return {
        "model_id": name.lower(),
        "name": name,
        "path": str(folder),
        "size_bytes": size,
        "architecture": "hyperNix0x-v2",
        "parameters_b": round(params / 1e9, 4) if params else 0.0,
        "context_limit": int(config.get("max_seq_len") or 0),
        "detail": detail,
    }


def _hub_url(repo: str, filename: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    if not endpoint.startswith("https://"):
        raise BrewedError("HF_ENDPOINT must be an https:// URL")
    return f"{endpoint}/{repo}/resolve/main/{filename}"


def download(
    repo: str = DEFAULT_MODEL_REPO,
    dest: str | Path | None = None,
    *,
    models_dir: str | Path | None = None,
    timeout: float = 120.0,
) -> Path:
    """Fetch *repo*'s safetensors, config and tokenizer into *dest*.

    Files already there are kept, so a second call is free and an
    interrupted one resumes at the file it stopped on. Each file is
    written to ``.part`` and renamed when complete, so a folder never
    holds half a weights file that looks whole.
    """
    if not _REPO_ID.fullmatch(repo or ""):
        raise BrewedError(f"not a Hugging Face repo id: {repo!r}")
    if dest is None:
        from .catalogue import DEFAULT_LOCAL_DIR

        base = Path(models_dir) if models_dir else DEFAULT_LOCAL_DIR
        dest = base / (DEFAULT_MODEL_DIR if repo == DEFAULT_MODEL_REPO else repo.split("/")[1])
    folder = Path(dest)
    folder.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("T1_HF_TOKEN") or ""
    for filename in DOWNLOAD_FILES:
        target = folder / filename
        if target.is_file() and target.stat().st_size > 0:
            continue
        partial = folder / f"{filename}.part"
        request = urllib.request.Request(_hub_url(repo, filename))  # noqa: S310 - https checked
        request.add_header("User-Agent", "hypernix-runner")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        logger.info("brewed: fetching %s/%s", repo, filename)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, \
                    partial.open("wb") as out:  # noqa: S310
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            if filename in ("special_tokens_map.json", "tokenizer_config.json") and exc.code == 404:
                continue  # optional: the tokenizer works without them
            raise BrewedError(f"could not fetch {repo}/{filename}: HTTP {exc.code}") from exc
        except (OSError, urllib.error.URLError) as exc:
            partial.unlink(missing_ok=True)
            raise BrewedError(
                f"could not fetch {repo}/{filename}: {getattr(exc, 'reason', exc)}. "
                f"Download the repo into {folder} by hand to use it offline."
            ) from exc
        partial.replace(target)
    if not is_brewed_dir(folder):
        raise BrewedError(f"{repo} did not download as a HyperNix checkpoint into {folder}")
    return folder
