"""hyperlink.hfdownload — fetch models from Hugging Face, with a token.

:mod:`hypernix.hyperlink.hfmerge` decides *what* to download. This
downloads it.

The split matters because the two have different failure modes.
Resolution fails fast and cheaply — a bad link, a repo with no GGUF —
and can be retried instantly. A download fails slowly and expensively,
usually after several gigabytes, and the only acceptable response is to
resume rather than start again. So everything here is built around
resumption:

* Files land at ``<name>.part`` and are renamed only when complete, so a
  killed process never leaves a truncated file that looks finished.
* A partial ``.part`` is resumed with a ``Range`` request. A server that
  ignores the range header is detected (it answers 200 rather than 206)
  and the file is restarted rather than being appended to, which would
  produce a corrupt model that downloads "successfully".
* Progress is reported per chunk through a callback, so the phone and
  the live-stream TUI show real movement rather than a spinner.

Both PyTorch repos and GGUF repos work: the resolver decides which files
are needed and this fetches them, in order, into one directory.

Tokens
------
A Hugging Face token is needed for gated and private repositories and is
useful everywhere else for the higher rate limit. It is read from the
argument, then ``HF_TOKEN``, then ``HUGGING_FACE_HUB_TOKEN`` — the two
names the ecosystem actually uses. It is sent as a bearer header and
never logged: :func:`_redact` exists because a 401 traceback that
includes the Authorization header is how tokens end up in issue reports.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hypernix.security.safeurl import urlopen as safe_urlopen

logger = logging.getLogger(__name__)

__all__ = [
    "HFDownloadError",
    "DownloadProgress",
    "DownloadResult",
    "HFDownloader",
    "resolve_token",
    "PYTORCH_PATTERNS",
]

#: Files a PyTorch repo needs to load. Config and tokeniser are small and
#: mandatory; the weights are whichever sharding scheme the repo uses.
PYTORCH_PATTERNS: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.bin.index.json",
)

_USER_AGENT = "hypernix-hyperlink/1.0.26.8.1.0"
_CHUNK = 1024 * 1024


class HFDownloadError(RuntimeError):
    """A download failed. ``code`` is stable for callers that branch."""

    def __init__(self, message: str, *, code: str = "error", hint: str = ""):
        super().__init__(message)
        self.code = code
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "hint": self.hint}


def resolve_token(token: str | None = None) -> str:
    """The HF token to use, from the argument or the usual environment.

    ``HF_TOKEN`` first because it is what the current CLI writes;
    ``HUGGING_FACE_HUB_TOKEN`` second because it is what older tooling
    set and plenty of machines still have.
    """
    return (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()


def _redact(text: str, token: str) -> str:
    """Never let a token reach a log line or an error message."""
    if token and token in text:
        return text.replace(token, "hf_***")
    return text


@dataclass
class DownloadProgress:
    """One progress tick. Passed to the caller's callback."""

    filename: str
    downloaded: int
    total: int
    file_index: int
    file_count: int
    bytes_per_second: float = 0.0
    resumed: bool = False

    @property
    def fraction(self) -> float:
        return (self.downloaded / self.total) if self.total else 0.0

    @property
    def eta_seconds(self) -> float:
        if not self.bytes_per_second or not self.total:
            return 0.0
        return max(0.0, (self.total - self.downloaded) / self.bytes_per_second)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "downloaded": self.downloaded,
            "total": self.total,
            "fraction": round(self.fraction, 4),
            "file_index": self.file_index,
            "file_count": self.file_count,
            "bytes_per_second": round(self.bytes_per_second, 1),
            "eta_seconds": round(self.eta_seconds, 1),
            "resumed": self.resumed,
        }


@dataclass
class DownloadResult:
    repo_id: str
    directory: Path
    files: list[dict[str, Any]] = field(default_factory=list)
    total_bytes: int = 0
    seconds: float = 0.0
    resumed_files: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "directory": str(self.directory),
            "files": list(self.files),
            "file_count": len(self.files),
            "total_bytes": self.total_bytes,
            "seconds": round(self.seconds, 2),
            "resumed_files": self.resumed_files,
        }


class HFDownloader:
    """Fetches files from a Hugging Face repository.

    Standard library only. ``huggingface_hub`` is a hypernix dependency
    and is excellent, but this runs inside the T1 API's request path and
    on a machine that may only have the client installed, so it uses
    plain HTTP against the same public endpoints.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        host: str = "huggingface.co",
        timeout: float = 60.0,
        progress: Callable[[DownloadProgress], None] | None = None,
        max_retries: int = 3,
    ) -> None:
        self.token = resolve_token(token)
        self.host = host
        self.timeout = float(timeout)
        self.progress = progress
        self.max_retries = int(max_retries)

    # -- metadata -----------------------------------------------------

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if extra:
            headers.update(extra)
        return headers

    def repo_info(self, repo_id: str, *, revision: str = "main") -> dict[str, Any]:
        """Repository metadata, including the file listing."""
        import json

        url = (
            f"https://{self.host}/api/models/{repo_id}"
            f"/revision/{urllib.parse.quote(revision, safe='')}"
        )
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with safe_urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise HFDownloadError(
                    f"{repo_id} is gated or private and this token does not open it.",
                    code="unauthorized",
                    hint=(
                        "Accept the licence on the model page, then set HF_TOKEN to a token "
                        "with read access."
                        if self.token
                        else "Set HF_TOKEN, or pass a token."
                    ),
                ) from exc
            if exc.code == 404:
                raise HFDownloadError(f"No repository {repo_id!r}", code="not_found") from exc
            raise HFDownloadError(
                f"Hugging Face returned HTTP {exc.code} for {repo_id}", code="http_error"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise HFDownloadError(
                _redact(f"Could not reach {self.host}: {exc}", self.token), code="offline"
            ) from exc

    def list_files(self, repo_id: str, *, revision: str = "main") -> list[str]:
        info = self.repo_info(repo_id, revision=revision)
        return [
            str(s.get("rfilename"))
            for s in (info.get("siblings") or [])
            if isinstance(s, dict) and s.get("rfilename")
        ]

    def select_pytorch_files(self, filenames: Sequence[str]) -> list[str]:
        """Pick the files a PyTorch model actually needs.

        Prefers safetensors and, when a repo ships both, drops the ``.bin``
        duplicates: downloading both formats of a 70B model is 260 GB to
        get 130 GB of model.
        """
        import fnmatch

        chosen: list[str] = []
        for pattern in PYTORCH_PATTERNS:
            for name in filenames:
                if fnmatch.fnmatch(name, pattern) and name not in chosen:
                    chosen.append(name)
        has_safetensors = any(n.endswith(".safetensors") for n in chosen)
        if has_safetensors:
            chosen = [
                n for n in chosen
                if not (n.endswith(".bin") or n.endswith(".bin.index.json"))
            ]
        return chosen

    # -- downloading --------------------------------------------------

    def download_file(
        self,
        repo_id: str,
        filename: str,
        destination: Path,
        *,
        revision: str = "main",
        file_index: int = 1,
        file_count: int = 1,
        expected_bytes: int = 0,
    ) -> dict[str, Any]:
        """Download one file, resuming a partial ``.part`` when present."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")

        if destination.exists() and (not expected_bytes or destination.stat().st_size == expected_bytes):
            return {
                "filename": filename,
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "skipped": True,
                "resumed": False,
            }

        already = partial.stat().st_size if partial.exists() else 0
        url = (
            f"https://{self.host}/{repo_id}/resolve/"
            f"{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(filename)}"
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            headers = self._headers({"Accept": "*/*"})
            if already:
                headers["Range"] = f"bytes={already}-"
            req = urllib.request.Request(url, headers=headers)
            try:
                with safe_urlopen(req, timeout=self.timeout) as resp:
                    # A server that ignores Range answers 200 with the whole
                    # file. Appending that to a partial would produce a
                    # corrupt model that downloaded "successfully", so the
                    # partial is discarded instead.
                    resuming = already > 0 and resp.status == 206
                    if already and not resuming:
                        logger.info(
                            "hfdownload: %s did not honour Range; restarting %s",
                            self.host, filename,
                        )
                        already = 0
                    mode = "ab" if resuming else "wb"
                    total = already + int(resp.headers.get("Content-Length") or 0)
                    if expected_bytes:
                        total = expected_bytes

                    downloaded = already
                    started = time.monotonic()
                    with open(partial, mode) as handle:
                        while True:
                            chunk = resp.read(_CHUNK)
                            if not chunk:
                                break
                            handle.write(chunk)
                            downloaded += len(chunk)
                            self._tick(
                                filename, downloaded, total, file_index, file_count,
                                started, already, resuming,
                            )
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise HFDownloadError(
                        f"Not authorised to download {filename} from {repo_id}.",
                        code="unauthorized",
                        hint="Accept the licence on the model page and set HF_TOKEN.",
                    ) from exc
                if exc.code == 416:
                    # Requested range beyond the file: the partial is at
                    # least as long as the file. Start over rather than
                    # trusting a length we cannot verify.
                    partial.unlink(missing_ok=True)
                    already = 0
                    last_error = exc
                    continue
                last_error = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                # Resume from whatever landed rather than restarting.
                already = partial.stat().st_size if partial.exists() else 0
                delay = 2 ** attempt
                logger.warning(
                    "hfdownload: %s attempt %d failed (%s); retrying in %ds",
                    filename, attempt, _redact(str(last_error), self.token), delay,
                )
                time.sleep(delay)
        else:
            raise HFDownloadError(
                _redact(
                    f"Could not download {filename} after {self.max_retries} attempts: "
                    f"{last_error}",
                    self.token,
                ),
                code="download_failed",
            )

        if expected_bytes and partial.stat().st_size != expected_bytes:
            raise HFDownloadError(
                f"{filename} finished at {partial.stat().st_size} bytes but the repository "
                f"says it is {expected_bytes}. Refusing to rename a short file into place.",
                code="size_mismatch",
            )
        # Rename only when complete: a killed process must never leave a
        # truncated file that looks finished.
        partial.replace(destination)
        return {
            "filename": filename,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "skipped": False,
            "resumed": bool(already),
        }

    def _tick(
        self, filename: str, downloaded: int, total: int, index: int, count: int,
        started: float, already: int, resumed: bool,
    ) -> None:
        if self.progress is None:
            return
        elapsed = max(1e-6, time.monotonic() - started)
        rate = (downloaded - already) / elapsed
        try:
            self.progress(
                DownloadProgress(
                    filename=filename, downloaded=downloaded, total=total,
                    file_index=index, file_count=count,
                    bytes_per_second=rate, resumed=resumed,
                )
            )
        except Exception:  # noqa: BLE001 - a broken listener must not fail a 40 GB download
            logger.debug("hfdownload: progress callback raised", exc_info=True)

    def download(
        self,
        repo_id: str,
        directory: str | Path,
        *,
        revision: str = "main",
        filenames: Sequence[str] | None = None,
        kind: str = "auto",
    ) -> DownloadResult:
        """Download a model into *directory*.

        ``kind`` is ``"gguf"``, ``"pytorch"`` or ``"auto"``. Auto looks at
        what the repo contains and prefers GGUF when it has any, because
        a repo with both is a GGUF conversion of a model whose PyTorch
        weights are somewhere else and much larger.
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()

        info = self.repo_info(repo_id, revision=revision)
        siblings = info.get("siblings") or []
        available = [
            str(s.get("rfilename")) for s in siblings
            if isinstance(s, dict) and s.get("rfilename")
        ]
        sizes: dict[str, int] = {}
        for sibling in siblings:
            if not isinstance(sibling, dict):
                continue
            name = str(sibling.get("rfilename") or "")
            lfs = sibling.get("lfs")
            if isinstance(lfs, dict) and isinstance(lfs.get("size"), int):
                sizes[name] = int(lfs["size"])
            elif isinstance(sibling.get("size"), int):
                sizes[name] = int(sibling["size"])

        if filenames:
            wanted = list(filenames)
            missing = [n for n in wanted if n not in available]
            if missing:
                raise HFDownloadError(
                    f"{repo_id} has no file(s): {', '.join(missing)}", code="not_found"
                )
        else:
            gguf = [n for n in available if n.lower().endswith(".gguf")]
            if kind == "gguf" or (kind == "auto" and gguf):
                if not gguf:
                    raise HFDownloadError(
                        f"{repo_id} has no .gguf files. It is probably a safetensors repo — "
                        "use kind='pytorch', or look for a matching '-GGUF' repository.",
                        code="no_gguf",
                    )
                wanted = gguf
            else:
                wanted = self.select_pytorch_files(available)
                if not wanted:
                    raise HFDownloadError(
                        f"{repo_id} has no recognisable model files.", code="not_found"
                    )

        result = DownloadResult(repo_id=repo_id, directory=target)
        for index, name in enumerate(wanted, start=1):
            record = self.download_file(
                repo_id, name, target / Path(name).name,
                revision=revision, file_index=index, file_count=len(wanted),
                expected_bytes=sizes.get(name, 0),
            )
            result.files.append(record)
            result.total_bytes += int(record["bytes"])
            result.resumed_files += 1 if record["resumed"] else 0

        result.seconds = time.monotonic() - started
        return result

    @staticmethod
    def free_space_bytes(directory: str | Path) -> int:
        return shutil.disk_usage(Path(directory)).free
