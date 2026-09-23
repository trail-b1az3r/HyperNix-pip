"""dilute.backends — turning a ``--model`` flag into something callable.

:mod:`dilute.core` asks for a callable and nothing more, which is right
for a library and useless at a command line. This is the bridge: a model
path or a server URL in, a ``(prompt, temperature=...) -> str`` out.

The thing to know about it
--------------------------
A run makes ``samples_per_prompt × attempts`` calls — five hundred
prompts at five samples is two and a half thousand. The convenient
one-shot helpers elsewhere in the codebase load the model, answer, and
close it, which at that call count means two and a half thousand model
loads and a run that never finishes. So :class:`LocalGenerator` holds
the model open across calls and closes it once, at the end.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "GeneratorError",
    "LocalGenerator",
    "ServerGenerator",
    "models_dir",
    "find_local_model",
    "resolve_generator",
]

DEFAULT_SERVER = "http://127.0.0.1:8000"


class GeneratorError(RuntimeError):
    """No model could be reached, and the message says what to do."""


def models_dir() -> Path:
    return Path.home() / ".hypernix" / "models"


def find_local_model(name: str = "", *, directory: Path | None = None) -> Path | None:
    """A GGUF under ``~/.hypernix/models`` matching *name*.

    An empty *name* takes the first one found, which is what someone
    with a single model downloaded means by "the model".
    """
    root = directory or models_dir()
    if name:
        direct = Path(name).expanduser()
        if direct.is_file():
            return direct
    if not root.is_dir():
        return None
    found = sorted(root.rglob("*.gguf"))
    if not found:
        return None
    if not name:
        return found[0]
    needle = name.lower()
    for path in found:
        if needle in path.name.lower():
            return path
    return None


@dataclass
class LocalGenerator:
    """A GGUF held open for the length of a run.

    Loaded lazily on the first call so that constructing one costs
    nothing — a CLI builds the generator and the evaluator before it
    knows whether the prompts file even parses, and paying for a model
    load to then fail on a missing file is a bad trade.
    """

    path: Path
    max_new_tokens: int = 512
    n_ctx: int = 8192
    backend: str = "vanilla"
    device: str = "auto"
    _model: Any = field(default=None, init=False, repr=False)

    def _ensure(self) -> Any:
        if self._model is None:
            from hypernix.models.ggufrun import load_gguf

            logger.info("dilute: loading %s", self.path.name)
            self._model = load_gguf(
                self.path, backend=self.backend, n_ctx=self.n_ctx,
                device=self.device,
            )
        return self._model

    def __call__(self, prompt: str, *, temperature: float = 0.8, **_: Any) -> str:
        model = self._ensure()
        return str(
            model.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=temperature,
            )
        )

    def close(self) -> None:
        close = getattr(self._model, "close", None)
        self._model = None
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - a failed teardown is not a failed run
                logger.debug("dilute: close() raised", exc_info=True)

    def __enter__(self) -> LocalGenerator:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


@dataclass
class ServerGenerator:
    """A T1 (or any OpenAI-shaped) server.

    Preferred over loading a GGUF here when a server is already up: it
    has the model in VRAM already, and a second llama.cpp beside it
    wants the same memory.
    """

    base_url: str = DEFAULT_SERVER
    token: str = ""
    model: str = ""
    max_tokens: int = 512
    timeout: float = 300.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def __call__(self, prompt: str, *, temperature: float = 0.8, **_: Any) -> str:
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if self.model:
            body["model"] = self.model
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read() or b"{}")
        choices = payload.get("choices") or []
        if not choices:
            raise GeneratorError("the server answered with no choices")
        return str(choices[0].get("message", {}).get("content", ""))

    def close(self) -> None:
        """Nothing to release. Here so callers need not care which they hold."""


def server_is_up(base_url: str, token: str = "", *, timeout: float = 2.0) -> bool:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/health", method="GET")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status < 400
    except (urllib.error.URLError, OSError, ValueError):
        return False


def resolve_generator(
    model: str = "",
    *,
    server: str = "",
    token: str = "",
    max_tokens: int = 512,
    models_directory: Path | None = None,
) -> Callable[..., str]:
    """A generator for *model*, from a running server or a local GGUF.

    The server is tried first when one was named or one is answering on
    the default port, for the VRAM reason above. A local file is used
    when there is no server. Neither available is an error with the two
    commands that fix it, not a stack trace.
    """
    url = server or DEFAULT_SERVER
    if server or server_is_up(url, token):
        if server_is_up(url, token):
            return ServerGenerator(url, token=token, model=model,
                                   max_tokens=max_tokens)
        raise GeneratorError(f"no server answering at {url}")

    path = find_local_model(model, directory=models_directory)
    if path is not None:
        return LocalGenerator(path, max_new_tokens=max_tokens)

    where = models_directory or models_dir()
    raise GeneratorError(
        "nothing to generate with. Either:\n"
        "  start a server   hypernix-t1 start\n"
        f"  or put a .gguf in {where}"
    )
