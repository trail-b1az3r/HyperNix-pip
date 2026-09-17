"""Which thing actually answers a chat turn.

For most of this project's life the answer was "LM Studio", and the code
said so in the field name: ``backend: "lmstudio"``, hard-coded, on every
message. Then :mod:`hypernix.hyperlink.managed` gave the server its own
llama.cpp process — and the chat path still went through the bridge, so
``/runner/load`` would start a model that nothing could talk to. A
server with no LM Studio installed could load a 70B and then refuse
every message with "this server has no chat backend configured".

This module is the missing choice. It picks, in order:

1. **The HyperNix runner**, when it has a model loaded. It is this
   server's own process, it was started deliberately, and something is
   already using the VRAM.
2. **The LM Studio bridge**, when it is enabled.
3. Neither — and the refusal says which of the two to set up, rather
   than naming only the one that happens to be checked first.

Why the same client class talks to both
---------------------------------------
``llama-server`` and LM Studio both speak the OpenAI chat API, so
:class:`~hypernix.bridge.lmstudio.LMStudioBridge` is already a correct
client for the runner — it is an OpenAI HTTP client with an unfortunate
name. Writing a second one would be a second set of retry, timeout and
error-shape decisions to keep in step.

What is *not* borrowed is the label. A reply that came from this
server's own llama.cpp is recorded as ``hypernix``, because "lmstudio"
on a machine with no LM Studio is the kind of small lie that costs
somebody an afternoon.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["Backend", "BackendUnavailable", "resolve_backend", "describe_backends"]

#: What goes in a message's ``backend`` field.
HYPERNIX = "hypernix"
LMSTUDIO = "lmstudio"


class BackendUnavailable(RuntimeError):
    """Nothing can answer. Carries what would make it possible."""

    def __init__(self, message: str, *, remedies: list[str] | None = None) -> None:
        super().__init__(message)
        self.remedies = remedies or []


@dataclass(frozen=True)
class Backend:
    """A client, and the honest name for where its answers come from."""

    client: Any
    name: str
    base_url: str
    #: The model it will use when the caller does not name one. The
    #: runner has exactly one loaded; LM Studio decides for itself.
    model_id: str = ""
    #: Shown in the app. Not the same as `name`, which is the wire value.
    label: str = ""

    @property
    def is_hypernix(self) -> bool:
        return self.name == HYPERNIX


def _openai_client(base_url: str, *, timeout: float = 300.0):
    """An OpenAI-protocol HTTP client for *base_url*.

    :class:`LMStudioBridge` under the name it should have had. Reused
    rather than reimplemented because a second client would be a second
    set of retry, timeout and error-shape decisions, and they would
    drift.
    """
    from ..bridge.lmstudio import LMStudioBridge

    return LMStudioBridge(base_url, timeout=timeout)


def resolve_backend(config: Any, runner: Any = None) -> Backend:
    """The backend that should answer, or a refusal that names both.

    *runner* is the :class:`~hypernix.hyperlink.managed.ManagedRunner`
    when the server has one. Checked first and deliberately: if somebody
    has loaded a model through ``/runner/load``, that is the model they
    meant, it is holding the VRAM, and quietly answering from LM Studio
    instead would be answering as a different model than the one the
    status screen is showing.
    """
    current = None
    if runner is not None:
        try:
            current = runner.current
        except Exception:  # noqa: BLE001 - a broken runner must not hide the bridge
            logger.debug("hyperlink: the runner could not be asked", exc_info=True)
            current = None

    if current is not None:
        base_url = f"{runner.base_url}/v1"
        return Backend(
            client=_openai_client(base_url),
            name=HYPERNIX,
            base_url=base_url,
            model_id=getattr(current, "model_id", ""),
            label="HyperNix runner",
        )

    if getattr(config, "lmstudio_enabled", False):
        base_url = getattr(config, "lmstudio_url", "") or ""
        return Backend(
            client=_openai_client(base_url or "http://127.0.0.1:1234/v1"),
            name=LMSTUDIO,
            base_url=base_url,
            label="LM Studio",
        )

    raise BackendUnavailable(
        "Nothing on this server can answer a message yet.",
        remedies=[
            "Load a model with `hypernix-t1 built-in-runner start`, or from the "
            "Runner screen in HyperLink — this server runs models itself and "
            "needs no other software.",
            "Or start LM Studio, load a model there, and set "
            "T1_LMSTUDIO_ENABLED=1.",
        ],
    )


def describe_backends(config: Any, runner: Any = None) -> list[dict[str, Any]]:
    """Every backend and whether it could answer right now.

    For the app: "no models" and "a model is loaded but the thing that
    serves it is switched off" look identical from the outside and need
    completely different actions.
    """
    rows: list[dict[str, Any]] = []

    loaded = None
    if runner is not None:
        try:
            loaded = runner.current
        except Exception:  # noqa: BLE001
            loaded = None
    rows.append({
        "name": HYPERNIX,
        "label": "HyperNix runner",
        "available": loaded is not None,
        "model_id": getattr(loaded, "model_id", "") if loaded else "",
        "base_url": getattr(runner, "base_url", "") if runner is not None else "",
        "detail": (
            f"Serving {getattr(loaded, 'model_id', '')}" if loaded
            else "Nothing loaded. This server can run models itself — no other "
                 "software needed."
        ),
    })

    enabled = bool(getattr(config, "lmstudio_enabled", False))
    rows.append({
        "name": LMSTUDIO,
        "label": "LM Studio",
        "available": enabled,
        "model_id": "",
        "base_url": getattr(config, "lmstudio_url", "") or "",
        "detail": (
            "Borrowing whatever LM Studio has loaded." if enabled
            else "Switched off. Set T1_LMSTUDIO_ENABLED=1 to use it."
        ),
    })
    return rows
