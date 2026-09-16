"""noodle.hyped — Noodle, driven from inside hyped-pro.

Noodle's own docstring has called it "the autonomous executor inside
Hyped Pro" since it shipped, and it was not one. It was a library and a
``noodle`` console script, and hyped-pro had no idea it existed: no
command, no bridge verb, nothing in the GUI. Running a swarm meant
leaving the TUI you were chatting in and starting a different program in
another terminal, which is exactly the workflow hyped-pro exists to
avoid.

This is the missing half. It gives the bridge four verbs —
:func:`providers`, :func:`start`, :func:`poll`, :func:`stop` — and holds
the sessions they address.

Why a session and not a call
-----------------------------
A swarm run is minutes of work with no output until it finishes, and the
bridge protocol is one JSON response per request. Running one inside a
request would hold a bridge thread for the whole run and give the TUI
nothing to draw until it ended, which for a five-task swarm is a frozen
screen for several minutes.

So :func:`start` returns immediately with a session id, the run proceeds
on its own thread, and :func:`poll` drains whatever events have happened
since the last poll. The TUI gets a live feed — the same
:class:`~hypernix.interfaces.noodle.AgentEvent` stream the ``noodle``
CLI prints — and the bridge's stdin loop is never blocked.

Events are drained, not accumulated
------------------------------------
:func:`poll` removes what it returns. A long run produces thousands of
events, and a poll that returned the whole history every second would
grow quadratically in a TUI that redraws on every one of them. The
session keeps a bounded tail for :func:`summary` and nothing else.

What it will not do without being told
---------------------------------------
Execution is off unless ``allow_execute`` is passed, exactly as in the
library, and :func:`start` refuses a workspace root outside the current
tree unless ``allow_outside`` is set. Both defaults are the conservative
one, because the thing being started here is a model with file tools
pointed at a directory, and a chat TUI is a context where somebody may
well paste a prompt without reading it closely.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "NoodleSessionError",
    "Session",
    "adopt_stored_keys",
    "providers",
    "start",
    "poll",
    "stop",
    "summary",
    "sessions",
    "clear_finished",
    "KEY_VENDORS",
    "MAX_TAIL",
]

#: hyped-pro's vendor id -> the environment variables Noodle's provider
#: specs look in.
#:
#: The two halves grew up apart and do not agree on names. hyped-pro
#: stores a Moonshot key under ``moonshot`` and an Alibaba one under
#: ``dashscope``; Noodle's providers are called ``kimi`` and ``qwen``.
#: Neither spelling is wrong and renaming either would break something
#: already on disk, so the mapping lives here.
#:
#: It is needed at all because
#: :meth:`~hypernix.interfaces.noodle.ProviderSpec.resolve_key` reads
#: ``os.environ`` and nothing else, while hyped-pro's ``/key`` writes to
#: ``~/.hypernix/config.json``. That was the whole of the integration
#: failure: somebody sets a key in the TUI, Noodle says it has no
#: provider with a key, and both are telling the truth.
KEY_VENDORS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
    "dashscope": ("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "grok": ("XAI_API_KEY", "GROK_API_KEY"),
    "t1": ("HYPERNIX_T1_KEY", "T1_KEY"),
    "t1api": ("HYPERNIX_T1_KEY", "T1_KEY"),
}


def adopt_stored_keys() -> list[str]:
    """Publish hyped-pro's saved API keys where Noodle will find them.

    Returns the vendors adopted. Called before anything that builds a
    client or reports what is available.

    An environment variable that is already set always wins — that is
    :func:`hypernix.system.config.get_provider_key`'s own resolution
    order, and a function that overwrote a key the operator exported into
    this shell would be the more surprising of the two behaviours by
    some margin.
    """
    import os

    try:
        from hypernix.system.config import get_provider_key
    except ImportError:  # pragma: no cover - config is part of the package
        return []

    adopted: list[str] = []
    for vendor, names in KEY_VENDORS.items():
        if any(os.environ.get(name, "").strip() for name in names):
            continue
        try:
            key = get_provider_key(vendor)
        except Exception:  # noqa: BLE001 - a corrupt config must not stop a run
            logger.debug("noodle: reading the %s key failed", vendor, exc_info=True)
            continue
        if not key:
            continue
        for name in names:
            os.environ.setdefault(name, key)
        adopted.append(vendor)
    return adopted

#: Events kept per session for :func:`summary` after :func:`poll` has
#: drained them. Enough to show what a run did at the end, bounded so a
#: long-lived TUI does not accumulate a swarm's whole history in memory.
MAX_TAIL = 400

#: How many sessions are remembered at once. A TUI that ran fifty swarms
#: over an afternoon should not be holding fifty reports.
MAX_SESSIONS = 16


class NoodleSessionError(Exception):
    """A session could not be started, found or stopped."""


@dataclass
class Session:
    """One swarm run, in flight or finished."""

    session_id: str
    prompt: str
    roster: list[str]
    root: Path
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    #: Events not yet drained by :func:`poll`.
    pending: deque = field(default_factory=deque)
    #: The most recent events, kept after draining for :func:`summary`.
    tail: deque = field(default_factory=lambda: deque(maxlen=MAX_TAIL))
    report: Any = None
    error: str = ""
    cancelled: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _swarm: Any = field(default=None, repr=False)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def record(self, event: Any) -> None:
        """Called from the swarm's worker threads, so it takes the lock."""
        try:
            payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        except Exception:  # noqa: BLE001 - an event must never break a run
            payload = {"kind": "event", "repr": repr(event)}
        with self._lock:
            self.pending.append(payload)
            self.tail.append(payload)

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self.pending)
            self.pending.clear()
        return events

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "roster": list(self.roster),
            "root": str(self.root),
            "running": self.running,
            "cancelled": self.cancelled,
            "elapsed": round(self.elapsed, 2),
            "error": self.error,
            "pending_events": len(self.pending),
            "report": self.report.to_dict() if self.report is not None else None,
        }


_SESSIONS: dict[str, Session] = {}
_SESSIONS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# What is available
# ---------------------------------------------------------------------------


def providers() -> dict[str, Any]:
    """Which models a swarm could be built from on this machine.

    Split into ``ready`` (a key is set, or the backend needs none) and
    ``needs_key``, because "Noodle says it has no providers" and "Noodle
    has nine providers and you have set a key for none of them" are
    different problems and the second one is fixable from inside
    hyped-pro with ``/key``.
    """
    from .providers import PROVIDERS, available_providers

    adopted = adopt_stored_keys()
    ready = {spec.provider.value for spec in available_providers()}
    # Which hyped-pro `/key` vendor to name in the fix-it line. The
    # provider's own name is often not it -- telling somebody to run
    # `/key qwen` when the key is stored under `dashscope` sends them
    # round the loop a second time.
    vendor_for = {
        "kimi": "moonshot", "qwen": "dashscope", "hypernix": "t1",
    }
    listed = []
    for name, spec in PROVIDERS.items():
        is_ready = spec.provider.value in ready
        vendor = vendor_for.get(name, name)
        if is_ready:
            reason = ""
        elif spec.paid:
            reason = f"no API key — /key {vendor} <key>"
        else:
            reason = "local backend; start it and it will appear"
        listed.append({
            "provider": spec.provider.value,
            "label": spec.label,
            "paid": spec.paid,
            "ready": is_ready,
            "default_model": spec.default_model,
            "env_keys": list(spec.env_keys),
            "key_vendor": vendor,
            "reason": reason,
        })
    listed.sort(key=lambda entry: (not entry["ready"], entry["paid"], entry["provider"]))
    return {
        "providers": listed,
        "ready": [entry["provider"] for entry in listed if entry["ready"]],
        "any_ready": any(entry["ready"] for entry in listed),
        # Named so that "it works now and I did not change anything" has
        # an answer: hyped-pro's stored keys were published into the
        # environment on the way in.
        "adopted_keys": adopted,
    }


def _default_roster() -> list[str]:
    """Something usable, when the caller named nothing.

    Free and local first. A default that silently picked a paid frontier
    model would turn ``/noodle do a thing`` into an invoice, and the one
    thing a default must not do is spend money nobody asked to spend.
    """
    from .providers import available_providers

    adopt_stored_keys()
    usable = available_providers()
    free = [spec for spec in usable if not spec.paid]
    paid = [spec for spec in usable if spec.paid]
    chosen = free or paid
    if not chosen:
        raise NoodleSessionError(
            "No provider is usable here. Set a key with `/key <vendor> <key>`, "
            "or start Ollama, and try again. `/noodle providers` lists them."
        )
    return [
        f"{spec.provider.value}:{spec.default_model}".rstrip(":")
        for spec in chosen[:2]
    ]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _resolve_root(root: str | Path | None, *, allow_outside: bool) -> Path:
    """Where the agents' file tools are allowed to write.

    Defaults to a directory under the working tree rather than the
    working tree itself. An agent with file tools pointed at the repo you
    are sitting in is a thing to opt into, not a thing to get by pressing
    enter.
    """
    if root is None:
        resolved = Path.cwd() / ".noodle"
    else:
        resolved = Path(root).expanduser().resolve()
    if not allow_outside:
        cwd = Path.cwd().resolve()
        try:
            resolved.resolve().relative_to(cwd)
        except ValueError:
            raise NoodleSessionError(
                f"{resolved} is outside {cwd}. Noodle's agents get file tools "
                f"rooted at that directory, so a root outside the tree you are "
                f"working in has to be asked for explicitly: pass "
                f"allow_outside=true."
            ) from None
    return resolved


def start(
    prompt: str,
    *,
    roster: list[str] | None = None,
    root: str | Path | None = None,
    tasks: list[str] | None = None,
    max_parallel: int = 4,
    allow_execute: bool = False,
    allow_outside: bool = False,
    memory_enabled: bool = False,
    max_turns: int = 12,
    verify: str = "",
) -> dict[str, Any]:
    """Begin a swarm run and return its session immediately.

    *tasks* splits the work across several agents; without it the whole
    *prompt* is one task. *verify* is a shell command each agent's output
    must survive — ``python3 -m pytest -q``, say — and an empty string
    means no verification beyond the agent's own judgement.

    Returns the session dict. The run is on its own thread; call
    :func:`poll` for events and :func:`summary` when it is done.
    """
    if not str(prompt).strip() and not tasks:
        raise NoodleSessionError("A swarm needs something to do.")

    from .swarm import Swarm
    from .validate import combine, command_verifier, syntax_verifier

    # Before build_client, which resolves keys from the environment.
    adopt_stored_keys()
    chosen_roster = [entry for entry in (roster or []) if entry] or _default_roster()
    resolved_root = _resolve_root(root, allow_outside=allow_outside)

    verifier = syntax_verifier()
    if verify:
        if not allow_execute:
            # A verifier is a shell command. Running one while claiming
            # execution is off would be the same capability under a
            # different name, which is worse than not having the flag.
            raise NoodleSessionError(
                f"--verify runs {verify!r} as a shell command, so it needs "
                f"allow_execute as well. They are the same permission."
            )
        verifier = combine(syntax_verifier(), command_verifier(verify))

    session = Session(
        session_id=uuid.uuid4().hex[:12],
        prompt=prompt,
        roster=list(chosen_roster),
        root=resolved_root,
    )

    try:
        swarm = Swarm(
            roster=chosen_roster,
            root=resolved_root,
            max_parallel=max_parallel,
            allow_execute=allow_execute,
            memory_enabled=memory_enabled,
            verifier=verifier,
            on_event=session.record,
        )
    except Exception as exc:  # noqa: BLE001 - build_client raises several types
        raise NoodleSessionError(
            f"Could not build a swarm from {', '.join(chosen_roster)}: {exc}"
        ) from exc

    for item in (tasks or [prompt]):
        swarm.submit(item, max_turns=max_turns)
    session._swarm = swarm

    def _run() -> None:
        try:
            session.report = swarm.run()
        except Exception as exc:  # noqa: BLE001 - a worker thread must not die silently
            logger.debug("noodle: swarm run failed", exc_info=True)
            session.error = f"{type(exc).__name__}: {exc}"
        finally:
            session.finished_at = time.time()
            session.record({"kind": "session_finished", "error": session.error})

    session._thread = threading.Thread(
        target=_run, name=f"noodle-{session.session_id}", daemon=True
    )
    _remember(session)
    session._thread.start()
    return session.to_dict()


def _remember(session: Session) -> None:
    with _SESSIONS_LOCK:
        _SESSIONS[session.session_id] = session
        if len(_SESSIONS) > MAX_SESSIONS:
            # Oldest *finished* first. Evicting a running session would
            # lose the handle to something still burning tokens.
            finished = sorted(
                (s for s in _SESSIONS.values() if not s.running),
                key=lambda s: s.started_at,
            )
            for stale in finished[: len(_SESSIONS) - MAX_SESSIONS]:
                _SESSIONS.pop(stale.session_id, None)


def _get(session_id: str) -> Session:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(session_id)
    if session is None:
        raise NoodleSessionError(
            f"No Noodle session {session_id!r}. It may have finished long "
            f"enough ago to be forgotten — only the last {MAX_SESSIONS} are kept."
        )
    return session


def poll(session_id: str, *, timeout: float = 0.0) -> dict[str, Any]:
    """Drain the events since the last poll.

    *timeout* waits up to that many seconds for at least one event rather
    than returning an empty list immediately, so a TUI can long-poll
    instead of spinning. It is capped well under the bridge's own
    timeout: a poll that outlived the request it arrived on would be
    answered into a closed pipe.
    """
    session = _get(session_id)
    deadline = time.monotonic() + min(max(0.0, timeout), 5.0)
    events = session.drain()
    while not events and session.running and time.monotonic() < deadline:
        time.sleep(0.05)
        events = session.drain()
    return {
        "session_id": session_id,
        "events": events,
        "running": session.running,
        "cancelled": session.cancelled,
        "elapsed": round(session.elapsed, 2),
        "error": session.error,
        "report": session.report.to_dict() if session.report is not None else None,
    }


def stop(session_id: str) -> dict[str, Any]:
    """Ask a run to stop after its in-flight turns finish.

    Not a kill. Every provider client here is blocking stdlib HTTP inside
    a thread pool, and Python cannot interrupt that from outside; what it
    can do is stop dispatching new turns, which it does. A request
    already on the wire is paid for whether or not its answer is used, so
    there is nothing to save by pretending otherwise.
    """
    session = _get(session_id)
    session.cancelled = True
    swarm = session._swarm
    stopped = False
    for attribute in ("request_stop", "cancel", "stop"):
        method = getattr(swarm, attribute, None)
        if callable(method):
            try:
                method()
                stopped = True
                break
            except Exception:  # noqa: BLE001
                logger.debug("noodle: %s() on the swarm failed", attribute, exc_info=True)
    session.record({"kind": "session_cancelled", "hard": stopped})
    return {
        "session_id": session_id,
        "cancelled": True,
        # False means the flag is set and the swarm had no cancellation
        # hook, so in-flight turns run to completion. Said out loud
        # because "stop" that does not stop things is worth knowing about.
        "swarm_notified": stopped,
        "running": session.running,
    }


def summary(session_id: str) -> dict[str, Any]:
    """A finished (or running) session, with its kept event tail."""
    session = _get(session_id)
    return {**session.to_dict(), "tail": list(session.tail)}


def sessions() -> list[dict[str, Any]]:
    """Every session this process remembers, newest first."""
    with _SESSIONS_LOCK:
        found = list(_SESSIONS.values())
    found.sort(key=lambda session: -session.started_at)
    return [session.to_dict() for session in found]


def clear_finished() -> int:
    """Forget every session that is not running. Returns how many went."""
    with _SESSIONS_LOCK:
        done = [s.session_id for s in _SESSIONS.values() if not s.running]
        for session_id in done:
            _SESSIONS.pop(session_id, None)
    return len(done)
