"""hyped_pro_bridge — stdio JSON worker used by the Node ``hyped_pro.ts`` TUI.

hyped-pro's TUI (hyped_pro.ts, run under Node) has no Python runtime of its
own, so every operation that needs real inference, a real HuggingFace
download, or the real T1 Gatekeeper — everything in
:mod:`hypernix.hyped_pro_core` — is delegated here over stdio.

Protocol
--------
One JSON object per line on stdin, one JSON object per line on stdout::

    -> {"id": 1, "cmd": "ping"}
    <- {"id": 1, "ok": true, "data": {"pong": true}}

    -> {"id": 2, "cmd": "chat", "model": "claude-sonnet-5",
        "messages": [{"role": "user", "content": "hi"}]}
    <- {"id": 2, "ok": true, "data": {"reply": "..."}}

    -> {"id": 3, "cmd": "download", "repo": "org/model"}
    <- {"id": 3, "ok": true, "data": {"path": "/home/user/.hypernix/models/model"}}

On failure: ``{"id": N, "ok": false, "code": "HPC-...", "error": "..."}``.

stderr is left connected to the parent's real terminal (Node spawns this
process with stdio ``["pipe", "pipe", "inherit"]``) so every
``huggingface_hub`` progress bar, ``[hypernix] ...`` download log line, and
Python traceback shows up live and unmodified — that's the "logs shown to
the user's terminal" requirement; stdout is reserved for the JSON protocol
and must never carry anything else.

The worker is a single Python process kept alive for the life of the TUI
session, not spawned per turn, so a local model loaded via
:func:`hypernix.hyped_pro_core.send_local_chat` stays resident in VRAM
across turns instead of reloading every message.
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
from typing import Any

from hypernix.interfaces import hyped_pro_core as core

BRIDGE_VERSION = "1.1.26.9.0.0"


# ---------------------------------------------------------------------------
# Cancellation
#
# Requests used to be dispatched inline, one at a time, straight off the
# stdin loop. That made "cancel" impossible to even deliver: a `chat` that
# was going to take ninety seconds held the loop for ninety seconds, so a
# cancel sent at second two wasn't read until second ninety-one. The TUI's
# Escape key therefore couldn't stop anything — it could only throw away an
# answer the model had already finished producing, on a GPU that stayed busy
# the whole time.
#
# So long-running commands now run on their own thread and the stdin loop
# stays free to read control commands. Each in-flight request owns a
# threading.Event; `cancel` sets it, and the generation loop polls it once
# per token.
#
# What can actually be interrupted is backend-specific, and the response
# says which happened rather than implying more than is true:
#
#   "stopped"   the generation loop is polling the event and will stop
#               within a token (local safetensors, via NeoOven.should_stop)
#   "pending"   the backend has no interruption point — an in-flight cloud
#               HTTP request, or llama.cpp inside multilama — so the call
#               runs to completion and its reply is discarded
#   "unknown"   nothing by that id is in flight (already finished, or a
#               stale cancel racing the reply)
# ---------------------------------------------------------------------------

# Commands that can take long enough to be worth cancelling, and are safe to
# run off the main loop. Everything else is a fast config read/write and
# stays inline, where it can't interleave with itself.
# noodle_poll is here because it long-polls: it waits up to `timeout`
# seconds for an event rather than returning empty and being asked again
# immediately. Inline, that wait would block the stdin loop and the TUI
# could not deliver a cancel while a swarm was running -- the same bug
# that made `chat` uncancellable before it moved off the loop.
#
# noodle_start is not here. It spawns a thread and returns at once, so
# backgrounding it would buy a thread to start a thread.
BACKGROUND_COMMANDS = frozenset({"chat", "download", "t1api_status", "noodle_poll"})

# Backends whose generation loop polls should_stop. Anything else can be
# asked to cancel, but the request will finish first.
INTERRUPTIBLE_KINDS = frozenset({"local"})

_inflight: dict[Any, threading.Event] = {}
_inflight_lock = threading.Lock()
_stdout_lock = threading.Lock()


def _interruptible(req: dict[str, Any]) -> bool:
    """Whether this request's backend polls the cancel event.

    Only the local safetensors path does — NeoOven's generation loop takes
    a ``should_stop``. A cloud HTTP call and llama.cpp inside multilama have
    no interruption point, so claiming otherwise would be the same lie the
    TUI was already telling.
    """
    if req.get("cmd") != "chat":
        return False
    try:
        model = core.get_model(req["model"])
    except Exception:  # noqa: BLE001 - an unknown model fails later, with a real error
        return False
    if model.kind not in INTERRUPTIBLE_KINDS:
        return False
    # GGUF local models go through multilama, which can't be interrupted.
    return model.format != "gguf"


def _register(id_: Any, req: dict[str, Any]) -> threading.Event:
    event = threading.Event()
    event.interruptible = _interruptible(req)  # type: ignore[attr-defined]
    with _inflight_lock:
        _inflight[id_] = event
    return event


def _unregister(id_: Any) -> bool:
    """Drop *id_*; returns whether it had been cancelled."""
    with _inflight_lock:
        event = _inflight.pop(id_, None)
    return bool(event and event.is_set())


def _cancel(id_: Any) -> str:
    with _inflight_lock:
        event = _inflight.get(id_)
    if event is None:
        return "unknown"
    event.set()
    return "stopped" if getattr(event, "interruptible", False) else "pending"


def _write(payload: dict[str, Any]) -> None:
    """Serialize one response. Held under a lock because worker threads and
    the main loop both write to the same pipe, and a half-written line would
    be unparseable JSON on the other end."""
    with _stdout_lock:
        try:
            sys.stdout.write(json.dumps(payload) + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass  # parent is gone; nothing to report to


def _err(id_: Any, code: str, message: str) -> dict[str, Any]:
    return {"id": id_, "ok": False, "code": code, "error": message}


def _ok(id_: Any, data: Any) -> dict[str, Any]:
    return {"id": id_, "ok": True, "data": data}


def dispatch(req: dict[str, Any], cancel: threading.Event | None = None) -> dict[str, Any]:
    id_ = req.get("id")
    cmd = req.get("cmd")

    try:
        if cmd == "ping":
            return _ok(id_, {"pong": True, "version": BRIDGE_VERSION})

        if cmd == "catalog":
            return _ok(id_, core.catalog_json())

        if cmd == "is_downloaded":
            model = core.get_model(req["model"])
            downloaded, path = core.is_downloaded(model)
            return _ok(id_, {"downloaded": downloaded, "path": str(path)})

        if cmd == "download":
            model = core.get_model(req["model"])
            print(f"[hyped-pro-bridge] fetching {model.repo} for {model.short} ...", file=sys.stderr)
            path = core.ensure_downloaded(model, quiet=False)
            print(f"[hyped-pro-bridge] {model.short} ready at {path}", file=sys.stderr)
            return _ok(id_, {"path": str(path)})

        if cmd == "chat":
            result = core.send_chat_message(
                model_short=req["model"],
                messages=req["messages"],
                system=req.get("system"),
                api_key=req.get("api_key"),
                max_tokens=req.get("max_tokens"),
                max_thinking_tokens=req.get("max_thinking_tokens"),
                hide_thinking=req.get("hide_thinking", True),
                enable_tools=req.get("enable_tools", True),
                t1_api_model_id=req.get("t1_api_model_id"),
                should_stop=(cancel.is_set if cancel is not None else None),
            )
            return _ok(id_, {
                "reply": result["content"],
                "thinking": result["thinking"],
                # True when the reply is whatever had been generated before
                # a cancel landed, not a finished answer.
                "cancelled": bool(cancel is not None and cancel.is_set()),
            })

        if cmd == "key_get":
            from hypernix.system.config import get_provider_key
            key = get_provider_key(req["vendor"])
            masked = (key[:6] + "..." + key[-4:]) if key and len(key) > 12 else ("set" if key else None)
            return _ok(id_, {"set": bool(key), "masked": masked})

        if cmd == "key_set":
            from hypernix.system.config import set_provider_key
            set_provider_key(req["vendor"], req["key"])
            return _ok(id_, {"saved": True})

        if cmd == "key_clear":
            from hypernix.system.config import clear_provider_key
            clear_provider_key(req["vendor"])
            return _ok(id_, {"cleared": True})

        # -- v0.71.5rc2 (Beta 4) ------------------------------------
        if cmd == "release":
            from hypernix.system.release import current_release
            info = current_release(force=bool(req.get("force")))
            return _ok(id_, info.to_dict())

        if cmd == "t1api_status":
            return _ok(id_, core.t1_api_status(url=req.get("url")))

        if cmd == "t1api_get_url":
            return _ok(id_, {"url": core.t1_api_url()})

        if cmd == "t1api_set_url":
            core.set_t1_api_url(req["url"])
            return _ok(id_, {"url": core.t1_api_url()})

        # -- 0.72.5: Noodle, the autonomous executor ------------------
        #
        # Noodle has described itself as "the autonomous executor inside
        # Hyped Pro" since it shipped and was not reachable from
        # hyped-pro at all. These five verbs are the connection.
        #
        # A run is a session rather than a call: it takes minutes and
        # produces nothing until it ends, so doing it inside one request
        # would hold a bridge thread and leave the TUI with nothing to
        # draw. `noodle_start` returns a session id at once and
        # `noodle_poll` drains events as they happen.
        if cmd == "noodle_providers":
            from hypernix.interfaces.noodle import hyped as noodle

            return _ok(id_, noodle.providers())

        if cmd == "noodle_start":
            from hypernix.interfaces.noodle import hyped as noodle

            try:
                return _ok(id_, noodle.start(
                    req.get("prompt", ""),
                    roster=req.get("roster"),
                    root=req.get("root"),
                    tasks=req.get("tasks"),
                    max_parallel=int(req.get("max_parallel", 4)),
                    allow_execute=bool(req.get("allow_execute", False)),
                    allow_outside=bool(req.get("allow_outside", False)),
                    memory_enabled=bool(req.get("memory_enabled", False)),
                    max_turns=int(req.get("max_turns", 12)),
                    verify=req.get("verify", ""),
                ))
            except noodle.NoodleSessionError as exc:
                return _err(id_, "HPB-NOODLE-001", str(exc))

        if cmd == "noodle_poll":
            from hypernix.interfaces.noodle import hyped as noodle

            try:
                return _ok(id_, noodle.poll(
                    req["session"], timeout=float(req.get("timeout", 0.0))
                ))
            except noodle.NoodleSessionError as exc:
                return _err(id_, "HPB-NOODLE-002", str(exc))

        if cmd == "noodle_stop":
            from hypernix.interfaces.noodle import hyped as noodle

            try:
                return _ok(id_, noodle.stop(req["session"]))
            except noodle.NoodleSessionError as exc:
                return _err(id_, "HPB-NOODLE-002", str(exc))

        if cmd == "noodle_sessions":
            from hypernix.interfaces.noodle import hyped as noodle

            if req.get("session"):
                try:
                    return _ok(id_, noodle.summary(req["session"]))
                except noodle.NoodleSessionError as exc:
                    return _err(id_, "HPB-NOODLE-002", str(exc))
            return _ok(id_, {"sessions": noodle.sessions()})

        if cmd == "cancel":
            return _ok(id_, {"target": req.get("target"),
                             "result": _cancel(req.get("target"))})

        return _err(id_, "HPB-PROTO-001", f"unknown command {cmd!r}")

    except core.HypedProError as exc:
        print(f"[hyped-pro-bridge] ERROR {exc.code}: {exc.message}", file=sys.stderr)
        return _err(id_, exc.code, exc.message)
    except KeyError as exc:
        msg = f"missing required field {exc}"
        print(f"[hyped-pro-bridge] ERROR HPB-PROTO-001: {msg}", file=sys.stderr)
        return _err(id_, "HPB-PROTO-001", msg)
    except Exception as exc:  # noqa: BLE001 — must always return JSON, never crash silently
        print("[hyped-pro-bridge] ERROR HPB-INTERNAL-001: unhandled exception", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return _err(id_, "HPB-INTERNAL-001", f"{type(exc).__name__}: {exc}")


def _run_background(req: dict[str, Any], event: threading.Event) -> None:
    """Run one long command off the stdin loop and write its response."""
    id_ = req.get("id")
    try:
        resp = dispatch(req, cancel=event)
    except Exception as exc:  # noqa: BLE001 - a worker thread must never die silently
        traceback.print_exc(file=sys.stderr)
        resp = _err(id_, "HPB-INTERNAL-001", f"{type(exc).__name__}: {exc}")
    finally:
        _unregister(id_)
    _write(resp)


def serve() -> int:
    """Read JSON requests from stdin, write JSON responses to stdout, forever.

    Long commands (see :data:`BACKGROUND_COMMANDS`) run on their own thread
    so the loop stays free to read ``cancel``. Everything else — key and URL
    reads and writes — runs inline: they finish in microseconds and running
    them concurrently would only add a way for two config writes to
    interleave.
    """
    print(f"[hyped-pro-bridge] ready (v{BRIDGE_VERSION}), waiting for requests on stdin", file=sys.stderr)
    workers: list[threading.Thread] = []
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[hyped-pro-bridge] ERROR HPB-PROTO-001: bad JSON line: {exc}", file=sys.stderr)
                _write(_err(None, "HPB-PROTO-001", f"invalid JSON: {exc}"))
                continue

            if req.get("cmd") in BACKGROUND_COMMANDS:
                event = _register(req.get("id"), req)
                thread = threading.Thread(
                    target=_run_background, args=(req, event),
                    name=f"hyped-pro-{req.get('cmd')}-{req.get('id')}", daemon=True,
                )
                workers.append(thread)
                thread.start()
                workers = [t for t in workers if t.is_alive()]
                continue

            _write(dispatch(req))
    except (BrokenPipeError, OSError):
        # The parent (hyped_pro.ts) exited and closed its end of the pipe
        # while we were mid-write — it's gone, there's no one left to
        # report an error to, so exit quietly rather than dump a traceback.
        pass

    # Ask anything still running to stop, then leave. The threads are
    # daemons, so a backend with no interruption point (a cloud call in
    # flight) doesn't hold the process open after its parent has gone.
    with _inflight_lock:
        for event in _inflight.values():
            event.set()
    print("[hyped-pro-bridge] stdin closed, exiting", file=sys.stderr)
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    """One-shot mode for scripting/debugging: build a single request from
    argv and print the response, instead of entering the stdio loop.

    Usage: python3 -m hypernix.hyped_pro_bridge <cmd> [key=value ...]
    Example: python3 -m hypernix.hyped_pro_bridge download model=deepseek-r1
    """
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("serve", "--serve"):
        return serve()

    cmd, *rest = argv
    req: dict[str, Any] = {"id": 1, "cmd": cmd}
    for arg in rest:
        if "=" not in arg:
            print(f"usage: hyped_pro_bridge <cmd> [key=value ...] — bad arg {arg!r}", file=sys.stderr)
            return 2
        k, v = arg.split("=", 1)
        req[k] = v
    resp = dispatch(req)
    # Compact, single-line JSON — same wire format as serve()'s stdout, so
    # callers that read "the last line of stdout" (hyped_pro.ts's
    # synchronous catalog fetch) get one parseable object either way.
    print(json.dumps(resp))
    return 0 if resp.get("ok") else 1


if __name__ == "__main__":
    sys.exit(cli_main())
