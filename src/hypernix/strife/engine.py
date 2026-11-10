"""strife.engine — load once, serve requests: sessions, streaming, stops, limits.

::

    from hypernix.strife import Engine

    engine = Engine("model.gguf", n_ctx=4096)
    for event in engine.stream(messages=[{"role": "user", "content": "Hi"}], max_tokens=64):
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
    print(event["stats"])            # the last event is "done"

**Isolation.** A request without ``session`` gets its own KV cache,
sampler and cancellation, so concurrent requests cannot see each
other's state. A request *with* a session reuses that session's cache --
only the part of the prompt after what the cache already holds is run --
and holds the session's lock while it does, so two requests on one
session take turns rather than corrupting it.

**Bounds.** At most ``max_concurrent`` requests compute at once and at
most ``max_queue`` wait; past that a request is refused with
``strife.engine.busy`` rather than queued without limit. At most
``max_sessions`` sessions are kept (least recently used dropped first),
each with a cache of exactly ``n_ctx`` tokens, so KV memory is
``max_sessions × n_ctx × per-token`` at worst and reported up front.

**Ending.** A request stops at an end-of-turn token, a stop string (the
text is cut before it, and text that might be the start of one is held
back until it is known not to be), ``max_tokens``, a full context,
``cancel`` being set, or ``timeout`` seconds passing -- and says which in
``stop_reason``.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import StrifeError
from .inspect import ModelReport, read_gguf, require_runnable
from .model import SeqState, Transformer
from .sampler import Sampler, SamplerConfig
from .tokenizer import StreamDecoder, Tokenizer
from .weights import WeightStore

__all__ = ["Engine", "GenerationResult"]


@dataclass
class GenerationResult:
    text: str
    tokens: list[int]
    stop_reason: str
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Session:
    cache: SeqState
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.monotonic)


class Engine:
    def __init__(self, model: str | Path, *, n_ctx: int | None = None,
                 cache_bytes: int | None = None, threads: int | None = None,
                 max_sessions: int = 8, max_concurrent: int = 1, max_queue: int = 16,
                 gpu_layers: int = 0) -> None:
        started = time.perf_counter()
        gguf, report = read_gguf(model)
        require_runnable(report)
        assert gguf is not None
        self.report: ModelReport = report
        self.path = Path(model)
        self.threads, self.threads_applied = _set_threads(threads)
        self.weights = WeightStore(gguf, cache_bytes=cache_bytes)
        self.model = Transformer(self.weights, report)
        self.tokenizer = Tokenizer(gguf.metadata)
        trained = report.context_length or 4096
        self.n_ctx = int(n_ctx) if n_ctx else min(trained, 4096)
        if self.n_ctx <= 0:
            raise StrifeError("strife.config.bad_context", "n_ctx must be positive")
        from .placement import plan

        self.placement = plan(report, self.model.plan, n_ctx=self.n_ctx, gpu_layers=gpu_layers,
                              max_sessions=max_sessions)
        self.max_sessions = max(1, int(max_sessions))
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._sessions_lock = threading.Lock()
        self._max_concurrent = max(1, int(max_concurrent))
        self._compute = threading.Semaphore(self._max_concurrent)
        self._active = 0
        self._waiting = 0
        self._max_queue = max(0, int(max_queue))
        self._queue_lock = threading.Lock()
        self._closed = False
        self.load_seconds = time.perf_counter() - started

    # -- sessions ------------------------------------------------------------

    def new_session(self) -> str:
        sid = uuid.uuid4().hex
        with self._sessions_lock:
            self._sessions[sid] = _Session(self.model.new_state(self.n_ctx))
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        return sid

    def session_for(self, ids: list[int]) -> str:
        """The session whose state shares the longest prefix with *ids*,
        or a new one when none shares a useful part (fewer than 16 tokens).

        For a server with several conversations going: each request lands
        on the cache that already holds most of it, and only the new
        message is run.
        """
        best, best_len = None, 15
        with self._sessions_lock:
            for sid, s in self._sessions.items():
                if s.lock.locked():
                    continue
                common = 0
                for a, b in zip(s.cache.tokens, ids, strict=False):
                    if a != b:
                        break
                    common += 1
                if common > best_len:
                    best, best_len = sid, common
        return best if best is not None else self.new_session()

    def close_session(self, sid: str) -> bool:
        with self._sessions_lock:
            return self._sessions.pop(sid, None) is not None

    def _session(self, sid: str) -> _Session:
        with self._sessions_lock:
            s = self._sessions.get(sid)
            if s is None:
                raise StrifeError("strife.engine.no_session", f"no session {sid!r}",
                                  remedy="Start one with new_session(); idle sessions are "
                                         "dropped when more than max_sessions exist.")
            self._sessions.move_to_end(sid)
            s.last_used = time.monotonic()
            return s

    # -- prompts -------------------------------------------------------------

    def prompt_tokens(self, prompt: str | None = None,
                      messages: list[dict[str, Any]] | None = None,
                      template_vars: dict[str, Any] | None = None) -> list[int]:
        if (prompt is None) == (messages is None):
            raise StrifeError("strife.request.bad", "give exactly one of prompt or messages")
        if messages is not None:
            text = self.tokenizer.apply_chat_template(messages, **(template_vars or {}))
            return self.tokenizer.encode(text, add_bos=False if self.tokenizer.templated_has_bos(text)
                                         else None)
        return self.tokenizer.encode(prompt or "")

    # -- generation ----------------------------------------------------------

    def generate(self, prompt: str | None = None, **kwargs: Any) -> GenerationResult:
        """:meth:`stream`, collected."""
        final: dict[str, Any] = {}
        for event in self.stream(prompt, **kwargs):
            if event["type"] == "done":
                final = event
        return GenerationResult(final["text"], final["tokens"], final["stop_reason"],
                                final["stats"])

    def stream(self, prompt: str | None = None, *, messages: list[dict[str, str]] | None = None,
               session: str | None = None, max_tokens: int = 256,
               stop: list[str] | None = None, sampling: SamplerConfig | None = None,
               timeout: float | None = None, cancel: threading.Event | None = None,
               template_vars: dict[str, Any] | None = None,
               prompt_ids: list[int] | None = None) -> Iterator[dict[str, Any]]:
        if self._closed:
            raise StrifeError("strife.engine.closed", "the engine was closed")
        if max_tokens < 0:
            raise StrifeError("strife.request.bad", "max_tokens must not be negative")
        stops = [s for s in (stop or []) if s]
        sampler = Sampler(sampling or SamplerConfig())
        ids = list(prompt_ids) if prompt_ids is not None else self.prompt_tokens(
            prompt, messages, template_vars)
        if not ids:
            raise StrifeError("strife.request.bad", "the prompt is empty")
        deadline = time.monotonic() + timeout if timeout else None

        def stopped() -> bool:
            return bool((cancel is not None and cancel.is_set())
                        or (deadline is not None and time.monotonic() > deadline))

        with self._queue_lock:
            # Refused only when every slot is busy *and* the queue is full.
            if self._active >= self._max_concurrent and self._waiting >= self._max_queue:
                raise StrifeError("strife.engine.busy",
                                  f"{self._active} requests are running and {self._waiting} "
                                  "waiting", remedy="Try again shortly, or raise max_queue.")
            self._waiting += 1
        acquired = False
        try:
            while not acquired:
                acquired = self._compute.acquire(timeout=0.1)
                if not acquired and stopped():
                    yield self._done("", [], "cancelled" if cancel is not None and cancel.is_set()
                                     else "timeout", {})
                    return
        finally:
            with self._queue_lock:
                self._waiting -= 1
                if acquired:
                    self._active += 1
        try:
            if session is not None:
                s = self._session(session)
                with s.lock:
                    yield from self._run(ids, s.cache, sampler, max_tokens, stops, stopped,
                                         cancel, reuse=True)
            else:
                cache = self.model.new_state(self.n_ctx)
                yield from self._run(ids, cache, sampler, max_tokens, stops, stopped, cancel,
                                     reuse=False)
        finally:
            with self._queue_lock:
                self._active -= 1
            self._compute.release()

    def _done(self, text: str, tokens: list[int], reason: str, stats: dict[str, Any]
              ) -> dict[str, Any]:
        return {"type": "done", "text": text, "tokens": tokens, "stop_reason": reason,
                "stats": stats}

    def _run(self, ids: list[int], cache: SeqState, sampler: Sampler, max_tokens: int,
             stops: list[str], stopped: Any, cancel: threading.Event | None, *, reuse: bool
             ) -> Iterator[dict[str, Any]]:
        if len(ids) >= cache.n_ctx:
            raise StrifeError("strife.runtime.context_full",
                              f"the prompt is {len(ids)} tokens; the context is {cache.n_ctx}",
                              remedy="Shorten the prompt or raise n_ctx.")
        reused = 0
        if reuse:
            common = 0
            for a, b in zip(cache.tokens, ids, strict=False):
                if a != b:
                    break
                common += 1
            common = min(common, len(ids) - 1)  # always run at least one token
            # Attention goes back anywhere; recurrent layers only to a
            # snapshot, so this may reuse less than the common prefix.
            reused = cache.rewind(common)
        else:
            reused = cache.rewind(0)
        t0 = time.perf_counter()
        try:
            logits = self.model.forward(ids[reused:], cache, cancelled=stopped)
        except StrifeError as exc:
            if exc.code == "strife.runtime.cancelled":
                cache.rewind(reused, force=True)
                yield self._done("", [], self._why(cancel), {"prompt_tokens": len(ids)})
                return
            raise
        # The next turn of a conversation starts with this prompt: remember
        # the recurrent state here so it can carry on instead of starting over.
        cache.checkpoint()
        t_prefill = time.perf_counter() - t0
        decoder = StreamDecoder(self.tokenizer)
        generated: list[int] = []
        text, shown = "", 0
        reason = "length"
        t_first: float | None = None
        history = list(ids)
        for _ in range(max_tokens):
            if stopped():
                reason = self._why(cancel)
                break
            token = sampler(logits, history)
            if token in self.tokenizer.stop_ids:
                reason = "stop_token"
                break
            generated.append(token)
            history.append(token)
            if t_first is None:
                t_first = time.perf_counter()
            text += decoder.push(token)
            hit = _find_stop(text, stops)
            if hit is not None:
                text = text[:hit]
                reason = "stop_string"
                if len(text) > shown:
                    yield {"type": "token", "text": text[shown:], "token": token}
                    shown = len(text)
                break
            safe = len(text) - _held_back(text, stops)
            if safe > shown:
                yield {"type": "token", "text": text[shown:safe], "token": token}
                shown = safe
            if cache.n + 1 > cache.n_ctx:
                reason = "context_full"
                break
            try:
                logits = self.model.forward([token], cache, cancelled=stopped)
            except StrifeError as exc:
                if exc.code == "strife.runtime.cancelled":
                    # A pass stopped partway leaves recurrent layers ahead of
                    # the tokens; go back to a state that is consistent.
                    cache.rewind(cache.n, force=True)
                    reason = self._why(cancel)
                    break
                raise
        if reason not in ("stop_string",):
            text += decoder.flush()
            if len(text) > shown and _find_stop(text, stops) is None:
                yield {"type": "token", "text": text[shown:], "token": None}
        t_end = time.perf_counter()
        decode_s = t_end - (t_first or t_end)
        n_gen = len(generated)
        stats = {
            "prompt_tokens": len(ids), "prompt_tokens_reused": reused,
            "prompt_tokens_run": len(ids) - reused, "generated_tokens": n_gen,
            "prefill_seconds": round(t_prefill, 6),
            "prefill_tokens_per_second": round((len(ids) - reused) / t_prefill, 3)
            if t_prefill > 0 else None,
            "time_to_first_token_seconds": round((t_first or t_end) - t0, 6),
            "decode_seconds": round(decode_s, 6),
            "decode_tokens_per_second": round((n_gen - 1) / decode_s, 3)
            if n_gen > 1 and decode_s > 0 else None,
            "total_seconds": round(t_end - t0, 6),
            "context_used": cache.n, "n_ctx": cache.n_ctx,
        }
        yield self._done(text, generated, reason, stats)

    @staticmethod
    def _why(cancel: threading.Event | None) -> str:
        return "cancelled" if cancel is not None and cancel.is_set() else "timeout"

    # -- reporting -----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._sessions_lock:
            sessions = len(self._sessions)
        return {"model": str(self.path), "architecture": self.report.architecture,
                "n_ctx": self.n_ctx, "threads": self.threads,
                "threads_applied": self.threads_applied, "sessions": sessions,
                "max_sessions": self.max_sessions, "load_seconds": round(self.load_seconds, 3),
                "weights": self.weights.stats(), "placement": self.placement}

    def close(self) -> None:
        self._closed = True
        with self._sessions_lock:
            self._sessions.clear()
        self.weights.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _find_stop(text: str, stops: list[str]) -> int | None:
    hits = [i for i in (text.find(s) for s in stops) if i >= 0]
    return min(hits) if hits else None


def _held_back(text: str, stops: list[str]) -> int:
    """How many trailing characters might be the start of a stop string."""
    held = 0
    for s in stops:
        for n in range(min(len(s) - 1, len(text)), 0, -1):
            if text.endswith(s[:n]):
                held = max(held, n)
                break
    return held


def _set_threads(threads: int | None) -> tuple[int, bool]:
    """Apply a BLAS thread count; say whether it took.

    With ``threadpoolctl`` it applies at any time. Without it, only the
    ``*_NUM_THREADS`` variables read when numpy first loads count, which
    ``hnx strife`` sets before importing numpy.
    """
    import os

    count = int(threads) if threads else (os.cpu_count() or 1)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(count)
        return count, True
    except ImportError:
        pass
    applied = os.environ.get("OPENBLAS_NUM_THREADS") == str(count) or (
        not threads and "OPENBLAS_NUM_THREADS" not in os.environ)
    return count, applied
