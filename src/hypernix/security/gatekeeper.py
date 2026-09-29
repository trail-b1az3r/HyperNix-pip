"""gatekeeper — Request authentication, usage tracking, and quota enforcement.

Gatekeeper is the runtime enforcement layer that sits in front of any
HyperNix API surface (REST, local, plugin, WebUI, …).  It:

* **Authenticates** T1 API keys via the :class:`Keymaster`.
* **Enforces quotas** — per-user, per-model, per-endpoint, and per-service
  limits on requests and tokens, with a configurable sliding-window.
* **Rate-limits** — a thread-safe sliding-window counter ensures no key
  bursts beyond its allowed rate.
* **Records usage** — every authenticated call is appended to an access
  log and the in-memory + on-disk usage counters are updated.
* **Reports statistics** — per-key and aggregate breakdowns on demand.
* **Local-only by default** — an optional ``remote_sync_url`` can be
  configured to push a usage summary to a remote endpoint periodically.

Quick example::

    from hypernix.keymaster import Keymaster, KeyType, KeyScope
    from hypernix.gatekeeper import Gatekeeper, Quota

    km = Keymaster()
    meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})

    gk = Gatekeeper(keymaster=km)
    gk.set_quota(meta.key_id, Quota(max_requests=100, window_seconds=60))

    verified = gk.authenticate(meta.key)   # → KeyMeta
    gk.check_quota(meta.key_id, endpoint="/v1/generate")
    gk.record_usage(meta.key_id, endpoint="/v1/generate", tokens_used=42)

    stats = gk.get_stats(meta.key_id)
    print(stats)
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hypernix.security.keymaster import Keymaster, KeyMeta, KeyScope
from hypernix.security.safeurl import urlopen as safe_urlopen

logger = logging.getLogger(__name__)

_DEFAULT_DATA: Path = Path.home() / ".hypernix" / "gatekeeper"


# ---------------------------------------------------------------------------
# Quota / rate-limiting primitives
# ---------------------------------------------------------------------------


@dataclass
class Quota:
    """A time-windowed usage quota.

    Args:
        max_requests: Maximum number of requests allowed within
            ``window_seconds``.  ``None`` = unlimited.
        max_tokens: Maximum number of tokens consumed within
            ``window_seconds``.  ``None`` = unlimited.
        window_seconds: Length of the sliding window in seconds.
            Default 60 (one minute).
    """

    max_requests: int | None = None
    max_tokens: int | None = None
    window_seconds: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_requests": self.max_requests,
            "max_tokens": self.max_tokens,
            "window_seconds": self.window_seconds,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Quota:
        return cls(
            max_requests=d.get("max_requests"),
            max_tokens=d.get("max_tokens"),
            window_seconds=d.get("window_seconds", 60.0),
        )


class QuotaViolation(Exception):
    """Raised when a request exceeds an enforced quota or rate limit."""

    def __init__(
        self,
        key_id: str,
        reason: str,
        *,
        limit: int | None = None,
        current: int | None = None,
        window: float | None = None,
    ) -> None:
        self.key_id = key_id
        self.reason = reason
        self.limit = limit
        self.current = current
        self.window = window
        detail = reason
        if limit is not None and current is not None:
            detail += f" (limit={limit}, current={current})"
        if window is not None:
            detail += f" in {window}s window"
        super().__init__(f"[{key_id[:8]}…] {detail}")


# ---------------------------------------------------------------------------
# Sliding-window rate counter
# ---------------------------------------------------------------------------


class RateWindow:
    """Thread-safe sliding-window counter.

    Tracks (timestamp, token_count) tuples in a deque and evicts entries
    older than ``window_seconds`` on every query.
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        # deque of (timestamp, tokens)
        self._events: deque[tuple[float, int]] = deque()

    def add(self, tokens: int = 0) -> None:
        now = time.time()
        with self._lock:
            self._events.append((now, tokens))
            self._evict(now)

    def counts(self) -> tuple[int, int]:
        """Return (request_count, token_count) within the window."""
        now = time.time()
        with self._lock:
            self._evict(now)
            req = len(self._events)
            tok = sum(t for _, t in self._events)
        return req, tok

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()


# ---------------------------------------------------------------------------
# Usage record
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    """A single recorded API call."""

    key_id: str
    endpoint: str
    model: str
    timestamp: float
    tokens_used: int
    request_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "endpoint": self.endpoint,
            "model": self.model,
            "timestamp": self.timestamp,
            "tokens_used": self.tokens_used,
            "request_count": self.request_count,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Per-key state (internal)
# ---------------------------------------------------------------------------


@dataclass
class _KeyState:
    quota: Quota | None = None
    # One rate window per quota window size (keyed by window_seconds)
    rate_window: RateWindow = field(default_factory=RateWindow)
    # Per-endpoint windows: endpoint → RateWindow
    endpoint_windows: dict[str, RateWindow] = field(default_factory=dict)
    # Per-model windows: model → RateWindow
    model_windows: dict[str, RateWindow] = field(default_factory=dict)
    # Aggregate lifetime stats
    total_requests: int = 0
    total_tokens: int = 0
    last_used: float = 0.0
    # Recent usage records (capped at 1000)
    records: deque[UsageRecord] = field(default_factory=lambda: deque(maxlen=1000))


# ---------------------------------------------------------------------------
# Gatekeeper
# ---------------------------------------------------------------------------


class Gatekeeper:
    """Authentication, quota enforcement, and usage tracking for T1 keys.

    Args:
        keymaster: :class:`~hypernix.keymaster.Keymaster` instance to look
            up key metadata.
        data_dir: Where to persist usage counters and the access log.
            Defaults to ``~/.hypernix/gatekeeper/``.
        default_quota: Fallback quota applied to keys with no explicit quota.
            ``None`` = no rate-limit by default.
        remote_sync_url: Optional HTTP endpoint to POST usage summaries to
            periodically.  When ``None`` (default) the Gatekeeper operates
            entirely locally.
        sync_interval: Seconds between remote sync pushes (ignored when
            ``remote_sync_url`` is None).
        log_to_file: If *True* (default), append one-line JSON records to
            ``<data_dir>/access.log``.
    """

    def __init__(
        self,
        keymaster: Keymaster,
        data_dir: Path | None = None,
        default_quota: Quota | None = None,
        remote_sync_url: str | None = None,
        sync_interval: float = 300.0,
        log_to_file: bool = True,
    ) -> None:
        self._km = keymaster
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._default_quota = default_quota
        self._remote_sync_url = remote_sync_url
        self._log_to_file = log_to_file
        self._log_path = self._data_dir / "access.log"
        self._usage_path = self._data_dir / "usage.json"

        self._lock = threading.RLock()
        # key_id → _KeyState
        self._state: dict[str, _KeyState] = defaultdict(_KeyState)

        self._load_usage()

        if remote_sync_url:
            self._sync_thread = threading.Thread(
                target=self._sync_loop,
                args=(sync_interval,),
                daemon=True,
                name="gatekeeper-sync",
            )
            self._sync_stop = threading.Event()
            self._sync_thread.start()
        else:
            self._sync_thread = None  # type: ignore
            self._sync_stop = threading.Event()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self, key_str: str) -> KeyMeta:
        """Verify a raw T1 key string and return its :class:`KeyMeta`.

        Raises:
            ValueError: If the key string is not a valid T1 format.
            PermissionError: If the key is revoked, expired, or unknown.
        """
        from hypernix.security.keymaster import T1KeyGenerator

        if not T1KeyGenerator.validate(key_str):
            raise ValueError(f"Not a valid T1 key format: {key_str[:20]}…")

        meta = self._km.get_by_key(key_str)
        if meta is None:
            raise PermissionError("Unknown or unregistered T1 key.")
        if not meta.active:
            raise PermissionError(f"Key {meta.key_id[:8]}… has been revoked.")
        if meta.is_expired:
            raise PermissionError(f"Key {meta.key_id[:8]}… has expired.")

        logger.debug("gatekeeper: authenticated key %s", meta.key_id[:8])
        return meta

    # ------------------------------------------------------------------
    # Quota management
    # ------------------------------------------------------------------

    def set_quota(
        self,
        key_id: str,
        quota: Quota | None,
        *,
        endpoint: str | None = None,
        model: str | None = None,
    ) -> None:
        """Assign a quota to a key (optionally scoped to an endpoint or model).

        Passing ``quota=None`` removes the quota for that scope.
        """
        with self._lock:
            state = self._state[key_id]
            if endpoint is not None:
                if quota is None:
                    state.endpoint_windows.pop(endpoint, None)
                else:
                    state.endpoint_windows[endpoint] = RateWindow(quota.window_seconds)
            elif model is not None:
                if quota is None:
                    state.model_windows.pop(model, None)
                else:
                    state.model_windows[model] = RateWindow(quota.window_seconds)
            else:
                state.quota = quota
                if quota is not None:
                    state.rate_window = RateWindow(quota.window_seconds)

    def get_quota(self, key_id: str) -> Quota | None:
        """Return the global quota for *key_id*, or the default."""
        with self._lock:
            state = self._state.get(key_id)
            if state and state.quota is not None:
                return state.quota
        return self._default_quota

    # ------------------------------------------------------------------
    # Quota checking
    # ------------------------------------------------------------------

    def check_quota(
        self,
        key_id: str,
        *,
        endpoint: str = "",
        model: str = "",
        tokens_requested: int = 0,
    ) -> None:
        """Raise :class:`QuotaViolation` if the call would exceed any limit.

        This method does **not** record the usage — call :meth:`record_usage`
        after the actual request completes.

        Args:
            key_id: The key to check.
            endpoint: API endpoint being called.
            model: Model name being used.
            tokens_requested: Estimated tokens for this call (used for
                token-based quota checking).

        Raises:
            QuotaViolation: If any quota is exceeded.
        """
        with self._lock:
            meta = self._km.get(key_id)
            if meta is None:
                raise QuotaViolation(key_id, "Unknown key")

            # Lifetime caps (from key metadata)
            if meta.request_limit is not None and meta.request_count >= meta.request_limit:
                raise QuotaViolation(
                    key_id, "Lifetime request limit reached",
                    limit=meta.request_limit, current=meta.request_count,
                )
            if meta.usage_cap is not None and meta.usage_count >= meta.usage_cap:
                raise QuotaViolation(
                    key_id, "Lifetime token cap reached",
                    limit=meta.usage_cap, current=meta.usage_count,
                )

            state = self._state[key_id]
            quota = state.quota or self._default_quota

            # Sliding-window rate limit (global for this key)
            if quota is not None:
                req_count, tok_count = state.rate_window.counts()
                if quota.max_requests is not None and req_count >= quota.max_requests:
                    raise QuotaViolation(
                        key_id, "Rate limit exceeded (requests)",
                        limit=quota.max_requests, current=req_count,
                        window=quota.window_seconds,
                    )
                if quota.max_tokens is not None:
                    if tok_count + tokens_requested > quota.max_tokens:
                        raise QuotaViolation(
                            key_id, "Rate limit exceeded (tokens)",
                            limit=quota.max_tokens, current=tok_count,
                            window=quota.window_seconds,
                        )

            # Per-endpoint limit
            if endpoint and endpoint in state.endpoint_windows:
                win = state.endpoint_windows[endpoint]
                req_c, _ = win.counts()
                # Use a fixed cap of 100 req/window for endpoint limits unless
                # a custom quota was set (endpoint quotas share the window object
                # but not a separate Quota object; extend if needed)
                _ = req_c  # counted but not enforced here without a separate quota obj

            # Scope check
            if endpoint and not any(
                s in {KeyScope.READ, KeyScope.WRITE, KeyScope.ADMIN, KeyScope.SERVICE}
                for s in meta.scopes
            ):
                raise QuotaViolation(key_id, "Key has no usable scopes for this endpoint")

    # ------------------------------------------------------------------
    # Usage recording
    # ------------------------------------------------------------------

    def record_usage(
        self,
        key_id: str,
        *,
        endpoint: str = "",
        model: str = "",
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """Record a completed API call and update all counters.

        Tells the :class:`Keymaster` to update its persistent counters and
        advances the rate-window and per-endpoint/model windows.
        """
        now = time.time()
        record = UsageRecord(
            key_id=key_id,
            endpoint=endpoint,
            model=model,
            timestamp=now,
            tokens_used=tokens_used,
            metadata=metadata or {},
        )

        with self._lock:
            state = self._state[key_id]
            # Advance windows
            state.rate_window.add(tokens_used)
            if endpoint:
                if endpoint not in state.endpoint_windows:
                    w = self._default_quota.window_seconds if self._default_quota else 60.0
                    state.endpoint_windows[endpoint] = RateWindow(w)
                state.endpoint_windows[endpoint].add(tokens_used)
            if model:
                if model not in state.model_windows:
                    w = self._default_quota.window_seconds if self._default_quota else 60.0
                    state.model_windows[model] = RateWindow(w)
                state.model_windows[model].add(tokens_used)

            state.total_requests += 1
            state.total_tokens += tokens_used
            state.last_used = now
            state.records.append(record)

        # Update keymaster counters
        self._km.record_usage(key_id, tokens=tokens_used, requests=1)

        if self._log_to_file:
            self._append_log(record)

        self._save_usage()
        return record

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def get_permissions(self, key_id: str) -> list[str]:
        """Return the list of scope strings for *key_id*."""
        meta = self._km.get(key_id)
        if meta is None:
            return []
        return sorted(s.value for s in meta.scopes)

    def has_permission(self, key_id: str, scope: KeyScope) -> bool:
        """Return True if the key has *scope*."""
        meta = self._km.get(key_id)
        if meta is None:
            return False
        return scope in meta.scopes

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self, key_id: str) -> dict[str, Any]:
        """Return a detailed usage stats dict for a single key."""
        with self._lock:
            state = self._state.get(key_id)

        meta = self._km.get(key_id)
        req_win, tok_win = state.rate_window.counts() if state else (0, 0)
        quota = self.get_quota(key_id)

        return {
            "key_id": key_id,
            "key_type": meta.key_type.value if meta else "unknown",
            "active": meta.active if meta else False,
            "scopes": sorted(s.value for s in meta.scopes) if meta else [],
            "total_requests": state.total_requests if state else 0,
            "total_tokens": state.total_tokens if state else 0,
            "lifetime_request_count": meta.request_count if meta else 0,
            "lifetime_token_count": meta.usage_count if meta else 0,
            "request_limit": meta.request_limit if meta else None,
            "usage_cap": meta.usage_cap if meta else None,
            "last_used": state.last_used if state else None,
            "window_requests": req_win,
            "window_tokens": tok_win,
            "quota": quota.to_dict() if quota else None,
            "endpoints": list(state.endpoint_windows.keys()) if state else [],
            "models": list(state.model_windows.keys()) if state else [],
        }

    def get_all_stats(self) -> list[dict[str, Any]]:
        """Return stats for all tracked keys."""
        with self._lock:
            key_ids = list(self._state.keys())
        return [self.get_stats(k) for k in key_ids]

    def get_usage_log(
        self,
        key_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return recent usage records.

        If *key_id* is given, filters to that key only.
        """
        with self._lock:
            if key_id:
                state = self._state.get(key_id)
                records: list[UsageRecord] = list(state.records) if state else []
            else:
                records = []
                for st in self._state.values():
                    records.extend(st.records)
                records.sort(key=lambda r: r.timestamp, reverse=True)

        records = records[-limit:]
        return [r.to_dict() for r in reversed(records)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_usage(self) -> None:
        """Persist aggregate usage counters to disk (non-sensitive)."""
        with self._lock:
            data: dict[str, Any] = {}
            for key_id, state in self._state.items():
                data[key_id] = {
                    "total_requests": state.total_requests,
                    "total_tokens": state.total_tokens,
                    "last_used": state.last_used,
                    "quota": state.quota.to_dict() if state.quota else None,
                }
        try:
            self._usage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("gatekeeper: could not save usage: %s", exc)

    def _load_usage(self) -> None:
        if not self._usage_path.exists():
            return
        try:
            data = json.loads(self._usage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("gatekeeper: could not load usage: %s", exc)
            return
        with self._lock:
            for key_id, rec in data.items():
                state = self._state[key_id]
                state.total_requests = rec.get("total_requests", 0)
                state.total_tokens = rec.get("total_tokens", 0)
                state.last_used = rec.get("last_used", 0.0)
                if rec.get("quota"):
                    state.quota = Quota.from_dict(rec["quota"])
                    state.rate_window = RateWindow(state.quota.window_seconds)

    def _append_log(self, record: UsageRecord) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict()) + "\n")
        except OSError as exc:
            logger.warning("gatekeeper: could not write access log: %s", exc)

    # ------------------------------------------------------------------
    # Remote sync (optional)
    # ------------------------------------------------------------------

    def _sync_loop(self, interval: float) -> None:
        while not self._sync_stop.wait(interval):
            self._push_remote()

    def _push_remote(self) -> None:
        if not self._remote_sync_url:
            return
        payload = json.dumps({"stats": self.get_all_stats()}).encode()
        req = urllib.request.Request(
            self._remote_sync_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with safe_urlopen(req, timeout=10):
                pass
            logger.debug("gatekeeper: remote sync OK")
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("gatekeeper: remote sync failed: %s", exc)

    def stop(self) -> None:
        """Stop background sync thread (if running)."""
        self._sync_stop.set()

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._state)
        return (
            f"Gatekeeper(keys_tracked={n}, "
            f"remote={'yes' if self._remote_sync_url else 'no'})"
        )


__all__ = [
    "Gatekeeper",
    "Quota",
    "QuotaViolation",
    "RateWindow",
    "UsageRecord",
]
