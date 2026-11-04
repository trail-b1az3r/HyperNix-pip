"""atc.traffic — how much is in flight, and what it has cost.

**In-flight limits.** At most ``max_inflight`` requests are inside the
application at once, and at most ``per_client_inflight`` from one client.
Past either, the request is answered 503 with ``Retry-After`` instead of
joining a queue that only grows: a server already working flat out is not
helped by accepting more, and one client opening a thousand slow
connections should not be able to take every worker. The rate limiter
counts requests per minute; this counts requests *right now*, which is
what a slow-loris or a stampede actually consumes.

**Statistics.** Per route (method and the first two path segments, so
the table stays small however many ids appear in paths): requests,
answers by status class, bytes in and out, and a latency sample for p50
and p95. Read at ``GET /atc/status``.
"""
from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Traffic", "route_key"]

#: Requests that never wait for a slot: an operator checking whether the
#: server is alive must not be told it is too busy to say.
_ALWAYS = ("/health", "/status", "/atc/status")


def route_key(method: str, path: str) -> str:
    parts = [p for p in path.split("/") if p][:2]
    return f"{method.upper()} /{'/'.join(parts)}"


@dataclass
class _RouteStats:
    requests: int = 0
    statuses: Counter = field(default_factory=Counter)
    bytes_in: int = 0
    bytes_out: int = 0
    latency_ms: deque = field(default_factory=lambda: deque(maxlen=256))

    def to_dict(self) -> dict[str, Any]:
        sample = sorted(self.latency_ms)

        def pct(p: float) -> float | None:
            return round(sample[min(len(sample) - 1, int(p * len(sample)))], 1) if sample else None

        return {"requests": self.requests, "statuses": dict(self.statuses),
                "bytes_in": self.bytes_in, "bytes_out": self.bytes_out,
                "latency_ms_p50": pct(0.5), "latency_ms_p95": pct(0.95)}


class Traffic:
    def __init__(self, *, max_inflight: int = 256, per_client_inflight: int = 32,
                 max_routes: int = 512) -> None:
        self.max_inflight = max(1, max_inflight)
        self.per_client_inflight = max(1, per_client_inflight)
        self.max_routes = max_routes
        self._lock = threading.Lock()
        self._inflight = 0
        self._per_client: Counter = Counter()
        self._routes: dict[str, _RouteStats] = {}
        self._refused = 0

    def enter(self, client: str, path: str) -> bool:
        """Take a slot. False when the server or this client has none left."""
        with self._lock:
            if not path.endswith(_ALWAYS):
                if self._inflight >= self.max_inflight or (
                        client and self._per_client[client] >= self.per_client_inflight):
                    self._refused += 1
                    return False
            self._inflight += 1
            if client:
                self._per_client[client] += 1
            return True

    def leave(self, client: str) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            if client:
                self._per_client[client] -= 1
                if self._per_client[client] <= 0:
                    del self._per_client[client]

    def record(self, method: str, path: str, status: int, *, bytes_in: int,
               bytes_out: int, ms: float) -> None:
        key = route_key(method, path)
        with self._lock:
            stats = self._routes.get(key)
            if stats is None:
                if len(self._routes) >= self.max_routes:
                    key, stats = "other", self._routes.setdefault("other", _RouteStats())
                else:
                    stats = self._routes[key] = _RouteStats()
            stats.requests += 1
            stats.statuses[f"{status // 100}xx"] += 1
            stats.bytes_in += bytes_in
            stats.bytes_out += bytes_out
            stats.latency_ms.append(ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "inflight": self._inflight, "max_inflight": self.max_inflight,
                "per_client_inflight": self.per_client_inflight,
                "busiest_clients": self._per_client.most_common(5),
                "refused_busy": self._refused,
                "routes": {k: v.to_dict() for k, v in sorted(self._routes.items())},
            }
