"""atc.shield — notice an attacker by what they do, and stop answering them.

Three behaviours mark a client, none of which an honest one shows:

* **Probing** for software this server does not run: ``/.env``,
  ``/.git/``, ``/wp-admin``, ``/phpmyadmin``, ``../`` in a path, and the
  rest of :data:`PROBE_PATTERNS`. A scanner tries hundreds of these; a
  person never types one. Each is a strike, answered 404 without reaching
  the application, and ``probe_strikes`` of them is a jail.
* **Guessing keys**: more than ``auth_fail_limit`` 401/403 answers inside
  ``window`` seconds.
* **Mapping the API**: more than ``notfound_limit`` 404 answers inside
  ``window`` seconds.

A jailed client is answered 429 with ``Retry-After`` until its time is
up, and each repeat offence doubles the time, up to ``jail_max``. This
sits in front of the rate limiter and the network policy rather than
replacing them: those count *how much*, this notices *what*.

Loopback is exempt unless ``jail_loopback`` is on, so a misbehaving
script on the server itself cannot lock its own operator out. Memory is
bounded: at most ``max_clients`` addresses are tracked, the oldest
dropped first.
"""
from __future__ import annotations

import ipaddress
import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PROBE_PATTERNS", "Shield", "is_probe"]

#: Paths no T1 server serves and every scanner asks for. Matched against
#: the path and the raw query, lower-cased and once percent-decoded.
PROBE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(re.compile(p) for p in (
    r"/\.env(\.|/|$)",
    r"/\.git(/|$)",
    r"/\.(svn|hg|aws|ssh|docker|kube)(/|$)",
    r"/\.ds_store$",
    r"/wp-(admin|login|content|includes|json)",
    r"/xmlrpc\.php",
    r"/php(my|my-)?admin",
    r"\.(php|asp|aspx|jsp|cgi)(\?|$)",
    r"/cgi-bin/",
    r"(^|/)\.\.(/|\\|$)",
    r"/etc/(passwd|shadow)",
    r"/(actuator|server-status|server-info)(/|$)",
    r"/vendor/phpunit",
    r"/(boaform|hnap1|goform)(/|$)",
    r"/console(/|$)",
    r"/solr/",
    r"\$\{jndi:",
))


def is_probe(path: str, query: str = "") -> bool:
    """Does this request look for something only an attacker would want?"""
    from urllib.parse import unquote

    text = (path + ("?" + query if query else "")).lower()
    decoded = unquote(text)
    return any(p.search(text) or p.search(decoded) for p in PROBE_PATTERNS)


@dataclass
class _Record:
    strikes: int = 0
    offences: int = 0
    jailed_until: float = 0.0
    reason: str = ""
    auth_failures: deque = field(default_factory=deque)
    not_found: deque = field(default_factory=deque)


class Shield:
    def __init__(
        self,
        *,
        probe_strikes: int = 3,
        auth_fail_limit: int = 30,
        notfound_limit: int = 120,
        window: float = 60.0,
        jail_base: float = 60.0,
        jail_max: float = 3600.0,
        jail_loopback: bool = False,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.probe_strikes = max(1, probe_strikes)
        self.auth_fail_limit = max(1, auth_fail_limit)
        self.notfound_limit = max(1, notfound_limit)
        self.window = window
        self.jail_base = jail_base
        self.jail_max = max(jail_base, jail_max)
        self.jail_loopback = jail_loopback
        self.max_clients = max(16, max_clients)
        self._clock = clock
        self._lock = threading.Lock()
        self._clients: OrderedDict[str, _Record] = OrderedDict()

    # -- who is exempt -----------------------------------------------------

    def exempt(self, client: str) -> bool:
        """Never jailed: no address to key on, or this machine itself."""
        if not client:
            return True
        if self.jail_loopback:
            return False
        try:
            return ipaddress.ip_address(client.split("%")[0]).is_loopback
        except ValueError:
            return client == "localhost"

    def _record(self, client: str) -> _Record:
        record = self._clients.get(client)
        if record is None:
            record = self._clients[client] = _Record()
            while len(self._clients) > self.max_clients:
                self._clients.popitem(last=False)
        else:
            self._clients.move_to_end(client)
        return record

    # -- the questions the middleware asks ---------------------------------

    def jailed(self, client: str) -> float:
        """Seconds of jail left for *client*, 0 when it may be served."""
        if self.exempt(client):
            return 0.0
        with self._lock:
            record = self._clients.get(client)
            if record is None:
                return 0.0
            left = record.jailed_until - self._clock()
            return left if left > 0 else 0.0

    def inspect(self, client: str, path: str, query: str = "") -> str | None:
        """A strike's reason when the request is a probe, else None.

        The caller answers a probe 404 without passing it on, whether or
        not this strike jailed the client.
        """
        if not is_probe(path, query):
            return None
        if self.exempt(client):
            return "probe"
        with self._lock:
            record = self._record(client)
            record.strikes += 1
            if record.strikes >= self.probe_strikes:
                self._jail(record, "probing for files and software this server does not run")
                record.strikes = 0
        return "probe"

    def observe(self, client: str, status: int) -> None:
        """Count the answers that, in bulk, mean someone is guessing."""
        if status not in (401, 403, 404) or self.exempt(client):
            return
        now = self._clock()
        with self._lock:
            record = self._record(client)
            bucket = record.not_found if status == 404 else record.auth_failures
            limit = self.notfound_limit if status == 404 else self.auth_fail_limit
            bucket.append(now)
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) > limit:
                bucket.clear()
                self._jail(record, "too many refused credentials" if status != 404
                           else "too many requests for things that do not exist")

    def _jail(self, record: _Record, reason: str) -> None:
        seconds = min(self.jail_max, self.jail_base * (2 ** record.offences))
        record.offences += 1
        record.jailed_until = self._clock() + seconds
        record.reason = reason

    # -- the operator's controls -------------------------------------------

    def jail(self, client: str, reason: str = "jailed by an administrator") -> float:
        with self._lock:
            record = self._record(client)
            self._jail(record, reason)
            return record.jailed_until - self._clock()

    def release(self, client: str) -> bool:
        with self._lock:
            record = self._clients.get(client)
            if record is None or record.jailed_until <= self._clock():
                return False
            record.jailed_until = 0.0
            record.strikes = 0
            return True

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            jailed = [
                {"client": client, "seconds_left": round(r.jailed_until - now, 1),
                 "offences": r.offences, "reason": r.reason}
                for client, r in self._clients.items() if r.jailed_until > now
            ]
            return {"tracked_clients": len(self._clients), "jailed": jailed,
                    "probe_strikes": self.probe_strikes, "auth_fail_limit": self.auth_fail_limit,
                    "notfound_limit": self.notfound_limit, "window_seconds": self.window,
                    "jail_loopback": self.jail_loopback}
