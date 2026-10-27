"""t1api.counter — how many public T1 servers are up, counted hourly.

Two halves (0.72.8a3):

**The heartbeat.** A T1 server that is *public* — it advertises an
address (``T1_SERVER_URL`` or ``T1_HYPERLINK_PUBLIC_URL``) that is not
this machine, a private network or the tailnet, or says ``T1_PUBLIC=1``
— and runs T1 v1.1.26.10.1.0 or newer sends one small POST an hour to
the counter, unless ``T1_PUBLIC_COUNTER=0``. It carries exactly three
things::

    {"id": "<32 hex>", "t1_version": "1.1.26.10.2.0", "version": "0.72.8a3"}

``id`` is a SHA-256 of a random value this server generated once and
keeps (``counter-id`` beside its database), so the counter can tell one
server from two without learning which server it is. No address, name,
URL, model, key or usage is sent; a private server sends nothing at all.

**The counter.** Any T1 server can be one with ``T1_COUNTER_HOST=1``:
``POST /counter/heartbeat`` takes a heartbeat (no key needed, small
bodies only, the sender's address never stored), and ``GET /counter``
answers how many distinct servers were heard from in the last two hours
and on which versions. The repository's hourly workflow reads that and
updates the badge in the README.

Where heartbeats go is ``T1_PUBLIC_COUNTER_URL``, else the address the
project publishes in ``docs/public/v1/t1-counter-endpoint.json`` (read
once a day), so the counter can move without a release. While that file
names no counter, nothing is sent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..security.safeurl import urlopen as safe_urlopen
from .version import T1Version

logger = logging.getLogger(__name__)

__all__ = ["COUNTED_SINCE", "Counter", "CounterPinger", "heartbeat", "instance_id", "is_public"]

#: The first T1 version that is counted.
COUNTED_SINCE = T1Version(api=1, major=1, year=2026, month=10, feature=1, fix=0)
ENDPOINT_FILE_URL = ("https://raw.githubusercontent.com/trail-b1az3r/HyperNix-pip/main/"
                     "docs/public/v1/t1-counter-endpoint.json")
INTERVAL = 3600.0
#: A server counts as up if heard from within this long (two missed beats are a restart, not an outage).
WINDOW = 2 * 3600.0


def _private_host(host: str) -> bool:
    import ipaddress

    host = host.strip().lower().strip("[]")
    if not host or host in ("localhost",) or host.endswith((".local", ".lan", ".internal", ".ts.net", ".home")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host            # a bare machine name is a local one
    return not address.is_global          # private, loopback, link-local, CGNAT/tailnet, reserved


def is_public(config: Any) -> bool:
    """Whether this server says it is reachable from the internet."""
    if getattr(config, "public", False):
        return True
    for url in (getattr(config, "server_url", ""), getattr(config, "hyperlink_public_url", "")):
        if url and not _private_host(urlsplit(url if "://" in url else f"https://{url}").hostname or ""):
            return True
    return False


def instance_id(folder: Path) -> str:
    """This server's anonymous counter id: a hash of a random value kept on disk."""
    # A seed, not a secret: it says nothing about the server, unlocks
    # nothing, and exists so the id is stable across restarts. (Named
    # `secret`, CodeQL read it as one and its file as clear-text storage.)
    path = folder / "counter-id"
    try:
        seed = path.read_text(encoding="utf-8").strip()
    except OSError:
        seed = ""
    if len(seed) < 32:
        seed = hashlib.sha256(b"hypernix-t1-counter-seed:" + os.urandom(32)).hexdigest()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path.write_text(seed, encoding="utf-8")
            path.chmod(0o600)
        except OSError:
            logger.debug("t1api.counter: could not keep the counter id", exc_info=True)
    return hashlib.sha256(f"hypernix-t1-counter:{seed}".encode()).hexdigest()[:32]


def heartbeat(folder: Path) -> dict[str, str]:
    from .. import __version__
    from .version import T1_VERSION

    return {"id": instance_id(folder), "t1_version": T1_VERSION.short, "version": __version__}


def should_count(config: Any) -> tuple[bool, str]:
    """``(yes, why not)`` for this server."""
    from .version import T1_VERSION

    if not getattr(config, "public_counter", True):
        return False, "T1_PUBLIC_COUNTER=0"
    if not is_public(config):
        return False, "not public (no public T1_SERVER_URL / T1_HYPERLINK_PUBLIC_URL, and T1_PUBLIC is off)"
    if T1_VERSION < COUNTED_SINCE:
        return False, f"T1 {T1_VERSION.short} is older than {COUNTED_SINCE.short}"
    return True, ""


def _post(url: str, body: dict[str, str]) -> bool:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "hypernix-t1-counter"})
    try:
        with safe_urlopen(request, timeout=15) as reply:  # nosec B310 - the configured counter
            return 200 <= reply.status < 300
    except Exception:  # noqa: BLE001 - a missed beat is the next hour's problem
        logger.debug("t1api.counter: heartbeat to %s failed", url, exc_info=True)
        return False


def published_endpoint() -> str:
    """The counter the project publishes, or ``""`` while it publishes none."""
    try:
        with safe_urlopen(ENDPOINT_FILE_URL, timeout=15) as reply:  # nosec B310 - fixed project URL
            data = json.loads(reply.read(4096))
    except Exception:  # noqa: BLE001
        return ""
    url = str(data.get("heartbeat") or "") if isinstance(data, dict) else ""
    return url if url.startswith("https://") else ""


class CounterPinger:
    """Sends the hourly heartbeat from a daemon thread."""

    def __init__(self, config: Any, folder: Path, *, post=_post, endpoint=published_endpoint) -> None:
        self.config = config
        self.folder = folder
        self._post = post
        self._endpoint = endpoint
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._url, self._url_at = "", 0.0
        self.sent = 0

    def url(self) -> str:
        configured = getattr(self.config, "public_counter_url", "")
        if configured:
            return configured
        if time.monotonic() - self._url_at > 86400 or not self._url_at:
            self._url, self._url_at = self._endpoint(), time.monotonic()
        return self._url

    def beat(self) -> bool:
        ok, _why = should_count(self.config)
        url = self.url() if ok else ""
        if not url:
            return False
        sent = self._post(url, heartbeat(self.folder))
        self.sent += int(sent)
        return sent

    def start(self) -> bool:
        ok, why = should_count(self.config)
        if not ok:
            logger.info("t1api.counter: not counted (%s)", why)
            return False
        logger.info("t1api.counter: this public server sends an hourly anonymous heartbeat "
                    "(an id hash and its version); T1_PUBLIC_COUNTER=0 stops it")
        self._thread = threading.Thread(target=self._loop, name="t1-counter", daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        # A first beat a minute or so after start, so a crash loop is not a flood.
        delay = random.uniform(30, 120)  # nosec B311 - jitter, not security
        while not self._stop.wait(delay):
            self.beat()
            delay = INTERVAL + random.uniform(-300, 300)  # nosec B311 - jitter

    def stop(self) -> None:
        self._stop.set()


class Counter:
    """The counting side: who was heard from, and when. No addresses are kept."""

    def __init__(self, path: Path | None = None, *, clock=time.time) -> None:
        self.path = path
        self.clock = clock
        self._seen: dict[str, tuple[float, str, str]] = {}
        self._lock = threading.Lock()
        if path and path.is_file():
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
                self._seen = {k: (float(v[0]), str(v[1]), str(v[2])) for k, v in rows.items()}
            except (ValueError, TypeError, IndexError):
                self._seen = {}

    def record(self, body: Any) -> None:
        if not isinstance(body, dict):
            raise ValueError("a heartbeat is a JSON object")
        ident = str(body.get("id", ""))
        if len(ident) != 32 or any(c not in "0123456789abcdef" for c in ident):
            raise ValueError("id is 32 lower-case hex characters")
        try:
            t1 = T1Version.parse(str(body.get("t1_version", "")))
        except ValueError as exc:
            raise ValueError(f"t1_version: {exc}") from exc
        if t1 < COUNTED_SINCE:
            raise ValueError(f"only T1 {COUNTED_SINCE.short} and newer are counted")
        version = str(body.get("version", ""))[:32]
        now = self.clock()
        with self._lock:
            self._seen[ident] = (now, t1.short, version)
            self._seen = {k: v for k, v in self._seen.items() if now - v[0] < 7 * 86400}
            self._save()

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: list(v) for k, v in self._seen.items()}), encoding="utf-8")
        tmp.replace(self.path)

    def summary(self) -> dict[str, Any]:
        now = self.clock()
        with self._lock:
            live = [v for v in self._seen.values() if now - v[0] < WINDOW]
        versions: dict[str, int] = {}
        for _seen, t1, _version in live:
            versions[t1] = versions.get(t1, 0) + 1
        return {"up": len(live), "t1_versions": dict(sorted(versions.items())),
                "window_hours": WINDOW / 3600, "counted_since": COUNTED_SINCE.short,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))}
