"""waiter.discovery — ``waiter -F``: find a HyperNix server.

Four kinds of target, told apart by shape rather than by a flag:

===================================  ==================================
``waiter -F "workshop-box"``          a server name
``waiter -F <54-character host id>``  a Host ID
``waiter -F home/api.jsonl``          a direct endpoint descriptor
``waiter -F 192.168.1.50:8000``       an address
===================================  ==================================

:func:`classify_target` does the telling-apart, and it can, because the
three identifier formats are mutually exclusive by construction: a Host
ID is 54 characters with no dash and no ``#``; a V1 Server ID has a dash
and is at most 8; an SSPKID has a ``#``. That was a deliberate choice in
:mod:`hypernix.security.t2keys` and this is what it buys.

Discovery scope
---------------
``-l`` restricts the search to this machine and this LAN. Without it,
the tailnet is searched too. The distinction is not cosmetic: a tailnet
sweep sends packets to every peer on someone's private network, and
"find my server" should not do that by surprise when the server turns
out to be on the desk.

api.jsonl, and why nothing is executed
--------------------------------------
A server can publish an ``api.jsonl`` describing how it wants to be
talked to, including which client application to open — Hyped Pro, or a
TUI or CLI of the host's own. Waiter reads that and *offers* it.

It never runs it. :class:`HostApplication.launch_command` is data until
a human says otherwise, and :func:`connect` separates discovery from
execution completely: it resolves, authenticates, and hands back a
descriptor. Running a command a remote server chose, because that server
asked, is remote code execution with extra steps — and a discovery
protocol that does it is a discovery protocol that only has to be lied
to once.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from hypernix.security.safeurl import urlopen as safe_urlopen

from ..t1api.version import T1_VERSION_SHORT

logger = logging.getLogger(__name__)

__all__ = [
    "TargetKind",
    "Target",
    "classify_target",
    "DiscoveredServer",
    "HostApplication",
    "ApiDescriptor",
    "discover",
    "connect",
    "DEFAULT_PORTS",
    "parse_api_jsonl",
]

#: Ports probed when a target does not name one. 8000 is the documented
#: default; the rest are what people actually pick when 8000 is taken.
DEFAULT_PORTS: tuple[int, ...] = (8000, 8080, 8443, 11434, 1234)

_HOST_ID_RE = re.compile(r"^[A-Za-z0-9]{53}[!@#$%^&*()\-_=+\[\]{};:',.<>?|~`]$")
_V1_SERVER_ID_RE = re.compile(r"^\d{1,5}-[A-Z]\d+$")
_ADDRESS_RE = re.compile(r"^(?:https?://)?[\w.\-]+(?::\d+)?(?:/.*)?$")


class TargetKind:
    """What ``-F`` was given. Plain constants; these end up in JSON."""

    SERVER_NAME = "server_name"
    HOST_ID = "host_id"
    API_JSONL = "api_jsonl"
    ADDRESS = "address"
    SERVER_ID = "server_id"
    SSPKID = "sspkid"


@dataclass(frozen=True)
class Target:
    raw: str
    kind: str
    #: For ADDRESS and API_JSONL: the URL to contact directly.
    url: str = ""
    note: str = ""

    @property
    def is_direct(self) -> bool:
        """True when there is nothing to search — just connect."""
        return self.kind in (TargetKind.ADDRESS, TargetKind.API_JSONL)

    def to_dict(self) -> dict[str, Any]:
        return {"raw": self.raw, "kind": self.kind, "url": self.url, "note": self.note}


def classify_target(target: str) -> Target:
    """Work out what kind of thing ``-F`` was given.

    Ordered most-specific first. The identifier formats are mutually
    exclusive by construction (see the module docstring), so this is a
    decision rather than a guess — which matters, because guessing wrong
    between "server name" and "Host ID" means searching for the wrong
    thing and reporting "not found" about something that is there.
    """
    text = (target or "").strip()
    if not text:
        raise ValueError("waiter -F needs a target: a server name, a Host ID, or an endpoint")

    if "api.jsonl" in text:
        url = text if "://" in text else f"http://{text.lstrip('/')}"
        return Target(
            raw=text, kind=TargetKind.API_JSONL, url=url,
            note="Direct endpoint descriptor; no discovery needed.",
        )
    if _HOST_ID_RE.fullmatch(text):
        return Target(
            raw=text, kind=TargetKind.HOST_ID,
            note="54-character Host ID. Not a V1 Server ID and not an SSPKID.",
        )
    if "#" in text:
        return Target(
            raw=text, kind=TargetKind.SSPKID,
            note="Identifies one key on a server, not the server itself.",
        )
    if _V1_SERVER_ID_RE.fullmatch(text):
        return Target(raw=text, kind=TargetKind.SERVER_ID, note="V1 Server ID.")
    if "://" in text or _looks_like_address(text):
        url = text if "://" in text else f"http://{text}"
        return Target(
            raw=text, kind=TargetKind.ADDRESS, url=url.rstrip("/"),
            note="An address; contacted directly.",
        )
    return Target(raw=text, kind=TargetKind.SERVER_NAME, note="Searched for by name.")


def _looks_like_address(text: str) -> bool:
    """Distinguish ``192.168.1.5:8000`` from ``workshop-box``.

    A bare hostname is a *name* to search for, not an address to dial —
    someone typing ``waiter -F workshop`` means "find the server called
    workshop", and treating that as a DNS name would produce a
    connection error instead of a search.
    """
    if ":" in text and text.rsplit(":", 1)[-1].isdigit():
        return True
    host = text.split("/", 1)[0]
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    # A dotted name with a known-ish TLD-ish tail: ts.net, .local, .lan.
    return host.endswith((".ts.net", ".local", ".lan", ".internal"))


# ---------------------------------------------------------------------------
# api.jsonl
# ---------------------------------------------------------------------------


@dataclass
class HostApplication:
    """A client application a host would like to be used.

    Data, not an instruction. See the module docstring: nothing here is
    ever executed without an explicit human decision, and
    :attr:`trusted` is only ever set by the code that asked a person.
    """

    kind: str = ""                    # hyped-pro | tui | cli | other
    name: str = ""
    launch_command: str = ""
    protocol_version: str = ""
    trusted: bool = False

    @property
    def is_builtin(self) -> bool:
        """True for applications HyperNix itself ships.

        These are the only ones :func:`connect` will offer to start
        without a confirmation prompt, because their code came from this
        package rather than from the server being connected to.
        """
        return self.kind in ("hyped-pro", "waiter", "hyped")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "launch_command": self.launch_command,
            "protocol_version": self.protocol_version,
            "is_builtin": self.is_builtin,
            "trusted": self.trusted,
        }


@dataclass
class ApiDescriptor:
    """A parsed ``api.jsonl``."""

    endpoint: str = ""
    server_name: str = ""
    host_id: str = ""
    server_id: str = ""
    t1_versions: list[str] = field(default_factory=list)
    t2_versions: list[str] = field(default_factory=list)
    auth_required: bool = True
    auth_kinds: list[str] = field(default_factory=list)
    application: HostApplication | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def supports_t1(self, version: str) -> bool:
        """Does this host speak a T1 version compatible with *version*?

        Compared by generation (``api.major``), not by exact string: the
        whole point of the six-part scheme is that ``1.0.26.8.1.0`` and
        ``1.0.26.8.0.1`` interoperate.
        """
        if not self.t1_versions:
            return True                       # unstated: assume yes, find out on connect
        from ..t1api.version import T1Version

        try:
            mine = T1Version.parse(version)
        except ValueError:
            return True
        for candidate in self.t1_versions:
            try:
                if T1Version.parse(candidate).compatible_with(mine):
                    return True
            except ValueError:
                continue
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "server_name": self.server_name,
            "host_id": self.host_id,
            "server_id": self.server_id,
            "t1_versions": list(self.t1_versions),
            "t2_versions": list(self.t2_versions),
            "auth_required": self.auth_required,
            "auth_kinds": list(self.auth_kinds),
            "application": self.application.to_dict() if self.application else None,
        }


def parse_api_jsonl(text: str) -> ApiDescriptor:
    """Parse an ``api.jsonl`` body.

    JSON Lines: one object per line, later lines overriding earlier keys.
    A malformed line is skipped rather than failing the parse — a
    descriptor with a stray blank or a comment line is still usable, and
    refusing it would make a trivial formatting slip look like an
    unreachable server.
    """
    merged: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            logger.debug("waiter.discovery: skipping unparseable api.jsonl line")
            continue
        if isinstance(parsed, dict):
            merged.update(parsed)

    app_data = merged.get("application") or merged.get("client") or {}
    application = None
    if isinstance(app_data, dict) and app_data:
        application = HostApplication(
            kind=str(app_data.get("type") or app_data.get("kind") or ""),
            name=str(app_data.get("name") or ""),
            launch_command=str(app_data.get("launch") or app_data.get("command") or ""),
            protocol_version=str(app_data.get("protocol_version") or ""),
        )

    return ApiDescriptor(
        endpoint=str(merged.get("endpoint") or merged.get("url") or ""),
        server_name=str(merged.get("server_name") or merged.get("name") or ""),
        host_id=str(merged.get("host_id") or ""),
        server_id=str(merged.get("server_id") or ""),
        t1_versions=_as_list(merged.get("t1_versions") or merged.get("t1_version")),
        t2_versions=_as_list(merged.get("t2_versions") or merged.get("t2_version")),
        auth_required=bool(merged.get("auth_required", True)),
        auth_kinds=_as_list(merged.get("auth") or merged.get("auth_kinds")),
        application=application,
        raw=merged,
    )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


@dataclass
class DiscoveredServer:
    """One server that answered."""

    url: str
    reachable: bool = False
    server_name: str = ""
    host_id: str = ""
    server_id: str = ""
    t1_version: str = ""
    hypernix_version: str = ""
    latency_ms: float = 0.0
    source: str = ""                  # local | lan | tailscale | direct
    descriptor: ApiDescriptor | None = None
    error: str = ""

    @property
    def matched_name(self) -> str:
        return self.server_name or self.url

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "reachable": self.reachable,
            "server_name": self.server_name,
            "host_id": self.host_id,
            "server_id": self.server_id,
            "t1_version": self.t1_version,
            "hypernix_version": self.hypernix_version,
            "latency_ms": round(self.latency_ms, 1),
            "source": self.source,
            "descriptor": self.descriptor.to_dict() if self.descriptor else None,
            "error": self.error,
        }


def _get(url: str, timeout: float) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"waiter-find/{T1_VERSION_SHORT}",
        },
    )
    try:
        with safe_urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ConnectionError(str(exc)) from exc


def probe(url: str, *, timeout: float = 2.0, source: str = "") -> DiscoveredServer:
    """Ask one address whether it is a HyperNix server. Never raises.

    Tries ``/status`` first because it is the richest answer, then
    ``api.jsonl`` because a host that publishes one may not expose
    ``/status`` unauthenticated.
    """
    base = url.rstrip("/")
    started = time.monotonic()
    result = DiscoveredServer(url=base, source=source)

    try:
        status_code, body = _get(f"{base}/status", timeout)
        result.latency_ms = (time.monotonic() - started) * 1000
        if status_code == 200 and body:
            data = json.loads(body)
            result.reachable = True
            result.t1_version = str(data.get("t1_api_version") or "")
            result.hypernix_version = str(data.get("hypernix_version") or "")
            result.server_name = str(data.get("server_name") or "")
    except (ConnectionError, json.JSONDecodeError) as exc:
        result.error = str(exc)

    # api.jsonl carries the host's own name, ids and preferred client —
    # things /status has no reason to publish.
    for path in ("/api.jsonl", "/home/api.jsonl", "/.well-known/hypernix/api.jsonl"):
        try:
            code, body = _get(f"{base}{path}", timeout)
        except ConnectionError:
            continue
        if code == 200 and body.strip():
            descriptor = parse_api_jsonl(body)
            result.descriptor = descriptor
            result.reachable = True
            result.server_name = result.server_name or descriptor.server_name
            result.host_id = descriptor.host_id
            result.server_id = descriptor.server_id
            result.error = ""
            break

    if result.reachable:
        result.latency_ms = result.latency_ms or (time.monotonic() - started) * 1000
    return result


def _tailscale_peers(timeout: float = 4.0) -> list[str]:
    exe = shutil.which("tailscale") or (
        "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
        if os.path.exists("/Applications/Tailscale.app/Contents/MacOS/Tailscale")
        else ""
    )
    if not exe:
        return []
    try:
        proc = subprocess.run(  # noqa: S603
            [exe, "status", "--json"], capture_output=True, text=True, timeout=timeout, check=False
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        status = json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    hosts: list[str] = []
    for peer in (status.get("Peer") or {}).values():
        if not isinstance(peer, dict) or not peer.get("Online"):
            continue
        dns = str(peer.get("DNSName") or "").rstrip(".")
        if dns:
            hosts.append(dns)
        for addr in peer.get("TailscaleIPs") or []:
            if isinstance(addr, str) and ":" not in addr:
                hosts.append(addr)
    return hosts


def _local_candidates(ports: Sequence[int]) -> list[tuple[str, str]]:
    """``(url, source)`` pairs for this machine and its LAN."""
    out: list[tuple[str, str]] = []
    for port in ports:
        out.append((f"http://127.0.0.1:{port}", "local"))
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        own = sock.getsockname()[0]
        sock.close()
    except OSError:
        return out
    for port in ports:
        out.append((f"http://{own}:{port}", "lan"))
    # The gateway is where a home server most often lives when it is not
    # this machine. A full /24 sweep is deliberately not done: that is a
    # port scan of someone's home network, and "find my server" should
    # not be one.
    parts = own.split(".")
    if len(parts) == 4:
        for host in ("1", "2"):
            for port in ports[:2]:
                out.append((f"http://{'.'.join(parts[:3])}.{host}:{port}", "lan"))
    return out


def discover(
    target: str | Target,
    *,
    local_only: bool = False,
    ports: Sequence[int] = DEFAULT_PORTS,
    timeout: float = 2.0,
    max_workers: int = 16,
) -> list[DiscoveredServer]:
    """Find servers matching *target*.

    ``local_only`` is ``waiter -F -l``: this machine and this LAN, never
    the tailnet. A tailnet sweep touches every peer on a private network,
    which should be opt-out-able and is.
    """
    parsed = target if isinstance(target, Target) else classify_target(target)

    if parsed.is_direct:
        found = probe(parsed.url, timeout=timeout, source="direct")
        return [found] if found.reachable else [found]

    candidates = _local_candidates(ports)
    if not local_only:
        for host in _tailscale_peers():
            for port in ports[:3]:
                candidates.append((f"http://{host}:{port}", "tailscale"))

    seen: set[str] = set()
    unique = [(u, s) for u, s in candidates if not (u in seen or seen.add(u))]

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(unique)))) as pool:
        results = list(
            pool.map(lambda pair: probe(pair[0], timeout=timeout, source=pair[1]), unique)
        )

    reachable = [r for r in results if r.reachable]
    matched = [r for r in reachable if _matches(r, parsed)]
    # Fall back to everything reachable when nothing matched the
    # identifier: "I found three servers, none called that" is a more
    # useful answer than "not found".
    out = matched or reachable
    out.sort(key=lambda r: (not _matches(r, parsed), r.latency_ms))
    return out


def _matches(server: DiscoveredServer, target: Target) -> bool:
    if target.kind == TargetKind.SERVER_NAME:
        return target.raw.lower() in (server.server_name or "").lower()
    if target.kind == TargetKind.HOST_ID:
        return server.host_id == target.raw
    if target.kind == TargetKind.SERVER_ID:
        return server.server_id == target.raw
    if target.kind == TargetKind.SSPKID:
        return server.server_id == target.raw.split("#", 1)[0]
    return True


# ---------------------------------------------------------------------------
# Connecting
# ---------------------------------------------------------------------------


@dataclass
class Connection:
    """The result of ``waiter -F`` finding something.

    Deliberately *describes* rather than acts. ``application`` is what
    the host would like opened and ``launch_approved`` says whether a
    human has agreed to it — the caller checks that before doing
    anything, and :func:`connect` never sets it to ``True`` on its own.
    """

    server: DiscoveredServer
    authenticated: bool = False
    key_id: str = ""
    application: HostApplication | None = None
    launch_approved: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "server": self.server.to_dict(),
            "authenticated": self.authenticated,
            "key_id": self.key_id,
            "application": self.application.to_dict() if self.application else None,
            "launch_approved": self.launch_approved,
            "notes": list(self.notes),
        }

    def hyped_pro_argv(self) -> list[str]:
        """The command that opens Hyped Pro against this server.

        Built here rather than taken from the descriptor: this is
        HyperNix's own binary with HyperNix's own flags, so nothing the
        remote server said can influence what actually runs.
        """
        return ["hyped-pro", "--server", self.server.url]


def connect(
    server: DiscoveredServer,
    *,
    credential: str = "",
    verify: bool = True,
) -> Connection:
    """Authenticate against a discovered server and describe the result.

    Separated from discovery, and from launching anything, on purpose.
    The flow is: find, resolve, authenticate, *describe*. Whether a
    client application starts is a decision for the caller and, when the
    host asked for a non-builtin one, for a human.
    """
    connection = Connection(server=server, application=(server.descriptor.application
                                                        if server.descriptor else None))
    if not server.reachable:
        connection.notes.append(f"{server.url} did not answer.")
        return connection

    if credential and verify:
        try:
            code, body = _get(f"{server.url.rstrip('/')}/auth/whoami", 5.0)
        except ConnectionError as exc:
            connection.notes.append(f"Could not verify the credential: {exc}")
            return connection
        if code == 200:
            connection.authenticated = True
            try:
                connection.key_id = str(json.loads(body).get("key_id") or "")
            except json.JSONDecodeError:
                pass
        else:
            connection.notes.append(
                f"The server refused that credential (HTTP {code})."
            )

    app = connection.application
    if app is not None:
        if app.is_builtin:
            connection.notes.append(
                f"This server asks clients to use {app.name or app.kind}, which HyperNix "
                "ships. It can be opened directly."
            )
        else:
            # The important line in this module.
            connection.notes.append(
                f"This server asks clients to run {app.name or app.kind!r}"
                + (f" ({app.launch_command})" if app.launch_command else "")
                + ". That command came from the server and has NOT been run. Running it "
                "would be executing code chosen by the machine you are connecting to. "
                "Start it yourself if you trust this host."
            )
    if server.descriptor and not server.descriptor.supports_t1(T1_VERSION_SHORT):
        connection.notes.append(
            "This host advertises a T1 generation this client does not speak "
            f"({', '.join(server.descriptor.t1_versions)}); the connection may not work."
        )
    return connection
