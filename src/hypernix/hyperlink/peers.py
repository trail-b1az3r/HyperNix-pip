"""Other HyperNix machines on this tailnet.

The problem this solves: someone has HyperNix on a desktop and a
laptop, pairs their phone with the desktop, and then wants the laptop
too — and has to go and find out what the laptop's tailnet name is.
The desktop already knows: it is on the same tailnet, and
``tailscale status`` lists every peer.

Three things about how this is done, each of which is the answer to
"why not the obvious thing":

**Discovery is not connection, and connection is not trust.** What comes
back is a list of *candidates*: addresses that answered a health check.
Nothing here authenticates anything, nothing here is paired, and every
result carries ``verified: false`` to say so in the payload rather than
only in the docs. A phone that acts on one of these still has to
authenticate against it, and the fingerprint it gets back is what tells
it whether the machine is the one it thinks.

**The name is not the identity.** A peer is reported with its tailnet
name because a human needs to pick from the list, but the name is
decoration. Two machines can claim the same one, and the tailnet name a
peer advertises is not something this machine can verify. The identity
is the fingerprint the peer returns from its own
``/hyperlink/endpoints`` — after a credential has been presented.

**It only ever looks; it never runs anything.** The probe is a GET to
``/health`` with a short timeout, and the only thing done with the
response is to check that it parses as HyperNix's health payload. No
part of a peer's response chooses a code path, names a file, or reaches
a shell.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .discovery import _tailscale_binary

logger = logging.getLogger(__name__)

__all__ = ["Peer", "tailnet_peers", "probe", "discover"]

#: Per-peer probe budget. A tailnet peer that is asleep or on a phone's
#: hotspot will not answer, and the scan must not sit on it: the whole
#: list is probed concurrently, so this is roughly the time the whole
#: call takes rather than the time per machine.
PROBE_TIMEOUT = 2.5

#: How many machines to look at. A personal tailnet is a handful; a work
#: one can be thousands, and scanning all of them from a request handler
#: is a way to make one HTTP call hold a connection for a minute.
MAX_PEERS = 64

#: Read from a peer's /health, and no more. A response bigger than this
#: is not a HyperNix server, and reading it in full would let a peer
#: decide how much memory this process spends.
MAX_BODY = 64 * 1024


@dataclass(frozen=True)
class Peer:
    """One tailnet machine, and whether it answered."""

    name: str
    address: str
    url: str
    online: bool = False
    #: True only when the address answered as a HyperNix server.
    reachable: bool = False
    t1_version: str = ""
    server_name: str = ""
    detail: str = ""
    os: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "address": self.address,
            "url": self.url,
            "online": self.online,
            "reachable": self.reachable,
            "t1_version": self.t1_version,
            # The name the peer calls itself. Decoration for a person
            # choosing from a list -- never an identity. See the module
            # docstring.
            "server_name": self.server_name,
            "detail": self.detail,
            "os": self.os,
            # Said in the payload, not only in the docs: nothing here has
            # been authenticated, and a client must not treat a match on
            # `name` or `server_name` as having found its server.
            "verified": False,
        }


def tailnet_peers(*, timeout: float = 5.0) -> list[Peer]:
    """Every peer ``tailscale status`` reports, unprobed.

    Never raises. No tailscale, not logged in, and no peers are all the
    same answer here — an empty list — because none of them is an error
    for a server that also works perfectly well on a LAN.
    """
    exe = _tailscale_binary()
    if not exe:
        return []
    try:
        proc = subprocess.run(  # noqa: S603
            [exe, "status", "--json"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        status = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []

    found: list[Peer] = []
    for entry in (status.get("Peer") or {}).values():
        if not isinstance(entry, dict):
            continue
        addresses = [
            a for a in (entry.get("TailscaleIPs") or [])
            if isinstance(a, str) and ":" not in a
        ]
        if not addresses:
            continue
        found.append(
            Peer(
                name=str(entry.get("DNSName") or "").rstrip("."),
                address=addresses[0],
                url="",
                online=bool(entry.get("Online")),
                os=str(entry.get("OS") or ""),
            )
        )
    found.sort(key=lambda p: (not p.online, p.name or p.address))
    return found[:MAX_PEERS]


def probe(peer: Peer, *, port: int, scheme: str = "http", timeout: float = PROBE_TIMEOUT) -> Peer:
    """Ask one peer whether it is a HyperNix server.

    A GET to ``/health``, nothing more. The response is parsed and two
    strings are copied out of it for display; it never selects
    behaviour.
    """
    host = peer.name or peer.address
    url = f"{scheme}://{host}:{port}"
    try:
        request = urllib.request.Request(  # noqa: S310 - scheme is ours, not a peer's
            f"{url}/health", headers={"accept": "application/json"}, method="GET"
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read(MAX_BODY)
            payload = json.loads(body)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        reason = getattr(exc, "reason", None) or exc
        return Peer(
            **{**peer.__dict__, "url": url, "reachable": False, "detail": str(reason)}
        )

    if not isinstance(payload, dict) or "status" not in payload:
        return Peer(
            **{
                **peer.__dict__, "url": url, "reachable": False,
                "detail": "something answered, but not a HyperNix server",
            }
        )
    return Peer(
        **{
            **peer.__dict__,
            "url": url,
            "reachable": True,
            "t1_version": str(payload.get("t1_api_version") or ""),
            "server_name": str(payload.get("server_name") or ""),
            "detail": "",
        }
    )


def discover(
    *, port: int, scheme: str = "http", timeout: float = PROBE_TIMEOUT,
    include_offline: bool = False,
) -> list[Peer]:
    """Tailnet peers that answer as HyperNix servers, best first.

    Probed concurrently: a machine that is asleep costs the timeout, and
    doing that one at a time turns a handful of sleeping laptops into a
    request that never returns.
    """
    peers = [p for p in tailnet_peers() if p.online or include_offline]
    if not peers:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(peers))) as pool:
        probed = list(
            pool.map(lambda p: probe(p, port=port, scheme=scheme, timeout=timeout), peers)
        )
    probed.sort(key=lambda p: (not p.reachable, p.name or p.address))
    return probed
