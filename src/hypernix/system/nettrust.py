"""Where a request came from, and how much that is worth.

0.72.4 adds a mode where a connection from the LAN or the tailnet may
act without a key. That is only safe if "from the LAN" is a fact about
the connection rather than a claim the connection makes, so this module
exists to keep one answer to that question in one place — the T1 API,
Waiter and the installer had three different notions of "local" between
them before it.

The rules, in the order they decide things
------------------------------------------
**The peer address is the evidence.** ``X-Forwarded-For`` and
``X-Real-IP`` are set by whoever is talking to you. Believing them turns
keyless LAN access into keyless access for anyone on the internet who
can spell a header, which is the single worst thing this feature could
do. They are read *only* when the immediate peer is a proxy the
administrator listed, and even then only the hop that proxy added.

**A tailnet address is a candidate, not a conclusion.** 100.64.0.0/10 is
shared address space. Anything on a LAN can number itself 100.x and
route to the server, so the range alone proves nothing. A tailnet origin
is confirmed by asking the local tailscaled who owns that address
(``tailscale whois``); without that confirmation it is treated as
public. This is what makes knowing the endpoint insufficient.

**Loopback is not the LAN.** It is strictly more trusted — the traffic
never touched a network — and it is what a reverse proxy on the same
host looks like, which is exactly the case where the forwarded header
matters.

What this module does not decide
---------------------------------
Whether trust is *enough*. :class:`TrustPolicy` answers "may this origin
act without a key", and it answers no unless an administrator turned
that on. The default is that nothing is keyless.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)

__all__ = [
    "Origin",
    "Trust",
    "TrustPolicy",
    "classify",
    "tailnet_identity",
    "trusted_proxies_from_env",
    "TAILNET_RANGE",
]

#: Tailscale's assigned range: RFC 6598 shared address space, *not*
#: RFC 1918. The distinction matters twice over — iOS's
#: NSAllowsLocalNetworking does not cover it either.
TAILNET_RANGE = ipaddress.ip_network("100.64.0.0/10")

#: What counts as "the LAN": RFC 1918, link-local, and IPv6 unique-local
#: and link-local. Spelled out rather than deferring to
#: ``ipaddress.is_private``, which is far broader than the name suggests
#: -- it is true for the documentation ranges (192.0.2/24, 198.51.100/24,
#: 203.0.113/24), for 0.0.0.0/8, for benchmarking and reserved space, and
#: for 100.64/10 itself. Trusting all of that as "local" would hand
#: keyless access to origins that are not on anybody's network, which is
#: exactly the mistake this module exists to not make. Caught by a test
#: address that happened to be TEST-NET-3.
_LAN_RANGES = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",   # link-local
        "fc00::/7",         # unique local
        "fe80::/10",        # link-local
    )
)

#: Where a tailscale binary hides when PATH is minimal, which is what a
#: systemd unit gives you.
_TAILSCALE_PATHS = (
    "/usr/bin/tailscale",
    "/usr/local/bin/tailscale",
    "/opt/homebrew/bin/tailscale",
    "/opt/bin/tailscale",
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
)


class Trust(StrEnum):
    """How a connection arrived, most trusted first."""

    LOOPBACK = "loopback"
    TAILNET = "tailnet"
    LAN = "lan"
    PUBLIC = "public"

    @property
    def rank(self) -> int:
        return {"loopback": 0, "tailnet": 1, "lan": 2, "public": 3}[self.value]


@dataclass(frozen=True)
class Origin:
    """A classified connection, and why it was classified that way."""

    address: str
    trust: Trust
    #: The tailnet node and user, when tailscaled confirmed them.
    tailnet_node: str = ""
    tailnet_user: str = ""
    #: True when a forwarded header was believed, and by whose leave.
    via_proxy: str = ""
    #: Plain-language reason, for logs and for `waiter whoami`.
    reason: str = ""

    @property
    def is_trusted_network(self) -> bool:
        """On the LAN or a confirmed tailnet, or the machine itself."""
        return self.trust is not Trust.PUBLIC

    def to_dict(self) -> dict:
        return {
            "address": self.address,
            "trust": self.trust.value,
            "tailnet_node": self.tailnet_node,
            "tailnet_user": self.tailnet_user,
            "via_proxy": self.via_proxy,
            "reason": self.reason,
        }


def trusted_proxies_from_env(
    value: str | None = None,
) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Proxies whose forwarded headers may be believed.

    Empty by default, and that default is the safe one: with no trusted
    proxy configured, no forwarded header is read at all.
    """
    raw = value if value is not None else os.environ.get("T1_TRUSTED_PROXIES", "")
    networks = []
    for chunk in raw.replace(";", ",").split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            networks.append(ipaddress.ip_network(text, strict=False))
        except ValueError:
            logger.warning(
                "system.nettrust: T1_TRUSTED_PROXIES entry %r is not an address "
                "or network; ignoring it", text
            )
    return networks


def _parse(address: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    text = (address or "").strip().strip("[]")
    if not text:
        return None
    # A "host:port" peer, as several servers report it.
    if text.count(":") == 1 and "." in text:
        text = text.rsplit(":", 1)[0]
    if "%" in text:  # scoped IPv6, fe80::1%eth0
        text = text.split("%", 1)[0]
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        return None


def tailnet_identity(address: str, *, timeout: float = 4.0) -> tuple[str, str]:
    """``(node, user)`` if tailscaled says *address* is on this tailnet.

    ``("", "")`` when it is not, when tailscale is absent, or when the
    query fails for any reason. Failing closed is the point: an
    unanswerable question about identity is not a yes.
    """
    binary = shutil.which("tailscale") or next(
        (p for p in _TAILSCALE_PATHS if os.path.exists(p)), ""
    )
    if not binary:
        return "", ""
    try:
        proc = subprocess.run(
            [binary, "whois", "--json", address],
            capture_output=True, text=True, encoding="utf-8",
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("system.nettrust: tailscale whois failed: %s", exc)
        return "", ""
    if proc.returncode != 0 or not proc.stdout.strip():
        return "", ""
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        return "", ""
    node = str((payload.get("Node") or {}).get("Name", ""))
    user = str((payload.get("UserProfile") or {}).get("LoginName", ""))
    return node, user


def classify(
    peer: str,
    *,
    forwarded_for: str = "",
    trusted_proxies: list | None = None,
    verify_tailnet: bool = True,
    timeout: float = 4.0,
) -> Origin:
    """Classify the connection from *peer*.

    ``forwarded_for`` is read **only** when *peer* is itself in
    ``trusted_proxies``. Anyone can send the header; only a proxy the
    administrator named gets to be believed about it.
    """
    proxies = trusted_proxies if trusted_proxies is not None else trusted_proxies_from_env()
    address = _parse(peer)
    if address is None:
        return Origin(
            address=peer, trust=Trust.PUBLIC,
            reason=f"{peer!r} is not an address this could classify, so it is "
                   f"treated as public.",
        )

    via = ""
    if forwarded_for and proxies and any(address in net for net in proxies):
        # The last hop the trusted proxy added is the only one it can
        # vouch for; everything to the left was supplied by the client.
        candidate = _parse(forwarded_for.split(",")[-1].strip())
        if candidate is not None:
            via, address = str(address), candidate
    elif forwarded_for:
        # A forwarded header from a peer we cannot vouch for. The header
        # itself is worthless -- but its *presence* means something is
        # probably relaying, and we cannot tell what.
        #
        # This is the reverse-proxy trap, and it is the dangerous
        # direction. nginx or caddy on the same host makes every request
        # in the world arrive from 127.0.0.1, which is the *most* trusted
        # level here. Trust that, turn keyless mode on, and the entire
        # internet has keyless access to a server whose operator believes
        # it is reachable only from their LAN.
        #
        # So an unverifiable forwarded header collapses the origin to
        # public. A direct client that sends a junk header only denies
        # itself keyless access, which is a much better failure than the
        # alternative.
        logger.debug(
            "system.nettrust: %s sent a forwarded header and is not a "
            "configured trusted proxy; treating the origin as public", peer,
        )
        return Origin(
            str(address), Trust.PUBLIC,
            reason=f"{peer} sent a forwarded header but is not a configured "
                   f"trusted proxy, so what is behind it cannot be established. "
                   f"Set T1_TRUSTED_PROXIES if this is your reverse proxy.",
        )

    text = str(address)
    if address.is_loopback:
        return Origin(text, Trust.LOOPBACK, via_proxy=via,
                      reason="the connection never left this machine.")

    if address.version == 4 and address in TAILNET_RANGE:
        if not verify_tailnet:
            return Origin(text, Trust.TAILNET, via_proxy=via,
                          reason="in the tailnet range (identity not checked).")
        node, user = tailnet_identity(text, timeout=timeout)
        if node or user:
            return Origin(
                text, Trust.TAILNET, tailnet_node=node, tailnet_user=user,
                via_proxy=via,
                reason=f"tailscaled identifies this as {node or 'a tailnet node'}"
                       + (f" belonging to {user}" if user else "") + ".",
            )
        # In the range and unconfirmed. Anything on a LAN can number
        # itself 100.x, so the range alone is not evidence.
        return Origin(
            text, Trust.PUBLIC, via_proxy=via,
            reason="in Tailscale's range but tailscaled does not recognise it, "
                   "so it is not treated as a tailnet peer.",
        )

    if any(address in net for net in _LAN_RANGES):
        return Origin(text, Trust.LAN, via_proxy=via,
                      reason="a private address on this network.")

    return Origin(text, Trust.PUBLIC, via_proxy=via,
                  reason="a public address.")


@dataclass(frozen=True)
class TrustPolicy:
    """Whether an origin may act without a key, and how far.

    Every field defaults to off. Trusted-network mode is something an
    administrator turns on knowing what it costs, not something that
    happens because a machine has a private address.
    """

    #: Keyless connections permitted from these levels.
    keyless: frozenset[Trust] = field(default_factory=frozenset)
    #: Levels that additionally get partial administrative functions.
    partial_admin: frozenset[Trust] = field(default_factory=frozenset)

    @classmethod
    def off(cls) -> TrustPolicy:
        return cls()

    @classmethod
    def from_config(
        cls, *, trusted_network: bool = False, include_lan: bool = True,
        include_tailnet: bool = True, partial_admin: bool = False,
    ) -> TrustPolicy:
        if not trusted_network:
            return cls.off()
        levels = {Trust.LOOPBACK}
        if include_lan:
            levels.add(Trust.LAN)
        if include_tailnet:
            levels.add(Trust.TAILNET)
        return cls(
            keyless=frozenset(levels),
            partial_admin=frozenset(levels) if partial_admin else frozenset(),
        )

    def allows_keyless(self, origin: Origin) -> bool:
        # Public is never keyless, whatever the configuration says. The
        # check is here as well as in from_config so that a hand-built
        # policy cannot express it either.
        if origin.trust is Trust.PUBLIC:
            return False
        return origin.trust in self.keyless

    def allows_partial_admin(self, origin: Origin) -> bool:
        if origin.trust is Trust.PUBLIC:
            return False
        return origin.trust in self.partial_admin

    @property
    def enabled(self) -> bool:
        return bool(self.keyless)
