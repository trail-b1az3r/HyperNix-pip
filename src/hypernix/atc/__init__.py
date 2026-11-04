"""atc — API traffic control for the T1 API (0.73.0 pre-alpha 1).

Manages, optimises, end-to-end encrypts and protects the traffic that
reaches a T1 server:

* :mod:`.shield` -- notices attackers by behaviour (probing, key guessing,
  API mapping) and jails them with growing timeouts;
* :mod:`.traffic` -- in-flight limits per server and per client, and
  per-route statistics;
* :mod:`.e2e` -- ``hypernix-atc/1`` end-to-end encryption: a handshake
  signed by the server's identity key, sealed requests and framed sealed
  replies;
* :mod:`.config` -- the ``T1_ATC_*`` settings.

* :mod:`.middleware` -- all of that as one ASGI layer, outermost in the
  T1 API;
* :mod:`.client` -- :class:`~.client.ATCTunnel`, a client for the tunnel.

The routes are ``/atc/*`` (``t1api/routers/atc.py``).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import ATCSettings
from .shield import Shield
from .traffic import Traffic

__all__ = ["ATC", "ATCSettings"]


class ATC:
    """The parts, built from one :class:`ATCSettings`, that the middleware
    and the ``/atc`` routes share (``app.state.t1_atc``)."""

    def __init__(self, settings: ATCSettings | None = None, *, signer: Any = None,
                 clock: Callable[[], float] | None = None) -> None:
        from . import e2e

        self.settings = settings or ATCSettings()
        s = self.settings
        shield_kwargs: dict[str, Any] = {"clock": clock} if clock else {}
        self.shield = Shield(probe_strikes=s.probe_strikes, auth_fail_limit=s.auth_fail_limit,
                             notfound_limit=s.notfound_limit, window=s.window_seconds,
                             jail_base=s.jail_seconds, jail_max=s.jail_max_seconds,
                             jail_loopback=s.jail_loopback, **shield_kwargs)
        self.traffic = Traffic(max_inflight=s.max_inflight, per_client_inflight=s.client_inflight)
        self.sessions = e2e.SessionStore(signer, ttl=s.session_seconds) if s.e2e else None
        self.stats: dict[str, int] = {
            "tunnelled": 0, "tunnel_refused": 0, "probes": 0, "jailed_refused": 0,
            "too_large": 0, "e2e_required_refused": 0, "compressed": 0,
            "compressed_saved_bytes": 0}

    def snapshot(self) -> dict[str, Any]:
        from . import e2e

        return {
            "settings": self.settings.to_dict(),
            "e2e": {"protocol": e2e.PROTOCOL, "algorithm": e2e.ALGORITHM,
                    "available": bool(self.sessions and self.sessions.ready),
                    "sessions": self.sessions.count() if self.sessions else 0},
            "shield": self.shield.snapshot(),
            "traffic": self.traffic.snapshot(),
            "counters": dict(self.stats),
        }
