"""atc.config — ATC's settings, from the environment.

| Variable | Default | Meaning |
|---|---|---|
| ``T1_ATC_ENABLED`` | ``1`` | the whole layer; ``0`` takes it out of the server |
| ``T1_ATC_SHIELD`` | ``1`` | probe detection and the jail |
| ``T1_ATC_JAIL_LOOPBACK`` | ``0`` | jail this machine's own addresses too |
| ``T1_ATC_PROBE_STRIKES`` | ``3`` | probes before a jail |
| ``T1_ATC_AUTH_FAIL_LIMIT`` | ``30`` | 401/403 answers per window before a jail |
| ``T1_ATC_NOTFOUND_LIMIT`` | ``120`` | 404 answers per window before a jail |
| ``T1_ATC_WINDOW_SECONDS`` | ``60`` | the window those two count over |
| ``T1_ATC_JAIL_SECONDS`` | ``60`` | the first jail; each repeat doubles it |
| ``T1_ATC_JAIL_MAX_SECONDS`` | ``3600`` | the longest jail |
| ``T1_ATC_MAX_INFLIGHT`` | ``256`` | requests inside the server at once |
| ``T1_ATC_CLIENT_INFLIGHT`` | ``32`` | of those, from one client |
| ``T1_ATC_MAX_BODY_MB`` | ``128`` | the largest request body accepted |
| ``T1_ATC_COMPRESS`` | ``1`` | gzip complete JSON/text replies of 1 KiB or more |
| ``T1_ATC_E2E`` | ``1`` | offer end-to-end encryption (``/atc/handshake``) |
| ``T1_ATC_E2E_REQUIRED`` | ``0`` | refuse unencrypted requests from other machines |
| ``T1_ATC_SESSION_SECONDS`` | ``3600`` | how long an encrypted session lasts |
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = ["ATCSettings"]


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _num(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class ATCSettings:
    enabled: bool = field(default_factory=lambda: _bool("T1_ATC_ENABLED", True))
    shield: bool = field(default_factory=lambda: _bool("T1_ATC_SHIELD", True))
    jail_loopback: bool = field(default_factory=lambda: _bool("T1_ATC_JAIL_LOOPBACK", False))
    probe_strikes: int = field(default_factory=lambda: int(_num("T1_ATC_PROBE_STRIKES", 3)))
    auth_fail_limit: int = field(default_factory=lambda: int(_num("T1_ATC_AUTH_FAIL_LIMIT", 30)))
    notfound_limit: int = field(default_factory=lambda: int(_num("T1_ATC_NOTFOUND_LIMIT", 120)))
    window_seconds: float = field(default_factory=lambda: _num("T1_ATC_WINDOW_SECONDS", 60))
    jail_seconds: float = field(default_factory=lambda: _num("T1_ATC_JAIL_SECONDS", 60))
    jail_max_seconds: float = field(default_factory=lambda: _num("T1_ATC_JAIL_MAX_SECONDS", 3600))
    max_inflight: int = field(default_factory=lambda: int(_num("T1_ATC_MAX_INFLIGHT", 256)))
    client_inflight: int = field(default_factory=lambda: int(_num("T1_ATC_CLIENT_INFLIGHT", 32)))
    max_body_bytes: int = field(default_factory=lambda: int(_num("T1_ATC_MAX_BODY_MB", 128) * 1024 * 1024))
    compress: bool = field(default_factory=lambda: _bool("T1_ATC_COMPRESS", True))
    e2e: bool = field(default_factory=lambda: _bool("T1_ATC_E2E", True))
    e2e_required: bool = field(default_factory=lambda: _bool("T1_ATC_E2E_REQUIRED", False))
    session_seconds: float = field(default_factory=lambda: _num("T1_ATC_SESSION_SECONDS", 3600))

    def to_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return asdict(self)
