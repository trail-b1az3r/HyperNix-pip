"""t1api.webauth — where the sign-up page lives, and what that implies.

:mod:`hypernix.t1api.accounts` is the account store. This is the part
that knows *where the browser is*, because four deployments that all
serve the same pages need four different sets of cookie flags, and
getting them wrong breaks login in one direction or security in the
other.

The four modes
--------------
``local``
    ``http://127.0.0.1:8000``. One person, their own machine. The cookie
    cannot be ``Secure``: a Secure cookie is not sent over plain HTTP and
    there is no TLS on loopback, so setting it would produce a login that
    appears to succeed and then forgets you on the next page. Loopback is
    not on a network, so this is not a downgrade.

``tailscale``
    Served on the tailnet address. WireGuard has already encrypted and
    authenticated the link by the time HTTP starts, so ``Secure`` is
    again not required — and again cannot be set, because plain HTTP over
    ``tailscale0`` is what Tailscale gives you unless you go and get a
    MagicDNS certificate. If you have one, use ``site``.

``site``
    The host's own domain behind their own TLS. ``Secure`` is mandatory
    and :func:`cookie_kwargs` refuses to produce a cookie without it,
    because the whole point of this mode is that the traffic crosses a
    network somebody else can see.

``cloudflare``
    Behind a Cloudflare tunnel or proxy. As ``site``, plus: the client's
    real address arrives in ``CF-Connecting-IP`` and
    ``request.client.host`` is Cloudflare. Reading the wrong one binds
    every session in the world to one address and rate-limits the whole
    internet as a single client.

Trusting a forwarded header is a decision, not a default
---------------------------------------------------------
``X-Forwarded-For`` and ``CF-Connecting-IP`` are trivially forged by
whoever connects. They are only meaningful when something in front
*sets* them, and only when that something also *strips* what the client
sent. So :func:`client_ip` reads them in the two modes where a proxy is
declared to exist and ignores them in the two where one does not — which
means a ``local`` deployment cannot be told "my IP is 10.0.0.1" by a
client that says so.
"""
from __future__ import annotations

import ipaddress
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "WebAuthError",
    "WebAuthMode",
    "MODES",
    "SESSION_COOKIE",
    "CSRF_HEADER",
    "resolve_mode",
    "cookie_kwargs",
    "client_ip",
    "is_origin_allowed",
]

#: The session cookie's name. ``__Host-`` would be better — it pins the
#: cookie to an exact origin with no subdomain or path wiggle room — but
#: browsers require ``Secure`` for a ``__Host-`` cookie, which loopback
#: and plain-HTTP-over-Tailscale cannot provide. Rather than have the
#: cookie's *name* differ between deployments (and every bug report begin
#: with working out which one you have), it is one name, with the flags
#: doing the work.
SESSION_COOKIE = "t1_session"

#: Where the CSRF token is expected on a state-changing request. A custom
#: header is itself most of the defence: a cross-site form post cannot
#: set one, and a cross-site fetch that tries is stopped by CORS
#: preflight before it reaches this server.
CSRF_HEADER = "X-T1-CSRF"


class WebAuthError(Exception):
    """The web-auth configuration is not usable as given."""


@dataclass(frozen=True)
class WebAuthMode:
    """One deployment shape, and everything that follows from it."""

    name: str
    #: Whether the browser will be talking TLS to this server.
    secure_cookies: bool
    #: Whether something in front of this server sets a real-client-IP
    #: header. False means any such header is a client's invention.
    behind_proxy: bool
    #: Which header carries the real client address, when there is one.
    forwarded_header: str = ""
    #: Cookie SameSite policy.
    same_site: str = "lax"
    #: What a person needs to know to use this mode.
    summary: str = ""
    #: What to type to serve it.
    how: str = ""

    @property
    def needs_tls(self) -> bool:
        return self.secure_cookies


MODES: dict[str, WebAuthMode] = {
    "local": WebAuthMode(
        name="local",
        secure_cookies=False,
        behind_proxy=False,
        # Strict, not Lax: nothing off-machine can navigate to loopback
        # in a way that should carry a session, and this is the one mode
        # where the tighter setting costs nothing.
        same_site="strict",
        summary="http://127.0.0.1:8000 — one machine, nothing on a network.",
        how="T1_ACCOUNTS_ENABLED=1 T1_ACCOUNTS_MODE=local",
    ),
    "tailscale": WebAuthMode(
        name="tailscale",
        secure_cookies=False,
        behind_proxy=False,
        same_site="lax",
        summary=(
            "Served on your tailnet address. WireGuard has already encrypted "
            "and authenticated the link, so plain HTTP here is not plain "
            "HTTP on a network."
        ),
        how="T1_ACCOUNTS_ENABLED=1 T1_ACCOUNTS_MODE=tailscale",
    ),
    "site": WebAuthMode(
        name="site",
        secure_cookies=True,
        behind_proxy=True,
        forwarded_header="X-Forwarded-For",
        same_site="lax",
        summary=(
            "Your own domain, your own TLS, your own reverse proxy. The "
            "proxy must strip the client's X-Forwarded-For and set its own."
        ),
        how=(
            "T1_ACCOUNTS_ENABLED=1 T1_ACCOUNTS_MODE=site "
            "T1_ACCOUNTS_PUBLIC_URL=https://t1.example.com"
        ),
    ),
    "cloudflare": WebAuthMode(
        name="cloudflare",
        secure_cookies=True,
        behind_proxy=True,
        forwarded_header="CF-Connecting-IP",
        same_site="lax",
        summary=(
            "Behind a Cloudflare tunnel. No port to open and no certificate "
            "to manage; Cloudflare terminates TLS and sets CF-Connecting-IP."
        ),
        how=(
            "T1_ACCOUNTS_ENABLED=1 T1_ACCOUNTS_MODE=cloudflare "
            "T1_ACCOUNTS_PUBLIC_URL=https://t1.example.com"
        ),
    ),
}


@dataclass
class WebAuthSettings:
    """Resolved web-auth configuration for one running server."""

    mode: WebAuthMode
    #: The origin a browser will see, e.g. ``https://t1.example.com``.
    #: Used for the allowed-origin check and for the links in emails and
    #: CLI output that tell somebody where to go.
    public_url: str = ""
    #: Extra origins permitted to make state-changing requests.
    extra_origins: tuple[str, ...] = ()
    enabled: bool = False
    registration: str = "closed"
    invite_codes: tuple[str, ...] = field(default_factory=tuple, repr=False)

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        found = [origin for origin in (self.public_url, *self.extra_origins) if origin]
        return tuple(dict.fromkeys(found))

    def to_dict(self) -> dict[str, Any]:
        """Safe to serve at ``GET /accounts/config``. No invite codes."""
        return {
            "enabled": self.enabled,
            "mode": self.mode.name,
            "registration": self.registration,
            "invite_required": self.registration == "invite",
            "public_url": self.public_url,
            "secure_cookies": self.mode.secure_cookies,
            "behind_proxy": self.mode.behind_proxy,
            "same_site": self.mode.same_site,
        }


def resolve_mode(name: str | None = None) -> WebAuthMode:
    """The :class:`WebAuthMode` *name* selects, or the one from the env."""
    chosen = (name or os.environ.get("T1_ACCOUNTS_MODE", "local")).strip().lower()
    mode = MODES.get(chosen)
    if mode is None:
        raise WebAuthError(
            f"Unknown accounts mode {chosen!r}. One of: {', '.join(MODES)}.\n"
            + "\n".join(f"  {m.name:10} {m.summary}" for m in MODES.values())
        )
    return mode


def settings_from_env(mode_name: str | None = None) -> WebAuthSettings:
    """Read the whole web-auth configuration from the environment."""
    from .accounts import invite_codes_from_env, registration_mode_from_env

    mode = resolve_mode(mode_name)
    public_url = os.environ.get("T1_ACCOUNTS_PUBLIC_URL", "").strip().rstrip("/")
    extra = tuple(
        origin.strip().rstrip("/")
        for origin in os.environ.get("T1_ACCOUNTS_ORIGINS", "").split(",")
        if origin.strip()
    )
    enabled = os.environ.get("T1_ACCOUNTS_ENABLED", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )
    settings = WebAuthSettings(
        mode=mode,
        public_url=public_url,
        extra_origins=extra,
        enabled=enabled,
        registration=registration_mode_from_env(),
        invite_codes=invite_codes_from_env(),
    )
    if enabled:
        validate(settings)
    return settings


def validate(settings: WebAuthSettings) -> None:
    """Refuse a configuration that cannot work, before it is serving.

    Each of these is a mistake that produces a *working-looking* server
    with a hole or a dead login in it, which is exactly the class of
    mistake worth refusing at startup rather than discovering later.
    """
    mode = settings.mode
    if mode.needs_tls and not settings.public_url:
        raise WebAuthError(
            f"Accounts mode {mode.name!r} serves over the network, so it needs "
            "T1_ACCOUNTS_PUBLIC_URL to know which origin to "
            "trust. Without it the CSRF origin check has nothing to check "
            "against."
        )
    if mode.needs_tls and settings.public_url.startswith("http://"):
        raise WebAuthError(
            f"Accounts mode {mode.name!r} sets Secure cookies, and a Secure "
            f"cookie is never sent over http://. {settings.public_url} would "
            "produce a login that appears to work and forgets you on the next "
            "page. Use https://, or mode 'local' / 'tailscale'."
        )
    if settings.registration == "invite" and not settings.invite_codes:
        raise WebAuthError(
            "Registration is set to 'invite' and no invite codes are "
            "configured, so nobody can register. Set T1_ACCOUNTS_INVITE_CODES, "
            "or choose another registration mode."
        )
    if settings.registration == "open" and mode.name in ("site", "cloudflare"):
        # A warning rather than a refusal: somebody running a community
        # server may genuinely mean it. But "open registration on the
        # public internet" is far more often a local-mode default that
        # followed the server outside.
        logger.warning(
            "t1api.webauth: registration is OPEN on a %s deployment. Anyone "
            "who can reach %s can create an account and mint API keys. Set "
            "T1_ACCOUNTS_REGISTRATION=invite if that is not intended.",
            mode.name, settings.public_url or "this server",
        )


def cookie_kwargs(mode: WebAuthMode, *, max_age: int) -> dict[str, Any]:
    """Keyword arguments for ``response.set_cookie``.

    ``httponly`` is not configurable. The session cookie is never read by
    JavaScript in any of these deployments, so exposing it to script
    would be all cost — it is the difference between an XSS that defaces
    a page and an XSS that walks away with the session.
    """
    return {
        "key": SESSION_COOKIE,
        "max_age": max_age,
        "httponly": True,
        "secure": mode.secure_cookies,
        "samesite": mode.same_site,
        "path": "/",
    }


def client_ip(headers: Any, peer: str, mode: WebAuthMode) -> str:
    """The client's real address.

    *peer* is the socket's own idea of it. In the two modes where a proxy
    is declared, the header wins; in the two where one is not, the header
    is a client's invention and is ignored — otherwise a ``local``
    deployment could be told "my address is 10.0.0.1" by the client whose
    sessions and rate limits that address governs.
    """
    if not mode.behind_proxy or not mode.forwarded_header:
        return peer or ""
    raw = ""
    try:
        raw = headers.get(mode.forwarded_header, "") or ""
    except AttributeError:
        raw = ""
    # X-Forwarded-For is a list, client first, with each proxy appending.
    # The first entry is the one the closest trusted proxy saw.
    candidate = raw.split(",")[0].strip()
    if not candidate:
        return peer or ""
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        # A proxy that sets this sets an address. Anything else is either
        # a misconfiguration or a client probing what this parser does
        # with it, and neither should become a session binding.
        logger.warning(
            "t1api.webauth: %s carried %r, which is not an IP address; "
            "using the socket peer instead",
            mode.forwarded_header, candidate[:64],
        )
        return peer or ""
    return candidate


def is_origin_allowed(origin: str, settings: WebAuthSettings) -> bool:
    """Whether a browser at *origin* may make a state-changing request.

    A missing Origin header passes. Browsers omit it on same-origin form
    posts in some versions, and every cross-site request that matters
    carries one — so requiring it would break real logins to catch
    nothing. The CSRF token is what actually stops the attack; this is
    the cheap check in front of it.
    """
    if not origin:
        return True
    normalised = origin.strip().rstrip("/")
    allowed = settings.allowed_origins
    if not allowed:
        # No public URL configured, which only validate() permits for the
        # two loopback-ish modes. There is nothing to compare against, so
        # the CSRF token carries it alone.
        return True
    return normalised in allowed
