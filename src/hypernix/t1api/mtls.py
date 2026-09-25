"""t1api.mtls — TLS and mutual-TLS support.

"Add optional mTLS" (spec, SECURITY). There are two genuinely different
deployments, and conflating them is the usual way mTLS ends up
decorative, so both are supported explicitly:

**1. Direct TLS termination** (uvicorn holds the certificates).
:meth:`TLSSettings.uvicorn_kwargs` produces the exact keyword arguments
``uvicorn.run()`` needs, including ``ssl_cert_reqs=ssl.CERT_REQUIRED``
when client certificates are mandatory. The client's certificate is
verified by the TLS stack itself before a single byte of HTTP is parsed
— the strongest position, and the default this module steers toward.

**2. Reverse-proxy termination** (nginx/Caddy/Traefik holds the
certificates and forwards headers). The TLS handshake happens before the
request reaches Python at all, so the app can only learn about the
client certificate from headers the proxy sets. That is trustworthy
*only* if two things hold, and :class:`ClientCertVerifier` enforces both
rather than assuming them:

  a. The connection genuinely came from a trusted proxy — checked
     against :attr:`TLSSettings.trusted_proxies`, because any client
     that can reach the app directly can otherwise just *send*
     ``X-Client-Verify: SUCCESS`` and be trusted.
  b. The proxy said verification actually succeeded — the header must
     say so; a present-but-unverified certificate is a failure, not a
     pass.

Getting (a) wrong turns mTLS into a header any attacker can set, so the
verifier refuses to trust proxy headers at all when
``trusted_proxies`` is empty. That refusal is deliberate: an empty
trusted-proxy list with proxy-mode mTLS enabled is a misconfiguration,
and it fails closed.

Optional allowlisting by certificate subject/fingerprint is supported
for the "only these three machines may talk to this server" case that a
private multi-server deployment usually wants.
"""
from __future__ import annotations

import ipaddress
import logging
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import T1APIError, T1ErrorCode

logger = logging.getLogger(__name__)

# Headers set by the common reverse proxies when they terminate mTLS.
# nginx: ssl_client_verify / ssl_client_s_dn / ssl_client_fingerprint.
HEADER_VERIFY = "x-client-verify"
HEADER_SUBJECT_DN = "x-client-subject-dn"
HEADER_FINGERPRINT = "x-client-fingerprint"
HEADER_ISSUER_DN = "x-client-issuer-dn"

_SUCCESS_VALUES = {"success", "ok", "verified", "true", "1"}


@dataclass
class TLSSettings:
    """TLS/mTLS configuration, resolved from environment variables.

    Args:
        certfile / keyfile: The server's own certificate and private key.
            Required for direct termination; unused behind a proxy.
        ca_certs: CA bundle used to verify *client* certificates.
        require_client_cert: Turns mTLS on. Without it, TLS is
            one-directional (ordinary HTTPS).
        behind_proxy: True when a reverse proxy terminates TLS and
            forwards the ``X-Client-*`` headers described above.
        trusted_proxies: Addresses/CIDRs permitted to assert client-cert
            headers. Mandatory when *behind_proxy* is set.
        allowed_subjects / allowed_fingerprints: Optional allowlists. A
            non-empty list means "only these clients", checked in
            addition to CA verification, never instead of it.
    """

    certfile: str | None = None
    keyfile: str | None = None
    ca_certs: str | None = None
    require_client_cert: bool = False
    behind_proxy: bool = False
    trusted_proxies: tuple[str, ...] = ()
    allowed_subjects: tuple[str, ...] = ()
    allowed_fingerprints: tuple[str, ...] = ()
    exempt_paths: tuple[str, ...] = ("/health",)

    @property
    def tls_enabled(self) -> bool:
        return bool(self.certfile and self.keyfile)

    @property
    def mtls_enabled(self) -> bool:
        return self.require_client_cert

    def validation_errors(self) -> list[str]:
        """Problems that make this configuration unusable or unsafe.

        Returned as a list (rather than raising on the first) so an
        operator fixing a deployment sees everything wrong at once.
        """
        problems: list[str] = []
        if self.certfile and not self.keyfile:
            problems.append("T1_TLS_CERTFILE is set but T1_TLS_KEYFILE is not.")
        if self.keyfile and not self.certfile:
            problems.append("T1_TLS_KEYFILE is set but T1_TLS_CERTFILE is not.")
        for label, path in (("T1_TLS_CERTFILE", self.certfile), ("T1_TLS_KEYFILE", self.keyfile), ("T1_TLS_CA_CERTS", self.ca_certs)):
            if path and not Path(path).exists():
                problems.append(f"{label}={path!r} does not exist.")
        if self.require_client_cert and not self.behind_proxy and not self.ca_certs:
            problems.append(
                "T1_MTLS_REQUIRE_CLIENT_CERT=1 needs T1_TLS_CA_CERTS (the CA that signs "
                "client certificates) when this process terminates TLS itself."
            )
        if self.require_client_cert and self.behind_proxy and not self.trusted_proxies:
            problems.append(
                "T1_MTLS_BEHIND_PROXY=1 requires T1_TRUSTED_PROXIES. Without it, any client "
                "able to reach this process directly could forge the X-Client-Verify header."
            )
        for fingerprint in self.allowed_fingerprints:
            if not _normalize_fingerprint(fingerprint):
                problems.append(f"Unparseable client fingerprint in allowlist: {fingerprint!r}")
        return problems

    def uvicorn_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``uvicorn.run()`` / ``uvicorn.Config``.

        Empty when TLS isn't configured, so a caller can always do
        ``uvicorn.run(app, **settings.uvicorn_kwargs())``.
        """
        if not self.tls_enabled:
            return {}
        kwargs: dict[str, Any] = {"ssl_certfile": self.certfile, "ssl_keyfile": self.keyfile}
        if self.ca_certs:
            kwargs["ssl_ca_certs"] = self.ca_certs
        if self.require_client_cert:
            kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
        return kwargs

    def public_dict(self) -> dict[str, Any]:
        """Safe for GET /config: says whether TLS/mTLS is on and how,
        never where the key material lives."""
        return {
            "tls_enabled": self.tls_enabled,
            "mtls_enabled": self.mtls_enabled,
            "mtls_mode": "proxy" if self.behind_proxy else ("direct" if self.mtls_enabled else "off"),
            "client_allowlist_configured": bool(self.allowed_subjects or self.allowed_fingerprints),
        }


def _normalize_fingerprint(value: str) -> str:
    """``AA:BB:cc`` and ``sha256/AABBCC`` both become ``aabbcc``.

    Proxies format fingerprints inconsistently; comparing normalized
    forms avoids an allowlist that silently never matches because nginx
    wrote colons and the config didn't.
    """
    cleaned = value.strip().lower()
    for prefix in ("sha256/", "sha256:", "sha1/", "sha1:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    cleaned = cleaned.replace(":", "").replace(" ", "")
    return cleaned if all(c in "0123456789abcdef" for c in cleaned) and cleaned else ""


@dataclass
class ClientCertificate:
    """What we know about the peer's certificate."""

    verified: bool
    subject_dn: str = ""
    issuer_dn: str = ""
    fingerprint: str = ""
    source: str = "none"  # "proxy" | "transport" | "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "subject_dn": self.subject_dn,
            "issuer_dn": self.issuer_dn,
            "fingerprint": self.fingerprint,
            "source": self.source,
        }


@dataclass
class ClientCertVerifier:
    """Extracts and validates the peer certificate for one request."""

    settings: TLSSettings
    _trusted_networks: list[Any] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        for entry in self.settings.trusted_proxies:
            try:
                self._trusted_networks.append(ipaddress.ip_network(entry.strip(), strict=False))
            except ValueError:
                logger.warning("t1api.mtls: ignoring unparseable trusted proxy entry")

    def is_trusted_proxy(self, peer_ip: str | None) -> bool:
        if not peer_ip or not self._trusted_networks:
            return False
        try:
            addr = ipaddress.ip_address(peer_ip.strip())
        except ValueError:
            return False
        return any(addr.version == net.version and addr in net for net in self._trusted_networks)

    def extract(
        self,
        *,
        headers: dict[str, str],
        peer_ip: str | None,
        transport_cert: dict[str, Any] | None = None,
    ) -> ClientCertificate:
        """Determine the client certificate for this request.

        *transport_cert* is the ASGI ``extensions['tls']`` peer
        certificate when uvicorn terminated TLS itself; it always wins
        over headers, since it can't be forged by the client.
        """
        if transport_cert:
            return ClientCertificate(
                verified=True,
                subject_dn=_format_dn(transport_cert.get("subject")),
                issuer_dn=_format_dn(transport_cert.get("issuer")),
                fingerprint=_normalize_fingerprint(str(transport_cert.get("fingerprint", ""))),
                source="transport",
            )
        if not self.settings.behind_proxy:
            return ClientCertificate(verified=False, source="none")
        if not self.is_trusted_proxy(peer_ip):
            # Headers from an untrusted peer are not evidence of
            # anything — treat them as absent rather than as a claim.
            logger.debug("t1api.mtls: ignoring client-cert headers from untrusted peer %s", peer_ip)
            return ClientCertificate(verified=False, source="none")
        lowered = {k.lower(): v for k, v in headers.items()}
        verify = lowered.get(HEADER_VERIFY, "").strip().lower()
        return ClientCertificate(
            verified=verify in _SUCCESS_VALUES,
            subject_dn=lowered.get(HEADER_SUBJECT_DN, ""),
            issuer_dn=lowered.get(HEADER_ISSUER_DN, ""),
            fingerprint=_normalize_fingerprint(lowered.get(HEADER_FINGERPRINT, "")),
            source="proxy",
        )

    def require(self, cert: ClientCertificate, *, path: str = "/") -> None:
        """Raise unless *cert* satisfies the configured mTLS policy."""
        if not self.settings.mtls_enabled:
            return
        if any(path == exempt or path.startswith(exempt.rstrip("/") + "/") for exempt in self.settings.exempt_paths):
            # /health stays reachable without a client certificate so
            # load balancers and uptime checks keep working; it exposes
            # nothing but liveness.
            return
        if not cert.verified:
            raise T1APIError(
                T1ErrorCode.MTLS_REQUIRED,
                "This server requires a verified client certificate (mTLS).",
                http_status=403,
            )
        if self.settings.allowed_subjects:
            if not any(_dn_matches(allowed, cert.subject_dn) for allowed in self.settings.allowed_subjects):
                raise T1APIError(
                    T1ErrorCode.MTLS_INVALID,
                    "Client certificate subject is not on this server's allowlist.",
                    details={"subject_dn": cert.subject_dn},
                    http_status=403,
                )
        if self.settings.allowed_fingerprints:
            allowed = {_normalize_fingerprint(f) for f in self.settings.allowed_fingerprints}
            if cert.fingerprint not in allowed:
                raise T1APIError(
                    T1ErrorCode.MTLS_INVALID,
                    "Client certificate fingerprint is not on this server's allowlist.",
                    http_status=403,
                )


def _dn_matches(pattern: str, subject_dn: str) -> bool:
    """Case-insensitive containment match against a subject DN.

    DN formatting varies between proxies and TLS stacks (``CN=a,O=b`` vs
    ``/CN=a/O=b``), so an exact-string allowlist is brittle in practice.
    Containment on a normalized form is the pragmatic middle ground: an
    operator writes ``CN=deploy-01`` and it matches either format.
    """
    return pattern.strip().lower() in subject_dn.strip().lower()


def _format_dn(rdns: Any) -> str:
    """Turn Python's nested-tuple DN into ``CN=x,O=y``."""
    if not rdns:
        return ""
    if isinstance(rdns, str):
        return rdns
    parts: list[str] = []
    for rdn in rdns:
        for item in rdn:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                parts.append(f"{item[0]}={item[1]}")
    return ",".join(parts)


__all__ = [
    "ClientCertVerifier",
    "ClientCertificate",
    "TLSSettings",
    "HEADER_FINGERPRINT",
    "HEADER_SUBJECT_DN",
    "HEADER_VERIFY",
]
