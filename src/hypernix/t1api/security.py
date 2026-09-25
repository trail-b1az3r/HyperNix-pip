"""t1api.security — shared guardrails used by the server registry and the
module system.

Two narrow, focused helpers rather than a general "security module" —
each maps to a specific spec requirement:

* :func:`validate_remote_address` — "Validate remote URLs before
  fetching" / "Prevent SSRF." Only ``http``/``https`` schemes are ever
  accepted; the cloud-metadata SSRF target (``169.254.169.254``) is
  always rejected; anything else classified as loopback/private/
  link-local requires the caller to explicitly opt in via
  ``allow_private=True`` — which is exactly the knob a local/Tailscale
  deployment (spec's own "Support local-only deployments" / "Example
  Tailscale deployment") needs to flip, since those addresses are
  *supposed* to be private there. The function doesn't decide policy on
  its own; it forces the caller to state which mode they're in.

* :func:`sanitize_module_path` — "Sanitize file paths" / "Prevent
  directory traversal." Resolves a caller-supplied relative path against
  a fixed base directory and rejects anything that would escape it (``..``
  segments, absolute-path overrides, symlink escapes).

Both raise :class:`T1APIError` with a stable code
(``SSRF_BLOCKED`` / ``PATH_TRAVERSAL_REJECTED``) rather than a bare
``ValueError``, so callers get the same error envelope as everywhere else
in the T1 API.
"""
from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

from .errors import T1APIError, T1ErrorCode

_ALLOWED_SCHEMES = {"http", "https"}
_ALWAYS_BLOCKED_HOSTS = {
    "169.254.169.254",  # AWS/GCP/Azure cloud-metadata SSRF classic
    "metadata.google.internal",
}


def _is_private_or_local(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # Not a literal IP — resolve it so "localhost" / a DNS name that
        # points at a private range doesn't slip past a string check.
        try:
            resolved = socket.getaddrinfo(host, None)
        except socket.gaierror:
            return False  # can't resolve — let the actual connection attempt fail later
        return any(_is_private_or_local(info[4][0]) for info in resolved)
    return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved


def validate_remote_address(address: str, *, allow_private: bool = False) -> str:
    """Validate a server/module-source URL before it's stored or fetched.

    Args:
        address: The URL to validate.
        allow_private: Set True for local/Tailscale deployments where
            private/loopback addresses are expected and desired. Defaults
            to False (reject) since the safe default for a
            general-purpose "register a remote server" call is to assume
            it's meant to be reachable on the public internet.

    Returns:
        The address, unchanged, if it passes.

    Raises:
        T1APIError: ``SSRF_BLOCKED`` if the address is unsafe.
    """
    parsed = urlparse(address)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise T1APIError(
            T1ErrorCode.SSRF_BLOCKED,
            f"Only http/https URLs are allowed, got scheme {parsed.scheme!r}.",
            details={"address": address},
            http_status=400,
        )
    host = parsed.hostname
    if not host:
        raise T1APIError(
            T1ErrorCode.SSRF_BLOCKED,
            "URL has no resolvable host.",
            details={"address": address},
            http_status=400,
        )
    if host.lower() in _ALWAYS_BLOCKED_HOSTS:
        raise T1APIError(
            T1ErrorCode.SSRF_BLOCKED,
            f"'{host}' is a cloud-metadata endpoint and is never allowed.",
            details={"address": address},
            http_status=400,
        )
    if not allow_private and _is_private_or_local(host):
        raise T1APIError(
            T1ErrorCode.SSRF_BLOCKED,
            f"'{host}' resolves to a private/loopback/link-local address. "
            "Pass allow_private=True for local/Tailscale deployments where this is expected.",
            details={"address": address},
            http_status=400,
        )
    return address


def _validate_path_component(component: str, *, original_path: str) -> str:
    if component in {"", ".", ".."} or "/" in component or "\\" in component:
        raise T1APIError(
            T1ErrorCode.PATH_TRAVERSAL_REJECTED,
            f"'{original_path}' contains an invalid path component.",
            details={"relative_path": original_path, "component": component},
            http_status=400,
        )
    return component


def sanitize_module_path(relative_path: str, base_dir: Path) -> Path:
    """Resolve *relative_path* under *base_dir* and reject any attempt to
    escape it (``..`` traversal, an absolute path override, or a symlink
    that resolves outside the base).

    Returns the resolved, safe absolute path. Does not check existence —
    callers create/read the file themselves after this returns.
    """
    requested = Path(relative_path)
    if requested.is_absolute() or requested.anchor:
        raise T1APIError(
            T1ErrorCode.PATH_TRAVERSAL_REJECTED,
            f"'{relative_path}' resolves outside the allowed module storage directory.",
            details={"relative_path": relative_path},
            http_status=400,
        )

    safe_parts = [_validate_path_component(part, original_path=relative_path) for part in requested.parts]
    if not safe_parts:
        raise T1APIError(
            T1ErrorCode.PATH_TRAVERSAL_REJECTED,
            f"'{relative_path}' resolves outside the allowed module storage directory.",
            details={"relative_path": relative_path},
            http_status=400,
        )

    base_resolved = base_dir.resolve()
    candidate = (base_resolved.joinpath(*safe_parts)).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise T1APIError(
            T1ErrorCode.PATH_TRAVERSAL_REJECTED,
            f"'{relative_path}' resolves outside the allowed module storage directory.",
            details={"relative_path": relative_path},
            http_status=400,
        ) from exc
    return candidate


__all__ = ["validate_remote_address", "sanitize_module_path"]
