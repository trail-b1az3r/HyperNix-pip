"""Where the T1 API on this machine is, for the command-line clients.

``hypernix-t1 runner`` used to try ``http://127.0.0.1:8000`` and nothing
else. ``hypernix-t1 start`` binds wherever ``T1_PORT`` says, and the
shell wrapper even exports ``T1_PORT`` from the server's ``.env`` before
running the client, so a server on 8001 answered to its own start
command and was "not running" to the runner command next to it:

    Could not reach the T1 API at http://127.0.0.1:8000/runner/plan:
    [Errno 111] Connection refused

The order here is the order a person would expect: a URL they gave,
``T1_URL``, then ``T1_HOST``/``T1_PORT`` from the environment, then the
server's own ``.env``, then 8000. And when nothing answers there, the
ports next to it are asked whether they are a T1 API, because the
commonest reason for a server not being where the config says is that
it was started with ``--port`` by hand, and saying "found it on 8001"
beats "is it running?" when it plainly is.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from hypernix.security.safeurl import urlopen as safe_urlopen

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "SCAN_PORTS",
    "config_dir",
    "configured_url",
    "discover",
    "is_t1_api",
]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: Asked, in order, when the configured address does not answer. Small
#: on purpose: this runs only after a refusal, on loopback, and each
#: miss costs a refused connection, not a timeout.
SCAN_PORTS = tuple(range(8000, 8011)) + (8080, 8443, 9000)


def config_dir() -> Path:
    configured = os.environ.get("T1_CONFIG_DIR", "")
    return Path(configured) if configured else Path.home() / ".hypernix" / "t1api"


def _env_file_values() -> dict[str, str]:
    path = config_dir() / ".env"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def _client_host(host: str) -> str:
    # A server bound to every interface is reachable on loopback, and
    # 0.0.0.0 is not an address a client can connect to.
    if host in ("", "0.0.0.0", "::", "[::]"):  # nosec B104 - compares against or binds 0.0.0.0 only where the caller chose LAN access
        return DEFAULT_HOST
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def configured_url(explicit: str = "") -> tuple[str, str]:
    """``(base_url, where_it_came_from)`` for the T1 API on this machine."""
    if explicit:
        return explicit.rstrip("/"), "--url"
    if os.environ.get("T1_URL"):
        return os.environ["T1_URL"].rstrip("/"), "T1_URL"
    port = os.environ.get("T1_PORT", "")
    host = os.environ.get("T1_HOST", "")
    source = "T1_PORT" if port else ""
    if not port:
        values = _env_file_values()
        port = values.get("T1_PORT", "")
        host = host or values.get("T1_HOST", "")
        if port:
            source = f"T1_PORT in {config_dir() / '.env'}"
    if not port.isdigit():
        port, source = str(DEFAULT_PORT), "the default port"
    return f"http://{_client_host(host)}:{port}", source


def is_t1_api(base_url: str, *, timeout: float = 0.6) -> bool:
    """Does *base_url* answer ``/health`` the way the T1 API does?

    The shape is checked, not just the status, so another service on a
    neighbouring port (a Jupyter, a dev server) is not mistaken for it.
    """
    try:
        with safe_urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as response:
            body = json.loads(response.read(4096) or b"{}")
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return isinstance(body, dict) and body.get("status") == "ok" and "request_id" in body


def discover(exclude: str = "", *, host: str = DEFAULT_HOST) -> str:
    """The first T1 API answering on loopback, or ``""``."""
    for port in SCAN_PORTS:
        url = f"http://{host}:{port}"
        if url == exclude.rstrip("/"):
            continue
        if is_t1_api(url):
            return url
    return ""
