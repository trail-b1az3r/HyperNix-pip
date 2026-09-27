"""HyperLink on the web: the T1 server's own site, on port 37965.

While the T1 API is running it also serves a page that works like the
HyperLink app: chats with streamed replies and tools, the model picker,
the runner (including moving a model out of LM Studio), memories, and
settings. It is for the machine itself and for the owner's other devices
on the tailnet, so it listens on exactly two kinds of address:

* ``127.0.0.1``, and
* each Tailscale address this machine has (100.64.0.0/10, fd7a:115c:a1e0::/48).

Never ``0.0.0.0``: the LAN is not the owner's network, and a café's
Wi-Fi is a LAN. Tailscale addresses are looked up again every minute,
so a tailnet that comes up after the server does is picked up without
a restart.

The page and the API share an origin. :func:`build_web_app` serves the
page's three files and hands every other path to the T1 API app itself,
so the page calls ``/hyperlink/...`` with the same authentication any
client uses: keyless where the server's trusted-network mode allows it,
otherwise a HyperLink pairing code or a key, which the page keeps in the
browser's storage for this origin only.
"""
from __future__ import annotations

import ipaddress
import logging
import shutil
import socket
import subprocess
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_WEB_PORT",
    "WEB_FILES",
    "WebListener",
    "build_web_app",
    "status_lines",
    "tailnet_addresses",
]

DEFAULT_WEB_PORT = 37965
WEB_DIR = Path(__file__).resolve().parent.parent / "hyperlink" / "web"
#: The only files served. A fixed map, not a directory listing, so no
#: request path ever becomes a file path.
WEB_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/web/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/web/app.css": ("app.css", "text/css; charset=utf-8"),
    "/web/icon.svg": ("icon.svg", "image/svg+xml"),
}
_TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
#: The page's own headers. No inline script anywhere, so the policy can
#: forbid it, and the page may not be framed.
_HEADERS = [
    (b"content-security-policy",
     b"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
     b"connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"),
    (b"x-frame-options", b"DENY"),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"cache-control", b"no-cache"),
]


def _is_tailnet(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return ip in (_TAILNET_V4 if ip.version == 4 else _TAILNET_V6)


#: Tailscale's own addresses inside the tailnet (MagicDNS). Only ever used
#: to ask the kernel which local address routes to them; nothing is sent.
_TAILNET_PROBES = (("100.100.100.100", socket.AF_INET), ("fd7a:115c:a1e0::53", socket.AF_INET6))


def _tailscale_binary() -> str:
    """The tailscale command, wherever it is, or ``""``.

    PATH alone misses it under a systemd unit, whose PATH is minimal,
    and on a Mac with Tailscale from the App Store, which keeps it inside
    the app. The same places the keyless check looks in.
    """
    import os

    from ..system.nettrust import _TAILSCALE_PATHS

    return shutil.which("tailscale") or next(
        (path for path in _TAILSCALE_PATHS if os.path.exists(path)), "")


def _routed_addresses() -> list[str]:
    """This machine's addresses on the routes to Tailscale's own addresses.

    A connected UDP socket sends nothing: the kernel only picks the
    local address it would use. With Tailscale up that is this machine's
    tailnet address, found with neither the command nor psutil; without
    it the route is the LAN's, which the caller's range check drops.
    """
    found = []
    for probe, family in _TAILNET_PROBES:
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as sock:
                sock.connect((probe, 53))
                found.append(str(sock.getsockname()[0]).split("%", 1)[0])
        except OSError:
            continue
    return found


def _interface_addresses() -> list[str]:
    try:
        import psutil
    except ImportError:
        return []
    found = []
    try:
        for addresses in psutil.net_if_addrs().values():
            for entry in addresses:
                found.append(str(entry.address).split("%", 1)[0])
    except Exception:  # noqa: BLE001 - interfaces unreadable: no tailnet from here
        logger.debug("hyperlink_web: interfaces unreadable", exc_info=True)
    return found


def tailnet_addresses() -> list[str]:
    """This machine's Tailscale addresses, or ``[]`` without Tailscale.

    Asked of ``tailscale ip`` when the command can be found, then of the
    kernel's routes to Tailscale's own addresses, then of the network
    interfaces (with psutil, when it is installed). Before 0.72.6.post2
    only PATH and psutil were tried, and the t1api extra does not bring
    psutil, so a server started by systemd or on a Mac listened on
    127.0.0.1 alone and a phone on the tailnet was refused. Only
    addresses inside Tailscale's ranges are returned, whatever any
    source says.
    """
    found: list[str] = []
    tailscale = _tailscale_binary()
    if tailscale:
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv
                [tailscale, "ip"], capture_output=True, text=True, timeout=3,
            )
            found = [line.strip() for line in done.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            found = []
    if not any(_is_tailnet(address) for address in found):
        found = _routed_addresses()
    if not any(_is_tailnet(address) for address in found):
        found = _interface_addresses()
    unique: list[str] = []
    for address in found:
        if _is_tailnet(address) and address not in unique:
            unique.append(address)
    return unique


def build_web_app(api_app: Any):
    """An ASGI app: the page's files, and the T1 API for everything else."""

    cache: dict[str, bytes] = {}

    def body_of(name: str) -> bytes:
        if name not in cache:
            cache[name] = (WEB_DIR / name).read_bytes()
        return cache[name]

    async def app(scope, receive, send):
        if scope["type"] == "http" and scope.get("method") in ("GET", "HEAD"):
            entry = WEB_FILES.get(scope.get("path", ""))
            if entry is not None:
                name, content_type = entry
                body = body_of(name)
                await send({
                    "type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", content_type.encode()),
                                (b"content-length", str(len(body)).encode()), *_HEADERS],
                })
                await send({"type": "http.response.body",
                            "body": b"" if scope["method"] == "HEAD" else body})
                return
        await api_app(scope, receive, send)

    return app


def _bind(address: str, port: int) -> socket.socket | None:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    try:
        sock.bind((address, port))
    except OSError as exc:
        sock.close()
        logger.warning("hyperlink_web: cannot listen on %s:%s: %s", address, port, exc)
        return None
    sock.listen(128)
    sock.setblocking(False)
    return sock


class WebListener:
    """Serves :func:`build_web_app` on loopback and the tailnet.

    A thread of its own running a second uvicorn server over the same
    app, so there is one server state (sessions, runner, stores) and
    two front doors. :meth:`start` and :meth:`stop` are what the T1
    app's lifespan calls.
    """

    RECHECK_SECONDS = 60.0

    def __init__(self, api_app: Any, port: int = DEFAULT_WEB_PORT) -> None:
        self.api_app = api_app
        self.port = int(port)
        self.addresses: list[str] = []
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._watch: threading.Thread | None = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()

    def urls(self) -> list[str]:
        return [f"http://[{a}]:{self.port}" if ":" in a else f"http://{a}:{self.port}"
                for a in self.addresses]

    def start(self) -> None:
        if self.port <= 0:
            return
        self._stopping.clear()
        self._serve(["127.0.0.1", *tailnet_addresses()])
        self._watch = threading.Thread(target=self._watch_tailnet, name="hyperlink-web-watch",
                                       daemon=True)
        self._watch.start()

    def _serve(self, addresses: list[str]) -> None:
        import uvicorn

        sockets = []
        bound = []
        for address in addresses:
            sock = _bind(address, self.port)
            if sock is not None:
                sockets.append(sock)
                bound.append(address)
        if not sockets:
            logger.warning("hyperlink_web: nothing to listen on; the web site is off. Is "
                           "something else using port %s? Set T1_WEB_PORT to move it.", self.port)
            return
        config = uvicorn.Config(build_web_app(self.api_app), log_level="warning",
                                lifespan="off", access_log=False)
        server = uvicorn.Server(config)
        # The main server owns signals; this one is stopped by `stop`.
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        thread = threading.Thread(target=server.run, kwargs={"sockets": sockets},
                                  name="hyperlink-web", daemon=True)
        with self._lock:
            self._server, self._thread, self.addresses = server, thread, bound
        thread.start()
        self._announce(addresses)

    def _announce(self, wanted: list[str]) -> None:
        """Say where the site is, in the server's log, where people look.

        An INFO line never reached ``hypernix-t1 logs``, so a site that
        was up said nothing and one missing its tailnet said nothing
        either. This is printed like the first-start key is.
        """
        import sys

        lines = [f"  HyperLink on the web: {', '.join(self.urls())}"]
        if len(wanted) == 1:
            lines.append("    No Tailscale address on this machine, so only this machine can "
                         "open it. It is picked up within a minute of Tailscale coming up.")
        missed = [a for a in wanted if a not in self.addresses]
        if missed:
            lines.append(f"    Could not listen on {', '.join(missed)} (see the warning above).")
        print("\n".join(lines), file=sys.stderr, flush=True)

    def _halt(self) -> None:
        with self._lock:
            server, thread = self._server, self._thread
            self._server, self._thread = None, None
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10)

    def _watch_tailnet(self) -> None:
        """Rebind when this machine's Tailscale addresses change."""
        while not self._stopping.wait(self.RECHECK_SECONDS):
            wanted = ["127.0.0.1", *tailnet_addresses()]
            if wanted != self.addresses:
                logger.info("hyperlink_web: tailnet addresses changed; rebinding")
                self._halt()
                if not self._stopping.is_set():
                    self._serve(wanted)

    def stop(self) -> None:
        self._stopping.set()
        self._halt()


def _answers(url: str, timeout: float = 2.0) -> bool:
    """Whether *url* serves this site's page. No proxy: these are local."""
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as page:
            return b"<title>HyperLink</title>" in page.read(4096)
    except OSError:
        return False


def status_lines(enabled: bool, port: int) -> list[str]:
    """What ``hypernix-t1 status`` says about the site: each address, and
    whether it answers there. Asked from outside the server, so it holds
    whether the server is healthy, wedged or reading another port."""
    if not enabled or port <= 0:
        return ["web       off (T1_WEB_ENABLED=0)"]
    lines = []
    for address in ["127.0.0.1", *tailnet_addresses()]:
        url = f"http://[{address}]:{port}/" if ":" in address else f"http://{address}:{port}/"
        lines.append(f"web       {url}  {'answering' if _answers(url) else 'NOT answering'}")
    if len(lines) == 1:
        lines.append("          no Tailscale address here, so only this machine can open it")
    return lines
