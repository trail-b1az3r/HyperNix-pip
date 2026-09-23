"""waiter.servops — what ``-u``, ``-S``, ``-Y`` and ``-T`` do (0.72.6).

Kept out of ``cli.py`` so each can be tested against a real app through
the SDK, without the argument handling in the way.
"""
from __future__ import annotations

import ipaddress
import os
import stat
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Finding",
    "security_check",
    "update_plan",
    "run_update",
    "local_version",
    "control_allowed",
    "format_info",
]


# ---------------------------------------------------------------------------
# -u / -ud — bring this client up to the server's version
# ---------------------------------------------------------------------------


def _version(text: str):
    try:
        from packaging.version import Version

        return Version(text)
    except Exception:  # noqa: BLE001 - packaging missing, or a local build string
        return None


def local_version() -> str:
    """The hypernix this client runs."""
    import hypernix

    return getattr(hypernix, "__version__", "unknown")


def update_plan(local: str, server: str, *, exact: bool) -> tuple[str | None, str]:
    """``(pip requirement or None, what and why)``.

    ``-u`` wants a client at least as new as the server; ``-ud`` wants
    exactly the server's version, down as well as up.
    """
    if not server or server == "unknown":
        return None, "the server did not say which hypernix it runs"
    ours, theirs = _version(local), _version(server)
    if exact:
        if local == server:
            return None, f"already on {server}, the server's version"
        return f"hypernix=={server}", f"{local} -> {server} (exactly the server's)"
    if ours is not None and theirs is not None:
        if ours >= theirs:
            return None, f"{local} is already at or above the server's {server}"
    elif local == server:
        return None, f"already on {server}"
    return f"hypernix>={server}", f"{local} -> {server} or newer"


def run_update(requirement: str, *, dry_run: bool = False) -> int:
    command = [sys.executable, "-m", "pip", "install", "--upgrade", requirement]
    print("  " + " ".join(command))
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode  # noqa: S603 - fixed argv, no shell


# ---------------------------------------------------------------------------
# -S — check this client, then the server
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    level: str      # high | medium | low | ok
    where: str      # client | server
    message: str
    fix: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "where": self.where, "message": self.message, "fix": self.fix}


def _is_private_host(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return host.endswith((".local", ".ts.net", ".internal", ".lan"))
    return ip.is_private or ip.is_loopback


def _client_findings(cfg: Any, store_path: Path, locked: bool, local_version: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        import cryptography  # noqa: F401
    except ImportError:
        findings.append(Finding("medium", "client", "the 'cryptography' package is not installed, "
                                "so v2.1 keys (-E) and a locked config (-e) are unavailable",
                                'pip install "hypernix[security]"'))
    if store_path.exists():
        if os.name != "nt":
            mode = stat.S_IMODE(store_path.stat().st_mode)
            if mode & 0o077:
                findings.append(Finding("high", "client", f"{store_path} can be read by other users "
                                        f"(mode {oct(mode)})", f"chmod 600 {store_path}"))
        key = getattr(cfg, "key", None) or ""
        if key and not locked:
            if key.startswith("T2CK_"):
                findings.append(Finding("medium", "client", "the v2.1 kit is stored unlocked; anyone "
                                        "who reads the config can make this key's daily keys",
                                        "waiter serv -e (lock the config with a password)"))
            elif key.startswith(("T1_", "T2_", "T2S_", "T2P_")):
                findings.append(Finding("medium", "client", "the key is stored as itself, and works "
                                        "forever for whoever reads the config",
                                        "waiter serv -E (seal it as a v2.1 key) and -e (lock)"))
        if locked:
            findings.append(Finding("ok", "client", "the config is locked with a password"))
    findings.append(Finding("ok", "client", f"hypernix {local_version}"))
    return findings


def security_check(client: Any, cfg: Any, *, store_path: Path, locked: bool) -> list[Finding]:
    """Look for problems in this client, then in the server it points at.

    Only the server's public endpoints are used, the way any client sees
    them: it is a check of your own setup, not a scan.
    """
    import hypernix

    local_version = getattr(hypernix, "__version__", "unknown")
    findings = _client_findings(cfg, store_path, locked, local_version)

    base = urllib.parse.urlsplit(client.base_url)
    host = base.hostname or ""
    if base.scheme == "http" and not _is_private_host(host):
        findings.append(Finding("high", "server", f"{client.base_url} is plain HTTP on a public "
                                "address: keys and replies cross the network unencrypted",
                                "use https://, a Tailscale address, or a private network"))
    elif base.scheme == "http":
        findings.append(Finding("low", "server", f"{client.base_url} is plain HTTP on a private "
                                "address; fine on a trusted network, not beyond it"))

    try:
        response = client.transport.request("GET", "/status")
        status, headers = response.body, {k.lower(): v for k, v in response.headers.items()}
    except Exception as exc:  # noqa: BLE001 - an unreachable server is a finding, not a crash
        findings.append(Finding("high", "server", f"could not reach {client.base_url}: {exc}"))
        return findings

    for warning in status.get("production_warnings") or []:
        findings.append(Finding("medium", "server", f"server configuration: {warning}"))
    if status.get("environment") == "production" and not status.get("tls_enabled"):
        if base.scheme != "https":
            findings.append(Finding("high", "server", "a production server without TLS"))
    for header in ("x-content-type-options", "x-frame-options"):
        if header not in headers:
            findings.append(Finding("low", "server", f"responses lack the {header} header"))

    server_version = str(status.get("hypernix_version") or "")
    plan, why = update_plan(local_version, server_version, exact=False)
    if plan:
        findings.append(Finding("medium", "client", f"this client is older than the server ({why})",
                                "waiter serv -u"))

    try:
        anonymous = client.transport.request("GET", "/usage/current", auth=False)
        if anonymous.status == 200:
            level = "medium" if not _is_private_host(host) else "low"
            findings.append(Finding(level, "server", "the server answers without a key from this "
                                    "address (trusted-network mode)"))
    except Exception:  # noqa: BLE001 - a refusal is the expected answer
        findings.append(Finding("ok", "server", "the server refuses requests without a key"))

    try:
        info = client.transport.request("GET", "/server/info").body
        key = getattr(cfg, "key", None) or ""
        if info.get("features", {}).get("t2c_keys") and key.startswith(("T1_", "T2_")):
            findings.append(Finding("low", "client", "the server supports v2.1 keys and this "
                                    "client still sends its key as itself", "waiter serv -E"))
    except Exception:  # noqa: BLE001 - an older server has no /server/info
        pass
    return findings


# ---------------------------------------------------------------------------
# -Y — the server's public card
# ---------------------------------------------------------------------------


def format_info(info: dict[str, Any]) -> list[str]:
    t1 = info.get("t1_version") or {}
    features = info.get("features") or {}
    rows = [
        ("name", info.get("name") or "—"),
        ("description", info.get("description") or "—"),
        ("owner", info.get("owner") or "—"),
        ("url", info.get("url") or "—"),
        ("public", "yes" if info.get("public") else "no"),
        ("environment", info.get("environment") or "—"),
        ("hypernix", info.get("hypernix_version") or "—"),
        ("t1 api", t1.get("short") or t1.get("long") or "—"),
        ("server id", info.get("server_id") or "—"),
        ("v2.1 keys", "yes" if features.get("t2c_keys") else "no"),
        ("conceal", f"yes, level {features.get('conceal_min_access_level', 3)}+, "
                    f"{features.get('retention_hours', 36)}h retention"
         if features.get("conceal") else "no"),
        ("accounts", "open" if features.get("accounts") else "no"),
    ]
    width = max(len(label) for label, _ in rows)
    return [f"  {label.ljust(width)}  {value}" for label, value in rows]


# ---------------------------------------------------------------------------
# -T — who may open the control screen
# ---------------------------------------------------------------------------


def control_allowed(validated: dict[str, Any]) -> tuple[bool, str]:
    """A verified administrator holding a level-9 T2 (or v2.1) key.

    The server still checks every action on its own; this decides only
    whether the control screen opens.
    """
    if not validated.get("is_admin"):
        return False, "this key is not an administrator key"
    family = validated.get("key_family") or "T1"
    if family not in ("T2", "T2C"):
        return False, f"control needs a T2 or v2.1 key; this is a {family} key"
    level = validated.get("access_level")
    if level != 9:
        return False, f"control needs access level 9; this key is level {level}"
    return True, "verified: administrator, level 9"
