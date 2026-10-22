"""netops — what ``hnx net`` did, as functions that return data.

``hnx net`` printed. The T1 API's ``/net`` routes (0.72.8a3) need the same
operations as answers, so they live here and both use them; ``hnx net``
now points people to the API.

Every command is an argv list, never a shell string, with the program
resolved on PATH first; a peer name is passed after ``--`` so one that
starts with ``-`` cannot become an option; a file name for a peer's shell
is quoted. Nothing here asks for privileges: a ``tailscale serve`` that
needs an operator says so and how to grant it.
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

__all__ = [
    "CONFIG_FILE",
    "NetError",
    "Run",
    "check_file",
    "expose",
    "load_config",
    "peers",
    "ping",
    "run_file",
    "save_config",
    "setup",
    "setup_steps",
    "status",
    "stop_file",
    "sync_storage",
    "tailnet_ip",
    "trust_peers",
]

CONFIG_FILE = Path.home() / ".hypernix" / "net.json"
TIMEOUT = 30.0


class NetError(RuntimeError):
    """A net operation that could not be done, and why."""


@dataclass
class Run:
    """One command's outcome."""

    argv: list[str]
    code: int
    out: str = ""
    err: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0

    def to_dict(self) -> dict[str, Any]:
        return {"command": " ".join(shlex.quote(a) for a in self.argv), "code": self.code,
                "out": self.out[-4000:], "err": self.err[-2000:]}


Runner = Callable[[Sequence[str], float], Run]


def _run(argv: Sequence[str], timeout: float = TIMEOUT) -> Run:
    argv = list(argv)
    exe = shutil.which(argv[0])
    if not exe:
        return Run(argv, 127, err=f"{argv[0]} is not installed")
    try:
        done = subprocess.run([exe, *argv[1:]], capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return Run(argv, 124, out=str(exc.stdout or ""), err=f"{argv[0]} did not finish within {timeout:.0f}s")
    except OSError as exc:
        return Run(argv, 126, err=str(exc))
    return Run(argv, done.returncode, done.stdout or "", done.stderr or "")


#: What every operation runs commands with, unless given another.
RUN: Runner = _run


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _defaults() -> dict[str, Any]:
    return {"ports": [], "storage_dir": str(Path.home() / "hypernix_storage"), "hub": False}


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_FILE
    config = _defaults()
    if path.is_file():
        try:
            config.update(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            pass
    config["ports"] = sorted({int(p) for p in config.get("ports") or [] if str(p).isdigit()})
    return config


def save_config(config: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_FILE
    clean = _defaults()
    clean.update({k: v for k, v in config.items() if k in clean})
    ports = sorted({int(p) for p in clean["ports"]})
    if any(not 0 < p < 65536 for p in ports):
        raise NetError("a port is 1-65535")
    clean["ports"] = ports
    clean["storage_dir"] = str(Path(str(clean["storage_dir"])).expanduser())
    clean["hub"] = bool(clean["hub"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return clean


# ---------------------------------------------------------------------------
# Tailscale
# ---------------------------------------------------------------------------

def _node(data: dict[str, Any]) -> dict[str, Any]:
    return {"hostname": data.get("HostName", ""), "dns_name": str(data.get("DNSName", "")).rstrip("."),
            "ips": list(data.get("TailscaleIPs") or []), "os": data.get("OS", ""),
            "online": bool(data.get("Online", False)), "last_seen": data.get("LastSeen", ""),
            "exit_node": bool(data.get("ExitNode", False))}


def status(run: Runner | None = None) -> dict[str, Any]:
    """This node and its peers, from ``tailscale status --json``."""
    run = run or RUN
    result = run(["tailscale", "status", "--json"], TIMEOUT)
    if result.code == 127:
        return {"installed": False, "running": False, "self": None, "peers": [],
                "hint": "Install Tailscale: https://tailscale.com/download"}
    try:
        data = json.loads(result.out or "{}")
    except ValueError:
        data = {}
    state = data.get("BackendState", "")
    return {"installed": True, "running": state == "Running", "backend_state": state or "unknown",
            "self": _node(data["Self"]) if data.get("Self") else None,
            "peers": sorted((_node(p) for p in (data.get("Peer") or {}).values()), key=lambda n: n["hostname"]),
            "hint": "" if state == "Running" else (result.err.strip() or "Run `tailscale up` (or POST /net/setup).")}


def peers(run: Runner | None = None, *, online_only: bool = True) -> list[dict[str, Any]]:
    run = run or RUN
    found = status(run)["peers"]
    return [p for p in found if p["online"] or not online_only]


def _addresses(run: Runner) -> list[str]:
    return [p["ips"][0] for p in peers(run) if p["ips"]]


def tailnet_ip(run: Runner | None = None) -> dict[str, str]:
    run = run or RUN
    v4, v6 = run(["tailscale", "ip", "-4"], TIMEOUT), run(["tailscale", "ip", "-6"], TIMEOUT)
    return {"ipv4": v4.out.strip().splitlines()[0] if v4.ok and v4.out.strip() else "",
            "ipv6": v6.out.strip().splitlines()[0] if v6.ok and v6.out.strip() else ""}


def ping(peer: str, run: Runner | None = None) -> Run:
    """``tailscale ping`` once: is the peer reachable, and over what path."""
    run = run or RUN
    return run(["tailscale", "ping", "--c", "1", "--timeout", "5s", "--", peer], 15.0)


def expose(port: int, *, apply: bool = False, run: Runner | None = None) -> dict[str, Any]:
    """Serve a local port across the tailnet with ``tailscale serve --bg``."""
    run = run or RUN
    if not 0 < int(port) < 65536:
        raise NetError("a port is 1-65535")
    argv = ["tailscale", "serve", "--bg", str(int(port))]
    if not apply:
        return {"applied": False, "command": " ".join(argv)}
    result = run(argv, TIMEOUT)
    out = {"applied": result.ok, **result.to_dict()}
    if not result.ok and ("access denied" in result.err.lower() or "permission" in result.err.lower()):
        out["hint"] = "Let this user configure Tailscale once: sudo tailscale set --operator=$USER"
    return out


def setup_steps() -> list[str]:
    return ["Install Tailscale: https://tailscale.com/download (Linux: curl -fsSL https://tailscale.com/install.sh | sh)",
            "Join the tailnet: sudo tailscale up (or POST /net/setup, which returns the login link)",
            "Let this user run tailscale serve: sudo tailscale set --operator=$USER",
            "An SSH key for the peers: ssh-keygen -t ed25519",
            "Copy it to each peer: ssh-copy-id <peer>, then POST /net/peers/trust"]


def setup(run: Runner | None = None, *, key_path: Path | None = None) -> dict[str, Any]:
    """``tailscale up`` (returning its login link if it prints one) and an SSH key."""
    run = run or RUN
    up = run(["tailscale", "up", "--timeout", "20s"], 30.0)
    link = next(
        (
            w
            for w in (up.out + " " + up.err).split()
            if (lambda p: p.scheme == "https" and p.hostname == "login.tailscale.com")(urlparse(w))
        ),
        "",
    )
    key_path = key_path or Path.home() / ".ssh" / "id_ed25519"
    made = None
    if not key_path.exists() and not key_path.with_name("id_rsa").exists():
        key_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        made = run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key_path), "-q"], TIMEOUT)
    return {"tailscale": up.to_dict(), "login_url": link,
            "ssh_key": str(key_path) if (made is None or made.ok) else "",
            "ssh_key_created": bool(made and made.ok), "next": setup_steps()[2:]}


# ---------------------------------------------------------------------------
# Peers over SSH
# ---------------------------------------------------------------------------

def _ssh(peer: str, *words: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--", peer,
            " ".join(shlex.quote(w) for w in words)]


def trust_peers(run: Runner | None = None, *, known_hosts: Path | None = None) -> list[dict[str, Any]]:
    """Add each online peer's SSH host keys to known_hosts (``a-il``)."""
    run = run or RUN
    known_hosts = known_hosts or Path.home() / ".ssh" / "known_hosts"
    rows = []
    for peer in _addresses(run):
        scan = run(["ssh-keyscan", "-H", "-T", "5", "--", peer], 20.0)
        added = 0
        if scan.ok and scan.out.strip():
            known_hosts.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with known_hosts.open("a", encoding="utf-8") as fh:
                lines = [ln for ln in scan.out.splitlines() if ln.strip() and not ln.startswith("#")]
                fh.write("\n".join(lines) + "\n")
                added = len(lines)
        rows.append({"peer": peer, "keys_added": added, "error": "" if scan.ok else scan.err.strip()})
    return rows


def sync_storage(run: Runner | None = None, *, config: dict[str, Any] | None = None,
                 only: Sequence[str] | None = None, dry_run: bool = False) -> list[dict[str, Any]]:
    """rsync the storage folder to each online peer (``s-storage``)."""
    run = run or RUN
    config = config or load_config()
    folder = Path(config["storage_dir"]).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for peer in _addresses(run):
        if only and peer not in only:
            continue
        argv = ["rsync", "-az", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10"]
        if dry_run:
            argv.append("--dry-run")
        argv += ["--", f"{folder}/", f"{peer}:{folder}/"]
        rows.append({"peer": peer, **run(argv, 3600.0).to_dict()})
    return rows


def check_file(path: str, run: Runner | None = None) -> list[dict[str, Any]]:
    """Whether *path* exists on each online peer (``tail acheck``)."""
    run = run or RUN
    return [{"peer": peer, "exists": run(_ssh(peer, "test", "-f", path), 20.0).ok} for peer in _addresses(run)]


def run_file(path: str, run: Runner | None = None) -> list[dict[str, Any]]:
    """Start ``python3 PATH`` in the background on each peer that has it (``tail acheck -r``)."""
    run = run or RUN
    rows = []
    for row in check_file(path, run):
        if not row["exists"]:
            rows.append({**row, "started": False})
            continue
        started = run(["ssh", "-o", "BatchMode=yes", "--", row["peer"],
                       f"nohup python3 {shlex.quote(path)} > /dev/null 2>&1 &"], 20.0)
        rows.append({**row, "started": started.ok, "error": started.err.strip()})
    return rows


def stop_file(path: str, run: Runner | None = None) -> list[dict[str, Any]]:
    """Stop a script started with :func:`run_file` on each peer (``tail stop``)."""
    run = run or RUN
    return [{"peer": peer, "stopped": run(_ssh(peer, "pkill", "-f", path), 20.0).code in (0, 1)}
            for peer in _addresses(run)]
