"""net — Advanced distributed network manager and Tailscale integration."""
from __future__ import annotations

import argparse
import ipaddress
import json
import re
import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path.home() / ".hypernix"
NET_CONFIG_FILE = CONFIG_DIR / "net.json"
_REMOTE_FILE_RE = re.compile(r"^[A-Za-z0-9_./-]+$")


def load_config() -> dict:
    if not NET_CONFIG_FILE.exists():
        return {"ports": [], "storage_dir": str(Path.home() / "hypernix_storage")}
    try:
        with open(NET_CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"ports": [], "storage_dir": str(Path.home() / "hypernix_storage")}


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(NET_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def run_cmd(cmd: list[str], capture: bool = True) -> str:
    """Run a fixed-argument subprocess command and return stdout.

    Callers must pass argv lists; no command is interpreted by a shell.
    """
    try:
        res = subprocess.run(cmd, check=True, capture_output=capture, text=True)
        return res.stdout.strip() if capture else ""
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        if capture and isinstance(exc, subprocess.CalledProcessError):
            return (exc.stdout or "").strip()
        return ""


def _peer_ip(value: str) -> str:
    """Accept only a Tailscale IPv4/IPv6 address returned by the daemon."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(f"invalid peer address: {value!r}") from exc
    if not address.is_private:
        raise ValueError(f"peer address is not private: {value!r}")
    return value


def _remote_file(value: str) -> str:
    """Validate paths before placing them in an intentional SSH command."""
    if (
        not value
        or not _REMOTE_FILE_RE.fullmatch(value)
        or value.startswith("/")
        or ".." in Path(value).parts
    ):
        raise ValueError("remote file must be a relative path without shell metacharacters")
    return value


def get_tailscale_peers() -> list[str]:
    """Return validated Tailscale peer IPs."""
    out = run_cmd(["tailscale", "status", "--json"])
    if not out:
        return []
    try:
        data = json.loads(out)
        peers = []
        for peer_info in data.get("Peer", {}).values():
            if peer_info.get("Online") and peer_info.get("TailscaleIPs"):
                try:
                    peers.append(_peer_ip(peer_info["TailscaleIPs"][0]))
                except (KeyError, ValueError):
                    continue
        return peers
    except (TypeError, ValueError):
        return []


def cmd_config(args: argparse.Namespace) -> None:
    print("Current Configuration:")
    print(json.dumps(load_config(), indent=2))
    print("\nTo update, edit:", NET_CONFIG_FILE)


def cmd_auto_setup(args: argparse.Namespace) -> None:
    print("[net] Bringing Tailscale up...")
    run_cmd(["tailscale", "up"], capture=False)
    ssh_key = Path.home() / ".ssh" / "id_rsa"
    if not ssh_key.exists():
        print("[net] Generating SSH keys...")
        run_cmd(["ssh-keygen", "-t", "rsa", "-b", "4096", "-f", str(ssh_key), "-N", ""], capture=False)
    print("[net] Auto-setup complete.")


def cmd_m_setup(args: argparse.Namespace) -> None:
    print("Manual Setup Instructions:")
    print("1. Install Tailscale using its official documentation.")
    print("2. Run: sudo tailscale up")
    print("3. Generate SSH keys: ssh-keygen -t rsa")
    print("4. Copy keys to peers: ssh-copy-id <peer-ip>")


def cmd_connect(args: argparse.Namespace) -> None:
    peer = _peer_ip(args.ip)
    print(f"[net] Connecting to {peer}...")
    subprocess.run(["ssh", "--", peer], check=False)


def cmd_status(args: argparse.Namespace) -> None:
    print("[net] Tailscale Status:")
    subprocess.run(["tailscale", "status"], check=False)


def cmd_m_ip(args: argparse.Namespace) -> None:
    ip = run_cmd(["tailscale", "ip", "-4"])
    print(ip or "Tailscale not running or not installed.")


def cmd_a_il(args: argparse.Namespace) -> None:
    print("[net] Auto-connecting to online HyperNix peers...")
    peers = get_tailscale_peers()
    if not peers:
        print("[net] No peers found online.")
        return
    for peer in peers:
        print(f" - Found peer: {peer}")
        # ssh-keyscan writes to stdout; do not pass a shell redirection as an argv item.
        result = subprocess.run(["ssh-keyscan", "-H", peer], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout:
            known_hosts = Path.home() / ".ssh" / "known_hosts"
            known_hosts.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with known_hosts.open("a", encoding="utf-8") as handle:
                handle.write(result.stdout)
    print("[net] Peer discovery and linkage complete.")


def cmd_multi_a_port(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.port not in cfg["ports"]:
        cfg["ports"].append(args.port)
        save_config(cfg)
    print(f"[net] Registered active port {args.port}. Current ports: {cfg['ports']}")


def cmd_ex_port(args: argparse.Namespace) -> None:
    print(f"[net] To expose port {args.port} securely across the tailnet, run:")
    print(f"  sudo tailscale serve --bg {args.port}")
    if args.apply:
        run_cmd(["sudo", "tailscale", "serve", "--bg", str(args.port)], capture=False)


def cmd_s_storage(args: argparse.Namespace) -> None:
    cfg = load_config()
    storage = Path(cfg["storage_dir"]).expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    peers = get_tailscale_peers()
    if not peers:
        print("[net] No peers available for storage sync.")
        return
    print(f"[net] Syncing {storage} across {len(peers)} peers...")
    for peer in peers:
        print(f" -> Syncing with {peer}...")
        run_cmd(["rsync", "-avz", f"{storage}{'/' if storage else ''}", f"{peer}:{storage}/"], capture=False)


def cmd_onef_all(args: argparse.Namespace) -> None:
    print("[net] Configuring node as onef-all (centralized storage).")
    print(f"[net] Tailscale IP: {run_cmd(['tailscale', 'ip', '-4'])}")
    print("[net] Use this IP in your client nodes' configs.")


def cmd_tail_acheck(args: argparse.Namespace) -> None:
    remote_file = _remote_file(args.file)
    for peer in get_tailscale_peers():
        print(f"[peer {peer}] Checking for {remote_file}...")
        # The filename is validated and passed as one remote command argument.
        check = f"test -f -- {remote_file!r}"
        res = subprocess.run(["ssh", "--", peer, "sh", "-c", check], capture_output=True, check=False)
        if res.returncode == 0:
            print(f"  -> File {remote_file} exists.")
            if args.r:
                print(f"  -> Running {remote_file}...")
                subprocess.Popen(["ssh", "--", peer, "sh", "-c", f"nohup python3 -- {remote_file!r} >/dev/null 2>&1 &"], start_new_session=True)
        else:
            print(f"  -> File {remote_file} NOT found.")


def cmd_tail_stop(args: argparse.Namespace) -> None:
    remote_file = _remote_file(args.file)
    for peer in get_tailscale_peers():
        print(f"[peer {peer}] Stopping {remote_file}...")
        subprocess.run(["ssh", "--", peer, "pkill", "-f", "--", remote_file], capture_output=True, check=False)


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hnx net", description="Distributed network manager.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    subparsers.add_parser("config")
    subparsers.add_parser("auto-setup")
    subparsers.add_parser("m-setup")
    p_conn = subparsers.add_parser("connect"); p_conn.add_argument("ip")
    subparsers.add_parser("status"); subparsers.add_parser("m-ip"); subparsers.add_parser("a-il")
    p_mport = subparsers.add_parser("mutli-a-port"); p_mport.add_argument("port", type=int)
    p_export = subparsers.add_parser("ex-port"); p_export.add_argument("port", type=int); p_export.add_argument("--apply", action="store_true")
    subparsers.add_parser("s-storage"); subparsers.add_parser("onef-all")
    p_tail = subparsers.add_parser("tail"); p_tail.add_argument("action", choices=["acheck", "stop"]); p_tail.add_argument("file"); p_tail.add_argument("-r", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    handlers = {"config": cmd_config, "auto-setup": cmd_auto_setup, "m-setup": cmd_m_setup, "connect": cmd_connect, "status": cmd_status, "m-ip": cmd_m_ip, "a-il": cmd_a_il, "mutli-a-port": cmd_multi_a_port, "ex-port": cmd_ex_port, "s-storage": cmd_s_storage, "onef-all": cmd_onef_all}
    if args.cmd == "tail":
        (cmd_tail_acheck if args.action == "acheck" else cmd_tail_stop)(args)
    else:
        handlers[args.cmd](args)
    return 0


if __name__ == "__main__":
    cli_main()
