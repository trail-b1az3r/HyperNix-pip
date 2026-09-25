"""net — Advanced distributed network manager and Tailscale integration.

Commands:
  config         - Configure local net settings.
  auto-setup     - Auto-configure Tailscale and SSH keys.
  m-setup        - Manual setup printouts.
  connect        - SSH connect/ping to a specific peer.
  status         - Show Tailscale status.
  m-ip           - Get local Tailscale IP.
  a-il           - Auto-connect to other hyperNix devices on the tailnet.
  mutli-a-port   - Configure multiple active ports.
  ex-port        - Expose/forward a port.
  s-storage      - Sync/share storage across net devices (rsync).
  onef-all       - Designate one node for centralized storage/compute.
  tail acheck    - Auto-check for a Python file on tailnet peers (use -r to run).
  tail stop      - Stop the script on tailnet peers.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path.home() / ".hypernix"
NET_CONFIG_FILE = CONFIG_DIR / "net.json"

def load_config() -> dict:
    if not NET_CONFIG_FILE.exists():
        return {"ports": [], "storage_dir": str(Path.home() / "hypernix_storage")}
    try:
        with open(NET_CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {"ports": [], "storage_dir": str(Path.home() / "hypernix_storage")}

def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(NET_CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def run_cmd(cmd: list[str], capture: bool = True) -> str:
    """Run a subprocess command and return stdout."""
    try:
        res = subprocess.run(cmd, check=True, capture_output=capture, text=True)
        return res.stdout.strip() if capture else ""
    except subprocess.CalledProcessError as e:
        if capture:
            return e.stdout.strip() if e.stdout else ""
        return ""
    except FileNotFoundError:
        return ""

def get_tailscale_peers() -> list[str]:
    """Return a list of Tailscale peer IPs."""
    out = run_cmd(["tailscale", "status", "--json"])
    if not out:
        return []
    try:
        data = json.loads(out)
        peers = []
        for _peer_id, peer_info in data.get("Peer", {}).items():
            if peer_info.get("Online") and peer_info.get("TailscaleIPs"):
                peers.append(peer_info["TailscaleIPs"][0])
        return peers
    except Exception:
        return []

# --- Command Handlers ---

def cmd_config(args: argparse.Namespace) -> None:
    cfg = load_config()
    print("Current Configuration:")
    print(json.dumps(cfg, indent=2))
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
    print("1. Install Tailscale: curl -fsSL https://tailscale.com/install.sh | sh")
    print("2. Run: sudo tailscale up")
    print("3. Generate SSH keys: ssh-keygen -t rsa")
    print("4. Copy keys to peers: ssh-copy-id <peer-ip>")

def cmd_connect(args: argparse.Namespace) -> None:
    print(f"[net] Connecting to {args.ip}...")
    subprocess.run(["ssh", args.ip])

def cmd_status(args: argparse.Namespace) -> None:
    print("[net] Tailscale Status:")
    subprocess.run(["tailscale", "status"])

def cmd_m_ip(args: argparse.Namespace) -> None:
    ip = run_cmd(["tailscale", "ip", "-4"])
    if ip:
        print(ip)
    else:
        print("Tailscale not running or not installed.")

def cmd_a_il(args: argparse.Namespace) -> None:
    print("[net] Auto-connecting to online HyperNix peers...")
    peers = get_tailscale_peers()
    if not peers:
        print("[net] No peers found online.")
        return
    for peer in peers:
        print(f" - Found peer: {peer}")
        # Automatically accept SSH keys for peers
        run_cmd(["ssh-keyscan", "-H", peer, ">>", str(Path.home() / ".ssh" / "known_hosts")], capture=True)
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
    storage = cfg["storage_dir"]
    Path(storage).mkdir(parents=True, exist_ok=True)
    peers = get_tailscale_peers()
    
    if not peers:
        print("[net] No peers available for storage sync.")
        return
        
    print(f"[net] Syncing {storage} across {len(peers)} peers...")
    for peer in peers:
        print(f" -> Syncing with {peer}...")
        # Sync to peer
        run_cmd(["rsync", "-avz", f"{storage}/", f"{peer}:{storage}/"], capture=False)

def cmd_onef_all(args: argparse.Namespace) -> None:
    print("[net] Configuring node as onef-all (centralized storage).")
    print(f"[net] Tailscale IP: {run_cmd(['tailscale', 'ip', '-4'])}")
    print("[net] Use this IP in your client nodes' configs.")

def cmd_tail_acheck(args: argparse.Namespace) -> None:
    peers = get_tailscale_peers()
    if not peers:
        print("No peers found.")
        return
        
    for peer in peers:
        print(f"[peer {peer}] Checking for {args.file}...")
        # Check if file exists
        res = subprocess.run(["ssh", peer, f"test -f {args.file}"], capture_output=True)
        if res.returncode == 0:
            print(f"  -> File {args.file} exists.")
            if args.r:
                print(f"  -> Running {args.file}...")
                subprocess.Popen(["ssh", peer, f"nohup python3 {args.file} > /dev/null 2>&1 &"])
        else:
            print(f"  -> File {args.file} NOT found.")

def cmd_tail_stop(args: argparse.Namespace) -> None:
    peers = get_tailscale_peers()
    for peer in peers:
        print(f"[peer {peer}] Stopping {args.file}...")
        subprocess.run(["ssh", peer, f"pkill -f {args.file}"], capture_output=False)


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hnx net", description="Distributed network manager.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("config", help="View/edit local net configuration.")
    subparsers.add_parser("auto-setup", help="Auto-configure Tailscale & SSH.")
    subparsers.add_parser("m-setup", help="Show manual setup instructions.")
    
    p_conn = subparsers.add_parser("connect", help="Connect to a peer.")
    p_conn.add_argument("ip", help="Peer IP address")

    subparsers.add_parser("status", help="Show tailscale status.")
    subparsers.add_parser("m-ip", help="Get local tailscale IP.")
    subparsers.add_parser("a-il", help="Auto-connect to all peers.")

    p_mport = subparsers.add_parser("mutli-a-port", help="Add an active port.")
    p_mport.add_argument("port", type=int)

    p_export = subparsers.add_parser("ex-port", help="Expose a port.")
    p_export.add_argument("port", type=int)
    p_export.add_argument("--apply", action="store_true", help="Actually run tailscale serve")

    subparsers.add_parser("s-storage", help="Sync storage across tailnet.")
    subparsers.add_parser("onef-all", help="Designate node as centralized storage.")

    # Tail subcommand (tail acheck / tail stop)
    p_tail = subparsers.add_parser("tail", help="Tailscale distributed execution.")
    p_tail.add_argument("action", choices=["acheck", "stop"], help="Action to perform")
    p_tail.add_argument("file", help="Python file path")
    p_tail.add_argument("-r", action="store_true", help="Run the file if found (only for acheck)")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.cmd == "config":
        cmd_config(args)
    elif args.cmd == "auto-setup":
        cmd_auto_setup(args)
    elif args.cmd == "m-setup":
        cmd_m_setup(args)
    elif args.cmd == "connect":
        cmd_connect(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "m-ip":
        cmd_m_ip(args)
    elif args.cmd == "a-il":
        cmd_a_il(args)
    elif args.cmd == "mutli-a-port":
        cmd_multi_a_port(args)
    elif args.cmd == "ex-port":
        cmd_ex_port(args)
    elif args.cmd == "s-storage":
        cmd_s_storage(args)
    elif args.cmd == "onef-all":
        cmd_onef_all(args)
    elif args.cmd == "tail":
        if args.action == "acheck":
            cmd_tail_acheck(args)
        elif args.action == "stop":
            cmd_tail_stop(args)

    return 0

if __name__ == "__main__":
    cli_main()
