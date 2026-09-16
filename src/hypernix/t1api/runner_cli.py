"""``hypernix-t1 runner`` — load, unload and inspect the served model.

Also reachable as ``python -m hypernix.t1api.runner_cli``.

Why this talks HTTP to a server on the same machine
---------------------------------------------------
Loading the model here instead would start a *second* llama.cpp, which
takes the VRAM the server's own copy is using, and the failure lands on
the one that was working rather than on the one being started. The
runner is a process the server owns; this is a client to it, exactly as
HyperLink is.

That also means one answer to "what is loaded" for everybody: the
phone, this command, and the server all read the same registry, so they
cannot disagree about which model is answering.

The key
-------
Changing what a shared server runs is gated server-side, so a request
needs a credential. It is read, in order, from ``--key``, then
``T1_ADMIN_KEY``, then the admin key in the server's own ``.env`` —
which is the common case: somebody at the keyboard of the machine that
is running the thing. A server in trusted-network mode with partial
admin on may accept the request with no key at all, and that works here
too; no key is not an error until the server says it is.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

__all__ = ["main", "cli_main"]

DEFAULT_URL = "http://127.0.0.1:8000"

_EPILOG = """\
Examples:

  hypernix-t1 runner status
  hypernix-t1 runner plan qwen3-8b
  hypernix-t1 runner load qwen3-8b --gpu-layers 24
  hypernix-t1 runner unload

`plan` says where a model's layers would go and changes nothing.
Loading evicts whatever people are currently talking to, so seeing the
consequence first is not a nicety.
"""


def _config_dir() -> Path:
    configured = os.environ.get("T1_CONFIG_DIR", "")
    return Path(configured) if configured else Path.home() / ".hypernix" / "t1api"


def _key_from_env_file() -> str:
    """The admin key out of the server's own ``.env``, or ``""``.

    Read rather than required: the caller is usually sitting at the
    machine running the server, and making them paste a key they already
    have on disk is the kind of friction that gets worked around with a
    shell alias that stores it somewhere worse.
    """
    path = _config_dir() / ".env"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() in ("T1_ADMIN_KEY", "T1_BOOTSTRAP_ADMIN_KEY"):
            return value.strip().strip('"').strip("'")
    return ""


def _resolve_key(explicit: str) -> str:
    return explicit or os.environ.get("T1_ADMIN_KEY", "") or _key_from_env_file()


def _base_url(explicit: str) -> str:
    url = explicit or os.environ.get("T1_URL", "") or DEFAULT_URL
    return url.rstrip("/")


def _request(
    method: str, url: str, key: str, payload: dict[str, Any] | None = None
) -> tuple[int, Any]:
    """``(status, parsed_body)``. Never raises for an HTTP error status.

    A 403 from this endpoint is *information* — it names what would be
    needed — so it is returned to be printed rather than turned into a
    traceback.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        body = error.read() or b"{}"
        try:
            return error.code, json.loads(body)
        except ValueError:
            return error.code, {"error": {"message": body.decode(errors="replace")}}
    except urllib.error.URLError as error:
        raise SystemExit(
            f"Could not reach the T1 API at {url}: {error.reason}\n"
            f"Is it running? `hypernix-t1 status` says."
        ) from error


def _fail(status: int, body: Any) -> int:
    """Print a refusal the way the server phrased it."""
    error = (body or {}).get("error") if isinstance(body, dict) else None
    message = (error or {}).get("message") if isinstance(error, dict) else None
    print(f"Refused ({status}): {message or body}", file=sys.stderr)
    details = (error or {}).get("details") if isinstance(error, dict) else None
    if isinstance(details, dict):
        remedy = details.get("remedy")
        if remedy:
            print(f"\n{remedy}", file=sys.stderr)
        loadable = details.get("loadable")
        if loadable:
            print("\nModels this server can load:", file=sys.stderr)
            for name in loadable:
                print(f"  {name}", file=sys.stderr)
    return 1


def _describe_placement(placement: dict[str, Any]) -> str:
    total = int(placement.get("total_layers") or 0)
    gpu = int(placement.get("gpu_layers") or 0)
    if total <= 0:
        return str(placement.get("reason") or "")
    if placement.get("fully_offloaded"):
        where = f"all {total} layers on the GPU"
    elif gpu <= 0:
        where = f"all {total} layers on the CPU"
    else:
        where = f"{gpu} of {total} on the GPU, {total - gpu} on the CPU"
    reason = placement.get("reason") or ""
    return f"{where}\n  {reason}" if reason else where


def _print_status(body: dict[str, Any]) -> None:
    if not body.get("loaded"):
        print("Nothing loaded.")
        backends = body.get("backends") or []
        if backends:
            print(f"Backends available: {', '.join(backends)}")
        return
    model = body.get("model") or {}
    print(f"Loaded: {model.get('model_id', '?')}")
    if model.get("base_url"):
        print(f"Serving: {model['base_url']}")
    placement = model.get("placement") or {}
    if placement:
        print(f"Layers: {_describe_placement(placement)}")
    if model.get("context_length"):
        print(f"Context: {model['context_length']}")
    if model.get("uptime_seconds"):
        print(f"Up: {model['uptime_seconds']}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypernix-t1 runner",
        description="Load, unload and inspect the model this server serves.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="", help=f"server base URL (default {DEFAULT_URL})")
    parser.add_argument("--key", default="", help="admin key (default: T1_ADMIN_KEY, then the server's .env)")
    parser.add_argument("--json", action="store_true", help="print the raw response")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="what is loaded, and where its layers are")

    for name, help_text in (
        ("plan", "where a model's layers would go — changes nothing"),
        ("load", "load a model, replacing whatever is running"),
    ):
        action = sub.add_parser(name, help=help_text)
        action.add_argument("model_id", help="the model to act on")
        action.add_argument("--gpu-layers", type=int, default=None,
                            help="layers on the GPU (default: work it out)")
        action.add_argument("--total-layers", type=int, default=None,
                            help="the model's layer count, if you know it")
        action.add_argument("--context-length", type=int, default=None,
                            help="context window (default: the model's own)")
        action.add_argument("--backend", default="auto",
                            help="auto, cuda, vulkan, cpu, hnx-cuda, hnx-cpu")

    sub.add_parser("unload", help="stop serving; unloading nothing is a success")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "status"

    url = _base_url(args.url)
    key = _resolve_key(args.key)

    if command == "status":
        status, body = _request("GET", f"{url}/runner/status", key)
    elif command == "unload":
        status, body = _request("POST", f"{url}/runner/unload", key, {})
    elif command in ("plan", "load"):
        payload = {
            "model_id": args.model_id,
            "gpu_layers": args.gpu_layers,
            "backend": args.backend,
            "context_length": args.context_length,
            "total_layers": args.total_layers,
        }
        status, body = _request("POST", f"{url}/runner/{command}", key, payload)
    else:  # pragma: no cover - argparse rejects anything else
        parser.error(f"unknown command {command}")

    if status >= 400:
        return _fail(status, body)

    if args.json:
        print(json.dumps(body, indent=2))
        return 0

    if command == "plan":
        print(f"{body.get('model_id', args.model_id)}")
        print(f"  file: {body.get('path', '?')}")
        placement = body.get("placement") or {}
        if placement:
            print(f"  {_describe_placement(placement)}")
        print("\nNothing has changed — this was a plan.")
    elif command == "unload":
        print("Unloaded." if body.get("was_running") else "Nothing was running.")
    else:
        _print_status(body)
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    sys.exit(main())
