"""``hnx claude`` — Claude Code, answered by a model HyperNix is serving.

::

    hnx claude                                   # this machine's T1 server, its loaded model
    hnx claude --model qwen3-4b-q4_k_m           # a registered model
    hnx claude --server http://100.64.0.7:8000 --key T1-...
    hnx claude --print-env                       # the variables, to export yourself
    hnx claude -- -p "explain this repo"         # anything after -- goes to claude

It points Claude Code at the T1 server's ``/v1/messages`` (the Anthropic
Messages API, :mod:`hypernix.t1api.routers.anthropic`) and starts it:

* ``ANTHROPIC_BASE_URL`` — the server;
* ``ANTHROPIC_AUTH_TOKEN`` — your T1 key, sent as ``Authorization:
  Bearer``. With no key, ``t1-keyless``, which the server reads as "no
  credential": it works where the server's trusted-network mode lets a
  keyless request in, and is refused everywhere else;
* ``ANTHROPIC_MODEL`` and the Opus/Sonnet/Haiku/subagent defaults — all
  the one model, because a local server has one model loaded;
* ``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1``, and a ten-minute
  ``API_TIMEOUT_MS``: a local model is slower than the API.

Claude Code itself is installed separately:
``npm install -g @anthropic-ai/claude-code``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

__all__ = ["KEYLESS", "cli_main", "claude_env"]

#: The token that means "I have no credential" (see the router).
KEYLESS = "t1-keyless"
_MODEL_VARS = ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
               "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")


def _default_server() -> str:
    for name in ("HYPERNIX_T1_URL", "T1_URL", "T1_SERVER_URL"):
        if os.environ.get(name):
            return os.environ[name].rstrip("/")
    return f"http://127.0.0.1:{os.environ.get('T1_HYPERLINK_PORT', '8000')}"


def _default_key() -> str:
    for name in ("HYPERNIX_T1_KEY", "T1_API_KEY", "T1_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    return ""


def available_models(server: str, key: str, *, timeout: float = 5.0) -> list[dict]:
    """``GET /v1/models`` on *server*, or a RuntimeError saying why not."""
    req = urllib.request.Request(server.rstrip("/") + "/v1/models",
                                 headers={"Authorization": f"Bearer {key or KEYLESS}",
                                          "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - the server the user named
            return list(json.loads(resp.read()).get("data") or [])
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read()).get("error", {}).get("message", "")
        except Exception:  # noqa: BLE001
            message = ""
        raise RuntimeError(f"{server} refused ({exc.code}){': ' + message if message else ''}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"no T1 server at {server} ({getattr(exc, 'reason', exc)}). Start one with "
                           "`hypernix-t1 serve`, or pass --server.") from exc


def claude_env(server: str, key: str, model: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment Claude Code runs in."""
    env = dict(os.environ if base is None else base)
    env.pop("ANTHROPIC_API_KEY", None)        # it would win over the token and go to the wrong place
    env["ANTHROPIC_BASE_URL"] = server.rstrip("/")
    env["ANTHROPIC_AUTH_TOKEN"] = key or KEYLESS
    for name in _MODEL_VARS:
        env[name] = model
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    env.setdefault("API_TIMEOUT_MS", "600000")
    return env


def _pick(models: list[dict], wanted: str) -> str:
    ids = [m.get("id", "") for m in models]
    if wanted:
        if wanted not in ids:
            raise RuntimeError(f"{wanted} is not on the server; it has: {', '.join(ids) or 'nothing'}")
        return wanted
    loaded = [m["id"] for m in models if str(m.get("display_name", "")).endswith("(loaded)")]
    if loaded:
        return loaded[0]
    if len(ids) == 1:
        return ids[0]
    raise RuntimeError("the server has no model loaded" + (f"; pick one with --model: {', '.join(ids)}"
                                                           if ids else "; load one with POST /runner/load"))


def cli_main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in args:
        at = args.index("--")
        args, passthrough = args[:at], args[at + 1:]
    p = argparse.ArgumentParser(prog="hnx claude", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=None, help="T1 server URL (default: this machine's, port 8000)")
    p.add_argument("--key", default=None, help="T1 key (default: $T1_API_KEY); none: keyless")
    p.add_argument("--model", default=None, help="model id (default: the one the server has loaded)")
    p.add_argument("--print-env", action="store_true", help="print the variables instead of starting claude")
    ns = p.parse_args(args)
    server = (ns.server or _default_server()).rstrip("/")
    key = ns.key if ns.key is not None else _default_key()
    try:
        model = _pick(available_models(server, key), ns.model or "")
    except RuntimeError as exc:
        print(f"hnx claude: {exc}", file=sys.stderr)
        return 1
    env = claude_env(server, key, model)
    if ns.print_env:
        for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", *_MODEL_VARS,
                     "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "API_TIMEOUT_MS"):
            print(f"export {name}={json.dumps(env[name])}")
        print("unset ANTHROPIC_API_KEY")
        return 0
    claude = shutil.which("claude")
    if not claude:
        print("hnx claude: Claude Code is not installed. Install it with "
              "`npm install -g @anthropic-ai/claude-code`, or use --print-env.", file=sys.stderr)
        return 1
    print(f"hnx claude: {model} on {server}", file=sys.stderr)
    return subprocess.call([claude, *passthrough], env=env)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
