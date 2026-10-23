"""``hnx-t1 setup`` — the T1 interactive installer, b1 (0.72.8a4).

    hnx-t1 setup                       # ask, show the result, write it, start it
    hnx-t1 setup --yes                 # every default, no questions
    hnx-t1 setup --answers ans.json    # scripted: answers by name, defaults for the rest
    hnx-t1 setup --dry-run             # print the configuration, write nothing

The updated installer. ``install-t1.sh`` still does a first install from
nothing (a venv, the package, then the questions), but it only exists in
a checkout; this runs from a pip install too, and asks about what the
server can do now: HyperLink on the web, Claude Code, remote MCP servers
(Hugging Face's), code sandboxes, the public counter and remote runs.

Run on a server that is already set up, it starts from that server's
answers. The token secret is kept -- a new one would invalidate every key
minted against the old -- the file it replaces is kept beside it as
``.env.bak-<time>``, and settings it does not ask about are carried over.

Every setting it writes is a real ``T1_*`` setting (the tests check them
against :mod:`hypernix.t1api.config`).
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["INSTALLER", "Answers", "Asker", "build_env", "cli_main", "read_env"]

#: Which installer this is. install-t1.sh is the one before.
INSTALLER = "b1"

LIMITS = {
    "relaxed": '[{"name":"per-key","capacity":600,"refill_per_second":10,"applies_to":["key"]},'
               '{"name":"per-ip","capacity":1200,"refill_per_second":20,"applies_to":["ip"]}]',
    "standard": '[{"name":"per-key","capacity":120,"refill_per_second":2,"applies_to":["key"]},'
                '{"name":"per-ip","capacity":240,"refill_per_second":4,"applies_to":["ip"]}]',
    "strict": '[{"name":"per-key","capacity":30,"refill_per_second":0.5,"applies_to":["key"]},'
              '{"name":"per-ip","capacity":60,"refill_per_second":1,"applies_to":["ip"]}]',
    "off": "",
}
REACH = {
    "local": "this machine only (127.0.0.1)",
    "network": "this machine, the LAN and the tailnet (0.0.0.0)",
    "tailnet": "this machine's Tailscale address only",
}
MODELS = {
    "runner": "HyperNix's own runner (llama.cpp on this machine)",
    "lmstudio": "LM Studio",
    "both": "both: the runner, and LM Studio when it is running",
}


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------

def _colour() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None and os.environ.get("TERM") != "dumb"


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _colour() else text


#: Where the questions' chrome goes. stderr for --dry-run, so stdout is the file.
_OUT = sys.stdout


def head(text: str) -> None:
    print("\n" + _c("1;36", text), file=_OUT)


def dim(text: str) -> None:
    print(_c("2", "     " + text), file=_OUT)


def ok(text: str) -> None:
    print(_c("32", "  ✓ ") + text)


def warn(text: str) -> None:
    print(_c("33", "  ! ") + text, file=sys.stderr)


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------

class SetupError(RuntimeError):
    """Setup cannot go on; the message says why and what to do."""


@dataclass
class Asker:
    """Questions, answered by a person, by an answers file, or by default."""

    interactive: bool = True
    preset: dict[str, Any] = field(default_factory=dict)
    read: Callable[[str], str] = input
    asked: list[str] = field(default_factory=list)

    def _given(self, key: str):
        return self.preset.get(key)

    def text(self, key: str, prompt: str, default: str = "", *, check: Callable[[str], str] | None = None) -> str:
        self.asked.append(key)
        given = self._given(key)
        if given is not None:
            value = str(given)
            problem = check(value) if check else ""
            if problem:
                raise SetupError(f"{_refused()}: answers file, {key}={value!r}: {problem}")
            return value
        if not self.interactive:
            return default
        while True:
            shown = f" {_c('2', f'[{default}]')}" if default else ""
            reply = self.read(f"{_c('1;35', '  ?')} {prompt}{shown} ").strip()
            value = reply or default
            problem = check(value) if check else ""
            if not problem:
                return value
            print(_c("31", f"    {_refused()}: {problem}"), file=_OUT)

    def yes_no(self, key: str, prompt: str, default: bool) -> bool:
        self.asked.append(key)
        given = self._given(key)
        if given is not None:
            if isinstance(given, bool):
                return given
            return str(given).strip().lower() in ("1", "y", "yes", "true", "on")
        if not self.interactive:
            return default
        hint = "Y/n" if default else "y/N"
        while True:
            reply = self.read(f"{_c('1;35', '  ?')} {prompt} {_c('2', f'[{hint}]')} ").strip().lower()
            if not reply:
                return default
            if reply in ("y", "yes"):
                return True
            if reply in ("n", "no"):
                return False
            print(_c("31", f"    {_refused()}: y or n"), file=_OUT)

    def choice(self, key: str, prompt: str, options: dict[str, str], default: str) -> str:
        self.asked.append(key)
        names = list(options)
        given = self._given(key)
        if given is not None:
            if str(given) not in options:
                raise SetupError(f"{_refused()}: answers file, {key}={given!r}: one of {', '.join(names)}")
            return str(given)
        if not self.interactive:
            return default
        print(f"{_c('1;35', '  ?')} {prompt}", file=_OUT)
        for number, name in enumerate(names, 1):
            mark = _c("36", "›") if name == default else " "
            print(f"    {mark} {number}. {options[name]}", file=_OUT)
        while True:
            reply = self.read(f"    {_c('2', f'[{names.index(default) + 1}]')} ").strip().lower()
            if not reply:
                return default
            if reply.isdigit() and 1 <= int(reply) <= len(names):
                return names[int(reply) - 1]
            if reply in options:
                return reply
            print(_c("31", f"    {_refused()}: a number from 1 to {len(names)}"), file=_OUT)


def _refused() -> str:
    from ..system.errorcatalogue import SETUP_ANSWER_REFUSED

    return SETUP_ANSWER_REFUSED.code


def _port(value: str) -> str:
    return "" if value.isdigit() and 1 <= int(value) <= 65535 else "a port is a number from 1 to 65535"


def _url(value: str) -> str:
    return "" if not value or value.startswith(("http://", "https://")) else "starts with http:// or https://"


# ---------------------------------------------------------------------------
# The answers, and the configuration they make
# ---------------------------------------------------------------------------

@dataclass
class Answers:
    name: str = ""
    reach: str = "local"
    port: int = 8000
    public_url: str = ""
    public: bool = False
    trusted: bool = False
    partial_admin: bool = False
    limits: str = "standard"
    models: str = "both"
    lmstudio_url: str = "http://127.0.0.1:1234"
    default_model: str = ""
    hyperlink: bool = True
    web: bool = True
    claude_code: bool = True
    claude_model: str = ""
    mcp: bool = True
    mcp_hf: bool = False
    code_sandbox: bool = False
    net_remote_run: bool = False
    public_counter: bool = True
    autostart: bool = False
    start: bool = True


def read_env(path: Path) -> dict[str, str]:
    """``KEY=value`` lines of a .env, quotes taken off. Never executed."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().rstrip("\r")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _on(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def defaults_from(old: dict[str, str]) -> Answers:
    """A server's present configuration, as the answers that would make it."""
    a = Answers(name=old.get("T1_SERVER_NAME") or socket.gethostname() or "hypernix")
    host = old.get("T1_HOST", "127.0.0.1")
    a.reach = "local" if host in ("127.0.0.1", "localhost", "") else "network" if host == "0.0.0.0" else "tailnet"
    a.port = int(old.get("T1_PORT", "8000") or 8000)
    a.public_url = old.get("T1_HYPERLINK_PUBLIC_URL", "") or old.get("T1_SERVER_URL", "")
    a.public = _on(old.get("T1_PUBLIC"), False)
    a.trusted = _on(old.get("T1_TRUSTED_NETWORK"), False)
    a.partial_admin = _on(old.get("T1_TRUSTED_NETWORK_PARTIAL_ADMIN"), False)
    rules = old.get("T1_RATE_LIMIT_RULES", None)
    if rules is not None:
        a.limits = next((k for k, v in LIMITS.items() if v and v == rules), "off" if not _on(
            old.get("T1_RATE_LIMIT_ENABLED"), True) else "standard")
    lms = _on(old.get("T1_LMSTUDIO_ENABLED"), True)
    a.models = "both" if lms else "runner"
    a.lmstudio_url = old.get("T1_LMSTUDIO_URL", a.lmstudio_url) or a.lmstudio_url
    # Unset: the package's default model. Empty: none. Anything else: that one.
    a.default_model = "none" if old.get("T1_DEFAULT_MODEL", None) == "" else old.get("T1_DEFAULT_MODEL", "")
    a.hyperlink = _on(old.get("T1_HYPERLINK_ENABLED"), True)
    a.web = _on(old.get("T1_WEB_ENABLED"), True)
    a.claude_code = _on(old.get("T1_ANTHROPIC_ENABLED"), True)
    a.claude_model = old.get("T1_CLAUDE_CODE_MODEL", "")
    a.mcp = _on(old.get("T1_MCP_ENABLED"), True)
    a.mcp_hf = _on(old.get("T1_MCP_HF"), False)
    a.code_sandbox = _on(old.get("T1_CODE_SANDBOX"), False)
    a.net_remote_run = _on(old.get("T1_NET_REMOTE_RUN"), False)
    a.public_counter = _on(old.get("T1_PUBLIC_COUNTER"), True)
    return a


def _tailnet_ip() -> str:
    try:
        from .hyperlink_web import tailnet_addresses

        found = [a for a in tailnet_addresses() if "." in a]
        return found[0] if found else ""
    except Exception:  # noqa: BLE001 - no tailnet is a normal answer
        return ""


def ask(asker: Asker, base: Answers) -> Answers:
    a = Answers(**{**base.__dict__})

    head("1 · The server")
    a.name = asker.text("name", "What should this server be called?", base.name)
    a.reach = asker.choice("reach", "Who can reach it?", REACH, base.reach)
    a.port = int(asker.text("port", "Port", str(base.port), check=_port))
    dim("A public address is one the internet can reach. Leave it blank if this")
    dim("server is for you and your tailnet.")
    a.public_url = asker.text("public_url", "Public URL clients use (blank: none)", base.public_url, check=_url)
    a.public = bool(a.public_url) and not _private_url(a.public_url) or (base.public and bool(a.public_url))

    head("2 · Access")
    dim("Keyless access lets this machine, the LAN and your tailnet in without a key;")
    dim("a public address never gets it.")
    a.trusted = asker.yes_no("trusted", "Allow keyless access from trusted networks?", base.trusted)
    a.partial_admin = a.trusted and asker.yes_no(
        "partial_admin", "Also let trusted connections use some admin routes?", base.partial_admin)
    a.limits = asker.choice("limits", "Request limits", {
        "relaxed": "relaxed — 600 burst, 10/s (a machine you trust)",
        "standard": "standard — 120 burst, 2/s (recommended)",
        "strict": "strict — 30 burst, 0.5/s (shared or public)",
        "off": "off — no limits (not on anything reachable)",
    }, "strict" if a.public and base.limits == "standard" else base.limits)

    head("3 · Models")
    a.models = asker.choice("models", "Where do models run?", MODELS, base.models)
    if a.models in ("lmstudio", "both"):
        a.lmstudio_url = asker.text("lmstudio_url", "LM Studio address", base.lmstudio_url, check=_url)
    dim("The model the runner loads when none is named: a name from ~/.hypernix/models")
    dim("or a .gguf path. Enter keeps HyperNix's default; `none` turns it off.")
    a.default_model = asker.text("default_model", "Default model (Enter: HyperNix's default)", base.default_model)

    head("4 · What it offers")
    a.hyperlink = asker.yes_no("hyperlink", "HyperLink: the phone app's pairing, chats and files?", base.hyperlink)
    a.web = a.hyperlink and asker.yes_no(
        "web", "HyperLink on the web, on this machine and the tailnet (port 37965)?", base.web)
    a.claude_code = asker.yes_no("claude_code", "Claude Code on this server's models (the Messages API)?",
                                 base.claude_code)
    if a.claude_code:
        dim("The model Claude Code's own model names map to; blank: whatever is loaded.")
        a.claude_model = asker.text("claude_model", "Model for Claude Code (blank: the loaded one)",
                                    base.claude_model)
    a.mcp = asker.yes_no("mcp", "MCP: let MCP clients use this server's tools?", base.mcp)
    a.mcp_hf = a.mcp and asker.yes_no(
        "mcp_hf", "Connect Hugging Face's MCP server (reads HF_TOKEN)?", base.mcp_hf)
    dim("Code sandboxes run code on this machine as this user. A boundary for code")
    dim("that is wrong, not for code that is hostile.")
    a.code_sandbox = asker.yes_no("code_sandbox", "Code sandboxes for HyperLink's Code tab and /code?",
                                  base.code_sandbox)
    a.net_remote_run = asker.yes_no(
        "net_remote_run", "Let admins start programs on other tailnet machines (/net)?", base.net_remote_run)
    if a.public:
        dim("A public server sends one anonymous heartbeat an hour (a hashed random id")
        dim("and its version, never an address) to the count in the project's README.")
        a.public_counter = asker.yes_no("public_counter", "Be counted?", base.public_counter)

    head("5 · Finishing")
    a.autostart = asker.yes_no("autostart", "Start it at login (systemd user service)?", base.autostart)
    a.start = asker.yes_no("start", "Start it now?", base.start)
    return a


def _private_url(url: str) -> bool:
    import ipaddress
    from urllib.parse import urlsplit

    host = urlsplit(url).hostname or ""
    if host in ("localhost",) or host.endswith((".local", ".ts.net", ".internal", ".lan")):
        return True
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False


def build_env(a: Answers, *, config_dir: Path, old: dict[str, str], version: str) -> str:
    """The .env text for these answers. Pure: the tests read it."""
    host = {"local": "127.0.0.1", "network": "0.0.0.0"}.get(a.reach) or _tailnet_ip() or "127.0.0.1"
    secret = old.get("T1_TOKEN_SECRET") or secrets.token_hex(32)
    b = lambda v: "1" if v else "0"  # noqa: E731
    lines = [
        f"# Written by hnx-t1 setup (installer {INSTALLER}, hypernix {version})",
        f"# on {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}. Mode 0600: it holds the token secret.",
        "# Run `hnx-t1 setup` again to change the answers; the old file is kept as .env.bak-*.",
        "",
        "# --- The server --------------------------------------------------------",
        f"T1_SERVER_NAME={a.name}",
        f"T1_ENVIRONMENT={'production' if a.public else 'development'}",
        f"T1_HOST={host}",
        f"T1_PORT={a.port}",
        f"T1_HYPERLINK_PORT={a.port}",
        *([f"T1_HYPERLINK_PUBLIC_URL={a.public_url}"] if a.public_url else []),
        f"T1_PUBLIC={b(a.public)}",
        "",
        "# --- Secrets and keys ---------------------------------------------------",
        "# Signs scoped tokens; kept across setups so existing keys keep working.",
        f"T1_TOKEN_SECRET={secret}",
        f"T1_KEYMASTER_DIR={config_dir / 'keymaster'}",
        "",
        "# --- Access -------------------------------------------------------------",
        f"T1_TRUSTED_NETWORK={b(a.trusted)}",
        f"T1_TRUSTED_NETWORK_PARTIAL_ADMIN={b(a.partial_admin)}",
        "T1_CORS_ALLOW_ORIGINS=",
        f"T1_RATE_LIMIT_ENABLED={b(a.limits != 'off')}",
        *([f"T1_RATE_LIMIT_RULES='{LIMITS[a.limits]}'"] if LIMITS[a.limits] else []),
        "T1_AUDIT_ENABLED=1",
        "",
        "# --- Storage ------------------------------------------------------------",
        f"T1_DB_PATH={config_dir / 't1api.sqlite3'}",
        f"T1_MODULE_STORAGE_DIR={config_dir / 'modules'}",
        f"T1_HYPERLINK_FILES_DIR={config_dir / 'files'}",
        f"T1_BACKUP_DIR={config_dir / 'backups'}",
        "",
        "# --- Models -------------------------------------------------------------",
        "T1_MODEL_SYNC=1",
        f"T1_MODELS_DIR={config_dir / 'models'}",
        f"T1_HF_DOWNLOAD_DIR={config_dir / 'models'}",
        f"T1_LMSTUDIO_ENABLED={b(a.models in ('lmstudio', 'both'))}",
        *([f"T1_LMSTUDIO_URL={a.lmstudio_url}"] if a.models in ("lmstudio", "both") else []),
        *([f"T1_DEFAULT_MODEL={'' if a.default_model.lower() == 'none' else a.default_model}"]
          if a.default_model else []),
        "",
        "# --- What it offers -----------------------------------------------------",
        f"T1_HYPERLINK_ENABLED={b(a.hyperlink)}",
        f"T1_WEB_ENABLED={b(a.web)}",
        "T1_WEB_PORT=37965",
        f"T1_ANTHROPIC_ENABLED={b(a.claude_code)}",
        f"T1_CLAUDE_CODE_MODEL={a.claude_model if a.claude_code else ''}",
        f"T1_MCP_ENABLED={b(a.mcp)}",
        f"T1_MCP_HF={b(a.mcp_hf)}",
        f"T1_CODE_SANDBOX={b(a.code_sandbox)}",
        f"T1_NET_REMOTE_RUN={b(a.net_remote_run)}",
        f"T1_PUBLIC_COUNTER={b(a.public_counter)}",
    ]
    written = {line.split("=", 1)[0] for line in lines if line and not line.startswith("#")}
    kept = [f"{k}={_quote(v)}" for k, v in old.items() if k not in written]
    if kept:
        lines += ["", "# --- Kept from the configuration before (setup does not ask about these) --", *kept]
    return "\n".join(lines) + "\n"


def _quote(value: str) -> str:
    return f"'{value}'" if any(ch in value for ch in " #\"'$`\\") and "'" not in value else value


def _redact_sensitive_env(text: str) -> str:
    sensitive_keys = {
        "T1_TOKEN_SECRET",
    }
    sensitive_markers = (
        "SECRET",
        "TOKEN",
        "PASSWORD",
        "PASS",
        "API_KEY",
        "PRIVATE",
        "CREDENTIAL",
        "AUTH",
        "KEY",
    )
    redacted_lines: list[str] = []
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            key = key.strip()
            key_upper = key.upper()
            is_sensitive = key_upper in sensitive_keys or any(marker in key_upper for marker in sensitive_markers)
            if is_sensitive and value:
                redacted_lines.append(f"{key}=<redacted>")
                continue
        redacted_lines.append(line)
    return "\n".join(redacted_lines) + ("\n" if text.endswith("\n") else "")


# ---------------------------------------------------------------------------
# Doing it
# ---------------------------------------------------------------------------

def _manager(given: str) -> list[str]:
    if given and Path(given).is_file():
        return [shutil.which("bash") or "bash", given]
    found = shutil.which("hypernix-t1") or shutil.which("hnx-t1")
    return [found] if found else []


def _summary(a: Answers, path: Path) -> None:
    head("This is what it will be")
    rows = [
        ("Server", f"{a.name}, {REACH[a.reach]}, port {a.port}"),
        ("Public URL", a.public_url or "none"),
        ("Keyless", "trusted networks" + (" (with some admin routes)" if a.partial_admin else "") if a.trusted
         else "off: every request needs a key"),
        ("Limits", a.limits),
        ("Models", MODELS[a.models] + {"": "; HyperNix's default model", "none": "; no default model"}.get(
            a.default_model.lower(), f"; default {a.default_model}")),
        ("HyperLink", ("on" + (", and on the web" if a.web else "")) if a.hyperlink else "off"),
        ("Claude Code", ("on" + (f" ({a.claude_model})" if a.claude_model else "")) if a.claude_code else "off"),
        ("MCP", ("on" + (", with Hugging Face's" if a.mcp_hf else "")) if a.mcp else "off"),
        ("Code sandboxes", "on" if a.code_sandbox else "off"),
        ("Remote runs", "on" if a.net_remote_run else "off"),
        ("Counted", ("yes" if a.public_counter else "no") if a.public else "no (not public)"),
        ("Config", str(path)),
    ]
    for label, value in rows:
        print(f"     {_c('2', f'{label:<15}')} {value}")


def _write(path: Path, text: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        from .envupgrade import backup_name

        backup = backup_name(path)
        shutil.copy2(path, backup)
        backup.chmod(0o600)
    tmp = path.with_name(".env.setup-tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
    for sub in ("keymaster", "models", "files", "modules", "backups"):
        folder = path.parent / sub
        folder.mkdir(exist_ok=True)
        if sub == "keymaster":
            folder.chmod(0o700)
    return backup


def _next_steps(a: Answers) -> None:
    head("Next")
    dim("hnx-t1 status                 # where it is, and the first admin key")
    dim("hnx list-all                  # every command")
    if a.web:
        dim("http://127.0.0.1:37965        # HyperLink on the web")
    if a.claude_code:
        dim("hnx claude                    # Claude Code on this server's model")
    if a.mcp:
        dim(f"claude mcp add --transport http hypernix http://127.0.0.1:{a.port}/mcp")
    dim("waiter                        # the T1 client, against any server")


def cli_main(argv: list[str] | None = None, *, read: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(prog="hnx-t1 setup", description=f"The T1 interactive installer ({INSTALLER}).")
    parser.add_argument("--yes", "-y", action="store_true", help="take every default; ask nothing")
    parser.add_argument("--answers", metavar="FILE", help="a JSON object of answers by name; defaults for the rest")
    parser.add_argument("--dry-run", action="store_true", help="print the configuration; write and start nothing")
    parser.add_argument("--config-dir", default="", help="default: $T1_CONFIG_DIR or ~/.hypernix/t1api")
    parser.add_argument("--no-start", action="store_true", help="write the configuration and stop there")
    parser.add_argument("--manager", default="", help=argparse.SUPPRESS)   # hypernix-t1 passes itself
    args = parser.parse_args(argv)
    global _OUT
    _OUT = sys.stderr if args.dry_run else sys.stdout

    config_dir = Path(args.config_dir or os.environ.get("T1_CONFIG_DIR") or Path.home() / ".hypernix" / "t1api")
    env_path = config_dir / ".env"
    preset: dict[str, Any] = {}
    if args.answers:
        try:
            preset = json.loads(Path(args.answers).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"hnx-t1 setup: {_refused()}: cannot read {args.answers}: {exc}", file=sys.stderr)
            return 2
        unknown = set(preset) - set(Answers.__dataclass_fields__)
        if unknown:
            print(f"hnx-t1 setup: {_refused()}: unknown answer(s) {', '.join(sorted(unknown))}; "
                  f"known: {', '.join(Answers.__dataclass_fields__)}", file=sys.stderr)
            return 2
    interactive = not args.yes and sys.stdin.isatty() if read is input else not args.yes
    from .. import __version__

    old = read_env(env_path)
    print(_c("1;36", "HyperNix T1 setup") + _c("2", f" · installer {INSTALLER} · hypernix {__version__}"), file=_OUT)
    if old:
        dim(f"{env_path} exists: its answers are the defaults, and its token secret is kept.")
    elif not interactive:
        dim("No questions asked: taking the defaults" + (" and the answers file." if preset else "."))
    try:
        answers = ask(Asker(interactive=interactive, preset=preset, read=read), defaults_from(old))
    except SetupError as exc:
        print(f"hnx-t1 setup: {exc}", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\nhnx-t1 setup: stopped; nothing was written.", file=sys.stderr)
        return 130
    text = build_env(answers, config_dir=config_dir, old=old, version=__version__)
    if args.dry_run:
        _summary(answers, env_path)
        print("hnx-t1 setup: dry-run; configuration was not written.")
        return 0
    _summary(answers, env_path)
    if interactive and not Asker(interactive=True, read=read).yes_no("write", "Write it?", True):
        print("hnx-t1 setup: nothing was written.")
        return 1
    backup = _write(env_path, text)
    ok(f"Wrote {env_path} (0600)")
    if backup:
        ok(f"The one before is {backup.name}")
    manager = _manager(args.manager)
    if not manager and (answers.autostart or answers.start) and not args.no_start:
        warn("hypernix-t1 is not on PATH, so it was not started; `hnx-t1 start` when it is.")
    elif not args.no_start:
        env = {**os.environ, "T1_CONFIG_DIR": str(config_dir)}
        if answers.autostart:
            subprocess.run([*manager, "autostart", "on"], env=env, check=False)
        elif answers.start:
            command = "restart" if old else "start"
            subprocess.run([*manager, command], env=env, check=False)
    _next_steps(answers)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
