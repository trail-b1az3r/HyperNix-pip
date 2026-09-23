"""waiter.cli — argparse-based CLI for the ``waiter`` console script.

Complete as of Beta 3: every flag in the spec's ``serv`` list is wired to
real behaviour. The ones Beta 1/2 could only record locally
(``-B``/``-W``/``-a``/``-r`` and the full ``-G``/``-Rf``/``-y``) now call
the server endpoints that Beta 3 added — see ``t1api/routers/security.py``
for the blacklist/whitelist/appeal/forced-limit surface and
``waiter/tui.py`` for the curses dashboard.

``-B``/``-W``/``-a``/``-r`` keep writing to the local config *as well as*
the server. That is deliberate rather than redundant: the local copy is
what ``waiter config`` shows and what a re-run of ``waiter serv -A``
re-applies against a rebuilt server, and it is the only record available
when the operator's key isn't admin (in which case the server call is
refused and the CLI says so plainly instead of pretending it worked).

Style matches ``hypernix.gkey_cli``: manual dispatch dict, rich-formatted
output with a plain-text fallback when ``rich`` isn't installed (it's a
core hypernix dependency, so the fallback is mostly defensive).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any

from .client import T1Client, T1ClientError
from .local_config import WaiterConfigLocked, WaiterConfigStore, WaiterLocalConfig

# ---------------------------------------------------------------------------
# Rich helpers (graceful degradation — mirrors hypernix.gkey_cli)
# ---------------------------------------------------------------------------


def _try_rich() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


_HAS_RICH = _try_rich()


def _ok(text: str) -> None:
    if _HAS_RICH:
        from rich.console import Console

        Console().print(f"[green]\u2713[/green] {text}")
    else:
        print(f"OK: {text}")


def _warn(text: str) -> None:
    if _HAS_RICH:
        from rich.console import Console

        Console(stderr=True).print(f"[yellow]![/yellow] {text}")
    else:
        print(f"WARNING: {text}", file=sys.stderr)


def _err_connection(exc: Exception, context: str = "") -> None:
    """Print an error, expanded into a diagnosis when it is a reachability one.

    "Could not reach http://127.0.0.1:1234/hyperlink/pair: [Errno 111]
    Connection refused" is accurate and answers none of the questions the
    reader has. Only unreachability gets the extra work — every other
    error already says what is wrong.

    ``context`` is what the caller was doing ("Automatic setup"), kept
    because "could not reach" alone does not say which step gave up.

    This used to be wired into exactly one of a dozen T1ClientError
    handlers, so whether you got a diagnosis or a bare errno depended on
    which subcommand you happened to run — and `serv -A`, the first
    command anyone runs after an install, was one of the bare ones.
    """
    text = str(exc)
    url = _failed_url(text)
    if url is None:
        _err(f"{context}: {text}" if context else text)
        return
    if context:
        # Just the context. The diagnosis below restates the failure in
        # full, and printing the raw errno line as well says the same
        # thing twice before saying anything useful.
        _err(f"{context}:")

    from .diagnose import diagnose, format_diagnosis

    reason = text.split(": ", 1)[1] if ": " in text else ""
    try:
        result = diagnose(url, reason, source=_ADDRESS_SOURCE)
    except Exception:  # noqa: BLE001
        # A diagnostic that fails must not replace the real error.
        _err(text)
        return
    lines = format_diagnosis(result).split("\n")
    if not context:
        _err(lines[0])
        lines = lines[1:]
    for line in lines:
        print(line, file=sys.stderr)


_UNREACHABLE = re.compile(r"Could not reach (https?://\S+?)(?::\s|$)")


def _failed_url(text: str) -> str | None:
    """The URL out of a transport error, or None if this is not one."""
    match = _UNREACHABLE.search(text)
    return match.group(1) if match else None


def _err(text: str) -> None:
    if _HAS_RICH:
        from rich.console import Console

        Console(stderr=True).print(f"[red]\u2717[/red] {text}", style="red")
    else:
        print(f"ERROR: {text}", file=sys.stderr)


def _info(text: str) -> None:
    if _HAS_RICH:
        from rich.console import Console

        Console().print(text)
    else:
        print(text)


def _print_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    if _HAS_RICH:
        from rich.console import Console
        from rich.table import Table

        t = Table(title=title, header_style="bold cyan", border_style="dim")
        for h in headers:
            t.add_column(h, overflow="fold")
        for row in rows:
            t.add_row(*row)
        Console().print(t)
    else:
        if title:
            print(f"\n{title}")
        print("  ".join(headers))
        for row in rows:
            print("  ".join(row))


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _mask(secret: str | None, keep: int = 8) -> str:
    if not secret:
        return "-"
    return secret[:keep] + "…" if len(secret) > keep else secret


# ---------------------------------------------------------------------------
# Config / client resolution shared by every subcommand
# ---------------------------------------------------------------------------


def _add_common_connection_args(parser: argparse.ArgumentParser) -> None:
    """-I/-K/-F/-P/-H are meaningful outside of `serv` too (e.g. `waiter
    models -I ... -K ...` without ever running `serv`), so every subcommand
    accepts them as overrides on top of the saved local config."""
    parser.add_argument("-I", "--server", dest="server", default=None, help="Server IP/Tailscale IP/URL")
    parser.add_argument("-K", "--key", dest="key", default=None, help="T1 token")
    parser.add_argument("-F", "--config-file", dest="config_file", default=None, help="Local config file path")
    parser.add_argument("-P", "--port", dest="port", type=int, default=None, help="Server port")
    parser.add_argument("-H", "--home", dest="home_url", default=None, help="Home page URL override")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Print raw JSON instead of a table")


def _ask_password() -> str | None:
    """For a locked config (waiter serv -e): the environment, or a prompt."""
    env = os.environ.get("HNX_WAITER_PASSWORD")
    if env:
        return env
    if sys.stdin.isatty():
        import getpass

        return getpass.getpass("waiter config password: ") or None
    return None


def _load_store(args: argparse.Namespace, *, encrypt: bool | None = None) -> WaiterConfigStore:
    enc = encrypt if encrypt is not None else bool(getattr(args, "encrypt", False))
    return WaiterConfigStore(args.config_file, encrypt=enc, password_provider=_ask_password)


#: Where the server address in the last resolved config came from. Set by
#: :func:`_resolve_config` and read only when a connection fails — "the
#: address you are using is the one saved three weeks ago" is frequently
#: the entire answer, and nothing else in the failure says it.
_ADDRESS_SOURCE = ""


def _resolve_config(args: argparse.Namespace, *, store: WaiterConfigStore | None = None) -> WaiterLocalConfig:
    global _ADDRESS_SOURCE
    store = store or _load_store(args)
    saved = store.load() or WaiterLocalConfig()
    _ADDRESS_SOURCE = f"the saved config ({store.path})" if saved.server else ""
    if args.server:
        saved.server = args.server
        _ADDRESS_SOURCE = "-I on the command line"
    if args.key:
        saved.key = args.key
    if getattr(args, "port", None) is not None:
        saved.port = args.port
        _ADDRESS_SOURCE += " with -P" if _ADDRESS_SOURCE else "-P on the command line"
    if getattr(args, "home_url", None):
        saved.home_url = args.home_url
    return saved


def _base_url(cfg: WaiterLocalConfig) -> str:
    if not cfg.server:
        raise T1ClientError(
            "No server configured. Run 'waiter serv -A -I <server> -K <key>' first, "
            "or pass -I explicitly."
        )
    server = cfg.server
    if not server.startswith(("http://", "https://")):
        scheme = "http" if cfg.local_only else "https"
        server = f"{scheme}://{server}"
    if cfg.port and f":{cfg.port}" not in server:
        server = f"{server}:{cfg.port}"
    return server


def _client_for(args: argparse.Namespace) -> tuple[T1Client, WaiterLocalConfig]:
    cfg = _resolve_config(args)
    return T1Client(base_url=_base_url(cfg), credential=cfg.key), cfg


# ---------------------------------------------------------------------------
# `waiter serv` — single-command automatic setup
# ---------------------------------------------------------------------------

_SERV_HELP = """\
waiter serv — set up, check and manage this machine's connection to a T1
API server.

    waiter serv -A -I <server> -K <key>          set up
    waiter serv -ArEK <key> -I <server>          set up, refresh, seal the key, check it
    waiter serv -bAE <server> <key>              the same, strings worked out by -b

Letters can be grouped in any order. A letter that takes a value (-I -K
-F -P -H -B -W -a -C -k, and a lone -r) ends its group, and the value is
the next word; inside a group r means refresh. -Rf is a full refresh and
-ud updates to exactly the server's version.

  -A  set up: check the key and save      -E  seal the key as a v2.1 key
  -R  refresh   -Rf full refresh           -y  sync settings from the server
  -Y  show the server's public details    -S  security check, client then server
  -u  update hypernix to at least the server's version   (-ud: exactly it)
  -c  conceal this key (address masked, data kept 36 hours; level 3+)
  -k  install a kit (a folder or .zip with a kit.json)
  -e  lock this config with a password    -T  control screen (level-9 T2 admin)
  -b  give bare strings to the options they look like
  -s  save   -L  local only   -G  dashboard   -g  prompt
  -B/-W/-a/-r  block / allow / remove / limit on the server (admin)
"""


def _build_serv_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="waiter serv", description=_SERV_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common_connection_args(p)
    p.add_argument("-A", "--auto", action="store_true", help="Automatic configuration: validate + save in one step")
    p.add_argument("-E", "--encrypt", action="store_true", help="Seal the key as a v2.1 (T2C) key when the server supports it, and encrypt the config at rest")
    p.add_argument("-s", "--save", action="store_true", help="Save current server/local configuration to a .jsonl file")
    p.add_argument("-L", "--local-only", action="store_true", help="Local/Tailscale/localhost-only mode")
    p.add_argument("-B", "--blacklist", action="append", default=[], metavar="IP_OR_CIDR", help="Blacklist an IP or CIDR range on the server (repeatable; admin key required)")
    p.add_argument("-W", "--whitelist", action="append", default=[], metavar="IP_OR_CIDR", help="Allowlist an IP or CIDR range on the server (repeatable; admin key required)")
    p.add_argument("-r", "--force-limit", action="append", default=[], metavar="SUBJECT=LIMIT", help="Force a limit on a T1 key/server, e.g. key:abc123=60/60s (admin key required)")
    p.add_argument("-a", "--appeal", action="append", default=[], metavar="IP_OR_CIDR", help="Appeal: remove an IP/CIDR from the server allow/block lists (admin key required)")
    p.add_argument("-C", "--config", dest="extra_config", action="append", default=[], metavar="KEY=VALUE", help="Additional configuration settings")
    p.add_argument("-G", "--gui", action="store_true", help="Open the full curses TUI dashboard")
    p.add_argument("-g", "--cli", action="store_true", help="Open an interactive CLI session")
    p.add_argument("-R", "--refresh", action="store_true", help="Quick refresh: re-validate + re-fetch models")
    p.add_argument("-Rf", "--force-refresh", dest="force_refresh", action="store_true", help="Force a full refresh: models, servers, modules, events, and config")
    p.add_argument("-y", "--sync", action="store_true", help="Synchronize local config against the server's current /config + /models")
    p.add_argument("--promote-admin", dest="promote_admin", action="store_true", help="After validating, request admin promotion for this key (requires the authenticating key to already be admin-scoped — see POST /auth/t1/admin/rotate)")
    # 0.72.6
    p.add_argument("-b", "--bundle", action="store_true", help="Give bare strings to the options they look like")
    p.add_argument("-u", "--update", action="store_true", help="Update hypernix to at least the server's version")
    p.add_argument("--update-exact", dest="update_exact", action="store_true", help="(-ud) Update hypernix to exactly the server's version")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="With -u: print the pip command instead of running it")
    p.add_argument("-k", "--kit-install", dest="kit_install", action="append", default=[], metavar="PATH", help="Install a kit: a folder or .zip with a kit.json")
    p.add_argument("-c", "--conceal", action="store_true", help="Conceal this key on the server: address masked, data kept 36 hours (access level 3+)")
    p.add_argument("--no-conceal", dest="no_conceal", action="store_true", help="Stop concealing this key")
    p.add_argument("-T", "--control", action="store_true", help="Open the control screen (a verified level-9 T2 administrator key)")
    p.add_argument("-Y", "--info", action="store_true", help="Show the server's public details: name, description, owner, url")
    p.add_argument("-S", "--security-check", dest="security_check", action="store_true", help="Check this client, then the server, for security problems")
    p.add_argument("-e", "--lock", action="store_true", help="Lock this config with a password (Rotorvault)")
    p.add_argument("--unlock", action="store_true", help="Remove the password lock from this config")
    return p


def _parse_kv_list(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            _warn(f"Ignoring malformed -C value (expected KEY=VALUE): {item!r}")
            continue
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip()
    return out


def _parse_forced_limit(spec: str) -> dict[str, object] | None:
    """Parse a ``-r`` value into a forced-limit request.

    Accepted forms (the subject prefix is optional and defaults to
    ``key``, since capping a key is the common case)::

        key:abc123=60/60s      60 requests per 60 seconds
        server:srv-1=120/1m    120 requests per minute
        abc123=1000t/1h        1000 tokens per hour

    Returns None (with a warning) rather than raising on a malformed
    value: one bad ``-r`` in a long command line should not discard the
    rest of the setup.
    """
    if "=" not in spec:
        _warn(f"Ignoring malformed -r value (expected SUBJECT=LIMIT/WINDOW): {spec!r}")
        return None
    subject, _, limit_spec = spec.partition("=")
    subject_type, _, subject_id = subject.partition(":")
    if not subject_id:
        subject_type, subject_id = "key", subject_type
    if subject_type not in ("key", "server"):
        _warn(f"Ignoring -r value with unknown subject type {subject_type!r} (use key: or server:)")
        return None
    if "/" not in limit_spec:
        _warn(f"Ignoring malformed -r limit {limit_spec!r} (expected COUNT/WINDOW, e.g. 60/60s)")
        return None
    count_text, _, window_text = limit_spec.partition("/")
    is_tokens = count_text.strip().lower().endswith("t")
    try:
        count = int(count_text.strip().rstrip("tT"))
        window = _parse_duration(window_text.strip())
    except ValueError:
        _warn(f"Ignoring -r value with unparseable numbers: {spec!r}")
        return None
    return {
        "subject_type": subject_type,
        "subject_id": subject_id,
        "tokens_per_window" if is_tokens else "requests_per_window": count,
        "window_seconds": window,
        "reason": "set via waiter serv -r",
    }


def _parse_duration(text: str) -> float:
    """``30s`` / ``5m`` / ``2h`` / ``1d`` / a bare number of seconds."""
    text = text.strip().lower()
    multipliers = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    if text and text[-1] in multipliers:
        return float(text[:-1]) * multipliers[text[-1]]
    return float(text)


def _apply_network_policy(client: T1Client, args: argparse.Namespace) -> None:
    """Push -B/-W/-a/-r to the server.

    Each entry is applied independently so one refusal doesn't abort the
    rest, and an admin-scope refusal is reported once with what it means
    rather than repeated per entry.
    """
    denied_reported = False

    def report(exc: T1ClientError, what: str) -> None:
        nonlocal denied_reported
        if exc.code in ("AUTH_ADMIN_REQUIRED", "AUTH_INSUFFICIENT_SCOPE"):
            if not denied_reported:
                _err(
                    "Server refused the network/limit changes: they need an admin-scoped T1 key. "
                    "They have still been saved to your local config."
                )
                denied_reported = True
        else:
            _err(f"{what}: {exc}")

    for cidr in args.blacklist:
        try:
            client.blacklist_ip(cidr, reason="set via waiter serv -B")
            _ok(f"Blacklisted {cidr} on the server")
        except T1ClientError as exc:
            report(exc, f"Could not blacklist {cidr}")

    for cidr in args.whitelist:
        try:
            client.whitelist_ip(cidr, reason="set via waiter serv -W")
            _ok(f"Allowlisted {cidr} on the server")
        except T1ClientError as exc:
            report(exc, f"Could not allowlist {cidr}")

    for cidr in args.appeal:
        try:
            client.appeal_ip(cidr)
            _ok(f"Appealed {cidr} — entry removed on the server")
        except T1ClientError as exc:
            if exc.code == "NOT_FOUND":
                _info(f"  {cidr} had no server-side entry (removed locally only).")
            else:
                report(exc, f"Could not appeal {cidr}")

    for spec in args.force_limit:
        parsed = _parse_forced_limit(spec)
        if parsed is None:
            continue
        try:
            client.set_forced_limit(**parsed)
            _ok(
                f"Forced limit on {parsed['subject_type']}:{str(parsed['subject_id'])[:12]} "
                f"applied on the server"
            )
        except T1ClientError as exc:
            report(exc, f"Could not force a limit for {spec}")


_SERV_ACTIONS = (
    "auto", "save", "refresh", "force_refresh", "sync", "cli", "gui", "update", "update_exact",
    "kit_install", "conceal", "no_conceal", "control", "info", "security_check", "lock",
    "unlock", "encrypt",
)


def _seal_key(client: T1Client, cfg: WaiterLocalConfig) -> None:
    """-E: replace a plain key with a v2.1 kit, when the server has them."""
    key = cfg.key or ""
    if not key or key.startswith(("T2CK_", "T2C_", "T1S.")):
        return
    try:
        import socket

        kit = client.seal_key(key, label=socket.gethostname()[:40])
    except ImportError:
        _warn('-E: v2.1 keys need the cryptography package (pip install "hypernix[security]"); '
              "the key is saved encrypted at rest instead.")
        return
    except T1ClientError as exc:
        if getattr(exc, "status", None) == 404:
            _warn("-E: this server has no v2.1 keys; the key is saved encrypted at rest instead.")
            return
        raise
    except ValueError as exc:
        # A scoped token or a key in no standard format has no T2 form to
        # seal. It is still saved, encrypted at rest.
        _warn(f"-E: this key cannot be sealed as a v2.1 key ({exc}); it is saved encrypted at rest instead.")
        return
    cfg.key = kit.to_text()
    client.credential = cfg.key
    _ok(f"Sealed the key as a v2.1 key (device {kit.device_id}); only the kit is kept, and "
        "each day's key is made from it.")


def _new_password() -> str | None:
    env = os.environ.get("HNX_WAITER_PASSWORD")
    if env:
        return env
    if not sys.stdin.isatty():
        _err("-e needs a password: run in a terminal to be asked, or set HNX_WAITER_PASSWORD.")
        return None
    import getpass

    first = getpass.getpass("Password to lock this config: ")
    if len(first) < 8:
        _err("Use at least 8 characters.")
        return None
    if getpass.getpass("Again: ") != first:
        _err("The two did not match.")
        return None
    return first


def _cmd_serv(rest: list[str]) -> int:
    from .servargs import ServArgsError, expand

    try:
        expanded = expand(rest)
    except ServArgsError as exc:
        _err(str(exc))
        return 2
    for text, option in expanded.assigned:
        shown = _mask(text) if option == "--key" else text
        _info(f"-b: read {shown!r} as {option}")
    args = _build_serv_parser().parse_args(expanded.argv)

    if not any(getattr(args, name) for name in _SERV_ACTIONS):
        _warn("No action flag given (-A/-s/-R/-Rf/-y/-g/-G/-u/-k/-c/-T/-Y/-S/-e). Nothing to do — see 'waiter serv --help'.")
        return 1

    store = _load_store(args, encrypt=args.encrypt)
    cfg = _resolve_config(args, store=store)
    cfg.local_only = cfg.local_only or args.local_only
    if args.blacklist:
        cfg.blacklist = sorted(set(cfg.blacklist) | set(args.blacklist))
    if args.appeal:
        before = set(cfg.blacklist)
        cfg.blacklist = sorted(before - set(args.appeal))
        cfg.whitelist = sorted(set(cfg.whitelist) - set(args.appeal))
        removed = before - set(cfg.blacklist)
        if removed:
            _info(f"Appealed (removed from local blacklist): {', '.join(sorted(removed))}")
    if args.whitelist:
        cfg.whitelist = sorted(set(cfg.whitelist) | set(args.whitelist))
    if args.extra_config:
        cfg.extra_config.update(_parse_kv_list(args.extra_config))
    if args.force_limit:
        cfg.forced_limits = sorted(set(cfg.forced_limits) | set(args.force_limit))

    policy_flags = bool(args.blacklist or args.whitelist or args.appeal or args.force_limit)
    exit_code = 0
    changed = False

    def client() -> T1Client:
        return T1Client(base_url=_base_url(cfg), credential=cfg.key)

    # -S and -Y look before anything authenticates: "check the server you
    # are about to connect to" means before the key is sent to it.
    if args.security_check:
        from .servops import security_check

        if not cfg.server:
            _err("-S needs a server: -I <server>.")
            return 1
        findings = security_check(client(), cfg, store_path=store.path, locked=store.is_locked())
        marks = {"high": "✗", "medium": "!", "low": "·", "ok": "✓"}
        for finding in findings:
            line = f"  {marks.get(finding.level, '?')} [{finding.where}] {finding.message}"
            print(line + (f"\n      fix: {finding.fix}" if finding.fix else ""))
        if any(f.level == "high" for f in findings):
            _warn("Security check: at least one serious problem (✗) above.")
            exit_code = 1
        else:
            _ok("Security check: nothing serious found.")

    if args.info:
        from .servops import format_info

        if not cfg.server:
            _err("-Y needs a server: -I <server>.")
            return 1
        try:
            info = client().server_info()
        except T1ClientError as exc:
            _err_connection(exc, "Server info")
            return 1
        print("\n".join(format_info(info)))

    if args.auto:
        if not cfg.server or not cfg.key:
            _err("-A requires both -I <server> and -K <T1_TOKEN>.")
            return 1
        c = client()
        if args.encrypt:
            _seal_key(c, cfg)
        try:
            validated = c.validate()
        except T1ClientError as exc:
            _err_connection(exc, "Automatic setup failed")
            return 1
        family = validated.get("key_family") or "T1"
        level = validated.get("access_level")
        _ok(f"Connected to {cfg.server} — key {_mask(validated.get('key_id'))} ({validated.get('key_type')}"
            + (f", {family} level {level}" if family != "T1" else "") + ")")
        if validated.get("scopes"):
            _info(f"  scopes: {', '.join(validated['scopes'])}")

        if args.promote_admin:
            if not validated.get("is_admin"):
                _warn("--promote-admin requested, but the authenticating key is not admin-scoped — skipped.")
            else:
                try:
                    promoted = c.admin_rotate(validated["key_id"], promote_to_admin=True)
                    cfg.key = promoted["key"]
                    _ok(f"Promoted to admin key {_mask(promoted['key_id'])}")
                except T1ClientError as exc:
                    _err_connection(exc, "Admin promotion failed")

        try:
            status = c.status()
            if not status.get("production_ready", True) and status.get("environment") == "production":
                _warn(
                    f"Server reports {len(status.get('production_warnings', []))} production "
                    "configuration warning(s) — run 'waiter doctor' to see them."
                )
        except T1ClientError:
            pass
        changed = True
    elif args.encrypt and cfg.server and cfg.key:
        _seal_key(client(), cfg)
        changed = True

    if policy_flags:
        if not cfg.server or not cfg.key:
            _warn("-B/-W/-a/-r were saved locally, but there's no configured server+key to apply them to.")
        else:
            _apply_network_policy(client(), args)
        changed = True

    if args.refresh or args.force_refresh:
        try:
            c = client()
            validated = c.validate()
            models = c.list_models()
        except T1ClientError as exc:
            _err_connection(exc, "Refresh failed")
            return 1
        _ok(f"Refreshed — key {_mask(validated.get('key_id'))} still valid, {models.get('count', 0)} model(s) visible.")
        if args.force_refresh:
            for label, fetch in (
                ("servers", lambda: c.list_servers().get("count", 0)),
                ("modules", lambda: c.list_modules().get("count", 0)),
                ("events", lambda: c.list_events(limit=50).get("count", 0)),
            ):
                try:
                    _info(f"  {label}: {fetch()}")
                except T1ClientError as exc:
                    _warn(f"  {label}: unavailable ({exc.code or exc})")
            try:
                remote_config = c.config()["config"]
                _info(f"  server config: {len(remote_config)} setting(s) visible")
            except T1ClientError as exc:
                _warn(f"  server config: unavailable ({exc.code or exc})")

    if args.sync:
        try:
            c = client()
            remote_config = c.config()["config"]
            models = c.list_models()
        except T1ClientError as exc:
            _err_connection(exc, "Sync failed")
            return 1
        cfg.extra_config.update(
            {
                "server_environment": str(remote_config.get("environment", "")),
                "server_default_plan": str(remote_config.get("default_plan", "")),
                "server_storage_backend": str(remote_config.get("storage_backend", "")),
                "server_model_count": str(models.get("count", 0)),
            }
        )
        _ok(f"Synchronized against {cfg.server} — {models.get('count', 0)} model(s), "
            f"plan '{remote_config.get('default_plan', '?')}'.")
        if args.as_json:
            _print_json(remote_config)
        changed = True

    if args.update or args.update_exact:
        from .servops import local_version, run_update, update_plan

        try:
            server_version = str(client().status().get("hypernix_version") or "")
        except T1ClientError as exc:
            _err_connection(exc, "Update")
            return 1
        requirement, why = update_plan(local_version(), server_version, exact=args.update_exact)
        if requirement is None:
            _ok(f"Update: nothing to do — {why}.")
        else:
            _info(f"Update: {why}")
            code = run_update(requirement, dry_run=args.dry_run)
            if code != 0:
                _err(f"pip exited with {code}; hypernix was not updated.")
                exit_code = code
            elif not args.dry_run:
                _ok("Updated. Restart anything that has hypernix loaded to use it.")

    if args.conceal or args.no_conceal:
        if not cfg.server or not cfg.key:
            _err("-c needs a server and a key: -I <server> -K <key>.")
            return 1
        try:
            result = client().conceal(not args.no_conceal)
        except T1ClientError as exc:
            _err_connection(exc, "Conceal")
            return 1
        cfg.concealed = bool(result.get("concealed"))
        if cfg.concealed:
            swept = result.get("swept") or {}
            _ok(f"Concealed: the server keeps your address as its network only, and what this "
                f"key makes for {result.get('retention_hours', 36)} hours (memories, preferences "
                "and usage counts excepted).")
            if any(swept.values()):
                _info("  deleted now: " + ", ".join(f"{v} {k}" for k, v in swept.items() if v))
        else:
            _ok("No longer concealed. Nothing already deleted comes back.")
        changed = True

    for source in args.kit_install:
        from .kits import KitError, install

        try:
            kit = install(source)
        except KitError as exc:
            _err(f"-k {source}: {exc}")
            return 1
        _ok(f"Installed kit {kit.name} {kit.version} ({kit.kind}) to {kit.path}")
        _warn("A kit is code that runs as you. Keep only kits you would run yourself (waiter kits remove <name>).")

    if args.unlock:
        store.password = None
        changed = True
    if args.lock:
        password = _new_password()
        if password is None:
            return 1
        store.password = password
        changed = True

    if changed or args.save:
        path = store.save(cfg)
        how = " (locked)" if store.password else (" (encrypted)" if args.encrypt else "")
        _ok(f"Saved config to {path}{how}")

    if args.control:
        from .servops import control_allowed

        try:
            validated = client().validate()
        except T1ClientError as exc:
            _err_connection(exc, "Control")
            return 1
        allowed, why = control_allowed(validated)
        if not allowed:
            _err(f"-T: {why}.")
            return 1
        _ok(f"-T: {why}.")
        from .tui import run as run_tui

        return run_tui(client(), control=True)

    if args.gui:
        return _launch_tui(cfg)

    if args.cli:
        return _interactive_session(cfg, store)

    return exit_code


def _cmd_kits(rest: list[str]) -> int:
    """``waiter kits [list | remove NAME | run NAME COMMAND [ARGS...]]`` (0.72.6)."""
    from .kits import KitError, installed, remove, run_command

    action = rest[0] if rest else "list"
    try:
        if action in ("list", "ls"):
            kits = installed()
            if "--json" in rest:
                _print_json([k.to_dict() for k in kits])
            elif not kits:
                _info("No kits installed. Install one with: waiter serv -k <folder-or-zip>")
            else:
                _print_table(["Kit", "Version", "Kind", "Commands", "Description"],
                             [[k.name, k.version, k.kind, ", ".join(sorted(k.commands)) or "—",
                               k.description or "—"] for k in kits], title="Installed kits")
            return 0
        if action in ("remove", "rm") and len(rest) == 2:
            if remove(rest[1]):
                _ok(f"Removed kit {rest[1]}")
                return 0
            _err(f"No kit called {rest[1]!r} is installed.")
            return 1
        if action == "run" and len(rest) >= 3:
            return run_command(rest[1], rest[2], rest[3:])
    except KitError as exc:
        _err(str(exc))
        return 1
    _err("usage: waiter kits [list [--json] | remove NAME | run NAME COMMAND [ARGS...]]")
    return 2


def _launch_tui(cfg: WaiterLocalConfig) -> int:
    """Open the curses dashboard (``-G`` / ``waiter tui``)."""
    if not cfg.server:
        _err("No server configured — run 'waiter serv -A -I <server> -K <key>' first.")
        return 1
    from .tui import run as run_tui

    return run_tui(T1Client(base_url=_base_url(cfg), credential=cfg.key))


def _interactive_session(cfg: WaiterLocalConfig, store: WaiterConfigStore) -> int:
    """-g: a minimal interactive REPL over the same subcommands available
    one-shot from the top level. Not the full curses TUI (-G, Beta 3) —
    just enough to poke at a server without re-typing -I/-K every time."""
    if not cfg.server:
        _err("No server configured — run 'waiter serv -A -I <server> -K <key>' first.")
        return 1
    client = T1Client(base_url=_base_url(cfg), credential=cfg.key)
    _info("waiter interactive session — commands: models, status, usage, whoami, quit")
    while True:
        try:
            line = input("waiter> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            return 0
        try:
            if line == "models":
                _render_model_list(client.list_models())
            elif line == "status":
                _print_json(client.status())
            elif line == "usage":
                _print_json(client.usage_current())
            elif line == "whoami":
                _print_json(client.validate())
            elif line == "help":
                _info("commands: models, status, usage, whoami, quit")
            else:
                _warn(f"Unknown command: {line!r} (try: models, status, usage, whoami, quit)")
        except T1ClientError as exc:
            _err_connection(exc)


def _render_model_list(payload: dict[str, Any], *, as_json: bool = False) -> None:
    if as_json:
        _print_json(payload)
        return
    models = payload.get("models", [])
    rows = [
        [
            m["model_id"],
            m["display_name"],
            m["status"],
            m["minimum_plan"],
            "yes" if m["free_tier_available"] else "no",
            str(m["routing_priority"]),
        ]
        for m in models
    ]
    _print_table(
        ["model_id", "display_name", "status", "min_plan", "free_tier", "priority"],
        rows,
        title=f"Models ({payload.get('count', len(models))})",
    )
    if not models:
        _info(
            "No models visible. If you expect the shipped example registry, the server "
            "needs T1_ENABLE_EXAMPLE_MODELS=1 — see wiki/T1-API.md#model-registry."
        )


# ---------------------------------------------------------------------------
# Other subcommands
# ---------------------------------------------------------------------------


def _cmd_models(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter models", description="List models visible in the server's registry.")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    try:
        client, _ = _client_for(args)
        payload = client.list_models()
    except T1ClientError as exc:
        _err_connection(exc)
        return 1
    _render_model_list(payload, as_json=args.as_json)
    return 0


def _cmd_model(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter model", description="Show detail, availability, and usage for one model.")
    p.add_argument("model_id")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    try:
        client, cfg = _client_for(args)
        detail = client.get_model(args.model_id)
        availability = client.model_availability(args.model_id)
        usage = None
        if cfg.key:
            try:
                usage = client.model_usage(args.model_id)
            except T1ClientError:
                usage = None  # unauthenticated or key lacks access — detail/availability still shown
    except T1ClientError as exc:
        _err_connection(exc)
        return 1

    if args.as_json:
        _print_json({"model": detail.get("model"), "availability": availability, "usage": usage})
        return 0

    m = detail["model"]
    _info(f"[bold]{m['display_name']}[/bold] ({m['model_id']})" if _HAS_RICH else f"{m['display_name']} ({m['model_id']})")
    rows = [
        ["status", m["status"]],
        ["architecture", m["architecture"]],
        ["total_parameters (B)", str(m["total_parameters"])],
        ["active_parameters (B)", str(m["active_parameters"])],
        ["context_limit", str(m["context_limit"])],
        ["input_token_limit", str(m["input_token_limit"])],
        ["output_token_limit", str(m["output_token_limit"])],
        ["minimum_plan", m["minimum_plan"]],
        ["free_tier_available", str(m["free_tier_available"])],
        ["fallback_model", str(m["fallback_model"])],
        ["available now", str(availability["available"])],
    ]
    _print_table(["field", "value"], rows, title=m["display_name"])
    if usage:
        _print_table(
            ["input_remaining", "output_remaining", "exhausted"],
            [[str(usage["input_remaining"]), str(usage["output_remaining"]), str(usage["is_exhausted"])]],
            title="Your usage",
        )
    elif cfg.key:
        _info("(usage unavailable for this key/model)")
    return 0


def _cmd_status(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter status", description="Show T1 API server status.")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    try:
        client, _ = _client_for(args)
        payload = client.status()
    except T1ClientError as exc:
        _err_connection(exc)
        return 1
    if args.as_json:
        _print_json(payload)
    else:
        rows = [[k, str(v)] for k, v in payload.items() if k != "request_id"]
        _print_table(["field", "value"], rows, title="T1 API status")
    return 0


def _cmd_health(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter health", description="Check T1 API server liveness.")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    try:
        client, cfg = _client_for(args)
        payload = client.health()
    except T1ClientError as exc:
        _err_connection(exc)
        return 1
    if args.as_json:
        _print_json(payload)
    else:
        _ok(f"{cfg.server} is healthy ({payload.get('status')})")
    return 0


def _cmd_whoami(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter whoami", description="Validate the configured key and show its scopes.")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    cfg = _resolve_config(args)
    if not cfg.key:
        _err("No key configured. Pass -K or run 'waiter serv -A' first.")
        return 1
    try:
        client = T1Client(base_url=_base_url(cfg), credential=cfg.key)
        payload = client.validate()
    except T1ClientError as exc:
        _err_connection(exc)
        return 1
    if args.as_json:
        _print_json(payload)
    else:
        rows = [[k, str(v)] for k, v in payload.items() if k != "request_id"]
        _print_table(["field", "value"], rows, title="Identity")
    return 0


def _cmd_usage(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter usage", description="Show current usage, or remaining allowance for one model with --model.")
    p.add_argument("--model", dest="model_id", default=None, help="Show remaining allowance for this model_id instead of the full summary")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    try:
        client, _ = _client_for(args)
        if args.model_id:
            payload = client.usage_remaining(args.model_id)
        else:
            payload = client.usage_current()
    except T1ClientError as exc:
        _err_connection(exc)
        return 1
    if args.as_json:
        _print_json(payload)
        return 0
    if args.model_id:
        rows = [[k, str(v)] for k, v in payload.items() if k != "request_id"]
        _print_table(["field", "value"], rows, title=f"Remaining — {args.model_id}")
    else:
        _print_table(
            ["window requests", "window tokens", "all-time requests", "all-time tokens"],
            [[
                str(payload["current_window"]["requests"]),
                str(payload["current_window"]["total_tokens"]),
                str(payload["all_time"]["requests"]),
                str(payload["all_time"]["total_tokens"]),
            ]],
            title="Usage summary",
        )
        by_model = payload.get("by_model", [])
        if by_model:
            _print_table(
                ["model_id", "requests", "input_tokens", "output_tokens"],
                [[r["model_id"], str(r["requests"]), str(r["input_tokens"]), str(r["output_tokens"])] for r in by_model],
                title="By model (current window)",
            )
    return 0


def _cmd_config(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter config", description="Show the locally saved waiter config (key is masked).")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    cfg = _resolve_config(args)
    d = cfg.to_dict()
    d["key"] = _mask(d.get("key"))
    if args.as_json:
        _print_json(d)
    else:
        _print_table(["field", "value"], [[k, str(v)] for k, v in d.items()], title="Local waiter config")
    return 0


# ---------------------------------------------------------------------------
# Beta 2 subcommands — servers, modules, jobs, events, billing, route
# ---------------------------------------------------------------------------


def _cmd_route(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter route", description="Ask the T1 API's routing engine which model to use.")
    p.add_argument("--plan", required=True, help="Your plan (e.g. free, paired)")
    p.add_argument("--model", dest="model_id", default=None, help="Manual selection instead of automatic routing")
    p.add_argument("--input-tokens", type=int, default=0)
    p.add_argument("--auto-fallback", action="store_true", help="Fall through the cascade if --model is exhausted")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, _ = _client_for(args)
    payload = client.route(
        plan=args.plan, model_id=args.model_id, input_tokens=args.input_tokens, automatic_fallback=args.auto_fallback
    )
    if args.as_json:
        _print_json(payload)
        return 0
    _ok(f"Routed to {payload['model_id']} ({payload['reason']})")
    return 0


def _cmd_servers(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter servers", description="List / register servers.")
    p.add_argument("--register", metavar="NAME", default=None, help="Register a new server with this name")
    p.add_argument("--address", default=None, help="Required with --register")
    p.add_argument("--allow-private", action="store_true", help="Allow a private/Tailscale/localhost address")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, _ = _client_for(args)
    if args.register:
        if not args.address:
            _err("--register requires --address")
            return 1
        payload = client.register_server(name=args.register, address=args.address, allow_private_address=args.allow_private)
        if args.as_json:
            _print_json(payload)
        else:
            s = payload["server"]
            _ok(f"Registered {s['name']} ({s['server_id'][:8]}…) — trust_level={s['trust_level']}")
        return 0
    payload = client.list_servers()
    if args.as_json:
        _print_json(payload)
        return 0
    rows = [[s["server_id"][:8] + "…", s["name"], s["address"], s["trust_level"], s["status"]] for s in payload["servers"]]
    _print_table(["server_id", "name", "address", "trust", "status"], rows, title=f"Servers ({payload['count']})")
    return 0


def _cmd_modules(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter modules", description="List modules, or create/upload/sync one.")
    p.add_argument("--create", metavar="NAME", default=None, help="Create a new module with this name")
    p.add_argument("--version", default="1.0.0", help="Version for --create (default 1.0.0)")
    p.add_argument("--upload", metavar="MODULE_ID", default=None, help="Upload a local file to this module_id")
    p.add_argument("--file", default=None, help="Local file path for --upload")
    p.add_argument("--sync", metavar="MODULE_ID", default=None, help="Sync this module_id to --server-id")
    p.add_argument("--server-id", default=None, help="Target server_id for --sync")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, _ = _client_for(args)

    if args.create:
        payload = client.create_module(name=args.create, version=args.version)
        m = payload["module"]
        _ok(f"Created {m['name']}@{m['version']} ({m['module_id'][:8]}…) status={m['status']}")
        return 0
    if args.upload:
        if not args.file:
            _err("--upload requires --file")
            return 1
        payload = client.upload_module_local(args.upload, args.file)
        m = payload["module"]
        _ok(f"Uploaded {m['size_bytes']} bytes, checksum={m['checksum'][:12]}…, status={m['status']}")
        return 0
    if args.sync:
        if not args.server_id:
            _err("--sync requires --server-id")
            return 1
        payload = client.sync_module(args.sync, args.server_id)
        _ok(f"Queued module_sync job {payload['job_id'][:8]}… (status={payload['status']})")
        _info("  check progress with: waiter jobs get " + payload["job_id"])
        return 0

    payload = client.list_modules()
    if args.as_json:
        _print_json(payload)
        return 0
    rows = [
        [m["module_id"][:8] + "…", f"{m['name']}@{m['version']}", m["status"], str(m["size_bytes"] or "-")]
        for m in payload["modules"]
    ]
    _print_table(["module_id", "name@version", "status", "size_bytes"], rows, title=f"Modules ({payload['count']})")
    return 0


def _cmd_jobs(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter jobs", description="Get or cancel a job by ID.")
    p.add_argument("action", choices=["get", "cancel"], help="get: show status/result. cancel: request cancellation.")
    p.add_argument("job_id")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, _ = _client_for(args)
    payload = client.cancel_job(args.job_id) if args.action == "cancel" else client.get_job(args.job_id)
    if args.as_json:
        _print_json(payload)
        return 0
    j = payload["job"]
    rows = [[k, str(v)] for k, v in j.items()]
    _print_table(["field", "value"], rows, title=f"Job {j['job_id'][:8]}…")
    return 0


def _cmd_events(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter events", description="Poll recent events.")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--since", dest="since_id", default=None, help="Only events after this event_id")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, _ = _client_for(args)
    payload = client.list_events(limit=args.limit, since_id=args.since_id)
    if args.as_json:
        _print_json(payload)
        return 0
    rows = [[e["type"], e["source"], str(e["data"])[:60]] for e in payload["events"]]
    _print_table(["type", "source", "data"], rows, title=f"Events ({payload['count']})")
    return 0


def _cmd_billing(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter billing", description="Show balance/transactions, or redeem a payment token.")
    p.add_argument("--redeem", metavar="TOKEN", default=None, help="Redeem a payment token into your account")
    p.add_argument("--transactions", action="store_true", help="Show transaction history instead of just balance")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, _ = _client_for(args)
    if args.redeem:
        payload = client.redeem_payment_token(args.redeem)
        _ok(f"Redeemed — new balance: {payload['balance']} ({payload['account_type']}:{payload['account_id']})")
        return 0
    if args.transactions:
        payload = client.billing_transactions()
        if args.as_json:
            _print_json(payload)
            return 0
        rows = [[t["transaction_id"], t["kind"], f"{t['amount']:+.2f}", f"{t['balance_after']:.2f}", t["note"]] for t in payload["transactions"]]
        _print_table(["txn_id", "kind", "amount", "balance_after", "note"], rows, title="Transactions")
        return 0
    payload = client.billing_balance()
    if args.as_json:
        _print_json(payload)
        return 0
    _info(f"Balance: {payload['balance']} ({payload['account_type']}:{payload['account_id']})")
    return 0


# ---------------------------------------------------------------------------
# Beta 3 subcommands — keys, audit, security, cost, deploy, tui, doctor, smoke
# ---------------------------------------------------------------------------


def _cmd_tui(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter tui", description="Open the full curses dashboard (same as `waiter serv -G`).")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    return _launch_tui(_resolve_config(args))


def _cmd_keys(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter keys", description="List keys, or assign a plan/account/models to one.")
    p.add_argument("--assign", metavar="KEY_ID", default=None, help="Assign settings to this key_id (admin)")
    p.add_argument("--plan", default=None, help="Plan to assign with --assign")
    p.add_argument("--account", dest="account_id", default=None)
    p.add_argument("--user", dest="user_id", default=None)
    p.add_argument("--models", default=None, help="Comma-separated model_ids to narrow this key to")
    p.add_argument("--servers", default=None, help="Comma-separated server_ids to bind this key to")
    p.add_argument("--import-file", dest="import_file", default=None, help="Import a Keymaster export JSON file (admin)")
    p.add_argument("--include-inactive", action="store_true")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, _ = _client_for(args)

    if args.import_file:
        with open(args.import_file, encoding="utf-8") as handle:
            payload = json.load(handle)
        result = client.import_keys(payload)
        _ok(f"Imported {result['imported']} key(s), skipped {result['skipped']} duplicate(s).")
        return 0

    if args.assign:
        fields: dict[str, Any] = {}
        if args.plan is not None:
            fields["plan"] = args.plan
        if args.account_id is not None:
            fields["account_id"] = args.account_id
        if args.user_id is not None:
            fields["user_id"] = args.user_id
        if args.models is not None:
            fields["allowed_models"] = [m.strip() for m in args.models.split(",") if m.strip()]
        if args.servers is not None:
            fields["server_ids"] = [s.strip() for s in args.servers.split(",") if s.strip()]
        if not fields:
            _err("--assign needs at least one of --plan/--account/--user/--models/--servers.")
            return 1
        assignment = client.assign_key(args.assign, **fields)["assignment"]
        _ok(f"Assigned {assignment['key_id']} → plan '{assignment['plan']}'")
        if assignment["allowed_models"]:
            _info(f"  restricted to: {', '.join(assignment['allowed_models'])}")
        return 0

    keys = client.list_keys(include_inactive=args.include_inactive)
    if args.as_json:
        _print_json([k.raw for k in keys])
        return 0
    rows = [
        [
            k.key_id,
            k.key_type,
            ",".join(k.scopes),
            k.plan or "-",
            "yes" if k.active else "no",
            str(k.request_count),
        ]
        for k in keys
    ]
    _print_table(["key_id", "type", "scopes", "plan", "active", "requests"], rows, title=f"Keys ({len(keys)})")
    if not keys:
        _info("No keys visible. A non-admin key only ever sees itself.")
    return 0


def _cmd_audit(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter audit", description="Read the server's audit trail (admin only).")
    p.add_argument("--category", default=None, help="admin | security | write | billing")
    p.add_argument("--action", default=None, help="e.g. keys.assign")
    p.add_argument("--outcome", default=None, help="success | denied | failure")
    p.add_argument("--limit", type=int, default=50)
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, _ = _client_for(args)
    payload = client.audit_events(
        category=args.category, action=args.action, outcome=args.outcome, limit=args.limit
    )
    if args.as_json:
        _print_json(payload)
        return 0
    rows = [
        [
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["ts"])),
            e["category"],
            e["action"],
            e["outcome"],
            e["actor_key_id"] or "-",
            e["client_ip"] or "-",
        ]
        for e in payload["events"]
    ]
    _print_table(
        ["when", "category", "action", "outcome", "actor", "ip"],
        rows,
        title=f"Audit ({payload['count']} of {payload['total']})",
    )
    return 0


def _cmd_security(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter security", description="Inspect and edit the server's network policy and forced limits.")
    p.add_argument("--block", metavar="CIDR", default=None, help="Blacklist an IP/CIDR")
    p.add_argument("--allow", metavar="CIDR", default=None, help="Allowlist an IP/CIDR")
    p.add_argument("--appeal", metavar="CIDR", default=None, help="Remove an allow/block entry")
    p.add_argument("--reason", default="", help="Reason recorded with --block/--allow")
    p.add_argument("--allow-unlisted", dest="allow_unlisted", choices=["on", "off"], default=None,
                   help="Whether clients on neither list may connect")
    p.add_argument("--limits", action="store_true", help="Show forced limits instead of the IP lists")
    p.add_argument("--my-rate-limits", action="store_true", help="Show your own remaining rate-limit budget")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, _ = _client_for(args)

    if args.my_rate_limits:
        payload = client.rate_limit_status()
        rows = [[r["rule"], f"{r['remaining']:.0f}/{r['capacity']:.0f}", f"{r['refill_per_second']}/s"] for r in payload["rules"]]
        _print_table(["rule", "remaining", "refill"], rows, title=f"Your rate limits (enabled={payload['enabled']})")
        return 0

    if args.block:
        client.blacklist_ip(args.block, reason=args.reason)
        _ok(f"Blocked {args.block}")
    if args.allow:
        client.whitelist_ip(args.allow, reason=args.reason)
        _ok(f"Allowlisted {args.allow}")
    if args.appeal:
        client.appeal_ip(args.appeal)
        _ok(f"Removed {args.appeal} from the server's lists")
    if args.allow_unlisted is not None:
        client.set_allow_unlisted(args.allow_unlisted == "on")
        _ok(f"Unlisted clients may now connect: {args.allow_unlisted == 'on'}")

    if args.limits:
        limits = client.list_forced_limits()
        rows = [
            [
                f"{limit['subject_type']}:{limit['subject_id']}",
                str(limit["requests_per_window"] or "-"),
                str(limit["tokens_per_window"] or "-"),
                f"{limit['window_seconds']:g}s",
                limit["reason"],
            ]
            for limit in limits
        ]
        _print_table(["subject", "requests", "tokens", "window", "reason"], rows, title=f"Forced limits ({len(limits)})")
        return 0

    payload = client.network_policy()
    if args.as_json:
        _print_json(payload)
        return 0
    rows = [[e["cidr"], e["kind"], e["reason"], "expired" if e["expired"] else "active"] for e in payload["entries"]]
    _print_table(["cidr", "kind", "reason", "state"], rows, title=f"Network policy ({payload['count']})")
    _info(
        f"  unlisted clients may connect: {payload['allow_unlisted_clients']} "
        f"(policy enforcement enabled: {payload['enabled']})"
    )
    return 0


def _cmd_cost(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter cost", description="Show spend, breakdowns, and a forecast.")
    p.add_argument("--group-by", dest="group_by", default="model_id",
                   help="model_id | key_id | server_id | module_id | user_id | account_id | endpoint")
    p.add_argument("--range", dest="range_", default="30d", help="1h | 24h | 7d | 30d | all")
    p.add_argument("--forecast", action="store_true", help="Include a spend forecast")
    p.add_argument("--estimate-model", dest="estimate_model", default=None, help="Estimate a prospective call instead")
    p.add_argument("--input-tokens", dest="input_tokens", type=int, default=0)
    p.add_argument("--output-tokens", dest="output_tokens", type=int, default=None)
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, _ = _client_for(args)

    if args.estimate_model:
        estimate = client.estimate_cost(
            model_id=args.estimate_model,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
        )
        if args.as_json:
            _print_json(estimate)
            return 0
        _print_table(
            ["field", "value"],
            [[k, str(v)] for k, v in estimate.items() if k != "request_id"],
            title=f"Estimate — {args.estimate_model}",
        )
        return 0

    report = client.usage_cost(group_by=args.group_by, range=args.range_, forecast=args.forecast)
    if args.as_json:
        _print_json(report.raw)
        return 0
    rows = [
        [line["value"], str(line["requests"]), str(line["total_tokens"]), f"{line['total_cost']:.6f}"]
        for line in report.lines
    ]
    _print_table(
        [args.group_by, "requests", "tokens", "cost"],
        rows,
        title=f"Cost — {report.total_cost:.6f} {report.currency} over {args.range_}",
    )
    if report.unpriced_models:
        _warn(
            "Usage recorded against models no longer in the registry (priced at 0): "
            + ", ".join(report.unpriced_models)
        )
    if report.forecast:
        f = report.forecast
        _info(
            f"  Forecast: {f['projected_cost']:.6f} {f['currency']} over "
            f"{f['horizon_seconds'] / 86400:.0f}d — {f['confidence']} confidence. {f['basis']}"
        )
    return 0


def _cmd_deploy(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter deploy", description="Push a module to one or more trusted servers and watch the job.")
    p.add_argument("module_id")
    p.add_argument("--to", dest="server_ids", required=True, help="Comma-separated server_ids")
    p.add_argument("--wait", action="store_true", help="Block until the deployment job settles")
    p.add_argument("--timeout", type=float, default=300.0)
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, _ = _client_for(args)
    server_ids = [s.strip() for s in args.server_ids.split(",") if s.strip()]

    queued = client.deploy_module(args.module_id, server_ids)
    _ok(f"Queued deployment job {queued['job_id'][:8]}… to {len(server_ids)} server(s)")
    if not args.wait:
        _info(f"  watch it with: waiter jobs get {queued['job_id']}")
        return 0

    job = client.wait_for_job(queued["job_id"], timeout=args.timeout)
    if not job.succeeded:
        _err(f"Deployment {job.status}: {job.error or 'no error recorded'}")
        for failure in (job.result or {}).get("failed", []):
            _info(f"  {failure['server_id']}: {failure['error_code']} — {failure['message']}")
        return 1
    delivered = (job.result or {}).get("delivered", [])
    _ok(f"Deployed to {len(delivered)} server(s)")
    for item in delivered:
        _info(f"  {item['server_id']}: {item['bytes_transferred']} bytes, sha256={item['checksum'][:12]}…")
    return 0


def _cmd_doctor(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="waiter doctor", description="Check a server's configuration and report anything unsafe for production.")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, cfg = _client_for(args)
    # waiter's T1Client overrides status() to return the raw envelope,
    # because that is what the CLI's table renderers consume. Doctor wants
    # the typed view, so it builds one rather than reaching past the
    # override — which is what it used to do, and it crashed with
    # "'dict' object has no attribute 'environment'" the first time it ran
    # against a real server.
    from ..t1sdk.models import ServerStatus

    raw = client.status()
    status = ServerStatus.from_dict(raw)

    if args.as_json:
        _print_json(raw)
        return 0

    _print_table(
        ["check", "value"],
        [
            ["environment", status.environment],
            ["beta", status.beta],
            ["t1 api version", status.t1_api_version],
            ["hypernix version", status.hypernix_version],
            ["storage backend", status.storage_backend],
            ["TLS", str(status.tls_enabled)],
            ["mTLS mode", status.mtls_mode],
            ["rate limiting", str(status.rate_limit_enabled)],
            ["audit logging", str(status.audit_enabled)],
            ["network policy", str(status.network_policy_enabled)],
            ["unlisted clients", str(status.allow_unlisted_clients)],
            ["remote deployment", str(status.remote_deployment_enabled)],
            ["production ready", str(status.production_ready)],
        ],
        title=f"Server health — {cfg.server}",
    )
    secrets = status.secrets_configured
    if secrets:
        _print_table(
            ["secret", "configured"],
            [[name, "yes" if value else "NO"] for name, value in secrets.items()],
            title="Secrets (set/unset only — values are never exposed)",
        )
    if status.production_warnings:
        _warn(f"{len(status.production_warnings)} configuration warning(s):")
        for warning in status.production_warnings:
            _info(f"  • {warning}")
        return 0 if status.environment != "production" else 1
    _ok("No configuration warnings reported.")
    return 0


def _cmd_smoke(rest: list[str]) -> int:
    """CLI smoke-testing tool (spec deliverable #11)."""
    from .smoke import run_smoke_tests

    p = argparse.ArgumentParser(prog="waiter smoke", description="Run read-only smoke tests against a T1 API server.")
    p.add_argument("--write", action="store_true", help="Also run write tests (creates and deletes a scratch module)")
    p.add_argument("--fail-fast", action="store_true", help="Stop at the first failure")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, cfg = _client_for(args)
    return run_smoke_tests(
        client,
        base_url=cfg.server or "",
        include_write=args.write,
        fail_fast=args.fail_fast,
        as_json=args.as_json,
    )


# ---------------------------------------------------------------------------
# T1 v1.0.26.8.0.1 — `waiter lmstudio`, `waiter hyperlink`, `waiter fetch`
# ---------------------------------------------------------------------------

_LMSTUDIO_HELP = """\
waiter lmstudio — the bridge to a model loaded in LM Studio.

    waiter lmstudio status              is it reachable, is anything loaded
    waiter lmstudio status --discover   also sweep localhost + the tailnet (admin)
    waiter lmstudio models              what LM Studio has, loaded ones marked
    waiter lmstudio chat "your prompt"  one completion through the bridge
    waiter lmstudio local               probe from *this* machine, no T1 server

`status`, `models` and `chat` ask the T1 API, which asks LM Studio — so
LM Studio only has to be reachable from the server, not from here.
`local` is the exception: it probes directly from this machine, which is
what you want when working out why the server cannot see it.
"""


def _cmd_lmstudio(rest: list[str]) -> int:
    if not rest or rest[0] in ("-h", "--help"):
        print(_LMSTUDIO_HELP)
        return 0
    sub, rest = rest[0], rest[1:]

    if sub == "local":
        return _lmstudio_local(rest)

    p = argparse.ArgumentParser(prog=f"waiter lmstudio {sub}")
    if sub == "status":
        p.add_argument("--discover", action="store_true", help="Sweep localhost and the tailnet (admin)")
    if sub == "chat":
        p.add_argument("prompt", nargs="+")
        p.add_argument("--model", default=None)
        p.add_argument("--system", default="", help="System prompt")
        p.add_argument("--temperature", type=float, default=None)
        p.add_argument("--max-tokens", dest="max_tokens", type=int, default=None)
    if sub in ("models", "chat"):
        p.add_argument("--base-url", dest="base_url", default=None, help="Admin-only override")
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, cfg = _client_for(args)

    if sub == "status":
        data = client.lmstudio_status(discover=args.discover)
        if args.as_json:
            _print_json(data)
            return 0
        if not data.get("enabled"):
            _err("The LM Studio bridge is disabled on this server (T1_LMSTUDIO_ENABLED=0).")
            return 1
        probe = data.get("probe")
        if not probe:
            _warn(f"No LM Studio address configured on {cfg.server}. Set T1_LMSTUDIO_URL there.")
        else:
            _print_table(
                ["check", "value"],
                [
                    ["address", probe["base_url"]],
                    ["reachable", str(probe["reachable"])],
                    ["models", str(probe["model_count"])],
                    ["loaded", str(probe["loaded_count"])],
                    ["native API", str(probe["native_api"])],
                    ["CORS", _cors_text(probe)],
                    ["latency", f"{probe['latency_ms']:.0f} ms"],
                ],
                title="LM Studio bridge",
            )
            if not probe["reachable"]:
                _err(probe.get("error") or "unreachable")
                return 1
            if not probe["loaded_count"]:
                _warn("LM Studio is answering but has nothing loaded — load a model in its UI.")
        for other in data.get("discovered", []):
            mark = "✓" if other["usable"] else ("·" if other["reachable"] else "×")
            _info(f"  {mark} {other['base_url']}  {other['loaded_count']}/{other['model_count']} loaded")
        return 0

    if sub == "models":
        data = client.lmstudio_models(base_url=args.base_url)
        if args.as_json:
            _print_json(data)
            return 0
        rows = [
            [
                m["model_id"],
                "yes" if m["loaded"] else "no",
                m["kind"],
                m["quantization"] or "-",
                str(m["max_context_length"] or m["context_length"] or "-"),
            ]
            for m in data["models"]
        ]
        _print_table(
            ["model", "loaded", "kind", "quant", "context"],
            rows,
            title=f"LM Studio at {data['base_url']} — {data['loaded_count']}/{data['count']} loaded",
        )
        return 0

    if sub == "chat":
        messages: list[dict[str, Any]] = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": " ".join(args.prompt)})
        data = client.lmstudio_chat(
            messages,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
        )
        if args.as_json:
            _print_json(data)
            return 0
        _info(data["content"])
        _info(
            f"\n[dim]{data['model']} via {data['base_url']} — "
            f"{data['input_tokens']} in / {data['output_tokens']} out[/dim]"
            if _HAS_RICH
            else f"\n{data['model']} via {data['base_url']} — "
            f"{data['input_tokens']} in / {data['output_tokens']} out"
        )
        return 0

    _err(f"Unknown lmstudio subcommand {sub!r}")
    print(_LMSTUDIO_HELP, file=sys.stderr)
    return 1


def _cors_text(probe: dict[str, Any]) -> str:
    """CORS state as something an operator can act on.

    Only matters for a browser or WKWebView talking to LM Studio
    directly; a Python client is not subject to it. Saying so inline
    saves the "do I need to turn this on?" round trip.
    """
    enabled = probe.get("cors_enabled")
    if enabled is None:
        return "not tested"
    if enabled:
        return f"on (allow-origin: {probe.get('cors_allow_origin') or '*'})"
    return "off — fine for this bridge, needed only for browser clients"


def _lmstudio_local(rest: list[str]) -> int:
    """Probe LM Studio from *this* machine, with no T1 server involved."""
    from ..bridge.lmstudio import LMStudioBridge, default_endpoints, discover

    p = argparse.ArgumentParser(prog="waiter lmstudio local")
    p.add_argument("--url", default=None, help="Probe just this address")
    p.add_argument("--timeout", type=float, default=1.5)
    p.add_argument("--json", dest="as_json", action="store_true")
    args = p.parse_args(rest)

    if args.url:
        probes = [LMStudioBridge(args.url, connect_timeout=args.timeout).probe()]
    else:
        probes = discover(connect_timeout=args.timeout)
    if args.as_json:
        _print_json([pr.to_dict() for pr in probes])
        return 0
    if not probes:
        _warn("Nowhere to probe. Set HYPERNIX_LMSTUDIO_URL or pass --url.")
        return 1
    _print_table(
        ["address", "reachable", "models", "loaded", "latency"],
        [
            [
                pr.base_url,
                "yes" if pr.reachable else "no",
                str(pr.model_count),
                str(pr.loaded_count),
                f"{pr.latency_ms:.0f} ms",
            ]
            for pr in probes
        ],
        title="LM Studio, probed from this machine",
    )
    usable = [pr for pr in probes if pr.usable]
    if usable:
        _ok(f"Use it with:  T1_LMSTUDIO_URL={usable[0].base_url}")
        return 0
    reachable = [pr for pr in probes if pr.reachable]
    if reachable:
        _warn(f"{reachable[0].base_url} answers but has nothing loaded — load a model in LM Studio.")
        return 1
    _err("No LM Studio server found. Checked: " + ", ".join(default_endpoints()))
    return 1


_HYPERLINK_HELP = """\
waiter hyperlink — pair phones, and manage what they can see.

    waiter hyperlink pair [--label "Mason iPhone"]   mint a pairing code
    waiter hyperlink pair --qr                        also print the QR payload
    waiter hyperlink devices [--all]                  list paired devices
    waiter hyperlink unpair <device_id>               revoke one device
    waiter hyperlink endpoints                        addresses this server answers on
    waiter hyperlink sessions                         chat sessions on the server
    waiter hyperlink chat <session_id> "message"      send a turn from here

`pair` requires an admin key: a device token is deliberately unable to
enrol another device.
"""


def _cmd_hyperlink(rest: list[str]) -> int:
    if not rest or rest[0] in ("-h", "--help"):
        print(_HYPERLINK_HELP)
        return 0
    sub, rest = rest[0], rest[1:]

    p = argparse.ArgumentParser(prog=f"waiter hyperlink {sub}")
    if sub == "pair":
        p.add_argument("--label", default="", help="A note for your own records")
        p.add_argument("--ttl", type=float, default=None, help="Seconds the code stays valid")
        p.add_argument("--qr", action="store_true", help="Print the QR payload JSON as well")
    if sub == "devices":
        p.add_argument("--all", dest="include_revoked", action="store_true")
    if sub == "unpair":
        p.add_argument("device_id")
    if sub == "sessions":
        p.add_argument("--all", dest="include_archived", action="store_true")
    if sub == "chat":
        p.add_argument("session_id")
        p.add_argument("message", nargs="+")
        p.add_argument("--model", default=None)
    _add_common_connection_args(p)
    args = p.parse_args(rest)
    client, cfg = _client_for(args)

    if sub == "pair":
        data = client.hyperlink_pair(label=args.label, ttl_seconds=args.ttl)
        if args.as_json:
            _print_json(data)
            return 0
        code = data["code"]
        # Spaced in the middle because that is how it will be read aloud
        # and typed: three and three, not six.
        pretty = f"{code[:3]} {code[3:]}"
        _ok(f"Pairing code:  {pretty}")
        _info(f"  valid for {data['seconds_remaining'] / 60:.0f} minutes, single use")
        _print_table(
            ["address", "kind", "note"],
            [[e["url"], e["kind"], e["note"]] for e in data.get("endpoints", [])],
            title="Enter one of these in the app, then the code",
        )
        if not any(e["kind"].startswith("tailscale") for e in data.get("endpoints", [])):
            _warn(
                "No Tailscale address on this server — the app will only work on the same "
                "network. Install Tailscale on both machines to use it from anywhere."
            )
        if args.qr:
            _info("\nQR payload:")
            _print_json(data["qr_payload"])
        return 0

    if sub == "devices":
        data = client.hyperlink_devices(include_revoked=args.include_revoked)
        if args.as_json:
            _print_json(data)
            return 0
        rows = [
            [
                d["device_id"][:12] + "…",
                d["name"],
                d["platform"],
                _ago(d.get("last_seen")),
                d.get("last_address") or "-",
                "revoked" if d["revoked"] else "active",
            ]
            for d in data["devices"]
        ]
        _print_table(
            ["device_id", "name", "platform", "last seen", "address", "state"],
            rows,
            title=f"Paired devices ({data['count']})",
        )
        return 0

    if sub == "unpair":
        data = client.hyperlink_revoke_device(args.device_id)
        _ok(f"Unpaired {data['device']['name']} — its token stops working immediately.")
        return 0

    if sub == "endpoints":
        data = client.hyperlink_endpoints()
        if args.as_json:
            _print_json(data)
            return 0
        _print_table(
            ["address", "kind", "note"],
            [[e["url"], e["kind"], e["note"]] for e in data["endpoints"]],
            title=f"{data['server_name']} — t1 v{data['t1_version']}",
        )
        if not data["reachable_off_lan"]:
            _warn("Nothing here is reachable off the LAN. Install Tailscale, or set T1_HYPERLINK_PUBLIC_URL.")
        return 0

    if sub == "sessions":
        data = client.hyperlink_sessions(include_archived=args.include_archived)
        if args.as_json:
            _print_json(data)
            return 0
        _print_table(
            ["session_id", "title", "model", "messages", "updated"],
            [
                [
                    s["session_id"],
                    s["title"],
                    s["model_id"] or "-",
                    str(s["message_count"]),
                    _ago(s["updated_at"]),
                ]
                for s in data["sessions"]
            ],
            title=f"Chat sessions ({data['count']})",
        )
        return 0

    if sub == "chat":
        data = client.hyperlink_chat(
            args.session_id, " ".join(args.message), model_id=args.model
        )
        if args.as_json:
            _print_json(data)
            return 0
        _info(data["assistant_message"]["content"])
        return 0

    _err(f"Unknown hyperlink subcommand {sub!r}")
    print(_HYPERLINK_HELP, file=sys.stderr)
    return 1


def _ago(timestamp: float | None) -> str:
    """``3m ago`` / ``2d ago`` / ``never``."""
    if not timestamp:
        return "never"
    delta = max(0.0, time.time() - float(timestamp))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= size:
            return f"{delta / size:.0f}{unit} ago"
    return f"{delta:.0f}s ago"


def _cmd_fetch(rest: list[str]) -> int:
    """`waiter fetch` — resolve a Hugging Face link into a download plan."""
    p = argparse.ArgumentParser(
        prog="waiter fetch",
        description=(
            "Turn a Hugging Face model page and/or a direct download link into a complete, "
            "runnable GGUF download plan — split parts and vision projectors included."
        ),
        epilog=(
            "Examples:\n"
            "  waiter fetch https://huggingface.co/bartowski/Qwen3-8B-GGUF\n"
            "  waiter fetch --page <model page> --file <download-arrow link>\n"
            "  waiter fetch bartowski/Qwen3-8B-GGUF:Q5_K_M\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("link", nargs="?", default="", help="A page link, a file link, or owner/repo[:QUANT]")
    p.add_argument("--page", default="", help="Model page URL")
    p.add_argument("--file", dest="file_url", default="", help="Direct file (download-arrow) URL")
    p.add_argument(
        "--prefer",
        choices=["strict", "file", "page"],
        default="strict",
        help="Which link wins if the two name different repositories",
    )
    p.add_argument("--no-vision", dest="include_vision", action="store_false")
    p.add_argument("--offline", action="store_true", help="Resolve from the links alone")
    p.add_argument("--local", action="store_true", help="Resolve here instead of asking the server")
    _add_common_connection_args(p)
    args = p.parse_args(rest)

    page, file_url = args.page, args.file_url
    if args.link:
        # A single positional link is a page or a file; work out which
        # rather than making the user remember two flag names.
        from ..hyperlink.hfmerge import parse_link

        try:
            ref = parse_link(args.link)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            _err(str(exc))
            return 1
        if ref.has_file:
            file_url = file_url or args.link
        else:
            page = page or args.link
    if not page and not file_url:
        _err("Give a link: a model page, a file link, or both (--page/--file).")
        return 1

    if args.local:
        from ..hyperlink.hfmerge import HFResolveError
        from ..hyperlink.hfmerge import resolve as hf_resolve

        try:
            data = hf_resolve(
                page,
                file_url,
                prefer=args.prefer,
                include_vision=args.include_vision,
                offline=args.offline,
            ).to_dict()
        except HFResolveError as exc:
            _err(str(exc))
            if exc.hint:
                _info(f"  {exc.hint}")
            return 1
    else:
        client, _ = _client_for(args)
        data = client.hyperlink_resolve_model(
            page_url=page,
            file_url=file_url,
            prefer=args.prefer,
            include_vision=args.include_vision,
            offline=args.offline,
        )

    if args.as_json:
        _print_json(data)
        return 0

    _print_table(
        ["field", "value"],
        [
            ["repo", data["repo_id"]],
            ["revision", data["revision"]],
            ["quantisation", data["quantization"] or "unknown"],
            ["primary file", data["primary_file"]],
            ["files", str(data["file_count"])],
            ["total size", data["total_size_human"]],
            ["split model", "yes" if data["is_split"] else "no"],
            ["vision projector", "yes" if data["has_vision"] else "no"],
            ["licence", data["license"] or "-"],
            ["gated", "yes" if data["gated"] else "no"],
        ],
        title="Resolved model",
    )
    _print_table(
        ["file", "role", "size"],
        [[f["filename"], f["role"], _human_bytes(f["size_bytes"])] for f in data["files"]],
        title="Download plan",
    )
    for warning in data.get("warnings", []):
        _warn(warning)
    return 0


def _human_bytes(size: int) -> str:
    value = float(size or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


# ---------------------------------------------------------------------------
# 0.72.1 — `waiter -F` (find)
# ---------------------------------------------------------------------------

_FIND_HELP = """\
waiter -F <target> — find a HyperNix server.

    waiter -F "workshop-box"          by server name
    waiter -F <54-char host id>       by Host ID
    waiter -F home/api.jsonl          a direct endpoint descriptor
    waiter -F 192.168.1.50:8000       an address
    waiter -F -l "workshop-box"       this machine and this LAN only

Without -l the tailnet is searched too. With it, never — a tailnet sweep
touches every peer on a private network, and "find my server" should not
do that by surprise when the server is on the desk.

After a match, --open launches hyped-pro against it. A server may ask
clients to run something else; waiter reports what it asked for and does
not run it. See `waiter -F --help` for why.
"""


def _cmd_find(rest: list[str]) -> int:
    from .discovery import classify_target, connect, discover

    p = argparse.ArgumentParser(
        prog="waiter -F",
        description="Find a HyperNix server by name, Host ID, api.jsonl endpoint, or address.",
        epilog=(
            "Security note: api.jsonl may name a client application the host would like "
            "opened. waiter reports it and never executes it — running a command chosen by "
            "the machine you are connecting to is remote code execution with extra steps. "
            "--open always launches HyperNix's own hyped-pro, never the host's command."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("target", help="Server name, Host ID, api.jsonl endpoint, or address")
    p.add_argument("-l", "--local", action="store_true",
                   help="Search only this machine and this LAN; never the tailnet")
    p.add_argument("--open", action="store_true",
                   help="Open hyped-pro against the first match")
    p.add_argument("--timeout", type=float, default=2.0)
    # Not --port: the shared connection flags already own -P/--port for
    # "the port of the server I am talking to", and this is "another
    # port to look on". Two flags with one name is how a CLI teaches
    # people to distrust it.
    p.add_argument("--probe-port", type=int, action="append", dest="ports",
                   help="Extra port to probe (repeatable)")
    _add_common_connection_args(p)
    args = p.parse_args(rest)

    target = classify_target(args.target)
    if not args.as_json:
        scope = "this machine and LAN" if args.local else "this machine, LAN and tailnet"
        _info(f"Looking for {target.kind.replace('_', ' ')} {args.target!r} across {scope}…")

    from .discovery import DEFAULT_PORTS

    ports = tuple(args.ports) + DEFAULT_PORTS if args.ports else DEFAULT_PORTS
    found = discover(target, local_only=args.local, ports=ports, timeout=args.timeout)
    reachable = [f for f in found if f.reachable]

    if args.as_json:
        _print_json([f.to_dict() for f in found])
        return 0 if reachable else 1

    if not reachable:
        _err(f"No HyperNix server found for {args.target!r}.")
        if args.local:
            _info("  Searched this machine and LAN only (-l). Drop -l to include the tailnet.")
        else:
            _info("  Searched this machine, the LAN, and the tailnet.")
        _info("  If you know the address, pass it directly: waiter -F 192.168.1.50:8000")
        return 1

    _print_table(
        ["address", "name", "t1", "where", "latency"],
        [
            [f.url, f.server_name or "-", f.t1_version or "-", f.source, f"{f.latency_ms:.0f} ms"]
            for f in reachable
        ],
        title=f"Found {len(reachable)} server(s)",
    )

    first = reachable[0]
    connection = connect(first, credential=_resolve_config(args).key or "")
    for note in connection.notes:
        _warn(note) if "NOT been run" in note else _info(f"  {note}")
    if connection.authenticated:
        _ok(f"Authenticated as {connection.key_id or 'a valid key'}")

    if args.open:
        return _open_hyped_pro(connection)
    if connection.application and connection.application.is_builtin:
        _info(f"\n  Open it with: waiter -F {args.target!r} --open")
    return 0


def _open_hyped_pro(connection: Any) -> int:
    """Launch hyped-pro against a discovered server.

    Always HyperNix's own binary with HyperNix's own flags — never a
    command the remote host supplied. That is the whole distinction this
    function exists to keep: discovery hands over an address, and the
    thing that runs is ours.
    """
    import shutil as _shutil
    import subprocess as _subprocess

    argv = connection.hyped_pro_argv()
    if _shutil.which(argv[0]) is None:
        _err(f"{argv[0]} is not on PATH. Install it with: pip install 'hypernix[t1api]'")
        return 1
    _ok(f"Opening {argv[0]} against {connection.server.url}")
    try:
        completed = _subprocess.run(argv, check=False)  # noqa: S603 - our own argv list
    except OSError as exc:
        _err(f"Could not start {argv[0]}: {exc}")
        return 1
    return completed.returncode


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# `waiter version` and `waiter help`
# ---------------------------------------------------------------------------

#: Longer help, by topic. Separate from the one-line usage table because
#: the questions people actually get stuck on ("which address?", "which
#: key?") need paragraphs, and putting paragraphs in the usage table makes
#: the table useless for the people who just want the subcommand name.
_HELP_TOPICS: dict[str, str] = {
    "connect": """\
waiter help connect — pointing waiter at a server

    waiter serv -A -I http://127.0.0.1:8000 -K <key>     validate and save
    waiter serv -A -I http://100.x.y.z:8000 -K <key> -L  a tailnet server
    waiter config                                        what is saved now
    waiter health                                        is it up?

The T1 API listens on 8000 by default. Two ports people reach for by
mistake are LM Studio's 1234 and Ollama's 11434 — those are model
backends, which the *server* talks to; waiter talks to the server. If you
point waiter at one directly you get a connection refused, or a puzzling
404 from something that is not a T1 API.

-I is remembered. A command failing against an address you do not
recognise usually means an -I from a while ago is still saved; `waiter
config` shows it and the failure message names the file.
""",
    "keys": """\
waiter help keys — which credential to use

    gkey create --type admin --scopes admin,read,write   mint one
    gkey create -v v2 --level 5                          a v2 spelling
    gkey version                                         formats this build mints
    waiter whoami                                        what is my key allowed to do?

-K takes either a raw key (T1_…, T2_…, T2S_…) or a scoped token (T1S.…).
A v2 key is a spelling of a v1 key rather than a separate credential, so
either form of the same key works against the same server.

Admin-only subcommands — `security`, `audit`, parts of `keys` and
`servers` — need a key whose *store record* is an admin. A key's access
level is not the same thing: a level-9 key with no admin record is a very
privileged user key.
""",
    "hyperlink": """\
waiter help hyperlink — the phone app

    waiter hyperlink pair --label "my iPhone"    mint a pairing code
    waiter hyperlink pair --qr                   print it as a QR payload
    waiter hyperlink devices                     what is paired
    waiter hyperlink unpair <device_id>          revoke one
    waiter hyperlink endpoints                   addresses the server answers on

Pairing runs against the T1 server, not against the phone: waiter asks
the server for a code and the phone redeems it. So the server has to be
running and reachable from here first — `waiter health` is the quick
check — and HyperLink has to be enabled on it (T1_HYPERLINK_ENABLED=1).

For a phone that is not on the LAN, pair over the tailnet and give the
app the 100.x address from `waiter hyperlink endpoints`.
""",
    "find": """\
waiter help find — locating a server you did not configure

    waiter -F my-server            search the tailnet and the LAN
    waiter -F my-server -l         local only; never touches Tailscale
    waiter -F <54-char-host-id>    by Host ID
    waiter -F http://host/api.jsonl  a descriptor endpoint directly

Discovery and connection are separate from running anything. A server's
api.jsonl may advertise commands; waiter shows them and never executes
them for you.
""",
}


_USAGE = """\
waiter — the official T1 API TUI/CLI

Usage:
  waiter serv     Setup and more: waiter serv -AE -I <server> -K <key>  (see waiter serv --help)
  waiter models   List models visible in the server's registry
  waiter model    Show detail/availability/usage for one model
  waiter route    Ask the routing engine which model to use (--plan, --model, --auto-fallback)
  waiter status   Server status
  waiter health   Server liveness check
  waiter whoami   Validate the configured key and show its scopes
  waiter usage    Show current usage (or --model <id> for remaining allowance)
  waiter servers  List / register servers (--register NAME --address ...)
  waiter modules  List modules, or --create/--upload/--sync one
  waiter jobs     get|cancel <job_id>
  waiter events   Poll recent events (--limit, --since)
  waiter billing  Balance/transactions, or --redeem <token>
  waiter cost     Spend, per-model/server breakdowns, forecasts, estimates
  waiter keys     List keys; --assign a plan/account/models; --import-file
  waiter deploy   Push a module to trusted servers (--to, --wait)
  waiter security Network policy (--block/--allow/--appeal), forced limits
  waiter audit    Read the server's audit trail (admin)
  waiter doctor   Check a server's configuration for production readiness
  waiter smoke    Run smoke tests against a server
  waiter tui      Open the full curses dashboard (same as `waiter serv -G`)
  waiter config   Show the locally saved config
  waiter kits     Installed kits: list, remove NAME, run NAME COMMAND (install: serv -k)

  waiter -F       Find a server by name / Host ID / api.jsonl (-l for local only)
  waiter version  Package, T1 API and key format versions
  waiter help     Longer help on one topic: `waiter help connect`

T1 v{t1_version}:
  waiter lmstudio  Bridge to a model loaded in LM Studio (status/models/chat/local)
  waiter hyperlink Pair phones, manage devices and server-side chat sessions
  waiter fetch     Resolve a Hugging Face page + file link into a GGUF download plan

Every subcommand accepts -I/-K/-F/-P/-H to override the saved config for
just that call, and --json for raw JSON output. (As the *first* argument,
-F means find; inside a subcommand it is the config-file override.)

Run `waiter <subcommand> --help` for detailed options.
"""


def _usage() -> str:
    """The usage text, with the T1 version filled in at call time.

    It used to be typed into the string and said 1.0.26.8.0.1 long after
    the API had moved to 1.0.26.8.1.0. A version that has to be updated
    by hand in a second place is a version that will be wrong.
    """
    from ..t1api.version import T1_VERSION_SHORT

    return _USAGE.format(t1_version=T1_VERSION_SHORT)


def _cmd_version(rest: list[str]) -> int:
    """Package, T1 API, waiter protocol, and key formats.

    Four numbers that move independently, which is exactly why they are
    printed together: "my key is refused" and "my client is too old" are
    diagnosed by comparing them, and hunting for each one separately is
    how people end up guessing.
    """
    import argparse

    p = argparse.ArgumentParser(prog="waiter version")
    p.add_argument("--json", dest="as_json", action="store_true")
    args = p.parse_args(rest)

    from hypernix import __version__ as package_version
    from hypernix.security.keyversions import (
        KEY_VERSIONS,
        LATEST_KEY_VERSION,
        RESERVED_KEY_VERSIONS,
    )
    from hypernix.t1api.version import (
        MIN_CLIENT_VERSION,
        T1_VERSION_LONG,
        T1_VERSION_SHORT,
    )

    from . import __waiter_version__

    if args.as_json:
        print(json.dumps({
            "hypernix": package_version,
            "waiter": __waiter_version__,
            "t1_api": {
                "short": T1_VERSION_SHORT,
                "long": T1_VERSION_LONG,
                "min_client": MIN_CLIENT_VERSION.short,
            },
            "key_versions": {
                "latest": LATEST_KEY_VERSION.name,
                "available": [v.name for v in KEY_VERSIONS],
                "reserved": [v.name for v in RESERVED_KEY_VERSIONS],
            },
        }, indent=2))
        return 0

    print(f"HyperNix:    {package_version}")
    print(f"waiter:      {__waiter_version__}")
    print(f"T1 API:      t1 v{T1_VERSION_SHORT}  ({T1_VERSION_LONG})")
    print(f"  speaks to: t1 v{MIN_CLIENT_VERSION.short} and newer")
    print(f"Key formats: {', '.join(v.name for v in KEY_VERSIONS)}"
          f"  (latest {LATEST_KEY_VERSION.name})")
    reserved = ", ".join(v.name for v in RESERVED_KEY_VERSIONS)
    if reserved:
        print(f"  not yet:   {reserved}")

    # The server's version, when one is configured and reachable.
    #
    # Fetched unauthenticated: /status is public, and requiring a key here
    # would mean `waiter version` — the command you run precisely when
    # something is wrong — failing for want of the credential you are
    # trying to diagnose. Left out entirely rather than guessed at when it
    # cannot be fetched: a version report that invents a server version is
    # worse than one that admits it did not look.
    status = _peek_server_status()
    if status:
        # `t1_api_version` is the flat string; `t1_version` is an object
        # with the parts broken out. Reading the wrong one prints a dict.
        remote = status.get("t1_api_version")
        if not remote:
            nested = status.get("t1_version")
            remote = nested.get("short") if isinstance(nested, dict) else nested
        name = status.get("server_name")
        if remote:
            print(f"Server:      t1 v{remote}" + (f"  ({name})" if name else ""))
    return 0


def _peek_server_status() -> dict | None:
    """GET /status from the configured server, or None. Never raises."""
    from .diagnose import get_json

    try:
        base = _base_url(_resolve_config(_bare_connection_args()))
    except Exception:  # noqa: BLE001
        return None
    return get_json(base.rstrip("/") + "/status", timeout=2.0)


def _bare_connection_args() -> argparse.Namespace:
    """A Namespace with every connection option unset.

    Built from the parser rather than by hand so it cannot drift from the
    real option names — spelling one wrong here produces an
    AttributeError swallowed by a broad `except`, which looks exactly like
    the server being unreachable.
    """
    parser = argparse.ArgumentParser(add_help=False)
    _add_common_connection_args(parser)
    return parser.parse_args([])


def _cmd_help(rest: list[str]) -> int:
    """Longer help on one topic."""
    if not rest or rest[0] in ("-h", "--help"):
        print("waiter help <topic>\n")
        print("Topics:")
        for name, body in _HELP_TOPICS.items():
            summary = body.splitlines()[0].split("—", 1)[-1].strip()
            print(f"  {name:<11} {summary}")
        print("\nRun `waiter --help` for the full subcommand list.")
        return 0
    topic = rest[0].lower()
    if topic not in _HELP_TOPICS:
        _err(f"No help topic {topic!r}. Try: {', '.join(_HELP_TOPICS)}")
        return 1
    print(_HELP_TOPICS[topic])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `waiter` console script."""
    raw = list(sys.argv[1:] if argv is None else argv)

    if not raw or raw[0] in ("-h", "--help"):
        print(_usage())
        return 0

    if raw[0] in ("-V", "--version"):
        from . import __waiter_version__

        print(f"waiter {__waiter_version__}")
        return 0

    # -F/--find is a flag rather than a subcommand, because that is how
    # the release specifies it. `waiter find ...` is accepted too and
    # runs the same function; there is one implementation.
    if raw[0] in ("-F", "--find"):
        return _cmd_find(raw[1:])

    cmd, rest = raw[0], raw[1:]
    dispatch = {
        "serv": _cmd_serv,
        "kits": _cmd_kits,
        "models": _cmd_models,
        "model": _cmd_model,
        "status": _cmd_status,
        "health": _cmd_health,
        "whoami": _cmd_whoami,
        "usage": _cmd_usage,
        "config": _cmd_config,
        "route": _cmd_route,
        "servers": _cmd_servers,
        "modules": _cmd_modules,
        "jobs": _cmd_jobs,
        "events": _cmd_events,
        "billing": _cmd_billing,
        "keys": _cmd_keys,
        "audit": _cmd_audit,
        "security": _cmd_security,
        "cost": _cmd_cost,
        "deploy": _cmd_deploy,
        "tui": _cmd_tui,
        "doctor": _cmd_doctor,
        "smoke": _cmd_smoke,
        # T1 v1.0.26.8.0.1
        "lmstudio": _cmd_lmstudio,
        "find": _cmd_find,
        "hyperlink": _cmd_hyperlink,
        "fetch": _cmd_fetch,
        "version": _cmd_version,
        "help": _cmd_help,
    }

    if cmd not in dispatch:
        print(f"Unknown subcommand: {cmd!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 1

    try:
        return dispatch[cmd](rest)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except T1ClientError as exc:
        _err_connection(exc)
        return 1
    except WaiterConfigLocked as exc:
        _err(str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[waiter {cmd}] Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
