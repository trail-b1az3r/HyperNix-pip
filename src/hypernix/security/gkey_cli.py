"""gkey_cli — Unified CLI for Gatekeeper and Keymaster.

Entry point: ``gkey`` (console script) or ``hypernix gkey``.

Subcommands
-----------
::

    gkey create     [--type dev|user|service|session|admin]
                    [--scopes read,write,admin,plugin,service]
                    [--expires YYYY-MM-DD] [--cap N] [--limit N]
                    [--prefix LABEL] [--tags k=v ...] [--body-len N]
                    [--note TEXT]

    gkey revoke     <key-id>  [--reason TEXT]

    gkey list       [--type TYPE] [--scope SCOPE]
                    [--all]  (include expired)
                    [--json]

    gkey list id    <key-id>    show full metadata for one key

    gkey stats      [--key KEY-ID]  [--log N]  [--json]

    gkey quota      --key KEY-ID
                    [--set max-requests=N,max-tokens=N,window=N]

    gkey permissions  --key KEY-ID

    gkey rotate     <key-id>

    gkey export     [--key KEY-ID] [--out FILE]

    gkey import     <FILE>

    gkey version    [--json]
                    HyperNix, T1 API, and key format versions

Key formats
-----------
``gkey create -v`` chooses which format the new key is presented in::

    gkey create -v v1                          # T1_… (default)
    gkey create -v v2 --level 5                # T2_…-5
    gkey create -v v2 --type admin             # T2_<password>_…-9
    gkey create -v v2short                     # T2S_…-1, for HyperLink

A v2 key is a *spelling* of a v1 key rather than a separate credential:
the key is minted into the store in its v1 form and presented in the
requested one, and the server converts it back on every request. Both
spellings therefore work, and ``gkey revoke`` on the key ID kills both.

All output is rich-formatted when the ``rich`` package is available,
plain text otherwise.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hypernix.security.keyversions import (
    DEFAULT_KEY_VERSION,
    KEY_VERSIONS,
    LATEST_KEY_VERSION,
    RESERVED_KEY_VERSIONS,
    key_version_names,
    resolve_key_version,
)

# ---------------------------------------------------------------------------
# Rich helpers (graceful degradation)
# ---------------------------------------------------------------------------


def _try_rich() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_RICH = _try_rich()


def _console():
    if _HAS_RICH:
        from rich.console import Console
        return Console()
    return None


#: The style names this module marks text up with. Deliberately a closed
#: list rather than a general ``\[[^]]*]`` pattern: ``[`` and ``]`` are
#: both valid characters inside a key, so a general stripper would
#: silently corrupt a key it was asked to print — the one string here
#: that has to survive byte for byte.
_MARKUP_STYLES = (
    "bold green", "bold red", "bold", "cyan", "dim", "green", "red", "yellow",
)
_MARKUP = re.compile(
    r"\[/?(?:" + "|".join(re.escape(name) for name in _MARKUP_STYLES) + r")]"
)


def _strip_markup(text: str) -> str:
    return _MARKUP.sub("", text)


def _literal(value: object) -> str:
    """A value that must reach the terminal exactly as it is.

    Every panel and table below is rendered with rich markup on, because
    this module writes markup into them. A *key* is not markup — but the
    T1/T2 special-character set includes ``[`` and ``]``, so roughly one
    key in three thousand carries a bracket pair that rich reads as a
    style tag, eats, and prints without.

    That is not a cosmetic bug. `gkey create` shows the operator a
    credential missing characters, they paste it, and it authenticates as
    nothing — with no error anywhere that says why. So anything that is
    data rather than markup goes through here first.

    Narrowing the key alphabet instead would be the wrong fix: the T2
    special set is deliberately identical to T1's so that T1 -> T2 -> T1
    is exact, and dropping two characters from it would break that.
    """
    text = str(value)
    if not _HAS_RICH:
        return text
    from rich.markup import escape

    return escape(text)


def _print_rich(text: str, style: str = "", plain: str | None = None) -> None:
    """Print *text*, which may carry rich markup.

    Without rich, markup would otherwise reach the terminal literally —
    ``[yellow]v2.1[/yellow]``. Pass *plain* when the fallback wants
    different wording; otherwise the known style tags are stripped.
    """
    if _HAS_RICH:
        from rich.console import Console
        Console().print(text, style=style or "default")
    else:
        print(plain if plain is not None else _strip_markup(text))


def _print_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    if _HAS_RICH:
        from rich.console import Console
        from rich.table import Table
        t = Table(title=title, header_style="bold cyan", border_style="dim")
        for h in headers:
            t.add_column(h, overflow="fold")
        for row in rows:
            t.add_row(*[_literal(cell) for cell in row])
        Console().print(t)
    else:
        if title:
            print(f"\n{title}")
            print("-" * len(title))
        widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in widths))
        for row in rows:
            print(fmt.format(*row))


def _print_panel(content: str, title: str = "") -> None:
    if _HAS_RICH:
        from rich.console import Console
        from rich.panel import Panel
        Console().print(Panel(content, title=title, border_style="cyan"))
    else:
        if title:
            print(f"\n=== {title} ===")
        print(content)


# ---------------------------------------------------------------------------
# Shared Keymaster / Gatekeeper factory
# ---------------------------------------------------------------------------


def _get_km(store: Path | None = None):
    """Return a Keymaster instance (auto_rotate=False for CLI use).

    Honours ``T1_KEYMASTER_DIR``, which is what the server reads. Without
    that, an install using ``--config-dir`` puts the server's key store
    under the config directory while gkey kept writing to
    ``~/.hypernix/keymaster`` — so every key the operator minted was
    invisible to their own server, and the key they were handed at first
    start was invisible to gkey. Two halves of one tool disagreeing about
    where the keys live is a hard failure to reason about, because both
    of them work.
    """
    from hypernix.security.keymaster import Keymaster

    if store is None:
        configured = os.environ.get("T1_KEYMASTER_DIR")
        if configured:
            store = Path(configured)
    return Keymaster(store_dir=store, auto_rotate=False)


def _get_gk(km, data: Path | None = None):
    """Return a Gatekeeper backed by *km*."""
    from hypernix.security.gatekeeper import Gatekeeper
    return Gatekeeper(km, data_dir=data)


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _parse_scopes(raw: str):
    from hypernix.security.keymaster import KeyScope
    mapping = {s.value: s for s in KeyScope}
    result = set()
    for part in raw.split(","):
        part = part.strip()
        if part not in mapping:
            raise SystemExit(
                f"Unknown scope: {part!r}. Valid: {', '.join(mapping)}"
            )
        result.add(mapping[part])
    return result


def _parse_expires(raw: str) -> float:
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC)
        return dt.timestamp()
    except ValueError:
        raise SystemExit(
            f"Invalid date format: {raw!r}. Use YYYY-MM-DD."
        ) from None


def _parse_tags(raw_list: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for item in raw_list:
        if "=" not in item:
            raise SystemExit(f"Invalid tag format: {item!r}. Use key=value.")
        k, v = item.split("=", 1)
        tags[k.strip()] = v.strip()
    return tags


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Subcommand: create
# ---------------------------------------------------------------------------


def _generate_password(include_word: bool) -> str:
    from hypernix.security.t2keys import generate_admin_password

    return generate_admin_password(include_word=include_word)


def _cmd_create(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="gkey create")
    p.add_argument("--type", dest="key_type", default="user",
                   choices=["dev", "development", "user", "service", "session", "admin"])
    p.add_argument("--scopes", default="read",
                   help="Comma-separated scopes: read,write,admin,plugin,service")
    p.add_argument("--expires", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--cap", type=int, default=None, metavar="TOKENS",
                   help="Lifetime token cap")
    p.add_argument("--limit", type=int, default=None, metavar="REQUESTS",
                   help="Lifetime request limit")
    p.add_argument("--prefix", default="", help="Short human label for the key")
    p.add_argument("--tags", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--body-len", type=int, default=24, metavar="N",
                   help="Body length of generated key (default 24, min 16)")
    p.add_argument("--note", default="", help="Free-text note attached to the key")
    p.add_argument("--rotation-window", type=int, default=24, metavar="HOURS",
                   help="Hours before expiry to auto-rotate (default 24)")
    p.add_argument("-v", "--key-version", dest="key_version",
                   default=DEFAULT_KEY_VERSION.name, metavar="VERSION",
                   help="Key format: " + ", ".join(key_version_names())
                        + f" (default {DEFAULT_KEY_VERSION.name})")
    p.add_argument("--level", type=int, default=None, metavar="1-9",
                   help="Access level for a v2/v2short key (default 1, "
                        "or 9 for an admin key)")
    p.add_argument("--password", default=None, metavar="PASSWORD",
                   help="Admin password for a v2 admin key. Generated when "
                        "omitted; a supplied one is validated, not trusted.")
    p.add_argument("--word", dest="include_word", action="store_true",
                   help="Embed a six-letter word in a generated admin password "
                        "(memorability, not entropy)")
    p.add_argument("-Con", "--config-source", dest="config_source", default=None,
                   metavar="ADDR|URL|PATH",
                   help="Take the V1 Server ID and/or SSPKID from a JSONL "
                        "config. A bare IP or host becomes "
                        "http://<host>/gkey.jsonl; a URL or path is used as "
                        "given. Sets identity only — never scopes, type or "
                        "expiry.")
    ns = p.parse_args(args)

    # Read the config before minting. A key created and then found to have
    # an unusable identity would still be in the store, valid, and known
    # to nobody but the operator who saw an error — the same reasoning as
    # every other precondition in this function.
    key_config = None
    if ns.config_source:
        from hypernix.security.keyconfig import KeyConfigError, load_key_config

        try:
            key_config = load_key_config(ns.config_source)
        except KeyConfigError as exc:
            print(f"[gkey create] -Con: {exc}", file=sys.stderr)
            return 2

    try:
        version = resolve_key_version(ns.key_version)
    except ValueError as exc:
        print(f"[gkey create] {exc}", file=sys.stderr)
        return 2

    from hypernix.security.keymaster import KeyType
    type_map = {
        "dev": KeyType.DEVELOPMENT,
        "development": KeyType.DEVELOPMENT,
        "user": KeyType.USER,
        "service": KeyType.SERVICE,
        "session": KeyType.SESSION,
        "admin": KeyType.ADMIN,
    }
    scopes = _parse_scopes(ns.scopes)
    expires = _parse_expires(ns.expires) if ns.expires else None
    tags = _parse_tags(ns.tags)
    key_type = type_map[ns.key_type]
    wants_admin = key_type is KeyType.ADMIN

    # Everything that makes the request impossible is checked here, before
    # a key exists. A key minted and then found unpresentable would still
    # be in the store, valid, and unknown to the operator who saw only an
    # error — a real credential nobody is tracking.
    if wants_admin and not version.supports_admin:
        print(
            f"[gkey create] A {version.name} key cannot be an administrator: "
            f"{version.summary}\n"
            f"              Use -v v2 for an admin key in the T2 family.",
            file=sys.stderr,
        )
        return 2
    if ns.level is not None and not version.supports_access_level:
        print(
            f"[gkey create] --level does not apply to a {version.name} key; "
            f"the format has no access-level field.",
            file=sys.stderr,
        )
        return 2
    if ns.level is not None and not 1 <= ns.level <= 9:
        print(f"[gkey create] --level must be 1-9, got {ns.level}", file=sys.stderr)
        return 2
    if ns.password is not None and not (wants_admin and version.supports_admin):
        print(
            "[gkey create] --password marks a key as an administrator; pass "
            "--type admin as well, or drop it.",
            file=sys.stderr,
        )
        return 2

    # A T2S body is exactly 26 characters, so the underlying T1 key has to
    # be minted at that length — the presentation cannot change it later.
    body_length = version.body_length or ns.body_len
    if version.body_length and ns.body_len != 24 and ns.body_len != version.body_length:
        print(
            f"[gkey create] --body-len {ns.body_len} conflicts with {version.name}, "
            f"whose body is fixed at {version.body_length}.",
            file=sys.stderr,
        )
        return 2

    access_level = ns.level if ns.level is not None else (9 if wants_admin else 1)

    # The key format is recorded on the key so `gkey list` can say which
    # spelling was issued. The store only ever holds the T1 form, so
    # without this the issued spelling is lost the moment the key scrolls
    # off the screen.
    if version is not DEFAULT_KEY_VERSION:
        tags = {**tags, "key_version": version.name}
        if version.supports_access_level:
            tags["access_level"] = str(access_level)

    km = _get_km()
    meta = km.create(
        key_type=key_type,
        scopes=scopes,
        expires_at=expires,
        usage_cap=ns.cap,
        request_limit=ns.limit,
        prefix=ns.prefix,
        tags=tags,
        rotation_window=ns.rotation_window,
        note=ns.note,
        body_length=body_length,
    )
    # Identity from the config source, applied to the key that now
    # exists. Failing here does not leave a key with a half-applied
    # identity: the server ID is validated before it is written, and an
    # SSPKID collision leaves the key on its default identity with the
    # reason printed, rather than silently taking an identifier that
    # belongs to a different key.
    config_applied: list[str] = []
    if key_config is not None:
        if key_config.server_id:
            try:
                meta = km.set_server_id(meta.key_id, key_config.server_id)
                config_applied.append(f"server_id={key_config.server_id}")
            except ValueError as exc:
                km.stop()
                print(f"[gkey create] -Con: {exc}", file=sys.stderr)
                return 2

        sspkid_text = key_config.sspkid
        if not sspkid_text and key_config.sspkid_index is not None:
            sspkid_text = f"{meta.server_id}#{key_config.sspkid_index}"
        if sspkid_text:
            from hypernix.security.t2keys import (
                SSPKID,
                ServerKeyRegistry,
                SSPKIDCollision,
            )

            try:
                parsed_sspkid = SSPKID.parse(sspkid_text)
            except ValueError as exc:
                km.stop()
                print(f"[gkey create] -Con: {exc}", file=sys.stderr)
                return 2
            try:
                ServerKeyRegistry().assign(meta.key_id, parsed_sspkid)
            except SSPKIDCollision as exc:
                km.stop()
                print(f"[gkey create] -Con: {exc}", file=sys.stderr)
                return 2
            config_applied.append(f"sspkid={parsed_sspkid}")

    km.stop()

    # Present the minted key in the requested format.
    #
    # The T1 key stays in the store and stays valid; a v2 spelling is
    # converted back to it on every authentication. That is why this is a
    # presentation step after minting rather than a different generator:
    # a T2 key generated on its own belongs to no key store and
    # authenticates as nothing.
    issued_key = meta.key
    admin_password = ""
    t2c_kit = None
    if version is not DEFAULT_KEY_VERSION:
        from hypernix.security.t2keys import T2KeyGenerator, T2Type

        # v2.1 seals a v2 spelling, so the key underneath is always T2.
        family = T2Type.T2 if version.family == "T2C" else T2Type(version.family)
        if wants_admin:
            # Guarded by the store's own record, not by the flag that was
            # typed: from_t1_admin's precondition is that the key really
            # is an administrator.
            if meta.key_type is not KeyType.ADMIN:
                print(
                    "[gkey create] Refusing to present a non-admin key as an "
                    "admin key.",
                    file=sys.stderr,
                )
                return 1
            t2 = T2KeyGenerator.from_t1_admin(
                meta.key,
                password=ns.password or _generate_password(ns.include_word),
                access_level=access_level,
            )
            admin_password = t2.password
        else:
            t2 = T2KeyGenerator.from_t1(
                meta.key,
                access_level=access_level,
                family=family,
            )
        issued_key = t2.raw
        if version.family == "T2C":
            from hypernix.security.t2c import T2CAuthority, T2CError

            try:
                t2c_kit = T2CAuthority(Path(km.store_dir) / "t2c").issue_kit(
                    t2.raw, bound_key=meta.key, access_level=access_level,
                    label=ns.prefix or meta.key_id[:8],
                )
            except (T2CError, ValueError) as exc:
                print(f"[gkey create] Could not seal the key as v2.1: {exc}\n"
                      f"              The key exists as {meta.key_id}; revoke it or "
                      f"present it as v2.", file=sys.stderr)
                return 1
            issued_key = t2c_kit.key_for()

    content_lines = [
        "[bold green]Key created successfully![/bold green]",
        "",
        f"[bold]Key ID:[/bold]     {_literal(meta.key_id)}",
        f"[bold]Key:[/bold]        [yellow]{_literal(issued_key)}[/yellow]",
        f"[bold]Format:[/bold]     {version.name} ({version.family})",
        f"[bold]Type:[/bold]       {meta.key_type.value}",
        f"[bold]Scopes:[/bold]     {', '.join(s.value for s in sorted(meta.scopes, key=lambda x: x.value))}",
        f"[bold]Expires:[/bold]    {_fmt_ts(meta.expires_at)}",
        f"[bold]Server ID:[/bold]  {_literal(meta.server_id)}",
        f"[bold]Prefix:[/bold]     {_literal(meta.prefix or '—')}",
        f"[bold]Note:[/bold]       {_literal(meta.note or '—')}",
    ]
    if version.supports_access_level:
        content_lines.insert(
            5, f"[bold]Level:[/bold]      {access_level}"
        )
    if admin_password:
        content_lines.append(
            f"[bold]Password:[/bold]   [yellow]{_literal(admin_password)}[/yellow]"
        )
    if config_applied:
        content_lines.append(
            f"[bold]Config:[/bold]     {', '.join(config_applied)}"
        )
        content_lines.append(f"[dim]             from {key_config.source}[/dim]")
    if tags:
        content_lines.append(f"[bold]Tags:[/bold]       {_literal(json.dumps(tags))}")
    if t2c_kit is not None:
        # Hundreds of characters each: boxed, they wrap inside the border
        # and cannot be copied. They are printed plainly below the panel.
        content_lines[3] = "[bold]Sealed:[/bold]     key and kit printed below the panel"
        content_lines.append(f"[bold]Device:[/bold]     {_literal(t2c_kit.device_id)}")
        content_lines.append("")
        content_lines.append(
            "[dim]The key above is today's. Give the kit to the client instead —\n"
            "'waiter serv -A -I <server> -K <kit>' keeps it and makes each day's key.\n"
            "Revoke the device with DELETE /auth/t2c/devices/"
            f"{t2c_kit.device_id}.[/dim]"
        )
    elif version is not DEFAULT_KEY_VERSION:
        content_lines.append("")
        content_lines.append(
            f"[dim]This is the {version.name} spelling of a v1 key that is in the "
            f"key store.\nThe server converts it back on every request, so the "
            f"{DEFAULT_KEY_VERSION.name} form below works too:[/dim]"
        )
        content_lines.append(f"[dim]  {_literal(meta.key)}[/dim]")

    if _HAS_RICH:
        _print_panel("\n".join(content_lines), title="gkey create")
        if t2c_kit is not None:
            print(f"Key: {issued_key}")
            print(f"Kit: {t2c_kit.to_text()}")
    else:
        print("Key created successfully!")
        print(f"  Key ID:    {meta.key_id}")
        print(f"  Key:       {issued_key}")
        print(f"  Format:    {version.name} ({version.family})")
        if version.supports_access_level:
            print(f"  Level:     {access_level}")
        print(f"  Type:      {meta.key_type.value}")
        print(f"  Scopes:    {', '.join(s.value for s in sorted(meta.scopes, key=lambda x: x.value))}")
        print(f"  Expires:   {_fmt_ts(meta.expires_at)}")
        print(f"  Server ID: {meta.server_id}")
        if admin_password:
            print(f"  Password:  {admin_password}")
        if t2c_kit is not None:
            print(f"  Kit:       {t2c_kit.to_text()}")
            print(f"  Device:    {t2c_kit.device_id}")
        elif version is not DEFAULT_KEY_VERSION:
            print(f"  v1 form:   {meta.key}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: revoke
# ---------------------------------------------------------------------------


def _cmd_revoke(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="gkey revoke")
    p.add_argument("key_id", help="Key ID to revoke")
    p.add_argument("--reason", default="", help="Reason for revocation")
    ns = p.parse_args(args)

    km = _get_km()
    try:
        km.revoke(ns.key_id, reason=ns.reason)
        km.stop()
        _print_rich(f"[bold red]✗[/bold red] Key [cyan]{ns.key_id[:8]}…[/cyan] revoked.")
        return 0
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        km.stop()
        return 1


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------


def _cmd_list(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="gkey list")
    p.add_argument("key_id", nargs="?", default=None,
                   help="If 'id <key-id>', show detailed info for one key")
    p.add_argument("--type", dest="key_type", default=None,
                   choices=["dev", "development", "user", "service", "session", "admin"])
    p.add_argument("--scope", default=None)
    p.add_argument("--all", dest="include_all", action="store_true",
                   help="Include expired keys")
    p.add_argument("--json", dest="as_json", action="store_true")
    ns = p.parse_args(args)

    km = _get_km()

    # `gkey list id <key-id>` form
    if ns.key_id == "id":
        print("Usage: gkey list id <key-id>", file=sys.stderr)
        km.stop()
        return 1

    if ns.key_id is not None:
        # Single-key detail view
        meta = km.get(ns.key_id)
        if meta is None:
            print(f"Key not found: {ns.key_id!r}", file=sys.stderr)
            km.stop()
            return 1
        km.stop()
        if ns.as_json:
            print(json.dumps(meta.to_dict(), indent=2))
        else:
            _print_detail(meta)
        return 0

    from hypernix.security.keymaster import KeyScope, KeyType
    type_map = {
        "dev": KeyType.DEVELOPMENT, "development": KeyType.DEVELOPMENT,
        "user": KeyType.USER, "service": KeyType.SERVICE,
        "session": KeyType.SESSION, "admin": KeyType.ADMIN,
    }
    key_type = type_map[ns.key_type] if ns.key_type else None
    scope = None
    if ns.scope:
        scope_map = {s.value: s for s in KeyScope}
        scope = scope_map.get(ns.scope)

    keys = km.list(
        key_type=key_type,
        scope=scope,
        active_only=not ns.include_all,
        include_expired=ns.include_all,
    )
    km.stop()

    if ns.as_json:
        print(json.dumps([m.to_dict() for m in keys], indent=2))
        return 0

    if not keys:
        _print_rich("[dim]No keys found.[/dim]")
        return 0

    headers = ["Key ID (short)", "Type", "Scopes", "Expires", "Status", "Prefix"]
    rows = []
    for m in keys:
        scopes_str = ",".join(s.value for s in sorted(m.scopes, key=lambda x: x.value))
        status = "active" if (m.active and not m.is_expired) else (
            "expired" if m.is_expired else "revoked"
        )
        rows.append([
            m.key_id[:12] + "…",
            m.key_type.value,
            scopes_str,
            _fmt_ts(m.expires_at),
            status,
            m.prefix or "—",
        ])
    _print_table(headers, rows, title=f"Keys ({len(keys)})")
    return 0


def _print_detail(meta: Any) -> None:
    """Print full key metadata."""
    lines = [
        f"Key ID:          {meta.key_id}",
        f"Key:             {_literal(meta.key)}",
        f"Type:            {meta.key_type.value}",
        f"Scopes:          {', '.join(s.value for s in sorted(meta.scopes, key=lambda x: x.value))}",
        f"Created:         {_fmt_ts(meta.created_at)}",
        f"Expires:         {_fmt_ts(meta.expires_at)}",
        f"Rotation window: {meta.rotation_window}h",
        f"Usage cap:       {meta.usage_cap or '—'}",
        f"Request limit:   {meta.request_limit or '—'}",
        f"Usage count:     {meta.usage_count}",
        f"Request count:   {meta.request_count}",
        f"Server ID:       {meta.server_id}",
        f"Prefix:          {_literal(meta.prefix or '—')}",
        f"Tags:            {json.dumps(meta.tags) if meta.tags else '—'}",
        f"Active:          {meta.active}",
        f"Note:            {_literal(meta.note or '—')}",
    ]
    if meta.rotated_from:
        lines.append(f"Rotated from:    {meta.rotated_from}")
    if meta.revoked_at:
        lines.append(f"Revoked at:      {_fmt_ts(meta.revoked_at)}")
    _print_panel("\n".join(lines), title=f"Key Detail — {meta.key_id[:8]}…")


# ---------------------------------------------------------------------------
# Subcommand: list id (separate positional form)
# ---------------------------------------------------------------------------


def _cmd_list_id(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="gkey list id")
    p.add_argument("key_id", help="Key ID to inspect")
    p.add_argument("--json", dest="as_json", action="store_true")
    ns = p.parse_args(args)

    km = _get_km()
    meta = km.get(ns.key_id)
    km.stop()
    if meta is None:
        print(f"Key not found: {ns.key_id!r}", file=sys.stderr)
        return 1
    if ns.as_json:
        print(json.dumps(meta.to_dict(), indent=2))
    else:
        _print_detail(meta)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------


def _cmd_stats(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="gkey stats")
    p.add_argument("--key", default=None, metavar="KEY-ID",
                   help="Show stats for a single key")
    p.add_argument("--log", type=int, default=0, metavar="N",
                   help="Also print the last N usage log entries")
    p.add_argument("--json", dest="as_json", action="store_true")
    ns = p.parse_args(args)

    km = _get_km()
    gk = _get_gk(km)

    if ns.key:
        data = gk.get_stats(ns.key)
        result: Any = data
    else:
        result = gk.get_all_stats()

    log_entries: list[dict[str, Any]] = []
    if ns.log > 0:
        log_entries = gk.get_usage_log(key_id=ns.key, limit=ns.log)

    km.stop()
    gk.stop()

    if ns.as_json:
        out: Any = result
        if log_entries:
            if isinstance(out, dict):
                out["log"] = log_entries
            else:
                out = {"stats": out, "log": log_entries}
        print(json.dumps(out, indent=2))
        return 0

    # Pretty print
    if isinstance(result, dict):
        _print_stats_single(result)
    else:
        if not result:
            _print_rich("[dim]No usage data recorded yet.[/dim]")
        else:
            headers = ["Key ID", "Type", "Requests", "Tokens", "Last Used"]
            rows = []
            for s in result:
                rows.append([
                    s["key_id"][:12] + "…",
                    s.get("key_type", "—"),
                    str(s.get("total_requests", 0)),
                    str(s.get("total_tokens", 0)),
                    _fmt_ts(s.get("last_used")),
                ])
            _print_table(headers, rows, title="Usage Statistics")

    if log_entries:
        log_headers = ["Time", "Key ID", "Endpoint", "Model", "Tokens"]
        log_rows = []
        for e in log_entries:
            log_rows.append([
                _fmt_ts(e.get("timestamp")),
                e.get("key_id", "")[:12] + "…",
                e.get("endpoint", "—"),
                e.get("model", "—"),
                str(e.get("tokens_used", 0)),
            ])
        _print_table(log_headers, log_rows, title="Recent Log Entries")

    return 0


def _print_stats_single(s: dict[str, Any]) -> None:
    lines = [
        f"Key ID:          {s['key_id']}",
        f"Type:            {s.get('key_type', '—')}",
        f"Active:          {s.get('active', False)}",
        f"Scopes:          {', '.join(s.get('scopes', []))}",
        f"Total requests:  {s.get('total_requests', 0)}",
        f"Total tokens:    {s.get('total_tokens', 0)}",
        f"Lifetime reqs:   {s.get('lifetime_request_count', 0)}",
        f"Lifetime tokens: {s.get('lifetime_token_count', 0)}",
        f"Request limit:   {s.get('request_limit') or '—'}",
        f"Token cap:       {s.get('usage_cap') or '—'}",
        f"Last used:       {_fmt_ts(s.get('last_used'))}",
        f"Window reqs:     {s.get('window_requests', 0)}",
        f"Window tokens:   {s.get('window_tokens', 0)}",
    ]
    if s.get("quota"):
        q = s["quota"]
        lines.append(
            f"Quota:           {q.get('max_requests', '∞')} req / "
            f"{q.get('max_tokens', '∞')} tok per {q.get('window_seconds', 60)}s"
        )
    _print_panel("\n".join(lines), title=f"Stats — {s['key_id'][:8]}…")


# ---------------------------------------------------------------------------
# Subcommand: quota
# ---------------------------------------------------------------------------


def _cmd_quota(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="gkey quota")
    p.add_argument("--key", required=True, metavar="KEY-ID")
    p.add_argument("--set", dest="quota_set", default=None,
                   metavar="max-requests=N,max-tokens=N,window=N",
                   help="Set quota values (comma-separated key=value pairs)")
    ns = p.parse_args(args)

    km = _get_km()
    gk = _get_gk(km)

    if ns.quota_set:
        from hypernix.security.gatekeeper import Quota
        qargs: dict[str, Any] = {}
        for part in ns.quota_set.split(","):
            part = part.strip()
            if "=" not in part:
                print(f"Invalid quota spec: {part!r}", file=sys.stderr)
                km.stop()
                gk.stop()
                return 1
            k, v = part.split("=", 1)
            k = k.strip().replace("-", "_")
            try:
                qargs[k] = float(v) if "." in v else int(v)
            except ValueError:
                print(f"Invalid value for {k}: {v!r}", file=sys.stderr)
                km.stop()
                gk.stop()
                return 1
        quota = Quota(
            max_requests=qargs.get("max_requests"),
            max_tokens=qargs.get("max_tokens"),
            window_seconds=float(qargs.get("window", 60)),
        )
        gk.set_quota(ns.key, quota)
        gk._save_usage()
        _print_rich(
            f"[green]✓[/green] Quota set for [cyan]{ns.key[:8]}…[/cyan]: "
            f"max_requests={quota.max_requests}, max_tokens={quota.max_tokens}, "
            f"window={quota.window_seconds}s"
        )
    else:
        quota = gk.get_quota(ns.key)
        if quota is None:
            _print_rich(f"[dim]No quota configured for {ns.key[:8]}…[/dim]")
        else:
            _print_panel(
                f"max_requests: {quota.max_requests or '∞'}\n"
                f"max_tokens:   {quota.max_tokens or '∞'}\n"
                f"window:       {quota.window_seconds}s",
                title=f"Quota — {ns.key[:8]}…",
            )
    km.stop()
    gk.stop()
    return 0


# ---------------------------------------------------------------------------
# Subcommand: permissions
# ---------------------------------------------------------------------------


def _cmd_permissions(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="gkey permissions")
    p.add_argument("--key", required=True, metavar="KEY-ID")
    ns = p.parse_args(args)

    km = _get_km()
    meta = km.get(ns.key)
    km.stop()

    if meta is None:
        print(f"Key not found: {ns.key!r}", file=sys.stderr)
        return 1

    scopes = sorted(s.value for s in meta.scopes)
    if _HAS_RICH:
        from rich.console import Console
        from rich.table import Table
        t = Table(title=f"Permissions — {ns.key[:8]}…", header_style="bold cyan")
        t.add_column("Scope")
        t.add_column("Granted")
        from hypernix.security.keymaster import KeyScope
        all_scopes = [s.value for s in KeyScope]
        for s in all_scopes:
            granted = s in scopes
            t.add_row(s, "[green]✓[/green]" if granted else "[red]✗[/red]")
        Console().print(t)
    else:
        from hypernix.security.keymaster import KeyScope
        print(f"Permissions for {ns.key[:8]}…:")
        for s in KeyScope:
            mark = "✓" if s.value in scopes else "✗"
            print(f"  {mark} {s.value}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: rotate
# ---------------------------------------------------------------------------


def _cmd_rotate(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="gkey rotate")
    p.add_argument("key_id", help="Key ID to rotate")
    ns = p.parse_args(args)

    km = _get_km()
    try:
        new_meta = km.rotate(ns.key_id)
        km.stop()
        _print_rich(
            f"[green]✓[/green] Key rotated.\n"
            f"  Old: [red]{ns.key_id[:8]}…[/red]\n"
            f"  New: [green]{_literal(new_meta.key_id)}[/green]\n"
            f"  Key: [yellow]{_literal(new_meta.key)}[/yellow]"
        )
        return 0
    except KeyError as exc:
        km.stop()
        print(f"Error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Subcommand: export
# ---------------------------------------------------------------------------


def _cmd_export(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="gkey export")
    p.add_argument("--key", default=None, metavar="KEY-ID",
                   help="Export a single key (default: all)")
    p.add_argument("--out", default=None, metavar="FILE",
                   help="Output file path (default: stdout)")
    ns = p.parse_args(args)

    km = _get_km()
    try:
        payload = km.export(path=ns.out, key_id=ns.key)
        km.stop()
    except KeyError as exc:
        km.stop()
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if ns.out:
        _print_rich(
            f"[green]✓[/green] Exported {len(payload['keys'])} key(s) to [cyan]{ns.out}[/cyan]"
        )
    else:
        print(json.dumps(payload, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: import
# ---------------------------------------------------------------------------


def _cmd_import(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="gkey import")
    p.add_argument("file", help="JSON file to import")
    ns = p.parse_args(args)

    path = Path(ns.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    km = _get_km()
    imported = km.import_keys(path)
    km.stop()
    _print_rich(
        f"[green]✓[/green] Imported [bold]{len(imported)}[/bold] key(s) from [cyan]{path}[/cyan]"
    )
    return 0


# ---------------------------------------------------------------------------
# Usage / dispatch
# ---------------------------------------------------------------------------

_USAGE = """\
gkey — Gatekeeper + Keymaster unified CLI

Usage:
  gkey create       Generate a new T1 API key
  gkey revoke       Revoke an existing key
  gkey list         List keys (gkey list id <id> for detail)
  gkey stats        Show usage statistics
  gkey quota        View or set rate-limit quotas
  gkey permissions  Show permission scopes for a key
  gkey rotate       Rotate (replace) a key with a fresh one
  gkey export       Export key(s) to JSON
  gkey import       Import key(s) from JSON
  gkey version      HyperNix, T1 API, and key format versions

Run `gkey <subcommand> --help` for detailed options.
"""


# ---------------------------------------------------------------------------
# Subcommand: version
# ---------------------------------------------------------------------------


def _cmd_version(args: list[str]) -> int:
    """What this install is, what it talks to, and what it can mint.

    Three separate version lines, because they move independently: the
    package ships on its own schedule, the T1 API carries its own
    six-part version inside that, and the key formats change more slowly
    than either. An operator debugging "my key is refused" needs to know
    which of the three is out of step.
    """
    import argparse

    p = argparse.ArgumentParser(prog="gkey version")
    p.add_argument("--json", dest="as_json", action="store_true")
    ns = p.parse_args(args)

    from hypernix import __version__ as hypernix_version
    from hypernix.t1api.version import T1_VERSION_LONG, T1_VERSION_SHORT

    if ns.as_json:
        print(json.dumps({
            "hypernix": hypernix_version,
            "t1_api": {"short": T1_VERSION_SHORT, "long": T1_VERSION_LONG},
            "key_versions": {
                "latest": LATEST_KEY_VERSION.name,
                "default": DEFAULT_KEY_VERSION.name,
                "available": [
                    {
                        "name": v.name,
                        "family": v.family,
                        "prefix": v.prefix,
                        "summary": v.summary,
                        "supports_admin": v.supports_admin,
                        "supports_access_level": v.supports_access_level,
                        "body_length": v.body_length,
                    }
                    for v in KEY_VERSIONS
                ],
                "reserved": [
                    {"name": v.name, "reason": v.unavailable_reason}
                    for v in RESERVED_KEY_VERSIONS
                ],
            },
        }, indent=2))
        return 0

    lines = [
        f"[bold]HyperNix:[/bold]    {hypernix_version}",
        f"[bold]T1 API:[/bold]      t1 v{T1_VERSION_SHORT}  [dim]({T1_VERSION_LONG})[/dim]",
        f"[bold]Key format:[/bold]  {LATEST_KEY_VERSION.name} is the latest; "
        f"[dim]gkey create mints {DEFAULT_KEY_VERSION.name} unless told otherwise[/dim]",
    ]
    if _HAS_RICH:
        _print_panel("\n".join(lines), title="gkey version")
    else:
        print(f"HyperNix:   {hypernix_version}")
        print(f"T1 API:     t1 v{T1_VERSION_SHORT} ({T1_VERSION_LONG})")
        print(f"Key format: {LATEST_KEY_VERSION.name} is the latest; "
              f"gkey create mints {DEFAULT_KEY_VERSION.name} unless told otherwise")
        print()

    rows = []
    for version in KEY_VERSIONS:
        notes = []
        if version is LATEST_KEY_VERSION:
            notes.append("latest")
        if version is DEFAULT_KEY_VERSION:
            notes.append("default")
        if not version.supports_admin:
            notes.append("never admin")
        if version.body_length:
            notes.append(f"{version.body_length}-char body")
        rows.append([
            version.name,
            version.prefix,
            version.summary,
            ", ".join(notes) or "—",
        ])
    _print_table(
        ["Version", "Prefix", "What it is", "Notes"], rows, title="Issuable key formats"
    )

    for version in RESERVED_KEY_VERSIONS:
        _print_rich(
            f"\n[yellow]{version.name}[/yellow] is not issuable. "
            f"{version.unavailable_reason}",
            style="yellow",
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `gkey` console script."""
    raw = list(sys.argv[1:] if argv is None else argv)

    if not raw or raw[0] in ("-h", "--help"):
        if _HAS_RICH:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            console = Console()
            title = Text("gkey", style="bold cyan")
            title.append(" — Gatekeeper + Keymaster unified CLI", style="dim")
            t = Table(show_header=True, header_style="bold magenta", border_style="cyan")
            t.add_column("Command")
            t.add_column("Description")
            cmds = [
                ("create", "Generate a new T1 API key"),
                ("revoke", "Revoke an existing key"),
                ("list", "List all keys; `list id <key-id>` for detail"),
                ("stats", "Show usage statistics and access logs"),
                ("quota", "View or set rate-limit quotas"),
                ("permissions", "Show permission scopes for a key"),
                ("rotate", "Rotate (replace) a key with a fresh one"),
                ("export", "Export key(s) to a JSON file"),
                ("import", "Import key(s) from a JSON file"),
                ("version", "HyperNix, T1 API, and key format versions"),
            ]
            for cmd, desc in cmds:
                t.add_row(f"[green]{cmd}[/green]", desc)
            console.print(Panel.fit(title))
            console.print(t)
            console.print("\n[dim]Run `gkey <subcommand> --help` for detailed options.[/dim]")
        else:
            print(_USAGE)
        return 0

    if raw[0] in ("-V", "--version"):
        from hypernix import __version__
        print(f"gkey (hypernix {__version__})")
        print("Run `gkey version` for the T1 API and key format versions.")
        return 0

    cmd, rest = raw[0], raw[1:]

    # `gkey list id <key-id>` — detect and reroute
    if cmd == "list" and rest and rest[0] == "id":
        return _cmd_list_id(rest[1:])

    dispatch = {
        "create": _cmd_create,
        "revoke": _cmd_revoke,
        "list": _cmd_list,
        "stats": _cmd_stats,
        "quota": _cmd_quota,
        "permissions": _cmd_permissions,
        "rotate": _cmd_rotate,
        "export": _cmd_export,
        "import": _cmd_import,
        "version": _cmd_version,
    }

    if cmd not in dispatch:
        print(f"Unknown subcommand: {cmd!r}\n", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        return 1

    try:
        return dispatch[cmd](rest)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:  # noqa: BLE001
        print(f"[gkey {cmd}] Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
