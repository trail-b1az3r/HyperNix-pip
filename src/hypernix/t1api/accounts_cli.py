"""``t1-accounts`` — manage web accounts from the machine they live on.

    t1-accounts list
    t1-accounts create mason
    t1-accounts reset mason
    t1-accounts disable mason
    t1-accounts sessions mason
    t1-accounts modes

Why this exists alongside the web pages: there is no password reset by
email, because there is no mail server here and inventing one would be
worse than the gap. Somebody who forgets their password gets it reset by
whoever has shell access to the server — which on a single-user install
is the same person.

Every command here bypasses the registration policy. An administrator
with shell access on the machine is past the point where a registration
policy means anything, and having to open registration to create the
first account would be theatre.

Passwords are read with :func:`getpass.getpass`, never taken as an
argument. A password on a command line is in the shell history, in
``ps``, and in whatever collects either.
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from typing import Any

__all__ = ["main", "cli_main"]


def _store(args) -> Any:
    """Open the account store over the same backend the server uses."""
    from .accounts import (
        AccountStore,
        invite_codes_from_env,
        registration_mode_from_env,
    )
    from .config import T1APIConfig
    from .db import make_backend

    cfg = T1APIConfig()
    backend = make_backend(
        db_path=args.db or cfg.db_path, database_url=cfg.database_url
    )
    return AccountStore(
        backend,
        registration=registration_mode_from_env(),
        invite_codes=invite_codes_from_env(),
    )


def _ask_password(confirm: bool = True) -> str:
    """Read a password twice, without echo.

    Never an argument. A password on a command line is in the shell
    history, in `ps`, and in whatever collects either.
    """
    from .accounts import PASSWORD_MIN_LENGTH

    while True:
        first = getpass.getpass("Password: ")
        if len(first) < PASSWORD_MIN_LENGTH:
            print(
                f"  At least {PASSWORD_MIN_LENGTH} characters, please.",
                file=sys.stderr,
            )
            continue
        if not confirm:
            return first
        again = getpass.getpass("Again: ")
        if first != again:
            print("  Those do not match.", file=sys.stderr)
            continue
        return first


def _fmt_time(when: float) -> str:
    if not when:
        return "never"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(when))


def _cmd_list(args) -> int:
    store = _store(args)
    accounts = store.list_accounts()
    if args.as_json:
        print(json.dumps(
            [a.to_dict(include_private=True) for a in accounts], indent=2
        ))
        return 0
    if not accounts:
        print("No accounts. Create one with `t1-accounts create <username>`.")
        return 0
    print(f"{'username':20} {'admin':6} {'state':9} {'keys':>5}  last login")
    for account in accounts:
        state = "disabled" if not account.active else (
            "locked" if account.locked else "active"
        )
        print(
            f"{account.username:20} {'yes' if account.is_admin else '-':6} "
            f"{state:9} {len(account.key_ids):>5}  "
            f"{_fmt_time(account.last_login_at)}"
        )
    return 0


def _cmd_create(args) -> int:
    from .accounts import AccountError

    store = _store(args)
    password = _ask_password()
    try:
        account = store.create(
            args.username, password,
            display_name=args.display_name or "",
            email=args.email or "",
            is_admin=args.admin,
            # See the module docstring: shell access outranks the policy.
            bypass_registration_check=True,
        )
    except AccountError as exc:
        print(f"t1-accounts: {exc.message}", file=sys.stderr)
        return 1
    print(f"Created {account.username}" + (" (admin)" if account.is_admin else ""))
    print("  Sign in at /accounts/login and mint a key there.")
    return 0


def _cmd_reset(args) -> int:
    from .accounts import AccountError

    store = _store(args)
    account = store.by_username(args.username)
    if account is None:
        print(f"t1-accounts: no account {args.username!r}", file=sys.stderr)
        return 1
    password = _ask_password()
    try:
        store.set_password(account.account_id, password)
    except AccountError as exc:
        print(f"t1-accounts: {exc.message}", file=sys.stderr)
        return 1
    print(f"Password set for {account.username}.")
    # Not a side effect worth hiding: a password reset that left the old
    # sessions alive would not be a password reset.
    print("  Every existing session for that account has been ended.")
    return 0


def _cmd_enable(args, active: bool) -> int:
    store = _store(args)
    account = store.by_username(args.username)
    if account is None:
        print(f"t1-accounts: no account {args.username!r}", file=sys.stderr)
        return 1
    store.set_active(account.account_id, active)
    print(f"{account.username} is now {'active' if active else 'disabled'}.")
    if not active:
        print("  Its sessions have been ended. Its API keys still work —")
        print("  revoke those with `gkey revoke` if that is the intent.")
    return 0


def _cmd_unlock(args) -> int:
    store = _store(args)
    account = store.by_username(args.username)
    if account is None:
        print(f"t1-accounts: no account {args.username!r}", file=sys.stderr)
        return 1
    store._update(account.account_id, {"failed_logins": 0, "locked_until": 0.0})
    print(f"{account.username} unlocked.")
    return 0


def _cmd_sessions(args) -> int:
    store = _store(args)
    account = store.by_username(args.username)
    if account is None:
        print(f"t1-accounts: no account {args.username!r}", file=sys.stderr)
        return 1
    if args.revoke:
        count = store.revoke_all_sessions(account.account_id)
        print(f"Ended {count} session(s) for {account.username}.")
        return 0
    sessions = store.sessions_for(account.account_id)
    if not sessions:
        print(f"{account.username} has no open sessions.")
        return 0
    print(f"{'started':17} {'expires':17} {'from':16} agent")
    for session in sessions:
        print(
            f"{_fmt_time(session.created_at):17} "
            f"{_fmt_time(session.expires_at):17} "
            f"{(session.client_ip or '-'):16} {session.user_agent[:40]}"
        )
    return 0


def _cmd_purge(args) -> int:
    store = _store(args)
    print(f"Removed {store.purge_expired_sessions()} expired session(s).")
    return 0


def _cmd_modes(_args) -> int:
    """Explain the four deployment shapes, since picking one is the
    first decision and the least obvious."""
    from .webauth import MODES

    print("Where the sign-up page can live:\n")
    for mode in MODES.values():
        print(f"  {mode.name}")
        for line in _wrap(mode.summary, 68):
            print(f"      {line}")
        print(f"      $ {mode.how} hypernix-t1 start")
        flags = []
        flags.append("Secure cookies" if mode.secure_cookies else "no TLS required")
        if mode.behind_proxy:
            flags.append(f"real client IP from {mode.forwarded_header}")
        print(f"      ({', '.join(flags)})")
        print()
    print("Set it with T1_ACCOUNTS_MODE, and turn accounts on with")
    print("T1_ACCOUNTS_ENABLED=1. Registration is 'closed' until you set")
    print("T1_ACCOUNTS_REGISTRATION to open, invite or first-user.")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width) or [""]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="t1-accounts",
        description=(
            "Manage T1 web accounts — the sign-up flow that lets somebody "
            "obtain their first T1 key without one."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Passwords are always prompted for, never taken as an argument:\n"
            "a password on a command line is in the shell history, in ps, and\n"
            "in whatever collects either.\n"
        ),
    )
    parser.add_argument("--db", help="SQLite path (default: the server's)")
    parser.add_argument("--json", dest="as_json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="Every account")
    listing.set_defaults(func=_cmd_list)

    create = sub.add_parser("create", help="Create an account")
    create.add_argument("username")
    create.add_argument("--display-name")
    create.add_argument("--email")
    create.add_argument("--admin", action="store_true",
                        help="Make it an admin. The first account is one anyway.")
    create.set_defaults(func=_cmd_create)

    reset = sub.add_parser("reset", help="Set a password without the old one")
    reset.add_argument("username")
    reset.set_defaults(func=_cmd_reset)

    disable = sub.add_parser("disable", help="Stop an account signing in")
    disable.add_argument("username")
    disable.set_defaults(func=lambda a: _cmd_enable(a, False))

    enable = sub.add_parser("enable", help="Let it sign in again")
    enable.add_argument("username")
    enable.set_defaults(func=lambda a: _cmd_enable(a, True))

    unlock = sub.add_parser("unlock", help="Clear a failed-login lockout")
    unlock.add_argument("username")
    unlock.set_defaults(func=_cmd_unlock)

    sessions = sub.add_parser("sessions", help="Open sessions for an account")
    sessions.add_argument("username")
    sessions.add_argument("--revoke", action="store_true",
                          help="End all of them")
    sessions.set_defaults(func=_cmd_sessions)

    purge = sub.add_parser("purge", help="Delete expired sessions")
    purge.set_defaults(func=_cmd_purge)

    modes = sub.add_parser("modes", help="The four places the pages can live")
    modes.set_defaults(func=_cmd_modes)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - a CLI reports, it does not traceback
        print(f"t1-accounts: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
