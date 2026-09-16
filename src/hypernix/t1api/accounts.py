"""t1api.accounts — sign up for a T1 key without already having one.

The T1 API authenticates with T1 keys, and until now the only way to get
one was for somebody who already had an admin key to mint it for you.
That is correct for a hosted service with a billing relationship and
absurd for the case this package is mostly used in: one person, their own
machine, wanting to point a client at their own server.

This is the other path. A person opens a web page, picks a username and a
password, and mints their own key from their own account page. No T1 key
is needed to get a T1 key.

It is not a second authentication system for the API
-----------------------------------------------------
Every API route still authenticates with a T1 key, exactly as before.
What an account gets you is a *session with the account pages*, and the
only privileged thing those pages do is mint and revoke keys belonging to
that account. An account is a way to obtain a credential, not a
credential.

That split is deliberate. It means a bug here cannot widen anyone's API
access beyond what their own keys already allow, and it means the
accounts system can be switched off entirely
(``T1_ACCOUNTS_ENABLED=0``, the default) on a deployment that mints keys
some other way.

Passwords
---------
``hashlib.scrypt``, 16-byte random salt per account, n=2^15, r=8, p=1 —
memory-hard, in the standard library, no new dependency for something as
load-bearing as this. Verification is :func:`hmac.compare_digest`.

The parameters are stored *in* the hash string, so raising them later
does not invalidate existing passwords: an account whose hash carries the
old cost verifies against the old cost and is re-hashed at the new one on
its next successful login. A migration that logs everyone out is a
migration nobody runs.

What this deliberately does not do
-----------------------------------
No password reset by email — there is no mail server here and inventing
one would be worse than the gap. An admin resets a password with
``hypernix-t1 accounts reset``; on a single-user install, the person with
the disk is the person with the account.

No "remember me" that outlives the session cookie. No account recovery
questions. No third-party OAuth. Every one of those is a way in, and the
smallest number of ways in is the right number for something that mints
API keys.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "AccountError",
    "Account",
    "Session",
    "AccountStore",
    "hash_password",
    "verify_password",
    "PASSWORD_MIN_LENGTH",
    "SESSION_TTL_SECONDS",
    "USERNAME_RE",
]

#: scrypt cost. n=2^15 with r=8 is ~32 MB per hash, which is the usual
#: interactive-login setting: enough that a GPU cannot try billions per
#: second, little enough that a login is not a noticeable pause.
_SCRYPT_N = 1 << 15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_KEY_BYTES = 32

#: Long enough that the scrypt cost is doing the work rather than being
#: the only thing between an attacker and a four-character password. Not
#: a complexity rule: length is the property that matters and a rule
#: demanding a symbol produces "Password1!" on every system that has one.
PASSWORD_MIN_LENGTH = 12

#: How long a login lasts. A day, because the thing behind it is a page
#: that mints API keys and not a mail client.
SESSION_TTL_SECONDS = 24 * 60 * 60

#: Deliberately narrow: these become part of a URL and a log line.
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")

#: Failed logins before an account stops accepting them for a while, and
#: for how long. Per-account rather than per-IP: an attacker picks their
#: own IP and cannot pick which account they are trying to get into.
MAX_FAILED_LOGINS = 8
LOCKOUT_SECONDS = 15 * 60


class AccountError(Exception):
    """An account could not be created, found, or authenticated.

    Carries a ``code`` so a router can map it to a status without
    matching on the message.
    """

    def __init__(self, message: str, *, code: str = "account_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def hash_password(password: str, *, n: int = _SCRYPT_N) -> str:
    """Hash *password* into a self-describing string.

    ``scrypt$<n>$<r>$<p>$<salt-hex>$<hash-hex>``. The cost parameters
    live in the string so that raising them later does not invalidate
    every existing password — see the module docstring.
    """
    if not isinstance(password, str) or not password:
        raise AccountError("A password is required.", code="password_required")
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_BYTES,
        # scrypt's memory ceiling defaults to 32 MB, which n=2^15 r=8
        # sits exactly on; without this, raising n at all raises
        # "memory limit exceeded" rather than working.
        maxmem=n * _SCRYPT_R * 256,
    )
    return f"scrypt${n}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Whether *password* produces *stored*. Constant-time.

    Returns False rather than raising on a malformed stored hash: a
    corrupt record should fail closed, and an exception here would
    distinguish "this account is broken" from "wrong password" to whoever
    is guessing.
    """
    if not password or not stored:
        return False
    try:
        scheme, n_raw, r_raw, p_raw, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        n, r, p = int(n_raw), int(r_raw), int(p_raw)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        logger.warning("t1api.accounts: a stored password hash is malformed")
        return False
    try:
        derived = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
            dklen=len(expected), maxmem=n * r * 256,
        )
    except ValueError:
        return False
    return hmac.compare_digest(derived, expected)


def needs_rehash(stored: str, *, n: int = _SCRYPT_N) -> bool:
    """Whether *stored* was hashed at a lower cost than the current one."""
    try:
        scheme, n_raw, *_rest = stored.split("$")
    except (ValueError, AttributeError):
        return True
    return scheme != "scrypt" or int(n_raw) < n


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Account:
    """One person who can sign in and mint keys for themselves."""

    username: str
    password_hash: str = field(repr=False)
    account_id: str = ""
    display_name: str = ""
    email: str = ""
    is_admin: bool = False
    active: bool = True
    created_at: float = field(default_factory=time.time)
    last_login_at: float = 0.0
    failed_logins: int = 0
    locked_until: float = 0.0
    #: T1 key ids minted through this account, so the account page can
    #: list and revoke them. The key *material* is never stored here —
    #: that lives in the Keymaster, encrypted, as it always did.
    key_ids: list[str] = field(default_factory=list)

    @property
    def locked(self) -> bool:
        return self.locked_until > time.time()

    def to_dict(self, *, include_private: bool = False) -> dict[str, Any]:
        """Serialise. ``password_hash`` is never included, at any level."""
        public = {
            "account_id": self.account_id,
            "username": self.username,
            "display_name": self.display_name,
            "is_admin": self.is_admin,
            "active": self.active,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
            "key_ids": list(self.key_ids),
        }
        if include_private:
            public["email"] = self.email
            public["locked"] = self.locked
            public["failed_logins"] = self.failed_logins
        return public


@dataclass
class Session:
    """A signed-in browser.

    The token is returned once, to be set as a cookie, and only its
    SHA-256 is kept. A stolen session store is then not a set of usable
    cookies — the same reason the Keymaster does not store key material
    in the clear.
    """

    session_id: str
    account_id: str
    token_hash: str = field(repr=False)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    #: Bound at login and checked on use. A session cookie that starts
    #: being presented from a different address is the shape of a stolen
    #: cookie, and refusing it costs a laptop moving between networks one
    #: extra login.
    client_ip: str = ""
    user_agent: str = ""
    #: Paired with the session and required on every state-changing
    #: request. See :meth:`AccountStore.check_csrf`.
    csrf_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))

    @property
    def expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class AccountStore:
    """Accounts and sessions, over the T1 API's SQL backend.

    Takes the same backend object every other T1 store takes, so an
    install on Postgres gets Postgres and one on SQLite gets SQLite
    without this module knowing which.
    """

    def __init__(
        self,
        backend: Any,
        *,
        registration: str = "closed",
        invite_codes: tuple[str, ...] = (),
        session_ttl: int = SESSION_TTL_SECONDS,
        bind_sessions_to_ip: bool = True,
    ) -> None:
        self.backend = backend
        self.session_ttl = int(session_ttl)
        self.bind_sessions_to_ip = bind_sessions_to_ip
        self.invite_codes = tuple(code for code in invite_codes if code)
        if registration not in ("open", "invite", "first-user", "closed"):
            raise AccountError(
                f"Unknown registration mode {registration!r}. One of: open, "
                "invite, first-user, closed.",
                code="bad_config",
            )
        self.registration = registration
        self._init_schema()

    # -- schema -------------------------------------------------------

    def _init_schema(self) -> None:
        with self.backend.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS t1_accounts (
                    account_id     TEXT PRIMARY KEY,
                    username       TEXT NOT NULL UNIQUE,
                    password_hash  TEXT NOT NULL,
                    display_name   TEXT NOT NULL DEFAULT '',
                    email          TEXT NOT NULL DEFAULT '',
                    is_admin       INTEGER NOT NULL DEFAULT 0,
                    active         INTEGER NOT NULL DEFAULT 1,
                    created_at     REAL NOT NULL,
                    last_login_at  REAL NOT NULL DEFAULT 0,
                    failed_logins  INTEGER NOT NULL DEFAULT 0,
                    locked_until   REAL NOT NULL DEFAULT 0,
                    key_ids        TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS t1_sessions (
                    session_id   TEXT PRIMARY KEY,
                    account_id   TEXT NOT NULL,
                    token_hash   TEXT NOT NULL UNIQUE,
                    csrf_token   TEXT NOT NULL,
                    created_at   REAL NOT NULL,
                    expires_at   REAL NOT NULL,
                    client_ip    TEXT NOT NULL DEFAULT '',
                    user_agent   TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS t1_sessions_account "
                "ON t1_sessions (account_id)"
            )

    # -- reading ------------------------------------------------------

    @staticmethod
    def _row_to_account(row: Any) -> Account:
        values = list(row)
        return Account(
            account_id=values[0],
            username=values[1],
            password_hash=values[2],
            display_name=values[3],
            email=values[4],
            is_admin=bool(values[5]),
            active=bool(values[6]),
            created_at=float(values[7]),
            last_login_at=float(values[8]),
            failed_logins=int(values[9]),
            locked_until=float(values[10]),
            key_ids=[k for k in str(values[11]).split(",") if k],
        )

    _COLUMNS = (
        "account_id, username, password_hash, display_name, email, is_admin, "
        "active, created_at, last_login_at, failed_logins, locked_until, key_ids"
    )

    def get(self, account_id: str) -> Account | None:
        with self.backend.connect() as conn:
            row = conn.execute(
                f"SELECT {self._COLUMNS} FROM t1_accounts WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        return self._row_to_account(row) if row else None

    def by_username(self, username: str) -> Account | None:
        with self.backend.connect() as conn:
            row = conn.execute(
                f"SELECT {self._COLUMNS} FROM t1_accounts WHERE username = ?",
                (self.normalise_username(username),),
            ).fetchone()
        return self._row_to_account(row) if row else None

    def count(self) -> int:
        with self.backend.connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM t1_accounts").fetchone()
        return int(row[0]) if row else 0

    def list_accounts(self, limit: int = 100) -> list[Account]:
        with self.backend.connect() as conn:
            rows = conn.execute(
                f"SELECT {self._COLUMNS} FROM t1_accounts "
                "ORDER BY created_at ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._row_to_account(row) for row in rows]

    # -- creating -----------------------------------------------------

    @staticmethod
    def normalise_username(username: str) -> str:
        return str(username or "").strip().lower()

    def _check_username(self, username: str) -> str:
        normalised = self.normalise_username(username)
        if not USERNAME_RE.match(normalised):
            raise AccountError(
                "A username is 2-32 characters of lowercase letters, digits, "
                "dot, dash or underscore, and starts with a letter or digit.",
                code="bad_username",
            )
        return normalised

    @staticmethod
    def _check_password(password: str) -> None:
        if len(password or "") < PASSWORD_MIN_LENGTH:
            raise AccountError(
                f"A password must be at least {PASSWORD_MIN_LENGTH} characters. "
                "Length is the property that matters; there is no rule here "
                "about symbols.",
                code="weak_password",
            )
        if len(password) > 1024:
            # Not a strength rule. scrypt's cost is in the salt and the
            # parameters, not the input length, but hashing a megabyte
            # someone pasted is still work this server will do on an
            # unauthenticated request.
            raise AccountError(
                "That password is longer than 1024 characters.",
                code="password_too_long",
            )

    def can_register(self, invite_code: str = "") -> tuple[bool, str]:
        """Whether registration is open right now, and why not if not."""
        if self.registration == "open":
            return True, ""
        if self.registration == "closed":
            return False, (
                "This server is not accepting new accounts. An administrator "
                "can create one with `hypernix-t1 accounts create`."
            )
        if self.registration == "first-user":
            if self.count() == 0:
                return True, ""
            return False, (
                "This server accepts one account, created by whoever got here "
                "first, and it already has it."
            )
        if not self.invite_codes:
            return False, (
                "This server requires an invite code and has none configured, "
                "so nobody can register. Set T1_ACCOUNTS_INVITE_CODES."
            )
        # Constant-time against every configured code: a comparison that
        # short-circuits tells the caller how much of a code they guessed.
        supplied = str(invite_code or "")
        if any(hmac.compare_digest(supplied, code) for code in self.invite_codes):
            return True, ""
        return False, "That invite code is not valid."

    def create(
        self,
        username: str,
        password: str,
        *,
        display_name: str = "",
        email: str = "",
        is_admin: bool | None = None,
        invite_code: str = "",
        bypass_registration_check: bool = False,
    ) -> Account:
        """Create an account. Raises :class:`AccountError` on refusal.

        *bypass_registration_check* is for the CLI: an administrator with
        shell access on the machine is past the point where a
        registration policy means anything, and needing to open
        registration to create the first account would be theatre.
        """
        if not bypass_registration_check:
            allowed, reason = self.can_register(invite_code)
            if not allowed:
                raise AccountError(reason, code="registration_closed")

        normalised = self._check_username(username)
        self._check_password(password)
        if self.by_username(normalised) is not None:
            raise AccountError(
                f"The username {normalised!r} is taken.", code="username_taken"
            )

        # The first account is the admin. On a single-user install that
        # is the whole of the permission model, and on a larger one the
        # person who set the server up is the right first admin.
        if is_admin is None:
            is_admin = self.count() == 0

        account = Account(
            account_id=secrets.token_hex(16),
            username=normalised,
            password_hash=hash_password(password),
            display_name=display_name or normalised,
            email=email,
            is_admin=bool(is_admin),
        )
        with self.backend.connect() as conn:
            conn.execute(
                "INSERT INTO t1_accounts (account_id, username, password_hash, "
                "display_name, email, is_admin, active, created_at, "
                "last_login_at, failed_logins, locked_until, key_ids) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    account.account_id, account.username, account.password_hash,
                    account.display_name, account.email, int(account.is_admin),
                    1, account.created_at, 0.0, 0, 0.0, "",
                ),
            )
        logger.info(
            "t1api.accounts: created account %s (admin=%s)",
            account.username, account.is_admin,
        )
        return account

    # -- authenticating -----------------------------------------------

    def authenticate(self, username: str, password: str) -> Account:
        """Check a username and password. Raises on any failure.

        Every failure raises the same message. "No such user" and "wrong
        password" are different facts and telling them apart hands an
        attacker a list of which usernames exist — which is the first
        thing they want and the cheapest thing to withhold.
        """
        generic = AccountError(
            "That username and password do not match an account.",
            code="bad_credentials",
        )
        normalised = self.normalise_username(username)
        account = self.by_username(normalised)

        if account is None:
            # Hash anyway. Returning immediately makes a non-existent
            # username measurably faster to reject than a real one with a
            # wrong password, which is the same enumeration leak by a
            # different route.
            verify_password(password, hash_password("dummy-for-timing"))
            raise generic

        if not account.active:
            raise AccountError(
                "That account has been disabled.", code="account_disabled"
            )
        if account.locked:
            remaining = int(account.locked_until - time.time())
            raise AccountError(
                f"Too many failed attempts. Try again in {remaining // 60 + 1} "
                f"minute(s).",
                code="account_locked",
            )

        if not verify_password(password, account.password_hash):
            self._record_failure(account)
            raise generic

        # Correct password. Clear the counter, and take the opportunity
        # to move an old hash up to the current cost while the plaintext
        # is in hand -- the only moment it ever is.
        updates: dict[str, Any] = {
            "failed_logins": 0, "locked_until": 0.0, "last_login_at": time.time(),
        }
        if needs_rehash(account.password_hash):
            updates["password_hash"] = hash_password(password)
            logger.info(
                "t1api.accounts: re-hashed %s at the current cost", account.username
            )
        self._update(account.account_id, updates)
        account.failed_logins = 0
        account.locked_until = 0.0
        account.last_login_at = updates["last_login_at"]
        return account

    def _record_failure(self, account: Account) -> None:
        failures = account.failed_logins + 1
        locked_until = 0.0
        if failures >= MAX_FAILED_LOGINS:
            locked_until = time.time() + LOCKOUT_SECONDS
            logger.warning(
                "t1api.accounts: locked %s after %d failed logins",
                account.username, failures,
            )
        self._update(
            account.account_id,
            {"failed_logins": failures, "locked_until": locked_until},
        )

    def _update(self, account_id: str, values: dict[str, Any]) -> None:
        if not values:
            return
        assignments = ", ".join(f"{column} = ?" for column in values)
        with self.backend.connect() as conn:
            conn.execute(
                f"UPDATE t1_accounts SET {assignments} WHERE account_id = ?",
                (*values.values(), account_id),
            )

    def set_password(self, account_id: str, password: str) -> None:
        """Set a password without checking the old one. Admin path only."""
        self._check_password(password)
        self._update(account_id, {
            "password_hash": hash_password(password),
            "failed_logins": 0,
            "locked_until": 0.0,
        })
        # Every existing session, on every device. A password change that
        # left a stolen session alive would not be a password change.
        self.revoke_all_sessions(account_id)

    def change_password(self, account_id: str, current: str, new: str) -> None:
        """Set a password, having checked the old one. The self-service path."""
        account = self.get(account_id)
        if account is None:
            raise AccountError("No such account.", code="not_found")
        if not verify_password(current, account.password_hash):
            raise AccountError(
                "The current password is wrong.", code="bad_credentials"
            )
        self.set_password(account_id, new)

    def set_active(self, account_id: str, active: bool) -> None:
        self._update(account_id, {"active": int(bool(active))})
        if not active:
            self.revoke_all_sessions(account_id)

    def remember_key(self, account_id: str, key_id: str) -> None:
        """Record that a key belongs to this account."""
        account = self.get(account_id)
        if account is None:
            raise AccountError("No such account.", code="not_found")
        if key_id in account.key_ids:
            return
        self._update(
            account_id, {"key_ids": ",".join([*account.key_ids, key_id])}
        )

    def forget_key(self, account_id: str, key_id: str) -> None:
        account = self.get(account_id)
        if account is None:
            return
        remaining = [k for k in account.key_ids if k != key_id]
        self._update(account_id, {"key_ids": ",".join(remaining)})

    # -- sessions -----------------------------------------------------

    def start_session(
        self, account: Account, *, client_ip: str = "", user_agent: str = ""
    ) -> tuple[Session, str]:
        """Open a session. Returns ``(session, token)``.

        The token is returned exactly once, for the cookie. Only its
        hash is stored, so a copy of the session table is not a set of
        usable cookies.
        """
        token = secrets.token_urlsafe(32)
        session = Session(
            session_id=secrets.token_hex(16),
            account_id=account.account_id,
            token_hash=_token_hash(token),
            expires_at=time.time() + self.session_ttl,
            client_ip=client_ip or "",
            user_agent=(user_agent or "")[:256],
        )
        with self.backend.connect() as conn:
            conn.execute(
                "INSERT INTO t1_sessions (session_id, account_id, token_hash, "
                "csrf_token, created_at, expires_at, client_ip, user_agent) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.session_id, session.account_id, session.token_hash,
                    session.csrf_token, session.created_at, session.expires_at,
                    session.client_ip, session.user_agent,
                ),
            )
        return session, token

    def resolve_session(
        self, token: str, *, client_ip: str = ""
    ) -> tuple[Session, Account] | None:
        """The session and account a cookie names, or ``None``.

        Looked up by the token's hash, which is an indexed equality
        match on a uniformly-distributed value — so this leaks neither
        the token nor how many sessions exist.
        """
        if not token:
            return None
        with self.backend.connect() as conn:
            row = conn.execute(
                "SELECT session_id, account_id, token_hash, csrf_token, "
                "created_at, expires_at, client_ip, user_agent "
                "FROM t1_sessions WHERE token_hash = ?",
                (_token_hash(token),),
            ).fetchone()
        if row is None:
            return None
        values = list(row)
        session = Session(
            session_id=values[0], account_id=values[1], token_hash=values[2],
            csrf_token=values[3], created_at=float(values[4]),
            expires_at=float(values[5]), client_ip=values[6],
            user_agent=values[7],
        )
        if session.expired:
            self.end_session(session.session_id)
            return None
        if (
            self.bind_sessions_to_ip
            and session.client_ip
            and client_ip
            and session.client_ip != client_ip
        ):
            # A session cookie that starts arriving from a different
            # address is the shape of a stolen cookie. Ending it costs a
            # laptop that changed networks one extra login and costs an
            # attacker the whole session.
            logger.warning(
                "t1api.accounts: session %s presented from %s, bound to %s; "
                "ending it",
                session.session_id[:8], client_ip, session.client_ip,
            )
            self.end_session(session.session_id)
            return None

        account = self.get(session.account_id)
        if account is None or not account.active:
            self.end_session(session.session_id)
            return None
        return session, account

    def check_csrf(self, session: Session, supplied: str) -> bool:
        """Whether *supplied* matches this session's CSRF token.

        Needed because the session lives in a cookie, and a cookie is
        sent by the browser on any request to this origin including one
        a different site caused. ``SameSite=Lax`` covers most of it and
        this covers the rest; both, because the failure mode is somebody
        else minting an API key in your name.
        """
        return bool(supplied) and hmac.compare_digest(
            str(supplied), session.csrf_token
        )

    def end_session(self, session_id: str) -> bool:
        with self.backend.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM t1_sessions WHERE session_id = ?", (session_id,)
            )
        return bool(getattr(cursor, "rowcount", 0))

    def revoke_all_sessions(self, account_id: str) -> int:
        with self.backend.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM t1_sessions WHERE account_id = ?", (account_id,)
            )
        return int(getattr(cursor, "rowcount", 0) or 0)

    def purge_expired_sessions(self) -> int:
        with self.backend.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM t1_sessions WHERE expires_at > 0 AND expires_at < ?",
                (time.time(),),
            )
        return int(getattr(cursor, "rowcount", 0) or 0)

    def sessions_for(self, account_id: str) -> list[Session]:
        with self.backend.connect() as conn:
            rows = conn.execute(
                "SELECT session_id, account_id, token_hash, csrf_token, "
                "created_at, expires_at, client_ip, user_agent "
                "FROM t1_sessions WHERE account_id = ? ORDER BY created_at DESC",
                (account_id,),
            ).fetchall()
        found = []
        for row in rows:
            values = list(row)
            found.append(Session(
                session_id=values[0], account_id=values[1], token_hash=values[2],
                csrf_token=values[3], created_at=float(values[4]),
                expires_at=float(values[5]), client_ip=values[6],
                user_agent=values[7],
            ))
        return found


def registration_mode_from_env() -> str:
    """``T1_ACCOUNTS_REGISTRATION``, defaulting to the safe one."""
    return os.environ.get("T1_ACCOUNTS_REGISTRATION", "closed").strip().lower()


def invite_codes_from_env() -> tuple[str, ...]:
    raw = os.environ.get("T1_ACCOUNTS_INVITE_CODES", "")
    return tuple(code.strip() for code in raw.split(",") if code.strip())
