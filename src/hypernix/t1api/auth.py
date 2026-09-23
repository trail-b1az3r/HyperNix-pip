"""t1api.auth — the T1 API's authentication/authorization glue.

Per the spec's "HYPERNIX-PIP INTEGRATION" requirement, this module does
**not** reimplement key storage, rotation, or quota enforcement — that
already exists in :mod:`hypernix.keymaster` (:class:`Keymaster`) and
:mod:`hypernix.gatekeeper` (:class:`Gatekeeper`), both of which already
speak the T1 key format and scope model described in the spec. T1AuthService
is a thin adapter that:

* turns Keymaster/Gatekeeper's exceptions into stable :class:`T1APIError`
  codes for the HTTP layer,
* adds a second, distinct credential type — short-lived **scoped tokens**
  (``POST /auth/token``) — for clients that shouldn't hold the raw T1 key
  on every request,
* implements the admin-only "convert a normal T1 token into an admin
  token" operation from the spec, expressed as promoting a target key to
  ``KeyType.ADMIN`` and re-issuing it via :meth:`Keymaster.rotate`-style
  recreate (never mutating scopes on a live key string in place — a scope
  change gets a new key, consistent with how Keymaster already treats
  rotation as "replace, don't mutate").

Scoped tokens are intentionally NOT a full JWT implementation — no new
required dependency for something this small. They're
``T1S.<payload_b64>.<sig_b64>``, HMAC-SHA256 signed with
``T1APIConfig.token_secret``. Losing/rotating the secret invalidates every
outstanding scoped token immediately, which is the desired revocation
story for a stateless token (see ``wiki/T1-API.md#authentication``).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass

from hypernix.security.gatekeeper import Gatekeeper, QuotaViolation
from hypernix.security.keymaster import Keymaster, KeyMeta, KeyScope, KeyType

from ..security.t2keys import looks_like_t2
from .errors import T1APIError, T1ErrorCode

logger = logging.getLogger(__name__)

_TOKEN_PREFIX = "T1S"


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


@dataclass
class AuthContext:
    """Everything a route handler needs after authenticating a request."""

    key_meta: KeyMeta
    scopes: set[KeyScope]
    via_scoped_token: bool = False

    # --- T2 (T1 v1.0.26.8.1.0) ---------------------------------------
    # Empty/zero for an ordinary T1 key. Present on a context that
    # authenticated with a T2 key, because T1 has nowhere to put these
    # and dropping them would make a level-1 key indistinguishable from
    # a level-9 one at every call site downstream.
    t2_family: str = ""
    t2_access_level: int = 0
    t2_is_admin: bool = False
    t2_sspkid: str = ""
    #: The registered device a T2C (v2.1) key came from (0.72.6).
    t2c_device_id: str = ""

    @property
    def key_id(self) -> str:
        return self.key_meta.key_id

    @property
    def is_admin(self) -> bool:
        return self.key_meta.key_type == KeyType.ADMIN and KeyScope.ADMIN in self.scopes

    @property
    def via_t2(self) -> bool:
        return bool(self.t2_family)

    def meets_access_level(self, required: int) -> bool:
        """Does this context clear access level *required* (1-9)?

        A plain T1 key has no level and clears everything: T1 predates
        the concept, and retroactively assigning existing keys a level
        would lock working deployments out of their own endpoints.
        """
        if not self.via_t2:
            return True
        return self.t2_access_level >= required


@dataclass
class ScopedToken:
    token: str
    key_id: str
    scopes: list[str]
    expires_at: float


class T1AuthService:
    """Adapter over Keymaster + Gatekeeper for the T1 API HTTP layer."""

    def __init__(
        self,
        keymaster: Keymaster,
        gatekeeper: Gatekeeper,
        *,
        token_secret: str,
        default_ttl_seconds: int = 3600,
        accept_t2_keys: bool = True,
        accept_t1_keys: bool = True,
        t2c_authority=None,
    ) -> None:
        self.keymaster = keymaster
        self.gatekeeper = gatekeeper
        #: Opens T2C (v2.1) keys. Made on first use, beside the key store,
        #: so a server that never sees one never writes an RSA key.
        self._t2c_authority = t2c_authority
        #: Recognise T2 keys (T1 v1.0.26.8.1.0). Configurable so a
        #: deployment can stay strictly T1 during a migration.
        self.accept_t2_keys = accept_t2_keys
        #: Accept keys presented in the bare T1 spelling. Turning this off
        #: is the other end of the same migration: a deployment that has
        #: finished moving to T2 stops accepting the older spelling.
        #:
        #: It narrows the *spelling*, not the key store. A T2 key still
        #: authenticates against the T1 key behind it, because
        #: :meth:`T2KeyGenerator.to_t1` is what does the lookup — so
        #: turning this off does not orphan any existing key, it only
        #: requires the holder to present its T2 form.
        self.accept_t1_keys = accept_t1_keys
        if not accept_t1_keys and not accept_t2_keys:
            # Nothing could ever authenticate. Refuse at construction
            # rather than serving a process that rejects every request
            # with a message about the key rather than the config.
            raise ValueError(
                "accept_t1_keys and accept_t2_keys are both off: no key of any "
                "family could authenticate. Enable at least one."
            )
        self._token_secret = token_secret
        self.default_ttl_seconds = default_ttl_seconds
        if not token_secret:
            logger.warning(
                "t1api.auth: T1_TOKEN_SECRET is not set — scoped tokens (POST /auth/token) "
                "will be signed with an ephemeral in-process secret and become invalid on "
                "restart. Set T1_TOKEN_SECRET for any deployment with more than one worker "
                "process or that needs tokens to survive a restart."
            )
            import secrets as _secrets

            self._token_secret = _secrets.token_hex(32)
        # 0.0, not "now": the throttle exists to stop repeated reloads, not
        # to delay the first one. Starting at the current time would make
        # the first unknown key — the exact case this refresh is for —
        # wait out the whole interval before it could be found.
        self._last_key_reload = 0.0

    # ------------------------------------------------------------------
    # Raw T1 key validation
    # ------------------------------------------------------------------

    # Keymaster loads its key files once, at construction. A key created
    # after the server started — which is exactly what the documented
    # quickstart tells you to do (`gkey create`, then point waiter at the
    # already-running server) — is therefore invisible until a restart.
    # validate_key() reloads once on an unknown key to close that gap.
    #
    # The reload is throttled because it is disk I/O reachable by an
    # unauthenticated caller: without a floor, a stream of garbage keys
    # would force a directory scan per request, which is a denial-of-
    # service handed out for free. One reload every few seconds bounds
    # that to noise while still picking up a newly-minted key promptly.
    _RELOAD_MIN_INTERVAL_SECONDS = 5.0

    def _maybe_reload_keys(self) -> bool:
        """Re-read the key store, at most once per interval.

        Returns True if a reload actually happened, so the caller knows
        whether a retry is worth attempting.
        """
        now = time.monotonic()
        if now - self._last_key_reload < self._RELOAD_MIN_INTERVAL_SECONDS:
            return False
        self._last_key_reload = now
        try:
            self.keymaster._load_all()
        except Exception:  # noqa: BLE001 - a failed refresh must not mask the auth error
            logger.warning("t1api.auth: could not refresh the key store", exc_info=True)
            return False
        return True

    @property
    def t2c(self):
        """The :class:`~hypernix.security.t2c.T2CAuthority` for this key store."""
        if self._t2c_authority is None:
            from pathlib import Path

            from ..security.t2c import T2CAuthority

            self._t2c_authority = T2CAuthority(Path(self.keymaster.store_dir) / "t2c")
        return self._t2c_authority

    def _validate_t2c(self, key_str: str) -> AuthContext:
        """Open a T2C key and authenticate the T2 key sealed inside it."""
        from ..security.rotorvault import RotorvaultError
        from ..security.t2c import T2CError

        try:
            t2_raw, device_id = self.t2c.open(key_str)
        except (T2CError, RotorvaultError) as exc:
            raise T1APIError(
                T1ErrorCode.AUTH_INVALID_KEY, str(exc),
                details={"key_family": "T2C"}, http_status=401,
            ) from exc
        context = self._validate_t2(t2_raw)
        context.t2_family = "T2C"
        context.t2c_device_id = device_id
        return context

    def _accepted_families(self) -> list[str]:
        families = []
        if self.accept_t1_keys:
            families.append("T1")
        if self.accept_t2_keys:
            families.extend(["T2", "T2S", "T2C"])
        return families

    def validate_key(self, key_str: str) -> AuthContext:
        """Validate a raw T1 **or T2** key string.

        Used by ``POST /auth/t1/validate`` and as the fallback path when
        a request presents a raw key instead of a scoped token.

        T1 v1.0.26.8.1.0 added the T2 branch. A T2 key is converted to
        its T1 form (:meth:`T2KeyGenerator.to_t1`, which preserves the
        body and is deterministic) and then authenticated exactly as a
        T1 key would be — so a T2 key works against the existing key
        store with no migration, and the access level and SSPKID that T1
        has nowhere to put are carried on the returned context instead of
        being silently discarded.
        """
        if looks_like_t2(key_str):
            if not self.accept_t2_keys:
                raise T1APIError(
                    T1ErrorCode.AUTH_INVALID_KEY,
                    "This server does not accept T2 keys.",
                    details={"key_family": "T2", "accepted": self._accepted_families()},
                    http_status=401,
                )
            if key_str.startswith("T2C_"):
                return self._validate_t2c(key_str)
            return self._validate_t2(key_str)
        if key_str.startswith("T2CK_"):
            # The kit is what makes keys, not a key. Sending it means it
            # has left the machine it was meant to stay on.
            raise T1APIError(
                T1ErrorCode.AUTH_INVALID_KEY,
                "That is a T2C kit, not a key. Keep the kit on the client and send the "
                "day's key it makes (waiter does this for you); if the kit was sent "
                "somewhere it should not have been, revoke its device.",
                details={"key_family": "T2CK"}, http_status=401,
            )
        if not self.accept_t1_keys:
            # Deliberately says what to present rather than just refusing:
            # the holder of a T1 key that has been wrapped as T2 has a
            # working credential and only needs the other spelling.
            raise T1APIError(
                T1ErrorCode.AUTH_INVALID_KEY,
                "This server accepts T2 keys only. Present the T2 form of this key.",
                details={"key_family": "T1", "accepted": self._accepted_families()},
                http_status=401,
            )
        return self._authenticate_with_reload(key_str)

    def _authenticate_with_reload(self, key_str: str) -> AuthContext:
        """Authenticate a T1 key, refreshing the store once if it is unknown.

        A key created after this process started is not in the in-memory
        table yet, so the first lookup misses and a refresh is worth one
        try before calling it invalid.

        Shared with the T2 path deliberately. It used to live inline in
        :meth:`validate_key`, which meant :meth:`_validate_t2` — reaching
        the store through :meth:`_validate_uncached` — never refreshed at
        all. A T1 key minted against a running server worked; the *same
        key* presented in its T2 or T2S spelling was refused until the
        server restarted, which is as confusing a failure as this codebase
        can produce: the key is real, registered, and correctly typed.
        """
        try:
            meta = self.gatekeeper.authenticate(key_str)
        except (ValueError, PermissionError):
            # Unknown key: it may simply have been created after this
            # process started. Refresh once and try again — if it is still
            # unknown, fall through to the normal error handling below so
            # the caller sees the real message.
            if self._maybe_reload_keys():
                try:
                    meta = self.gatekeeper.authenticate(key_str)
                except (ValueError, PermissionError):
                    return self._validate_uncached(key_str)
                return AuthContext(key_meta=meta, scopes=set(meta.scopes))
            return self._validate_uncached(key_str)
        return AuthContext(key_meta=meta, scopes=set(meta.scopes))

    def _validate_t2(self, key_str: str) -> AuthContext:
        """Authenticate a T2 key by way of its T1 equivalent.

        Two things are deliberate here. The conversion is *not* stored —
        the T1 form is derived on every call, which is safe because
        :meth:`to_t1` is deterministic, and which avoids keeping a second
        copy of a credential around. And the T2-only facts (access level,
        family, SSPKID) ride on the returned :class:`AuthContext` rather
        than being dropped, because the endpoints added in this release
        need them and a context that quietly loses them would make a
        level-1 key indistinguishable from a level-9 one.
        """
        from ..security.t2keys import T2KeyGenerator, T2Type

        try:
            parsed = T2KeyGenerator.parse(key_str)
        except ValueError as exc:
            raise T1APIError(T1ErrorCode.AUTH_INVALID_KEY, str(exc), http_status=401) from exc

        equivalent = T2KeyGenerator.to_t1(parsed)
        try:
            context = self._authenticate_with_reload(equivalent)
        except T1APIError as exc:
            # The underlying path only ever knew about T1 keys, so it says
            # "Unknown or unregistered T1 key" to someone holding a T2S
            # key — which reads as though the wrong *kind* of key was
            # presented, when in fact the right kind simply is not in this
            # server's key store.
            #
            # That distinction is the whole difficulty with a T2 key: it
            # is a spelling of a T1 key, so it authenticates by being
            # converted back and looked up. A T2 key generated on its own
            # belongs to no key store and authenticates as nothing, no
            # matter how well-formed it is. Saying so, and naming the
            # command that mints a registered one, is the difference
            # between a two-minute fix and an afternoon.
            if exc.code is T1ErrorCode.AUTH_INVALID_KEY:
                raise T1APIError(
                    T1ErrorCode.AUTH_INVALID_KEY,
                    f"This {parsed.family.value} key is well-formed but is not "
                    f"registered on this server.",
                    details={
                        "key_family": parsed.family.value,
                        "reason": "not_in_key_store",
                        "explanation": (
                            f"A {parsed.family.value} key is a spelling of a T1 key "
                            "rather than a separate credential: it is converted back "
                            "to its T1 form and looked up in the key store. A key "
                            "generated on its own is in no key store and authenticates "
                            "as nothing."
                        ),
                        "remedy": (
                            "Mint one on the server: "
                            f"gkey create -v {'v2short' if parsed.family is T2Type.T2S else 'v2'}"
                        ),
                    },
                    http_status=401,
                ) from exc
            raise
        context.t2_family = parsed.family.value
        context.t2_access_level = parsed.access_level
        context.t2_is_admin = parsed.is_admin
        context.t2_sspkid = str(parsed.sspkid) if parsed.sspkid else ""
        if parsed.family is T2Type.T2S:
            # A T2S key is 26 typeable characters. Outside HyperLink it
            # gets read and non-admin write and nothing else, whatever
            # the underlying T1 key is scoped for — see t2keys.T2Key.permits.
            context.scopes = {
                scope for scope in context.scopes if scope.value in ("read", "write")
            }
        return context

    def _validate_uncached(self, key_str: str) -> AuthContext:
        """The original validation path, raising the stable error codes."""
        try:
            meta = self.gatekeeper.authenticate(key_str)
        except ValueError as exc:
            raise T1APIError(T1ErrorCode.AUTH_INVALID_KEY, str(exc), http_status=401) from exc
        except PermissionError as exc:
            msg = str(exc)
            # NOTE: Keymaster.revoke() removes the key from the active
            # lookup table entirely (archived, not just flagged inactive —
            # see keymaster.py::revoke), so Gatekeeper.authenticate() can
            # never actually produce a "has been revoked" message for a
            # key looked up by its raw string: a revoked key and a key
            # that never existed are indistinguishable at this layer, both
            # surfacing as "Unknown or unregistered T1 key." We map that
            # case to AUTH_INVALID_KEY rather than pretending we can tell
            # them apart. AUTH_REVOKED_KEY stays reachable in principle
            # (e.g. Keymaster.get(key_id)-based lookups by key_id, used
            # elsewhere) and is kept in T1ErrorCode for that reason.
            if "expired" in msg:
                code = T1ErrorCode.AUTH_EXPIRED_KEY
            else:
                code = T1ErrorCode.AUTH_INVALID_KEY
            raise T1APIError(code, msg, http_status=401) from exc
        return AuthContext(key_meta=meta, scopes=set(meta.scopes))

    # ------------------------------------------------------------------
    # Scoped tokens
    # ------------------------------------------------------------------

    def issue_scoped_token(
        self,
        key_str: str,
        *,
        ttl_seconds: int | None = None,
        scopes: list[str] | None = None,
    ) -> ScopedToken:
        """Exchange a validated raw T1 key for a short-lived scoped token.

        If *scopes* is given, it must be a subset of the underlying key's
        scopes (a token can only narrow permissions, never widen them).
        """
        ctx = self.validate_key(key_str)
        granted = {KeyScope(s) for s in scopes} if scopes else set(ctx.scopes)
        if not granted.issubset(ctx.scopes):
            raise T1APIError(
                T1ErrorCode.AUTH_INSUFFICIENT_SCOPE,
                "Requested scopes exceed the underlying T1 key's scopes.",
                details={"key_scopes": sorted(s.value for s in ctx.scopes)},
                http_status=403,
            )
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        exp = time.time() + max(1, ttl)
        payload = {
            "key_id": ctx.key_id,
            "scopes": sorted(s.value for s in granted),
            "exp": exp,
        }
        payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        sig = hmac.new(self._token_secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256)
        token = f"{_TOKEN_PREFIX}.{payload_b64}.{_b64url_encode(sig.digest())}"
        return ScopedToken(token=token, key_id=ctx.key_id, scopes=payload["scopes"], expires_at=exp)

    def verify_scoped_token(self, token: str) -> AuthContext:
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != _TOKEN_PREFIX:
            raise T1APIError(T1ErrorCode.AUTH_INVALID_TOKEN, "Malformed scoped token.", http_status=401)
        _, payload_b64, sig_b64 = parts
        expected_sig = hmac.new(
            self._token_secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256
        ).digest()
        try:
            given_sig = _b64url_decode(sig_b64)
        except Exception as exc:  # noqa: BLE001 - malformed input, not a real error path
            raise T1APIError(T1ErrorCode.AUTH_INVALID_TOKEN, "Malformed scoped token.", http_status=401) from exc
        if not hmac.compare_digest(expected_sig, given_sig):
            raise T1APIError(T1ErrorCode.AUTH_INVALID_TOKEN, "Scoped token signature mismatch.", http_status=401)
        try:
            payload = json.loads(_b64url_decode(payload_b64))
        except Exception as exc:  # noqa: BLE001
            raise T1APIError(T1ErrorCode.AUTH_INVALID_TOKEN, "Malformed scoped token payload.", http_status=401) from exc
        if payload.get("exp", 0) < time.time():
            raise T1APIError(T1ErrorCode.AUTH_EXPIRED_TOKEN, "Scoped token has expired.", http_status=401)

        key_meta = self.keymaster.get(payload["key_id"])
        if key_meta is None or not key_meta.is_valid:
            raise T1APIError(
                T1ErrorCode.AUTH_INVALID_TOKEN,
                "The T1 key backing this scoped token is no longer valid.",
                http_status=401,
            )
        scopes = {KeyScope(s) for s in payload.get("scopes", [])}
        return AuthContext(key_meta=key_meta, scopes=scopes, via_scoped_token=True)

    # ------------------------------------------------------------------
    # Quota check wrapper (used by routers before expensive operations)
    # ------------------------------------------------------------------

    def check_quota(self, key_id: str, *, endpoint: str = "", model: str = "", tokens_requested: int = 0) -> None:
        try:
            self.gatekeeper.check_quota(
                key_id, endpoint=endpoint, model=model, tokens_requested=tokens_requested
            )
        except QuotaViolation as exc:
            raise T1APIError(T1ErrorCode.RATE_LIMITED, str(exc), http_status=429) from exc

    # ------------------------------------------------------------------
    # Admin operations
    # ------------------------------------------------------------------

    def rotate_own_key(self, key_id: str) -> KeyMeta:
        try:
            return self.keymaster.rotate(key_id)
        except KeyError as exc:
            raise T1APIError(T1ErrorCode.NOT_FOUND, str(exc), http_status=404) from exc

    def admin_rotate(self, *, requester: AuthContext, target_key_id: str, promote_to_admin: bool = False) -> KeyMeta:
        """POST /auth/t1/admin/rotate — admin-only. Rotates *target_key_id*
        and, if requested, promotes the newly-issued key to an admin key.

        "support conversion of a normal T1 token into an admin token only
        when the authenticated user has the required permission" — the
        permission check here IS the admin-scope requirement on
        *requester*, enforced before anything else runs.
        """
        if not requester.is_admin:
            raise T1APIError(
                T1ErrorCode.AUTH_ADMIN_REQUIRED,
                "Only an admin-scoped T1 key may rotate another key or grant admin.",
                http_status=403,
            )
        target = self.keymaster.get(target_key_id)
        if target is None:
            raise T1APIError(T1ErrorCode.NOT_FOUND, f"Key {target_key_id} not found.", http_status=404)

        new_meta = self.keymaster.rotate(target_key_id)
        if promote_to_admin and new_meta.key_type != KeyType.ADMIN:
            # Recreate as an admin-typed key preserving the rest of the
            # settings, then revoke the just-rotated intermediate key —
            # mirrors Keymaster.rotate's own "replace, don't mutate" model.
            promoted = self.keymaster.create(
                key_type=KeyType.ADMIN,
                scopes=set(new_meta.scopes) | {KeyScope.ADMIN},
                expires_at=new_meta.expires_at,
                usage_cap=new_meta.usage_cap,
                request_limit=new_meta.request_limit,
                prefix=new_meta.prefix,
                tags={**new_meta.tags, "promoted_from": new_meta.key_id},
                rotation_window=new_meta.rotation_window,
                note=f"promoted to admin by {requester.key_id[:8]}",
            )
            self.keymaster.revoke(new_meta.key_id, reason="promoted to admin key")
            logger.info(
                "t1api.auth: key %s promoted to admin key %s by %s",
                target_key_id[:8],
                promoted.key_id[:8],
                requester.key_id[:8],
            )
            return promoted
        return new_meta


__all__ = ["AuthContext", "ScopedToken", "T1AuthService"]
