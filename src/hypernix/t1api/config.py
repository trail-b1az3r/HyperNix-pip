"""t1api.config — environment-variable configuration for the T1 API.

Every value is overridable via an environment variable so the module
stays "easy to mount into an existing Python server" without editing
code. A ``.env`` file is loaded when present (and ``python-dotenv`` is
installed); a missing python-dotenv is a soft failure, since env vars
set some other way still work.

Two properties this file is responsible for and that the rest of the
package depends on:

**Secrets never leak through the config surface.**
:meth:`T1APIConfig.public_dict` — what ``GET /config`` returns — is an
explicit *allowlist* rather than "everything except a blocklist", so a
secret field added later cannot leak by being forgotten. Nothing in this
module logs a secret's value, and :meth:`describe_secrets` reports only
whether each one is *set*.

**Production is validated, not assumed.**
:meth:`validate_for_production` is the Beta 3 "production configuration
validation" deliverable. It refuses to start a production deployment
that is missing a token secret, still on SQLite, running mTLS behind a
proxy with no trusted-proxy list, or accepting any CORS origin — the
handful of settings that are merely inconvenient in development and
genuinely dangerous in production. It returns *every* problem rather
than the first, because an operator fixing a deployment should get one
list, not five restarts.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import T1APIError, T1ErrorCode
from .mtls import TLSSettings

logger = logging.getLogger(__name__)

PRODUCTION_ENVIRONMENTS = ("production", "prod")


def _load_dotenv_if_present(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=p)
    except ImportError:
        # Soft-fail: minimal hand-rolled parser so .env still works
        # without the optional python-dotenv dependency installed.
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _tuple_env(name: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in os.environ.get(name, "").split(",") if v.strip())


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        logger.warning("t1api.config: %s is not an integer; using default %d", name, default)
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        logger.warning("t1api.config: %s is not a number; using default %s", name, default)
        return default


def _is_local_or_tailscale(host: str) -> bool:
    """True for loopback, ``*.ts.net``, and the 100.64/10 tailnet range.

    Used by the production check: these are the addresses where plain
    HTTP is not a plaintext-over-the-network problem, either because the
    traffic never leaves the machine or because WireGuard already
    encrypted it.
    """
    import ipaddress

    lowered = host.strip().lower().strip("[]")
    if lowered in ("localhost", "127.0.0.1", "::1") or lowered.endswith(".ts.net"):
        return True
    try:
        addr = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return addr.is_loopback or addr in ipaddress.ip_network("100.64.0.0/10")


@dataclass
class T1APIConfig:
    """Runtime configuration, resolved once at ``create_app()`` time."""

    # --- General -------------------------------------------------------
    mount_prefix: str = field(default_factory=lambda: os.environ.get("T1_MOUNT_PREFIX", ""))
    environment: str = field(default_factory=lambda: os.environ.get("T1_ENVIRONMENT", "development"))

    # --- Storage ---------------------------------------------------------
    db_path: str | None = field(default_factory=lambda: os.environ.get("T1_DB_PATH"))
    # Beta 3: set this and every store moves to PostgreSQL. Unset = SQLite.
    database_url: str | None = field(default_factory=lambda: os.environ.get("T1_DATABASE_URL"))

    # --- Model registry --------------------------------------------------
    registry_path: str | None = field(default_factory=lambda: os.environ.get("T1_MODEL_REGISTRY_PATH"))
    enable_example_models: bool = field(
        default_factory=lambda: _bool_env("T1_ENABLE_EXAMPLE_MODELS", False)
    )

    # --- Trusted network (0.72.4) -----------------------------------------
    # Off by default, and that default is the whole point: keyless access
    # is something an administrator turns on knowing what it costs, never
    # something that happens because a machine has a private address.
    trusted_network: bool = field(
        default_factory=lambda: _bool_env("T1_TRUSTED_NETWORK", False)
    )
    trusted_network_lan: bool = field(
        default_factory=lambda: _bool_env("T1_TRUSTED_NETWORK_LAN", True)
    )
    trusted_network_tailnet: bool = field(
        default_factory=lambda: _bool_env("T1_TRUSTED_NETWORK_TAILNET", True)
    )
    #: Partial administrative functions for trusted origins. Separate
    #: from keyless *access* because they are separate decisions: letting
    #: the phone on your sofa read models is not letting it stop training.
    trusted_network_partial_admin: bool = field(
        default_factory=lambda: _bool_env("T1_TRUSTED_NETWORK_PARTIAL_ADMIN", False)
    )

    # --- Routing ----------------------------------------------------------
    routing_policy_path: str | None = field(
        default_factory=lambda: os.environ.get("T1_ROUTING_POLICY_PATH")
    )
    default_plan: str = field(default_factory=lambda: os.environ.get("T1_DEFAULT_PLAN", "free"))

    # --- Auth --------------------------------------------------------------
    # HMAC secret used to sign scoped tokens minted by POST /auth/token.
    # Never logged; never returned by any endpoint. Rotate by changing this
    # value — outstanding scoped tokens become invalid immediately (they
    # are stateless, so there is nothing else to revoke).
    token_secret: str = field(default_factory=lambda: os.environ.get("T1_TOKEN_SECRET", ""))
    scoped_token_default_ttl_seconds: int = field(
        default_factory=lambda: _int_env("T1_TOKEN_DEFAULT_TTL", 3600)
    )

    # --- Usage -------------------------------------------------------------
    usage_reset_period_seconds: float = field(
        default_factory=lambda: _float_env("T1_USAGE_RESET_PERIOD_SECONDS", 86400.0)
    )

    # --- Networking --------------------------------------------------------
    cors_allow_origins: tuple[str, ...] = field(
        default_factory=lambda: _tuple_env("T1_CORS_ALLOW_ORIGINS")
    )
    # Addresses permitted to set X-Forwarded-For / client-cert headers.
    # Empty means "trust no proxy", so the peer address is used directly.
    trusted_proxies: tuple[str, ...] = field(default_factory=lambda: _tuple_env("T1_TRUSTED_PROXIES"))
    # The three-way access decision from the design principle: may a
    # client that is neither allowlisted nor blocklisted connect at all?
    allow_unlisted_clients: bool = field(
        default_factory=lambda: _bool_env("T1_ALLOW_UNLISTED_CLIENTS", True)
    )
    network_policy_enabled: bool = field(
        default_factory=lambda: _bool_env("T1_NETWORK_POLICY_ENABLED", True)
    )

    # --- Rate limiting ------------------------------------------------------
    rate_limit_enabled: bool = field(default_factory=lambda: _bool_env("T1_RATE_LIMIT_ENABLED", True))
    # JSON: either a list of rule objects or {"rules": [...]}. Empty =
    # t1api.ratelimit.DEFAULT_RULES, so "unset" never means "unlimited".
    rate_limit_rules_json: str = field(default_factory=lambda: os.environ.get("T1_RATE_LIMIT_RULES", ""))

    # --- Audit --------------------------------------------------------------
    audit_enabled: bool = field(default_factory=lambda: _bool_env("T1_AUDIT_ENABLED", True))

    # --- TLS / mTLS ----------------------------------------------------------
    tls_certfile: str | None = field(default_factory=lambda: os.environ.get("T1_TLS_CERTFILE"))
    tls_keyfile: str | None = field(default_factory=lambda: os.environ.get("T1_TLS_KEYFILE"))
    tls_ca_certs: str | None = field(default_factory=lambda: os.environ.get("T1_TLS_CA_CERTS"))
    mtls_require_client_cert: bool = field(
        default_factory=lambda: _bool_env("T1_MTLS_REQUIRE_CLIENT_CERT", False)
    )
    mtls_behind_proxy: bool = field(default_factory=lambda: _bool_env("T1_MTLS_BEHIND_PROXY", False))
    mtls_allowed_subjects: tuple[str, ...] = field(
        default_factory=lambda: _tuple_env("T1_MTLS_ALLOWED_SUBJECTS")
    )
    mtls_allowed_fingerprints: tuple[str, ...] = field(
        default_factory=lambda: _tuple_env("T1_MTLS_ALLOWED_FINGERPRINTS")
    )

    # --- Remote deployment ----------------------------------------------------
    # Shared HMAC secret for server-to-server module transfer. Both ends
    # must hold the same value; unset means this server neither pushes
    # nor accepts module transfers.
    deploy_secret: str = field(default_factory=lambda: os.environ.get("T1_DEPLOY_SECRET", ""))
    max_transfer_bytes: int = field(
        default_factory=lambda: _int_env("T1_MAX_TRANSFER_BYTES", 256 * 1024 * 1024)
    )
    allow_private_deploy_targets: bool = field(
        default_factory=lambda: _bool_env("T1_ALLOW_PRIVATE_DEPLOY_TARGETS", False)
    )
    module_storage_dir: str | None = field(default_factory=lambda: os.environ.get("T1_MODULE_STORAGE_DIR"))

    # --- LM Studio bridge (T1 v1.0.26.8.0.1) ----------------------------------
    # The bridge is *off* unless an address is configured or discovery is
    # explicitly enabled. A server that probes the LAN for open ports on
    # every start is a surprise nobody asked for; a server that talks to
    # localhost:1234 because someone typed a URL is not.
    lmstudio_enabled: bool = field(default_factory=lambda: _bool_env("T1_LMSTUDIO_ENABLED", True))
    lmstudio_url: str = field(default_factory=lambda: os.environ.get("T1_LMSTUDIO_URL", ""))
    lmstudio_api_key: str = field(default_factory=lambda: os.environ.get("T1_LMSTUDIO_API_KEY", ""))
    lmstudio_discovery: bool = field(
        default_factory=lambda: _bool_env("T1_LMSTUDIO_DISCOVERY", False)
    )
    lmstudio_timeout_seconds: float = field(
        default_factory=lambda: _float_env("T1_LMSTUDIO_TIMEOUT", 300.0)
    )

    # --- HyperLink (T1 v1.0.26.8.0.1) -----------------------------------------
    # The phone-facing surface: pairing, chat sessions, attachments,
    # Hugging Face resolution. On by default because the T1 API is
    # already authenticated and HyperLink adds no *unauthenticated*
    # surface beyond the pairing redemption endpoint, which needs a code
    # an operator minted by hand.
    hyperlink_enabled: bool = field(default_factory=lambda: _bool_env("T1_HYPERLINK_ENABLED", True))
    hyperlink_public_url: str = field(
        default_factory=lambda: os.environ.get("T1_HYPERLINK_PUBLIC_URL", "")
    )
    hyperlink_files_dir: str | None = field(
        default_factory=lambda: os.environ.get("T1_HYPERLINK_FILES_DIR")
    )
    hyperlink_max_upload_bytes: int = field(
        default_factory=lambda: _int_env("T1_HYPERLINK_MAX_UPLOAD_BYTES", 64 * 1024 * 1024)
    )
    hyperlink_advertised_port: int = field(
        default_factory=lambda: _int_env("T1_HYPERLINK_PORT", 8000)
    )
    #: Whether that port was actually configured, or is just the default.
    #:
    #: The two cannot be told apart from the value — 8000 is both — and
    #: they mean opposite things: a configured port must be advertised as
    #: given (a proxy knows something the request cannot), while a default
    #: one must give way to the port the request really arrived on.
    #: Captured here, beside the value, so the answer cannot drift from
    #: what was read.
    hyperlink_port_is_explicit: bool = field(
        default_factory=lambda: bool(os.environ.get("T1_HYPERLINK_PORT"))
    )
    hyperlink_pairing_ttl_seconds: float = field(
        default_factory=lambda: _float_env("T1_HYPERLINK_PAIRING_TTL", 600.0)
    )
    # Hugging Face token used for gated-repo resolution. Never returned
    # by any endpoint — only its set/unset state, via describe_secrets().
    hf_token: str = field(
        default_factory=lambda: os.environ.get("T1_HF_TOKEN") or os.environ.get("HF_TOKEN", "")
    )

    # --- Identity (0.72.1) ----------------------------------------------------
    # What this deployment calls itself. Reported by GET /status so
    # `waiter -F <name>` has something to match; falls back to the
    # hostname, which is what an operator would have typed anyway.
    server_name: str = field(default_factory=lambda: os.environ.get("T1_SERVER_NAME", ""))
    # The 54-character Host ID, when this deployment has been issued one.
    # Distinct from a V1 Server ID and from an SSPKID by construction —
    # see hypernix.security.t2keys.
    host_id: str = field(default_factory=lambda: os.environ.get("T1_HOST_ID", ""))
    server_id: str = field(default_factory=lambda: os.environ.get("T1_SERVER_ID", ""))

    # --- Hugging Face downloads (0.72.1) --------------------------------------
    # HyperLink can ask the server to fetch a model. The server does it
    # because the server is the machine that will run it; pulling
    # gigabytes onto a phone to send them back is the wrong direction on
    # every axis.
    hf_downloads_enabled: bool = field(
        default_factory=lambda: _bool_env("T1_HF_DOWNLOADS_ENABLED", True)
    )
    hf_download_dir: str | None = field(
        default_factory=lambda: os.environ.get("T1_HF_DOWNLOAD_DIR")
    )

    # --- Backups (T1 v1.0.26.8.1.0) -------------------------------------------
    # Where /backup/list and /backup/restore keep snapshots. Never
    # contains key material — see hypernix.t1api.backup.EXCLUDED.
    backup_dir: str | None = field(default_factory=lambda: os.environ.get("T1_BACKUP_DIR"))
    backup_max_count: int = field(default_factory=lambda: _int_env("T1_BACKUP_MAX_COUNT", 20))
    # Recognise and accept T2 keys. On by default in this release: T1
    # v1.0.26.8.1.0's whole point is that a T2 key works against a T1
    # server. The switch exists for a deployment that wants to stay
    # strictly T1 during a migration.
    accept_t2_keys: bool = field(default_factory=lambda: _bool_env("T1_ACCEPT_T2_KEYS", True))
    # The other end of the same migration: stop accepting the bare T1
    # spelling once a deployment has finished moving to T2. On by
    # default, because turning it off is a breaking change for every
    # client that has not been reissued.
    #
    # This narrows the accepted spelling, not the key store — a T2 key
    # authenticates against the T1 key behind it either way — so no
    # existing key is orphaned by turning it off. Both off is refused at
    # startup rather than serving a process nothing can authenticate to.
    accept_t1_keys: bool = field(default_factory=lambda: _bool_env("T1_ACCEPT_T1_KEYS", True))

    # Where the key store lives. Unset means Keymaster's own default
    # (~/.hypernix/keymaster), which is the long-standing behaviour and
    # stays the default here.
    #
    # Setting it makes a deployment self-contained, which matters for two
    # cases the default cannot serve: an installer that mints a key into
    # a config directory and needs the server to read that same store,
    # and two servers on one machine, which would otherwise share one
    # store and see each other's keys.
    # Mint a loopback-only admin key on first start, so a new server is
    # reachable without running gkey on the box by hand. On by default:
    # the alternative is a server nobody can configure, which is how
    # "HyperLink cannot connect" starts. Off for a deployment that
    # provisions credentials some other way.
    # What this server does with a key that bills somewhere else.
    #
    # "allow" accepts the caller's own payment arrangement. "deny"
    # refuses it and points at this server's payment page, which is the
    # right answer for an operator who sells access themselves. "separate"
    # accepts the request but requires payment on a *different* key, so
    # the credential that identifies the caller and the one that spends
    # money have separate lifetimes.
    billing_key_policy: str = field(
        default_factory=lambda: (
            os.environ.get("T1_BILLING_KEY_POLICY", "allow").strip().lower() or "allow"
        )
    )
    #: Where to send someone whose billing key was refused. Without it
    #: "deny" is a dead end, so the refusal says so rather than pretending
    #: there is somewhere to go.
    payment_url: str = field(default_factory=lambda: os.environ.get("T1_PAYMENT_URL", ""))

    bootstrap_key_enabled: bool = field(
        default_factory=lambda: _bool_env("T1_BOOTSTRAP_KEY", True)
    )

    keymaster_dir: str | None = field(
        default_factory=lambda: os.environ.get("T1_KEYMASTER_DIR") or None
    )

    # --- Misc -----------------------------------------------------------------
    # Destructive operations (DELETE /servers/{id}, DELETE /modules/{id})
    # require ?confirm=true when this is on. "Require explicit
    # confirmation for destructive operations", made a setting so a
    # scripted environment can opt out deliberately rather than by
    # accident.
    require_destructive_confirmation: bool = field(
        default_factory=lambda: _bool_env("T1_REQUIRE_DESTRUCTIVE_CONFIRMATION", True)
    )
    expose_docs: bool = field(default_factory=lambda: _bool_env("T1_EXPOSE_DOCS", True))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, *, load_dotenv: bool = True, dotenv_path: str | Path = ".env") -> T1APIConfig:
        if load_dotenv:
            _load_dotenv_if_present(dotenv_path)
        return cls()

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() in PRODUCTION_ENVIRONMENTS

    @property
    def storage_backend(self) -> str:
        return "postgres" if self.database_url else "sqlite"

    def tls_settings(self) -> TLSSettings:
        return TLSSettings(
            certfile=self.tls_certfile,
            keyfile=self.tls_keyfile,
            ca_certs=self.tls_ca_certs,
            require_client_cert=self.mtls_require_client_cert,
            behind_proxy=self.mtls_behind_proxy,
            trusted_proxies=self.trusted_proxies,
            allowed_subjects=self.mtls_allowed_subjects,
            allowed_fingerprints=self.mtls_allowed_fingerprints,
        )

    def rate_limit_rules(self) -> Any:
        """Parsed rate-limit rules, or None to use the defaults.

        A malformed ``T1_RATE_LIMIT_RULES`` falls back to the defaults
        with a warning rather than raising: unparseable rules must not
        mean *no* rules, which is what raising here would effectively
        produce if a deployment caught the error and carried on.
        """
        if not self.rate_limit_rules_json.strip():
            return None
        try:
            return json.loads(self.rate_limit_rules_json)
        except json.JSONDecodeError:
            logger.warning(
                "t1api.config: T1_RATE_LIMIT_RULES is not valid JSON — falling back to default rules."
            )
            return None

    def describe_secrets(self) -> dict[str, bool]:
        """Which secrets are set. Values never included — this is what
        ``GET /status`` uses to tell an operator what's missing without
        telling anyone what it is."""
        return {
            "token_secret": bool(self.token_secret),
            "deploy_secret": bool(self.deploy_secret),
            "tls_keyfile": bool(self.tls_keyfile),
            "tls_ca_certs": bool(self.tls_ca_certs),
            "database_url": bool(self.database_url),
            "lmstudio_api_key": bool(self.lmstudio_api_key),
            "hf_token": bool(self.hf_token),
        }

    def public_dict(self) -> dict[str, object]:
        """Explicit allowlist of config surfaced by GET /config. Never
        includes ``token_secret``, ``deploy_secret``, ``database_url``,
        or any future secret field."""
        return {
            "environment": self.environment,
            "mount_prefix": self.mount_prefix,
            "storage_backend": self.storage_backend,
            "enable_example_models": self.enable_example_models,
            "routing_policy_path": self.routing_policy_path,
            "default_plan": self.default_plan,
            "usage_reset_period_seconds": self.usage_reset_period_seconds,
            "scoped_token_default_ttl_seconds": self.scoped_token_default_ttl_seconds,
            "rate_limit_enabled": self.rate_limit_enabled,
            "audit_enabled": self.audit_enabled,
            "network_policy_enabled": self.network_policy_enabled,
            "allow_unlisted_clients": self.allow_unlisted_clients,
            "require_destructive_confirmation": self.require_destructive_confirmation,
            "remote_deployment_enabled": bool(self.deploy_secret),
            "max_transfer_bytes": self.max_transfer_bytes,
            # T1 v1.0.26.8.0.1. The LM Studio *address* is not a secret
            # (it is a LAN or tailnet host an operator wants to see in
            # /config when debugging "why can't it find my GPU"); the
            # API key for it is, and lives in describe_secrets() only.
            "lmstudio_enabled": self.lmstudio_enabled,
            "lmstudio_url": self.lmstudio_url,
            "lmstudio_discovery": self.lmstudio_discovery,
            "hyperlink_enabled": self.hyperlink_enabled,
            "hyperlink_public_url": self.hyperlink_public_url,
            "hyperlink_max_upload_bytes": self.hyperlink_max_upload_bytes,
            **self.tls_settings().public_dict(),
        }

    # ------------------------------------------------------------------
    # Production validation
    # ------------------------------------------------------------------

    def production_problems(self) -> list[str]:
        """Every reason this config is unsafe for production, as text.

        Split from :meth:`validate_for_production` so a CLI (``waiter
        doctor``, the smoke tester) can *report* problems without the
        raise, and the server can refuse to start on the same list.
        """
        problems: list[str] = []

        if not self.token_secret:
            problems.append(
                "T1_TOKEN_SECRET is not set. Scoped tokens would be signed with an ephemeral "
                "per-process secret, so they break across restarts and across workers."
            )
        elif len(self.token_secret) < 32:
            problems.append(
                f"T1_TOKEN_SECRET is only {len(self.token_secret)} characters. Use at least 32 "
                "(e.g. `python -c 'import secrets;print(secrets.token_hex(32))'`)."
            )

        if not self.database_url:
            problems.append(
                "T1_DATABASE_URL is not set. SQLite is the documented development backend; "
                "production deployments should use PostgreSQL."
            )

        if "*" in self.cors_allow_origins:
            problems.append(
                "T1_CORS_ALLOW_ORIGINS contains '*'. A wildcard origin on an authenticated API "
                "lets any site drive it with a user's credentials."
            )

        if not self.network_policy_enabled:
            problems.append(
                "T1_NETWORK_POLICY_ENABLED=0 disables IP allowlist/blocklist enforcement entirely."
            )

        if not self.rate_limit_enabled:
            problems.append(
                "T1_RATE_LIMIT_ENABLED=0 disables rate limiting. The spec requires limits to be "
                "applied before expensive model operations."
            )

        if not self.audit_enabled:
            problems.append(
                "T1_AUDIT_ENABLED=0 disables the audit log. Administrator actions would go unrecorded."
            )

        tls = self.tls_settings()
        if not tls.tls_enabled and not self.mtls_behind_proxy:
            problems.append(
                "No TLS configured (T1_TLS_CERTFILE/T1_TLS_KEYFILE) and not marked as running "
                "behind a TLS-terminating proxy (T1_MTLS_BEHIND_PROXY). Credentials would cross "
                "the network in plaintext."
            )
        problems.extend(tls.validation_errors())

        # T1 v1.0.26.8.0.1. An LM Studio bridge pointed at a plaintext
        # address that is not loopback sends prompts — and whatever the
        # user pasted into them — across the network in the clear.
        # Tailscale is the exception: it is encrypted at the transport,
        # so http:// over 100.64/10 is not the same mistake.
        if self.lmstudio_enabled and self.lmstudio_url.startswith("http://"):
            host = self.lmstudio_url.split("//", 1)[1].split(":")[0].split("/")[0]
            if not _is_local_or_tailscale(host):
                problems.append(
                    f"T1_LMSTUDIO_URL points at {host} over plain HTTP. Prompts and completions "
                    "would cross the network unencrypted. Use loopback, a Tailscale address, "
                    "or put TLS in front of LM Studio."
                )

        if self.deploy_secret and len(self.deploy_secret) < 32:
            problems.append(
                f"T1_DEPLOY_SECRET is only {len(self.deploy_secret)} characters; use at least 32."
            )

        if self.enable_example_models:
            problems.append(
                "T1_ENABLE_EXAMPLE_MODELS=1 exposes the placeholder registry entries. Those models "
                "are documented as not real; point T1_MODEL_REGISTRY_PATH at real data instead."
            )

        if self.expose_docs:
            # Not fatal on its own — plenty of internal deployments want
            # Swagger. Reported so the choice is deliberate.
            problems.append(
                "T1_EXPOSE_DOCS=1 serves Swagger/ReDoc publicly. Set T1_EXPOSE_DOCS=0 if the "
                "OpenAPI surface should not be reachable in production."
            )

        return problems

    def validate_for_production(self, *, strict: bool | None = None) -> None:
        """Raise ``CONFIG_INVALID`` if this config is unsafe for production.

        Args:
            strict: Force validation on (or off) regardless of
                ``environment``. Defaults to "validate iff
                ``T1_ENVIRONMENT`` says production", which is what makes
                this safe to call unconditionally from ``create_app``.
        """
        should_validate = self.is_production if strict is None else strict
        if not should_validate:
            return
        problems = self.production_problems()
        if not problems:
            return
        raise T1APIError(
            T1ErrorCode.CONFIG_INVALID,
            "This configuration is not safe for a production deployment:\n  - "
            + "\n  - ".join(problems)
            + "\n\nFix these, or set T1_ENVIRONMENT to something other than 'production' "
            "to run with development defaults.",
            details={"problems": problems, "count": len(problems)},
            http_status=500,
        )


__all__ = ["PRODUCTION_ENVIRONMENTS", "T1APIConfig"]
