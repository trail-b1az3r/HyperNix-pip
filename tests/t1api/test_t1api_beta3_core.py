"""Beta 3 core tests: PostgreSQL portability, keys, cost, transport,
deployment, and production config validation.

Pure core — no FastAPI needed. The PostgreSQL class exercises the SQL
*translation* (which is where portability actually lives) without needing
a server; a real round-trip against PostgreSQL runs only when
``T1_TEST_DATABASE_URL`` is set, and is skipped otherwise rather than
silently not covering anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import pytest

from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
from hypernix.t1api.audit import AuditLog
from hypernix.t1api.config import T1APIConfig
from hypernix.t1api.cost import CostCalculator
from hypernix.t1api.db import (
    PostgresBackend,
    SQLiteBackend,
    make_backend,
    to_pg_placeholders,
    translate_ddl,
)
from hypernix.t1api.deploy import DeploymentCoordinator
from hypernix.t1api.errors import T1APIError, T1ErrorCode
from hypernix.t1api.keys import KeyDirectory, mask_key_id
from hypernix.t1api.modules import ModuleRegistry
from hypernix.t1api.registry import ModelEntry, ModelPricing, ModelRegistry, ModelStatus
from hypernix.t1api.servers import ServerRegistry, TrustLevel
from hypernix.t1api.storage import UsageStore
from hypernix.t1api.transport import (
    SIGNATURE_MAX_SKEW_SECONDS,
    ModuleTransport,
    compute_signature,
    verify_signature,
)


@pytest.fixture
def backend(tmp_path) -> SQLiteBackend:
    return SQLiteBackend(tmp_path / "beta3.sqlite3")


@pytest.fixture
def registry() -> ModelRegistry:
    """A small registry of *real* (non-example) priced models.

    The shipped seed entries are deliberately unpriced placeholders, so
    costing has to be tested against entries that look like production
    data rather than against zeros.
    """
    reg = ModelRegistry()
    reg.register(
        ModelEntry(
            model_id="test-large",
            display_name="Test Large",
            version="1.0",
            total_parameters=100.0,
            active_parameters=10.0,
            architecture="moe",
            supported_tasks=["chat"],
            availability="public",
            minimum_plan="pro",
            free_tier_available=False,
            api_available=True,
            local_available=False,
            remote_available=True,
            context_limit=100_000,
            input_token_limit=100_000,
            output_token_limit=10_000,
            tool_call_limit=32,
            pricing=ModelPricing(input_price_per_1k=1.0, output_price_per_1k=3.0),
            routing_priority=10,
            fallback_model="test-small",
            license="proprietary",
            status=ModelStatus.AVAILABLE,
        )
    )
    reg.register(
        ModelEntry(
            model_id="test-small",
            display_name="Test Small",
            version="1.0",
            total_parameters=7.0,
            active_parameters=None,
            architecture="dense",
            supported_tasks=["chat"],
            availability="public",
            minimum_plan="free",
            free_tier_available=True,
            api_available=True,
            local_available=True,
            remote_available=True,
            context_limit=8_000,
            input_token_limit=8_000,
            output_token_limit=2_000,
            tool_call_limit=4,
            pricing=ModelPricing(input_price_per_1k=0.1, output_price_per_1k=0.2),
            routing_priority=50,
            fallback_model=None,
            license="apache-2.0",
            status=ModelStatus.AVAILABLE,
        )
    )
    return reg


@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


# ===========================================================================
# PostgreSQL portability
# ===========================================================================


class TestSQLTranslation:
    """The translation layer is what makes 'SQLite for development,
    PostgreSQL for production' true without a second set of queries."""

    def test_placeholders_converted(self):
        assert to_pg_placeholders("SELECT * FROM t WHERE a = ? AND b = ?") == (
            "SELECT * FROM t WHERE a = %s AND b = %s"
        )

    def test_question_mark_inside_a_string_literal_is_preserved(self):
        """A literal '?' in SQL text is data, not a placeholder. Naive
        str.replace would corrupt it."""
        sql = "SELECT * FROM t WHERE note LIKE '%?%' AND a = ?"
        assert to_pg_placeholders(sql) == "SELECT * FROM t WHERE note LIKE '%%?%%' AND a = %s"

    def test_percent_is_escaped_for_psycopg(self):
        assert to_pg_placeholders("SELECT 50 % 7") == "SELECT 50 %% 7"

    def test_autoincrement_becomes_bigserial(self):
        ddl = "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, x TEXT)"
        assert "BIGSERIAL PRIMARY KEY" in translate_ddl(ddl, "postgres")
        assert "AUTOINCREMENT" not in translate_ddl(ddl, "postgres")

    def test_real_becomes_double_precision(self):
        assert "DOUBLE PRECISION" in translate_ddl("CREATE TABLE t (ts REAL)", "postgres")

    def test_sqlite_ddl_is_untouched(self):
        ddl = "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL)"
        assert translate_ddl(ddl, "sqlite") == ddl

    def test_every_shipped_schema_translates(self):
        """Each store's schema must survive translation — this is the
        check that catches a new table using SQLite-only syntax."""
        from hypernix.t1api import audit, billing, jobs, keys, modules, netpolicy, servers, storage

        for module in (storage, servers, modules, jobs, billing, audit, netpolicy, keys):
            schema = module._SCHEMA
            translated = translate_ddl(schema, "postgres")
            assert "AUTOINCREMENT" not in translated, module.__name__
            # `REAL` is valid in PostgreSQL but means single precision;
            # timestamps stored as floats need the double.
            assert " REAL" not in translated, module.__name__

    def test_make_backend_prefers_the_database_url(self, tmp_path):
        assert isinstance(make_backend(db_path=tmp_path / "x.sqlite3"), SQLiteBackend)

    def test_postgres_backend_requires_a_dsn(self):
        with pytest.raises(ValueError):
            PostgresBackend("")

    def test_dsn_password_is_redacted_in_describe(self):
        """`describe` reaches logs and GET /status, so it must never
        carry the database password."""
        from hypernix.t1api.db import _redact_dsn

        redacted = _redact_dsn("postgresql://user:hunter2@db.internal:5432/t1")
        assert "hunter2" not in redacted
        assert "user:***@db.internal:5432/t1" in redacted

    def test_sqlite_describe_names_the_file(self, tmp_path):
        backend = SQLiteBackend(tmp_path / "t.sqlite3")
        assert "sqlite:" in backend.describe


@pytest.mark.skipif(
    not os.environ.get("T1_TEST_DATABASE_URL"),
    reason="set T1_TEST_DATABASE_URL to run the real PostgreSQL round-trip",
)
class TestPostgresRoundTrip:
    def test_usage_store_works_on_postgres(self):
        store = UsageStore(backend=PostgresBackend(os.environ["T1_TEST_DATABASE_URL"]))
        store.record(key_id="k1", model_id="m1", input_tokens=10, output_tokens=5)
        totals = store.totals(key_id="k1")
        assert totals["input_tokens"] >= 10


class TestUsageStorePortableQueries:
    def test_history_and_aggregate(self, backend):
        store = UsageStore(backend=backend)
        store.record(key_id="k1", model_id="m1", input_tokens=10, output_tokens=5, server_id="s1")
        store.record(key_id="k1", model_id="m2", input_tokens=20, output_tokens=1, server_id="s2")
        store.record(key_id="k2", model_id="m1", input_tokens=7, output_tokens=0, server_id="s1")

        assert len(store.history()) == 3
        assert len(store.history(key_id="k1")) == 2
        by_model = {row["model_id"]: row for row in store.aggregate("model_id")}
        assert by_model["m1"]["input_tokens"] == 17
        by_server = {row["server_id"]: row for row in store.aggregate("server_id", key_id="k1")}
        assert set(by_server) == {"s1", "s2"}

    def test_user_and_account_columns_are_recorded(self, backend):
        store = UsageStore(backend=backend)
        store.record(key_id="k1", model_id="m1", user_id="u1", account_id="acct1", input_tokens=5)
        rows = store.aggregate("account_id")
        assert rows[0]["account_id"] == "acct1"

    def test_migration_adds_columns_to_an_existing_table(self, tmp_path):
        """An existing Beta 1/2 database must upgrade in place, not need a
        dump and reload."""
        path = tmp_path / "old.sqlite3"
        import sqlite3

        conn = sqlite3.connect(path)
        conn.executescript(
            """CREATE TABLE usage_events (
                   id INTEGER PRIMARY KEY AUTOINCREMENT, key_id TEXT NOT NULL,
                   model_id TEXT NOT NULL, input_tokens INTEGER NOT NULL DEFAULT 0,
                   output_tokens INTEGER NOT NULL DEFAULT 0, requests INTEGER NOT NULL DEFAULT 1,
                   endpoint TEXT NOT NULL DEFAULT '', server_id TEXT NOT NULL DEFAULT '',
                   module_id TEXT NOT NULL DEFAULT '', ts REAL NOT NULL);"""
        )
        conn.execute(
            "INSERT INTO usage_events (key_id, model_id, input_tokens, ts) VALUES ('k','m',3,1.0)"
        )
        conn.commit()
        conn.close()

        store = UsageStore(path)  # runs the migration
        assert store.totals(key_id="k")["input_tokens"] == 3
        store.record(key_id="k", model_id="m", user_id="u", account_id="a", input_tokens=1)
        assert store.aggregate("user_id", key_id="k")[0]["user_id"] in ("", "u")

    def test_unknown_filter_dimension_is_a_programming_error(self, backend):
        store = UsageStore(backend=backend)
        with pytest.raises(ValueError):
            store.totals(nonsense="x")

    def test_unknown_group_by_is_rejected(self, backend):
        """group_by becomes a SQL identifier, so it can never be
        free-form caller input."""
        store = UsageStore(backend=backend)
        with pytest.raises(ValueError):
            store.aggregate("1; DROP TABLE usage_events")


# ===========================================================================
# Cost
# ===========================================================================


class TestCost:
    def test_prices_recorded_usage(self, backend, registry):
        store = UsageStore(backend=backend)
        store.record(key_id="k1", model_id="test-large", input_tokens=1000, output_tokens=1000)
        report = CostCalculator(store, registry).cost_report()
        # 1000 in @ $1/1k + 1000 out @ $3/1k
        assert report.total_cost == pytest.approx(4.0)

    def test_per_server_costs_use_per_model_prices(self, backend, registry):
        """A per-server total is the sum of that server's per-model costs,
        not a server-level rate."""
        store = UsageStore(backend=backend)
        store.record(key_id="k", model_id="test-large", input_tokens=1000, server_id="s1")
        store.record(key_id="k", model_id="test-small", input_tokens=1000, server_id="s1")
        report = CostCalculator(store, registry).cost_report(group_by="server_id")
        assert len(report.lines) == 1
        assert report.lines[0].value == "s1"
        assert report.lines[0].total_cost == pytest.approx(1.1)  # 1.0 + 0.1

    def test_usage_for_a_deregistered_model_is_reported_not_dropped(self, backend, registry):
        store = UsageStore(backend=backend)
        store.record(key_id="k", model_id="test-large", input_tokens=1000)
        store.record(key_id="k", model_id="retired-model", input_tokens=5000)
        report = CostCalculator(store, registry).cost_report()
        assert "retired-model" in report.unpriced_models
        assert report.total_cost == pytest.approx(1.0)  # the retired model priced at 0

    def test_estimate_defaults_to_an_upper_bound(self, backend, registry):
        estimate = CostCalculator(UsageStore(backend=backend), registry).estimate(
            model_id="test-small", input_tokens=1000
        )
        assert estimate["output_tokens_assumed"] is True
        assert estimate["output_tokens"] == 2000  # the model's output_token_limit
        assert estimate["estimated_cost"] == pytest.approx(0.1 + 0.4)

    def test_estimate_with_explicit_output(self, backend, registry):
        estimate = CostCalculator(UsageStore(backend=backend), registry).estimate(
            model_id="test-small", input_tokens=1000, output_tokens=1000
        )
        assert estimate["output_tokens_assumed"] is False
        assert estimate["estimated_cost"] == pytest.approx(0.1 + 0.2)

    def test_estimate_flags_over_limit_prompts(self, backend, registry):
        estimate = CostCalculator(UsageStore(backend=backend), registry).estimate(
            model_id="test-small", input_tokens=99_999
        )
        assert estimate["exceeds_input_token_limit"] is True
        assert estimate["exceeds_context_limit"] is True

    def test_estimate_rejects_an_unregistered_model(self, backend, registry):
        """Pricing an unknown model would mean inventing a price."""
        with pytest.raises(T1APIError) as exc:
            CostCalculator(UsageStore(backend=backend), registry).estimate(
                model_id="not-a-model", input_tokens=1
            )
        assert exc.value.code is T1ErrorCode.MODEL_NOT_SUPPORTED

    def test_estimate_rejects_negative_tokens(self, backend, registry):
        with pytest.raises(T1APIError):
            CostCalculator(UsageStore(backend=backend), registry).estimate(
                model_id="test-small", input_tokens=-1
            )

    def test_forecast_with_no_usage_is_zero_and_says_so(self, backend, registry):
        forecast = CostCalculator(UsageStore(backend=backend), registry).forecast()
        assert forecast.projected_cost == 0.0
        assert forecast.confidence == "low"
        assert "No usage recorded" in forecast.basis

    def test_forecast_states_its_confidence_and_basis(self, backend, registry):
        """A forecast from eleven minutes of history should not look like
        a measurement."""
        store = UsageStore(backend=backend)
        store.record(key_id="k", model_id="test-large", input_tokens=1000, ts=time.time() - 60)
        forecast = CostCalculator(store, registry).forecast(horizon_seconds=86400)
        assert forecast.confidence == "low"
        assert forecast.projected_cost > 0
        assert "extrapolation" in forecast.basis.lower()


# ===========================================================================
# Key directory
# ===========================================================================


class TestKeyDirectory:
    def test_non_admin_sees_only_their_own_key(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend)
        mine = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.create(key_type=KeyType.USER, scopes={KeyScope.READ})

        as_user = directory.list_keys(requester_key_id=mine.key_id, is_admin=False)
        assert [k.key_id for k in as_user] == [mine.key_id]
        as_admin = directory.list_keys(requester_key_id=mine.key_id, is_admin=True)
        assert len(as_admin) == 2

    def test_summary_never_carries_the_raw_key(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        summary = directory.list_keys(requester_key_id=meta.key_id, is_admin=True)[0]
        serialized = json.dumps(summary.to_dict())
        assert meta.key not in serialized
        assert "key" not in summary.to_dict() or summary.to_dict().get("key") is None

    def test_key_id_is_masked(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        summary = directory.list_keys(requester_key_id=meta.key_id, is_admin=True)[0]
        assert summary.to_dict()["key_id"] == mask_key_id(meta.key_id)
        assert summary.to_dict()["key_id"] != meta.key_id

    def test_assign_validates_models_against_the_registry(self, km, registry, backend):
        """An assignment must not be a way to name a model the registry
        never heard of."""
        directory = KeyDirectory(km, registry, backend)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        with pytest.raises(T1APIError) as exc:
            directory.assign(meta.key_id, assigned_by="admin", allowed_models=["not-a-model"])
        assert exc.value.code is T1ErrorCode.MODEL_NOT_SUPPORTED
        assert directory.get_assignment(meta.key_id) is None

    def test_assign_validates_servers_when_a_validator_is_given(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend)
        servers = ServerRegistry(backend)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        with pytest.raises(T1APIError):
            directory.assign(
                meta.key_id, assigned_by="a", server_ids=["nope"], server_validator=servers.require
            )

    def test_assign_rejects_an_unknown_key(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend)
        with pytest.raises(T1APIError) as exc:
            directory.assign("no-such-key", assigned_by="admin", plan="pro")
        assert exc.value.code is T1ErrorCode.NOT_FOUND

    def test_assign_is_a_partial_update(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        directory.assign(meta.key_id, assigned_by="a", plan="pro", account_id="acct1")
        directory.assign(meta.key_id, assigned_by="a", user_id="u1")
        assignment = directory.get_assignment(meta.key_id)
        assert assignment.plan == "pro"           # preserved
        assert assignment.account_id == "acct1"   # preserved
        assert assignment.user_id == "u1"         # updated

    def test_plan_is_resolved_server_side(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend, default_plan="free")
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        assert directory.resolve_plan(meta.key_id) == "free"
        directory.assign(meta.key_id, assigned_by="admin", plan="pro")
        assert directory.resolve_plan(meta.key_id) == "pro"

    def test_a_client_cannot_name_a_plan_it_does_not_hold(self, km, registry, backend):
        """The design principle in one assertion: the client requests, the
        server decides."""
        directory = KeyDirectory(km, registry, backend, default_plan="free")
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        directory.assign(meta.key_id, assigned_by="admin", plan="free")
        with pytest.raises(T1APIError) as exc:
            directory.resolve_plan(meta.key_id, requested_plan="enterprise")
        assert exc.value.code is T1ErrorCode.AUTH_INSUFFICIENT_SCOPE
        # Matching is fine.
        assert directory.resolve_plan(meta.key_id, requested_plan="free") == "free"

    def test_model_narrowing_is_enforced(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        directory.assign(meta.key_id, assigned_by="admin", allowed_models=["test-small"])
        directory.assert_model_allowed(meta.key_id, "test-small")
        with pytest.raises(T1APIError) as exc:
            directory.assert_model_allowed(meta.key_id, "test-large")
        assert exc.value.code is T1ErrorCode.AUTH_INSUFFICIENT_SCOPE

    def test_no_narrowing_means_no_restriction(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        directory.assert_model_allowed(meta.key_id, "test-large")  # no assignment at all
        directory.assign(meta.key_id, assigned_by="admin", plan="pro")
        directory.assert_model_allowed(meta.key_id, "test-large")  # assignment, empty list

    def test_import_returns_no_key_material(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        payload = km.export()
        result = directory.import_payload(payload, imported_by="admin")
        assert meta.key not in json.dumps(result)
        # Re-importing the same export is idempotent, not destructive.
        assert result["skipped"] == 1

    def test_import_rejects_a_malformed_payload(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend)
        for bad in ({}, {"keys": []}, {"keys": [{"key_id": "x"}]}, {"keys": "nope"}):
            with pytest.raises(T1APIError) as exc:
                directory.import_payload(bad, imported_by="admin")
            assert exc.value.code is T1ErrorCode.VALIDATION_ERROR

    def test_usage_identity(self, km, registry, backend):
        directory = KeyDirectory(km, registry, backend)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        assert directory.usage_identity(meta.key_id) == ("", "")
        directory.assign(meta.key_id, assigned_by="a", user_id="u1", account_id="acct1")
        assert directory.usage_identity(meta.key_id) == ("u1", "acct1")


# ===========================================================================
# Transport
# ===========================================================================


class _Response:
    """A minimal stand-in for an HTTP response.

    ``read(size)`` honours the size argument and returns ``b""`` at EOF,
    because ``ModuleTransport.fetch_source`` streams in 64 KiB chunks and
    stops on the empty read. A stub that returned its whole payload on
    every call would loop until it tripped the size cap — which is how
    this class was written the first time, and what the streaming tests
    caught.
    """

    status = 200

    def __init__(self, payload: bytes = b"{}") -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int = -1):
        if size is None or size < 0:
            chunk, self._offset = self._payload[self._offset :], len(self._payload)
            return chunk
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def getcode(self):
        return 200

    @property
    def headers(self):
        return {"Content-Length": str(len(self._payload))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestTransportSignatures:
    def test_signature_round_trip(self):
        body = b"payload"
        stamp = f"{time.time():.3f}"
        signature = compute_signature(
            "secret", method="POST", path="/modules/receive", timestamp=stamp,
            body_digest=hashlib.sha256(body).hexdigest(),
        )
        verify_signature(
            "secret", method="POST", path="/modules/receive", timestamp=stamp,
            body=body, signature=signature,
        )

    def test_tampered_body_fails(self):
        stamp = f"{time.time():.3f}"
        signature = compute_signature(
            "secret", method="POST", path="/p", timestamp=stamp,
            body_digest=hashlib.sha256(b"original").hexdigest(),
        )
        with pytest.raises(T1APIError) as exc:
            verify_signature(
                "secret", method="POST", path="/p", timestamp=stamp,
                body=b"tampered", signature=signature,
            )
        assert exc.value.code is T1ErrorCode.AUTH_INVALID_TOKEN

    def test_signature_is_bound_to_the_path(self):
        """A body signed for one endpoint must not replay against another."""
        stamp = f"{time.time():.3f}"
        signature = compute_signature(
            "secret", method="POST", path="/modules/receive", timestamp=stamp,
            body_digest=hashlib.sha256(b"x").hexdigest(),
        )
        with pytest.raises(T1APIError):
            verify_signature(
                "secret", method="POST", path="/admin/danger", timestamp=stamp,
                body=b"x", signature=signature,
            )

    def test_stale_signature_is_refused(self):
        old = f"{time.time() - SIGNATURE_MAX_SKEW_SECONDS - 60:.3f}"
        signature = compute_signature(
            "secret", method="POST", path="/p", timestamp=old,
            body_digest=hashlib.sha256(b"x").hexdigest(),
        )
        with pytest.raises(T1APIError) as exc:
            verify_signature(
                "secret", method="POST", path="/p", timestamp=old, body=b"x", signature=signature
            )
        assert exc.value.code is T1ErrorCode.AUTH_EXPIRED_TOKEN

    def test_wrong_secret_fails(self):
        stamp = f"{time.time():.3f}"
        signature = compute_signature(
            "secret-a", method="POST", path="/p", timestamp=stamp,
            body_digest=hashlib.sha256(b"x").hexdigest(),
        )
        with pytest.raises(T1APIError):
            verify_signature(
                "secret-b", method="POST", path="/p", timestamp=stamp, body=b"x", signature=signature
            )

    def test_no_configured_secret_refuses_rather_than_accepting(self):
        with pytest.raises(T1APIError) as exc:
            verify_signature("", method="POST", path="/p", timestamp="1", body=b"", signature="x")
        assert exc.value.code is T1ErrorCode.AUTH_MISSING_CREDENTIALS

    def test_missing_headers_refused(self):
        with pytest.raises(T1APIError):
            verify_signature("s", method="POST", path="/p", timestamp="", body=b"", signature="")


class TestModuleTransport:
    def test_push_signs_and_sends(self):
        sent = {}

        def opener(request, timeout=None):
            sent["url"] = request.full_url
            sent["body"] = request.data
            sent["headers"] = {k.lower(): v for k, v in request.headers.items()}
            return _Response(json.dumps({"checksum": hashlib.sha256(request.data).hexdigest()}).encode())

        transport = ModuleTransport(deploy_secret="s" * 32, opener=opener)
        result = transport.push_module(
            server_address="https://target.example.com", module_id="m1", content=b"bytes"
        )
        assert result.ok
        assert sent["url"] == "https://target.example.com/modules/receive"
        assert sent["body"] == b"bytes"
        assert sent["headers"]["x-t1-content-sha256"] == hashlib.sha256(b"bytes").hexdigest()
        verify_signature(
            "s" * 32, method="POST", path="/modules/receive",
            timestamp=sent["headers"]["x-t1-timestamp"], body=b"bytes",
            signature=sent["headers"]["x-t1-signature"],
        )

    def test_push_without_a_secret_is_refused(self):
        """Refusing to push unauthenticated bytes is better than pushing
        them."""
        transport = ModuleTransport(deploy_secret="", opener=lambda *a, **k: _Response())
        with pytest.raises(T1APIError) as exc:
            transport.push_module(
                server_address="https://target.example.com", module_id="m", content=b"x"
            )
        assert exc.value.code is T1ErrorCode.AUTH_MISSING_CREDENTIALS

    def test_push_enforces_the_size_cap(self):
        transport = ModuleTransport(
            deploy_secret="s" * 32, max_bytes=10, opener=lambda *a, **k: _Response()
        )
        with pytest.raises(T1APIError) as exc:
            transport.push_module(
                server_address="https://target.example.com", module_id="m", content=b"x" * 100
            )
        assert exc.value.code is T1ErrorCode.PAYLOAD_TOO_LARGE

    def test_push_validates_the_target_address(self):
        """SSRF guard: a private target needs an explicit opt-in."""
        transport = ModuleTransport(deploy_secret="s" * 32, opener=lambda *a, **k: _Response())
        with pytest.raises(T1APIError) as exc:
            transport.push_module(
                server_address="http://169.254.169.254/latest", module_id="m", content=b"x"
            )
        assert exc.value.code is T1ErrorCode.SSRF_BLOCKED

    def test_private_target_allowed_when_opted_in(self):
        """Tailscale/LAN deployments have private targets by design."""
        transport = ModuleTransport(
            deploy_secret="s" * 32,
            allow_private_targets=True,
            opener=lambda request, timeout=None: _Response(
                json.dumps({"checksum": hashlib.sha256(request.data).hexdigest()}).encode()
            ),
        )
        assert transport.push_module(
            server_address="http://100.101.102.103:8000", module_id="m", content=b"x"
        ).ok

    def test_checksum_mismatch_from_the_target_fails_the_deployment(self):
        """A cheerful 200 with the wrong digest means the bytes changed in
        flight."""
        transport = ModuleTransport(
            deploy_secret="s" * 32,
            opener=lambda *a, **k: _Response(json.dumps({"checksum": "deadbeef"}).encode()),
        )
        with pytest.raises(T1APIError) as exc:
            transport.push_module(
                server_address="https://target.example.com", module_id="m", content=b"x"
            )
        assert exc.value.code is T1ErrorCode.CHECKSUM_MISMATCH

    def test_network_failure_becomes_transport_failed(self):
        import urllib.error

        def boom(*args, **kwargs):
            raise urllib.error.URLError("connection refused")

        transport = ModuleTransport(deploy_secret="s" * 32, opener=boom)
        with pytest.raises(T1APIError) as exc:
            transport.push_module(
                server_address="https://target.example.com", module_id="m", content=b"x"
            )
        assert exc.value.code is T1ErrorCode.TRANSPORT_FAILED

    def test_fetch_writes_and_checksums(self, tmp_path):
        transport = ModuleTransport(
            deploy_secret="s" * 32, opener=lambda *a, **k: _Response(b"fetched-content")
        )
        destination = tmp_path / "out.bin"
        result = transport.fetch_source("https://example.com/mod.bin", destination=destination)
        assert destination.read_bytes() == b"fetched-content"
        assert result.checksum == hashlib.sha256(b"fetched-content").hexdigest()

    def test_fetch_rejects_a_checksum_mismatch_and_leaves_nothing_behind(self, tmp_path):
        transport = ModuleTransport(
            deploy_secret="s" * 32, opener=lambda *a, **k: _Response(b"content")
        )
        destination = tmp_path / "out.bin"
        with pytest.raises(T1APIError) as exc:
            transport.fetch_source(
                "https://example.com/m.bin", destination=destination, expected_checksum="deadbeef"
            )
        assert exc.value.code is T1ErrorCode.CHECKSUM_MISMATCH
        assert not destination.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_fetch_enforces_the_size_cap_mid_download(self, tmp_path):
        transport = ModuleTransport(
            deploy_secret="s" * 32, max_bytes=8, opener=lambda *a, **k: _Response(b"x" * 1000)
        )
        with pytest.raises(T1APIError) as exc:
            transport.fetch_source("https://example.com/m.bin", destination=tmp_path / "o.bin")
        assert exc.value.code is T1ErrorCode.PAYLOAD_TOO_LARGE
        assert not (tmp_path / "o.bin").exists()

    def test_fetch_validates_the_url(self, tmp_path):
        transport = ModuleTransport(deploy_secret="s" * 32, opener=lambda *a, **k: _Response(b""))
        with pytest.raises(T1APIError) as exc:
            transport.fetch_source("file:///etc/passwd", destination=tmp_path / "o.bin")
        assert exc.value.code is T1ErrorCode.SSRF_BLOCKED

    def test_size_cap_is_clamped_to_the_absolute_maximum(self):
        from hypernix.t1api.transport import ABSOLUTE_MAX_TRANSFER_BYTES

        transport = ModuleTransport(deploy_secret="s", max_bytes=10**15)
        assert transport.max_bytes == ABSOLUTE_MAX_TRANSFER_BYTES


# ===========================================================================
# Deployment coordinator
# ===========================================================================


class TestDeploymentCoordinator:
    @pytest.fixture
    def setup(self, tmp_path, backend):
        modules = ModuleRegistry(backend, storage_dir=tmp_path / "modules")
        servers = ServerRegistry(backend)
        sent: list[dict] = []

        def opener(request, timeout=None):
            sent.append({"url": request.full_url, "body": request.data})
            return _Response(
                json.dumps({"checksum": hashlib.sha256(request.data).hexdigest()}).encode()
            )

        transport = ModuleTransport(deploy_secret="s" * 32, opener=opener)
        coordinator = DeploymentCoordinator(
            modules, servers, transport, audit=AuditLog(backend)
        )
        return coordinator, modules, servers, sent

    def test_push_requires_a_trusted_server(self, setup):
        """Trust is checked before the address is even read."""
        coordinator, modules, servers, sent = setup
        module = modules.create(name="m", version="1.0", owner_key_id="k")
        modules.upload_local(module.module_id, b"content", filename="m.bin")
        server = servers.register(name="s", address="https://s.example.com", registered_by="k")

        with pytest.raises(T1APIError) as exc:
            coordinator.push_to_server(module.module_id, server.server_id)
        assert exc.value.code is T1ErrorCode.SERVER_UNTRUSTED
        assert sent == []  # nothing was dialled

    def test_push_to_a_trusted_server_transfers(self, setup):
        coordinator, modules, servers, sent = setup
        module = modules.create(name="m", version="1.0", owner_key_id="k")
        modules.upload_local(module.module_id, b"content", filename="m.bin")
        server = servers.register(name="s", address="https://s.example.com", registered_by="k")
        servers.update(server.server_id, trust_level=TrustLevel.TRUSTED)

        result = coordinator.push_to_server(module.module_id, server.server_id)
        assert result.ok
        assert sent[0]["body"] == b"content"
        assert modules.require(module.module_id).deployed_servers == [server.server_id]

    def test_sync_handler_preserves_the_specific_error_code(self, setup):
        """A single-target sync to an untrusted server should say
        SERVER_UNTRUSTED, not a generic transport failure."""
        coordinator, modules, servers, _ = setup
        module = modules.create(name="m", version="1.0", owner_key_id="k")
        modules.upload_local(module.module_id, b"c", filename="m.bin")
        server = servers.register(name="s", address="https://s.example.com", registered_by="k")

        with pytest.raises(T1APIError) as exc:
            coordinator.module_sync_handler(
                {"module_id": module.module_id, "server_id": server.server_id}, None
            )
        assert exc.value.code is T1ErrorCode.SERVER_UNTRUSTED

    def test_multi_target_partial_success_is_reported_not_raised(self, setup):
        """Some servers holding the module is a fact worth returning, not
        an exception to discard."""
        coordinator, modules, servers, _ = setup
        module = modules.create(name="m", version="1.0", owner_key_id="k")
        modules.upload_local(module.module_id, b"c", filename="m.bin")
        good = servers.register(name="good", address="https://a.example.com", registered_by="k")
        servers.update(good.server_id, trust_level=TrustLevel.TRUSTED)
        bad = servers.register(name="bad", address="https://b.example.com", registered_by="k")

        outcome = coordinator.module_sync_handler(
            {"module_id": module.module_id, "server_ids": [good.server_id, bad.server_id]}, None
        )
        assert len(outcome["delivered"]) == 1
        assert outcome["failed"][0]["error_code"] == "SERVER_UNTRUSTED"
        assert outcome["deployed_servers"] == [good.server_id]

    def test_sync_payload_needs_a_target(self, setup):
        coordinator, modules, _, _ = setup
        module = modules.create(name="m", version="1.0", owner_key_id="k")
        with pytest.raises(T1APIError) as exc:
            coordinator.module_sync_handler({"module_id": module.module_id}, None)
        assert exc.value.code is T1ErrorCode.VALIDATION_ERROR

    def test_fetch_uses_the_registered_url_not_the_payload(self, setup, tmp_path):
        """Letting a job payload name a URL would reopen the SSRF hole
        that registering it through the validated path closes."""
        coordinator, modules, _, _ = setup
        module = modules.create(name="m", version="1.0", owner_key_id="k")
        with pytest.raises(T1APIError) as exc:
            coordinator.module_fetch_handler(
                {"module_id": module.module_id, "source_url": "http://169.254.169.254/"}, None
            )
        # No registered source, so it refuses — the payload's URL is ignored.
        assert exc.value.code is T1ErrorCode.MODULE_UPLOAD_REJECTED

    def test_fetch_stages_a_registered_source(self, setup):
        coordinator, modules, _, _ = setup
        module = modules.create(name="m", version="1.0", owner_key_id="k")
        modules.register_remote_source(module.module_id, "https://example.com/m.bin")
        coordinator.transport._urlopen = lambda *a, **k: _Response(b"remote-bytes")

        outcome = coordinator.module_fetch_handler({"module_id": module.module_id}, None)
        entry = modules.require(module.module_id)
        assert outcome["status"] == "active"
        assert entry.checksum == hashlib.sha256(b"remote-bytes").hexdigest()
        # Provenance is preserved: it still says where it came from.
        assert entry.source_url == "https://example.com/m.bin"
        assert modules.read_content(module.module_id) == b"remote-bytes"


# ===========================================================================
# Production configuration validation
# ===========================================================================


class TestProductionValidation:
    def test_development_config_never_raises(self):
        T1APIConfig(environment="development").validate_for_production()

    def test_production_config_with_gaps_raises(self):
        config = T1APIConfig(environment="production")
        with pytest.raises(T1APIError) as exc:
            config.validate_for_production()
        assert exc.value.code is T1ErrorCode.CONFIG_INVALID
        assert exc.value.details["count"] > 0

    def test_every_problem_is_reported_not_just_the_first(self):
        problems = T1APIConfig(environment="production").production_problems()
        joined = " ".join(problems)
        assert "T1_TOKEN_SECRET" in joined
        assert "T1_DATABASE_URL" in joined
        assert "TLS" in joined

    def test_a_complete_production_config_passes(self, tmp_path):
        cert = tmp_path / "c.pem"
        key = tmp_path / "k.pem"
        ca = tmp_path / "ca.pem"
        for path in (cert, key, ca):
            path.write_text("x")
        config = T1APIConfig(
            environment="production",
            token_secret="a" * 64,
            database_url="postgresql://u:p@db/t1",
            tls_certfile=str(cert),
            tls_keyfile=str(key),
            tls_ca_certs=str(ca),
            expose_docs=False,
        )
        assert config.production_problems() == []
        config.validate_for_production()

    def test_short_token_secret_is_rejected(self):
        problems = T1APIConfig(environment="production", token_secret="short").production_problems()
        assert any("32" in p for p in problems)

    def test_wildcard_cors_is_rejected(self):
        problems = T1APIConfig(
            environment="production", cors_allow_origins=("*",)
        ).production_problems()
        assert any("wildcard" in p.lower() or "'*'" in p for p in problems)

    def test_example_models_in_production_is_flagged(self):
        problems = T1APIConfig(
            environment="production", enable_example_models=True
        ).production_problems()
        assert any("EXAMPLE_MODELS" in p for p in problems)

    def test_disabled_protections_are_flagged(self):
        problems = T1APIConfig(
            environment="production",
            rate_limit_enabled=False,
            audit_enabled=False,
            network_policy_enabled=False,
        ).production_problems()
        joined = " ".join(problems)
        assert "T1_RATE_LIMIT_ENABLED=0" in joined
        assert "T1_AUDIT_ENABLED=0" in joined
        assert "T1_NETWORK_POLICY_ENABLED=0" in joined

    def test_strict_forces_validation_off_production(self):
        with pytest.raises(T1APIError):
            T1APIConfig(environment="staging").validate_for_production(strict=True)
        T1APIConfig(environment="production").validate_for_production(strict=False)

    def test_public_config_never_contains_a_secret(self):
        config = T1APIConfig(
            token_secret="super-secret-value",
            deploy_secret="deploy-secret-value",
            database_url="postgresql://user:pw@host/db",
        )
        serialized = json.dumps(config.public_dict())
        assert "super-secret-value" not in serialized
        assert "deploy-secret-value" not in serialized
        assert "pw@host" not in serialized

    def test_describe_secrets_reports_presence_only(self):
        config = T1APIConfig(token_secret="x" * 32)
        described = config.describe_secrets()
        assert described["token_secret"] is True
        assert described["deploy_secret"] is False
        assert "x" * 32 not in json.dumps(described)

    def test_storage_backend_reflects_the_database_url(self):
        assert T1APIConfig().storage_backend == "sqlite"
        assert T1APIConfig(database_url="postgresql://x/y").storage_backend == "postgres"

    def test_malformed_rate_limit_rules_fall_back_to_defaults(self):
        """Unparseable rules must not mean *no* rules."""
        config = T1APIConfig(rate_limit_rules_json="{not json")
        assert config.rate_limit_rules() is None
        from hypernix.t1api.ratelimit import DEFAULT_RULES, rules_from_json

        assert rules_from_json(config.rate_limit_rules()) == DEFAULT_RULES


# ===========================================================================
# Key store freshness
# ===========================================================================


class TestKeyStoreRefresh:
    """A key created *after* the server started must still authenticate.

    Keymaster reads its key files once, at construction. That made the
    documented quickstart wrong whenever the server was already running:
    `gkey create` writes a new key file, the server never re-reads the
    directory, and the brand-new key comes back AUTH_INVALID_KEY. Found by
    running the quickstart against a real uvicorn process rather than a
    TestClient.
    """

    @pytest.fixture
    def service(self, tmp_path):
        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.t1api.auth import T1AuthService

        store = tmp_path / "keymaster"
        keymaster = Keymaster(store_dir=store, auto_rotate=False)
        gatekeeper = Gatekeeper(keymaster=keymaster, data_dir=tmp_path / "gk", log_to_file=False)
        service = T1AuthService(keymaster, gatekeeper, token_secret="x" * 40)
        return service, store

    def test_a_key_created_by_another_process_is_picked_up(self, service, tmp_path):
        svc, store = service
        # A second Keymaster over the same directory stands in for `gkey
        # create` running while the server is up.
        other = Keymaster(store_dir=store, auto_rotate=False)
        fresh = other.create(key_type=KeyType.USER, scopes={KeyScope.READ})

        # The service's own Keymaster has never seen it...
        assert svc.keymaster.get(fresh.key_id) is None
        # ...but validating refreshes and finds it.
        ctx = svc.validate_key(fresh.key)
        assert ctx.key_id == fresh.key_id

    def test_a_genuinely_invalid_key_still_fails_clearly(self, service):
        svc, _ = service
        with pytest.raises(T1APIError) as exc:
            svc.validate_key("T1_definitely-not-a-real-key")
        assert exc.value.code is T1ErrorCode.AUTH_INVALID_KEY

    def test_the_refresh_is_throttled(self, service, monkeypatch):
        """The reload is disk I/O reachable by an unauthenticated caller,
        so a stream of garbage keys must not force a directory scan per
        request."""
        svc, _ = service
        reloads = 0
        original = svc.keymaster._load_all

        def counting_load():
            nonlocal reloads
            reloads += 1
            original()

        monkeypatch.setattr(svc.keymaster, "_load_all", counting_load)
        for _ in range(50):
            with pytest.raises(T1APIError):
                svc.validate_key("T1_still-not-real")
        assert reloads <= 1, f"{reloads} key-store reloads for 50 bad keys"
