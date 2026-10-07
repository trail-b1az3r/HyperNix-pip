"""``/code`` — sandboxes that run code, and their permissions grammar.

The grammar is ``/web/v1/config``'s, so the tests of it here are about
what is new: numbered settings with names, ``;`` and a clause written
straight after ``perms``, ``s1=?`` as a read, and ``s2=on`` refused
rather than taken as a change. The boundary tests run real processes:
a sandbox's promises are about what a child process cannot do, and a
mock of the child would test nothing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from hypernix.t1api import codebox as cb

POSIX = pytest.mark.skipif(sys.platform == "win32", reason="runs bash and POSIX limits")
NETNS = pytest.mark.skipif(cb.network_isolation() is None,
                           reason="no user namespaces on this machine")


# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------


class TestThePermsGrammar:
    def test_the_web_form(self):
        perms, changed, asked = cb.apply_perms(cb.CodePerms(), "s2?:=on|s3?:=120|s1?=k")
        assert perms.network is True and perms.timeout == 120
        assert changed == ["s2", "s3"] and asked == ["s1"]

    def test_a_clause_straight_after_perms_and_semicolons(self):
        """The form as it was asked for: `perms|s1=?;s2=...`."""
        perms, changed, asked = cb.apply_perms(cb.CodePerms(), "|s1=?;s2?:=on")
        assert asked == ["s1"] and changed == ["s2"] and perms.network

    def test_s1_equals_question_mark_reads(self):
        perms, changed, asked = cb.apply_perms(cb.CodePerms(), "s1=?;s6=?")
        assert perms == cb.CodePerms() and changed == [] and asked == ["s1", "s6"]

    def test_an_empty_equals_reads_too(self):
        """What `perms|s2=?` arrives as: HTTP drops a `?` that ends a URL."""
        _, changed, asked = cb.apply_perms(cb.CodePerms(), "|s1=?;s2=")
        assert changed == [] and asked == ["s1", "s2"]

    def test_an_equals_with_no_marker_is_refused_not_taken_as_a_change(self):
        """`s2=on` reads like "network on". Treated as a change or as
        nothing, somebody would be wrong about their sandbox's network."""
        with pytest.raises(cb.GrammarError, match=r"s2\?:=on"):
            cb.apply_perms(cb.CodePerms(), "s2=on")

    def test_the_web_grammars_own_trap_is_refused_too(self):
        with pytest.raises(cb.GrammarError, match="without the colon"):
            cb.apply_perms(cb.CodePerms(), "s2?=on")

    def test_names_work_as_well_as_numbers(self):
        perms, changed, _ = cb.apply_perms(cb.CodePerms(), "network?:=on|timeout?:=9|ttl?:=5")
        assert (perms.network, perms.timeout, perms.ttl_minutes) == (True, 9, 5)
        assert changed == ["s2", "s3", "s7"]

    def test_setting_a_value_it_already_has_is_not_a_change(self):
        _, changed, _ = cb.apply_perms(cb.CodePerms(), "s2?:=off")
        assert changed == []

    @pytest.mark.parametrize("raw,message", [
        ("s9?:=1", "no setting"),
        ("s3?:=0", "outside 1..600"),
        ("s3?:=soon", "whole number"),
        ("s2?:=maybe", "not on or off"),
        ("s4?:=cobol", "unknown language"),
    ])
    def test_a_bad_value_says_which_setting_and_why(self, raw, message):
        with pytest.raises(cb.GrammarError, match=message):
            cb.apply_perms(cb.CodePerms(), raw)

    def test_all_or_nothing(self):
        before = cb.CodePerms()
        with pytest.raises(cb.GrammarError):
            cb.apply_perms(before, "s2?:=on|s3?:=never")
        assert before.network is False

    def test_languages_take_a_list_or_all(self):
        perms, _, _ = cb.apply_perms(cb.CodePerms(), "s4?:=python,node")
        assert perms.languages == ("python", "node")
        perms, _, _ = cb.apply_perms(cb.CodePerms(), "s4?:=all")
        assert perms.languages == tuple(cb.LANGUAGES)

    def test_the_defaults_are_the_cautious_ones(self):
        perms = cb.CodePerms()
        assert perms.network is False
        assert perms.languages == ("python", "bash")
        assert perms.to_dict()["s2"] == {"name": "network", "value": "off",
                                         "means": "off: no network inside the sandbox"}


# ---------------------------------------------------------------------------
# Sandboxes, directly
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path) -> cb.SandboxStore:
    return cb.SandboxStore(tmp_path / "code")


# Network on for runs that are not about the network, so they pass on a
# machine without user namespaces.
OPEN = cb.CodePerms(network=True, timeout=10)


class TestTheBoundary:
    @pytest.mark.parametrize("path", ["../../etc/passwd", "/etc/passwd", ".hnx/tmp/x", ""])
    def test_paths_stay_inside(self, store, path):
        box = store.create("me", OPEN)
        with pytest.raises(cb.SandboxError):
            store.read(box, path)

    @POSIX
    def test_a_planted_symlink_does_not_lead_out(self, store):
        box = store.create("me", OPEN)
        (box.root / "out").symlink_to("/etc")
        with pytest.raises(cb.SandboxError) as caught:
            store.read(box, "out/passwd")
        assert caught.value.code == "outside_workspace"

    def test_somebody_elses_sandbox_is_not_found(self, store):
        box = store.create("alice", OPEN)
        with pytest.raises(cb.SandboxError) as caught:
            store.get(box.id, "bob")
        assert caught.value.status == 404
        assert store.get(box.id, "bob", is_admin=True) is box

    def test_a_caller_may_hold_only_so_many(self, store):
        for _ in range(cb.MAX_PER_OWNER):
            store.create("me", OPEN)
        with pytest.raises(cb.SandboxError) as caught:
            store.create("me", OPEN)
        assert caught.value.status == 409
        store.create("someone-else", OPEN)

    def test_writes_are_held_to_the_disk_budget(self, store):
        box = store.create("me", cb.CodePerms(disk_mb=1))
        with pytest.raises(cb.SandboxError) as caught:
            store.write(box, "big.txt", "x" * (2 * 2**20), cb.CodePerms(disk_mb=1))
        assert caught.value.status == 413

    def test_an_idle_sandbox_expires(self, store):
        box = store.create("me", OPEN)
        box.last_used = time.time() - 3600
        assert store.expire(cb.CodePerms(ttl_minutes=30)) == [box.id]
        assert not box.root.exists()

    def test_only_sandbox_directories_are_ever_removed(self, tmp_path):
        """A T1_CODE_SANDBOX_DIR pointed somewhere unwise loses nothing
        but sandboxes when the server restarts."""
        base = tmp_path / "shared"
        (base / "notes").mkdir(parents=True)
        (base / "sbx_0123456789abcdef").mkdir()
        cb.SandboxStore(base)
        assert (base / "notes").is_dir()
        assert not (base / "sbx_0123456789abcdef").exists()


@POSIX
class TestRunning:
    def test_code_runs_and_reads_stdin(self, store):
        box = store.create("me", OPEN)
        result = store.run(box, OPEN, code="import sys; print(sys.stdin.read()[::-1])",
                           stdin="olleh")
        assert (result.exit_code, result.stdout.strip()) == (0, "hello")

    def test_a_file_runs_by_its_suffix(self, store):
        box = store.create("me", OPEN, files={"go.sh": "echo from bash $1"})
        result = store.run(box, OPEN, path="go.sh", args=["now"])
        assert result.language == "bash" and result.stdout.strip() == "from bash now"

    def test_the_environment_carries_no_secrets(self, store, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        box = store.create("me", OPEN)
        result = store.run(box, OPEN, code="import os; print(sorted(os.environ))")
        assert "HF_TOKEN" not in result.stdout
        assert "HOME" in result.stdout

    def test_a_run_past_its_timeout_is_killed(self, store):
        perms = cb.CodePerms(network=True, timeout=1)
        box = store.create("me", perms)
        started = time.monotonic()
        result = store.run(box, perms, code="import time; time.sleep(60)")
        assert result.timed_out and result.exit_code is None
        assert time.monotonic() - started < 10

    def test_memory_is_limited(self, store):
        perms = cb.CodePerms(network=True, memory_mb=64, timeout=10)
        box = store.create("me", perms)
        result = store.run(box, perms, code="b = bytearray(512 * 2**20)")
        if "memory_bytes" in result.limits.get("not_enforced", ()):
            # macOS takes no data-segment limit. The run still happens,
            # and the result says the limit did not hold rather than
            # reporting one that never applied.
            assert sys.platform == "darwin", result.limits
            assert result.limits["memory_bytes"] is None
        else:
            assert result.exit_code != 0 and "MemoryError" in result.stderr

    def test_a_file_cannot_outgrow_the_disk_budget(self, store):
        perms = cb.CodePerms(network=True, disk_mb=1, timeout=10)
        box = store.create("me", perms)
        result = store.run(box, perms, code="open('big', 'wb').write(b'x' * (4 * 2**20))")
        assert result.exit_code != 0

    def test_output_is_capped(self, store):
        box = store.create("me", OPEN)
        result = store.run(box, OPEN, code=f"print('x' * {cb.MAX_OUTPUT * 2})")
        assert result.truncated and len(result.stdout) <= cb.MAX_OUTPUT

    def test_execute_off_runs_nothing(self, store):
        perms = cb.CodePerms(execute=False)
        box = store.create("me", perms)
        with pytest.raises(cb.SandboxError) as caught:
            store.run(box, perms, code="print(1)")
        assert caught.value.code == "execute_off"

    def test_a_language_not_allowed_is_refused(self, store):
        box = store.create("me", OPEN)
        with pytest.raises(cb.SandboxError) as caught:
            store.run(box, OPEN, language="node", code="1")
        assert caught.value.code == "language_off"

    @NETNS
    def test_network_off_means_no_network(self, store):
        perms = cb.CodePerms(timeout=10)
        box = store.create("me", perms)
        result = store.run(box, perms, code=(
            "import socket\n"
            "s = socket.socket(); s.settimeout(3)\n"
            "try:\n    s.connect(('1.1.1.1', 80)); print('connected')\n"
            "except OSError as e:\n    print('refused', e.errno)\n"))
        assert result.stdout.startswith("refused")

    def test_network_off_where_it_cannot_be_enforced_is_refused(self, store, monkeypatch):
        """Not run with the network on: that is the one wrong answer."""
        monkeypatch.setattr(cb, "network_isolation", lambda: None)
        perms = cb.CodePerms()
        box = store.create("me", perms)
        with pytest.raises(cb.SandboxError) as caught:
            store.run(box, perms, code="print(1)")
        assert caught.value.code == "network_unenforceable"


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402


@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


def _client(km, tmp_path, *, enabled: bool) -> TestClient:
    gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)
    config = T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough",
        db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"),
        hyperlink_files_dir=str(tmp_path / "files"),
        default_plan="free",
        code_sandbox=enabled,
        code_sandbox_dir=str(tmp_path / "code"),
    )
    app = create_app(config=config, keymaster=km, gatekeeper=gk)
    # The cautious defaults, but with the network on: these tests are
    # about HTTP, and must pass on a machine without user namespaces.
    app.state.t1_code_perms = cb.CodePerms(network=True, timeout=10)
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture
def client(km, tmp_path) -> TestClient:
    return _client(km, tmp_path, enabled=True)


def _key(km, *scopes) -> dict[str, str]:
    kind = KeyType.ADMIN if KeyScope.ADMIN in scopes else KeyType.USER
    return {"Authorization": f"Bearer {km.create(key_type=kind, scopes=set(scopes)).key}"}


@pytest.fixture
def admin(km):
    return _key(km, KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE)


@pytest.fixture
def writer(km):
    return _key(km, KeyScope.READ, KeyScope.WRITE)


@pytest.fixture
def reader(km):
    return _key(km, KeyScope.READ)


class TestOffByDefault:
    def test_the_reads_say_how_to_turn_it_on(self, km, tmp_path, admin):
        off = _client(km, tmp_path, enabled=False)
        body = off.get("/code", headers=admin).json()
        assert body["enabled"] is False and "T1_CODE_SANDBOX=1" in body["how_to_enable"]

    def test_nothing_runs(self, km, tmp_path, admin):
        off = _client(km, tmp_path, enabled=False)
        got = off.post("/code/create", json={"code": "print(1)"}, headers=admin)
        assert got.status_code == 403 and "T1_CODE_SANDBOX=1" in got.json()["error"]["message"]
        assert off.post("/code/create/sandbox", json={}, headers=admin).status_code == 403

    def test_the_config_default_is_off(self, monkeypatch):
        monkeypatch.delenv("T1_CODE_SANDBOX", raising=False)
        assert T1APIConfig().code_sandbox is False


class TestTheTree:
    @pytest.mark.parametrize("path", ["/code", "/code/", "/code/create",
                                      "/code/create/sandbox", "/code/create/sandbox/perms"])
    def test_every_level_answers_a_read(self, client, reader, path):
        assert client.get(path, headers=reader).status_code == 200

    def test_nothing_without_a_credential(self, client):
        assert client.get("/code").status_code == 401


@POSIX
class TestRunningOverHTTP:
    def test_run_once(self, client, writer):
        got = client.post("/code/create", json={"code": "print(6 * 7)"}, headers=writer)
        assert got.status_code == 200 and got.json()["stdout"] == "42\n"

    def test_run_once_leaves_no_sandbox_behind(self, client, writer):
        client.post("/code/create", json={"code": "print(1)"}, headers=writer)
        assert client.get("/code/create/sandbox", headers=writer).json()["sandboxes"] == []

    def test_a_read_only_key_cannot_run(self, client, reader):
        got = client.post("/code/create", json={"code": "print(1)"}, headers=reader)
        assert got.status_code == 403

    def test_a_sandbox_from_creation_to_deletion(self, client, writer):
        made = client.post("/code/create/sandbox",
                           json={"name": "demo", "files": {"a.py": "print(open('b.txt').read())"}},
                           headers=writer)
        assert made.status_code == 201
        box = made.json()["id"]
        put = client.put(f"/code/sandbox/{box}/files/b.txt", json={"content": "from a file"},
                         headers=writer)
        assert put.status_code == 200
        ran = client.post(f"/code/sandbox/{box}/run", json={"path": "a.py"}, headers=writer)
        assert ran.json()["stdout"] == "from a file\n"
        client.post(f"/code/sandbox/{box}/run",
                    json={"code": "open('out.txt', 'w').write('written by code')"},
                    headers=writer)
        read = client.get(f"/code/sandbox/{box}/files/out.txt", headers=writer)
        assert read.json()["content"] == "written by code"
        listing = client.get(f"/code/sandbox/{box}", headers=writer).json()
        assert {f["path"] for f in listing["files"]} >= {"a.py", "b.txt", "out.txt"}
        assert client.delete(f"/code/sandbox/{box}/files/b.txt", headers=writer).status_code == 200
        assert client.delete(f"/code/sandbox/{box}", headers=writer).status_code == 200
        assert client.get(f"/code/sandbox/{box}", headers=writer).status_code == 404

    def test_an_encoded_traversal_is_refused(self, client, writer):
        box = client.post("/code/create/sandbox", json={}, headers=writer).json()["id"]
        got = client.get(f"/code/sandbox/{box}/files/%2e%2e/%2e%2e/etc/passwd", headers=writer)
        assert got.status_code in (400, 404)
        assert "root:" not in got.text


class TestPermsOverHTTP:
    def test_the_documented_form(self, client, admin):
        got = client.get("/code/create/sandbox/perms/s3?:=45|s6?:=256", headers=admin)
        assert got.status_code == 200
        body = got.json()
        assert body["changed"] == ["s3", "s6"]
        assert body["perms"]["s3"]["value"] == 45

    def test_the_form_as_asked_for(self, client, admin):
        got = client.get("/code/create/sandbox/perms|s1=?;s2?:=off", headers=admin)
        body = got.json()
        assert got.status_code == 200
        assert body["asked"] == {"s1": body["perms"]["s1"]}
        assert body["changed"] == ["s2"]

    def test_a_change_sticks(self, client, admin, reader):
        client.get("/code/create/sandbox/perms/s7?:=15", headers=admin)
        body = client.get("/code/create/sandbox/perms", headers=reader).json()
        assert body["perms"]["s7"]["value"] == 15

    def test_reading_is_open_changing_is_admin(self, client, reader, writer):
        assert client.get("/code/create/sandbox/perms|s2=?", headers=reader).status_code == 200
        got = client.get("/code/create/sandbox/perms/s2?:=off", headers=writer)
        assert got.status_code == 403
        assert got.json()["error"]["code"] == "AUTH_ADMIN_REQUIRED"

    def test_a_bad_clause_is_a_400_that_teaches_the_grammar(self, client, admin):
        got = client.get("/code/create/sandbox/perms|s2=on", headers=admin)
        assert got.status_code == 400
        error = got.json()["error"]
        assert "s2?:=on" in error["message"] and "grammar" in error["details"]

    @POSIX
    def test_execute_off_stops_runs_immediately(self, client, admin, writer):
        client.get("/code/create/sandbox/perms/s1?:=off", headers=admin)
        got = client.post("/code/create", json={"code": "print(1)"}, headers=writer)
        assert got.status_code == 403
        assert got.json()["error"]["details"]["reason"] == "execute_off"


def test_the_sandbox_folder_is_under_the_configured_dir(client, writer, tmp_path):
    client.get("/code", headers=writer)
    assert Path(client.app.state.t1_code_store.base) == tmp_path / "code"
