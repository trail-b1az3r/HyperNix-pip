"""SSPKID that survives the process, and identity from a config source.

Two halves of one release item.

**The registry used to be memory-only.** That is the same defect that
made every key `gkey` minted carry server ID ``00001-A1``: each `gkey`
invocation is its own process and the server is another, so an
assignment made in one was invisible to the other and gone on the next
restart. An identifier that does not outlive the process that issued it
cannot identify anything — and worse, it comes back around, handing
``#1`` to a second key while an audit trail still names the first.

**`gkey create -Con`** takes a key's V1 Server ID and/or SSPKID from a
JSONL config, which only means something across processes if the first
half holds.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hypernix.security.keyconfig import (
    DEFAULT_CONFIG_PATH,
    MAX_CONFIG_BYTES,
    KeyConfigError,
    load_key_config,
    resolve_config_source,
)
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
from hypernix.security.t2keys import (
    SSPKID,
    ServerKeyRegistry,
    SSPKIDCollision,
    default_sspkid_store_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestTheRegistryOutlivesTheProcess:
    def test_assignments_survive_a_new_instance(self, tmp_path):
        first = ServerKeyRegistry(store_dir=tmp_path)
        first.assign("key-a", SSPKID(server_id="00042-C1", index=1))

        second = ServerKeyRegistry(store_dir=tmp_path)

        assert second.resolve("00042-C1#1") == "key-a"
        assert str(second.sspkid_for("key-a")) == "00042-C1#1"

    def test_an_index_is_not_handed_out_twice_after_a_restart(self, tmp_path):
        """The bug this exists to prevent, stated directly."""
        ServerKeyRegistry(store_dir=tmp_path).allocate("key-a", "00042-C1")
        ServerKeyRegistry(store_dir=tmp_path).allocate("key-b", "00042-C1")

        final = ServerKeyRegistry(store_dir=tmp_path)
        assert str(final.sspkid_for("key-a")) == "00042-C1#1"
        assert str(final.sspkid_for("key-b")) == "00042-C1#2"
        assert str(final.allocate("key-c", "00042-C1")) == "00042-C1#3"

    def test_a_collision_is_refused_across_processes(self, tmp_path):
        ServerKeyRegistry(store_dir=tmp_path).assign(
            "key-a", SSPKID(server_id="00042-C1", index=1)
        )
        later = ServerKeyRegistry(store_dir=tmp_path)
        with pytest.raises(SSPKIDCollision):
            later.assign("key-b", SSPKID(server_id="00042-C1", index=1))

    def test_a_release_persists_too(self, tmp_path):
        first = ServerKeyRegistry(store_dir=tmp_path)
        first.assign("key-a", SSPKID(server_id="00042-C1", index=1))
        first.release("key-a")

        assert ServerKeyRegistry(store_dir=tmp_path).resolve("00042-C1#1") is None

    def test_it_lives_beside_the_keys_but_not_among_them(self, tmp_path):
        """The Keymaster globs *.json in its store, and CI counts them.

        A registry file sitting at the top level is read as a malformed
        key on every start and counted as a leaked key at teardown.
        """
        registry = ServerKeyRegistry(store_dir=tmp_path)
        registry.assign("key-a", SSPKID(server_id="00042-C1", index=1))

        assert registry.path is not None
        assert registry.path.parent.name == "sspkid"
        assert not list(tmp_path.glob("*.json")), "registry is in the key glob"

    def test_a_keymaster_sharing_the_directory_is_undisturbed(self, tmp_path):
        registry = ServerKeyRegistry(store_dir=tmp_path)
        registry.assign("key-a", SSPKID(server_id="00042-C1", index=1))

        km = Keymaster(store_dir=tmp_path, auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.stop()

        reloaded = Keymaster(store_dir=tmp_path, auto_rotate=False)
        assert reloaded.get(meta.key_id) is not None
        reloaded.stop()

    def test_an_ephemeral_registry_touches_no_disk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("T1_KEYMASTER_DIR", str(tmp_path))
        registry = ServerKeyRegistry(store_dir=None)
        registry.assign("key-a", SSPKID(server_id="00042-C1", index=1))
        assert registry.path is None
        assert not list(tmp_path.rglob("*.json"))

    def test_a_corrupt_registry_does_not_stop_a_start(self, tmp_path):
        """Losing assignments is bad; refusing to boot is worse."""
        path = tmp_path / "sspkid"
        path.mkdir()
        (path / "registry.json").write_text("{ this is not json", encoding="utf-8")

        registry = ServerKeyRegistry(store_dir=tmp_path)
        assert len(registry) == 0
        registry.assign("key-a", SSPKID(server_id="00042-C1", index=1))
        assert registry.resolve("00042-C1#1") == "key-a"

    def test_the_default_follows_the_keymaster_directory(self, tmp_path, monkeypatch):
        """Both halves read T1_KEYMASTER_DIR, for the reason it exists."""
        monkeypatch.setenv("T1_KEYMASTER_DIR", str(tmp_path / "elsewhere"))
        assert default_sspkid_store_dir() == tmp_path / "elsewhere"


class TestReadingTheConfig:
    def _write(self, tmp_path: Path, *lines: str) -> Path:
        path = tmp_path / "fleet.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_later_lines_win(self, tmp_path):
        """It is an append-only log, so the current setting is the last."""
        path = self._write(
            tmp_path,
            json.dumps({"server_id": "00007-B1"}),
            json.dumps({"server_id": "00042-C1"}),
        )
        assert load_key_config(str(path)).server_id == "00042-C1"

    def test_a_malformed_line_is_skipped_not_fatal(self, tmp_path):
        """One bad entry must not break every key minted after it."""
        path = self._write(
            tmp_path,
            "this is not json",
            json.dumps({"server_id": "00042-C1"}),
            "{ neither is this",
        )
        config = load_key_config(str(path))
        assert config.server_id == "00042-C1"
        assert config.lines_skipped == 2

    def test_comments_and_blank_lines_are_not_counted_as_errors(self, tmp_path):
        path = self._write(
            tmp_path,
            "# the fleet config",
            "",
            json.dumps({"server_id": "00042-C1"}),
        )
        config = load_key_config(str(path))
        assert config.lines_skipped == 0
        assert config.lines_read == 1

    def test_it_reads_only_identity_fields(self, tmp_path):
        """A config source is somewhere else, and possibly someone else."""
        path = self._write(tmp_path, json.dumps({
            "server_id": "00042-C1",
            "scopes": "admin",
            "key_type": "admin",
            "expires": "2099-01-01",
            "access_level": 9,
        }))
        config = load_key_config(str(path))
        assert config.server_id == "00042-C1"
        for forbidden in ("scopes", "key_type", "expires", "access_level"):
            assert not hasattr(config, forbidden)

    def test_a_config_that_sets_nothing_is_an_error(self, tmp_path):
        path = self._write(tmp_path, json.dumps({"unrelated": 1}))
        with pytest.raises(KeyConfigError, match="set none of"):
            load_key_config(str(path))

    def test_an_index_without_a_server_is_refused(self, tmp_path):
        path = self._write(tmp_path, json.dumps({"sspkid_index": 3}))
        with pytest.raises(KeyConfigError, match="does not say which server"):
            load_key_config(str(path))

    @pytest.mark.parametrize("bad", [0, -1, "abc"])
    def test_a_bad_index_is_refused(self, tmp_path, bad):
        path = self._write(
            tmp_path, json.dumps({"server_id": "00042-C1", "sspkid_index": bad})
        )
        with pytest.raises(KeyConfigError):
            load_key_config(str(path))

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(KeyConfigError, match="No config at"):
            load_key_config(str(tmp_path / "absent.jsonl"))

    def test_an_oversized_config_is_refused(self, tmp_path):
        path = tmp_path / "huge.jsonl"
        path.write_text('{"server_id": "00042-C1"}\n' * (MAX_CONFIG_BYTES // 10), encoding="utf-8")
        with pytest.raises(KeyConfigError, match="log, not a config"):
            load_key_config(str(path))

    def test_an_unsupported_scheme_is_refused(self):
        with pytest.raises(KeyConfigError, match="Unsupported source scheme"):
            load_key_config("ftp://example.com/keys.jsonl")

    @pytest.mark.parametrize("source", [r"C:\keys\fleet.jsonl", "C:/keys/fleet.jsonl"])
    def test_a_windows_path_is_a_path_not_a_scheme(self, source):
        """``urlparse`` calls the drive letter a scheme named ``c``.

        Left alone that turns every local config on Windows into
        "Unsupported source scheme 'c'" -- a refusal that names nothing
        the operator can act on, for a file that is sitting right there.
        This runs everywhere because the bug is in the parsing, not in
        the filesystem: the path need not exist for the wrong branch to
        be taken.
        """
        with pytest.raises(KeyConfigError, match="No config at"):
            load_key_config(source)


class TestResolvingWhatWasTyped:
    @pytest.mark.parametrize("source", ["10.0.0.5", "10.0.0.5:8080", "config.example"])
    def test_a_bare_host_gets_the_default_path(self, source):
        assert resolve_config_source(source) == f"http://{source}{DEFAULT_CONFIG_PATH}"

    @pytest.mark.parametrize(
        "source",
        [
            "https://config.example/keys.jsonl",
            "http://10.0.0.5/other.jsonl",
            "./fleet.jsonl",
            "/etc/hypernix/fleet.jsonl",
            "~/fleet.jsonl",
            "fleet.jsonl",
        ],
    )
    def test_a_url_or_path_is_left_alone(self, source):
        assert resolve_config_source(source) == source

    def test_a_windows_path_is_not_read_as_host_and_port(self):
        assert resolve_config_source(r"C:\fleet.jsonl") == r"C:\fleet.jsonl"

    def test_an_empty_source_is_refused(self):
        with pytest.raises(KeyConfigError, match="needs an address"):
            resolve_config_source("   ")


def _gkey(*argv: str, store: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "hypernix.security.gkey_cli", *argv],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        env={
            **os.environ,
            "T1_KEYMASTER_DIR": str(store),
            "NO_COLOR": "1",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
    )


class TestGkeyCreateCon:
    def test_it_applies_both_fields(self, tmp_path):
        config = tmp_path / "fleet.jsonl"
        config.write_text(json.dumps({"server_id": "00042-C1", "sspkid_index": 4}) + "\n", encoding="utf-8")
        store = tmp_path / "store"

        result = _gkey("create", "--type", "user", "-Con", str(config), store=store)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "00042-C1" in result.stdout
        assert "00042-C1#4" in result.stdout

    def test_the_assignment_is_visible_to_another_process(self, tmp_path):
        """The whole point of persisting it."""
        config = tmp_path / "fleet.jsonl"
        config.write_text(json.dumps({"server_id": "00042-C1", "sspkid_index": 4}) + "\n", encoding="utf-8")
        store = tmp_path / "store"
        _gkey("create", "--type", "user", "-Con", str(config), store=store)

        registry = ServerKeyRegistry(store_dir=store)
        assert registry.resolve("00042-C1#4") is not None

    def test_a_second_key_cannot_take_the_same_sspkid(self, tmp_path):
        config = tmp_path / "fleet.jsonl"
        config.write_text(json.dumps({"server_id": "00042-C1", "sspkid_index": 4}) + "\n", encoding="utf-8")
        store = tmp_path / "store"
        _gkey("create", "--type", "user", "-Con", str(config), store=store)

        second = _gkey("create", "--type", "user", "-Con", str(config), store=store)

        assert second.returncode != 0
        assert "already assigned" in second.stderr

    def test_an_unreadable_config_fails_before_a_key_exists(self, tmp_path):
        """A key minted and then found unusable is a credential nobody
        is tracking."""
        store = tmp_path / "store"
        result = _gkey(
            "create", "--type", "user", "-Con", str(tmp_path / "absent.jsonl"),
            store=store,
        )
        assert result.returncode == 2
        assert not list(store.glob("*.json")), "a key was minted anyway"

    def test_a_bad_server_id_is_refused(self, tmp_path):
        config = tmp_path / "fleet.jsonl"
        config.write_text(json.dumps({"server_id": "not-an-id"}) + "\n", encoding="utf-8")
        result = _gkey(
            "create", "--type", "user", "-Con", str(config), store=tmp_path / "store"
        )
        assert result.returncode == 2
        assert "not a V1 Server ID" in result.stderr

    def test_without_con_nothing_changes(self, tmp_path):
        store = tmp_path / "store"
        result = _gkey("create", "--type", "user", store=store)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Config:" not in result.stdout

    def test_the_help_describes_it(self, tmp_path):
        result = _gkey("create", "--help", store=tmp_path / "store")
        assert "-Con" in result.stdout
        assert "jsonl" in result.stdout.lower() or "JSONL" in result.stdout


class TestNobodyWritesToTheOperatorsStoreByAccident:
    """A bare ServerKeyRegistry() reaches ~/.hypernix/keymaster.

    That is right for `gkey`, which is the operator running a command on
    their own machine. It is wrong for anything constructed incidentally
    — and `create_app()` used to construct one bare, so every test in
    this suite that built an app wrote SSPKID assignments into the
    developer's real key store. It did, once, before this test existed.
    """

    def test_create_app_keeps_the_registry_beside_its_own_keys(self, tmp_path):
        fastapi = pytest.importorskip("fastapi")  # noqa: F841
        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.t1api.app import create_app
        from hypernix.t1api.config import T1APIConfig

        store = tmp_path / "keys"
        km = Keymaster(store_dir=store, auto_rotate=False)
        gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False)
        config = T1APIConfig(
            token_secret="test-secret-value-that-is-long-enough",
            db_path=str(tmp_path / "t1.sqlite3"),
            module_storage_dir=str(tmp_path / "modules"),
            hyperlink_files_dir=str(tmp_path / "files"),
        )
        # Note: keymaster_dir is unset, which is the case that made the
        # config an unreliable source for this — it is None whenever
        # T1_KEYMASTER_DIR is not exported.
        assert config.keymaster_dir is None

        app = create_app(config=config, keymaster=km, gatekeeper=gk)
        registry = app.state.t1_server_key_registry

        assert registry.path is not None
        assert store in registry.path.parents

    def test_the_default_is_the_shared_store_not_an_ephemeral_one(self, tmp_path, monkeypatch):
        """`store_dir=None` means ephemeral; omitting it must not."""
        monkeypatch.setenv("T1_KEYMASTER_DIR", str(tmp_path))
        assert ServerKeyRegistry().path is not None
        assert ServerKeyRegistry(store_dir=None).path is None
