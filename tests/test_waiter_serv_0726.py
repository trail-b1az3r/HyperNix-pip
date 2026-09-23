"""``waiter serv`` in 0.72.6: grouped letters, -b, and the new flags.

The argument rules are tested on their own; the flags that talk to a
server run the real ``waiter`` CLI against the real app, through the
SDK bridge.
"""
from __future__ import annotations

import json
import zipfile

import pytest

from hypernix.waiter.servargs import ServArgsError, expand

# ---------------------------------------------------------------------------
# Grouped letters and -b
# ---------------------------------------------------------------------------


class TestGroupedLetters:
    def test_the_example_from_the_request(self):
        got = expand(["-ArEK", "T1_key", "-I", "10.0.0.2"]).argv
        assert got == ["--auto", "--refresh", "--encrypt", "--key", "T1_key", "-I", "10.0.0.2"]

    def test_any_order_and_any_combination(self):
        assert expand(["-EA"]).argv == ["--encrypt", "--auto"]
        assert expand(["-YSc"]).argv == ["--info", "--security-check", "--conceal"]

    def test_a_value_letter_can_stand_alone_before_its_flags(self):
        assert expand(["-K", "T1_k", "-A"]).argv == ["-K", "T1_k", "-A"]

    def test_a_value_letter_in_the_middle_is_refused_not_guessed(self):
        with pytest.raises(ServArgsError, match="end its group"):
            expand(["-AKE", "T1_k"])

    def test_a_lone_r_still_forces_a_limit(self):
        assert expand(["-r", "key:abc=5/60s"]).argv == ["-r", "key:abc=5/60s"]

    def test_rf_and_ud_keep_their_meanings(self):
        assert expand(["-ARf"]).argv == ["--auto", "--force-refresh"]
        assert expand(["-Rf"]).argv == ["--force-refresh"]
        assert expand(["-ud"]).argv == ["--update-exact"]
        assert expand(["-Aud"]).argv == ["--auto", "--update-exact"]

    def test_an_unknown_letter_says_which(self):
        with pytest.raises(ServArgsError, match="-Q"):
            expand(["-AQ"])


class TestBundle:
    def test_without_b_a_bare_string_is_an_error(self):
        with pytest.raises(ServArgsError, match="add -b"):
            expand(["-A", "10.0.0.2"])

    def test_b_works_strings_out(self, tmp_path):
        cfg = tmp_path / "cfg.jsonl"
        cfg.write_text("{}")
        result = expand(["-bA", "T2_keyish", "100.64.0.7", "8123", "plan=pro", str(cfg)])
        assert result.argv[:2] == ["--bundle", "--auto"]
        assert dict(result.assigned) == {
            "T2_keyish": "--key", "100.64.0.7": "--server", "8123": "--port",
            "plan=pro": "--config", str(cfg): "--config-file",
        }

    @pytest.mark.parametrize("text,option", [
        ("https://t1.example.net", "--server"),
        ("t1.example.net", "--server"),
        ("localhost", "--server"),
        ("::1", "--server"),
        ("T2CK_abc", "--key"),
        ("key:abc=60/60s", "--force-limit"),
    ])
    def test_what_each_looks_like(self, text, option):
        assert expand(["-b", text]).assigned == [(text, option)]

    def test_a_range_is_never_guessed(self):
        with pytest.raises(ServArgsError, match="-B .*-W"):
            expand(["-b", "10.0.0.0/8"])

    def test_two_servers_is_ambiguous(self):
        with pytest.raises(ServArgsError, match="both look like"):
            expand(["-b", "10.0.0.1", "10.0.0.2"])

    def test_an_explicit_option_wins_and_a_duplicate_is_refused(self):
        with pytest.raises(ServArgsError, match="already given"):
            expand(["-b", "-I", "10.0.0.1", "10.0.0.2"])

    def test_a_kit_is_recognised(self, tmp_path):
        kit = _make_kit(tmp_path)
        assert expand(["-b", str(kit)]).assigned == [(str(kit), "--kit-install")]


# ---------------------------------------------------------------------------
# Kits
# ---------------------------------------------------------------------------


def _make_kit(root, name="hello-kit", kind="waiter", body="def main(argv):\n    print('hi', *argv)\n    return 3\n"):
    folder = root / name
    folder.mkdir()
    (folder / "kit.json").write_text(json.dumps({
        "name": name, "version": "1.0.0", "kind": kind, "description": "says hi",
        "commands": {"hi": "hello:main"} if kind == "waiter" else {},
    }))
    (folder / "hello.py").write_text(body)
    return folder


@pytest.fixture
def kit_home(tmp_path, monkeypatch):
    import hypernix.waiter.local_config as local_config

    monkeypatch.setattr(local_config, "_DEFAULT_CONFIG_DIR", tmp_path / "waiter")
    return tmp_path / "waiter" / "kits"


class TestKits:
    def test_install_list_run_remove(self, tmp_path, kit_home, capsys):
        from hypernix.waiter.cli import main

        source = _make_kit(tmp_path)
        assert main(["serv", "-k", str(source)]) == 0
        assert (kit_home / "hello-kit" / "kit.json").is_file()
        capsys.readouterr()
        assert main(["kits", "list", "--json"]) == 0
        listed = json.loads(capsys.readouterr().out)
        assert [k["name"] for k in listed] == ["hello-kit"]
        assert main(["kits", "run", "hello-kit", "hi", "there"]) == 3
        assert "hi there" in capsys.readouterr().out
        assert main(["kits", "remove", "hello-kit"]) == 0
        assert not (kit_home / "hello-kit").exists()

    def test_a_zip_installs_the_same(self, tmp_path, kit_home):
        from hypernix.waiter.kits import install

        folder = _make_kit(tmp_path)
        archive = tmp_path / "kit.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for path in folder.rglob("*"):
                zf.write(path, path.relative_to(tmp_path))
        assert install(archive).name == "hello-kit"

    def test_a_zip_cannot_write_outside_its_folder(self, tmp_path, kit_home):
        from hypernix.waiter.kits import KitError, install

        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("kit.json", json.dumps({"name": "evil", "version": "1", "kind": "other"}))
            zf.writestr("../../escaped.txt", "x")
        with pytest.raises(KitError, match="outside"):
            install(archive)
        assert not (tmp_path / "escaped.txt").exists()

    def test_a_bad_manifest_is_refused(self, tmp_path, kit_home):
        from hypernix.waiter.kits import KitError, install

        folder = tmp_path / "bad"
        folder.mkdir()
        (folder / "kit.json").write_text(json.dumps({"name": "Bad Name", "version": "1"}))
        with pytest.raises(KitError, match="lowercase"):
            install(folder)

    def test_other_kinds_install_for_their_program_to_find(self, tmp_path, kit_home):
        from hypernix.waiter.kits import install, installed

        install(_make_kit(tmp_path, name="sdk-extra", kind="t1api-client"))
        assert [k.name for k in installed(kind="t1api-client")] == ["sdk-extra"]
        assert installed(kind="waiter") == []


# ---------------------------------------------------------------------------
# -e: a locked config
# ---------------------------------------------------------------------------


class TestLock:
    def test_lock_then_read_with_the_password(self, tmp_path, monkeypatch):
        pytest.importorskip("cryptography")
        from hypernix.waiter.local_config import (
            LOCK_PREFIX,
            WaiterConfigLocked,
            WaiterConfigStore,
            WaiterLocalConfig,
        )

        path = tmp_path / "cfg.jsonl"
        WaiterConfigStore(path, password="correct horse").save(WaiterLocalConfig(server="s", key="T1_k"))
        assert path.read_text().startswith(LOCK_PREFIX)
        assert "T1_k" not in path.read_text()
        with pytest.raises(WaiterConfigLocked, match="locked"):
            WaiterConfigStore(path).load()
        with pytest.raises(WaiterConfigLocked, match="wrong password"):
            WaiterConfigStore(path, password="nope").load()
        store = WaiterConfigStore(path, password_provider=lambda: "correct horse")
        cfg = store.load()
        assert cfg.key == "T1_k"
        # Once opened it stays locked on the next save.
        store.save(cfg)
        assert path.read_text().startswith(LOCK_PREFIX)

    def test_the_cli_locks_and_later_commands_read_it(self, tmp_path, monkeypatch, capsys):
        pytest.importorskip("cryptography")
        from hypernix.waiter.cli import main

        path = tmp_path / "cfg.jsonl"
        path.write_text(json.dumps({"server": "10.9.9.9", "key": "T1_saved"}))
        monkeypatch.setenv("HNX_WAITER_PASSWORD", "a long password")
        assert main(["serv", "-e", "-F", str(path)]) == 0
        assert "T1_saved" not in path.read_text()
        capsys.readouterr()
        assert main(["config", "-F", str(path), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["server"] == "10.9.9.9"
        monkeypatch.delenv("HNX_WAITER_PASSWORD")
        assert main(["config", "-F", str(path), "--json"]) == 1


# ---------------------------------------------------------------------------
# Against the real app
# ---------------------------------------------------------------------------


@pytest.fixture
def live(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from conftest import clear_t1_config
    from fastapi.testclient import TestClient
    from sdk_bridge import sdk_client

    import hypernix.waiter.cli as waiter_cli
    from hypernix.security.gatekeeper import Gatekeeper
    from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
    from hypernix.t1api.app import create_app
    from hypernix.t1api.config import T1APIConfig
    from hypernix.waiter.client import T1Client as WaiterClient

    clear_t1_config(monkeypatch)
    km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
    user = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE})
    admin = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.READ, KeyScope.WRITE, KeyScope.ADMIN})
    app = create_app(
        config=T1APIConfig(
            token_secret="test-secret-value-that-is-long-enough",
            db_path=str(tmp_path / "t1.sqlite3"),
            module_storage_dir=str(tmp_path / "m"),
            hyperlink_files_dir=str(tmp_path / "f"),
            server_name="live-box", server_owner="Robin", server_description="for tests",
        ),
        keymaster=km,
        gatekeeper=Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False),
    )
    test_client = TestClient(app)

    def factory(base_url, credential=None, *args, **kwargs):
        return sdk_client(test_client, credential, cls=WaiterClient)

    monkeypatch.setattr(waiter_cli, "T1Client", factory)
    return {"cfg": tmp_path / "cfg.jsonl", "user": user, "admin": admin, "app": app}


def _serv(*argv):
    from hypernix.waiter.cli import main

    return main(["serv", *argv])


class TestAgainstTheServer:
    def test_info(self, live, capsys):
        assert _serv("-Y", "-I", "http://testserver", "-F", str(live["cfg"])) == 0
        out = capsys.readouterr().out
        assert "live-box" in out and "Robin" in out and "for tests" in out

    def test_the_request_example_seals_the_key(self, live, capsys):
        pytest.importorskip("cryptography")
        from hypernix.waiter.local_config import WaiterConfigStore

        assert _serv("-ArEK", live["user"].key, "-I", "http://testserver", "-F", str(live["cfg"])) == 0
        saved = WaiterConfigStore(live["cfg"]).load()
        assert saved.key.startswith("T2CK_")
        assert live["user"].key not in live["cfg"].read_text()
        out = capsys.readouterr()
        assert "v2.1" in out.out + out.err

    def test_bundle_does_the_same_without_letters_for_the_strings(self, live):
        pytest.importorskip("cryptography")
        from hypernix.waiter.local_config import WaiterConfigStore

        assert _serv("-bAE", "http://testserver", live["user"].key, "-F", str(live["cfg"])) == 0
        assert WaiterConfigStore(live["cfg"]).load().key.startswith("T2CK_")

    def test_conceal(self, live):
        from hypernix.waiter.local_config import WaiterConfigStore

        assert _serv("-Ac", "-I", "http://testserver", "-K", live["user"].key, "-F", str(live["cfg"])) == 0
        assert WaiterConfigStore(live["cfg"]).load().concealed is True
        assert live["app"].state.t1_conceal.is_concealed(live["user"].key_id)

    def test_update_is_nothing_to_do_when_the_versions_match(self, live, capsys, monkeypatch):
        import subprocess

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("pip ran"))
        assert _serv("-u", "-I", "http://testserver", "-F", str(live["cfg"])) == 0
        captured = capsys.readouterr()
        assert "nothing to do" in (captured.out + captured.err).lower()

    def test_update_exact_runs_pip_for_the_servers_version(self, live, monkeypatch):
        from hypernix.waiter import servops

        monkeypatch.setattr(servops, "local_version", lambda: "0.1.0")
        calls = []
        monkeypatch.setattr(servops.subprocess, "run",
                            lambda argv, check=False: calls.append(argv) or type("R", (), {"returncode": 0})())
        assert _serv("-ud", "-I", "http://testserver", "-F", str(live["cfg"])) == 0
        assert calls and calls[0][-1].startswith("hypernix==")

    def test_security_check(self, live, capsys):
        code = _serv("-S", "-I", "http://testserver", "-K", live["user"].key, "-F", str(live["cfg"]))
        out = capsys.readouterr().out
        assert "[client]" in out and "[server]" in out
        assert "refuses requests without a key" in out
        assert code in (0, 1)

    def test_control_needs_a_level_nine_t2_admin(self, live, capsys, monkeypatch):
        from hypernix.security.t2keys import T2KeyGenerator

        opened = []
        monkeypatch.setattr("hypernix.waiter.tui.run", lambda client, control=False: opened.append(control) or 0)
        # A T1 admin key: refused, because -T is for T2 keys.
        assert _serv("-T", "-I", "http://testserver", "-K", live["admin"].key, "-F", str(live["cfg"])) == 1
        # A level-8 T2 admin key: refused.
        eight = T2KeyGenerator.from_t1_admin(live["admin"].key, access_level=8).raw
        assert _serv("-T", "-I", "http://testserver", "-K", eight, "-F", str(live["cfg"])) == 1
        # A level-9 T2 user key: refused.
        user_nine = T2KeyGenerator.from_t1(live["user"].key, access_level=9).raw
        assert _serv("-T", "-I", "http://testserver", "-K", user_nine, "-F", str(live["cfg"])) == 1
        assert opened == []
        nine = T2KeyGenerator.from_t1_admin(live["admin"].key, access_level=9).raw
        assert _serv("-T", "-I", "http://testserver", "-K", nine, "-F", str(live["cfg"])) == 0
        assert opened == [True]


class TestControlPane:
    class Fake:
        credential = "T2_admin"
        base_url = "http://x"

        def __init__(self):
            self.calls = []
            self.allow = True

        def __getattr__(self, name):
            return lambda *a, **k: {}

        def list_models(self):
            return {"models": []}

        def list_servers(self):
            return {"servers": []}

        def list_modules(self):
            return {"modules": []}

        def list_events(self, **_):
            return {"events": []}

        def status(self):
            return {"server_name": "box", "hypernix_version": "0.72.6", "environment": "production",
                    "production_warnings": ["T1_TOKEN_SECRET is short"]}

        def network_policy(self):
            return {"allow_unlisted_clients": self.allow,
                    "entries": [{"kind": "blacklist", "cidr": "203.0.113.0/24", "reason": "spam"}]}

        def list_keys(self):
            from hypernix.t1sdk.models import KeyInfo

            return [KeyInfo(key_id="abcdef123456", key_type="user", scopes=["read"])]

        def audit_events(self, **filters):
            self.calls.append(("audit", filters))
            return {"events": [{"action": "security.auth_invalid_key", "outcome": "denied",
                                "client_ip": "198.51.100.9"}]}

        def blacklist_ip(self, cidr, reason=""):
            self.calls.append(("block", cidr))

        def set_allow_unlisted(self, enabled):
            self.calls.append(("unlisted", enabled))
            self.allow = enabled

    def test_control_mode_adds_the_pane_and_its_data(self):
        from hypernix.waiter.tui import CONTROL_PANE, DashboardController, render_pane

        client = self.Fake()
        controller = DashboardController(client, control=True)
        assert controller.panes[0] == CONTROL_PANE
        state = controller.refresh()
        text = "\n".join(render_pane(state, CONTROL_PANE))
        assert "203.0.113.0/24" in text and "abcdef123456" in text
        assert "security.auth_invalid_key" in text and "T1_TOKEN_SECRET" in text
        assert ("audit", {"category": "security", "limit": 20}) in client.calls

    def test_actions_go_to_the_server_and_refresh(self):
        from hypernix.waiter.tui import DashboardController

        client = self.Fake()
        controller = DashboardController(client, control=True)
        controller.refresh()
        assert "done" in controller.block_address("192.0.2.1")
        controller.toggle_allow_unlisted()
        assert ("block", "192.0.2.1") in client.calls and ("unlisted", False) in client.calls

    def test_without_control_mode_there_is_no_pane_and_no_action(self):
        from hypernix.waiter.tui import CONTROL_PANE, DashboardController

        client = self.Fake()
        controller = DashboardController(client)
        assert CONTROL_PANE not in controller.panes
        assert "-T" in controller.block_address("192.0.2.1")
        assert ("block", "192.0.2.1") not in client.calls
