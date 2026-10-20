"""``/net``: what ``hnx net`` did, as the T1 API (0.72.8a3).

Tailscale, ssh and rsync are replaced by a recorder, so nothing here
touches a network; what is checked is the argv each operation builds,
the JSON it answers, and who may call it.
"""
from __future__ import annotations

import json

import pytest

from hypernix.system import netops
from hypernix.system.netops import Run

STATUS = {
    "BackendState": "Running",
    "Self": {"HostName": "box", "DNSName": "box.tail.ts.net.", "TailscaleIPs": ["100.64.0.1"], "OS": "linux",
             "Online": True},
    "Peer": {"a": {"HostName": "gpu", "TailscaleIPs": ["100.64.0.2", "fd7a::2"], "OS": "linux", "Online": True},
             "b": {"HostName": "old", "TailscaleIPs": ["100.64.0.3"], "OS": "windows", "Online": False}},
}


class Recorder:
    def __init__(self, answers=None):
        self.calls: list[list[str]] = []
        self.answers = answers or {}

    def __call__(self, argv, timeout):
        argv = list(argv)
        self.calls.append(argv)
        key = " ".join(argv[:2])
        if key == "tailscale status":
            return Run(argv, 0, json.dumps(STATUS))
        if key in self.answers:
            return self.answers[key](argv)
        return Run(argv, 0, "")


class TestOperations:
    def test_status(self):
        out = netops.status(Recorder())
        assert out["running"] and out["self"]["dns_name"] == "box.tail.ts.net"
        assert [p["hostname"] for p in out["peers"]] == ["gpu", "old"]

    def test_tailscale_missing(self):
        out = netops.status(lambda argv, timeout: Run(list(argv), 127, err="not installed"))
        assert out == {"installed": False, "running": False, "self": None, "peers": [],
                       "hint": "Install Tailscale: https://tailscale.com/download"}

    def test_only_online_peers_are_worked_on(self):
        run = Recorder()
        netops.check_file("train.py", run)
        ssh = [c for c in run.calls if c[0] == "ssh"]
        assert len(ssh) == 1 and ssh[0][-2] == "100.64.0.2"

    def test_a_file_name_is_quoted_and_a_peer_is_not_an_option(self):
        run = Recorder()
        netops.check_file("x; rm -rf ~", run)
        ssh = next(c for c in run.calls if c[0] == "ssh")
        assert "--" in ssh and ssh[-1] == "test -f 'x; rm -rf ~'"

    def test_expose_without_apply_runs_nothing(self):
        run = Recorder()
        assert netops.expose(8080, run=run) == {"applied": False, "command": "tailscale serve --bg 8080"}
        assert run.calls == []

    def test_expose_without_permission_says_how(self):
        run = Recorder({"tailscale serve": lambda argv: Run(argv, 1, err="Access denied: serve config denied")})
        out = netops.expose(8080, apply=True, run=run)
        assert not out["applied"] and "--operator" in out["hint"]

    def test_trust_appends_host_keys(self, tmp_path):
        run = Recorder({"ssh-keyscan -H": lambda argv: Run(argv, 0, "|1|abc ssh-ed25519 AAAA\n# comment\n")})
        rows = netops.trust_peers(run, known_hosts=tmp_path / "known_hosts")
        assert rows == [{"peer": "100.64.0.2", "keys_added": 1, "error": ""}]
        assert (tmp_path / "known_hosts").read_text() == "|1|abc ssh-ed25519 AAAA\n"

    def test_sync(self, tmp_path):
        run = Recorder()
        rows = netops.sync_storage(run, config={"storage_dir": str(tmp_path / "s"), "ports": []}, dry_run=True)
        rsync = next(c for c in run.calls if c[0] == "rsync")
        assert "--dry-run" in rsync and rsync[-1] == f"100.64.0.2:{tmp_path / 's'}/" and rows[0]["code"] == 0

    def test_setup_returns_the_login_link(self, tmp_path):
        run = Recorder({"tailscale up": lambda argv: Run(argv, 1, err="To authenticate, visit:\n\n\thttps://login.tailscale.com/a/abc123\n")})
        out = netops.setup(run, key_path=tmp_path / "id_ed25519")
        assert out["login_url"] == "https://login.tailscale.com/a/abc123"
        assert any(c[0] == "ssh-keygen" for c in run.calls)

    def test_config_round_trip(self, tmp_path):
        path = tmp_path / "net.json"
        saved = netops.save_config({"ports": [8080, "443", 8080], "storage_dir": "~/x", "hub": 1}, path)
        assert saved["ports"] == [443, 8080] and saved["hub"] is True and "~" not in saved["storage_dir"]
        assert netops.load_config(path) == saved
        with pytest.raises(netops.NetError):
            netops.save_config({"ports": [70000]}, path)


@pytest.fixture
def api(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from conftest import clear_t1_config
    from fastapi.testclient import TestClient

    from hypernix.security.gatekeeper import Gatekeeper
    from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
    from hypernix.t1api.app import create_app
    from hypernix.t1api.config import T1APIConfig

    clear_t1_config(monkeypatch)
    monkeypatch.setattr(netops, "CONFIG_FILE", tmp_path / "net.json")
    run = Recorder()
    monkeypatch.setattr(netops, "RUN", run)
    km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
    gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False)
    config = T1APIConfig(token_secret="x" * 40, db_path=str(tmp_path / "t1.sqlite3"),
                         module_storage_dir=str(tmp_path / "m"), hyperlink_files_dir=str(tmp_path / "f"),
                         web_enabled=False)
    client = TestClient(create_app(config=config, keymaster=km, gatekeeper=gk), client=("203.0.113.5", 4000))
    admin = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}).key
    user = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key
    return client, {"Authorization": f"Bearer {admin}"}, {"Authorization": f"Bearer {user}"}, run, config


class TestTheRoutes:
    def test_status_and_peers(self, api):
        client, admin, _user, _run, _cfg = api
        assert client.get("/net/status", headers=admin).json()["running"] is True
        peers = client.get("/net/peers", headers=admin).json()["peers"]
        assert [p["hostname"] for p in peers] == ["gpu"]

    def test_a_user_key_is_refused(self, api):
        client, _admin, user, _run, _cfg = api
        assert client.get("/net/status", headers=user).status_code == 403

    def test_ports(self, api):
        client, admin, *_ = api
        assert client.post("/net/ports", json={"port": 8080}, headers=admin).json()["ports"] == [8080]
        assert client.delete("/net/ports/8080", headers=admin).json()["ports"] == []
        assert client.delete("/net/ports/9", headers=admin).status_code == 404

    def test_expose_plan_and_apply(self, api):
        client, admin, _user, run, _cfg = api
        plan = client.post("/net/expose", json={"port": 8080}, headers=admin).json()
        assert plan == {"applied": False, "command": "tailscale serve --bg 8080"}
        applied = client.post("/net/expose?confirm=true", json={"port": 8080, "apply": True}, headers=admin)
        assert applied.status_code == 200 and ["tailscale", "serve", "--bg", "8080"] in run.calls

    def test_running_on_peers_is_off_by_default(self, api):
        client, admin, *_ = api
        reply = client.post("/net/peers/run?confirm=true", json={"file": "train.py"}, headers=admin)
        assert reply.status_code == 501 and "T1_NET_REMOTE_RUN" in reply.json()["error"]["message"]

    def test_check_is_read_only_and_allowed(self, api):
        client, admin, *_ = api
        reply = client.post("/net/peers/check", json={"file": "train.py"}, headers=admin).json()
        assert reply["peers"] == [{"peer": "100.64.0.2", "exists": True}]

    def test_the_cli_says_it_is_deprecated(self, capsys, monkeypatch):
        from hypernix.system import net

        monkeypatch.setattr(net, "cmd_m_ip", lambda args: None)
        net.cli_main(["m-ip"])
        assert "deprecated" in capsys.readouterr().err and "GET /net/ip" in net.ROUTES["m-ip"]
