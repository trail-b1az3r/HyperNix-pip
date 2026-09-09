"""Finding a server, and knowing it is the right one.

0.72.4 item 1: a phone should find the machines its owner runs rather
than being told to type a tailnet name it does not know. Item 14 draws
the line that shapes how: *never allow a public unauthenticated
connection to escalate into administrator access*, and the stated
constraints add two more —

* discovery must be separated from connection, and neither is execution;
* a human-readable server name must not, on its own, authenticate
  anything.

Most of this file is those three properties, because the feature is a
list and the properties are where the damage lives.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hypernix.hyperlink import identity, peers


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture(autouse=True)
def _no_identity_cache():
    """The fingerprint is memoised per seed path; tests make new paths."""
    identity._CACHE.clear()
    yield
    identity._CACHE.clear()


# ===========================================================================
# Identity
# ===========================================================================


class TestTheFingerprint:
    def test_it_is_stable_across_calls(self, tmp_path):
        assert identity.fingerprint(tmp_path) == identity.fingerprint(tmp_path)

    def test_it_survives_a_restart(self, tmp_path):
        """The whole value of pinning it. A fingerprint that changed on
        restart would train people to click through the warning."""
        first = identity.fingerprint(tmp_path)
        identity._CACHE.clear()

        assert identity.fingerprint(tmp_path) == first

    def test_two_installations_differ(self, tmp_path):
        assert identity.fingerprint(tmp_path / "a") != identity.fingerprint(tmp_path / "b")

    def test_it_is_not_derived_from_the_hostname(self, tmp_path, monkeypatch):
        """A name is what an attacker can look up and claim. If the
        fingerprint moved with it, it would authenticate nothing."""
        import socket

        first = identity.fingerprint(tmp_path / "a")
        monkeypatch.setattr(socket, "gethostname", lambda: "somebody-elses-desktop")
        identity._CACHE.clear()

        assert identity.fingerprint(tmp_path / "a") == first

    def test_the_seed_is_not_world_readable(self, tmp_path):
        """Anyone who can read the seed can claim to be this machine."""
        identity.fingerprint(tmp_path)
        mode = identity.seed_path(tmp_path).stat().st_mode & 0o777

        assert mode == 0o600, f"seed is {oct(mode)}"

    def test_the_fingerprint_does_not_contain_the_seed(self, tmp_path):
        value = identity.fingerprint(tmp_path)
        seed = identity.seed_path(tmp_path).read_bytes()

        assert seed.hex() not in value
        assert len(value) == identity.FINGERPRINT_LENGTH

    def test_an_unwritable_config_dir_still_answers(self, tmp_path):
        """A read-only config directory is worse — the fingerprint is
        then per-process — but it is not a reason to fail the request
        that asked for it."""
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")

        value = identity.fingerprint(blocked)

        assert len(value) == identity.FINGERPRINT_LENGTH

    def test_a_truncated_seed_is_replaced(self, tmp_path):
        """A half-written seed file is weak key material, and using it
        because it happens to exist is the wrong recovery."""
        path = identity.seed_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"short")

        identity.fingerprint(tmp_path)

        assert len(path.read_bytes()) >= 16


# ===========================================================================
# Peer discovery
# ===========================================================================


def _status(*entries: dict) -> str:
    return json.dumps({"Peer": {f"key{i}": e for i, e in enumerate(entries)}})


class _Result:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


class TestReadingTheTailnet:
    def test_no_tailscale_is_an_empty_list_not_an_error(self, monkeypatch):
        """A tailnet is optional. A server on a plain LAN works fine and
        must not raise on the way to saying so."""
        monkeypatch.setattr(peers, "_tailscale_binary", lambda: "")

        assert peers.tailnet_peers() == []

    def test_garbage_from_the_cli_is_an_empty_list(self, monkeypatch):
        monkeypatch.setattr(peers, "_tailscale_binary", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(
            peers.subprocess, "run", lambda *a, **k: _Result("not json at all")
        )

        assert peers.tailnet_peers() == []

    def test_it_reads_names_addresses_and_liveness(self, monkeypatch):
        monkeypatch.setattr(peers, "_tailscale_binary", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(peers.subprocess, "run", lambda *a, **k: _Result(_status(
            {"DNSName": "laptop.tail1234.ts.net.", "TailscaleIPs": ["100.64.0.2"],
             "Online": True, "OS": "linux"},
            {"DNSName": "phone.tail1234.ts.net.", "TailscaleIPs": ["100.64.0.3"],
             "Online": False, "OS": "iOS"},
        )))

        found = peers.tailnet_peers()

        assert [p.name for p in found] == [
            "laptop.tail1234.ts.net", "phone.tail1234.ts.net"
        ]
        assert found[0].online is True
        assert found[1].online is False

    def test_online_peers_come_first(self, monkeypatch):
        monkeypatch.setattr(peers, "_tailscale_binary", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(peers.subprocess, "run", lambda *a, **k: _Result(_status(
            {"DNSName": "a.ts.net", "TailscaleIPs": ["100.64.0.2"], "Online": False},
            {"DNSName": "z.ts.net", "TailscaleIPs": ["100.64.0.3"], "Online": True},
        )))

        assert [p.name for p in peers.tailnet_peers()] == ["z.ts.net", "a.ts.net"]

    def test_ipv6_only_peers_are_skipped(self, monkeypatch):
        """HyperLink needs an IPv4 address; a peer with only a v6 one
        cannot be reached and listing it would only waste a probe."""
        monkeypatch.setattr(peers, "_tailscale_binary", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(peers.subprocess, "run", lambda *a, **k: _Result(_status(
            {"DNSName": "v6.ts.net", "TailscaleIPs": ["fd7a::1"], "Online": True},
        )))

        assert peers.tailnet_peers() == []

    def test_a_huge_tailnet_is_capped(self, monkeypatch):
        """A work tailnet can be thousands of machines, and probing all
        of them from a request handler holds the connection for a
        minute."""
        monkeypatch.setattr(peers, "_tailscale_binary", lambda: "/usr/bin/tailscale")
        many = [
            {"DNSName": f"host{i}.ts.net", "TailscaleIPs": [f"100.64.1.{i % 250}"],
             "Online": True}
            for i in range(500)
        ]
        monkeypatch.setattr(
            peers.subprocess, "run", lambda *a, **k: _Result(_status(*many))
        )

        assert len(peers.tailnet_peers()) == peers.MAX_PEERS


class TestProbing:
    def test_a_peer_that_answers_as_hypernix_is_reachable(self, monkeypatch):
        peer = peers.Peer(name="laptop.ts.net", address="100.64.0.2", url="")
        monkeypatch.setattr(
            peers, "probe",
            lambda p, **k: peers.Peer(**{
                **p.__dict__, "reachable": True, "server_name": "laptop",
                "t1_version": "1.0.26", "url": "http://laptop.ts.net:8000",
            }),
        )

        assert peers.probe(peer, port=8000).reachable is True

    def test_something_that_is_not_hypernix_is_not_reachable(self, monkeypatch, tmp_path):
        """A web server on the same port answers 200 with HTML. Treating
        that as a HyperNix server would put a stranger in the list."""
        class _Response:
            def read(self, _n=None):
                return b"<html>hello</html>"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(peers.urllib.request, "urlopen", lambda *a, **k: _Response())
        result = peers.probe(peers.Peer(name="x.ts.net", address="100.64.0.2", url=""), port=8000)

        assert result.reachable is False

    def test_a_json_object_without_status_is_refused(self, monkeypatch):
        class _Response:
            def read(self, _n=None):
                return b'{"hello": "world"}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(peers.urllib.request, "urlopen", lambda *a, **k: _Response())
        result = peers.probe(peers.Peer(name="x.ts.net", address="100.64.0.2", url=""), port=8000)

        assert result.reachable is False
        assert "not a HyperNix server" in result.detail

    def test_a_refused_connection_is_recorded_not_raised(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("Connection refused")

        monkeypatch.setattr(peers.urllib.request, "urlopen", boom)
        result = peers.probe(peers.Peer(name="x.ts.net", address="100.64.0.2", url=""), port=8000)

        assert result.reachable is False
        assert "refused" in result.detail

    def test_the_probe_body_is_bounded(self, monkeypatch):
        """A peer must not get to decide how much memory this process
        spends answering one discovery call."""
        seen = {}

        class _Response:
            def read(self, n=None):
                seen["limit"] = n
                return b'{"status": "ok"}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(peers.urllib.request, "urlopen", lambda *a, **k: _Response())
        peers.probe(peers.Peer(name="x.ts.net", address="100.64.0.2", url=""), port=8000)

        assert seen["limit"] == peers.MAX_BODY


class TestDiscoveryIsNotTrust:
    def test_every_result_says_it_is_unverified(self, monkeypatch):
        """In the payload, not only in the docs. A client reading this
        list has been told, in the data it parses, that nothing here has
        been authenticated."""
        monkeypatch.setattr(peers, "_tailscale_binary", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(peers.subprocess, "run", lambda *a, **k: _Result(_status(
            {"DNSName": "laptop.ts.net", "TailscaleIPs": ["100.64.0.2"], "Online": True},
        )))

        class _Response:
            def read(self, _n=None):
                return b'{"status": "ok", "server_name": "laptop"}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(peers.urllib.request, "urlopen", lambda *a, **k: _Response())
        found = peers.discover(port=8000)

        assert found and all(p.to_dict()["verified"] is False for p in found)

    def test_a_peer_cannot_name_itself_into_being_trusted(self, monkeypatch):
        """The name in the payload is what the peer calls itself. It is
        carried for a human choosing from a list and is never an
        identity — so a machine claiming to be "desktop" gets exactly
        the same unverified row as any other."""
        monkeypatch.setattr(peers, "_tailscale_binary", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(peers.subprocess, "run", lambda *a, **k: _Result(_status(
            {"DNSName": "impostor.ts.net", "TailscaleIPs": ["100.64.0.9"], "Online": True},
        )))

        class _Response:
            def read(self, _n=None):
                return b'{"status": "ok", "server_name": "desktop"}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(peers.urllib.request, "urlopen", lambda *a, **k: _Response())
        row = peers.discover(port=8000)[0].to_dict()

        assert row["server_name"] == "desktop"
        assert row["verified"] is False

    def test_a_peers_reply_never_chooses_a_code_path(self, monkeypatch):
        """The constraint behind all of this: discovery must not become
        execution. Everything taken from a peer's response is copied
        into a string field; nothing is dispatched on, and the only
        thing that changes behaviour is whether the JSON had a "status"
        key at all."""
        source = Path(peers.__file__).read_text(encoding="utf-8")
        body = source.split('"""', 2)[2]

        for forbidden in ("eval(", "exec(", "subprocess.Popen", "os.system", "__import__"):
            assert forbidden not in body, f"{forbidden} in hyperlink.peers"
        # The one subprocess call is the tailscale CLI, with a fixed argv.
        assert body.count("subprocess.run") == 1
        assert '[exe, "status", "--json"]' in body


# ===========================================================================
# Over HTTP
# ===========================================================================

# The exact spelling the lint-regression check looks for: adding a
# reason= kwarg breaks its substring match, and the guard then reads as
# absent even though it is right there.
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ambient_trust(monkeypatch):
    for name in (
        "T1_TRUSTED_NETWORK",
        "T1_TRUSTED_NETWORK_LAN",
        "T1_TRUSTED_NETWORK_TAILNET",
        "T1_TRUSTED_NETWORK_PARTIAL_ADMIN",
        "T1_TRUSTED_PROXIES",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


@pytest.fixture
def gk(km, tmp_path) -> Gatekeeper:
    return Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)


def build_client(km, gk, tmp_path, *, peer: str = "127.0.0.1", **trust) -> TestClient:
    config = T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough",
        db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"),
        hyperlink_files_dir=str(tmp_path / "files"),
        **trust,
    )
    return TestClient(create_app(config=config, keymaster=km, gatekeeper=gk),
                      client=(peer, 51234))


@pytest.fixture
def client(km, gk, tmp_path) -> TestClient:
    return build_client(km, gk, tmp_path)


@pytest.fixture
def admin_key(km) -> str:
    return km.create(
        key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}
    ).key


@pytest.fixture
def device_token(client, admin_key) -> str:
    code = client.post(
        "/hyperlink/pair", json={"label": "phone"},
        headers={"Authorization": f"Bearer {admin_key}"},
    ).json()["code"]
    return client.post(
        "/hyperlink/pair/redeem", json={"code": code, "device_name": "Test iPhone"}
    ).json()["device_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestEndpointsCarryAnIdentity:
    def test_it_reports_a_fingerprint(self, client, admin_key):
        body = client.get("/hyperlink/endpoints", headers=_auth(admin_key)).json()

        assert len(body["server_fingerprint"]) == identity.FINGERPRINT_LENGTH

    def test_the_fingerprint_is_the_same_on_every_call(self, client, admin_key):
        """A client pins it. One that moved between calls would be
        useless for exactly the check it exists for."""
        first = client.get("/hyperlink/endpoints", headers=_auth(admin_key)).json()
        second = client.get("/hyperlink/endpoints", headers=_auth(admin_key)).json()

        assert first["server_fingerprint"] == second["server_fingerprint"]

    def test_it_is_not_served_to_an_unauthenticated_caller(self, client):
        """A stable machine identifier handed to anyone who can reach
        the port is reconnaissance for no gain."""
        assert client.get("/hyperlink/endpoints").status_code == 401

    def test_a_device_token_may_read_it(self, client, device_token):
        """The phone is the thing that pins it."""
        body = client.get("/hyperlink/endpoints", headers=_auth(device_token)).json()

        assert body["server_fingerprint"]


class TestEndpointsSayWhetherKeylessIsAvailable:
    def test_off_by_default(self, client, admin_key):
        body = client.get("/hyperlink/endpoints", headers=_auth(admin_key)).json()

        assert body["trusted_network"] is False
        assert body["keyless_available_here"] is False

    def test_it_answers_for_this_caller_not_in_general(self, km, gk, tmp_path, admin_key):
        """"The server allows keyless connections" and "this phone can
        make one" are different questions, and the app can only act on
        the second. Answering only the first makes it find out by
        failing."""
        lan = build_client(km, gk, tmp_path, peer="192.168.1.40", trusted_network=True)
        public = build_client(km, gk, tmp_path, peer="203.0.113.9", trusted_network=True)

        from_lan = lan.get("/hyperlink/endpoints", headers=_auth(admin_key)).json()
        from_public = public.get("/hyperlink/endpoints", headers=_auth(admin_key)).json()

        assert from_lan["trusted_network"] is True
        assert from_lan["keyless_available_here"] is True
        assert from_public["trusted_network"] is True
        assert from_public["keyless_available_here"] is False
        assert from_public["origin_trust"] == "public"


class TestPeers:
    def test_it_is_admin_only(self, client, device_token):
        """The answer is a map of somebody's private network. A phone's
        credential for *this* server is not authority to enumerate every
        other machine its owner runs."""
        assert client.get("/hyperlink/peers", headers=_auth(device_token)).status_code == 403

    def test_unauthenticated_is_refused(self, client):
        assert client.get("/hyperlink/peers").status_code == 401

    def test_no_tailnet_explains_itself(self, client, admin_key, monkeypatch):
        """An empty list with no reason reads as "discovery is broken"."""
        monkeypatch.setattr(peers, "_tailscale_binary", lambda: "")
        body = client.get("/hyperlink/peers", headers=_auth(admin_key)).json()

        assert body["peers"] == []
        assert body["tailscale"] is False
        assert body["detail"]

    def test_it_lists_probed_peers(self, client, admin_key, monkeypatch):
        monkeypatch.setattr(peers, "_tailscale_binary", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(peers.subprocess, "run", lambda *a, **k: _Result(_status(
            {"DNSName": "laptop.ts.net", "TailscaleIPs": ["100.64.0.2"], "Online": True},
        )))

        class _Response:
            def read(self, _n=None):
                return b'{"status": "ok", "server_name": "laptop", "t1_api_version": "1.0"}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(peers.urllib.request, "urlopen", lambda *a, **k: _Response())
        body = client.get("/hyperlink/peers", headers=_auth(admin_key)).json()

        assert body["count"] == 1
        assert body["reachable"] == 1
        row = body["peers"][0]
        assert row["name"] == "laptop.ts.net"
        assert row["verified"] is False
