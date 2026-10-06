"""Why HyperLink could not reach a server that was running fine.

Three separate causes, each of which alone produces "the app times out
and there is nothing in the server log", because in all three cases
nothing ever arrives:

* The server advertised the wrong port. uvicorn owns the bind address and
  passes it to nobody, so the config's default — 8000 — went out in the
  endpoint list whatever port the server was actually on.
* iOS blocked the request before it left the phone. ATS exempts RFC 1918,
  and Tailscale is 100.64.0.0/10, which is not RFC 1918.
* Tailscale was not detected, and the server said nothing about why.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hypernix.hyperlink.discovery import (
    _TAILSCALE_PATHS,
    tailscale_diagnosis,
    tailscale_self,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import fastapi  # noqa: F401

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

needs_server = pytest.mark.skipif(not _HAS_FASTAPI, reason="needs the [t1api] extra")


@pytest.fixture
def stub_tailscale(tmp_path, monkeypatch):
    """Put a scripted `tailscale` first on PATH.

    The stub is Python rather than shell, and named for the platform.
    The code under test finds the binary with ``shutil.which`` and runs
    it with ``subprocess``; on Windows neither step can do anything with
    an extensionless ``#!/bin/sh`` file, so every diagnosis test was
    really asserting against "tailscale is not installed" -- the one
    message none of them is about.
    """

    def install(stdout: str = "", stderr: str = "", returncode: int = 0):
        body = (
            "import sys\n"
            f"sys.stdout.write({stdout!r})\n"
            f"sys.stderr.write({stderr!r})\n"
            f"raise SystemExit({returncode})\n"
        )
        if os.name == "nt":
            # PATHEXT is what lets shutil.which find this without an
            # extension, and CreateProcess runs a .cmd through cmd.exe.
            script = tmp_path / "tailscale_stub.py"
            script.write_text(body, encoding="utf-8")
            binary = tmp_path / "tailscale.cmd"
            binary.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
        else:
            binary = tmp_path / "tailscale"
            binary.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
            binary.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
        return binary

    return install


class TestTailscaleDiagnosis:
    """`tailscale_self` returns empty for four different reasons and says
    which applies to none of them. Fine for the server — a tailnet is
    optional — and useless for the operator staring at a phone that
    cannot connect, which is the case this exists for."""

    def test_logged_out_says_to_log_in(self, stub_tailscale):
        stub_tailscale(stdout='{"BackendState":"NeedsLogin","Self":{"TailscaleIPs":[]}}')
        assert "tailscale up" in tailscale_diagnosis()

    def test_a_dead_daemon_reports_the_real_error(self, stub_tailscale):
        stub_tailscale(stderr="failed to connect to tailscaled", returncode=1)
        assert "failed to connect to tailscaled" in tailscale_diagnosis()

    def test_garbage_output_is_survivable(self, stub_tailscale):
        stub_tailscale(stdout="not json")
        assert "not JSON" in tailscale_diagnosis()

    def test_a_v6_only_node_is_named_as_such(self, stub_tailscale):
        """HyperLink needs IPv4, and "no address" would be wrong here."""
        stub_tailscale(
            stdout='{"BackendState":"Running","Self":{"DNSName":"pc.ts.net.",'
                   '"TailscaleIPs":["fd7a::1"]}}'
        )
        assert "no IPv4" in tailscale_diagnosis()

    def test_a_working_node_diagnoses_nothing(self, stub_tailscale):
        stub_tailscale(
            stdout='{"BackendState":"Running","Self":{"DNSName":"pc.tail1.ts.net.",'
                   '"TailscaleIPs":["100.101.102.103","fd7a::1"]}}'
        )
        assert tailscale_diagnosis() == ""
        assert tailscale_self() == ("pc.tail1.ts.net", ["100.101.102.103"])

    def test_a_missing_binary_blames_PATH_not_the_install(self, monkeypatch, tmp_path):
        """A service manager's minimal PATH is the usual reason.

        The server came up, found no tailnet, advertised only its LAN
        address, and that reads as "Tailscale is broken" when Tailscale
        is fine and PATH is not.
        """
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(
            "hypernix.hyperlink.discovery._TAILSCALE_PATHS", ()
        )
        assert "PATH" in tailscale_diagnosis()

    def test_the_usual_install_locations_are_searched(self):
        for expected in ("/usr/bin/tailscale", "/usr/local/bin/tailscale"):
            assert expected in _TAILSCALE_PATHS


class TestATSCoversTailscale:
    """iOS blocked the request before it left the phone.

    `NSAllowsLocalNetworking` exempts link-local, .local and RFC 1918.
    Tailscale is 100.64.0.0/10 — shared address space, not RFC 1918 — so
    a tailnet address was refused by ATS with no packet sent and nothing
    for the server to log.
    """

    @pytest.fixture(scope="class")
    def ats(self):
        yaml = pytest.importorskip("yaml")
        project = yaml.safe_load((REPO_ROOT / "ios" / "project.yml").read_text(encoding="utf-8"))
        return project["targets"]["HyperLink"]["info"]["properties"][
            "NSAppTransportSecurity"
        ]

    def test_ts_net_is_excepted(self, ats):
        exception = ats["NSExceptionDomains"]["ts.net"]
        assert exception["NSExceptionAllowsInsecureHTTPLoads"] is True
        assert exception["NSIncludesSubdomains"] is True, (
            "MagicDNS names are subdomains of ts.net; without this only the "
            "bare domain is covered"
        )

    def test_arbitrary_loads_stay_off(self, ats):
        """The exception is scoped. A public http:// endpoint is still refused."""
        assert ats["NSAllowsArbitraryLoads"] is False

    def test_local_networking_is_still_allowed(self, ats):
        assert ats["NSAllowsLocalNetworking"] is True


@needs_server
class TestTheAdvertisedPort:
    """The server told the phone to connect somewhere it was not."""

    @pytest.fixture
    def app(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            "hypernix.security.keymaster._DEFAULT_STORE", tmp_path / "keymaster"
        )
        monkeypatch.setattr(
            "hypernix.security.gatekeeper._DEFAULT_DATA", tmp_path / "gatekeeper"
        )
        monkeypatch.setenv("T1_TOKEN_SECRET", "x" * 64)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1api.sqlite3"))
        monkeypatch.setenv("T1_HYPERLINK_ENABLED", "1")
        monkeypatch.delenv("T1_HYPERLINK_PORT", raising=False)
        from hypernix.t1api.app import create_app

        return create_app()

    @staticmethod
    def _endpoints(app, base_url: str):
        from fastapi.testclient import TestClient

        key = app.state.t1_bootstrap_key.key
        with TestClient(app, client=("127.0.0.1", 5000), base_url=base_url) as client:
            response = client.get(
                "/hyperlink/endpoints", headers={"Authorization": "Bearer " + key}
            )
        assert response.status_code == 200, response.text
        return response.json()["endpoints"]

    def test_it_advertises_the_port_the_request_arrived_on(self, app):
        """Whatever address a client just reached the server on is, by
        construction, one that works."""
        endpoints = self._endpoints(app, "http://127.0.0.1:8091")
        assert endpoints, "no endpoints advertised at all"
        assert all(":8091" in e["url"] for e in endpoints), endpoints

    def test_a_different_port_is_reflected(self, app):
        assert all(
            ":9999" in e["url"] for e in self._endpoints(app, "http://127.0.0.1:9999")
        )

    def test_it_no_longer_hardcodes_8000(self, app):
        """The bug: a server on any other port advertised 8000, the phone
        connected there, and both sides looked fine."""
        assert not any(
            ":8000" in e["url"] for e in self._endpoints(app, "http://127.0.0.1:8091")
        )

    def test_an_explicit_port_still_wins(self, tmp_path, monkeypatch):
        """A proxy forwarding to another port knows something the request
        cannot.

        Set before the app is built, which is the only ordering that
        happens in practice — and the config now records whether the port
        was configured at the moment it reads it, so the two cannot
        disagree.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            "hypernix.security.keymaster._DEFAULT_STORE", tmp_path / "keymaster"
        )
        monkeypatch.setattr(
            "hypernix.security.gatekeeper._DEFAULT_DATA", tmp_path / "gatekeeper"
        )
        monkeypatch.setenv("T1_TOKEN_SECRET", "x" * 64)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1api.sqlite3"))
        monkeypatch.setenv("T1_HYPERLINK_ENABLED", "1")
        monkeypatch.setenv("T1_HYPERLINK_PORT", "443")
        from hypernix.t1api.app import create_app

        assert all(
            ":443" in e["url"]
            for e in self._endpoints(create_app(), "http://127.0.0.1:8091")
        )

    def test_a_default_port_url_is_handled(self, app):
        """No port in the Host header means the scheme's default, not None."""
        endpoints = self._endpoints(app, "http://127.0.0.1")
        assert endpoints and all(":80" in e["url"] for e in endpoints)


@needs_server
class TestPairingWorksWithATwoKey:
    """`waiter hyperlink pair` was believed to need a T1 key."""

    def test_a_t2_admin_key_can_mint_a_pairing_code(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            "hypernix.security.keymaster._DEFAULT_STORE", tmp_path / "keymaster"
        )
        monkeypatch.setattr(
            "hypernix.security.gatekeeper._DEFAULT_DATA", tmp_path / "gatekeeper"
        )
        monkeypatch.setenv("T1_TOKEN_SECRET", "x" * 64)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1api.sqlite3"))
        monkeypatch.setenv("T1_HYPERLINK_ENABLED", "1")
        from fastapi.testclient import TestClient

        from hypernix.t1api.app import create_app

        app = create_app()
        key = app.state.t1_bootstrap_key.key
        assert key.startswith("T2_")
        with TestClient(app, client=("127.0.0.1", 5000)) as client:
            response = client.post(
                "/hyperlink/pair", json={"label": "my iPhone"},
                headers={"Authorization": "Bearer " + key},
            )
        assert response.status_code == 200, response.text
        assert response.json()["code"]
