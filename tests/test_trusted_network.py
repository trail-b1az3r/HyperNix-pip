"""Keyless access from the LAN and the tailnet, and its edges.

0.72.4 item 3: an origin on the LAN or the tailnet may connect without
a key *when the administrator has turned that on*. Item 14 sets the
boundary it must never cross — a public unauthenticated connection can
never escalate into anything.

Most of this file is the boundary rather than the feature, because the
feature is one config flag and the boundary is where the damage lives.

The three that would each be a hole
------------------------------------
**Off by default.** A private address is not consent. Nothing is keyless
until someone says so.

**The reverse-proxy trap.** nginx or caddy on the same host makes every
request in the world arrive from 127.0.0.1 — the *most* trusted level.
An operator who enables keyless LAN access behind an unconfigured proxy
would be publishing it to the internet while believing it was reachable
only from their sofa. An unverifiable forwarded header collapses the
origin to public.

**A bad key is not "no key".** The keyless path runs only when the
request brings no credential at all. If a failed key fell through to it,
revoking a key would stop working from the LAN, which is the opposite of
what revoking means.
"""
from __future__ import annotations

import logging

import pytest

# The exact spelling the lint-regression check looks for: adding a
# reason= kwarg breaks its substring match, and the guard then
# reads as absent even though it is right there.
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.keymaster import KeyScope  # noqa: E402

#: An authenticated route with no side effects.
GUARDED = "/usage/current"


@pytest.fixture(autouse=True)
def _quiet_and_clean(monkeypatch):
    logging.disable(logging.CRITICAL)
    for name in (
        "T1_TRUSTED_NETWORK",
        "T1_TRUSTED_NETWORK_LAN",
        "T1_TRUSTED_NETWORK_TAILNET",
        "T1_TRUSTED_NETWORK_PARTIAL_ADMIN",
        "T1_TRUSTED_PROXIES",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    logging.disable(logging.NOTSET)


def client(peer: str, **env) -> TestClient:
    import os

    from hypernix.t1api.app import create_app

    for key, value in env.items():
        os.environ[key] = value
    return TestClient(create_app(), client=(peer, 54321))


class TestOffByDefault:
    """A private address is not consent."""

    @pytest.mark.parametrize("peer", ["127.0.0.1", "192.168.1.5", "10.0.0.9", "8.8.8.8"])
    def test_every_origin_needs_a_key(self, peer):
        assert client(peer).get(GUARDED).status_code == 401

    def test_the_config_default_is_false(self, monkeypatch):
        from hypernix.t1api.config import T1APIConfig

        monkeypatch.delenv("T1_TRUSTED_NETWORK", raising=False)

        assert T1APIConfig().trusted_network is False


class TestWhenItIsOn:
    def test_loopback_and_lan_are_keyless(self):
        for peer in ("127.0.0.1", "192.168.1.5"):
            response = client(peer, T1_TRUSTED_NETWORK="1").get(GUARDED)

            assert response.status_code == 200, peer

    def test_a_public_origin_is_still_refused(self):
        """The line item 14 draws, and the one that matters most."""
        response = client("8.8.8.8", T1_TRUSTED_NETWORK="1").get(GUARDED)

        assert response.status_code == 401

    def test_lan_can_be_excluded(self):
        response = client(
            "192.168.1.5", T1_TRUSTED_NETWORK="1", T1_TRUSTED_NETWORK_LAN="0"
        ).get(GUARDED)

        assert response.status_code == 401

    def test_loopback_survives_excluding_both(self):
        response = client(
            "127.0.0.1", T1_TRUSTED_NETWORK="1",
            T1_TRUSTED_NETWORK_LAN="0", T1_TRUSTED_NETWORK_TAILNET="0",
        ).get(GUARDED)

        assert response.status_code == 200


class TestTheReverseProxyTrap:
    """The failure that would publish a LAN-only server to the internet."""

    def test_an_unverifiable_forwarded_header_is_not_trusted(self):
        response = client("127.0.0.1", T1_TRUSTED_NETWORK="1").get(
            GUARDED, headers={"x-forwarded-for": "203.0.113.9"}
        )

        assert response.status_code == 401

    def test_a_lan_peer_sending_a_header_is_also_refused(self):
        """Fails closed rather than guessing. A direct client that sends
        a junk header only denies itself."""
        response = client("192.168.1.5", T1_TRUSTED_NETWORK="1").get(
            GUARDED, headers={"x-forwarded-for": "8.8.8.8"}
        )

        assert response.status_code == 401

    def test_a_configured_proxy_relaying_a_lan_client_works(self):
        response = client(
            "127.0.0.1", T1_TRUSTED_NETWORK="1", T1_TRUSTED_PROXIES="127.0.0.1/32",
        ).get(GUARDED, headers={"x-forwarded-for": "192.168.1.5"})

        assert response.status_code == 200

    def test_a_configured_proxy_relaying_a_public_client_does_not(self):
        response = client(
            "127.0.0.1", T1_TRUSTED_NETWORK="1", T1_TRUSTED_PROXIES="127.0.0.1/32",
        ).get(GUARDED, headers={"x-forwarded-for": "203.0.113.9"})

        assert response.status_code == 401


class TestACredentialIsAlwaysJudgedOnItsOwnMerits:
    def test_a_bad_key_does_not_fall_through_to_keyless(self):
        """Otherwise revoking a key would stop working from the LAN."""
        response = client("192.168.1.5", T1_TRUSTED_NETWORK="1").get(
            GUARDED, headers={"authorization": "Bearer T1_definitely_not_a_key"}
        )

        assert response.status_code == 401

    def test_the_same_request_without_the_header_succeeds(self):
        """So the test above is about the key, not the origin."""
        response = client("192.168.1.5", T1_TRUSTED_NETWORK="1").get(GUARDED)

        assert response.status_code == 200


class TestWhatAKeylessCallerMayDo:
    def _context(self, peer: str, **env):
        import os

        from hypernix.t1api import deps
        from hypernix.t1api.app import create_app

        for key, value in env.items():
            os.environ[key] = value
        app = create_app()

        class Req:
            client = type("C", (), {"host": peer})()
            headers: dict = {}
            state = type("S", (), {})()

        Req.app = app
        return deps.trusted_network_context(Req())

    def test_read_only_by_default(self):
        ctx = self._context("192.168.1.5", T1_TRUSTED_NETWORK="1")

        assert ctx.scopes == {KeyScope.READ}

    def test_partial_admin_adds_write_but_never_admin(self):
        """"Partial administrative functionality" — the partial part is
        load-bearing. A caller that presented no credential must not
        reach what an admin key exists to gate."""
        ctx = self._context(
            "192.168.1.5", T1_TRUSTED_NETWORK="1",
            T1_TRUSTED_NETWORK_PARTIAL_ADMIN="1",
        )

        assert KeyScope.WRITE in ctx.scopes
        assert KeyScope.ADMIN not in ctx.scopes
        assert not ctx.is_admin

    def test_it_carries_no_key_material(self):
        """There was no credential. Recording a plausible-looking one
        would invent evidence of an authentication that never happened."""
        ctx = self._context("192.168.1.5", T1_TRUSTED_NETWORK="1")

        assert ctx.key_meta.key == ""

    def test_it_records_where_the_trust_came_from(self):
        ctx = self._context("192.168.1.5", T1_TRUSTED_NETWORK="1")

        assert ctx.key_meta.tags["trust"] == "lan"
        assert ctx.key_meta.tags["origin"] == "192.168.1.5"

    def test_a_public_origin_gets_no_context_at_all(self):
        assert self._context("8.8.8.8", T1_TRUSTED_NETWORK="1") is None

    def test_nothing_is_granted_when_the_mode_is_off(self):
        assert self._context("192.168.1.5") is None


class TestTheInstallerAsksBeforeLoweringAuthentication:
    """`--yes` answers confirmations. This is not a confirmation.

    `ask_yes_no` returns yes for everything under `--yes`, which is
    right for "are you sure" and wrong for the one question that lowers
    an authentication requirement: an unattended install would come up
    serving the LAN without a key and nobody would have chosen it.
    """

    import subprocess as _subprocess
    from pathlib import Path as _Path

    SCRIPT = _Path(__file__).resolve().parent.parent / "install-t1.sh"

    def _run(self, tmp_path, *flags) -> str:
        import os
        import shutil
        import subprocess

        bash = shutil.which("bash")
        if bash is None:  # pragma: no cover
            pytest.skip("no bash")
        result = subprocess.run(
            [bash, str(self.SCRIPT), "--install", "skip", "--dry-run",
             "--config-dir", str(tmp_path / "cfg"), *flags],
            capture_output=True, text=True, encoding="utf-8", timeout=300,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "NO_COLOR": "1", "HOME": str(tmp_path / "home")},
        )
        return result.stdout + result.stderr

    def test_yes_does_not_enable_keyless_access(self, tmp_path):
        output = self._run(tmp_path, "--yes")

        assert "does not enable keyless access" in output
        assert "Keyless access is on" not in output

    def test_the_flag_enables_it(self, tmp_path):
        output = self._run(tmp_path, "--trusted-network", "--non-interactive")

        assert "Enabled by --trusted-network" in output
        assert "Keyless access is on" in output

    def test_enabling_it_warns_plainly(self, tmp_path):
        """Item 3 asks for the administrator to be told what it costs."""
        output = self._run(tmp_path, "--trusted-network", "--non-interactive")

        assert "lowers the authentication requirement" in output
        assert "guest on your wifi" in output

    def test_non_interactive_alone_leaves_it_off(self, tmp_path):
        output = self._run(tmp_path, "--non-interactive")

        assert "Keyless access is on" not in output

    def test_the_flag_survives_the_defaults(self, tmp_path):
        """The defaults are assigned after argument parsing, so a plain
        assignment there would silently undo the flag."""
        output = self._run(tmp_path, "--trusted-network", "--non-interactive")

        assert "Enabled by --trusted-network" in output

    def test_the_env_block_documents_the_proxy_requirement(self):
        source = self.SCRIPT.read_text(encoding="utf-8")

        assert "T1_TRUSTED_NETWORK=" in source
        assert "T1_TRUSTED_PROXIES" in source
