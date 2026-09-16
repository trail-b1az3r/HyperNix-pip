"""HyperLink over a trusted network, which did not work at all.

The report was "you can not auth using Tailscale in hyperlink, it says
it needs an authorized t1 key". There turned out to be three separate
reasons for that, and the first one means the feature had never worked
on any trusted network, tailnet or otherwise.

1. **Every /hyperlink route refused a keyless request.**
   ``get_hyperlink_principal`` called ``_extract_credential`` before it
   looked at anything else, and that function *raises* on a missing
   header. So the keyless branch below it could not be reached — the
   exact mistake ``get_auth_context`` carries a comment warning about,
   repeated in the function next to it. ``/usage/current`` worked
   keyless from the LAN while ``/hyperlink/sessions`` returned 401.

2. **A Tailscale IPv6 address was never a tailnet address.**
   ``TAILNET_RANGE`` is ``100.64.0.0/10`` and the check was guarded on
   ``address.version == 4``. Every Tailscale node also has an address in
   ``fd7a:115c:a1e0::/48``, and a phone resolving a MagicDNS name gets
   both and generally prefers the v6 one — so the connection arrived,
   was classified public, and was refused.

3. **``verify_tailnet`` was unreachable.** The parameter existed and no
   caller ever set it, so when ``tailscale whois`` could not answer —
   not installed for the service user, no permission on the socket, a
   peer it has not seen traffic from — every tailnet peer silently
   became public with nothing the operator could do about it.

What is *not* relaxed
---------------------
Off by default, still. A bad key still never falls through to the
keyless path. An unverifiable forwarded header still collapses the
origin to public. And a keyless caller still never gets ADMIN.
"""
from __future__ import annotations

import ipaddress
import logging

import pytest

# The exact spelling the lint-regression check looks for.
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

TAILNET_V4 = "100.109.195.71"
TAILNET_V6 = "fd7a:115c:a1e0::1234"
LAN = "192.168.1.50"

#: Reachable with a device token, a T1 key, or keyless on a trusted
#: network. The one the phone calls first.
HYPERLINK = "/hyperlink/endpoints"
#: An ordinary authenticated route, for comparison. This one already
#: worked, which is what made the difference diagnostic.
ORDINARY = "/usage/current"

_ENV = (
    "T1_TRUSTED_NETWORK",
    "T1_TRUSTED_NETWORK_LAN",
    "T1_TRUSTED_NETWORK_TAILNET",
    "T1_TRUSTED_NETWORK_TAILNET_VERIFY",
    "T1_TRUSTED_NETWORK_PARTIAL_ADMIN",
    "T1_TRUSTED_PROXIES",
)


@pytest.fixture(autouse=True)
def _quiet_and_clean(monkeypatch):
    logging.disable(logging.CRITICAL)
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    yield
    logging.disable(logging.NOTSET)




def client(peer: str, **env) -> TestClient:
    import os

    from hypernix.t1api.app import create_app


    for key, value in env.items():
        os.environ[key] = value
    return TestClient(create_app(), client=(peer, 54321))


class TestAKeylessRequestReachesHyperLink:
    """Bug 1. Nothing to do with Tailscale — it was every trusted
    network, which is why a LAN peer is the clearest way to show it."""

    def test_the_lan_can_read_hyperlink_keyless(self):
        assert client(LAN, T1_TRUSTED_NETWORK="1").get(HYPERLINK).status_code == 200

    def test_it_matches_what_an_ordinary_route_already_did(self):
        """The two disagreeing is the bug. They have to agree in both
        directions, or one of them is wrong."""
        both = client(LAN, T1_TRUSTED_NETWORK="1")
        assert both.get(ORDINARY).status_code == both.get(HYPERLINK).status_code

    def test_a_session_list_is_readable_too(self):
        """Not just the discovery route: the phone is useless if it can
        see the server and none of its conversations."""
        assert (
            client(LAN, T1_TRUSTED_NETWORK="1").get("/hyperlink/sessions").status_code
            == 200
        )

    def test_it_is_still_off_by_default(self):
        assert client(LAN).get(HYPERLINK).status_code == 401

    def test_a_public_origin_is_still_refused(self):
        assert client("8.8.8.8", T1_TRUSTED_NETWORK="1").get(HYPERLINK).status_code == 401

    def test_a_bad_key_does_not_fall_through_to_keyless(self):
        """The one that would make revoking a key stop meaning anything.
        A presented credential is judged on its own merits; the keyless
        path is for requests that bring nothing."""
        response = client(LAN, T1_TRUSTED_NETWORK="1").get(
            HYPERLINK, headers={"Authorization": "Bearer T1_definitely-not-a-key"}
        )
        assert response.status_code == 401

    def test_a_keyless_caller_is_never_an_admin(self):
        """Pairing a device is an operator action. A connection that
        presented no credential must not be able to enrol another one."""
        response = client(
            LAN, T1_TRUSTED_NETWORK="1", T1_TRUSTED_NETWORK_PARTIAL_ADMIN="1"
        ).post("/hyperlink/pair", json={"name": "a phone"})
        assert response.status_code in (401, 403)


class TestTailscaleIPv6:
    """Bug 2. Tailscale gives every node a v4 and a v6 address, and a
    phone resolving a MagicDNS name will often take the v6."""

    def test_the_v6_range_is_known(self):
        from hypernix.system.nettrust import TAILNET_RANGES

        network = ipaddress.ip_network("fd7a:115c:a1e0::/48")
        assert any(net == network for net in TAILNET_RANGES), TAILNET_RANGES

    def test_the_v4_range_is_still_known(self):
        from hypernix.system.nettrust import TAILNET_RANGES

        network = ipaddress.ip_network("100.64.0.0/10")
        assert any(net == network for net in TAILNET_RANGES)

    @pytest.mark.parametrize("peer", [TAILNET_V4, TAILNET_V6])
    def test_both_families_classify_as_tailnet(self, peer):
        from hypernix.system.nettrust import Trust, classify

        origin = classify(peer, verify_tailnet=False)
        assert origin.trust is Trust.TAILNET, origin

    @pytest.mark.parametrize("peer", [TAILNET_V4, TAILNET_V6])
    def test_both_families_can_connect(self, peer):
        response = client(
            peer,
            T1_TRUSTED_NETWORK="1",
            # tailscaled is not running here, which is the case bug 3 is
            # about. Verification off isolates this test to the range.
            T1_TRUSTED_NETWORK_TAILNET_VERIFY="0",
        ).get(HYPERLINK)
        assert response.status_code == 200

    def test_an_ordinary_ula_is_not_a_tailnet_address(self):
        """`fd00::/8` is the whole private-ULA space and anyone can number
        themselves in it. Only Tailscale's own /48 counts."""
        from hypernix.system.nettrust import Trust, classify

        origin = classify("fd00:1234:5678::1", verify_tailnet=False)
        assert origin.trust is not Trust.TAILNET


class TestVerificationCanBeTurnedOff:
    """Bug 3. `tailscale whois` failing is not rare, and when it fails
    there was nothing the operator could say about it."""

    def test_verification_is_on_by_default(self):
        from hypernix.t1api.config import T1APIConfig

        assert T1APIConfig().trusted_network_tailnet_verify is True

    def test_an_unverifiable_peer_is_public_by_default(self):
        """tailscaled is not running in this test environment, so this is
        the real behaviour rather than a mocked one: in the range, not
        confirmed, therefore not trusted."""
        from hypernix.system.nettrust import Trust, classify

        origin = classify(TAILNET_V4, verify_tailnet=True)
        assert origin.trust is Trust.PUBLIC
        assert "does not recognise" in origin.reason

    def test_turning_it_off_trusts_the_range(self):
        from hypernix.system.nettrust import Trust, classify

        origin = classify(TAILNET_V4, verify_tailnet=False)
        assert origin.trust is Trust.TAILNET
        assert "not checked" in origin.reason

    def test_the_default_still_refuses_an_unverified_peer_end_to_end(self):
        """With verification left on and no tailscaled, a 100.x peer is
        exactly as untrusted as any other stranger."""
        assert client(TAILNET_V4, T1_TRUSTED_NETWORK="1").get(
            HYPERLINK
        ).status_code == 401

    def test_the_lan_is_unaffected_by_the_tailnet_switch(self):
        """Turning off tailnet verification must not quietly widen
        anything else."""
        assert client(
            "8.8.8.8", T1_TRUSTED_NETWORK="1", T1_TRUSTED_NETWORK_TAILNET_VERIFY="0"
        ).get(HYPERLINK).status_code == 401
