"""Where a request came from, and how much that is worth.

0.72.4 lets a connection from the LAN or the tailnet act without a key.
That is only safe if "from the LAN" is a *fact about the connection*
rather than a claim the connection makes, so most of this file is about
the ways a public client could try to be mistaken for a local one.

Two of those are real and both are covered:

**The forwarded header.** ``X-Forwarded-For`` is set by whoever is
talking to you. A server that believes it has turned keyless LAN access
into keyless access for anyone who can spell a header. It is read only
when the immediate peer is a proxy the administrator listed.

**The tailnet range.** 100.64.0.0/10 is shared address space, not
RFC 1918. Anything on a LAN can number itself 100.x and route to the
server, so the range alone proves nothing; a tailnet origin has to be
confirmed by tailscaled. Unconfirmed means public.

And one bug of this module's own, found while writing these:
``ipaddress.is_private`` is far broader than RFC 1918 — true for the
documentation ranges, 0.0.0.0/8, benchmarking and reserved space — so
using it as "the LAN" would have granted keyless access to addresses
that are on nobody's network. The ranges are spelled out instead.
"""
from __future__ import annotations

import ipaddress

import pytest

from hypernix.system.nettrust import (
    TAILNET_RANGE,
    Origin,
    Trust,
    TrustPolicy,
    classify,
    trusted_proxies_from_env,
)


class TestTheObviousCases:
    @pytest.mark.parametrize(
        ("peer", "expected"),
        [
            ("127.0.0.1", Trust.LOOPBACK),
            ("::1", Trust.LOOPBACK),
            ("192.168.1.95", Trust.LAN),
            ("10.0.0.4", Trust.LAN),
            ("172.20.1.1", Trust.LAN),
            ("169.254.1.1", Trust.LAN),
            ("fd00::1", Trust.LAN),
            ("8.8.8.8", Trust.PUBLIC),
            ("2606:4700::1111", Trust.PUBLIC),
        ],
    )
    def test_classification(self, peer, expected):
        assert classify(peer, verify_tailnet=False).trust is expected

    def test_an_unparseable_peer_is_public(self):
        """Failing closed. An address that cannot be understood is not
        an address that can be trusted."""
        origin = classify("not-an-address")

        assert origin.trust is Trust.PUBLIC

    def test_a_host_port_peer_is_understood(self):
        assert classify("192.168.1.95:54321", verify_tailnet=False).trust is Trust.LAN

    def test_a_scoped_ipv6_peer_is_understood(self):
        assert classify("fe80::1%eth0", verify_tailnet=False).trust is Trust.LAN


class TestIsPrivateIsNotTheLAN:
    """The bug this module nearly shipped.

    `ipaddress.is_private` is true for a great deal that is not a local
    network. Every address here would have been LAN — and therefore
    eligible for keyless access — under the obvious implementation.
    """

    @pytest.mark.parametrize(
        "peer",
        [
            "203.0.113.9",    # TEST-NET-3
            "192.0.2.5",      # TEST-NET-1
            "198.51.100.7",   # TEST-NET-2
            "0.1.2.3",        # "this network"
            "240.0.0.1",      # reserved
            "192.0.0.170",    # IETF protocol assignments
        ],
    )
    def test_these_are_public(self, peer):
        assert ipaddress.ip_address(peer).is_private, (
            f"{peer} is not is_private, so this test proves nothing"
        )

        assert classify(peer, verify_tailnet=False).trust is Trust.PUBLIC


class TestTheForwardedHeader:
    LOCAL_CLAIM = "192.168.1.50"

    def test_it_is_ignored_without_a_configured_proxy(self):
        """The attack: a public client claiming to be on the LAN."""
        origin = classify("8.8.8.8", forwarded_for=self.LOCAL_CLAIM)

        assert origin.trust is Trust.PUBLIC
        assert origin.address == "8.8.8.8"
        assert origin.via_proxy == ""

    def test_it_is_believed_from_a_configured_proxy(self):
        proxy = [ipaddress.ip_network("203.0.113.9/32")]

        origin = classify(
            "203.0.113.9", forwarded_for=self.LOCAL_CLAIM, trusted_proxies=proxy,
            verify_tailnet=False,
        )

        assert origin.trust is Trust.LAN
        assert origin.via_proxy == "203.0.113.9"

    def test_only_the_hop_the_proxy_added_is_read(self):
        """Everything left of the last entry came from the client."""
        proxy = [ipaddress.ip_network("203.0.113.9/32")]

        origin = classify(
            "203.0.113.9",
            forwarded_for="10.0.0.1, 8.8.8.8",
            trusted_proxies=proxy, verify_tailnet=False,
        )

        assert origin.address == "8.8.8.8"
        assert origin.trust is Trust.PUBLIC

    def test_a_proxy_that_is_not_the_peer_does_not_count(self):
        proxy = [ipaddress.ip_network("198.51.100.1/32")]

        origin = classify("8.8.8.8", forwarded_for=self.LOCAL_CLAIM, trusted_proxies=proxy)

        assert origin.trust is Trust.PUBLIC

    def test_the_default_is_no_trusted_proxies(self, monkeypatch):
        monkeypatch.delenv("T1_TRUSTED_PROXIES", raising=False)

        assert trusted_proxies_from_env() == []

    def test_junk_in_the_proxy_list_is_dropped_not_fatal(self):
        assert trusted_proxies_from_env("10.0.0.0/8, nonsense, ") == [
            ipaddress.ip_network("10.0.0.0/8")
        ]


class TestTheTailnetRangeIsNotProof:
    def test_the_range_is_the_documented_one(self):
        assert TAILNET_RANGE == ipaddress.ip_network("100.64.0.0/10")

    def test_an_unconfirmed_tailnet_address_is_public(self, monkeypatch):
        """Anything on a LAN can number itself 100.x. Without tailscaled
        confirming it, the range alone means nothing."""
        monkeypatch.setattr(
            "hypernix.system.nettrust.tailnet_identity", lambda *a, **k: ("", "")
        )

        origin = classify("100.109.195.71")

        assert origin.trust is Trust.PUBLIC
        assert "does not recognise" in origin.reason

    def test_a_confirmed_one_is_tailnet(self, monkeypatch):
        monkeypatch.setattr(
            "hypernix.system.nettrust.tailnet_identity",
            lambda *a, **k: ("desktop", "someone@example.com"),
        )

        origin = classify("100.109.195.71")

        assert origin.trust is Trust.TAILNET
        assert origin.tailnet_node == "desktop"
        assert origin.tailnet_user == "someone@example.com"

    def test_tailnet_space_is_not_reached_by_the_lan_branch(self):
        """100.64/10 is is_private, so the ordering matters: it must be
        judged as a tailnet candidate, never fall through to LAN."""
        origin = classify("100.109.195.71", verify_tailnet=False)

        assert origin.trust is Trust.TAILNET


class TestThePolicy:
    LOCAL = Origin("192.168.1.95", Trust.LAN)
    TAILNET = Origin("100.64.0.1", Trust.TAILNET)
    PUBLIC = Origin("8.8.8.8", Trust.PUBLIC)

    def test_nothing_is_keyless_by_default(self):
        policy = TrustPolicy.off()

        assert not policy.allows_keyless(self.LOCAL)
        assert not policy.allows_keyless(self.TAILNET)
        assert not policy.enabled

    def test_enabling_it_covers_lan_and_tailnet(self):
        policy = TrustPolicy.from_config(trusted_network=True)

        assert policy.allows_keyless(self.LOCAL)
        assert policy.allows_keyless(self.TAILNET)

    def test_public_is_never_keyless_however_it_is_configured(self):
        """The one thing that must not be expressible."""
        policy = TrustPolicy.from_config(trusted_network=True)

        assert not policy.allows_keyless(self.PUBLIC)

    def test_public_is_not_keyless_even_in_a_hand_built_policy(self):
        """Constructed directly, bypassing from_config. The refusal has
        to live in the check, not only in the constructor."""
        policy = TrustPolicy(keyless=frozenset(Trust), partial_admin=frozenset(Trust))

        assert not policy.allows_keyless(self.PUBLIC)
        assert not policy.allows_partial_admin(self.PUBLIC)

    def test_partial_admin_is_separately_opted_into(self):
        keyless_only = TrustPolicy.from_config(trusted_network=True)

        assert keyless_only.allows_keyless(self.LOCAL)
        assert not keyless_only.allows_partial_admin(self.LOCAL)

    def test_lan_and_tailnet_can_be_enabled_separately(self):
        tailnet_only = TrustPolicy.from_config(
            trusted_network=True, include_lan=False
        )

        assert tailnet_only.allows_keyless(self.TAILNET)
        assert not tailnet_only.allows_keyless(self.LOCAL)

    def test_loopback_is_always_included_when_enabled(self):
        policy = TrustPolicy.from_config(
            trusted_network=True, include_lan=False, include_tailnet=False
        )

        assert policy.allows_keyless(Origin("127.0.0.1", Trust.LOOPBACK))


class TestTheOriginItself:
    def test_trust_ranks_most_trusted_first(self):
        assert Trust.LOOPBACK.rank < Trust.TAILNET.rank < Trust.LAN.rank < Trust.PUBLIC.rank

    def test_is_trusted_network_excludes_public_only(self):
        assert Origin("10.0.0.1", Trust.LAN).is_trusted_network
        assert Origin("100.64.0.1", Trust.TAILNET).is_trusted_network
        assert Origin("127.0.0.1", Trust.LOOPBACK).is_trusted_network
        assert not Origin("8.8.8.8", Trust.PUBLIC).is_trusted_network

    def test_it_serialises_for_logs_and_whoami(self):
        payload = classify("192.168.1.95", verify_tailnet=False).to_dict()

        assert payload["trust"] == "lan"
        assert payload["reason"]
