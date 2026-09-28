"""T2P keys, and a server's right to refuse them.

A T2P key is an ordinary T2 key with a billing binding attached, so a key
can be issued to someone who pays for their own usage. Two things are
deliberately absent from the credential: any card data, and the binding
itself. A key gets pasted into terminals and config files and lands in
shell history — a payment token that lives there is one that leaks.

And a server does not have to take one. Somebody else's payment
arrangement is somebody else's business relationship.
"""
from __future__ import annotations

import pytest

from hypernix.security.keymaster import KeyScope, KeyType
from hypernix.security.t2keys import T2KeyGenerator, T2Type, looks_like_t2
from hypernix.t1api.billingkeys import (
    BillingBindingStore,
    BillingKeyPolicy,
    PaymentRequired,
)
from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.errors import T1APIError

try:
    import fastapi  # noqa: F401

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

needs_server = pytest.mark.skipif(not _HAS_FASTAPI, reason="needs the [t1api] extra")


@pytest.fixture
def store(tmp_path):
    return BillingBindingStore(SQLiteBackend(str(tmp_path / "b.sqlite3")))


class TestTheFamily:
    def test_a_t2p_key_parses_and_round_trips(self):
        """A T2P key is a spelling of a T1 key, like every other T2."""
        from hypernix.security.keymaster import T1KeyGenerator

        t1 = T1KeyGenerator.generate()
        key = T2KeyGenerator.from_t1(t1, access_level=3, family=T2Type.T2P)
        assert key.family is T2Type.T2P
        assert T2KeyGenerator.to_t1(key) == t1

    def test_it_can_never_be_an_admin(self):
        """Spending money and reconfiguring a server are separate
        authorities with no reason to travel on one credential."""
        with pytest.raises(ValueError, match="cannot be an admin"):
            T2KeyGenerator.generate(family=T2Type.T2P, admin=True)

    def test_the_other_families_still_parse(self):
        """T2P was inserted into an ordered alternation.

        `T2S|T2P|T2C|T2` — put T2 first and "T2P_…" matches family "T2"
        with a stray P on the body.
        """
        for family in (T2Type.T2, T2Type.T2S, T2Type.T2P):
            raw = T2KeyGenerator.generate(family=family).raw
            assert T2KeyGenerator.parse(raw).family is family

    @pytest.mark.parametrize(
        "key,expected",
        [("T2_a", True), ("T2S_a", True), ("T2P_a", True), ("T2C_a", True),
         ("T1_a", False), ("T2X_a", False), ("", False), ("nope", False)],
    )
    def test_family_detection_is_derived_not_listed(self, key, expected):
        """Two call sites tested `key[:3] in ("T2_", "T2S")`.

        That silently stopped covering the family when T2P arrived: the
        key fell through to the T1 path and was rejected as malformed,
        several layers before the code that had an opinion about it.
        """
        assert looks_like_t2(key) is expected


class TestTheBinding:
    def test_a_binding_holds_provider_references(self, store):
        binding = store.bind(
            "key-1", provider="stripe", customer_ref="cus_ABC", method_ref="pm_XYZ"
        )
        assert binding.provider == "stripe"
        assert binding.key_id == "key-1"

    def test_the_api_view_withholds_the_provider_references(self, store):
        """Not secrets like a card number, but they identify a payment
        method at the provider and no client needs them."""
        public = store.bind(
            "key-1", provider="stripe", customer_ref="cus_ABC", method_ref="pm_XYZ"
        ).public_dict()
        assert "customer_ref" not in public
        assert "method_ref" not in public
        assert public["provider"] == "stripe"

    @pytest.mark.parametrize(
        "card", ["4242424242424242", "4242 4242 4242 4242", "4242-4242-4242-4242"]
    )
    def test_a_card_number_is_refused_at_the_boundary(self, store, card):
        """The check cannot prove a string is safe, but a bare 13-19 digit
        run is the one shape that must never be stored here — and catching
        it at the boundary beats finding it in a database dump."""
        with pytest.raises(T1APIError, match="card number"):
            store.bind("k", provider="stripe", customer_ref=card, method_ref="pm_1")

    def test_a_provider_reference_containing_digits_is_fine(self, store):
        assert store.bind(
            "k", provider="stripe", customer_ref="cus_9999999999999", method_ref="pm_1"
        )

    def test_a_zero_cap_is_refused(self, store):
        """Zero would refuse every request and read as a bug. No cap means
        unlimited; that is what to use."""
        with pytest.raises(T1APIError, match="positive"):
            store.bind("k", provider="s", customer_ref="c", method_ref="m", spend_cap=0)

    def test_one_binding_per_key(self, store):
        """Two bindings on one key is an unanswerable question at charge
        time. Re-binding replaces."""
        store.bind("k", provider="stripe", customer_ref="c1", method_ref="m1")
        store.bind("k", provider="adyen", customer_ref="c2", method_ref="m2")
        assert len(store.list_bindings()) == 1
        assert store.get("k").provider == "adyen"


class TestTheSpendCap:
    def test_spending_accumulates(self, store):
        store.bind("k", provider="s", customer_ref="c", method_ref="m", spend_cap=10.0)
        store.record_spend("k", 4.0)
        assert store.get("k").remaining == 6.0

    def test_it_refuses_before_the_work(self, store):
        """An over-cap request found at charge time has already cost the
        operator the inference. The estimate is what is checked."""
        store.bind("k", provider="s", customer_ref="c", method_ref="m", spend_cap=10.0)
        store.record_spend("k", 4.0)
        store.assert_within_cap("k", estimated=2.0)  # fine
        with pytest.raises(PaymentRequired) as excinfo:
            store.assert_within_cap("k", estimated=9.0)
        assert excinfo.value.details["reason"] == "spend_cap_reached"
        assert excinfo.value.http_status == 402

    def test_no_cap_means_no_limit(self, store):
        store.bind("k", provider="s", customer_ref="c", method_ref="m")
        store.assert_within_cap("k", estimated=1_000_000.0)

    def test_a_key_with_no_binding_is_not_an_error_here(self, store):
        """The quota cascade has its own answer for that."""
        store.assert_within_cap("no-such-key", estimated=5.0)

    def test_releasing_a_binding_removes_it(self, store):
        """A binding that outlives its key is a standing authorisation to
        charge someone for a credential that no longer exists."""
        store.bind("k", provider="s", customer_ref="c", method_ref="m")
        assert store.release("k")
        assert store.get("k") is None


@needs_server
class TestTheServerPolicy:
    """Enforced at authentication. Refusing after the work is a refund."""

    @pytest.fixture
    def make(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            "hypernix.security.keymaster._DEFAULT_STORE", tmp_path / "keymaster"
        )
        monkeypatch.setattr(
            "hypernix.security.gatekeeper._DEFAULT_DATA", tmp_path / "gatekeeper"
        )
        monkeypatch.setenv("T1_TOKEN_SECRET", "x" * 64)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1api.sqlite3"))

        def build(policy: str, payment_url: str = ""):
            monkeypatch.setenv("T1_BILLING_KEY_POLICY", policy)
            monkeypatch.setenv("T1_PAYMENT_URL", payment_url)
            from hypernix.t1api.app import create_app

            app = create_app()
            meta = app.state.t1_keymaster.create(
                key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}
            )
            return (
                app,
                T2KeyGenerator.from_t1(meta.key, access_level=3, family=T2Type.T2P).raw,
                T2KeyGenerator.from_t1(meta.key, access_level=3, family=T2Type.T2).raw,
            )

        return build

    @staticmethod
    def _get(app, key):
        from fastapi.testclient import TestClient

        with TestClient(app, client=("127.0.0.1", 5000)) as client:
            return client.get("/keys", headers={"Authorization": "Bearer " + key})

    def test_allow_accepts_a_billing_key(self, make):
        app, t2p, _ = make(BillingKeyPolicy.ALLOW)
        assert self._get(app, t2p).status_code == 200

    def test_deny_refuses_and_says_where_to_pay(self, make):
        app, t2p, _ = make(BillingKeyPolicy.DENY, "https://pay.example.com")
        response = self._get(app, t2p)
        assert response.status_code == 402
        error = response.json()["error"]
        assert error["code"] == "BILLING_KEY_REFUSED"
        assert error["details"]["payment_url"] == "https://pay.example.com"

    def test_deny_without_a_url_says_so_rather_than_bluffing(self, make):
        """"Pay here" without saying where is a dead end, and the operator
        is the one who can fix it."""
        app, t2p, _ = make(BillingKeyPolicy.DENY)
        message = self._get(app, t2p).json()["error"]["message"]
        assert "No payment URL is configured" in message

    def test_separate_requires_the_payment_key_in_its_own_header(self, make):
        app, t2p, _ = make(BillingKeyPolicy.SEPARATE)
        error = self._get(app, t2p).json()["error"]
        assert error["details"]["policy"] == "separate"
        assert error["details"]["payment_header"] == "X-Payment-Key"

    @pytest.mark.parametrize(
        "policy",
        [BillingKeyPolicy.ALLOW, BillingKeyPolicy.DENY, BillingKeyPolicy.SEPARATE],
    )
    def test_an_ordinary_key_is_never_affected(self, make, policy):
        """The policy is about billing keys and nothing else."""
        app, _, t2 = make(policy)
        assert self._get(app, t2).status_code == 200

    def test_the_default_is_allow_so_nothing_changes_for_existing_servers(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("T1_BILLING_KEY_POLICY", raising=False)
        from hypernix.t1api.config import T1APIConfig

        assert T1APIConfig().billing_key_policy == "allow"


class TestTheBindingFollowsTheKey:
    """`release` and `transfer` were methods nobody called.

    Revocation happens in the security layer, which knows nothing about
    billing. Without the lifecycle hooks a binding outlives the key it
    belongs to — and a spend cap with a recorded spend, sitting on a dead
    key ID, is waiting to be inherited by whatever holds that ID next.
    """

    @pytest.fixture
    def km(self, tmp_path):
        from hypernix.security.keymaster import Keymaster

        return Keymaster(store_dir=tmp_path / "keys", auto_rotate=False)

    def test_revoking_a_key_releases_its_binding(self, store, km):
        key = km.create(key_type=KeyType.SERVICE, scopes={KeyScope.READ})
        store.bind(
            key.key_id,
            provider="stripe",
            customer_ref="cus_test",
            method_ref="pm_test",
            spend_cap=25.0,
        )
        store.attach_to(km)
        assert store.get(key.key_id) is not None

        km.revoke(key.key_id, reason="test")

        assert store.get(key.key_id) is None

    def test_rotating_a_key_moves_the_binding_and_its_spend(self, store, km):
        """A rotation that reset `spent` would mint spend out of a cap."""
        key = km.create(key_type=KeyType.SERVICE, scopes={KeyScope.READ})
        store.bind(
            key.key_id,
            provider="stripe",
            customer_ref="cus_test",
            method_ref="pm_test",
            spend_cap=10.0,
        )
        store.record_spend(key.key_id, 7.5)
        store.attach_to(km)

        new_meta = km.rotate(key.key_id)

        assert store.get(key.key_id) is None
        moved = store.get(new_meta.key_id)
        assert moved is not None
        assert moved.spend_cap == 10.0
        assert moved.spent == pytest.approx(7.5)

        # And the cap still bites on the new key.
        with pytest.raises(PaymentRequired):
            store.assert_within_cap(new_meta.key_id, estimated=5.0)

    def test_a_failing_observer_cannot_block_a_revocation(self, km):
        """Revoking is the safety-critical operation; bookkeeping is not."""
        def explode(_key_id, _reason=""):
            raise RuntimeError("bookkeeping is down")

        km.on_revoke(explode)
        key = km.create(key_type=KeyType.SERVICE, scopes={KeyScope.READ})

        km.revoke(key.key_id, reason="test")

        assert key.key_id not in {k.key_id for k in km.list()}

    def test_attach_to_tolerates_a_keymaster_without_the_hooks(self, store):
        """A version mismatch degrades to the old behaviour, not a crash."""
        class Older:
            pass

        store.attach_to(Older())  # must not raise

    def test_transfer_of_a_key_with_no_binding_is_a_no_op(self, store):
        assert store.transfer("nope", "also-nope") is None
