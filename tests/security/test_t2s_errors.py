"""What a T2S key holder is told when it does not work.

The two failures reported: "forbidden" and "invalid key". Both were
accurate and both sent the reader somewhere useless.

* An unregistered T2S key produced ``Unknown or unregistered T1 key.`` —
  which reads as "you brought the wrong kind of key" to someone holding a
  T2S key, when in fact the right kind simply is not in this server's
  store. Until ``gkey create -v v2short`` existed there was no supported
  way to mint one that was.
* A registered T2S key on an admin operation produced ``This HyperLink
  operation requires an admin T1 key.`` — sending the reader off to widen
  their key's scopes, which can never work: a T2S key is never an admin,
  and that is a property of the format, not of the grant.
"""
from __future__ import annotations

import contextlib
import io
import re

import pytest

from hypernix.security.gkey_cli import main as gkey
from hypernix.security.t2keys import T2KeyGenerator, T2Type

# Everything that drives a server needs the [t1api] extra; the version
# check does not. See tests/t1api/test_t1_v1_0_26_8_1_1.py for the same split.
try:  # a real import, not find_spec
    import fastapi  # noqa: F401

    _HAS_FASTAPI = True
except ImportError:
    # find_spec would answer "yes" for an installed-but-broken fastapi —
    # it locates the module without executing it. Importing is what the
    # tests themselves do, so it is what the guard should do.
    _HAS_FASTAPI = False
needs_server = pytest.mark.skipif(
    not _HAS_FASTAPI, reason="needs the [t1api] extra (fastapi)"
)

ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "hypernix.security.keymaster._DEFAULT_STORE", tmp_path / "keymaster"
    )
    monkeypatch.setattr(
        "hypernix.security.gatekeeper._DEFAULT_DATA", tmp_path / "gatekeeper"
    )
    monkeypatch.setenv("T1_HYPERLINK_ENABLED", "1")
    monkeypatch.setenv("T1_TOKEN_SECRET", "x" * 64)
    return tmp_path


def mint(*argv: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert gkey(list(argv)) == 0
    text = ANSI.sub("", buf.getvalue())
    return next(
        line.split("Key:")[1].strip().rstrip("│").strip()
        for line in text.splitlines()
        if re.search(r"\bKey:", line)
    )


@pytest.fixture
def client(store):
    from fastapi.testclient import TestClient

    from hypernix.t1api.app import create_app

    with TestClient(create_app(), client=("127.0.0.1", 5000)) as test_client:
        yield test_client


def error(response) -> dict:
    return response.json()["error"]


@needs_server
class TestUnregisteredKey:
    """"invalid key" — well-formed, but in no key store."""

    def test_it_names_the_family_the_caller_actually_presented(self, client):
        standalone = T2KeyGenerator.generate(family=T2Type.T2S).raw
        body = error(client.post("/auth/t1/validate", json={"key": standalone}))
        assert "T2S" in body["message"]
        assert "T1 key" not in body["message"], (
            "tells a T2S holder about a T1 key, which reads as the wrong key type"
        )

    def test_it_says_the_key_is_well_formed(self, client):
        """Separates "malformed" from "not registered" — different fixes."""
        standalone = T2KeyGenerator.generate(family=T2Type.T2S).raw
        body = error(client.post("/auth/t1/validate", json={"key": standalone}))
        assert "well-formed" in body["message"]
        assert body["details"]["reason"] == "not_in_key_store"

    def test_it_names_the_command_that_mints_a_registered_one(self, client):
        standalone = T2KeyGenerator.generate(family=T2Type.T2S).raw
        body = error(client.post("/auth/t1/validate", json={"key": standalone}))
        assert "gkey create -v v2short" in body["details"]["remedy"]

    def test_a_t2_key_is_told_about_v2_not_v2short(self, client):
        standalone = T2KeyGenerator.generate(family=T2Type.T2).raw
        body = error(client.post("/auth/t1/validate", json={"key": standalone}))
        assert "gkey create -v v2" in body["details"]["remedy"]
        assert "v2short" not in body["details"]["remedy"]

    def test_it_explains_why_a_generated_key_authenticates_as_nothing(self, client):
        """The misconception, not just the symptom.

        A T2 key looks self-contained, so "I generated one and it does not
        work" is the natural next move. It is a spelling of a T1 key.
        """
        standalone = T2KeyGenerator.generate(family=T2Type.T2S).raw
        explanation = error(
            client.post("/auth/t1/validate", json={"key": standalone})
        )["details"]["explanation"]
        assert "spelling of a T1 key" in explanation
        assert "key store" in explanation

    def test_it_is_still_a_401_with_the_same_code(self, client):
        """The wording changed; the contract did not."""
        standalone = T2KeyGenerator.generate(family=T2Type.T2S).raw
        response = client.post("/auth/t1/validate", json={"key": standalone})
        assert response.status_code == 401
        assert error(response)["code"] == "AUTH_INVALID_KEY"

    def test_a_malformed_key_still_says_it_is_malformed(self, client):
        """Not swallowed by the new branch."""
        response = client.post("/auth/t1/validate", json={"key": "T2S_nope"})
        assert response.status_code == 401
        assert "well-formed" not in error(response)["message"]

    def test_an_unregistered_plain_t1_key_is_unaffected(self, client):
        from hypernix.security.keymaster import T1KeyGenerator

        body = error(
            client.post("/auth/t1/validate", json={"key": T1KeyGenerator.generate()})
        )
        assert "Unknown or unregistered T1 key" in body["message"]


@needs_server
class TestAdminRefusal:
    """"forbidden" — and no amount of scope-widening will help."""

    def test_a_t2s_key_is_told_the_refusal_is_permanent(self, store, client):
        key = mint("create", "-v", "v2short", "--scopes", "read,write")
        response = client.post(
            "/hyperlink/pair", json={"label": "x"},
            headers={"Authorization": "Bearer " + key},
        )
        assert response.status_code == 403
        body = error(response)
        assert "never" in body["message"]
        assert body["details"]["reason"] == "t2s_is_never_admin"

    def test_it_says_widening_scopes_will_not_help(self, store, client):
        """The dead end this is meant to close off."""
        key = mint("create", "-v", "v2short", "--scopes", "read,write")
        body = error(client.post(
            "/hyperlink/pair", json={"label": "x"},
            headers={"Authorization": "Bearer " + key},
        ))
        assert "scopes will not change this" in body["details"]["explanation"]

    def test_it_gives_the_route_that_does_work(self, store, client):
        key = mint("create", "-v", "v2short", "--scopes", "read,write")
        remedy = error(client.post(
            "/hyperlink/pair", json={"label": "x"},
            headers={"Authorization": "Bearer " + key},
        ))["details"]["remedy"]
        assert "waiter hyperlink pair" in remedy
        assert "six-character" in remedy

    def test_a_non_admin_t1_key_gets_the_ordinary_message(self, store, client):
        """Its problem *is* fixable, so it gets the fixable answer."""
        key = mint("create", "-v", "v1", "--scopes", "read,write")
        body = error(client.post(
            "/hyperlink/pair", json={"label": "x"},
            headers={"Authorization": "Bearer " + key},
        ))
        assert "never" not in body["message"]
        assert "gkey create --type admin" in body["details"]["remedy"]


@needs_server
class TestWhatAT2SKeyCanDo:
    """The refusals must not have narrowed what actually works."""

    @pytest.mark.parametrize("path", ["/hyperlink/endpoints", "/hyperlink/sessions"])
    def test_a_registered_t2s_key_reaches_hyperlink(self, store, client, path):
        key = mint("create", "-v", "v2short", "--scopes", "read,write")
        response = client.get(path, headers={"Authorization": "Bearer " + key})
        assert response.status_code == 200

    def test_it_can_create_a_session(self, store, client):
        """Non-admin write, which is exactly what T2S is allowed."""
        key = mint("create", "-v", "v2short", "--scopes", "read,write")
        response = client.post(
            "/hyperlink/sessions", json={"title": "t"},
            headers={"Authorization": "Bearer " + key},
        )
        assert response.status_code == 200

    def test_a_read_only_t2s_key_still_reads(self, store, client):
        key = mint("create", "-v", "v2short")
        response = client.get(
            "/hyperlink/endpoints", headers={"Authorization": "Bearer " + key}
        )
        assert response.status_code == 200

    def test_it_must_be_a_bearer_header(self, store, client):
        """X-API-Key is not the HyperLink contract; the error says so."""
        key = mint("create", "-v", "v2short")
        response = client.get("/hyperlink/endpoints", headers={"X-API-Key": key})
        assert response.status_code == 401
        assert "Authorization" in error(response)["message"]


@needs_server
class TestKeysMintedWhileTheServerIsRunning:
    """The bug behind the report.

    ``validate_key`` refreshes the key store once when a key is unknown,
    because a key minted against a running server is not in the in-memory
    table yet. That retry lived inline in the T1 branch, and the T2 branch
    reached the store by another route — so a T1 key minted against a
    running server worked, and *the same key* in its T2 or T2S spelling
    was refused until the server restarted.

    Which produces the reported symptom exactly: mint a T2S key, use it,
    "invalid key" — on a key that is real, registered and correctly typed.
    """

    @staticmethod
    def _server():
        from fastapi.testclient import TestClient

        from hypernix.t1api.app import create_app

        return TestClient, create_app

    def test_a_t2s_key_minted_after_startup_is_accepted(self, store):
        TestClient, create_app = self._server()
        app = create_app()
        key = mint("create", "-v", "v2short", "--scopes", "read,write")
        with TestClient(app, client=("127.0.0.1", 5000)) as client:
            response = client.get(
                "/hyperlink/endpoints", headers={"Authorization": "Bearer " + key}
            )
        assert response.status_code == 200, response.text

    def test_a_t2_key_minted_after_startup_is_accepted(self, store):
        TestClient, create_app = self._server()
        app = create_app()
        key = mint("create", "-v", "v2", "--scopes", "read,write")
        with TestClient(app, client=("127.0.0.1", 5000)) as client:
            response = client.get("/models", headers={"Authorization": "Bearer " + key})
        assert response.status_code == 200

    def test_both_spellings_of_one_key_agree(self, store):
        """The sharpest form of the bug: same credential, two answers."""
        TestClient, create_app = self._server()
        app = create_app()
        t2s = mint("create", "-v", "v2short", "--scopes", "read,write")
        t1 = T2KeyGenerator.to_t1(t2s)
        with TestClient(app, client=("127.0.0.1", 5000)) as client:
            first = client.get(
                "/hyperlink/endpoints", headers={"Authorization": "Bearer " + t1}
            ).status_code
            second = client.get(
                "/hyperlink/endpoints", headers={"Authorization": "Bearer " + t2s}
            ).status_code
        assert first == second == 200

    def test_a_genuinely_unknown_key_is_still_refused(self, store):
        """The refresh must not turn into "accept anything"."""
        TestClient, create_app = self._server()
        app = create_app()
        mint("create", "-v", "v2short")  # something real in the store
        stranger = T2KeyGenerator.generate(family=T2Type.T2S).raw
        with TestClient(app, client=("127.0.0.1", 5000)) as client:
            response = client.get(
                "/hyperlink/endpoints", headers={"Authorization": "Bearer " + stranger}
            )
        assert response.status_code == 401


class TestVersionsAgree:
    """Three files carry the package version; they must not drift.

    ``__init__.__version__`` is what ``gkey version`` and ``waiter
    version`` report, ``pyproject.toml`` is what pip installs, and
    ``setup.cfg`` is the legacy declaration. A release that bumps two of
    the three ships a package whose self-reported version is a lie —
    which is exactly how ``waiter --help`` came to advertise a T1 version
    the API had left behind two releases earlier.
    """

    @staticmethod
    def _read(pattern: str, filename: str) -> str:
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2]
        match = re.search(pattern, (root / filename).read_text(), re.M)
        assert match, f"no version found in {filename}"
        return match.group(1)

    def test_all_three_match(self):
        import hypernix

        assert (
            hypernix.__version__
            == self._read(r'^version = "(\S+)"', "pyproject.toml")
            == self._read(r"^version = (\S+)", "setup.cfg")
        )
