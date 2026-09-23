"""``gkey create -v`` and ``gkey version`` — minting v1, v2 and v2short keys.

The property everything here rests on: **a v2 key is a spelling of a v1
key, not a separate credential.** Authentication converts it back to its
T1 form and looks that up in the key store, so a T2 key that was never
minted into the store authenticates as nothing at all. Every test that
checks a key "works" therefore authenticates it against a real
:class:`T1AuthService` over the same store rather than checking its shape.
"""
from __future__ import annotations

import contextlib
import io
import json
import re

import pytest

from hypernix.security.gatekeeper import Gatekeeper
from hypernix.security.gkey_cli import main
from hypernix.security.keymaster import Keymaster, KeyType
from hypernix.security.keyversions import (
    DEFAULT_KEY_VERSION,
    KEY_VERSIONS,
    LATEST_KEY_VERSION,
    RESERVED_KEY_VERSIONS,
    key_version_names,
    resolve_key_version,
)
from hypernix.security.t2keys import T2KeyGenerator
from hypernix.t1api.auth import T1AuthService
from hypernix.t1api.errors import T1APIError

ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An isolated key store.

    The default paths are module-level constants evaluated at import
    (``keymaster._DEFAULT_STORE``, ``gatekeeper._DEFAULT_DATA``), so
    setting ``$HOME`` or patching ``Path.home`` inside a test is far too
    late — the value was computed when the module first loaded. Patching
    the constants is what actually redirects them, and getting this wrong
    means the suite writes real credentials into the developer's own
    ``~/.hypernix/keymaster``.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "hypernix.security.keymaster._DEFAULT_STORE", tmp_path / "keymaster"
    )
    monkeypatch.setattr(
        "hypernix.security.gatekeeper._DEFAULT_DATA", tmp_path / "gatekeeper"
    )
    return tmp_path


def run(*argv: str) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, ANSI.sub("", out.getvalue() + err.getvalue())


def field(text: str, label: str) -> str:
    """Pull one ``Label: value`` out of gkey's panel or plain output."""
    pattern = re.compile(rf"\b{re.escape(label)}:\s*(.+)")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            return match.group(1).strip().rstrip("│").strip()
    raise AssertionError(f"no {label!r} in:\n{text}")


def mint(*argv: str) -> tuple[str, str]:
    code, text = run("create", *argv)
    assert code == 0, text
    return field(text, "Key"), text


def auth_service() -> T1AuthService:
    """A service built the way the server builds it: fresh, same store.

    Takes no path: it picks up the patched default, which is the point —
    if the redirect were not working, this would read a different store
    than gkey wrote to and every test here would fail loudly rather than
    quietly passing against the real one.
    """
    km = Keymaster(auto_rotate=False)
    return T1AuthService(km, Gatekeeper(keymaster=km), token_secret="x" * 32)


class TestTheRegistry:
    def test_the_four_issuable_versions(self):
        assert key_version_names() == ["v1", "v2", "v2short", "v2.1"]

    def test_latest_is_v2_1_not_v2short(self):
        """v2short is a variant for a constrained client, not a later version.

        Calling it "latest" would push HyperLink's deliberately restricted
        key at everyone as the newest thing. v2.1 is the newer format.
        """
        assert LATEST_KEY_VERSION.name == "v2.1"

    def test_the_default_stays_v1(self):
        """Changing what a bare `gkey create` mints is a breaking surprise."""
        assert DEFAULT_KEY_VERSION.name == "v1"

    @pytest.mark.parametrize(
        "alias,expected",
        [("v2s", "v2short"), ("t2s", "v2short"), ("2short", "v2short"),
         ("T2", "v2"), ("t1", "v1"), ("  V2  ", "v2")],
    )
    def test_aliases_resolve(self, alias, expected):
        assert resolve_key_version(alias).name == expected

    def test_v2_1_is_issuable(self):
        """Reserved until 0.72.6, when it got a real key-agreement step."""
        assert RESERVED_KEY_VERSIONS == ()
        version = resolve_key_version("v2.1")
        assert (version.family, version.prefix, version.issuable) == ("T2C", "T2C_", True)
        assert resolve_key_version("t2c") is version

    def test_an_unknown_version_lists_what_exists(self):
        with pytest.raises(ValueError, match="v1, v2, v2short, v2.1"):
            resolve_key_version("v9")

    def test_only_v2short_pins_a_body_length(self):
        assert resolve_key_version("v2short").body_length == 26
        assert resolve_key_version("v2").body_length is None

    def test_v2short_can_never_be_an_administrator(self):
        assert resolve_key_version("v2short").supports_admin is False


class TestMintedKeysAuthenticate:
    """The point of the feature: these keys must actually work."""

    @pytest.mark.parametrize(
        "version,prefix",
        [("v1", "T1_"), ("v2", "T2_"), ("v2short", "T2S_")],
    )
    def test_each_format_is_accepted_by_the_server(self, store, version, prefix):
        key, _ = mint("-v", version)
        assert key.startswith(prefix)
        context = auth_service().validate_key(key)
        assert context.key_id

    @pytest.mark.parametrize(
        "version,prefix",
        [("v1", "T1_"), ("v2", "T2_"), ("v2short", "T2S_")],
    )
    def test_a_key_containing_brackets_is_printed_intact(
        self, store, monkeypatch, version, prefix
    ):
        """The panel is rendered with rich markup on, and a key is not markup.

        The T1/T2 special-character set includes ``[`` and ``]``, so
        about one key in three thousand carries a bracket pair that rich
        reads as a style tag, eats, and prints without. That is not
        cosmetic: `gkey create` shows the operator a credential missing
        characters, they paste it, and it authenticates as nothing with
        no error anywhere saying why. It surfaced as a CI flake on one
        run out of many, which is exactly how often it happens.

        Forcing the special set to brackets makes it happen every time.
        """
        monkeypatch.setattr("hypernix.security.keymaster._SPECIAL_CHARS", "[]")
        monkeypatch.setattr("hypernix.security.t2keys._T2_SPECIAL_CHARS", "[]")

        key, text = mint("-v", version)
        assert key.startswith(prefix)
        assert "[" in key or "]" in key, "the special set was not forced"
        # The printed key is the minted key: it still authenticates.
        assert auth_service().validate_key(key).key_id
        assert key in text, "the panel printed something other than the key"

    def test_a_value_that_looks_like_markup_survives_the_panel(self):
        """The exact failure, without waiting for the dice.

        `[bold]` inside a value is what rich eats. Five random specials
        land on a pair like it about once in three thousand keys, which
        is a CI flake and a support ticket rather than a visible bug.
        """
        from hypernix.security.gkey_cli import _literal, _print_panel

        forged = "T2S_abcdefghijklmnopqrstuvwxyzAB[bold]/7-1"
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            _print_panel(f"[bold]Key:[/bold] [yellow]{_literal(forged)}[/yellow]")
        assert forged in ANSI.sub("", out.getvalue() + err.getvalue())

    def test_the_unescaped_form_really_would_lose_it(self):
        """Otherwise the test above proves nothing about the escaping."""
        pytest.importorskip("rich")
        forged = "T2S_abcdefghijklmnopqrstuvwxyzAB[bold]/7-1"
        out, err = io.StringIO(), io.StringIO()
        from hypernix.security.gkey_cli import _print_panel

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            _print_panel(f"Key: {forged}")
        assert forged not in ANSI.sub("", out.getvalue() + err.getvalue())

    def test_a_bracketed_note_is_printed_intact(self, store):
        """Same hazard, arbitrary text: --note is whatever was typed."""
        _key, text = mint("--note", "for the [alpha] cluster")
        assert "for the [alpha] cluster" in text

    def test_a_v2_key_carries_its_access_level(self, store):
        key, _ = mint("-v", "v2", "--level", "5")
        assert auth_service().validate_key(key).t2_access_level == 5

    def test_the_level_defaults_to_one(self, store):
        key, _ = mint("-v", "v2")
        assert auth_service().validate_key(key).t2_access_level == 1

    def test_both_spellings_of_the_same_key_work(self, store):
        """A v2 key is the same credential wearing a different name."""
        v2_key, _ = mint("-v", "v2", "--level", "3")
        v1_key = T2KeyGenerator.to_t1(v2_key)
        service = auth_service()
        assert service.validate_key(v2_key).key_id == service.validate_key(v1_key).key_id

    def test_revoking_kills_both_spellings(self, store):
        """Otherwise the v1 form is a live credential nobody is tracking."""
        v2_key, text = mint("-v", "v2")
        v1_key = T2KeyGenerator.to_t1(v2_key)
        assert run("revoke", field(text, "Key ID"), "--reason", "test")[0] == 0
        for key in (v2_key, v1_key):
            with pytest.raises(T1APIError):
                auth_service().validate_key(key)

    def test_a_v2_admin_key_is_really_an_admin(self, store):
        """Both senses of it, which are not the same sense.

        T2's own ``is_admin`` is the password component in the prefix; the
        authority a request gets comes from the key store. A key that
        looks administrative and is refused by every admin endpoint would
        be worse than one that never claimed to be.
        """
        key, text = mint("-v", "v2", "--type", "admin", "--scopes", "admin,read,write")
        assert field(text, "Password")
        context = auth_service().validate_key(key)
        assert context.is_admin, "the key store does not consider this an admin"
        assert context.t2_is_admin, "the key does not carry a password component"
        assert context.key_meta.key_type is KeyType.ADMIN

    def test_a_supplied_admin_password_is_used(self, store):
        key, text = mint(
            "-v", "v2", "--type", "admin", "--scopes", "admin,read,write",
            "--password", "Correct7Horse",
        )
        assert field(text, "Password") == "Correct7Horse"
        assert "Correct7Horse" in key

    def test_a_weak_supplied_password_is_refused(self, store):
        """Validated, not trusted — a pasted memorable string is the case."""
        code, _ = run(
            "create", "-v", "v2", "--type", "admin",
            "--scopes", "admin,read,write", "--password", "abc",
        )
        assert code != 0

    def test_a_v2short_body_is_exactly_26_characters(self, store):
        """The whole reason the format exists: it has to be typeable.

        The length is fixed at mint time, because presentation cannot
        change the body of a key that is already in the store.
        """
        key, _ = mint("-v", "v2short")
        assert len(T2KeyGenerator.parse(key).body) == 26

    def test_a_v2short_key_is_narrowed_outside_hyperlink(self, store):
        """Read and non-admin write, whatever the underlying key allows."""
        key, _ = mint("-v", "v2short", "--scopes", "read,write,plugin")
        scopes = {s.value for s in auth_service().validate_key(key).scopes}
        assert scopes == {"read", "write"}


class TestRefusalsLeaveNothingBehind:
    """A refusal must not leave a live credential nobody knows about.

    Every impossible combination is checked before the key is minted. If
    it were checked after, the key would already be in the store — valid,
    usable, and known to nobody, because the operator saw only an error.
    """

    @pytest.mark.parametrize(
        "argv,expected",
        [
            (["-v", "v2short", "--type", "admin"], "cannot be an administrator"),
            (["-v", "v9"], "Unknown key version"),
            (["-v", "v1", "--level", "5"], "does not apply"),
            (["-v", "v2", "--level", "12"], "must be 1-9"),
            (["-v", "v2", "--level", "0"], "must be 1-9"),
            (["-v", "v2", "--password", "Hunter2xyz"], "marks a key as an admin"),
            (["-v", "v2short", "--body-len", "30"], "conflicts with v2short"),
        ],
    )
    def test_refused_with_a_reason(self, store, argv, expected):
        code, text = run("create", *argv)
        assert code == 2, text
        assert expected in text, text

    def test_no_key_is_minted_by_a_refused_request(self, store):
        for argv in (
            ["-v", "v2short", "--type", "admin"],
            ["-v", "v1", "--level", "5"],
            ["-v", "v2", "--level", "12"],
        ):
            assert run("create", *argv)[0] == 2
        code, listing = run("list", "--json")
        assert code == 0
        assert json.loads(listing) == [], "a refused request left a key in the store"


class TestTheIssuedFormatIsRecorded:
    """The store only ever holds the T1 form.

    Without a note of which spelling was handed out, that fact is lost the
    moment the key scrolls off the screen — and the raw key is shown once.
    """

    def test_v2_records_its_version_and_level(self, store):
        _, text = mint("-v", "v2", "--level", "4")
        _, detail = run("list", "id", field(text, "Key ID"), "--json")
        tags = json.loads(detail)["tags"]
        assert tags["key_version"] == "v2"
        assert tags["access_level"] == "4"

    def test_v1_stays_untagged(self, store):
        """v1 is the default; tagging every key with it is noise."""
        _, text = mint("-v", "v1")
        _, detail = run("list", "id", field(text, "Key ID"), "--json")
        assert json.loads(detail)["tags"] == {}

    def test_the_v1_form_is_shown_alongside_a_v2_key(self, store):
        """Both spellings work, and the operator should be told so."""
        key, text = mint("-v", "v2")
        assert T2KeyGenerator.to_t1(key) in text


class TestVersionCommand:
    def test_it_reports_all_three_versions(self, store):
        import hypernix
        from hypernix.t1api.version import T1_VERSION_SHORT

        code, text = run("version")
        assert code == 0
        assert hypernix.__version__ in text
        assert T1_VERSION_SHORT in text
        assert LATEST_KEY_VERSION.name in text

    def test_it_lists_every_issuable_format(self, store):
        _, text = run("version")
        for version in KEY_VERSIONS:
            assert version.name in text
            assert version.prefix in text

    def test_it_lists_v2_1_as_the_latest(self, store):
        _, text = run("version")
        assert "v2.1" in text
        assert "T2C_" in text

    def test_json_output(self, store):
        import hypernix
        from hypernix.t1api.version import T1_VERSION_LONG, T1_VERSION_SHORT

        code, text = run("version", "--json")
        assert code == 0
        data = json.loads(text)
        assert data["hypernix"] == hypernix.__version__
        assert data["t1_api"]["short"] == T1_VERSION_SHORT
        assert data["t1_api"]["long"] == T1_VERSION_LONG
        assert data["key_versions"]["latest"] == "v2.1"
        assert data["key_versions"]["default"] == "v1"
        assert [v["name"] for v in data["key_versions"]["available"]] == [
            "v1", "v2", "v2short", "v2.1"
        ]
        assert data["key_versions"]["reserved"] == []

    def test_it_is_in_the_help(self, store):
        _, text = run("--help")
        assert "version" in text


class TestFromT1Admin:
    """The one place a format conversion may produce an admin key."""

    def test_plain_from_t1_still_never_produces_an_admin(self):
        """The guarantee that makes conversion safe for arbitrary keys."""
        from hypernix.security.keymaster import T1KeyGenerator

        wrapped = T2KeyGenerator.from_t1(T1KeyGenerator.generate(), access_level=9)
        assert wrapped.is_admin is False
        assert wrapped.password == ""

    def test_the_admin_form_round_trips_to_the_same_t1_key(self):
        """The password rides in the prefix and must not disturb the body.

        If it did, the key would convert back to a *different* T1 key and
        authenticate as nothing.
        """
        from hypernix.security.keymaster import T1KeyGenerator

        original = T1KeyGenerator.generate()
        admin = T2KeyGenerator.from_t1_admin(original, access_level=9)
        assert admin.is_admin
        assert T2KeyGenerator.to_t1(admin) == original

    def test_a_weak_password_is_refused(self):
        from hypernix.security.keymaster import T1KeyGenerator

        with pytest.raises(ValueError):
            T2KeyGenerator.from_t1_admin(T1KeyGenerator.generate(), password="aaa")


class TestPlainOutput:
    """Without ``rich``, markup must not reach the terminal literally."""

    def test_markup_is_stripped(self):
        from hypernix.security.gkey_cli import _strip_markup

        assert _strip_markup("[yellow]v2.1[/yellow] is reserved") == "v2.1 is reserved"

    def test_a_key_containing_brackets_survives(self):
        """``[`` and ``]`` are both in the key alphabet.

        A general ``\\[[^]]*]`` stripper would eat part of a key — the one
        string in this CLI that has to survive byte for byte — so the
        stripper works from a closed list of style names instead.
        """
        from hypernix.security.gkey_cli import _strip_markup

        for key in ("T1_abc[[&*./4", "T2_pw_body[]{}ll;:.<>/4-9", "T2S_a[b]c!@/2-1"):
            assert _strip_markup(key) == key

    def test_the_version_command_prints_no_markup(self, store, monkeypatch):
        monkeypatch.setattr("hypernix.security.gkey_cli._HAS_RICH", False)
        code, text = run("version")
        assert code == 0
        for tag in ("[yellow]", "[/yellow]", "[bold]", "[dim]", "[/dim]"):
            assert tag not in text, f"{tag} leaked into plain output"

    def test_keys_print_intact_without_rich(self, store, monkeypatch):
        """The end the stripper exists to protect."""
        monkeypatch.setattr("hypernix.security.gkey_cli._HAS_RICH", False)
        key, text = mint("-v", "v2", "--level", "2")
        assert key in text
        assert auth_service().validate_key(key).t2_access_level == 2
