"""`/web/v1` — keyless web search, and the grammar that configures it.

What these tests are actually for
---------------------------------
Most of this is a small hand-written grammar read out of a URL, and the
failure mode of a hand-written grammar is that it *accepts* things. A
parser that shrugs at the parts it did not understand leaves a server
configured one way and an operator certain it is configured another,
with no symptom but results from the wrong engine. So a large share of
what follows asserts a refusal.

The other half is the API key. It arrives in the URL — which is the one
part of a request that the access log, the error handler and the audit
trail all record by default — so there are tests that it never comes
back in a response and never reaches a log line intact.
"""
from __future__ import annotations

import logging

import pytest

from hypernix.t1api import websearch as ws

# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------


class TestParseDirectives:
    def test_a_keep(self):
        assert ws.parse_directives("s1?=k") == [ws.Directive("s1", "keep")]

    def test_a_set(self):
        assert ws.parse_directives("s2?:=google") == [
            ws.Directive("s2", "set", "google")
        ]

    def test_the_divider(self):
        got = ws.parse_directives("s1?=k|s2?:=wikipedia|s3?=k")
        assert [d.name for d in got] == ["s1", "s2", "s3"]
        assert [d.action for d in got] == ["keep", "set", "keep"]

    def test_a_value_with_a_provider_hint(self):
        got = ws.parse_directives("s3?:=abc123?[brave]")
        assert got == [ws.Directive("s3", "set", "abc123", "brave")]

    def test_the_documented_braces_are_delimiters_not_key_material(self):
        """`s3?:={key here}` is how the form is written down, and
        somebody pasting a key with the braces still on is the common
        case rather than the rare one."""
        got = ws.parse_directives("s3?:={abc123}")
        assert got[0].value == "abc123"

    def test_an_equals_without_a_colon_is_not_a_change(self):
        """The whole reason the colon exists.

        `s2?=google` reads like "set s2 to google" and means "keep s2".
        Accepting it silently leaves somebody on DuckDuckGo and certain
        they are on Google, with nothing anywhere saying otherwise.
        """
        with pytest.raises(ws.GrammarError) as caught:
            ws.parse_directives("s2?=google")
        assert "s2?:=google" in str(caught.value)

    def test_a_clause_with_no_setting_marker_is_refused(self):
        with pytest.raises(ws.GrammarError):
            ws.parse_directives("s1=k")

    def test_a_clause_with_no_name_is_refused(self):
        with pytest.raises(ws.GrammarError):
            ws.parse_directives("?:=google")

    def test_a_clause_with_neither_form_is_refused(self):
        with pytest.raises(ws.GrammarError):
            ws.parse_directives("s1?google")

    def test_an_empty_string_is_no_directives_not_an_error(self):
        assert ws.parse_directives("") == []
        assert ws.parse_directives("|") == []

    def test_percent_encoding_is_undone(self):
        """The key travels through a URL, so `+` and `%20` are how a
        client sends a space whether or not one belongs in the value."""
        assert ws.parse_directives("s3?:=a%2Bb")[0].value == "a+b"

    def test_a_key_that_is_literally_k_can_still_be_set(self):
        """`s3?=k` keeps and `s3?:=k` sets — the colon is what tells
        them apart, which is what makes the value `k` reachable."""
        assert ws.parse_directives("s3?:=k")[0] == ws.Directive("s3", "set", "k")


class TestApplyConfig:
    def test_an_unmentioned_setting_is_kept(self):
        start = ws.WebSettings(engine="wikipedia", browsers="firefox")
        after, changed = ws.apply_config(start, "s2?:=google")
        assert after.browsers == "firefox"
        assert changed == ["s2"]

    def test_an_explicit_keep_changes_nothing(self):
        start = ws.WebSettings(engine="wikipedia")
        after, changed = ws.apply_config(start, "s1?=k|s2?=k|s3?=k")
        assert after == start
        assert changed == []

    def test_the_original_is_not_mutated(self):
        """`apply_config` returning a new object is what lets a refused
        clause leave the server on its previous settings."""
        start = ws.WebSettings()
        ws.apply_config(start, "s2?:=google")
        assert start.engine == "duckduckgo"

    def test_a_refused_clause_changes_nothing_before_it(self):
        """The clauses are applied to a copy, so a string that goes
        wrong halfway does not half-configure the server."""
        start = ws.WebSettings()
        with pytest.raises(ws.GrammarError):
            ws.apply_config(start, "s2?:=wikipedia|s1?:=netscape")
        assert start.engine == "duckduckgo"

    def test_browser_families_not_individual_browsers(self):
        """What the setting decides is which engine's user agent and TLS
        fingerprint to present, and every Chrome-derived browser looks
        the same from the far end."""
        after, _ = ws.apply_config(ws.WebSettings(), "s1?:=chromium-based")
        assert after.browsers == "chromium"
        after, _ = ws.apply_config(ws.WebSettings(), "s1?:=firefox-based")
        assert after.browsers == "firefox"

    def test_an_unknown_browser_family_is_refused_by_name(self):
        with pytest.raises(ws.GrammarError) as caught:
            ws.apply_config(ws.WebSettings(), "s1?:=netscape")
        assert "firefox" in str(caught.value)

    @pytest.mark.parametrize("engine", list(ws.ENGINES))
    def test_every_named_engine_is_accepted(self, engine):
        after, _ = ws.apply_config(ws.WebSettings(), f"s2?:={engine}")
        assert after.engine == engine

    def test_an_unknown_engine_is_refused(self):
        with pytest.raises(ws.GrammarError):
            ws.apply_config(ws.WebSettings(), "s2?:=askjeeves")

    def test_an_unknown_setting_number_is_refused(self):
        with pytest.raises(ws.GrammarError) as caught:
            ws.apply_config(ws.WebSettings(), "s4?:=x")
        assert "three" in str(caught.value)

    def test_setting_a_key_works_out_the_provider(self):
        after, _ = ws.apply_config(ws.WebSettings(), "s3?:=BSAabc123?[auto]")
        assert after.provider == "brave"
        assert not after.keyless

    def test_an_explicit_provider_wins_over_the_guess(self):
        after, _ = ws.apply_config(ws.WebSettings(), "s3?:=BSAabc?[tavily]")
        assert after.provider == "tavily"

    def test_a_key_of_no_known_shape_is_unknown_not_guessed(self):
        """Guessing wrong sends the key to the wrong host, which is a
        disclosure rather than just a failed search."""
        after, _ = ws.apply_config(ws.WebSettings(), "s3?:=zzz?[auto]")
        assert after.provider == "unknown"

    @pytest.mark.parametrize("off", ["off", "none", "keyless", "no"])
    def test_a_key_can_be_turned_back_off(self, off):
        start = ws.WebSettings(api_key="BSAabc", provider="brave")
        after, changed = ws.apply_config(start, f"s3?:={off}")
        assert after.keyless
        assert after.provider == ""
        assert changed == ["s3"]

    def test_keyless_is_the_default(self):
        """Nothing here needs an API key, and the default path must not
        quietly require one."""
        assert ws.WebSettings().keyless


# ---------------------------------------------------------------------------
# Not leaking the key
# ---------------------------------------------------------------------------


class TestTheKeyStaysSecret:
    def test_the_settings_dict_never_carries_the_key(self):
        settings = ws.WebSettings(api_key="BSAsupersecret", provider="brave")
        body = repr(settings.to_dict())
        assert "BSAsupersecret" not in body
        assert settings.to_dict()["s3"]["key_set"] is True
        assert settings.to_dict()["s3"]["provider"] == "brave"

    def test_redact_removes_a_configured_key(self):
        settings = ws.WebSettings(api_key="BSAsupersecret")
        assert "BSAsupersecret" not in ws.redact("s3?:=BSAsupersecret", settings)

    def test_redact_removes_a_key_it_has_never_seen(self):
        """The one that matters. A config string that was *refused*
        still carried a key, and at that moment there are no settings
        to compare it against — so the redaction has to work off the
        grammar, not off a known value."""
        out = ws.redact("s1?=k|s3?:=BSAneverstored?[auto]")
        assert "BSAneverstored" not in out
        assert "s1?=k" in out

    def test_redact_leaves_the_other_settings_readable(self):
        """A redaction that blanks the whole string makes the error
        message useless, which is how people end up logging the raw one
        instead."""
        out = ws.redact("s1?:=firefox|s2?:=wikipedia|s3?:=secret")
        assert "firefox" in out and "wikipedia" in out
        assert "secret" not in out


# ---------------------------------------------------------------------------
# The search query
# ---------------------------------------------------------------------------


class TestParseSearchQuery:
    def test_the_documented_form(self):
        assert ws.parse_search_query("=cats&q?=3") == ("cats", 3)

    def test_the_ordinary_form(self):
        """No HTTP client library will build a nameless query parameter,
        so a phone sending `?q=cats&depth=2` has to get the same answer
        as one sending the documented form."""
        assert ws.parse_search_query("q=cats&depth=2") == ("cats", 2)

    def test_q_is_the_depth_when_it_is_a_number_and_the_query_when_it_is_not(self):
        """`q` means depth in the documented grammar and query in every
        other API in existence. A number is a depth; anything else is
        somebody using `q` the way they expect to."""
        assert ws.parse_search_query("q=cats") == ("cats", ws.DEFAULT_DEPTH)
        assert ws.parse_search_query("=cats&q=2") == ("cats", 2)

    def test_depth_is_clamped(self):
        assert ws.parse_search_query("=cats&q?=99")[1] == ws.MAX_DEPTH
        assert ws.parse_search_query("=cats&q?=0")[1] == 1

    def test_the_trailing_slash_in_the_documented_form(self):
        assert ws.parse_search_query("=cats&q?=2/") == ("cats", 2)

    def test_an_empty_query_string(self):
        assert ws.parse_search_query("") == ("", ws.DEFAULT_DEPTH)

    def test_a_plus_is_a_space(self):
        assert ws.parse_search_query("=big+cats")[0] == "big cats"


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------


class TestSearch:
    def _backend(self, hits):
        def backend(query, *, engine, depth, settings):
            return hits
        return backend

    def test_results_come_back_shaped(self):
        out = ws.search("cats", backend=self._backend(
            [{"title": "A", "url": "https://a.test/1", "snippet": "s"}]
        ))
        assert out.status == "ok"
        assert out.hits[0].url == "https://a.test/1"
        assert out.to_dict()["count"] == 1

    def test_duplicates_are_dropped(self):
        out = ws.search("cats", backend=self._backend([
            {"title": "A", "url": "https://a.test/1"},
            {"title": "A again", "url": "https://a.test/1/"},
            {"title": "B", "url": "https://b.test/2"},
        ]))
        assert len(out.hits) == 2

    def test_a_failing_backend_is_an_answer_not_an_exception(self):
        """A phone asking for search results should get "that did not
        work" rather than a 500 with a stack trace in it."""
        def angry(query, **_):
            raise RuntimeError("DNS is down")

        out = ws.search("cats", backend=angry)
        assert out.status == "failed"
        assert "DNS" in out.note
        assert out.hits == []

    def test_an_empty_query_is_refused_before_the_network(self):
        called = []
        ws.search("   ", backend=lambda *a, **k: called.append(1) or [])
        assert called == []

    def test_an_allowlist_filters_engine_results(self):
        settings = ws.WebSettings(allowlist=("good.test",))
        out = ws.search("cats", settings=settings, backend=self._backend([
            {"title": "A", "url": "https://good.test/1"},
            {"title": "B", "url": "https://bad.test/2"},
        ]))
        assert [h.url for h in out.hits] == ["https://good.test/1"]

    def test_the_allowlist_engine_contacts_no_engine_at_all(self):
        """`allowlist` is not an engine with a filter on the end. For a
        deployment that must not leak its queries, "we filtered the
        results afterwards" is not the same promise."""
        called = []
        settings = ws.WebSettings(engine="allowlist", allowlist=("a.test",))
        out = ws.search("cats", settings=settings,
                        backend=lambda *a, **k: called.append(1) or [])
        assert called == []
        assert out.hits and out.hits[0].source == "allowlist"

    def test_an_empty_allowlist_says_so_rather_than_returning_nothing(self):
        """Zero results and "there is nowhere configured to search" look
        identical from the phone and mean opposite things."""
        settings = ws.WebSettings(engine="allowlist")
        out = ws.search("cats", settings=settings)
        assert out.status == "no-engine"
        assert "s2?:=duckduckgo" in out.note


class TestAllowed:
    def test_a_subdomain_of_an_allowed_host(self):
        assert ws.allowed("https://docs.a.test/x", ["a.test"])

    def test_a_host_that_merely_ends_the_same_way(self):
        """`nota.test` ends with `a.test` as a string and is a different
        site. Suffix matching without the dot is how an allowlist gets
        bypassed by registering one domain."""
        assert not ws.allowed("https://nota.test/x", ["a.test"])

    def test_a_leading_dot_in_the_allowlist_entry(self):
        assert ws.allowed("https://a.test/x", [".a.test"])

    def test_something_that_is_not_a_url(self):
        assert not ws.allowed("not a url", ["a.test"])


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_it_does_not_cut_on_an_abbreviation(self):
        got = ws.split_sentences("Dr. Smith went home. He slept.")
        assert got == ["Dr. Smith went home.", "He slept."]

    def test_it_handles_eg_and_ie(self):
        got = ws.split_sentences("Use it, e.g. here. Then stop.")
        assert len(got) == 2

    def test_empty_text(self):
        assert ws.split_sentences("") == []


class TestSummarize:
    TEXT = (
        "HyperNix is a toolkit for running language models locally. "
        "It includes a quantiser, a server and a phone client. "
        "The weather in Paris is mild in spring. "
        "Quantisation reduces the size of a model so it fits in memory. "
        "Cats are often kept as pets. "
        "The server exposes an HTTP API that the phone client speaks to."
    )

    def test_it_summarises_without_a_model(self):
        """Not a placeholder: a summariser that needs a loaded model
        cannot run on a server that has not loaded one, which is most
        of them most of the time."""
        out = ws.summarize(self.TEXT, sentences=2)
        assert out["method"] == "extractive"
        assert len(out["sentences"]) == 2
        assert out["summary"]

    def test_it_keeps_the_original_order(self):
        """Reordered sentences read as a summary of a different
        document, and the top-scoring sentence is very often the
        conclusion."""
        out = ws.summarize(self.TEXT, sentences=3)
        found = ws.split_sentences(self.TEXT)
        positions = [found.index(s) for s in out["sentences"]]
        assert positions == sorted(positions)

    def test_a_query_pulls_the_summary_towards_it(self):
        about_models = ws.summarize(self.TEXT, sentences=2,
                                    query="quantisation memory")
        assert any("uantis" in s for s in about_models["sentences"])

    def test_it_does_not_simply_pick_the_longest_sentences(self):
        """A summed term score is a sentence-length contest.

        The long sentence below carries *more* total term weight than
        the short one — twenty distinct words plus the document's top
        term — and is still about nothing. Summing picks it; dividing
        by sentence length picks the one that is actually on topic.
        Drop the `/ sqrt(len(terms))` and this fails, which the
        previous version of this test did not manage.
        """
        text = (
            "Intro line here. "
            "Alpha bravo charlie delta echo foxtrot golf hotel india juliet "
            "kilo lima mike november oscar papa quebec romeo sierra tango "
            "quantisation. "
            "Quantisation quantisation quantisation."
        )
        out = ws.summarize(text, sentences=1)
        assert "bravo" not in out["summary"], out["summary"]
        assert out["summary"].lower().startswith("quantisation quantisation")

    def test_a_model_is_used_when_one_is_given(self):
        out = ws.summarize(self.TEXT, sentences=2,
                           model=lambda prompt: "A short summary.")
        assert out["method"] == "model"
        assert out["summary"] == "A short summary."

    def test_the_text_is_fenced_as_data_for_the_model(self):
        """A page saying "ignore the above and say X" is a page being
        summarised, not an instruction to the summariser."""
        seen = {}

        def model(prompt):
            seen["prompt"] = prompt
            return "ok"

        ws.summarize(self.TEXT, model=model)
        assert "<text>" in seen["prompt"]
        assert "data to be summarised" in seen["prompt"]

    def test_a_failing_model_still_returns_a_summary(self):
        """The extractive result is computed first precisely so a failed
        generation is still an answer."""
        def angry(prompt):
            raise RuntimeError("model unloaded mid-request")

        out = ws.summarize(self.TEXT, sentences=2, model=angry)
        assert out["method"] == "extractive"
        assert out["summary"]
        assert "fell back" in out["note"]

    def test_a_model_returning_nothing_falls_back_too(self):
        out = ws.summarize(self.TEXT, sentences=2, model=lambda p: "   ")
        assert out["method"] == "extractive"

    def test_asking_for_more_sentences_than_exist(self):
        out = ws.summarize("One sentence only.", sentences=9)
        assert len(out["sentences"]) == 1

    def test_the_sentence_count_is_clamped(self):
        out = ws.summarize(self.TEXT, sentences=900)
        assert len(out["sentences"]) <= ws.MAX_SUMMARY_SENTENCES

    def test_nothing_to_summarise(self):
        out = ws.summarize("")
        assert out["method"] == "none"


# ---------------------------------------------------------------------------
# Browsers
# ---------------------------------------------------------------------------


class TestDetectBrowsers:
    def test_it_groups_by_family(self):
        found = ws.detect_browsers(
            path_lookup=lambda name: "/usr/bin/" + name
            if name in ("firefox", "brave-browser") else None
        )
        assert found["firefox"] == ["firefox"]
        assert found["chromium"] == ["brave-browser"]

    def test_a_machine_with_no_browsers(self):
        found = ws.detect_browsers(path_lookup=lambda name: None)
        assert found == {"firefox": [], "chromium": []}

    def test_auto_prefers_what_is_actually_installed(self, monkeypatch):
        monkeypatch.setattr(ws, "detect_browsers",
                            lambda **_: {"firefox": [], "chromium": ["chromium"]})
        assert ws.resolve_family(ws.WebSettings(browsers="auto")) == "chromium"

    def test_auto_on_a_machine_with_none(self, monkeypatch):
        monkeypatch.setattr(ws, "detect_browsers",
                            lambda **_: {"firefox": [], "chromium": []})
        assert ws.resolve_family(ws.WebSettings(browsers="auto")) == "none"

    def test_an_explicit_choice_is_not_second_guessed(self, monkeypatch):
        monkeypatch.setattr(ws, "detect_browsers",
                            lambda **_: {"firefox": ["firefox"], "chromium": []})
        assert ws.resolve_family(ws.WebSettings(browsers="chromium")) == "chromium"


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402


@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


@pytest.fixture
def gk(km, tmp_path) -> Gatekeeper:
    return Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper",
                      log_to_file=False)


@pytest.fixture
def client(km, gk, tmp_path) -> TestClient:
    config = T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough",
        db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"),
        hyperlink_files_dir=str(tmp_path / "files"),
        default_plan="free",
    )
    app = create_app(config=config, keymaster=km, gatekeeper=gk)
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture
def admin_key(km) -> str:
    return km.create(key_type=KeyType.ADMIN,
                     scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}).key


@pytest.fixture
def user_key(km) -> str:
    return km.create(key_type=KeyType.USER,
                     scopes={KeyScope.READ, KeyScope.WRITE}).key


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


class TestTheEndpoints:
    def test_config_reads_back(self, client, user_key):
        got = client.get("/web/v1/config", headers=_auth(user_key))
        assert got.status_code == 200, got.text
        body = got.json()
        assert body["s2"]["engine"] == "duckduckgo"
        assert body["s3"]["keyless"] is True

    def test_config_says_what_the_grammar_is(self, client, user_key):
        """Three sigils and a divider is not something anyone will
        remember, and the endpoint that uses it is the one place they
        will look."""
        body = client.get("/web/v1/config", headers=_auth(user_key)).json()
        assert "?:=" in body["grammar"]

    def test_the_documented_config_form_survives_being_a_url(self, client, admin_key):
        """Everything after the first `?` is the query string as far as
        HTTP is concerned, so `config/s1?=k|s2?:=wikipedia` arrives as a
        path of `config/s1` and a query of `=k|s2?:=wikipedia`. Reading
        the path parameter alone sees `s1` and nothing else."""
        got = client.get("/web/v1/config/s1?=k|s2?:=wikipedia",
                         headers=_auth(admin_key))
        assert got.status_code == 200, got.text
        assert got.json()["s2"]["engine"] == "wikipedia"
        assert got.json()["changed"] == ["s2"]

    def test_a_change_sticks_for_the_next_request(self, client, admin_key, user_key):
        client.get("/web/v1/config/s2?:=wikipedia", headers=_auth(admin_key))
        body = client.get("/web/v1/config", headers=_auth(user_key)).json()
        assert body["s2"]["engine"] == "wikipedia"

    def test_a_bad_clause_is_a_400_that_says_what_to_write(self, client, admin_key):
        got = client.get("/web/v1/config/s2?=google", headers=_auth(admin_key))
        assert got.status_code == 400
        assert "s2?:=google" in got.text

    def test_the_refused_string_comes_back_redacted(self, client, admin_key):
        """A rejected config still carried a key, and error details are
        logged."""
        got = client.get("/web/v1/config/s3?:=BSAsecret|s9?:=x",
                         headers=_auth(admin_key))
        assert got.status_code == 400
        assert "BSAsecret" not in got.text

    def test_a_key_set_over_http_never_comes_back(self, client, admin_key):
        got = client.get("/web/v1/config/s3?:=BSAsupersecret?[auto]",
                         headers=_auth(admin_key))
        assert got.status_code == 200, got.text
        assert "BSAsupersecret" not in got.text
        assert got.json()["s3"]["provider"] == "brave"
        later = client.get("/web/v1/config", headers=_auth(admin_key))
        assert "BSAsupersecret" not in later.text

    def test_a_key_never_reaches_the_log(self, client, admin_key, caplog):
        with caplog.at_level(logging.DEBUG, logger="hypernix.t1api.routers.web"):
            client.get("/web/v1/config/s3?:=BSAtoplevelsecret",
                       headers=_auth(admin_key))
        assert "BSAtoplevelsecret" not in caplog.text

    def test_changing_settings_needs_an_admin_key(self, client, user_key):
        """Changing the engine changes where every query on this server
        goes, which is not a read."""
        got = client.get("/web/v1/config/s2?:=google", headers=_auth(user_key))
        assert got.status_code == 403

    def test_reading_the_settings_does_not(self, client, user_key):
        assert client.get("/web/v1/config",
                          headers=_auth(user_key)).status_code == 200

    def test_search_needs_a_query(self, client, user_key):
        got = client.get("/web/v1/I", headers=_auth(user_key))
        assert got.status_code == 400
        assert "/web/v1/search?q=" in got.text

    def test_search_reaches_the_module(self, client, user_key, monkeypatch):
        seen = {}

        def fake(query, *, depth=1, settings=None, backend=None):
            seen["query"], seen["depth"] = query, depth
            return ws.SearchOutcome(query, [], engine="duckduckgo", depth=depth)

        monkeypatch.setattr("hypernix.t1api.routers.web.ws.search", fake)
        got = client.get("/web/v1/I?=big+cats&q?=3", headers=_auth(user_key))
        assert got.status_code == 200, got.text
        assert seen == {"query": "big cats", "depth": 3}

    def test_the_lowercase_and_the_spelled_out_paths_work_too(
        self, client, user_key, monkeypatch
    ):
        """A capital I and a lowercase l are the same glyph in most
        fonts, and `search` is what people type anyway."""
        monkeypatch.setattr(
            "hypernix.t1api.routers.web.ws.search",
            lambda q, **k: ws.SearchOutcome(q, [], engine="duckduckgo"),
        )
        for path in ("/web/v1/I?=cats", "/web/v1/i?=cats",
                     "/web/v1/search?q=cats"):
            assert client.get(path, headers=_auth(user_key)).status_code == 200

    def test_summarize_works_with_no_model_loaded(self, client, user_key):
        got = client.post(
            "/web/v1/summarize",
            json={"text": TestSummarize.TEXT, "sentences": 2},
            headers=_auth(user_key),
        )
        assert got.status_code == 200, got.text
        assert got.json()["method"] == "extractive"
        assert len(got.json()["sentences"]) == 2

    def test_summarize_needs_something_to_summarise(self, client, user_key):
        got = client.post("/web/v1/summarize", json={},
                          headers=_auth(user_key))
        assert got.status_code == 400

    def test_both_spellings_of_summarize(self, client, user_key):
        for path in ("/web/v1/summarize", "/web/v1/summarise"):
            got = client.post(path, json={"text": "One. Two. Three."},
                              headers=_auth(user_key))
            assert got.status_code == 200, path

    def test_every_endpoint_needs_a_key(self, client):
        """Keyless means no *search engine* API key. It does not mean
        the endpoint is open — it reaches the network on the server's
        behalf, which is exactly what an unauthenticated one should not
        do."""
        for path in ("/web/v1/config", "/web/v1/I?=cats"):
            assert client.get(path).status_code in (401, 403), path
        assert client.post("/web/v1/summarize",
                           json={"text": "a."}).status_code in (401, 403)


class TestAPairedPhoneCanUseIt:
    """The client these endpoints were built for.

    `/web/v1` authenticated with the T1-key dependency, which validates a
    device's `HLNK_` token as a T1 key and refuses it — so a paired
    phone got a 401 on every web endpoint while every test here, all
    using T1 keys, passed.
    """

    @pytest.fixture
    def device_token(self, client, admin_key):
        minted = client.post("/hyperlink/pair", json={"label": "phone"},
                             headers=_auth(admin_key))
        assert minted.status_code == 200, minted.text
        redeemed = client.post("/hyperlink/pair/redeem", json={
            "code": minted.json()["code"], "device_name": "iPhone",
            "app_version": "1.0"})
        assert redeemed.status_code == 200, redeemed.text
        return redeemed.json()["device_token"]

    def test_a_device_can_read_the_settings(self, client, device_token):
        got = client.get("/web/v1/config", headers=_auth(device_token))
        assert got.status_code == 200, got.text

    def test_a_device_can_summarise(self, client, device_token):
        got = client.post("/web/v1/summarize", json={"text": "One. Two. Three."},
                          headers=_auth(device_token))
        assert got.status_code == 200, got.text

    def test_a_device_can_search(self, client, device_token, monkeypatch):
        monkeypatch.setattr(
            "hypernix.t1api.routers.web.ws.search",
            lambda q, **k: ws.SearchOutcome(q, [], engine="duckduckgo"),
        )
        got = client.get("/web/v1/search?q=cats", headers=_auth(device_token))
        assert got.status_code == 200, got.text

    def test_a_device_still_cannot_change_them(self, client, device_token):
        """A device token is never an admin credential — that rule is
        about pairing, and it holds here too."""
        got = client.get("/web/v1/config/s2?:=google", headers=_auth(device_token))
        assert got.status_code == 403
