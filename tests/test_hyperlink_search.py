"""``hypernix.hyperlink.search`` — finding the conversation you half remember.

A phone accumulates months of conversations and offers a date-sorted
list to find them in. What people want is "the one where I worked out
the CUDA thing", and scrolling is the only tool for it.

The implementation narrows in SQL and matches in Python rather than
using FTS5, because the T1 API runs on SQLite *or* PostgreSQL and FTS5
is SQLite-only. The tests below are mostly about what that buys:
Unicode-correct folding and no wildcard injection, neither of which
``LIKE`` gives.
"""
from __future__ import annotations

import pathlib

import pytest

from hypernix.hyperlink.search import (
    MAX_TERMS,
    SEARCHABLE_ROLES,
    SearchIndex,
    normalise,
    tokenize,
)
from hypernix.hyperlink.sessions import ChatSessionStore
from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.errors import T1APIError, T1ErrorCode


@pytest.fixture
def library(tmp_path: pathlib.Path):
    backend = SQLiteBackend(db_path=tmp_path / "search.sqlite3")
    sessions = ChatSessionStore(backend)
    index = SearchIndex(backend)

    def add(owner: str, title: str, messages: list[tuple[str, str]]):
        session = sessions.create(owner=owner, title=title)
        for role, content in messages:
            sessions.append(session_id=session.session_id, role=role, content=content)
        return session

    return sessions, index, add


@pytest.fixture
def stocked(library):
    sessions, index, add = library
    add("alice", "CUDA memory notes", [
        ("user", "why does the GPU run out of VRAM at batch size 8?"),
        ("assistant", "Gradient checkpointing trades compute for VRAM."),
    ])
    add("alice", "Dinner plans", [("user", "pasta or curry tonight")])
    add("alice", "Straße names", [("user", "Die Straße ist lang")])
    add("bob", "Bob's VRAM notes", [("user", "VRAM everywhere")])
    return sessions, index, add


class TestUnicodeCorrectness:
    """The first reason this is not a `LIKE`.

    SQLite's `LIKE` is case-insensitive for ASCII only, so a German or
    Turkish user's search silently returns nothing.
    """

    def test_folding_handles_the_german_sharp_s(self, stocked):
        _, index, _ = stocked
        assert normalise("Straße") == "strasse"
        hits = index.search("STRASSE", owner="alice").hits
        assert hits, "'STRASSE' did not find 'Straße'"

    def test_folding_handles_the_turkish_dotted_i(self):
        """`lower()` gets this wrong; `casefold()` does not."""
        assert normalise("İ") != "i̇".upper()
        assert normalise("İSTANBUL") == normalise("i̇stanbul")

    def test_nfkc_makes_alternate_encodings_compare_equal(self):
        """A phone keyboard and a desktop keyboard produce different
        codepoints for what the user considers one string."""
        assert normalise("ﬁle") == normalise("file")
        assert normalise("Ａ") == normalise("a")

    def test_a_search_matches_across_encodings(self, library):
        sessions, index, add = library
        add("alice", "ﬁle handling", [("user", "opening a ﬁle")])
        assert index.search("file", owner="alice").hits

    def test_accents_are_not_stripped(self, library):
        """Folding is not transliteration. "resume" must not match
        "résumé" -- they are different words, and pretending otherwise
        makes every search noisier."""
        sessions, index, add = library
        add("alice", "résumé", [("user", "my résumé")])
        assert not index.search("resume", owner="alice").hits
        assert index.search("RÉSUMÉ", owner="alice").hits


class TestNoWildcardInjection:
    """The second reason this is not a `LIKE`.

    In a `LIKE`, an unescaped `%` matches everything. The call that
    forgets to escape it is the one that returns the whole database.
    """

    @pytest.mark.parametrize("query", ["%", "%%", "*", "[]"])
    def test_a_bare_wildcard_matches_nothing(self, stocked, query):
        """In a `LIKE`, `%` matches every row. Here it is punctuation,
        so it tokenises to nothing and finds nothing."""
        _, index, _ = stocked
        assert index.search(query, owner="alice").hits == []

    @pytest.mark.parametrize(("wrapped", "bare"), [("%a%", "a"), ("%vram%", "vram")])
    def test_percent_is_dropped_not_expanded(self, stocked, wrapped, bare):
        """The real question is not "few hits" -- a one-letter query
        legitimately matches a lot -- but whether the wildcard changed
        the answer. It must not: `%a%` is a search for `a`.
        """
        _, index, _ = stocked
        wrapped_ids = {
            (h.session_id, h.message_id) for h in index.search(wrapped, owner="alice").hits
        }
        bare_ids = {
            (h.session_id, h.message_id) for h in index.search(bare, owner="alice").hits
        }
        assert wrapped_ids == bare_ids

    def test_underscore_is_a_word_character_not_a_single_char_wildcard(self):
        """It differs from `%`, and the difference is worth stating.

        `_` is `\w`, so `_vram_` tokenises as one literal word rather
        than being dropped as punctuation. In a `LIKE` it would match
        any single character either side; here it matches the literal
        string and therefore nothing. Safe, but not the same rule as
        `%`, and a reader who assumes otherwise will write a search that
        silently finds nothing.
        """
        assert tokenize("_vram_") == ["_vram_"]
        assert tokenize("%vram%") == ["vram"]

    def test_an_underscore_query_finds_nothing_rather_than_anything(self, stocked):
        _, index, _ = stocked
        assert index.search("_vram_", owner="alice").hits == []

    def test_a_wildcard_does_not_widen_a_specific_query(self, stocked):
        _, index, _ = stocked
        assert len(index.search("%zzzznotpresent%", owner="alice").hits) == 0

    def test_a_literal_percent_is_findable(self, library):
        sessions, index, add = library
        add("alice", "stats", [("user", "GPU at 97% utilisation")])
        # Tokenising drops the punctuation, so this finds the number --
        # the point is that it does not match every row.
        assert len(index.search("97", owner="alice").hits) == 1

    def test_a_sql_quote_does_not_break_anything(self, stocked):
        _, index, _ = stocked
        assert index.search("'; DROP TABLE hyperlink_messages; --", owner="alice") is not None
        # And the table is still there.
        assert index.search("vram", owner="alice").hits


class TestTokenizing:
    def test_it_lowercases_and_splits(self):
        assert set(tokenize("The CUDA Thing")) == {"the", "cuda", "thing"}

    def test_longest_first_so_the_snippet_follows_the_signal(self):
        """A search for "the cuda thing" should show the line with
        *cuda* in it, not the first "the"."""
        assert tokenize("the cuda thing")[0] == "thing"
        assert len(tokenize("a cuda")[0]) >= len(tokenize("a cuda")[-1])

    def test_repeats_do_not_multiply_weight(self):
        """"cuda cuda cuda" is one intent, not three."""
        assert tokenize("cuda cuda cuda") == ["cuda"]

    def test_punctuation_is_dropped(self):
        assert tokenize("what's --this-- ?!") == sorted(
            tokenize("what's --this-- ?!"), key=len, reverse=True
        )
        assert "?" not in tokenize("what?")

    def test_a_paste_is_capped(self):
        assert len(tokenize(" ".join(f"word{i}" for i in range(200)))) == MAX_TERMS

    @pytest.mark.parametrize("query", ["", "   ", "!!!", "-- --"])
    def test_a_query_with_no_terms_is_empty(self, query):
        assert tokenize(query) == []


class TestFindingThings:
    def test_it_finds_a_message(self, stocked):
        _, index, _ = stocked
        hits = index.search("checkpointing", owner="alice").hits
        assert len(hits) == 1
        assert hits[0].role == "assistant"

    def test_it_finds_a_title(self, stocked):
        _, index, _ = stocked
        hits = index.search("dinner", owner="alice").hits
        assert hits and hits[0].title == "Dinner plans"

    def test_an_empty_query_finds_nothing_rather_than_everything(self, stocked):
        _, index, _ = stocked
        result = index.search("   ", owner="alice")
        assert result.hits == [] and result.terms == []

    def test_one_owner_never_sees_anothers_conversations(self, stocked):
        """Bob has a session with VRAM in the title and the body."""
        _, index, _ = stocked
        for hit in index.search("vram", owner="alice").hits:
            assert "Bob" not in hit.title

    def test_partial_matches_still_appear(self, stocked):
        """Requiring every term turns a slightly misremembered query
        into no results, which reads as "the conversation is gone"."""
        _, index, _ = stocked
        hits = index.search("vram elephant", owner="alice").hits
        assert hits, "a two-term query found nothing when one term matched"

    def test_a_system_prompt_is_not_searched(self, library):
        """It appears in every session, so matching it returns
        everything and ranks it all identically."""
        sessions, index, add = library
        add("alice", "chat", [("system", "You are a helpful assistant."),
                              ("user", "hello")])
        assert "system" not in SEARCHABLE_ROLES
        assert not index.search("helpful", owner="alice").hits

    def test_it_can_be_scoped_to_one_session(self, stocked):
        sessions, index, _ = stocked
        target = [
            s for s in sessions.list_sessions(owner="alice")
            if s.title == "CUDA memory notes"
        ][0]
        hits = index.search("vram", owner="alice", session_id=target.session_id).hits
        assert all(h.session_id == target.session_id for h in hits)

    def test_it_can_be_scoped_to_a_role(self, stocked):
        _, index, _ = stocked
        hits = index.search("vram", owner="alice", roles=["assistant"]).hits
        assert hits and all(h.role == "assistant" for h in hits)

    def test_an_unsearchable_role_is_refused_with_the_valid_list(self, stocked):
        _, index, _ = stocked
        with pytest.raises(T1APIError) as caught:
            index.search("x", owner="alice", roles=["system"])
        assert caught.value.code == T1ErrorCode.VALIDATION_ERROR
        assert "user" in str(caught.value)

    def test_an_owner_is_required(self, stocked):
        _, index, _ = stocked
        with pytest.raises(T1APIError):
            index.search("x", owner="")

    def test_archived_sessions_are_excluded_unless_asked_for(self, library):
        sessions, index, add = library
        session = add("alice", "old CUDA thread", [("user", "cuda")])
        sessions.update(session.session_id, archived=True)
        assert not index.search("cuda", owner="alice").hits
        assert index.search("cuda", owner="alice", include_archived=True).hits


class TestRanking:
    def test_a_title_match_outranks_a_body_match(self, stocked):
        """A title is a deliberate summary."""
        _, index, _ = stocked
        hits = index.search("cuda", owner="alice").hits
        assert hits[0].message_id == "", "a body match came first"

    def test_matching_every_term_outranks_matching_some(self, library):
        sessions, index, add = library
        add("alice", "a", [("user", "alpha beta together in one message")])
        add("alice", "b", [("user", "alpha only, no second word")])
        hits = index.search("alpha beta", owner="alice").hits
        assert "together" in hits[0].snippet.text

    def test_an_exact_phrase_outranks_scattered_words(self, library):
        sessions, index, add = library
        add("alice", "a", [("user", "the quick brown fox jumped")])
        add("alice", "b", [("user", "quick, and separately, brown")])
        hits = index.search("quick brown", owner="alice").hits
        assert "fox" in hits[0].snippet.text

    def test_more_occurrences_outrank_fewer(self, library):
        sessions, index, add = library
        add("alice", "a", [("user", "cuda " * 10)])
        add("alice", "b", [("user", "cuda once")])
        hits = index.search("cuda", owner="alice").hits
        assert hits[0].score > hits[1].score

    def test_occurrences_are_damped(self, library):
        """Without damping the longest document wins every search."""
        sessions, index, add = library
        add("alice", "a", [("user", "cuda " * 100)])
        add("alice", "b", [("user", "cuda " * 4)])
        hits = index.search("cuda", owner="alice").hits
        assert hits[0].score < hits[1].score * 25, "linear in occurrences"

    def test_recency_breaks_a_tie(self, library):
        sessions, index, add = library
        old = add("alice", "old", [("user", "identical text here")])
        new = add("alice", "new", [("user", "identical text here")])
        with sessions.backend.connect() as conn:
            conn.execute(
                "UPDATE hyperlink_messages SET created_at = 0 WHERE session_id = ?",
                (old.session_id,),
            )
        hits = [h for h in index.search("identical", owner="alice").hits if h.message_id]
        assert hits[0].session_id == new.session_id


class TestSnippets:
    def test_a_snippet_locates_the_match(self, stocked):
        _, index, _ = stocked
        hit = [h for h in index.search("checkpointing", owner="alice").hits if h.message_id][0]
        lo, hi = hit.snippet.ranges[0]
        assert hit.snippet.text[lo:hi] == "checkpointing"

    def test_it_returns_offsets_not_markup(self, stocked):
        """A server that returns HTML has decided how a SwiftUI view
        should highlight, which it cannot use."""
        _, index, _ = stocked
        hit = index.search("vram", owner="alice").hits[0]
        assert "<" not in hit.snippet.text
        assert isinstance(hit.snippet.ranges[0], tuple)

    def test_it_says_when_it_cut_the_text(self, library):
        sessions, index, add = library
        add("alice", "long", [("user", "x" * 400 + " needle " + "y" * 400)])
        snippet = index.search("needle", owner="alice").hits[0].snippet
        assert snippet.truncated_start and snippet.truncated_end

    def test_overlapping_matches_are_merged(self, library):
        """Two terms in the same place is one highlight, not two."""
        sessions, index, add = library
        add("alice", "t", [("user", "checkpointing")])
        hit = index.search("checkpoint checkpointing", owner="alice").hits[0]
        assert len(hit.snippet.ranges) == 1

    def test_the_wire_form_is_json_safe(self, stocked):
        import json
        _, index, _ = stocked
        json.dumps(index.search("vram", owner="alice").to_dict())


class TestBounds:
    def test_the_result_says_when_it_did_not_search_everything(self, library):
        """A silent partial answer makes someone conclude the
        conversation is gone."""
        sessions, index, add = library
        add("alice", "t", [("user", "hello")])
        result = index.search("hello", owner="alice")
        assert result.capped is False
        assert result.scanned >= 1

    def test_the_limit_is_capped(self, stocked):
        _, index, _ = stocked
        assert len(index.search("vram", owner="alice", limit=10_000).hits) <= 200

    @pytest.mark.parametrize("limit", [0, -3])
    def test_a_nonsense_limit_still_returns_something(self, stocked, limit):
        _, index, _ = stocked
        assert len(index.search("vram", owner="alice", limit=limit).hits) == 1
