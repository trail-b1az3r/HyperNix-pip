"""Settings: who you are, and how you want to be answered.

These live on the server rather than the phone for two reasons, and the
second is the one that decides it. A person with a phone and a tablet is
one person, so device-local settings mean two different answers to "how
do you want to be answered". And these are *inputs to generation* — the
system prompt, the effort level and the context bounds all have to be in
the process that builds the request.

Most of this file is about the clamps, because a settings screen that
silently stores something other than what you typed is one nobody can
trust afterwards. A context maximum of four million is not a preference:
it is a number that makes every reply fail with an out-of-memory two
minutes later and somewhere unrelated, so it reads as the model being
broken. It is lowered — and *said*.
"""
from __future__ import annotations

import pytest
from conftest import clear_t1_config

from hypernix.hyperlink.preferences import (
    CONTEXT_CEILING,
    CONTEXT_FLOOR,
    DEFAULT_EFFORT,
    EFFORT_LEVELS,
    MAX_BIO,
    MAX_SYSTEM_PROMPT,
    PreferenceStore,
    sampling_for,
    system_prompt_for,
)
from hypernix.t1api.errors import T1APIError


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
    return PreferenceStore()


class TestDefaults:
    def test_somebody_who_has_set_nothing_gets_working_defaults(self, store):
        settings = store.get(owner="me")
        assert settings.effort == DEFAULT_EFFORT
        assert settings.auto_memory is True
        assert settings.tools_enabled is False

    def test_reading_does_not_create_a_row(self, store):
        """A GET that writes is a database that grows every time an
        unauthorised caller guesses an owner name."""
        store.get(owner="me")
        with store.backend.connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM hyperlink_preferences"
            ).fetchone()["n"]
        assert count == 0

    def test_tools_are_off_until_asked_for(self, store):
        """Letting a model write files on somebody's machine is not a
        default."""
        assert store.get(owner="me").tools_enabled is False


class TestPartialUpdates:
    def test_unsent_fields_are_left_alone(self, store):
        """The reason this is a PATCH and not a PUT: a client that knows
        about six settings must not blank the four it has never heard of
        by sending them as defaults."""
        store.save(owner="me", display_name="Mason", effort="high")
        store.save(owner="me", bio="Runs a home lab.")
        settings = store.get(owner="me")
        assert settings.display_name == "Mason"
        assert settings.effort == "high"
        assert settings.bio == "Runs a home lab."

    def test_an_empty_string_does_clear_a_field(self, store):
        """Sending "" is a decision; not sending the key is not. The two
        have to mean different things or nothing can ever be cleared."""
        store.save(owner="me", bio="something")
        store.save(owner="me", bio="")
        assert store.get(owner="me").bio == ""

    def test_two_owners_do_not_share(self, store):
        store.save(owner="me", display_name="Mason")
        store.save(owner="you", display_name="Someone else")
        assert store.get(owner="me").display_name == "Mason"


class TestTheClampsAreReported:
    def test_a_context_maximum_past_the_ceiling_is_lowered(self, store):
        settings, notes = store.save(owner="me", context_maximum=9_999_999)
        assert settings.context_maximum == CONTEXT_CEILING
        assert any("lowered" in note for note in notes)

    def test_a_context_minimum_below_the_floor_is_raised(self, store):
        """Below the floor, a system prompt plus two memories is the
        whole window and the conversation has nowhere to go — which
        reads as the model forgetting everything instantly."""
        settings, notes = store.save(owner="me", context_minimum=10)
        assert settings.context_minimum == CONTEXT_FLOOR
        assert any("raised" in note for note in notes)

    def test_zero_means_no_opinion_and_is_not_clamped(self, store):
        """0 is the default and means "use whatever the model says it
        can do", which is what most people should be doing."""
        settings, notes = store.save(owner="me", context_minimum=0, context_maximum=0)
        assert settings.context_minimum == 0
        assert settings.context_maximum == 0
        assert notes == []

    def test_a_backwards_range_is_swapped_not_refused(self, store):
        """Somebody who typed them the wrong way round meant the range.
        A 422 here is a settings screen that will not save."""
        settings, notes = store.save(
            owner="me", context_minimum=8000, context_maximum=2000
        )
        assert settings.context_minimum == 2000
        assert settings.context_maximum == 8000
        assert any("swapped" in note for note in notes)

    def test_an_over_long_bio_is_cut_and_said(self, store):
        settings, notes = store.save(owner="me", bio="x" * (MAX_BIO + 500))
        assert len(settings.bio) == MAX_BIO
        assert any("bio" in note.lower() for note in notes)

    def test_a_massive_system_prompt_is_allowed(self, store):
        """The request asked for a massive one. 32,000 characters is
        roughly 8,000 words — more instruction than most models can
        follow, and comfortably more than anybody writes by hand."""
        prompt = "Follow these rules. " * 1000
        assert len(prompt) < MAX_SYSTEM_PROMPT
        settings, notes = store.save(owner="me", system_prompt=prompt)
        assert settings.system_prompt == prompt
        assert notes == []

    def test_nothing_valid_produces_a_note(self, store):
        _settings, notes = store.save(owner="me", display_name="Mason", effort="low")
        assert notes == []


class TestWhatItRefuses:
    def test_an_unknown_effort_level(self, store):
        with pytest.raises(T1APIError) as refused:
            store.save(owner="me", effort="ludicrous")
        assert "minimal" in str(refused.value)

    def test_a_negative_context(self, store):
        with pytest.raises(T1APIError):
            store.save(owner="me", context_maximum=-1)

    def test_context_that_is_not_a_number(self, store):
        with pytest.raises(T1APIError):
            store.save(owner="me", context_minimum="lots")

    def test_no_owner(self, store):
        with pytest.raises(T1APIError):
            store.get(owner="")


class TestEffortLevels:
    @pytest.mark.parametrize("level", EFFORT_LEVELS)
    def test_every_level_has_sampling(self, level):
        settings = sampling_for(level)
        assert settings["temperature"] is not None
        assert settings["max_tokens"] > 0

    def test_higher_effort_allows_a_longer_answer(self):
        assert (
            sampling_for("maximum")["max_tokens"]
            > sampling_for("minimal")["max_tokens"]
        )

    def test_minimal_is_deterministic(self):
        """"Minimal effort" and "creative" are different requests."""
        assert sampling_for("minimal")["temperature"] == 0.0

    def test_the_level_is_passed_through_by_name(self):
        """A backend with a real reasoning-effort control gets the level
        itself. One without gets the approximation above — which is why
        both are always set."""
        assert sampling_for("high")["reasoning_effort"] == "high"

    def test_an_unknown_level_falls_back_rather_than_raising(self):
        """This runs inside a chat turn. A row written by an older build
        must not make every reply fail."""
        assert sampling_for("nonsense") == sampling_for(DEFAULT_EFFORT)


class TestTheComposedPrompt:
    def test_the_order_is_widest_scope_first(self, store):
        """Who the person is, how they want to be answered generally,
        what this conversation is for, then what is known about them."""
        settings, _ = store.save(
            owner="me", display_name="Mason", bio="Runs a home lab.",
            system_prompt="Be concise.",
        )
        composed = system_prompt_for(
            settings, session_prompt="Answer in British English.",
            memory_block="Known: prefers Vulkan.",
        )
        assert composed.index("Mason") < composed.index("Be concise.")
        assert composed.index("Be concise.") < composed.index("British English")
        assert composed.index("British English") < composed.index("prefers Vulkan")

    def test_the_session_prompt_comes_after_the_global_one(self, store):
        """So a conversation can override the default rather than fight
        it: the later instruction is the one a model follows when two
        conflict."""
        settings, _ = store.save(owner="me", system_prompt="Always answer in English.")
        composed = system_prompt_for(settings, session_prompt="Answer in French.")
        assert composed.index("English") < composed.index("French")

    def test_nothing_set_composes_to_nothing(self, store):
        """An empty system message is worse than none: it is a turn of
        context spent saying nothing."""
        assert system_prompt_for(store.get(owner="me")) == ""

    def test_a_name_with_no_bio_still_says_who(self, store):
        settings, _ = store.save(owner="me", display_name="Mason")
        assert "Mason" in system_prompt_for(settings)

    def test_a_bio_with_no_name_still_works(self, store):
        settings, _ = store.save(owner="me", bio="Runs a home lab.")
        assert "home lab" in system_prompt_for(settings)


class TestReset:
    def test_it_goes_back_to_defaults(self, store):
        store.save(owner="me", display_name="Mason", effort="maximum")
        store.reset(owner="me")
        assert store.get(owner="me").effort == DEFAULT_EFFORT
        assert store.get(owner="me").display_name == ""

    def test_it_deletes_the_row_rather_than_blanking_it(self, store):
        """"Never set anything" and "set everything back" should be the
        same state, which is what somebody pressing Reset means."""
        store.save(owner="me", display_name="Mason")
        store.reset(owner="me")
        with store.backend.connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM hyperlink_preferences"
            ).fetchone()["n"]
        assert count == 0

    def test_it_leaves_other_owners_alone(self, store):
        store.save(owner="me", display_name="Mason")
        store.save(owner="you", display_name="Someone")
        store.reset(owner="me")
        assert store.get(owner="you").display_name == "Someone"


class TestThroughTheAPI:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        clear_t1_config(monkeypatch)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")

        from hypernix.t1api.app import create_app

        return TestClient(create_app(), client=("192.168.1.9", 5000))

    def test_reading_works_before_anything_is_set(self, client):
        response = client.get("/hyperlink/preferences")
        assert response.status_code == 200
        assert response.json()["preferences"]["effort"] == DEFAULT_EFFORT

    def test_the_bounds_come_back_with_the_values(self, client):
        """So the app does not carry its own copy of a list the server
        owns. An effort level the phone offers and the server rejects is
        a settings screen that cannot save."""
        body = client.get("/hyperlink/preferences").json()
        assert body["effort_levels"] == list(EFFORT_LEVELS)
        assert body["context_ceiling"] == CONTEXT_CEILING
        assert body["max_system_prompt"] == MAX_SYSTEM_PROMPT

    def test_a_patch_only_changes_what_it_sends(self, client):
        client.patch("/hyperlink/preferences", json={"display_name": "Mason"})
        client.patch("/hyperlink/preferences", json={"effort": "high"})
        settings = client.get("/hyperlink/preferences").json()["preferences"]
        assert settings["display_name"] == "Mason"
        assert settings["effort"] == "high"

    def test_the_clamp_notes_reach_the_client(self, client):
        body = client.patch(
            "/hyperlink/preferences", json={"context_maximum": 9_999_999}
        ).json()
        assert body["notes"]
        assert body["preferences"]["context_maximum"] == CONTEXT_CEILING

    def test_a_bad_effort_is_a_422(self, client):
        assert client.patch(
            "/hyperlink/preferences", json={"effort": "ludicrous"}
        ).status_code == 422

    def test_reset_works(self, client):
        client.patch("/hyperlink/preferences", json={"effort": "maximum"})
        body = client.post("/hyperlink/preferences/reset").json()
        assert body["preferences"]["effort"] == DEFAULT_EFFORT
