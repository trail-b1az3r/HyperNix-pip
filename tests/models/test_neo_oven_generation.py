"""Generation-path regressions in neo_oven.

Four things were wrong, and each one is pinned here by the property it
broke rather than by an output string — an untrained model emits noise, so
asserting on text would test the RNG, not the fix:

* ``stream()`` decoded every token on its own, which splits multi-byte
  UTF-8 characters across chunks. "café" streamed as ``caf`` + two
  replacement characters.
* ``stream()`` ignored stop sequences and had no ``seed``, so the streamed
  answer and the non-streamed one for a prompt were different text.
* ``fill()`` passed no ``eos_ids``, so it could never stop early — every
  call ran the full ``max_new_tokens``.
* ``max_position_embeddings`` was read unguarded, turning a loadable model
  into an AttributeError at the first generated token.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hypernix.models.neo_oven import (  # noqa: E402
    DEFAULT_STOPS,
    FIM_STOPS,
    NeoOven,
    new_oven,
)


@pytest.fixture(scope="module")
def oven(tmp_path_factory) -> NeoOven:
    """A tiny untrained model. Fast, and its output is deliberately noise."""
    out = tmp_path_factory.mktemp("neo-oven")
    return new_oven(
        out, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, vocab_size=256, max_position_embeddings=256, seed=0,
    )


# ---------------------------------------------------------------------------
# Incremental decoding
# ---------------------------------------------------------------------------


class TestIncrementalDecoding:
    @pytest.mark.parametrize("text", [
        "café",
        "a → b",
        "🚀 launch",
        "naïve café → 🚀",
        "ascii only",
    ])
    def test_incremental_decode_reconstructs_the_text_exactly(self, oven, text):
        """The rule stream() applies, checked against the tokenizer directly
        so it doesn't depend on what the model happens to sample."""
        ids = oven._encode(text)
        emitted, chunks = "", []
        for n in range(1, len(ids) + 1):
            stable = NeoOven._stable_prefix(oven._decode(ids[:n]))
            if stable.startswith(emitted) and len(stable) > len(emitted):
                chunks.append(stable[len(emitted):])
                emitted = stable
        assert "".join(chunks) == oven._decode(ids)

    def test_per_token_decoding_is_what_was_broken(self, oven):
        """The old behaviour, so the regression is legible: decoding tokens
        one at a time mangles anything that isn't ASCII."""
        ids = oven._encode("café")
        assert "".join(oven._decode([i]) for i in ids) != "café"

    def test_stable_prefix_holds_back_a_partial_character(self):
        assert NeoOven._stable_prefix("caf�") == "caf"
        assert NeoOven._stable_prefix("caf��") == "caf"

    def test_stable_prefix_leaves_complete_text_alone(self):
        assert NeoOven._stable_prefix("café → 🚀") == "café → 🚀"

    def test_stable_prefix_only_trims_the_tail(self):
        """A replacement character in the middle is genuinely part of the
        text — the bytes around it already arrived."""
        assert NeoOven._stable_prefix("a�b") == "a�b"


# ---------------------------------------------------------------------------
# stream() agrees with complete()
# ---------------------------------------------------------------------------


class TestStreamMatchesComplete:
    @pytest.mark.parametrize("seed", [7, 11, 42])
    def test_joined_stream_equals_complete(self, oven, seed):
        streamed = "".join(oven.stream("def f():", max_new_tokens=64,
                                       temperature=0.9, seed=seed))
        whole = oven.complete("def f():", max_new_tokens=64,
                              temperature=0.9, seed=seed)
        assert streamed == whole

    def test_stream_is_reproducible_for_a_seed(self, oven):
        first = "".join(oven.stream("x =", max_new_tokens=32, temperature=0.9, seed=3))
        second = "".join(oven.stream("x =", max_new_tokens=32, temperature=0.9, seed=3))
        assert first == second

    def test_stream_accepts_a_seed_at_all(self):
        """It didn't, which is why it couldn't agree with complete()."""
        import inspect

        assert "seed" in inspect.signature(NeoOven.stream).parameters

    def test_stream_honours_stop_sequences(self, oven, monkeypatch):
        """Forced through a fixed decode so the assertion is about the stop
        logic, not about what an untrained model samples."""
        monkeypatch.setattr(NeoOven, "_decode",
                            lambda self, ids: "abc\nclass Foo:\ndef bar():"[:len(ids)])
        out = "".join(oven.stream("p", max_new_tokens=40, temperature=0.9,
                                  seed=1, stop=("\nclass ",)))
        assert "\nclass " not in out
        assert out == "abc"

    def test_stream_emits_nothing_past_a_stop_at_position_zero(self, oven, monkeypatch):
        monkeypatch.setattr(NeoOven, "_decode", lambda self, ids: "</s>tail"[:len(ids)])
        assert "".join(oven.stream("p", max_new_tokens=10, seed=1, stop=("</s>",))) == ""


# ---------------------------------------------------------------------------
# EOS handling
# ---------------------------------------------------------------------------


class TestEosHandling:
    def test_eos_is_not_part_of_the_output(self, oven, monkeypatch):
        """EOS terminates a sequence; it isn't a token in it. This only ever
        looked correct because HF decode is asked to skip special tokens."""
        monkeypatch.setattr(NeoOven, "_eos_ids", lambda self: (5,))
        seen: list[list[int]] = []
        monkeypatch.setattr(NeoOven, "_decode",
                            lambda self, ids: seen.append(list(ids)) or "")
        # Force the sampler to produce EOS on the third token.
        import hypernix.models.neo_oven as mod

        calls = {"n": 0}

        def fake_sample(logits, **_kw):
            calls["n"] += 1
            return torch.tensor(5 if calls["n"] == 3 else 1)

        monkeypatch.setattr(mod, "_sample_next", fake_sample)
        oven.complete("p", max_new_tokens=10, seed=1)
        assert seen and 5 not in seen[-1]
        assert seen[-1] == [1, 1]

    def test_fill_stops_at_eos(self, oven, monkeypatch):
        """fill() passed no eos_ids, so it always ran the full budget."""
        import hypernix.models.neo_oven as mod

        monkeypatch.setattr(NeoOven, "_eos_ids", lambda self: (5,))
        monkeypatch.setattr(NeoOven, "_decode", lambda self, ids: "x" * len(ids))
        calls = {"n": 0}

        def fake_sample(logits, **_kw):
            calls["n"] += 1
            return torch.tensor(5 if calls["n"] == 4 else 1)

        monkeypatch.setattr(mod, "_sample_next", fake_sample)
        out = oven.fill("pre", "suf", max_new_tokens=200, seed=1)
        assert calls["n"] == 4, "generation should have stopped at the EOS token"
        assert out == "xxx"

    def test_fill_trims_at_fim_stops(self, oven, monkeypatch):
        monkeypatch.setattr(NeoOven, "_decode",
                            lambda self, ids: "body<fim_suffix>trailing")
        out = oven.fill("pre", "suf", max_new_tokens=4, seed=1)
        assert out == "body"

    def test_fim_stops_differ_from_completion_stops(self):
        """"\\ndef " is a normal thing to generate when filling a hole in
        existing code, so FIM must not stop on it."""
        assert "\ndef " in DEFAULT_STOPS
        assert "\ndef " not in FIM_STOPS


# ---------------------------------------------------------------------------
# Cooperative stop
# ---------------------------------------------------------------------------


class TestShouldStop:
    def test_complete_stops_when_asked(self, oven):
        calls = {"n": 0}

        def stop_after_five():
            calls["n"] += 1
            return calls["n"] > 5

        oven.complete("p", max_new_tokens=500, temperature=0.9,
                      seed=1, should_stop=stop_after_five)
        assert calls["n"] <= 7, "the loop kept running after should_stop returned True"

    def test_partial_output_is_returned_not_discarded(self, oven, monkeypatch):
        monkeypatch.setattr(NeoOven, "_decode", lambda self, ids: "y" * len(ids))
        calls = {"n": 0}

        def stop_after_three():
            calls["n"] += 1
            return calls["n"] > 3

        out = oven.complete("p", max_new_tokens=500, seed=1,
                            should_stop=stop_after_three, stop=())
        assert out == "yyy"

    def test_stream_stops_when_asked(self, oven):
        calls = {"n": 0}

        def stop_after_four():
            calls["n"] += 1
            return calls["n"] > 4

        chunks = list(oven.stream("p", max_new_tokens=500, temperature=0.9,
                                  seed=1, should_stop=stop_after_four))
        assert calls["n"] <= 6
        assert len(chunks) <= 5

    def test_a_stop_that_never_fires_changes_nothing(self, oven):
        with_hook = oven.complete("p", max_new_tokens=24, temperature=0.9,
                                  seed=5, should_stop=lambda: False)
        without = oven.complete("p", max_new_tokens=24, temperature=0.9, seed=5)
        assert with_hook == without

    def test_chat_accepts_a_stop_hook(self, oven):
        calls = {"n": 0}

        def stop_after_two():
            calls["n"] += 1
            return calls["n"] > 2

        oven.chat([{"role": "user", "content": "hi"}], max_new_tokens=500,
                  seed=1, should_stop=stop_after_two)
        assert calls["n"] <= 4

    def test_generate_batch_stops_mid_batch(self, oven, monkeypatch):
        monkeypatch.setattr(NeoOven, "_decode", lambda self, ids: "z" * len(ids))
        state = {"done": 0}

        def stop_after_first_prompt():
            return state["done"] >= 1

        original = NeoOven.complete

        def counting_complete(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            state["done"] += 1
            return result

        monkeypatch.setattr(NeoOven, "complete", counting_complete)
        out = oven.generate_batch(["a", "b", "c"], max_new_tokens=4, seed=1,
                                  should_stop=stop_after_first_prompt)
        assert len(out) == 3
        assert out[0]
        assert out[1] == "" and out[2] == ""


# ---------------------------------------------------------------------------
# Config robustness
# ---------------------------------------------------------------------------


class TestMaxContext:
    def test_reads_the_configured_limit(self, oven):
        assert oven._max_context() == 256

    def test_falls_back_when_the_config_lacks_the_field(self, oven, monkeypatch):
        """Some architectures' configs don't carry max_position_embeddings.
        Reading it unguarded failed at the first generated token."""

        class Bare:
            pass

        monkeypatch.setattr(oven.model, "config", Bare())
        assert oven._max_context() == 2048

    def test_falls_back_on_a_zero_or_none_value(self, oven, monkeypatch):
        class Zeroed:
            max_position_embeddings = 0

        monkeypatch.setattr(oven.model, "config", Zeroed())
        assert oven._max_context() == 2048


# ---------------------------------------------------------------------------
# Prompt preparation is shared
# ---------------------------------------------------------------------------


class TestPromptIds:
    def test_empty_prompt_still_produces_a_token(self, oven):
        assert oven._prompt_ids("") == [0]

    def test_byte_tokenizer_gets_no_bos(self, oven):
        assert oven._prompt_ids("hi") == oven._encode("hi")

    def test_hf_tokenizer_gets_exactly_one_bos(self, oven, monkeypatch):
        class FakeTok:
            bos_token_id = 9

            def encode(self, text, add_special_tokens=False):
                return [9, 1, 2] if text == "with-bos" else [1, 2]

        monkeypatch.setattr(oven, "tokenizer", FakeTok())
        monkeypatch.setattr(oven, "tokenizer_kind", "hf")
        assert oven._prompt_ids("plain") == [9, 1, 2]
        assert oven._prompt_ids("with-bos") == [9, 1, 2]


class TestStopMarkerHoldback:
    """A stop sequence arrives one character at a time; emitting eagerly
    leaks its prefix into the stream before it is recognisable."""

    def test_holds_back_a_partial_marker(self):
        assert NeoOven._emit_safe_length("abc\nclas", ("\nclass ",)) == 3

    def test_emits_everything_when_no_marker_is_forming(self):
        assert NeoOven._emit_safe_length("abcdef", ("\nclass ",)) == 6

    def test_holds_back_the_longest_candidate(self):
        assert NeoOven._emit_safe_length("xx\nd", ("\ndef ", "\nd")) == 2

    def test_no_stops_holds_nothing_back(self):
        assert NeoOven._emit_safe_length("anything", ()) == 8

    def test_a_complete_marker_is_not_held_back_here(self):
        """A finished marker is the early-return path's job, not this one's."""
        assert NeoOven._emit_safe_length("ab</s>", ("</s>",)) == 6
