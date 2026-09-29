"""Tests for v0.51 chat infrastructure.

Covers:
- cookbook chat-template registry + for_model resolver
- menu system-prompt registry + persistence
- bell streaming primitive (with a stub oven that doesn't need real weights)
- countertop multi-turn session (incl. trim, persist, persona)
- flour logits processor (repetition penalty, no-repeat n-gram, role-leak
  suppression, decoded-text stop-sequence detection, clean_reply)
- hyper-Nix.2 wiring: KNOWN_MODELS entry, ARCH_PRESETS, default repo id
- CodeOven.repo_id propagates to _format_chat → cookbook fallback
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from hypernix import DEFAULT_REPO_ID, KNOWN_MODELS, bell, cookbook, countertop, flour, menu
from hypernix.bell import Bell, silent_bell, stdout_bell
from hypernix.cookbook import (
    COOKBOOK,
    HYPER_NIX_2,
    ChatTemplate,
    Cookbook,
    apply_template,
    for_model,
    list_templates,
)
from hypernix.countertop import Countertop
from hypernix.flour import ROLE_LEAK_MARKERS, Flour
from hypernix.menu import MENU, Menu

# ---------------------------------------------------------------------------
# cookbook
# ---------------------------------------------------------------------------

class TestCookbook:
    def test_chatml_renders_with_im_start_im_end(self) -> None:
        prompt = COOKBOOK.get("chatml").apply(
            [
                {"role": "system", "content": "you are terse"},
                {"role": "user", "content": "hi"},
            ],
            add_generation_prompt=True,
        )
        assert "<|im_start|>system\nyou are terse<|im_end|>" in prompt
        assert "<|im_start|>user\nhi<|im_end|>" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_hyper_nix_2_injects_default_system_when_absent(self) -> None:
        prompt = HYPER_NIX_2.apply(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
        )
        assert "You are HyperNix" in prompt

    def test_hyper_nix_2_keeps_caller_system_prompt(self) -> None:
        prompt = HYPER_NIX_2.apply(
            [
                {"role": "system", "content": "MY OWN SYSTEM"},
                {"role": "user", "content": "hi"},
            ],
        )
        assert "MY OWN SYSTEM" in prompt
        assert "You are HyperNix" not in prompt

    def test_llama3_uses_header_id_tags(self) -> None:
        prompt = COOKBOOK.get("llama3").apply(
            [{"role": "user", "content": "hi"}],
        )
        assert "<|start_header_id|>user<|end_header_id|>" in prompt
        assert "<|eot_id|>" in prompt
        assert prompt.startswith("<|begin_of_text|>")

    def test_llama2_uses_inst_tags(self) -> None:
        prompt = COOKBOOK.get("llama2").apply(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
        )
        assert "[INST]" in prompt
        assert "[/INST]" in prompt
        assert "<<SYS>>" in prompt

    def test_alpaca_and_vicuna_render(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        assert "### Instruction:" in COOKBOOK.get("alpaca").apply(msgs)
        assert "USER: hello" in COOKBOOK.get("vicuna").apply(msgs)

    def test_plain_fallback_works_for_any_role(self) -> None:
        prompt = COOKBOOK.get("plain").apply(
            [{"role": "user", "content": "hi"}],
        )
        assert "user: hi" in prompt
        assert prompt.endswith("assistant: ")

    def test_for_model_resolves_hyper_nix_2_repo_id(self) -> None:
        tmpl = for_model("ray0rf1re/hyper-Nix.2")
        assert tmpl.name == "hyper-nix.2"

    def test_for_model_resolves_llama_family(self) -> None:
        assert for_model("meta-llama/Llama-3.1-8B-Instruct").name == "llama3"
        assert for_model("meta-llama/Llama-2-7b-chat-hf").name == "llama2"

    def test_for_model_resolves_qwen_to_chatml(self) -> None:
        assert for_model("Qwen/Qwen2.5-7B-Instruct").name == "chatml"
        assert for_model("Qwen/Qwen3-8B").name == "chatml"

    def test_for_model_falls_back_to_plain_for_unknown(self) -> None:
        assert for_model("totally-unknown/model").name == "plain"

    def test_apply_template_one_shot_helper(self) -> None:
        prompt = apply_template(
            [{"role": "user", "content": "hi"}],
            template="chatml",
        )
        assert "<|im_start|>user" in prompt

    def test_apply_template_with_template_object(self) -> None:
        out = apply_template([{"role": "user", "content": "x"}], template=HYPER_NIX_2)
        assert "<|im_start|>" in out

    def test_register_custom_template(self) -> None:
        book = Cookbook.from_builtins()
        custom = ChatTemplate(
            name="my-bot",
            role_prefixes={"user": "Q: ", "assistant": "A: "},
            role_suffixes={"user": "\n", "assistant": "\n"},
            assistant_prefix="A: ",
        )
        book.add("my-bot", custom)
        assert book.get("my-bot") is custom
        # case-insensitive lookup
        assert book.get("MY-BOT") is custom

    def test_register_rejects_non_chat_template(self) -> None:
        book = Cookbook.from_builtins()
        with pytest.raises(TypeError):
            book.add("bad", "not a template")  # type: ignore[arg-type]

    def test_get_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            COOKBOOK.get("does-not-exist")

    def test_stop_tokens_include_eos(self) -> None:
        stops = COOKBOOK.get("chatml").stop_tokens()
        assert any("im_end" in s for s in stops)

    def test_list_templates_returns_names_and_notes(self) -> None:
        d = list_templates()
        assert "chatml" in d
        assert "hyper-nix.2" in d
        assert all(isinstance(v, str) for v in d.values())


# ---------------------------------------------------------------------------
# menu
# ---------------------------------------------------------------------------

class TestMenu:
    def test_builtins_present(self) -> None:
        for name in ("default", "code-helper", "judge", "creative", "chef"):
            assert name in MENU

    def test_default_returns_default_entry(self) -> None:
        assert "helpful assistant" in MENU.default().lower()

    def test_add_then_get(self) -> None:
        m = Menu.from_builtins()
        m.add("custom", "be a custom bot")
        assert m.get("custom") == "be a custom bot"

    def test_add_rejects_empty(self) -> None:
        m = Menu()
        with pytest.raises(ValueError):
            m.add("x", "   ")

    def test_unknown_get_raises(self) -> None:
        m = Menu()
        with pytest.raises(KeyError):
            m.get("nope")

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        m = Menu.from_builtins()
        m.add("alpha", "hello there")
        m.save(tmp_path / "menu.json")
        loaded = Menu.load(tmp_path / "menu.json")
        assert loaded.get("alpha") == "hello there"
        # Round-tripped JSON keeps all builtins.
        assert "code-helper" in loaded


# ---------------------------------------------------------------------------
# flour
# ---------------------------------------------------------------------------

class TestFlour:
    def test_smart_default_sets_repetition_penalty_and_template(self) -> None:
        f = Flour.smart_default(template="hyper-nix.2")
        assert f.repetition_penalty == pytest.approx(1.1)
        assert f.no_repeat_ngram == 4
        assert f.suppress_role_leaks is True
        assert f.template_name == "hyper-nix.2"

    def test_off_is_noop(self) -> None:
        f = Flour.off()
        logits = torch.zeros(8)
        out = f.process(logits, [1, 2, 3])
        assert torch.equal(out, logits)

    def test_repetition_penalty_demotes_seen_positive_logits(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        f = Flour(repetition_penalty=2.0)
        out = f.process(logits, [2])
        # Token 2 had logit 3.0 > 0, so it should be /= 2.0 = 1.5
        assert out[0, 2].item() == pytest.approx(1.5)
        # Other tokens unchanged
        assert out[0, 0].item() == 1.0

    def test_no_repeat_ngram_blocks_completion(self) -> None:
        f = Flour(no_repeat_ngram=3)
        # Sequence [1, 2, 3, 1, 2] — tail = (1, 2). The n-gram (1, 2, 3)
        # appeared at i=0, so token 3 should be banned.
        logits = torch.zeros(8)
        out = f.process(logits, [1, 2, 3, 1, 2])
        assert out[3].item() == float("-inf")
        assert out[5].item() == 0.0  # untouched

    def test_role_leak_suppression_adds_chatml_markers(self) -> None:
        f = Flour(suppress_role_leaks=True, template_name="hyper-nix.2")
        bad = f.effective_bad_words()
        assert "<|im_start|>" in bad
        # Stop sequences pick up the close marker.
        stops = f.effective_stop_sequences()
        assert any("im_end" in s for s in stops)

    def test_matched_stop_detects_decoded_text_suffix(self) -> None:
        f = Flour(stop_sequences=["<|im_end|>"])
        assert f.matched_stop("hello world<|im_end|>") == "<|im_end|>"
        assert f.matched_stop("hello world") is None

    def test_strip_stop_strips_trailing_whitespace_too(self) -> None:
        f = Flour(stop_sequences=["END"])
        assert f.strip_stop("hi   END") == "hi"

    def test_clean_reply_cuts_role_leak_marker(self) -> None:
        f = Flour(suppress_role_leaks=True, template_name="hyper-nix.2")
        cleaned = f.clean_reply("the answer is 42<|im_start|>user\nfollowup?")
        assert cleaned == "the answer is 42"

    def test_clean_reply_cuts_user_role_text_leak(self) -> None:
        f = Flour(suppress_role_leaks=True, template_name="plain")
        cleaned = f.clean_reply("answer\nuser: thanks!")
        assert cleaned == "answer"

    def test_factory_function_aggressive(self) -> None:
        f = flour.flour(aggressive=True, template="hyper-nix.2")
        assert f.repetition_penalty > 1.1
        assert f.frequency_penalty > 0


# ---------------------------------------------------------------------------
# bell
# ---------------------------------------------------------------------------

class _StubOven:
    """A toy oven that can drive Bell's streaming code path without
    loading any weights.  Tokens come from a hard-coded id sequence;
    ``model`` is a tiny nn.Module that always returns the same logits
    so we can assert the loop terminates and produces decoded tokens."""

    tokenizer_kind = "byte"
    tokenizer = None
    device = torch.device("cpu")
    repo_id = None

    def __init__(self, vocab: int = 16) -> None:
        self.vocab = vocab

        class _M(torch.nn.Module):
            def __init__(self_inner, vocab: int) -> None:
                super().__init__()
                self_inner.linear = torch.nn.Linear(vocab, vocab)

            def forward(self_inner, x: torch.Tensor):  # noqa: N805
                bsz, seq = x.shape
                onehot = torch.nn.functional.one_hot(x, num_classes=self.vocab).float()
                logits = self_inner.linear(onehot)
                # Wrapped to mirror the HF .logits attribute path.
                return type("O", (), {"logits": logits})()

        self.model = _M(vocab)
        self.model.eval()

    def _encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab for c in text] or [1]

    def _decode(self, ids: list[int]) -> str:
        return "".join(chr(int(i) + 65) for i in ids)

    def _format_chat(self, messages: list[dict[str, str]]) -> list[int]:
        return self._encode("|".join(m.get("content", "") for m in messages))


class TestBell:
    def test_iter_complete_yields_max_new_tokens(self) -> None:
        oven = _StubOven()
        b = Bell()
        toks = list(b.iter_complete(oven, "hi", max_new_tokens=5, temperature=0.0, seed=0))
        assert len(toks) == 5

    def test_iter_chat_routes_through_format_chat(self) -> None:
        oven = _StubOven()
        b = Bell()
        toks = list(b.iter_chat(
            oven,
            [{"role": "user", "content": "hi"}],
            max_new_tokens=3, temperature=0.0, seed=0,
        ))
        assert len(toks) == 3

    def test_token_callback_fires_per_token(self) -> None:
        oven = _StubOven()
        seen: list[tuple[str, int]] = []
        b = Bell()
        b.on_token(lambda tok, idx: seen.append((tok, idx)))
        b.stream_complete(oven, "hi", max_new_tokens=4, temperature=0.0, seed=0)
        assert [idx for _, idx in seen] == [0, 1, 2, 3]

    def test_done_callback_receives_full_reply(self) -> None:
        oven = _StubOven()
        full: list[str] = []
        b = Bell()
        b.on_done(lambda s: full.append(s))
        out = b.stream_complete(oven, "hi", max_new_tokens=4, temperature=0.0, seed=0)
        assert full == [out]

    def test_silent_and_stdout_factories(self) -> None:
        assert isinstance(silent_bell(), Bell)
        b = stdout_bell()
        assert isinstance(b, Bell)
        assert len(b.token_callbacks) == 1
        assert len(b.done_callbacks) == 1

    def test_flour_attached_to_bell_runs_during_streaming(self) -> None:
        oven = _StubOven()
        f = Flour(repetition_penalty=2.0)
        b = Bell(flour=f)
        toks = list(b.iter_complete(oven, "hi", max_new_tokens=5, temperature=0.0, seed=0))
        assert len(toks) == 5  # still works with flour active


# ---------------------------------------------------------------------------
# countertop
# ---------------------------------------------------------------------------

class _ChatStubOven(_StubOven):
    """A stub oven with a chat() method that captures the messages
    it was called with — Countertop only needs ``.chat()`` and (for
    template auto-resolve) ``.repo_id``."""

    repo_id = "ray0rf1re/hyper-Nix.2"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[dict[str, str]]] = []
        self.next_reply = "hello back"

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:  # noqa: ARG002
        self.calls.append(list(messages))
        return self.next_reply


class TestCountertop:
    def test_say_appends_user_and_assistant_turn(self) -> None:
        oven = _ChatStubOven()
        oven.next_reply = "hi there"
        ct = Countertop(oven=oven, system="be terse")
        reply = ct.say("hello")
        assert reply == "hi there"
        assert ct.history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

    def test_system_prompt_prepended_to_messages_sent_to_oven(self) -> None:
        oven = _ChatStubOven()
        ct = Countertop(oven=oven, system="ROOT_SYSTEM")
        ct.say("question")
        sent = oven.calls[0]
        assert sent[0] == {"role": "system", "content": "ROOT_SYSTEM"}
        assert sent[-1] == {"role": "user", "content": "question"}

    def test_template_auto_resolves_from_repo_id(self) -> None:
        oven = _ChatStubOven()  # repo_id = ray0rf1re/hyper-Nix.2
        ct = Countertop(oven=oven)
        rendered = ct.render()
        assert "<|im_start|>" in rendered

    def test_render_without_generation_prompt(self) -> None:
        oven = _ChatStubOven()
        ct = Countertop(oven=oven, template="chatml")
        ct.history.append({"role": "user", "content": "hi"})
        rendered = ct.render(add_generation_prompt=False)
        assert not rendered.endswith("<|im_start|>assistant\n")

    def test_reset_clears_history_but_keeps_system(self) -> None:
        oven = _ChatStubOven()
        ct = Countertop(oven=oven, system="ROOT")
        ct.say("a")
        ct.say("b")
        assert len(ct.history) == 4
        ct.reset()
        assert ct.history == []
        assert ct.system == "ROOT"

    def test_max_history_tokens_trims_oldest_pair(self) -> None:
        oven = _ChatStubOven()
        oven.next_reply = "x" * 200
        ct = Countertop(
            oven=oven,
            system="sys",
            template="plain",
            max_history_tokens=300,
        )
        for _ in range(5):
            ct.say("q" * 50)
        # After trimming, the rendered transcript fits within the budget.
        rendered = ct.render(add_generation_prompt=False)
        assert len(rendered) <= 300 + 200  # +last-reply slack

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        oven = _ChatStubOven()
        ct = Countertop(
            oven=oven,
            system="ROOT_SYS",
            template="chatml",
            sampling={"temperature": 0.5},
        )
        ct.say("hello")
        ct.save(tmp_path / "session.json")

        loaded = Countertop.load(tmp_path / "session.json", oven=oven)
        assert loaded.system == "ROOT_SYS"
        assert loaded.history == ct.history
        assert loaded.sampling == {"temperature": 0.5}
        # Persisted template name is the cookbook key.
        assert loaded._template_name() in ("chatml", None)

    def test_persona_factory_picks_menu_entry(self) -> None:
        oven = _ChatStubOven()
        ct = countertop.countertop(oven, persona="judge")
        assert "impartial judge" in (ct.system or "")

    def test_persona_and_system_conflict_raises(self) -> None:
        oven = _ChatStubOven()
        with pytest.raises(ValueError):
            countertop.countertop(oven, persona="judge", system="custom")

    def test_flour_cleans_reply_after_generation(self) -> None:
        oven = _ChatStubOven()
        oven.next_reply = "the answer<|im_start|>user\nignore me"
        f = Flour(suppress_role_leaks=True, template_name="hyper-nix.2")
        ct = Countertop(oven=oven, system="be terse", flour=f)
        reply = ct.say("hi")
        assert reply == "the answer"
        assert ct.history[-1]["content"] == "the answer"


# ---------------------------------------------------------------------------
# hyper-Nix.2 wiring
# ---------------------------------------------------------------------------

class TestHyperNix2Wiring:
    def test_default_repo_id_points_to_hyper_nix_2(self) -> None:
        assert DEFAULT_REPO_ID == "ray0rf1re/hyper-Nix.2"

    def test_known_models_has_hyper_nix_2_and_aliases(self) -> None:
        for key in ("hyper-nix.2", "hyper-nix2", "hypernix2", "hyper-nix", "hypernix"):
            assert key in KNOWN_MODELS
            assert KNOWN_MODELS[key].repo_id == "ray0rf1re/hyper-Nix.2"

    def test_arch_presets_has_hypernix2(self) -> None:
        from hypernix import ARCH_PRESETS
        assert "hypernix2" in ARCH_PRESETS
        assert ARCH_PRESETS["hypernix2"]["model_type"] == "hypernix"

    def test_oven_repo_id_routes_to_cookbook_when_no_hf_template(
        self, tmp_path: Path,
    ) -> None:
        from hypernix import new_oven
        oven = new_oven(
            tmp_path / "n2", arch="hypernix2",
            vocab_size=256, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, device="cpu", seed=0,
        )
        # new_oven doesn't set repo_id — set it here to simulate a
        # downloaded snapshot and verify the cookbook fallback path.
        oven.repo_id = "ray0rf1re/hyper-Nix.2"
        ids = oven._format_chat([{"role": "user", "content": "hi"}])
        decoded = oven._decode(ids)
        assert "<|im_start|>" in decoded


# ---------------------------------------------------------------------------
# Module-level smoke
# ---------------------------------------------------------------------------

def test_v051_modules_importable() -> None:
    for mod in (cookbook, countertop, menu, bell, flour):
        assert hasattr(mod, "__all__")


def test_role_leak_markers_table_covers_all_builtin_templates() -> None:
    for name in ("chatml", "hyper-nix.2", "llama3", "llama2", "alpaca", "vicuna", "plain"):
        assert name in ROLE_LEAK_MARKERS
        assert len(ROLE_LEAK_MARKERS[name]) >= 2
