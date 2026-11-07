"""hypernix.runtime.relay: long prompts read by small-context instances.

The instances here are fakes that record what they were sent and check
it fits the context they were given -- the one promise a relay makes.
"""
from __future__ import annotations

import threading

import pytest

from hypernix.runtime import relay
from hypernix.runtime.relay import CHAIN, MAP, ContextTooSmall, Relay, Reply, estimate_tokens


def _document(paragraphs: int, words: int = 60) -> str:
    return "\n\n".join(
        " ".join(f"p{i}w{j}" for j in range(words)) + "." for i in range(paragraphs)
    )


class Fake:
    """An instance that notes which parts it has seen and refuses overflow."""

    def __init__(self, name: int, context: int, calls: list):
        self.name, self.context, self.calls = name, context, calls

    def __call__(self, messages, max_tokens):
        text = "".join(m["content"] for m in messages)
        used = estimate_tokens(text) + relay.OVERHEAD_TOKENS // 2 + max_tokens
        assert used <= self.context, f"instance {self.name} sent {used} tokens for a {self.context} context"
        user = messages[-1]["content"]
        seen = sorted({word.split("w")[0] for word in user.split() if word.startswith("p") and "w" in word})
        self.calls.append((self.name, threading.current_thread().name, user))
        if messages[0]["content"] == relay.ANSWER_PROMPT:
            return Reply(f"answer from {user.count('p')} mentions", estimate_tokens(text), 5)
        return Reply(" ".join(seen) or "nothing", estimate_tokens(text), len(seen) or 1)


def _fakes(n: int, context: int) -> tuple[list[Fake], list]:
    calls: list = []
    return [Fake(i, context, calls) for i in range(n)], calls


class TestSplit:
    def test_parts_are_within_budget_and_lose_nothing(self):
        text = _document(30)
        parts = relay.split(text, 200)
        assert len(parts) > 1
        assert all(estimate_tokens(p) <= 200 for p in parts)
        assert " ".join(" ".join(parts).split()) == " ".join(text.split())

    def test_paragraphs_are_kept_whole_where_they_fit(self):
        parts = relay.split(_document(6, words=10), 1000)
        assert parts == [_document(6, words=10)]

    def test_a_giant_word_is_cut_anyway(self):
        parts = relay.split("x" * 1000, 50)
        assert all(len(p) <= 200 for p in parts) and "".join(parts) == "x" * 1000


class TestPlan:
    def test_a_prompt_that_fits_is_one_call(self):
        plan = relay.plan("short document", "what?", context_tokens=4096)
        assert plan.single and plan.requests == 1 and plan.overhead_factor == 1.0

    def test_a_long_prompt_is_cut_to_the_context(self):
        doc = _document(80)
        plan = relay.plan(doc, "what is p3?", context_tokens=1024, instances=3)
        assert not plan.single
        assert plan.requests == len(plan.chunks) + 1
        assert plan.largest_call_tokens <= 1024
        assert plan.overhead_factor > 1.0
        assert plan.direct_input_tokens >= estimate_tokens(doc)

    def test_map_reads_no_notes_so_its_parts_are_larger(self):
        doc = _document(80)
        chain = relay.plan(doc, "q", context_tokens=1024, mode=CHAIN)
        mapped = relay.plan(doc, "q", context_tokens=1024, mode=MAP)
        assert mapped.chunk_tokens > chain.chunk_tokens
        assert len(mapped.chunks) <= len(chain.chunks)

    def test_a_context_too_small_says_so(self):
        with pytest.raises(ContextTooSmall, match="larger context"):
            relay.plan(_document(50), "q", context_tokens=300)

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            relay.plan("x", "q", context_tokens=1024, mode="sideways")
        with pytest.raises(ValueError):
            relay.plan("x", "q", context_tokens=1024, instances=0)


class TestChain:
    def test_every_hop_fits_and_the_instances_take_turns_in_order(self):
        instances, calls = _fakes(3, 1024)
        result = Relay(instances, context_tokens=1024).run(_document(60), "what is in p59?")
        plan = result.plan
        assert [h.kind for h in result.hops] == ["read"] * len(plan.chunks) + ["answer"]
        assert [h.part for h in result.hops[:-1]] == list(range(len(plan.chunks)))
        assert [h.instance for h in result.hops] == [i % 3 for i in range(len(result.hops))]
        assert [c[0] for c in calls] == [h.instance for h in result.hops]
        assert result.answer.startswith("answer from")

    def test_notes_are_carried_from_hop_to_hop(self):
        instances, calls = _fakes(2, 1024)
        Relay(instances, context_tokens=1024).run(_document(60), "q")
        reads = [user for _, _, user in calls[:-1]]
        assert "(none yet)" in reads[0]
        # Hop 2 is shown what hop 1 wrote: the paragraphs it had seen.
        first_seen = calls[0][2].split("Part 1")[1].split()[3].split("w")[0]
        assert first_seen in reads[1].split("Part 2")[0]
        assert "Notes:" in calls[-1][2]

    def test_usage_is_summed_and_reported_per_hop(self):
        instances, _ = _fakes(2, 1024)
        seen = []
        result = Relay(instances, context_tokens=1024, on_hop=seen.append).run(_document(40), "q")
        assert seen == result.hops
        assert result.input_tokens == sum(h.input_tokens for h in result.hops) > 0
        assert result.output_tokens == sum(h.output_tokens for h in result.hops) > 0
        assert result.requests == len(result.hops) == result.plan.requests
        assert result.input_tokens <= result.plan.input_tokens
        body = result.to_dict()
        assert body["plan"]["chunks"] == len(result.plan.chunks) and len(body["hops"]) == result.requests

    def test_a_short_prompt_is_one_direct_call(self):
        instances, calls = _fakes(4, 4096)
        result = Relay(instances, context_tokens=4096).run("tiny doc", "q?")
        assert [h.kind for h in result.hops] == ["direct"] and len(calls) == 1
        assert calls[0][2] == "tiny doc\n\nq?"

    def test_over_long_notes_are_clipped_to_the_plan(self):
        def chatty(messages, max_tokens):
            return Reply("z" * 100_000, 10, max_tokens)

        sent: list[int] = []

        def counting(messages, max_tokens):
            sent.append(estimate_tokens("".join(m["content"] for m in messages)))
            return chatty(messages, max_tokens)

        result = Relay([counting], context_tokens=1024).run(_document(40), "q")
        assert max(sent) + result.plan.notes_tokens <= 1024
        assert len(result.notes) <= result.plan.notes_tokens * relay.CHARS_PER_TOKEN


class TestChatMessages:
    def test_a_short_chat_is_sent_unchanged_and_keeps_its_envelope(self):
        messages = [
            {"role": "system", "content": "Answer accurately."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is in this picture?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]},
            {"role": "assistant", "content": "I see a tree."},
            {"role": "user", "content": "What color are its leaves?"},
        ]
        envelope = {
            "model": "small",
            "choices": [{"message": {"role": "assistant", "content": "Green."},
                         "finish_reason": "stop"}],
        }
        sent = []

        def answer(call_messages, max_tokens):
            sent.append(call_messages)
            return Reply("Green.", 23, 2, envelope)

        result = Relay([answer], context_tokens=4096).run_messages(messages)
        assert sent == [messages]
        assert result.plan.single
        assert result.envelope is envelope
        assert result.answer == "Green."

    def test_a_long_chat_keeps_instructions_roles_and_latest_image(self):
        messages = [
            {"role": "system", "content": "Use the conversation as context. Be precise."},
            {"role": "user", "content": "Earlier user detail. " + "alpha " * 160},
            {"role": "assistant", "content": "Earlier assistant response. " + "beta " * 140},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image in light of our earlier discussion."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,latest"}},
            ]},
        ]
        calls = []

        def instance(call_messages, max_tokens):
            calls.append(call_messages)
            system = call_messages[0]["content"]
            if system.endswith(relay.READ_PROMPT):
                return Reply("Earlier context preserved.", 40, 8)
            raw = {
                "model": "small",
                "choices": [{"message": {"role": "assistant", "content": "It is green."},
                             "finish_reason": "stop"}],
            }
            return Reply("It is green.", 45, 4, raw)

        result = Relay(
            [instance], context_tokens=512, notes_tokens=40, answer_tokens=48
        ).run_messages(messages)

        assert not result.plan.single
        assert [hop.kind for hop in result.hops] == ["read"] * len(result.plan.chunks) + ["answer"]
        read_calls = calls[:-1]
        assert all(call[0]["content"].startswith("[system]\nUse the conversation") for call in read_calls)
        assert all(call[0]["content"].endswith(relay.READ_PROMPT) for call in read_calls)
        read_input = read_calls[0][-1]["content"]
        assert "USER:\nEarlier user detail." in read_input
        assert "ASSISTANT:\nEarlier assistant response." in read_input
        final_content = calls[-1][-1]["content"]
        assert isinstance(final_content, list)
        assert final_content[-1] == messages[-1]["content"][-1]
        assert result.envelope["choices"][0]["message"]["content"] == "It is green."

    def test_system_only_messages_are_rejected(self):
        with pytest.raises(ValueError, match="user or conversation"):
            Relay([lambda messages, max_tokens: Reply("ok")],
                  context_tokens=1024).run_messages([
                {"role": "system", "content": "instructions only"}
            ])


class TestMap:
    def test_parts_are_digested_on_every_instance_then_answered(self):
        instances, calls = _fakes(3, 1024)
        result = Relay(instances, context_tokens=1024, mode=MAP).run(_document(60), "q")
        digests = [h for h in result.hops if h.kind == "digest"]
        assert sorted(h.part for h in digests) == list(range(len(result.plan.chunks)))
        # Part i goes to instance i % N, so each copy reads one at a time.
        assert all(h.instance == h.part % 3 for h in digests)
        assert {c[1] for c in calls if c[2].startswith("Question")} >= {"relay_0", "relay_1", "relay_2"}
        assert result.hops[-1].kind == "answer"

    def test_many_digests_are_folded_in_series(self):
        instances, _ = _fakes(2, 1024)

        def wordy(inst):
            def call(messages, max_tokens):
                reply = inst(messages, max_tokens)
                if messages[0]["content"] == relay.DIGEST_PROMPT:
                    return Reply("d" * (max_tokens * relay.CHARS_PER_TOKEN), reply.input_tokens, max_tokens)
                return reply
            return call

        result = Relay([wordy(i) for i in instances], context_tokens=1024, mode=MAP).run(_document(120), "q")
        kinds = [h.kind for h in result.hops]
        assert "fold" in kinds and kinds[-1] == "answer"
        assert kinds.index("fold") > max(i for i, k in enumerate(kinds) if k == "digest")


def test_openai_instance_reads_usage_or_estimates_it():
    class Client:
        def __init__(self, usage):
            self.usage, self.sent = usage, []

        def chat(self, messages, **kwargs):
            self.sent.append(kwargs)
            return {"choices": [{"message": {"content": "hi there"}}], "usage": self.usage}

    reported = relay.openai_instance(Client({"prompt_tokens": 7, "completion_tokens": 2}), "m")
    assert reported([{"role": "user", "content": "x"}], 50) == Reply("hi there", 7, 2)
    silent_client = Client({})
    silent = relay.openai_instance(silent_client, "m")
    reply = silent([{"role": "user", "content": "x" * 40}], 50)
    assert reply.input_tokens == 10 and reply.output_tokens == 2
    assert silent_client.sent[0]["max_tokens"] == 50 and silent_client.sent[0]["model"] == "m"


def test_it_is_reachable_by_its_flat_name():
    import hypernix.relay as flat

    assert flat is relay
