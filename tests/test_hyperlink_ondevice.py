"""``hypernix.hyperlink.ondevice`` — will this model run on that phone?

HyperLink can download a GGUF from Hugging Face and run it with no
server involved. The hard part is answering "will this one work?"
*before* a 4 GB download, and being right.

Wrong in one direction hides a model that would have run. Wrong in the
other means the phone downloads for twenty minutes on cellular and is
then killed by the OS partway through the first reply. The tests below
are mostly about keeping the error on the first side.
"""
from __future__ import annotations

import pytest

from hypernix.hyperlink.ondevice import (
    ANE_EXPLANATION,
    GIB,
    SAFETY_MARGIN,
    SUPPLEMENTARY_GGUF_BITS,
    ComputeBackend,
    DeviceBudget,
    GGUFCandidate,
    ModelShape,
    Verdict,
    choose_candidate,
    kv_cache_bytes,
    largest_context,
    plan,
    quant_bits,
    quant_from_filename,
)

GB = 10**9  # decimal, which is the unit Hugging Face file listings use

# Real shapes from published model configs.
SHAPES = {
    "Llama-3.2-1B": dict(
        parameters=1_235_814_400, layers=16, kv_heads=8, head_dim=64,
        vocab_size=128256, embedding_dim=2048, tied_embeddings=True,
        train_context=131072,
    ),
    "Llama-3.2-3B": dict(
        parameters=3_212_749_824, layers=28, kv_heads=8, head_dim=128,
        vocab_size=128256, embedding_dim=3072, tied_embeddings=True,
        train_context=131072,
    ),
    "Llama-3.1-8B": dict(
        parameters=8_030_261_248, layers=32, kv_heads=8, head_dim=128,
        vocab_size=128256, embedding_dim=4096, tied_embeddings=False,
        train_context=131072,
    ),
    "Qwen2.5-7B": dict(
        parameters=7_615_616_512, layers=28, kv_heads=4, head_dim=128,
        vocab_size=152064, embedding_dim=3584, tied_embeddings=False,
        train_context=32768,
    ),
}

# Published GGUF file sizes, in decimal GB.
PUBLISHED = [
    ("Llama-3.2-1B", "Q4_K_M", 0.81),
    ("Llama-3.2-3B", "Q4_K_M", 2.02),
    ("Llama-3.1-8B", "Q4_K_M", 4.92),
    ("Llama-3.1-8B", "Q4_K_S", 4.69),
    ("Llama-3.1-8B", "Q5_K_M", 5.73),
    ("Llama-3.1-8B", "Q6_K", 6.60),
    ("Llama-3.1-8B", "Q8_0", 8.54),
    ("Qwen2.5-7B", "Q4_K_M", 4.68),
]


def shape(name: str, quant: str, **overrides) -> ModelShape:
    return ModelShape(quant=quant, **{**SHAPES[name], **overrides})


# An iPhone 15 Pro: 8 GB of RAM, and roughly 3 GB of it available to a
# normal app before jetsam intervenes.
PHONE = DeviceBudget(
    available_bytes=int(3.0 * GIB), total_ram_bytes=8 * GIB, device_model="iPhone15,2"
)


class TestTheSizeEstimateNeverUnderCounts:
    """The asymmetry that shapes this whole module.

    Over-estimating hides a model that would have run — annoying.
    Under-estimating approves one that cannot, after a multi-gigabyte
    download, and the failure is an immediate process kill rather than
    a slowdown. So every estimate must land at or above the real file.
    """

    @pytest.mark.parametrize(("name", "quant", "published_gb"), PUBLISHED)
    def test_it_is_never_below_the_real_file(self, name, quant, published_gb):
        estimate = shape(name, quant).weights_bytes() / GB
        assert estimate >= published_gb, (
            f"{name} {quant}: estimated {estimate:.2f} GB for a file that is "
            f"{published_gb:.2f} GB — this approves a model that will be killed"
        )

    @pytest.mark.parametrize(("name", "quant", "published_gb"), PUBLISHED)
    def test_it_is_not_absurdly_above_it_either(self, name, quant, published_gb):
        """A planner that doubles everything refuses everything."""
        estimate = shape(name, quant).weights_bytes() / GB
        assert estimate <= published_gb * 1.12, (
            f"{name} {quant}: {estimate:.2f} GB against a real {published_gb:.2f} GB"
        )

    def test_a_flat_bits_times_parameters_would_under_count(self):
        """Which is why the embedding tensors are priced separately.

        Llama-3.2-1B has a 128k vocabulary over 2048 dimensions — 21% of
        its parameters — and llama.cpp keeps that table at a higher
        precision than the file's name suggests. Sizing it flat
        under-counts by about 8%.
        """
        flat = 1_235_814_400 * quant_bits("Q4_K_M") / 8
        assert flat / GB < 0.81, "the premise of the correction no longer holds"
        assert shape("Llama-3.2-1B", "Q4_K_M").weights_bytes() / GB >= 0.81

    def test_the_correction_matters_most_for_small_models(self):
        """The embedding table is a bigger fraction of a small model."""
        def ratio(name: str) -> float:
            with_meta = shape(name, "Q4_K_M").weights_bytes()
            without = ModelShape(
                quant="Q4_K_M",
                **{k: v for k, v in SHAPES[name].items()
                   if k not in ("vocab_size", "embedding_dim", "tied_embeddings")},
            ).weights_bytes()
            return with_meta / without

        assert ratio("Llama-3.2-1B") > ratio("Llama-3.1-8B")

    def test_a_known_file_size_wins_over_arithmetic(self):
        """Once downloaded, the file is the truth."""
        exact = shape("Llama-3.1-8B", "Q4_K_M", file_bytes=4_920_000_000)
        assert exact.weights_bytes() == 4_920_000_000

    def test_an_unknown_quantisation_sizes_to_nothing_not_zero_bytes(self):
        """Callers must treat 0 as "cannot size this", and plan() says so."""
        unknown = ModelShape(parameters=7_000_000_000, quant="Q9_MAGIC",
                             layers=32, kv_heads=8, head_dim=128)
        assert unknown.weights_bytes() == 0
        result = plan(unknown, PHONE)
        assert any("No size" in w for w in result.warnings)


class TestTotalRamIsNotTheBudget:
    """The mistake that makes a planner confidently wrong.

    iOS gives each process a jetsam limit well below the device's RAM,
    and exceeding it is an immediate kill with no exception to catch.
    An 8 GB iPhone will not run a 5 GB model just because the marketing
    page says 8 GB.
    """

    def test_an_8gb_phone_refuses_an_8b_q4_model(self):
        result = plan(shape("Llama-3.1-8B", "Q4_K_M"), PHONE, context=4096)
        assert result.verdict == Verdict.NO

    def test_and_says_why_rather_than_just_no(self):
        result = plan(shape("Llama-3.1-8B", "Q4_K_M"), PHONE, context=4096)
        joined = " ".join(result.notes)
        assert "jetsam" in joined.lower()
        assert "8.0 GiB" in joined and "3.0 GiB" in joined

    def test_it_mentions_the_entitlement_that_would_change_the_answer(self):
        result = plan(shape("Llama-3.1-8B", "Q4_K_M"), PHONE, context=4096)
        assert any("increased-memory-limit" in n for n in result.notes)

    def test_the_entitlement_note_is_dropped_when_it_is_already_on(self):
        entitled = DeviceBudget(
            available_bytes=int(3.0 * GIB), total_ram_bytes=8 * GIB,
            has_increased_limit=True,
        )
        result = plan(shape("Llama-3.1-8B", "Q4_K_M"), entitled, context=4096)
        assert not any("increased-memory-limit" in n for n in result.notes)

    def test_the_entitlement_never_inflates_an_estimate(self):
        """It is recorded for reporting. Claiming headroom the process
        may not have been granted is the same bug in a new place."""
        plain = plan(shape("Llama-3.2-3B", "Q4_K_M"), PHONE, context=4096)
        entitled = plan(
            shape("Llama-3.2-3B", "Q4_K_M"),
            DeviceBudget(available_bytes=int(3.0 * GIB), total_ram_bytes=8 * GIB,
                         has_increased_limit=True),
            context=4096,
        )
        assert plain.total_bytes == entitled.total_bytes

    def test_a_missing_availability_figure_is_refused_not_guessed(self):
        blind = DeviceBudget(available_bytes=0, total_ram_bytes=8 * GIB)
        result = plan(shape("Llama-3.2-1B", "Q4_K_M"), blind)
        assert result.verdict == Verdict.NO
        assert any("os_proc_available_memory" in w for w in result.warnings)

    def test_a_1b_model_does_run_on_that_phone(self):
        """The planner must not simply refuse everything."""
        assert plan(shape("Llama-3.2-1B", "Q4_K_M"), PHONE, context=4096).verdict == Verdict.RUNS


class TestTheKVCache:
    def test_it_scales_with_context(self):
        model = shape("Llama-3.1-8B", "Q4_K_M")
        assert kv_cache_bytes(model, 8192) == 4 * kv_cache_bytes(model, 2048)

    def test_it_can_exceed_the_weights(self):
        """At 32k on an 8B model it is larger than the quantised weights,
        which is exactly when someone tries to use the long context they
        chose the model for."""
        model = shape("Llama-3.1-8B", "Q4_K_M")
        assert kv_cache_bytes(model, 32768) > 0.8 * model.weights_bytes()

    def test_it_is_sized_by_kv_heads_not_attention_heads(self):
        """A grouped-query model shares each KV head across several
        attention heads. Using the attention count overestimates by the
        GQA ratio and refuses models that run."""
        few = ModelShape(parameters=1, quant="Q4_K_M", layers=32, kv_heads=8, head_dim=128)
        many = ModelShape(parameters=1, quant="Q4_K_M", layers=32, kv_heads=32, head_dim=128)
        assert kv_cache_bytes(many, 4096) == 4 * kv_cache_bytes(few, 4096)

    def test_an_8_bit_cache_halves_it(self):
        model = shape("Llama-3.1-8B", "Q4_K_M")
        assert kv_cache_bytes(model, 4096, kv_bits=8) == kv_cache_bytes(model, 4096) // 2

    def test_the_plan_says_when_the_cache_dominates(self):
        result = plan(shape("Llama-3.1-8B", "Q4_K_M"), PHONE, context=32768)
        assert any("KV cache" in n for n in result.notes)

    @pytest.mark.parametrize("context", [0, -1])
    def test_no_context_is_no_cache(self, context):
        assert kv_cache_bytes(shape("Llama-3.1-8B", "Q4_K_M"), context) == 0


class TestLargestContext:
    """Usually the more useful question than "does it fit"."""

    def test_a_model_that_does_not_fit_at_32k_may_fit_at_4k(self):
        model = shape("Llama-3.2-3B", "Q4_K_M")
        assert plan(model, PHONE, context=32768).verdict == Verdict.NO
        assert largest_context(model, PHONE) > 0

    def test_the_answer_actually_runs(self):
        for name in ("Llama-3.2-1B", "Llama-3.2-3B"):
            model = shape(name, "Q4_K_M")
            best = largest_context(model, PHONE)
            if best:
                assert plan(model, PHONE, context=best).verdict == Verdict.RUNS

    def test_one_token_more_does_not(self):
        """Otherwise it is not the largest."""
        model = shape("Llama-3.2-3B", "Q4_K_M")
        best = largest_context(model, PHONE)
        if best:
            assert plan(model, PHONE, context=best + 4096).verdict != Verdict.RUNS

    def test_a_model_that_cannot_fit_at_all_returns_zero(self):
        assert largest_context(shape("Llama-3.1-8B", "Q8_0"), PHONE) == 0

    def test_it_never_exceeds_the_trained_context(self):
        roomy = DeviceBudget(available_bytes=64 * GIB, total_ram_bytes=64 * GIB)
        model = shape("Qwen2.5-7B", "Q4_K_M")
        assert largest_context(model, roomy) <= 32768

    def test_the_answer_is_a_round_number(self):
        """6,143 implies a precision the estimate does not have."""
        best = largest_context(shape("Llama-3.2-1B", "Q4_K_M"), PHONE)
        assert best % 256 == 0


class TestUnifiedMemory:
    """On a phone, GPU offload does not reduce memory.

    A Metal buffer and a malloc come from the same pool. A planner that
    subtracts offloaded layers approves models that cannot run.
    """

    def test_metal_and_cpu_cost_the_same(self):
        model = shape("Llama-3.2-3B", "Q4_K_M")
        on_gpu = plan(model, PHONE, context=2048, backend=ComputeBackend.METAL)
        on_cpu = plan(model, PHONE, context=2048, backend=ComputeBackend.CPU)
        assert on_gpu.total_bytes == on_cpu.total_bytes

    def test_and_the_plan_says_so(self):
        result = plan(shape("Llama-3.2-1B", "Q4_K_M"), PHONE, backend=ComputeBackend.METAL)
        assert any("unified memory" in n for n in result.notes)

    def test_an_unknown_backend_falls_back_rather_than_raising(self):
        result = plan(shape("Llama-3.2-1B", "Q4_K_M"), PHONE, backend="quantum")
        assert result.backend in ComputeBackend.ALL


class TestTheNeuralEngine:
    """It cannot run a GGUF, and saying otherwise would be a lie in a
    settings screen."""

    def test_it_is_not_offered_as_a_backend(self):
        assert not any("ane" in b or "neural" in b for b in ComputeBackend.ALL)

    def test_there_is_an_explanation_to_show_instead(self):
        assert "Core ML" in ANE_EXPLANATION
        assert "Metal" in ANE_EXPLANATION

    def test_the_explanation_says_what_to_use(self):
        """A refusal that names no alternative is not an answer."""
        assert "Metal is the acceleration that exists" in ANE_EXPLANATION


class TestQuantNames:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", "Q4_K_M"),
            ("llama-3-8b-instruct-q4_k_m.gguf", "Q4_K_M"),
            ("model-IQ2_M.gguf", "IQ2_M"),
            ("Qwen2.5-7B-Instruct-Q6_K.gguf", "Q6_K"),
            ("thing-Q4_K_S.gguf", "Q4_K_S"),
            ("m-Q2_K.gguf", "Q2_K"),
            ("m-IQ3_M.gguf", "IQ3_M"),
            ("m-IQ4_NL.gguf", "IQ4_NL"),
            ("some-model.gguf", ""),
        ],
    )
    def test_it_reads_the_convention(self, filename, expected):
        assert quant_from_filename(filename) == expected

    def test_a_longer_name_wins_over_a_prefix_of_it(self):
        """`Q4_K_M` matched as `Q4_K` would carry the wrong bit width,
        and the whole estimate follows from that number."""
        assert quant_from_filename("m-Q4_K_M.gguf") == "Q4_K_M"

    def test_the_common_names_hugging_face_actually_uses_are_all_known(self):
        for name in ("Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "Q2_K", "Q3_K_M", "IQ4_NL"):
            assert quant_bits(name) > 0, f"cannot size {name}"

    def test_the_supplementary_table_is_ordered_sensibly(self):
        assert SUPPLEMENTARY_GGUF_BITS["Q2_K"] < SUPPLEMENTARY_GGUF_BITS["Q4_K_S"]
        assert SUPPLEMENTARY_GGUF_BITS["Q4_K_S"] < SUPPLEMENTARY_GGUF_BITS["Q5_K_S"]
        assert SUPPLEMENTARY_GGUF_BITS["F16"] == 16.0


class TestChoosingACandidate:
    def _shape_for(self, candidate: GGUFCandidate):
        return ModelShape(quant=candidate.resolved_quant, **SHAPES["Llama-3.2-1B"])

    def test_it_picks_the_widest_that_fits(self):
        """Given two files that both run, the wider one is the better
        model."""
        candidates = [
            GGUFCandidate(repo_id="x/y", filename="m-Q4_K_M.gguf"),
            GGUFCandidate(repo_id="x/y", filename="m-Q8_0.gguf"),
            GGUFCandidate(repo_id="x/y", filename="m-Q2_K.gguf"),
        ]
        chosen = choose_candidate(candidates, self._shape_for, PHONE)
        assert chosen is not None and chosen.filename == "m-Q8_0.gguf"

    def test_it_returns_nothing_rather_than_something_unrunnable(self):
        """Falling back to the smallest would download a model that
        cannot run."""
        tiny = DeviceBudget(available_bytes=64 * 1024 * 1024, total_ram_bytes=GIB)
        candidates = [GGUFCandidate(repo_id="x/y", filename="m-Q8_0.gguf")]
        assert choose_candidate(candidates, self._shape_for, tiny) is None

    def test_an_unsizeable_candidate_is_skipped(self):
        candidates = [
            GGUFCandidate(repo_id="x/y", filename="mystery.gguf"),
            GGUFCandidate(repo_id="x/y", filename="m-Q4_K_M.gguf"),
        ]
        chosen = choose_candidate(candidates, self._shape_for, PHONE)
        assert chosen is not None and chosen.filename == "m-Q4_K_M.gguf"

    def test_the_download_url_is_the_resolve_endpoint(self):
        candidate = GGUFCandidate(repo_id="org/model", filename="m-Q4_K_M.gguf")
        assert candidate.download_url == (
            "https://huggingface.co/org/model/resolve/main/m-Q4_K_M.gguf"
        )


class TestTheWireShape:
    def test_a_plan_serialises_whole(self):
        import json
        json.dumps(plan(shape("Llama-3.2-1B", "Q4_K_M"), PHONE).to_dict())

    def test_it_reports_headroom_and_utilisation(self):
        result = plan(shape("Llama-3.2-1B", "Q4_K_M"), PHONE, context=4096)
        assert result.headroom_bytes == result.available_bytes - result.total_bytes
        assert 0 < result.utilisation < 1

    def test_the_safety_margin_leaves_real_headroom(self):
        assert 0.5 < SAFETY_MARGIN < 1.0

    def test_a_tight_verdict_carries_a_warning(self):
        """Otherwise "tight" is indistinguishable from "runs" in a UI."""
        model = shape("Llama-3.2-3B", "Q4_K_M")
        for context in (1024, 2048, 3072, 4096, 6144):
            result = plan(model, PHONE, context=context)
            if result.verdict == Verdict.TIGHT:
                assert result.warnings
                return
        pytest.skip("no context produced a tight verdict for this model")
