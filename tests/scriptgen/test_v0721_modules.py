"""0.72.1: quant formats, steamroller, Pascal, 6-bit, scriptgen, noodle, livestream.

Grouped by module rather than by feature because that is how they fail:
a change to the quant registry breaks the Pascal tuner's format filter,
and having those next to each other makes the connection visible.
"""
from __future__ import annotations

import ast
import math
import random
import socket

import pytest

# ---------------------------------------------------------------------------
# Quantisation formats
# ---------------------------------------------------------------------------
from hypernix.quant.formats import (  # noqa: E402
    SIX_BIT_MODES,
    get_format,
    list_formats,
)


class TestQuantFormats:
    @pytest.mark.parametrize(
        "name",
        ["NF4", "INT8", "FP8", "FP4", "Q4_K_M", "Q5_K_M", "Q8_0", "EXL2", "AWQ", "GPTQ"],
    )
    def test_every_requested_format_is_registered(self, name):
        assert get_format(name).name == name

    def test_aliases_resolve(self):
        assert get_format("nf4").name == "NF4"
        assert get_format("q4km").name == "Q4_K_M"
        assert get_format("qlora").name == "NF4"

    def test_unknown_formats_list_what_exists(self):
        with pytest.raises(KeyError, match="Known"):
            get_format("Q9_ULTRA")

    def test_effective_bits_include_scale_overhead(self):
        # A "4-bit" model is never 0.5 bytes per parameter, and a
        # registry that says it is will size a GPU wrong.
        assert get_format("NF4").effective_bits > 4.0
        assert get_format("Q4_K_M").effective_bits > 4.0

    def test_fp8_and_fp4_are_absent_on_pascal(self):
        ok, reason = get_format("FP8").supported_on((6, 1))
        assert not ok
        assert "missing instruction" in reason
        assert not get_format("FP4").supported_on((6, 1))[0]

    def test_gptq_does_run_on_pascal(self):
        assert get_format("GPTQ").supported_on((6, 1))[0]

    def test_filtering_by_capability_excludes_what_the_card_cannot_do(self):
        pascal = [f.name for f in list_formats(compute_capability=(6, 1))]
        assert "FP8" not in pascal and "FP4" not in pascal
        assert "NF4" in pascal and "Q4_K_M" in pascal

    def test_size_estimates_are_sane(self):
        seven_b = 7_000_000_000
        fp16 = get_format("FP16").estimate_bytes(seven_b)
        nf4 = get_format("NF4").estimate_bytes(seven_b)
        assert 13e9 < fp16 < 15e9
        assert nf4 < fp16 / 3

    def test_the_three_six_bit_modes_exist(self):
        assert set(SIX_BIT_MODES) == {"packed", "aligned", "hybrid"}
        assert SIX_BIT_MODES["packed"].bits_per_value == 6.0


# ---------------------------------------------------------------------------
# Steamroller
# ---------------------------------------------------------------------------

from hypernix.quant.steamroller import (  # noqa: E402
    STAGING_TIER,
    TIERS,
    SteamrollerError,
    get_tier,
    plan,
)


class TestSteamroller:
    @pytest.mark.parametrize(
        "name", ["Q8_0", "Q3_K_L", "IQ1_M", "IQ0.9_L", "IQ0.75_M", "IQ0.5_XXXL"]
    )
    def test_every_requested_tier_exists(self, name):
        assert get_tier(name).name == name

    def test_the_ladder_descends(self):
        bits = [TIERS[n].bits_per_weight for n in
                ("Q8_0", "Q3_K_L", "IQ1_M", "IQ0.9_L", "IQ0.75_M", "IQ0.5_XXXL")]
        assert bits == sorted(bits, reverse=True)

    def test_narrow_targets_stage_through_q3_k_l(self):
        for target in ("IQ1_M", "IQ0.9_L", "IQ0.5_XXXL"):
            steps = plan("FP16", target).steps
            assert steps[0].target == STAGING_TIER, target

    def test_a_wide_target_does_not_stage(self):
        assert not plan("FP16", "Q8_0").staged

    def test_forcing_staging_where_it_is_pointless_says_so(self):
        result = plan("FP16", "Q8_0", force_staging=True)
        assert any("buys nothing" in w for w in result.warnings)

    def test_extension_tiers_cannot_skip_staging_and_say_so(self):
        # They are packed *from* Q3_K_L, so the pass is required rather
        # than an optimisation. Claiming it was skipped would be a lie.
        result = plan("FP16", "IQ0.75_M", force_staging=False)
        assert result.staged
        assert any("cannot be skipped" in w for w in result.warnings)

    def test_the_sub_bit_tiers_are_marked_as_extensions(self):
        for name in ("IQ0.9_L", "IQ0.75_M", "IQ0.5_XXXL"):
            tier = TIERS[name]
            assert tier.is_extension
            assert "extension type" in tier.honest_warning

    def test_sub_bit_plans_carry_the_quality_warning(self):
        warnings = " ".join(plan("FP16", "IQ0.5_XXXL").warnings)
        assert "1.5 bits" in warnings
        assert "importance matrix" in warnings

    def test_an_imatrix_removes_that_particular_warning(self):
        warnings = " ".join(plan("FP16", "IQ1_M", have_imatrix=True).warnings)
        assert "importance matrix" not in warnings

    def test_a_bad_source_format_points_at_the_fix(self):
        with pytest.raises(SteamrollerError) as excinfo:
            plan("AWQ", "Q8_0")
        assert excinfo.value.code == "bad_source_format"
        assert "convert" in excinfo.value.hint.lower()


# ---------------------------------------------------------------------------
# Pascal
# ---------------------------------------------------------------------------

from hypernix.system.pascal import (  # noqa: E402
    PASCAL_GPUS,
    FP16Guard,
    GPUInfo,
    autotune,
    identify,
    kernel_tuning_for,
)


class TestPascal:
    @pytest.mark.parametrize(
        "name", ["GTX 1080", "GTX 1080 Ti", "Tesla P40", "Tesla P4", "Tesla P100"]
    )
    def test_every_requested_card_is_in_the_table(self, name):
        assert PASCAL_GPUS[name].compute[0] == 6

    def test_identification_survives_driver_name_variation(self):
        assert identify("NVIDIA GeForce GTX 1080 Ti").name == "GTX 1080 Ti"
        assert identify("Tesla P100-PCIE-16GB").name == "Tesla P100"
        # Longest alias wins, or "1080 Ti" would match "GTX 1080".
        assert identify("GeForce GTX 1080").name == "GTX 1080"

    def test_only_gp100_has_a_fast_fp16_path(self):
        assert PASCAL_GPUS["Tesla P100"].fp16_is_fast
        for name in ("GTX 1080", "GTX 1080 Ti", "Tesla P40", "Tesla P4"):
            assert not PASCAL_GPUS[name].fp16_is_fast

    def test_the_tuner_computes_in_fp32_on_a_1080_and_fp16_on_a_p100(self):
        # The single most important thing this module knows.
        def tune(name):
            gpu = PASCAL_GPUS[name]
            return autotune(GPUInfo(name=name, compute=gpu.compute,
                                    vram_gb=gpu.vram_gb, matched=gpu))

        assert tune("GTX 1080").compute_dtype == "fp32"
        assert tune("GTX 1080").storage_dtype == "fp16"
        assert tune("Tesla P100").compute_dtype == "fp16"

    def test_the_tuner_never_offers_fp8_on_pascal(self):
        gpu = PASCAL_GPUS["GTX 1080 Ti"]
        tuning = autotune(GPUInfo(name=gpu.name, compute=gpu.compute,
                                  vram_gb=gpu.vram_gb, matched=gpu))
        assert "FP8" not in tuning.allowed_quant_formats
        assert "FP4" not in tuning.allowed_quant_formats

    def test_every_decision_has_a_reason(self):
        gpu = PASCAL_GPUS["GTX 1080"]
        tuning = autotune(GPUInfo(name=gpu.name, compute=gpu.compute,
                                  vram_gb=gpu.vram_gb, matched=gpu))
        for field in ("compute_dtype", "micro_batch", "attention", "six_bit_mode"):
            assert tuning.reasons.get(field), f"{field} has no stated reason"

    def test_a_smaller_card_gets_a_smaller_batch(self):
        def batch(name):
            gpu = PASCAL_GPUS[name]
            return autotune(
                GPUInfo(name=name, compute=gpu.compute, vram_gb=gpu.vram_gb, matched=gpu),
                parameters=1_500_000_000,
            ).micro_batch

        assert batch("GTX 1080") <= batch("Tesla P40")

    def test_kernel_tuning_targets_the_right_arch(self):
        assert "-arch=sm_61" in kernel_tuning_for(PASCAL_GPUS["GTX 1080 Ti"]).nvcc_flags()
        assert "-arch=sm_60" in kernel_tuning_for(PASCAL_GPUS["Tesla P100"]).nvcc_flags()


class TestFP16Guard:
    def test_a_clean_run_grows_the_scale(self):
        guard = FP16Guard(growth_interval=4)
        start = guard.scale
        for _ in range(8):
            assert guard.step(overflow=False)
        assert guard.scale > start
        assert guard.healthy

    def test_an_overflow_skips_the_step_and_halves_the_scale(self):
        guard = FP16Guard()
        before = guard.scale
        assert guard.step(overflow=True) is False
        assert guard.scale == before / 2

    def test_persistent_overflow_falls_back_to_fp32(self):
        # Loss scaling cannot rescue a run that overflows every step, and
        # Pascal has no BF16 to switch to.
        guard = FP16Guard(max_consecutive_overflows=5)
        for _ in range(5):
            guard.step(overflow=True)
        assert guard.fell_back_to_fp32
        assert "no BF16" in guard.fallback_reason
        assert not guard.healthy

    def test_an_isolated_overflow_does_not_trigger_the_fallback(self):
        guard = FP16Guard(max_consecutive_overflows=5)
        for i in range(20):
            guard.step(overflow=(i == 7))
        assert not guard.fell_back_to_fp32

    def test_after_falling_back_every_step_applies(self):
        guard = FP16Guard(max_consecutive_overflows=2)
        guard.step(overflow=True)
        guard.step(overflow=True)
        assert guard.fell_back_to_fp32
        assert guard.step(overflow=True) is True

    def test_the_scale_never_goes_below_its_floor(self):
        guard = FP16Guard(min_scale=1.0, max_consecutive_overflows=10_000)
        for _ in range(200):
            guard.step(overflow=True)
        assert guard.scale >= 1.0

    def test_invalid_factors_are_refused(self):
        with pytest.raises(ValueError):
            FP16Guard(backoff_factor=1.5)
        with pytest.raises(ValueError):
            FP16Guard(growth_factor=0.5)


# ---------------------------------------------------------------------------
# 6-bit momentum
# ---------------------------------------------------------------------------

from hypernix.optimizers.sixbit import (  # noqa: E402
    bits_per_value,
    pack,
    quantize_group,
    resolve_mode,
    roundtrip,
    unpack,
)


class TestSixBit:
    @pytest.mark.parametrize("mode", ["packed", "aligned", "hybrid"])
    def test_codes_round_trip_exactly(self, mode):
        rng = random.Random(11)
        codes = [rng.randint(-31, 31) for _ in range(4096)]
        assert unpack(pack(codes, mode), len(codes), mode) == codes

    @pytest.mark.parametrize("mode", ["packed", "aligned", "hybrid"])
    def test_values_survive_within_the_expected_error(self, mode):
        rng = random.Random(3)
        values = [rng.gauss(0, 0.02) for _ in range(4096)]
        restored = roundtrip(values, mode)
        absmax = max(abs(v) for v in values)
        assert max(abs(a - b) for a, b in zip(values, restored, strict=True)) < absmax / 31

    def test_packed_and_hybrid_beat_int8_on_size(self):
        assert bits_per_value("packed") < 8.5
        assert bits_per_value("hybrid") < 8.5

    def test_aligned_trades_size_for_unpack_speed(self):
        assert bits_per_value("aligned") > bits_per_value("packed")

    def test_an_all_zero_group_does_not_produce_nan(self):
        codes, scale = quantize_group([0.0] * 32)
        assert scale == 1.0 and set(codes) == {0}
        assert all(v == 0.0 for v in roundtrip([0.0] * 64, "packed"))

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="Unknown 6-bit mode"):
            resolve_mode("sixish")


# ---------------------------------------------------------------------------
# scriptgen
# ---------------------------------------------------------------------------

from hypernix.scriptgen import audit_palette, defaults, generate, inject  # noqa: E402
from hypernix.scriptgen.app import FormModel  # noqa: E402
from hypernix.scriptgen.params import ALL_PARAMS, GROUPS, validate_all  # noqa: E402
from hypernix.scriptgen.templates import BEGIN_MARKER, as_config  # noqa: E402


class TestScriptgenTheme:
    def test_the_palette_has_no_purple_and_passes_contrast(self):
        report = audit_palette()
        assert report["purple_violations"] == []
        assert report["contrast_violations"] == []
        assert report["accent_violations"] == []
        assert report["ok"]

    def test_body_text_clears_wcag_aa_everywhere(self):
        ratios = audit_palette()["ratios"]
        for key in ("text/obsidian", "text/charcoal", "text/slate"):
            assert ratios[key] >= 4.5, key


class TestScriptgenParams:
    def test_the_form_is_dense(self):
        assert len(GROUPS) == 6
        assert len(ALL_PARAMS) >= 40

    def test_the_requested_parameters_are_all_present(self):
        for name in (
            "learning_rate", "warmup_ratio", "epochs", "micro_batch",
            "gradient_accumulation", "loss_function", "optimizer",
        ):
            assert name in ALL_PARAMS

    def test_defaults_validate_cleanly(self):
        errors, _ = validate_all(defaults())
        assert errors == []

    def test_an_out_of_range_value_is_an_error(self):
        values = defaults() | {"learning_rate": 5.0}
        assert validate_all(values)[0]

    def test_a_legal_but_unwise_value_is_advice_not_an_error(self):
        values = defaults() | {"learning_rate": 5e-3}
        errors, warnings = validate_all(values)
        assert errors == []
        assert any("high" in w for w in warnings)

    def test_six_bit_with_a_non_pressure_cooker_optimiser_is_an_error(self):
        values = defaults() | {"six_bit_mode": "packed", "optimizer": "AdamW"}
        errors, _ = validate_all(values)
        assert any("Pressure Cooker" in e for e in errors)

    def test_effective_batch_of_one_is_flagged(self):
        values = defaults() | {"micro_batch": 1, "gradient_accumulation": 1}
        assert any("Effective batch is 1" in w for w in validate_all(values)[1])


class TestScriptgenGeneration:
    def test_the_generated_script_is_valid_python(self):
        ast.parse(generate(defaults()))

    def test_hostile_values_still_produce_valid_python(self):
        # Values are emitted as literals, never interpolated as text.
        values = defaults() | {
            "dataset": 'has "quotes" and \\ backslash',
            "notes": "line\nbreak",
        }
        ast.parse(generate(values))

    def test_floats_round_trip_exactly(self):
        values = defaults() | {"learning_rate": 3e-5, "eps": 1e-8}
        source = generate(values)
        tree = ast.parse(source)
        config = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "CONFIG"
        )
        parsed = ast.literal_eval(config.value)
        assert parsed["learning_rate"] == 3e-5
        assert parsed["eps"] == 1e-8

    def test_injecting_twice_does_not_stack_blocks(self):
        script = generate(defaults())
        once = inject(script, defaults() | {"epochs": 2.0})
        twice = inject(once, defaults() | {"epochs": 5.0})
        assert twice.count(BEGIN_MARKER) == 1
        ast.parse(twice)

    def test_injection_into_a_foreign_script_lands_after_the_imports(self):
        foreign = '"""Mine."""\nimport torch\n\n\ndef train():\n    pass\n'
        out = inject(foreign, defaults())
        ast.parse(out)
        assert out.index("CONFIG") > out.index("import torch")

    def test_config_only_formats(self):
        values = defaults()
        assert as_config(values, fmt="json").startswith("{")
        assert "HYPERNIX_" in as_config(values, fmt="env")
        with pytest.raises(ValueError):
            as_config(values, fmt="yaml")


class TestFormModel:
    def test_a_rejected_value_is_not_stored(self):
        # Storing it would make the preview show a script that cannot run.
        model = FormModel()
        model.set("learning_rate", 3e-5)
        ok, _ = model.set("learning_rate", 99.0)
        assert not ok
        assert model.get("learning_rate") == 3e-5

    def test_saving_is_refused_while_invalid(self, tmp_path):
        model = FormModel()
        model.values["six_bit_mode"] = "packed"
        model.values["optimizer"] = "AdamW"
        with pytest.raises(ValueError, match="Fix these"):
            model.save(tmp_path / "train.py")

    def test_saving_writes_a_parseable_script(self, tmp_path):
        model = FormModel()
        model.set("epochs", 1.5)
        path = model.save(tmp_path / "train.py")
        ast.parse(path.read_text())
        assert not model.dirty

    def test_presets_round_trip(self):
        model = FormModel()
        model.set("learning_rate", 7e-6)
        other = FormModel()
        assert other.load_json(model.to_json()) == []
        assert other.get("learning_rate") == 7e-6

    def test_an_unknown_preset_key_is_skipped_not_fatal(self):
        model = FormModel()
        rejected = model.load_json('{"values": {"from_the_future": 1}}')
        assert rejected

    def test_advanced_parameters_are_hidden_by_default(self):
        model = FormModel()
        hidden = len(model.visible_params("optim"))
        model.show_advanced = True
        assert len(model.visible_params("optim")) > hidden


# ---------------------------------------------------------------------------
# noodle
# ---------------------------------------------------------------------------

from hypernix.interfaces.noodle import (  # noqa: E402
    PROVIDERS,
    TOOLS,
    SearchSpace,
    ToolContext,
    run_tool,
    successive_halving,
    syntax_verifier,
    tool_schemas,
)
from hypernix.interfaces.noodle.providers import Provider  # noqa: E402


class TestNoodleProviders:
    @pytest.mark.parametrize(
        "provider",
        [Provider.OPENAI, Provider.ANTHROPIC, Provider.KIMI, Provider.GEMINI,
         Provider.QWEN, Provider.GROK, Provider.HYPERNIX, Provider.OLLAMA, Provider.VLLM],
    )
    def test_every_requested_provider_is_registered(self, provider):
        assert PROVIDERS[provider].label

    def test_the_three_wire_formats_are_distinguished(self):
        assert PROVIDERS[Provider.ANTHROPIC].wire == "anthropic"
        assert PROVIDERS[Provider.GEMINI].wire == "gemini"
        assert PROVIDERS[Provider.OPENAI].wire == "openai"

    def test_local_providers_are_marked_unpaid(self):
        for provider in (Provider.OLLAMA, Provider.VLLM, Provider.HYPERNIX):
            assert not PROVIDERS[provider].paid


class TestNoodleTools:
    @pytest.mark.parametrize(
        "name",
        ["create_file", "edit_file", "execute_file", "web_search",
         "update_memory", "read_memory", "compact_context",
         "create_todo", "update_todo"],
    )
    def test_every_requested_tool_exists(self, name):
        assert name in TOOLS

    def test_schemas_render_for_every_tool(self):
        schemas = tool_schemas()
        assert len(schemas) == len(TOOLS)
        assert all(s["name"] and s["description"] and s["parameters"] for s in schemas)

    def test_paths_cannot_escape_the_workspace(self, tmp_path):
        ctx = ToolContext(root=tmp_path)
        result = run_tool(ctx, "create_file", {"path": "../escaped.txt", "content": "x"})
        assert not result.ok and result.code == "outside_workspace"
        assert not (tmp_path.parent / "escaped.txt").exists()

    def test_a_symlink_cannot_be_used_to_escape(self, tmp_path):
        # resolve() follows links before the containment check; checking
        # the textual path first is the version that looks correct.
        ctx = ToolContext(root=tmp_path)
        outside = tmp_path.parent / "outside_target"
        outside.mkdir(exist_ok=True)
        try:
            (tmp_path / "link").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable here")
        result = run_tool(ctx, "create_file", {"path": "link/x.txt", "content": "x"})
        assert not result.ok and result.code == "outside_workspace"

    def test_execution_is_off_by_default(self, tmp_path):
        ctx = ToolContext(root=tmp_path)
        run_tool(ctx, "create_file", {"path": "a.py", "content": "print(1)\n"})
        result = run_tool(ctx, "execute_file", {"path": "a.py"})
        assert not result.ok and result.code == "execute_disabled"

    def test_execution_works_when_enabled(self, tmp_path):
        ctx = ToolContext(root=tmp_path, allow_execute=True)
        run_tool(ctx, "create_file", {"path": "a.py", "content": "print('hi')\n"})
        result = run_tool(ctx, "execute_file", {"path": "a.py"})
        assert result.ok and "hi" in result.content

    def test_memory_is_off_unless_the_server_enabled_it(self, tmp_path):
        ctx = ToolContext(root=tmp_path)
        assert run_tool(ctx, "update_memory", {"key": "k", "value": "v"}).code == "memory_disabled"
        assert run_tool(ctx, "read_memory", {}).code == "memory_disabled"

    def test_memory_round_trips_when_enabled(self, tmp_path):
        ctx = ToolContext(root=tmp_path, memory_enabled=True)
        assert run_tool(ctx, "update_memory", {"key": "project", "value": "hypernix"}).ok
        assert "hypernix" in run_tool(ctx, "read_memory", {"key": "project"}).content

    def test_an_ambiguous_edit_is_refused(self, tmp_path):
        # Silently replacing the first occurrence and reporting success
        # is the worst available outcome.
        ctx = ToolContext(root=tmp_path)
        run_tool(ctx, "create_file", {"path": "a.txt", "content": "x\nx\nx\n"})
        result = run_tool(ctx, "edit_file", {"path": "a.txt", "old_text": "x", "new_text": "y"})
        assert not result.ok and result.code == "ambiguous"

    def test_replace_all_resolves_it(self, tmp_path):
        ctx = ToolContext(root=tmp_path)
        run_tool(ctx, "create_file", {"path": "a.txt", "content": "x\nx\n"})
        result = run_tool(
            ctx, "edit_file",
            {"path": "a.txt", "old_text": "x", "new_text": "y", "replace_all": True},
        )
        assert result.ok
        assert (tmp_path / "a.txt").read_text() == "y\ny\n"

    def test_the_write_budget_is_enforced(self, tmp_path):
        ctx = ToolContext(root=tmp_path, max_writes=3)
        results = [
            run_tool(ctx, "create_file", {"path": f"f{i}.txt", "content": "x"})
            for i in range(5)
        ]
        assert results[-1].code == "write_budget"

    def test_todos_track_state(self, tmp_path):
        ctx = ToolContext(root=tmp_path)
        run_tool(ctx, "create_todo", {"text": "do the thing"})
        assert run_tool(ctx, "update_todo", {"id": "t1", "status": "done"}).ok
        assert not run_tool(ctx, "update_todo", {"id": "t9", "status": "done"}).ok
        assert not run_tool(ctx, "update_todo", {"id": "t1", "status": "maybe"}).ok

    def test_an_unknown_tool_lists_the_real_ones(self, tmp_path):
        result = run_tool(ToolContext(root=tmp_path), "teleport", {})
        assert not result.ok and "create_file" in result.content

    def test_the_syntax_verifier_catches_a_broken_file(self, tmp_path):
        ctx = ToolContext(root=tmp_path)
        run_tool(ctx, "create_file", {"path": "bad.py", "content": "def f(:\n"})
        ok, feedback = syntax_verifier()(ctx)
        assert not ok and "bad.py" in feedback


class TestNoodleHPO:
    def test_successive_halving_finds_the_optimum(self):
        space = SearchSpace(log_uniform={"lr": (1e-6, 1e-2)})
        result = successive_halving(
            space, lambda c, b: abs(math.log10(c["lr"]) + 5), trials=27, seed=5
        )
        assert result.best is not None
        assert abs(math.log10(result.best.config["lr"]) + 5) < 0.5

    def test_budgets_increase_across_rungs(self):
        space = SearchSpace(uniform={"x": (0.0, 1.0)})
        result = successive_halving(space, lambda c, b: c["x"], trials=9, seed=1)
        assert result.best.budget > 10

    def test_a_nan_objective_is_a_failed_trial(self):
        # NaN sorts unpredictably and could otherwise win a comparison.
        space = SearchSpace(uniform={"x": (0.0, 1.0)})
        result = successive_halving(space, lambda c, b: float("nan"), trials=9, seed=1)
        assert all(t.failed for t in result.trials)
        assert result.best is None

    def test_log_uniform_actually_spans_decades(self):
        space = SearchSpace(log_uniform={"lr": (1e-6, 1e-3)})
        rng = random.Random(0)
        samples = [space.sample(rng)["lr"] for _ in range(500)]
        # A uniform sampler would put almost nothing below 1e-4.
        assert sum(1 for s in samples if s < 1e-4) > 100

    def test_a_non_positive_log_bound_is_refused(self):
        with pytest.raises(ValueError):
            SearchSpace(log_uniform={"lr": (0.0, 1.0)}).sample(random.Random(0))


# ---------------------------------------------------------------------------
# livestream
# ---------------------------------------------------------------------------

from hypernix.monitoring.livestream import (  # noqa: E402
    Event,
    EventKind,
    LiveStreamServer,
    sample_hardware,
)


class TestLivestream:
    def test_hardware_sampling_never_raises(self):
        sample = sample_hardware()
        assert "gpus" in sample and "ram" in sample and "cpu_percent" in sample

    def test_events_serialise_flat(self):
        payload = Event(EventKind.LOG, {"message": "hello"}, source="trainer").to_json()
        assert '"message": "hello"' in payload
        assert '"kind": "log"' in payload

    def test_the_server_binds_and_serves_its_viewer(self):
        server = LiveStreamServer(port=0, sample_hardware_every=0)
        port = server.start()
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=5)
            sock.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            page = sock.recv(65536).decode(errors="replace")
            sock.close()
            assert "200 OK" in page.splitlines()[0]
            assert "hypernix livestream" in page
        finally:
            server.stop()

    def test_a_token_gates_the_upgrade(self):
        import base64
        import os as _os

        server = LiveStreamServer(port=0, token="s3cret", sample_hardware_every=0)
        port = server.start()

        def handshake(token=""):
            sock = socket.create_connection(("127.0.0.1", port), timeout=5)
            key = base64.b64encode(_os.urandom(16)).decode()
            path = f"/?token={token}" if token else "/"
            sock.sendall(
                f"GET {path} HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n".encode()
            )
            line = sock.recv(4096).decode(errors="replace").splitlines()[0]
            sock.close()
            return line

        try:
            assert "401" in handshake()
            assert "101" in handshake("s3cret")
        finally:
            server.stop()

    def test_publishing_with_no_clients_is_harmless(self):
        server = LiveStreamServer(port=0, sample_hardware_every=0)
        server.start()
        try:
            for i in range(100):
                server.log(f"line {i}")
            assert server.client_count == 0
        finally:
            server.stop()
