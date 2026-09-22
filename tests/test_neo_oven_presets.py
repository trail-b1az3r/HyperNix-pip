"""neo_oven's architecture presets, against what transformers registers.

`model_type` is the one field in a preset that is not a preference. It
is written into config.json and it is what `AutoModel` dispatches on,
so a preset saying "llama" for a Gemma builds Gemma-shaped weights and
then declares them to be a Llama. Nothing complains at build time. It
fails at load, much later, somewhere that looks like a corrupt
checkpoint.

Most of the table said that. gemma, gemma2, gemma3, phi3, glm, glm4,
qwen3, llama4, nemotron and gpt_oss have all had their own model_type
in transformers for a long time, and every one of them was mapped to
"llama" or "qwen2".

So this checks the table against the live CONFIG_MAPPING rather than
against a list written beside it — a hardcoded list of valid types is
the same class of mistake one level up.
"""
from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from hypernix.models.neo_oven import ARCH_PRESETS, FAMILY_ASSUMED  # noqa: E402


@pytest.fixture(scope="module")
def registered() -> set[str]:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    # hypernix is this project's own and is registered at import time by
    # the package rather than by transformers.
    return set(CONFIG_MAPPING_NAMES) | {"hypernix"}


class TestEveryPresetNamesARealArchitecture:
    def test_no_preset_invents_a_model_type(self, registered):
        wrong = {
            name: preset["model_type"]
            for name, preset in ARCH_PRESETS.items()
            if preset["model_type"] not in registered
        }
        assert not wrong, f"model_types transformers does not know: {wrong}"

    @pytest.mark.parametrize(
        "preset,expected",
        [
            # The ones that were wrong, named individually so a
            # regression says which family came back.
            ("gemma", "gemma"),
            ("gemma2", "gemma2"),
            ("gemma3", "gemma3"),
            ("phi3", "phi3"),
            ("phi4", "phi3"),
            ("glm4", "glm4"),
            ("glm5", "glm4"),
            ("glm5.1", "glm4"),
            ("qwen3", "qwen3"),
            ("qwen3.5", "qwen3"),
            ("qwen3.6", "qwen3"),
            ("llama4", "llama4"),
            ("nemotron", "nemotron"),
            ("gpt-oss", "gpt_oss"),
            ("deepseek", "deepseek_v3"),
        ],
    )
    def test_the_corrected_ones_stay_corrected(self, preset, expected):
        assert ARCH_PRESETS[preset]["model_type"] == expected

    def test_the_r1_distills_are_still_llama_on_purpose(self):
        """A distill really is Llama underneath — that is what a distill
        is — so this one is not a mistake and should not be "fixed"."""
        assert ARCH_PRESETS["deepseek-r1"]["model_type"] == "llama"

    def test_no_gemma_or_phi_is_a_llama_any_more(self):
        """The shape of the original bug, as one assertion."""
        mislabelled = [
            name for name, preset in ARCH_PRESETS.items()
            if name.startswith(("gemma", "phi", "glm", "qwen3"))
            and preset["model_type"] in ("llama", "qwen2")
        ]
        assert not mislabelled, mislabelled


class TestTheNumbers:
    def test_glm_normalisation_is_not_off_by_two_orders(self):
        """It was 1e-5. The real default is 1.5625e-07."""
        for name in ("glm4", "glm5", "glm5.1", "glm5.3"):
            assert ARCH_PRESETS[name]["rms_norm_eps"] == pytest.approx(1.5625e-07)

    def test_qwen2_keeps_its_qkv_bias_and_qwen3_drops_it(self):
        """The reason they are different types rather than one with a
        flag."""
        assert ARCH_PRESETS["qwen2"]["attention_bias"] is True
        assert ARCH_PRESETS["qwen3"]["attention_bias"] is False

    def test_glm_and_gpt_oss_carry_qkv_bias(self):
        assert ARCH_PRESETS["glm4"]["attention_bias"] is True
        assert ARCH_PRESETS["gpt-oss"]["attention_bias"] is True

    def test_gemma_ties_its_embeddings(self):
        for name in ("gemma", "gemma2", "gemma3"):
            assert ARCH_PRESETS[name]["tie_word_embeddings"] is True

    def test_every_preset_has_the_four_fields(self):
        for name, preset in ARCH_PRESETS.items():
            missing = {
                "attention_bias", "model_type", "rope_theta",
                "rms_norm_eps", "tie_word_embeddings",
            } - set(preset)
            assert not missing, f"{name} is missing {missing}"

    def test_the_numbers_are_the_right_types(self):
        for name, preset in ARCH_PRESETS.items():
            assert isinstance(preset["rope_theta"], float), name
            assert isinstance(preset["rms_norm_eps"], float), name
            assert isinstance(preset["attention_bias"], bool), name
            assert isinstance(preset["tie_word_embeddings"], bool), name


class TestTheNewOnes:
    @pytest.mark.parametrize(
        "name",
        ["glm5.3", "qwen3.8", "qwen3.8-flash", "muse-spark",
         "spark-x2.5-1.7b", "spark-x2.5-4b"],
    )
    def test_they_are_there(self, name):
        assert name in ARCH_PRESETS

    def test_they_are_marked_as_inferred(self):
        """These were added by name before their configs were available,
        so the arch is the family's convention rather than something
        read off a released config. That is a reasonable starting point
        and it is not a verified one — and FAMILY_ASSUMED is how anybody
        reading the table can tell which is which."""
        for name in ("glm5.3", "qwen3.8", "qwen3.8-flash", "muse-spark",
                     "spark-x2.5-1.7b", "spark-x2.5-4b"):
            assert name in FAMILY_ASSUMED

    def test_everything_in_family_assumed_is_a_real_preset(self):
        assert FAMILY_ASSUMED <= set(ARCH_PRESETS)

    def test_the_verified_ones_are_not_marked_assumed(self):
        """Otherwise the marker means nothing."""
        for name in ("llama", "qwen2", "gemma", "phi3", "glm4", "mistral"):
            assert name not in FAMILY_ASSUMED
